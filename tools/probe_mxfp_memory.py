#!/usr/bin/env python3
"""
Memory probe for Qwen3 MXFP/MSD PPL forward paths.

Run from the onlinearith repo root with PYTHONPATH pointing at the sibling
transformers/src, or let this script add the sibling checkout automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import MethodType

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRANSFORMERS_SRC = (REPO_ROOT / ".." / "transformers" / "src").resolve()
if TRANSFORMERS_SRC.exists() and str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_config import (
    SETUPS,
    apply_config,
    clear_mxfp_weight_cache,
    reconfigure_mlp_layers,
    reset_to_baseline,
)
from ppl_utils import (
    clear_mxfp_progress_hook,
    gib,
    install_mxfp_progress_hook,
    prepare_tail_logits_loss_kwargs,
)
from transformers.models.qwen3.modeling_qwen3 import _MXFPLinearBase


def set_config_if_present(config, name: str, value):
    # Use setattr even if not present so Codex can add the fields incrementally.
    setattr(config, name, value)


def summarize_mxfp_weight_cache(model) -> dict:
    by_dtype: dict[str, int] = {}
    by_device: dict[str, int] = {}
    modules_with_cache = 0
    total_bytes = 0

    for module in model.modules():
        cache = getattr(module, "_w_cache", None)
        if cache is None:
            continue
        modules_with_cache += 1
        for tensor in cache:
            if not torch.is_tensor(tensor):
                continue
            nbytes = tensor.numel() * tensor.element_size()
            total_bytes += nbytes
            by_dtype[str(tensor.dtype)] = by_dtype.get(str(tensor.dtype), 0) + nbytes
            by_device[str(tensor.device)] = by_device.get(str(tensor.device), 0) + nbytes

    return {
        "modules_with_cache": modules_with_cache,
        "total_gib": gib(total_bytes),
        "by_dtype_gib": {k: gib(v) for k, v in sorted(by_dtype.items())},
        "by_device_gib": {k: gib(v) for k, v in sorted(by_device.items())},
    }


def install_forward_probe(records: list[dict], include_all: bool = False):
    original = _MXFPLinearBase.forward

    def wrapped(self, x, compute_context=None):
        device = x.device
        layer_name = getattr(self, "layer_name", None) or "<unregistered>"
        batch_shape = tuple(x.shape[:-1])
        N = math.prod(batch_shape) if batch_shape else 1
        rec = {
            "layer_name": layer_name,
            "class": type(self).__name__,
            "input_shape": list(x.shape),
            "N": int(N),
            "in_features": int(getattr(self, "in_features", -1)),
            "out_features": int(getattr(self, "out_features", -1)),
            "block_size": int(getattr(self, "block_size", -1)),
            "ok": False,
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            rec["alloc_before_gib"] = gib(torch.cuda.memory_allocated(device))
            rec["reserved_before_gib"] = gib(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        try:
            out = original(self, x, compute_context)
            rec["ok"] = True
            return out
        except torch.cuda.OutOfMemoryError:
            rec["oom"] = True
            rec["traceback"] = traceback.format_exc(limit=8)
            raise
        finally:
            rec["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                rec["alloc_after_gib"] = gib(torch.cuda.memory_allocated(device))
                rec["reserved_after_gib"] = gib(torch.cuda.memory_reserved(device))
                rec["peak_alloc_gib"] = gib(torch.cuda.max_memory_allocated(device))
                rec["peak_reserved_gib"] = gib(torch.cuda.max_memory_reserved(device))
            if include_all or (rec.get("peak_alloc_gib", 0.0) >= 0.25) or rec.get("oom"):
                records.append(rec)

    _MXFPLinearBase.forward = wrapped
    return original


def restore_forward_probe(original):
    _MXFPLinearBase.forward = original


def build_input_ids(tokenizer, model, seq_len: int, device: torch.device, random_tokens: bool):
    if random_tokens:
        vocab = int(getattr(model.config, "vocab_size", len(tokenizer)))
        return torch.randint(0, vocab, (1, seq_len), dtype=torch.long, device=device)
    text = "The quick brown fox jumps over the lazy dog. " * max(1, seq_len // 10)
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :seq_len]
    if ids.shape[1] < seq_len:
        repeats = math.ceil(seq_len / ids.shape[1])
        ids = ids.repeat(1, repeats)[:, :seq_len]
    return ids.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe MXFP/MSD forward memory by wrapping _MXFPLinearBase.forward")
    parser.add_argument("--model-path", default="../Qwen3-8B")
    parser.add_argument("--setup", type=int, default=2, help="onlinearith setup ID")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--gpus", default=None, help="Single physical GPU id, e.g. 0 or 3")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--mx-chunk-target-mib", type=int, default=None)
    parser.add_argument("--msd-chunk-target-mib", type=int, default=None)
    parser.add_argument("--weight-cache-dtype", choices=["float16", "float32", "none"], default=None)
    parser.add_argument("--stats", choices=["off", "lite", "full"], default="off")
    parser.add_argument("--random-tokens", action="store_true")
    parser.add_argument("--include-all", action="store_true", help="Record every MXFP layer, not just heavy layers")
    parser.add_argument("--min-headroom-gib", type=float, default=None,
                        help="Fail if CUDA peak reserved-memory headroom is below this many GiB.")
    parser.add_argument("--progress-interval-sec", type=float, default=30.0,
                        help="Print throttled MX/MSD chunk progress every N seconds. Use 0 for every chunk.")
    parser.add_argument("--progress-file", default=None,
                        help="Optional path updated atomically with the latest progress event.")
    parser.add_argument("--output", default="probe_mxfp_memory.json")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    selected = next((s for s in SETUPS if s[0] == args.setup), None)
    if selected is None:
        raise SystemExit(f"unknown setup {args.setup}; valid IDs: {[s[0] for s in SETUPS]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, dtype=dtype)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    _, tag, desc, overrides = selected
    reset_to_baseline(model.config)
    apply_config(model.config, overrides)
    if args.mx_chunk_target_mib is not None:
        set_config_if_present(model.config, "mxfp_chunk_target_mib", args.mx_chunk_target_mib)
        set_config_if_present(model.config, "mxfp_use_chunked_exact", True)
    if args.msd_chunk_target_mib is not None:
        set_config_if_present(model.config, "msd_chunk_target_mib", args.msd_chunk_target_mib)
    if args.weight_cache_dtype is not None:
        set_config_if_present(model.config, "mxfp_weight_cache_dtype", args.weight_cache_dtype)
    if args.stats == "off":
        model.config.msd_perf_stats_enabled = False
        model.config.msd_perf_stats_lite = False
    elif args.stats == "lite":
        model.config.msd_perf_stats_enabled = True
        model.config.msd_perf_stats_lite = True
    else:
        model.config.msd_perf_stats_enabled = True
        model.config.msd_perf_stats_lite = False

    reconfigure_mlp_layers(model, device)
    clear_mxfp_weight_cache(model)
    install_mxfp_progress_hook(
        model,
        interval_sec=args.progress_interval_sec,
        progress_file=args.progress_file,
        device=device,
        extra={"setup": args.setup},
    )

    records: list[dict] = []
    original = install_forward_probe(records, include_all=args.include_all)
    status = "ok"
    error = None
    cache_summary = None
    try:
        input_ids = build_input_ids(tokenizer, model, args.seq_len, device, args.random_tokens)
        labels = input_ids.clone()
        loss_kwargs = prepare_tail_logits_loss_kwargs(
            labels,
            args.seq_len,
            loss_token_chunk_size=512,
            output_logits=False,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            try:
                outputs = model(input_ids, labels=labels, use_cache=False, **loss_kwargs)
            except TypeError:
                outputs = model(input_ids, labels=labels)
        loss = float(outputs.loss.detach().cpu()) if getattr(outputs, "loss", None) is not None else None
        elapsed = time.perf_counter() - t0
        del outputs, input_ids, labels, loss_kwargs
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc)
        elapsed = None
        loss = None
    finally:
        restore_forward_probe(original)
        clear_mxfp_progress_hook(model)
        cache_summary = summarize_mxfp_weight_cache(model)
        clear_mxfp_weight_cache(model)

    cuda_total_gib = None
    cuda_peak_reserved_headroom_gib = None
    meets_min_headroom = None
    if device.type == "cuda":
        cuda_total_gib = gib(torch.cuda.get_device_properties(device).total_memory)
        peak_reserved_gib = gib(torch.cuda.max_memory_reserved(device))
        cuda_peak_reserved_headroom_gib = cuda_total_gib - peak_reserved_gib
        if args.min_headroom_gib is not None:
            meets_min_headroom = status == "ok" and cuda_peak_reserved_headroom_gib >= args.min_headroom_gib

    summary = {
        "status": status,
        "error": error,
        "setup": {"id": args.setup, "tag": tag, "description": desc},
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "device": str(device),
        "elapsed_sec": elapsed,
        "loss": loss,
        "torch_cuda_peak_alloc_gib": gib(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "torch_cuda_peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "torch_cuda_total_gib": cuda_total_gib,
        "torch_cuda_peak_reserved_headroom_gib": cuda_peak_reserved_headroom_gib,
        "min_headroom_gib": args.min_headroom_gib,
        "meets_min_headroom": meets_min_headroom,
        "mxfp_weight_cache": cache_summary,
        "records_top_by_peak_alloc": sorted(records, key=lambda r: r.get("peak_alloc_gib", 0.0), reverse=True)[:40],
        "records": records if args.include_all else None,
    }
    out = Path(args.output)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "records_top_by_peak_alloc" and k != "records"}, indent=2))
    print(f"wrote {out}")
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if status == "oom":
        return 2
    if meets_min_headroom is False:
        print(
            f"headroom check failed: peak reserved-memory headroom "
            f"{cuda_peak_reserved_headroom_gib:.2f} GiB < {args.min_headroom_gib:.2f} GiB"
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
