"""
Batch PPL evaluation across all MSD / MXFP configuration combinations.

Supports **multi-GPU** via torchrun:
    torchrun --nproc_per_node=8 ppl_batch.py             # all 21 setups on 8 GPUs
    torchrun --nproc_per_node=4 ppl_batch.py --only 2 6  # subset on 4 GPUs
    python ppl_batch.py                                   # single-GPU fallback

Each GPU loads its own model copy and processes a shard of the setups.
Setups are partitioned round-robin across ranks; each rank writes its
result files independently (no inter-rank communication during eval).
Rank 0 prints the summary after all ranks finish.

Loads the model ONCE per rank, then iterates through every assigned setup
by patching config fields in-memory.  Each run's results are saved to
    ppl_results_{tag}.json

Skips setups whose result file already exists (resume-safe).

Todo: Add autmatic load ballancing before the batch run, detect which gpus are free and 
assign more setups to them.  For now, just assign round-robin and let the user manually re-run any stragglers.

Usage:
    cd /home/xzj/coding/onlinearith
    torchrun --nproc_per_node=8 ppl_batch.py                 # run all setups
    torchrun --nproc_per_node=8 ppl_batch.py --list          # list setups (rank 0)
    torchrun --nproc_per_node=8 ppl_batch.py --only 1 6 10   # run only selected
    torchrun --nproc_per_node=8 ppl_batch.py --force         # re-run even if done
    torchrun --nproc_per_node=3 ppl_batch.py --gpus 0,2,5    # use specific GPUs
    python ppl_batch.py                                       # single-GPU fallback
    python ppl_batch.py --gpus 3                              # single specific GPU
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dist_utils import (
    cleanup_distributed,
    file_barrier,
    init_distributed_lite,
    is_main,
    restrict_gpus,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "../Qwen3-0.6B"
DATASET     = ("wikitext", "wikitext-2-raw-v1", "test")
MAX_LENGTH  = 4096
STRIDE      = 512
RESULTS_DIR = Path(__file__).parent   # save next to this script

# ── Setup definitions ─────────────────────────────────────────────────────────
# Each setup is (id, tag, description, config_overrides_dict)
# config_overrides are applied ON TOP of a clean baseline (all mxfp/msd off).

_MSD_DEFAULTS = {
    "use_msd_truncation": True,
    "msd_cycle_budget": 16,
    "msd_online_delay": 2,
    "msd_budget_dynamic_scale": 1.0,
    "msd_budget_dynamic_threshold": 0.0,
    "msd_budget_dynamic_mode": "linear",
    "msd_deep_pipeline": False,
    "msd_pipeline_precision_loss": 2,
    "msd_calibration_data": None,
}

def _msd(budget=16, pipeline=False, **extra):
    d = dict(_MSD_DEFAULTS)
    d["msd_cycle_budget"] = budget
    d["msd_deep_pipeline"] = pipeline
    d.update(extra)
    return d


SETUPS = [
    # ── Tier 1: Baseline & MX-only ──
    (1,  "baseline",          "FP16 baseline (no quantization)",
     {"use_mxfp8": False, "use_mxfp6": False, "use_mxfp4": False}),

    (2,  "MXFP8",             "MXFP8 only",
     {"use_mxfp8": True}),

    (3,  "MXFP6_E2M3",       "MXFP6 E2M3 only",
     {"use_mxfp6": True, "mxfp6_format": "e2m3"}),

    (4,  "MXFP6_E3M2",       "MXFP6 E3M2 only",
     {"use_mxfp6": True, "mxfp6_format": "e3m2"}),

    (5,  "MXFP4",             "MXFP4 only",
     {"use_mxfp4": True}),

    # ── Tier 2: MSD default budget (B=16) across formats ──
    (6,  "MXFP8_MSD_B16",    "MXFP8 + MSD B=16",
     {"use_mxfp8": True, **_msd(16)}),

    (7,  "MXFP6_E2M3_MSD_B16", "MXFP6 E2M3 + MSD B=16",
     {"use_mxfp6": True, "mxfp6_format": "e2m3", **_msd(16)}),

    (8,  "MXFP6_E3M2_MSD_B16", "MXFP6 E3M2 + MSD B=16",
     {"use_mxfp6": True, "mxfp6_format": "e3m2", **_msd(16)}),

    (9,  "MXFP4_MSD_B16",    "MXFP4 + MSD B=16",
     {"use_mxfp4": True, **_msd(16)}),

    # ── Tier 3: Budget sweep (MXFP8) ──
    (10, "MXFP8_MSD_B8",     "MXFP8 + MSD B=8",
     {"use_mxfp8": True, **_msd(8)}),

    (11, "MXFP8_MSD_B12",    "MXFP8 + MSD B=12",
     {"use_mxfp8": True, **_msd(12)}),

    # (B=16 already covered by setup #6)

    (12, "MXFP8_MSD_B20",    "MXFP8 + MSD B=20",
     {"use_mxfp8": True, **_msd(20)}),

    (13, "MXFP8_MSD_B24",    "MXFP8 + MSD B=24",
     {"use_mxfp8": True, **_msd(24)}),

    (14, "MXFP8_MSD_B32",    "MXFP8 + MSD B=32",
     {"use_mxfp8": True, **_msd(32)}),

    # ── Tier 3b: Budget sweep (MXFP4) ──
    (15, "MXFP4_MSD_B8",     "MXFP4 + MSD B=8",
     {"use_mxfp4": True, **_msd(8)}),

    (16, "MXFP4_MSD_B12",    "MXFP4 + MSD B=12",
     {"use_mxfp4": True, **_msd(12)}),

    # (B=16 already covered by setup #9)

    (17, "MXFP4_MSD_B20",    "MXFP4 + MSD B=20",
     {"use_mxfp4": True, **_msd(20)}),

    (18, "MXFP4_MSD_B24",    "MXFP4 + MSD B=24",
     {"use_mxfp4": True, **_msd(24)}),

    (19, "MXFP4_MSD_B32",    "MXFP4 + MSD B=32",
     {"use_mxfp4": True, **_msd(32)}),

    # ── Tier 4: Deep pipeline ──
    (20, "MXFP8_MSD_B16_pipeline", "MXFP8 + MSD B=16 + pipeline",
     {"use_mxfp8": True, **_msd(16, pipeline=True)}),

    (21, "MXFP4_MSD_B16_pipeline", "MXFP4 + MSD B=16 + pipeline",
     {"use_mxfp4": True, **_msd(16, pipeline=True)}),
]


# ── Baseline config (everything off) ─────────────────────────────────────────
_BASELINE_OVERRIDES = {
    "use_mxfp8": False, "use_mxfp6": False, "use_mxfp4": False,
    "use_msd_truncation": False, "msd_deep_pipeline": False,
    "msd_calibration_data": None,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def reconfigure_mlp_layers(model, device):
    """
    Replace every MLP linear layer with the correct type for the current config.

    ``_make_linear()`` is called at model construction and bakes in the linear
    class (MXFP8Linear, MXFP6Linear, MXFP4Linear, or nn.Linear).  Changing
    ``model.config`` later does NOT change the existing layer objects.  This
    function walks all MLP modules and rebuilds the three projections to
    match the current config, sharing the weight tensor so no data is copied.
    """
    from transformers.models.qwen3.modeling_qwen3 import _make_linear, Qwen3MLP

    config = model.config
    for module in model.modules():
        if not isinstance(module, Qwen3MLP):
            continue
        for attr in ("gate_proj", "up_proj", "down_proj"):
            old = getattr(module, attr)
            new = _make_linear(old.in_features, old.out_features, config)
            new.weight = old.weight  # share the nn.Parameter (no copy)
            if hasattr(old, "bias_param") and old.bias_param is not None:
                new.bias_param = old.bias_param
            new = new.to(device)
            setattr(module, attr, new)

    # Invalidate MSD context so it re-walks modules & re-sets layer_name etc.
    if hasattr(model, "_msd_context"):
        model._msd_context = None
        model._msd_context_config_hash = None


def peak_memory_str(device):
    if device.type == "cuda":
        return f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB"
    return "N/A"


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def apply_config(config, overrides):
    """Apply a dict of overrides to a model config (in-place)."""
    for k, v in overrides.items():
        setattr(config, k, v)


def evaluate_ppl(model, encodings, device, seq_len, num_words, num_chars, num_bytes,
                 show_progress=True):
    """Run sliding-window PPL and return a results dict."""
    reset_peak_memory(device)

    total_nll_sum = 0.0
    total_tokens  = 0
    per_chunk_nlls = []
    prev_end_loc = 0
    t_start = time.perf_counter()

    for begin_loc in tqdm(range(0, seq_len, STRIDE), desc="  PPL windows",
                          leave=False, disable=not show_progress):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids  = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            avg_nll = outputs.loss.item()

        total_nll_sum += avg_nll * trg_len
        total_tokens  += trg_len
        per_chunk_nlls.append(avg_nll)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    elapsed  = time.perf_counter() - t_start
    mean_nll = total_nll_sum / total_tokens

    token_ppl  = math.exp(mean_nll)
    word_ppl   = math.exp(mean_nll * total_tokens / num_words)
    bpb        = (mean_nll * total_tokens / num_bytes) / math.log(2)
    bpc        = (mean_nll * total_tokens / num_chars) / math.log(2)
    throughput = total_tokens / elapsed
    chunk_arr  = np.array(per_chunk_nlls)

    return {
        "metrics": {
            "token_perplexity": round(token_ppl, 4),
            "word_perplexity":  round(word_ppl, 4),
            "bits_per_byte":    round(bpb, 4),
            "bits_per_char":    round(bpc, 4),
            "mean_nll_nats":    round(mean_nll, 4),
        },
        "reliability": {
            "num_chunks":     len(per_chunk_nlls),
            "chunk_nll_mean": round(float(chunk_arr.mean()), 4),
            "chunk_nll_std":  round(float(chunk_arr.std()), 4),
            "scored_tokens":  total_tokens,
        },
        "performance": {
            "throughput_tokens_per_sec": round(throughput, 1),
            "wall_time_sec":             round(elapsed, 2),
            "peak_memory":               peak_memory_str(device),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch PPL evaluation for all MXFP/MSD setups")
    parser.add_argument("--list", action="store_true", help="List all setups and exit")
    parser.add_argument("--only", nargs="+", type=int, metavar="ID",
                        help="Run only these setup IDs (e.g. --only 1 6 10)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result file already exists")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated physical GPU IDs to use, e.g. '0,2,5'. "
                             "Must match --nproc_per_node when using torchrun.")
    args = parser.parse_args()
    restrict_gpus(args.gpus)

    # ── Distributed init (no NCCL — ranks work independently) ──
    rank, world_size, local_rank, device = init_distributed_lite()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # ── List mode (rank 0 only, then exit) ──
    if args.list:
        if is_main(rank):
            print(f"{'ID':>3}  {'Tag':<30}  Description")
            print("-" * 75)
            for sid, tag, desc, _ in SETUPS:
                result_file = RESULTS_DIR / f"ppl_results_{tag}.json"
                exists = "  (done)" if result_file.exists() else ""
                print(f"{sid:3d}  {tag:<30}  {desc}{exists}")
        cleanup_distributed()
        return

    # ── Filter setups ──
    if args.only:
        selected = {s for s in args.only}
        run_setups = [(s, t, d, c) for s, t, d, c in SETUPS if s in selected]
        if not run_setups:
            if is_main(rank):
                print(f"No matching setup IDs: {args.only}")
                print(f"Valid IDs: {[s[0] for s in SETUPS]}")
            cleanup_distributed()
            return
    else:
        run_setups = list(SETUPS)

    # Partition setups across ranks (round-robin)
    my_setups = run_setups[rank::world_size]

    if is_main(rank):
        print(f"World size: {world_size}  |  Device: {device}  |  dtype: {dtype}")
        print(f"Total setups: {len(run_setups)}  |  Setups on this rank: {len(my_setups)}")
        print()

    # ── Device & model ──
    if is_main(rank):
        print("Loading tokenizer & model ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model_kwargs = {"local_files_only": True, "torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    model.to(device)
    model.eval()

    if is_main(rank):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {MODEL_PATH}  |  Params: {num_params/1e6:.1f}M")

    # ── Dataset ──
    ds_name, ds_config, ds_split = DATASET
    if is_main(rank):
        print(f"Loading dataset: {ds_name}/{ds_config} ({ds_split}) ...")

    test_data = load_dataset(ds_name, ds_config, split=ds_split)
    raw_text  = "\n\n".join(test_data["text"])
    encodings = tokenizer(raw_text, return_tensors="pt")
    seq_len   = encodings.input_ids.size(1)
    num_words = len(raw_text.split())
    num_chars = len(raw_text)
    num_bytes = len(raw_text.encode("utf-8"))

    if is_main(rank):
        print(f"Tokens: {seq_len:,}  |  Words: {num_words:,}")
        print()

    # ── Run assigned setups ──
    local_summary = []
    total_start = time.perf_counter()

    for i, (sid, tag, desc, overrides) in enumerate(my_setups):
        result_file = RESULTS_DIR / f"ppl_results_{tag}.json"
        print(f"[rank {rank}] {'='*55}")
        print(f"[rank {rank}]   [{i+1}/{len(my_setups)}]  Setup #{sid}: {desc}")
        print(f"[rank {rank}]   Tag: {tag}  ->  {result_file.name}")
        print(f"[rank {rank}] {'='*55}")

        # Skip if exists
        if result_file.exists() and not args.force:
            with open(result_file) as f:
                existing = json.load(f)
            ppl = existing.get("metrics", {}).get("token_perplexity", "?")
            print(f"[rank {rank}]   Already exists (PPL={ppl}). Use --force to re-run.\n")
            local_summary.append((sid, tag, ppl, "skipped"))
            continue

        # Reset to baseline, then apply this setup's overrides
        apply_config(model.config, _BASELINE_OVERRIDES)
        apply_config(model.config, overrides)

        # Rebuild MLP linear layers to match the new config flags
        reconfigure_mlp_layers(model, device)

        # Show active config
        active_flags = []
        for k in ["use_mxfp8", "use_mxfp6", "use_mxfp4", "use_msd_truncation",
                   "msd_cycle_budget", "msd_deep_pipeline"]:
            v = getattr(model.config, k, None)
            if v is not None and v is not False:
                active_flags.append(f"{k}={v}")
        print(f"[rank {rank}]   Config: {', '.join(active_flags) or '(baseline fp16)'}")

        # Evaluate
        results = evaluate_ppl(model, encodings, device, seq_len,
                               num_words, num_chars, num_bytes,
                               show_progress=is_main(rank))

        # Add metadata
        results["setup"] = {"id": sid, "tag": tag, "description": desc,
                            "config_overrides": {k: v for k, v in overrides.items()
                                                 if not callable(v)}}
        results["model"] = MODEL_PATH
        results["dataset"] = f"{ds_name}/{ds_config}/{ds_split}"
        results["config"] = {"max_length": MAX_LENGTH, "stride": STRIDE,
                             "dtype": str(dtype)}

        # Save
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        ppl = results["metrics"]["token_perplexity"]
        wall = results["performance"]["wall_time_sec"]
        print(f"[rank {rank}]   PPL={ppl:.4f}  |  {wall:.0f}s  |  saved -> {result_file.name}\n")
        local_summary.append((sid, tag, ppl, f"{wall:.0f}s"))

    local_elapsed = time.perf_counter() - total_start

    # ── Wait for all ranks to finish (file-based, no NCCL timeout) ──
    file_barrier(rank, world_size, RESULTS_DIR)
    total_elapsed = time.perf_counter() - total_start

    # ── Summary (rank 0 reads all result files) ──
    if is_main(rank):
        print()
        print(f"{'='*60}")
        print(f"  BATCH COMPLETE  ({total_elapsed/60:.1f} min total, {world_size} GPUs)")
        print(f"{'='*60}")
        print(f"{'ID':>3}  {'Tag':<30}  {'PPL':>10}  {'Time':>8}")
        print("-" * 60)

        summary_rows = []
        for sid, tag, desc, overrides in run_setups:
            result_file = RESULTS_DIR / f"ppl_results_{tag}.json"
            if result_file.exists():
                with open(result_file) as f:
                    res = json.load(f)
                metrics = res.get("metrics", {})
                perf    = res.get("performance", {})
                ppl  = metrics.get("token_perplexity", "?")
                wall = perf.get("wall_time_sec", "?")
                ppl_str  = f"{ppl:.4f}" if isinstance(ppl, float) else str(ppl)
                wall_str = f"{wall:.0f}s" if isinstance(wall, (int, float)) else str(wall)
                summary_rows.append({
                    "id": sid, "tag": tag, "description": desc,
                    "token_perplexity": metrics.get("token_perplexity"),
                    "word_perplexity":  metrics.get("word_perplexity"),
                    "bits_per_byte":    metrics.get("bits_per_byte"),
                    "bits_per_char":    metrics.get("bits_per_char"),
                    "mean_nll_nats":    metrics.get("mean_nll_nats"),
                    "wall_time_sec":    perf.get("wall_time_sec"),
                    "peak_memory":      perf.get("peak_memory"),
                    "status": "ok",
                })
            else:
                ppl_str = "MISSING"
                wall_str = "-"
                summary_rows.append({
                    "id": sid, "tag": tag, "description": desc,
                    "status": "missing",
                })
            print(f"{sid:3d}  {tag:<30}  {ppl_str:>10}  {wall_str:>8}")

        print("-" * 60)

        # ── Save consolidated summary JSON ──
        from datetime import datetime, timezone
        summary_file = RESULTS_DIR / "ppl_batch_summary.json"
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_PATH,
            "dataset": "/".join(DATASET),
            "eval_config": {"max_length": MAX_LENGTH, "stride": STRIDE,
                            "dtype": str(dtype)},
            "world_size": world_size,
            "total_wall_time_sec": round(total_elapsed, 2),
            "setups_requested": len(run_setups),
            "setups_completed": sum(1 for r in summary_rows if r["status"] == "ok"),
            "results": summary_rows,
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file.name}")
        print(f"Results saved in: {RESULTS_DIR}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
