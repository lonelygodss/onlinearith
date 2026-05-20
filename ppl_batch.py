"""
Batch PPL evaluation across all MSD / MXFP configuration combinations.

Supports **multi-GPU** — either auto-launched (recommended) or via torchrun:
    python ppl_batch.py --nproc 8                        # auto-launch 8 GPUs (picks free port)
    python ppl_batch.py --nproc 4 --only 2 6             # subset on 4 GPUs
    python ppl_batch.py --nproc 4 --gpus 4,5,6,7         # specific GPUs, free port
    python ppl_batch.py                                   # single-GPU fallback

    # Manual torchrun (you must pick a free port yourself if 29500 is taken):
    torchrun --nproc_per_node=8 --master-port=29501 ppl_batch.py
    torchrun --nproc_per_node=3 --master-port=29501 ppl_batch.py --gpus 0,2,5

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
    cd /path/to/onlinearith
    python ppl_batch.py --nproc 8                             # run all setups
    python ppl_batch.py --nproc 8 --list                      # list setups (rank 0)
    python ppl_batch.py --nproc 8 --only 1 6 10               # run only selected
    python ppl_batch.py --nproc 4 --gpus 4,5,6,7              # specific GPUs
    python ppl_batch.py --nproc 8 --force                     # re-run even if done
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
    maybe_relaunch_with_torchrun,
    restrict_gpus,
    suppress_warnings,
)
from experiment_config import (
    BASELINE_CONFIG,
    SETUPS,
    apply_config,
    clear_mxfp_weight_cache,
    format_config_banner,
    get_active_flags,
    get_config_snapshot,
    peak_memory_str,
    reconfigure_mlp_layers,
    reset_peak_memory,
    reset_to_baseline,
)
from ppl_utils import (
    accumulate_weighted_nll,
    clear_mxfp_progress_hook,
    install_mxfp_progress_hook,
    mask_context_labels,
    precompute_windows,
    prepare_tail_logits_loss_kwargs,
)
from runtime_paths import default_model_path, describe_missing_model_path, normalize_output_dir

# Re-export for backward compatibility (calibrate.py, ppltest.py may import these)
_BASELINE_OVERRIDES = BASELINE_CONFIG

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = default_model_path("Qwen3-0.6B")
DATASET     = ("wikitext", "wikitext-2-raw-v1", "test")
MAX_LENGTH  = 4096
STRIDE      = 512
RESULTS_DIR = Path(__file__).parent   # save next to this script


# ── Helpers ───────────────────────────────────────────────────────────────────


def evaluate_ppl(model, encodings, device, seq_len, num_words, num_chars, num_bytes,
                 show_progress=True):
    """Run sliding-window PPL and return a results dict."""
    reset_peak_memory(device)

    total_nll_sum = 0.0
    total_tokens  = 0
    per_chunk_nlls = []
    t_start = time.perf_counter()
    windows = precompute_windows(seq_len, MAX_LENGTH, STRIDE)

    for begin_loc, end_loc, trg_len in tqdm(windows, desc="  PPL windows",
                                            leave=False, disable=not show_progress):
        input_ids  = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = mask_context_labels(input_ids, trg_len)
        loss_kwargs = prepare_tail_logits_loss_kwargs(
            target_ids,
            trg_len,
            loss_token_chunk_size=512,
            output_logits=False,
        )

        with torch.inference_mode():
            try:
                outputs = model(input_ids, labels=target_ids, use_cache=False, **loss_kwargs)
            except TypeError:
                outputs = model(input_ids, labels=target_ids)
            avg_nll = outputs.loss.item()

        total_nll_sum, total_tokens = accumulate_weighted_nll(
            total_nll_sum, total_tokens, avg_nll, trg_len
        )
        per_chunk_nlls.append(avg_nll)
        del input_ids, target_ids, loss_kwargs, outputs

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


def progress_file_for_rank_setup(path: str | None, rank: int, setup_id: int, world_size: int) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if world_size <= 1:
        return str(p)
    return str(p.with_name(f"{p.stem}.rank{rank}.setup{setup_id}{p.suffix}"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global MODEL_PATH, RESULTS_DIR
    parser = argparse.ArgumentParser(description="Batch PPL evaluation for all MXFP/MSD setups")
    parser.add_argument("--list", action="store_true", help="List all setups and exit")
    parser.add_argument("--only", nargs="+", type=int, metavar="ID",
                        help="Run only these setup IDs (e.g. --only 1 6 10)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result file already exists")
    parser.add_argument("--nproc", type=int, default=None, metavar="N",
                        help="Number of GPU workers. Auto-launches via torchrun "
                             "with a free port (avoids EADDRINUSE). "
                             "Replaces manual 'torchrun --nproc_per_node=N'.")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated physical GPU IDs to use, e.g. '4,5,6,7'. "
                             "Must match --nproc (or --nproc_per_node if using torchrun).")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, metavar="DIR",
                        help=f"Local model directory (default: {MODEL_PATH})")
    parser.add_argument("--results-dir", type=str, default=None, metavar="DIR",
                        help=f"Directory for result JSON files (default: {RESULTS_DIR})")
    parser.add_argument("--stats", choices=["off", "lite", "full"], default="off",
                        help="MSD performance-statistics mode for PPL runs. (default: off)")
    parser.add_argument("--lite", action="store_true",
                        help="Backward-compatible alias for --stats lite.")
    parser.add_argument("--mx-chunk-target-mib", type=int, default=None,
                        help="Exact MX-only output chunk target in MiB.")
    parser.add_argument("--msd-chunk-target-mib", type=int, default=None,
                        help="MSD output chunk target in MiB.")
    parser.add_argument("--weight-cache-dtype", choices=["float16", "float32", "none"], default=None,
                        help="Persistent MXFP quantized-weight cache storage.")
    parser.add_argument("--mxfp-progress-interval-sec", type=float, default=30.0,
                        help="Print throttled MX/MSD chunk progress every N seconds. "
                             "Use 0 for every chunk, or a negative value to disable.")
    parser.add_argument("--mxfp-progress-file", default=None,
                        help="Optional path updated atomically with the latest MX/MSD progress event. "
                             "In multi-rank runs, rank/setup suffixes are added.")
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    RESULTS_DIR = normalize_output_dir(args.results_dir, RESULTS_DIR)
    if args.lite:
        args.stats = "lite"
    restrict_gpus(args.gpus)
    maybe_relaunch_with_torchrun(args.nproc)

    # ── Distributed init (no NCCL — ranks work independently) ──
    rank, world_size, local_rank, device = init_distributed_lite()
    suppress_warnings(rank)
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    if not Path(MODEL_PATH).exists():
        if is_main(rank):
            print(f"Error: {describe_missing_model_path(MODEL_PATH)}")
        cleanup_distributed()
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model_kwargs = {"local_files_only": True, "dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    model.to(device)
    model.eval()
    model.config.use_cache = False

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
        reset_to_baseline(model.config)
        apply_config(model.config, overrides)
        if args.mx_chunk_target_mib is not None:
            model.config.mxfp_chunk_target_mib = args.mx_chunk_target_mib
            model.config.mxfp_use_chunked_exact = True
        if args.msd_chunk_target_mib is not None:
            model.config.msd_chunk_target_mib = args.msd_chunk_target_mib
        if args.weight_cache_dtype is not None:
            model.config.mxfp_weight_cache_dtype = args.weight_cache_dtype
        if args.stats == "off":
            model.config.msd_perf_stats_enabled = False
            model.config.msd_perf_stats_lite = False
        elif args.stats == "lite":
            model.config.msd_perf_stats_enabled = True
            model.config.msd_perf_stats_lite = True
        else:
            model.config.msd_perf_stats_enabled = True
            model.config.msd_perf_stats_lite = False

        # Rebuild MLP linear layers to match the new config flags
        reconfigure_mlp_layers(model, device)
        install_mxfp_progress_hook(
            model,
            interval_sec=args.mxfp_progress_interval_sec,
            progress_file=progress_file_for_rank_setup(args.mxfp_progress_file, rank, sid, world_size),
            device=device,
            extra={"rank": rank, "setup": sid},
        )

        # Show active config
        banner = format_config_banner(model.config, setup_id=sid, setup_desc=desc)
        for line in banner.splitlines():
            print(f"[rank {rank}]   {line}")

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
                             "dtype": str(dtype), "world_size": world_size,
                             "stats": args.stats}
        results["config_snapshot"] = get_config_snapshot(model.config)

        # Save
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        ppl = results["metrics"]["token_perplexity"]
        wall = results["performance"]["wall_time_sec"]
        print(f"[rank {rank}]   PPL={ppl:.4f}  |  {wall:.0f}s  |  saved -> {result_file.name}\n")
        local_summary.append((sid, tag, ppl, f"{wall:.0f}s"))
        clear_mxfp_progress_hook(model)
        clear_mxfp_weight_cache(model)

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
                            "dtype": str(dtype), "stats": args.stats},
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

    clear_mxfp_progress_hook(model)
    clear_mxfp_weight_cache(model)
    cleanup_distributed()


if __name__ == "__main__":
    main()
