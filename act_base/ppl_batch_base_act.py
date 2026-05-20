"""
Batch PPL evaluation for activation-only n:m sparsity across MXFP setups.

Supports **multi-GPU** — either auto-launched (recommended) or via torchrun:
    python act_base/ppl_batch_base_act.py --nproc 8                        # auto-launch 8 GPUs (picks free port)
    python act_base/ppl_batch_base_act.py --nproc 4 --only 1 2             # subset on 4 GPUs
    python act_base/ppl_batch_base_act.py --nproc 4 --gpus 4,5,6,7         # specific GPUs, free port
    python act_base/ppl_batch_base_act.py                                   # single-GPU fallback

    # Manual torchrun (you must pick a free port yourself if 29500 is taken):
    torchrun --nproc_per_node=8 --master-port=29501 act_base/ppl_batch_base_act.py
    torchrun --nproc_per_node=3 --master-port=29501 act_base/ppl_batch_base_act.py --gpus 0,2,5

Each GPU loads its own model copy and processes a shard of the setups.
Setups are partitioned round-robin across ranks; each rank writes its
result files independently (no inter-rank communication during eval).
Rank 0 prints the summary after all ranks finish.

Loads the model ONCE per rank, then iterates through every assigned setup
by patching config fields in-memory. Runtime activation-only n:m sparsity is
enabled from args for each setup. Each run's results are saved to
    ppl_results_{tag}.json

Skips setups whose result file already exists (resume-safe).

Todo: Add autmatic load ballancing before the batch run, detect which gpus are free and 
assign more setups to them.  For now, just assign round-robin and let the user manually re-run any stragglers.

Usage:
    cd /path/to/onlinearith
    python act_base/ppl_batch_base_act.py --nproc 8                             # run all setups
    python act_base/ppl_batch_base_act.py --nproc 8 --list                      # list setups (rank 0)
    python act_base/ppl_batch_base_act.py --nproc 8 --only 1 6 10               # run only selected
    python act_base/ppl_batch_base_act.py --nproc 4 --gpus 4,5,6,7              # specific GPUs
    python act_base/ppl_batch_base_act.py --nproc 8 --force                     # re-run even if done
    python act_base/ppl_batch_base_act.py                                       # single-GPU fallback
    python act_base/ppl_batch_base_act.py --gpus 3                              # single specific GPU
"""


SETUPS = [
    (1, "MXFP8",      "MXFP8 (E4M3FN)", {"use_mxfp8": True}),
    (2, "MXFP6_E2M3", "MXFP6 E2M3", {"use_mxfp6": True, "mxfp6_format": "e2m3"}),
    (3, "MXFP6_E3M2", "MXFP6 E3M2", {"use_mxfp6": True, "mxfp6_format": "e3m2"}),
    (4, "MXFP4",      "MXFP4 (E2M1)", {"use_mxfp4": True}),
]

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ONLINEARITH_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = ONLINEARITH_ROOT.parent

if str(ONLINEARITH_ROOT) not in sys.path:
    sys.path.insert(0, str(ONLINEARITH_ROOT))

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
    BASELINE_CONFIG, apply_config, reset_to_baseline,
    reconfigure_mlp_layers, get_config_snapshot, get_active_flags,
    format_config_banner,peak_memory_str,
    reset_peak_memory,
)
from runtime_paths import describe_missing_model_path, normalize_output_dir

# Re-export for backward compatibility (calibrate.py, ppltest.py may import these)
_BASELINE_OVERRIDES = BASELINE_CONFIG

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = str((WORKSPACE_ROOT / "Qwen3-0.6B").resolve())
DATASET     = ("wikitext", "wikitext-2-raw-v1", "test")
MAX_LENGTH  = 4096
STRIDE      = 512
RESULTS_ROOT = (WORKSPACE_ROOT / "data" / "act_base").resolve()
RESULTS_DIR = None   # set in main()


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def discover_nm_cases(results_root):
    """Return sorted [(n, m, path), ...] for subdirs named like 'n-m'."""
    cases = []
    if not results_root.exists():
        return cases

    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", child.name)
        if not match:
            continue
        n, m = int(match.group(1)), int(match.group(2))
        cases.append((n, m, child))

    cases.sort(key=lambda x: (x[0], x[1]))
    return cases


def run_complete_mode(args):
    """Run this script serially for every discovered n-m case."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        print("ERROR: --complete must be launched as a single process (not inside torchrun).")
        print("Run: python act_base/ppl_batch_base_act.py --complete [--nproc N] [--gpus ...]")
        return 2

    cases = discover_nm_cases(RESULTS_ROOT)
    if not cases:
        print(f"No n-m result directories found in {RESULTS_ROOT}.")
        return 0

    print(f"[complete] Found {len(cases)} n-m cases under {RESULTS_ROOT}")
    start_all = time.perf_counter()

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    for idx, (n, m, case_dir) in enumerate(cases, start=1):
        cmd = [
            sys.executable,
            str(script_path),
            "-n", str(n),
            "-m", str(m),
        ]
        if args.nproc is not None:
            cmd.extend(["--nproc", str(args.nproc)])
        if args.gpus is not None:
            cmd.extend(["--gpus", args.gpus])
        cmd.extend(["--model-path", MODEL_PATH])
        cmd.extend(["--results-root", str(RESULTS_ROOT)])
        if args.force:
            cmd.append("--force")
        if args.only:
            cmd.extend(["--only", *[str(s) for s in args.only]])

        print()
        print(f"[complete] ({idx}/{len(cases)}) Running n={n}, m={m} in {case_dir.name}")
        print(f"[complete] Command: {' '.join(cmd)}")
        start_case = time.perf_counter()

        completed = subprocess.run(cmd, cwd=str(script_dir))
        case_elapsed = time.perf_counter() - start_case
        if completed.returncode != 0:
            print(f"[complete] FAILED n={n}, m={m} (exit={completed.returncode}, {case_elapsed:.1f}s)")
            return completed.returncode

        print(f"[complete] DONE n={n}, m={m} ({case_elapsed:.1f}s)")

    all_elapsed = time.perf_counter() - start_all
    print()
    print(f"[complete] All cases finished in {all_elapsed/60:.1f} min")
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global MODEL_PATH, RESULTS_ROOT, RESULTS_DIR
    parser = argparse.ArgumentParser(description="Batch PPL evaluation for activation-only n:m sparsity")
    parser.add_argument("--complete", action="store_true",
                        help="Run all discovered n-m cases serially (one ppl_batch_base run per case).")
    parser.add_argument("--list", action="store_true", help="List all setups and exit")
    parser.add_argument("--only", nargs="+", type=int, metavar="ID",
                        help="Run only these setup IDs (e.g. --only 1 6 10)")
    parser.add_argument("-n", type=int, default=2, help="Sparsify n elements")
    parser.add_argument("-m", type=int, default=4, help="Group size")
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
    parser.add_argument("--results-root", type=str, default=None, metavar="DIR",
                        help=f"Root directory containing n-m result directories (default: {RESULTS_ROOT})")
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    RESULTS_ROOT = normalize_output_dir(args.results_root, RESULTS_ROOT)

    if args.m <= 0:
        raise SystemExit("ERROR: -m must be > 0.")
    if args.n < 0:
        raise SystemExit("ERROR: -n must be >= 0.")
    if args.n >= args.m:
        raise SystemExit(f"ERROR: activation n:m requires n < m, got n={args.n}, m={args.m}.")

    if args.complete:
        exit_code = run_complete_mode(args)
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    restrict_gpus(args.gpus)
    maybe_relaunch_with_torchrun(args.nproc)
    RESULTS_DIR = RESULTS_ROOT / f"{args.n}-{args.m}"

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

        # Activation-only n:m mode: no offline mask file required.
        apply_config(model.config, {
            "use_activation_nm_sparsity": True,
            "activation_nm_n": args.n,
            "activation_nm_m": args.m,
            "use_msd_truncation": False,
        })

        # Rebuild MLP linear layers to match the new config flags
        reconfigure_mlp_layers(model, device)
        model.eval()

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
                             "dtype": str(dtype), "world_size": world_size}
        results["activation_sparsity"] = {
            "mode": "runtime_activation_nm",
            "n": args.n,
            "m": args.m,
        }
        results["config_snapshot"] = get_config_snapshot(model.config)

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
        summary_file = RESULTS_DIR / "ppl_batch_base_act_summary.json"
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
