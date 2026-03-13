"""
Perplexity (PPL) evaluation script with richer metrics and reliable computation.

Supports **multi-GPU** — either auto-launched (recommended) or via torchrun:
    python ppltest.py --nproc 8                            # auto-launch 8 GPUs (picks free port)
    python ppltest.py --nproc 3 --gpus 4,5,6              # specific GPUs, free port
    python ppltest.py                                      # single-GPU fallback
    python ppltest.py --gpus 3                             # single specific GPU

    # Manual torchrun (you must pick a free port yourself if 29500 is taken):
    torchrun --nproc_per_node=8 --master-port=29501 ppltest.py
    torchrun --nproc_per_node=3 --master-port=29501 ppltest.py --gpus 0,2,5

Predefined setups (same numbering as ppl_batch.py):
    python ppltest.py --list                               # list all setups
    python ppltest.py --setup 6                            # MXFP8 + MSD B=16
    python ppltest.py --nproc 3 --gpus 4,5,6 --setup 2

Each GPU loads a full model copy and processes a shard of the 578 sliding
windows.  Partial NLL sums are aggregated via NCCL all_reduce.

Metrics reported:
  - Token-level Perplexity  (standard NLP metric)
  - Word-level  Perplexity  (normalized by whitespace-split word count)
  - Bits-per-Byte   (BPB)   (compression-quality measure)
  - Bits-per-Char   (BPC)
  - Average NLL (nats)
  - Throughput  (tokens / second)
  - Peak GPU/CPU memory
  - Per-chunk NLL stats  (mean +/- std, useful for reliability)

Fix vs. original script:
  The original averaged per-window losses directly.  Because windows have
  different numbers of scored tokens (trg_len), that is WRONG.  The correct
  approach is to accumulate  (loss * trg_len)  and divide by total scored tokens.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
    all_reduce_sum,
    barrier,
    cleanup_distributed,
    gather_list,
    init_distributed,
    is_main,
    maybe_relaunch_with_torchrun,
    restrict_gpus,
    suppress_warnings,
)
from experiment_config import (
    BASELINE_CONFIG,
    SETUPS,
    apply_config,
    format_config_banner,
    get_active_flags,
    get_config_snapshot,
    peak_memory_str,
    reconfigure_mlp_layers,
    reset_peak_memory,
    reset_to_baseline,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "../Qwen3-0.6B"
DATASET     = ("wikitext", "wikitext-2-raw-v1", "test")   # (name, config, split)
MAX_LENGTH  = 4096   # max context window fed to the model
STRIDE      = 512    # sliding-window stride (<= MAX_LENGTH)
RESULTS_OUT = "ppl_results_MXFP8_MSD_B16.json"
# ─────────────────────────────────────────────────────────────────────────────


def precompute_windows(seq_len: int, max_length: int, stride: int):
    """
    Pre-compute all sliding windows as a list of (begin_loc, end_loc, trg_len).

    trg_len is the number of *newly scored* tokens in each window
    (tokens not masked by the -100 label).
    """
    windows = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        windows.append((begin_loc, end_loc, trg_len))
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return windows


def main():
    # ── 0. Parse args & setup selection ──────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation (single setup)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python ppltest.py --list                                # list all setups
  python ppltest.py --nproc 8 --setup 6                   # MXFP8 + MSD B=16
  python ppltest.py --nproc 4 --gpus 4,5,6,7 --setup 6   # specific GPUs
  python ppltest.py --setup 6 --calibration calibration_MXFP8.json

calibration workflow:
  1. python calibrate.py --setup 1          # produce calibration_MXFP8.json
  2. python ppltest.py --setup 6 --calibration calibration_MXFP8.json
""",
    )
    parser.add_argument("--nproc", type=int, default=None, metavar="N",
                        help="Number of GPU workers. Auto-launches via torchrun "
                             "with a free port (avoids EADDRINUSE). "
                             "Replaces manual 'torchrun --nproc_per_node=N'.")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated physical GPU IDs to use, e.g. '4,5,6,7'. "
                             "Must match --nproc (or --nproc_per_node if using torchrun).")
    parser.add_argument("--setup", type=int, default=None, metavar="ID",
                        help="Use a predefined setup by ID (same numbering as ppl_batch.py). "
                             "See --list for available setups. If omitted, uses config.json as-is.")
    parser.add_argument("--list", action="store_true",
                        help="List all predefined setups and exit")
    parser.add_argument("--output", type=str, default=None, metavar="FILE",
                        help="Override output JSON filename "
                             "(default: auto-generated from setup tag, "
                             "or RESULTS_OUT constant if --setup is not used)")
    parser.add_argument("--calibration", type=str, default=None, metavar="FILE",
                        help="Path to calibration JSON file (produced by calibrate.py). "
                             "Injects per-channel B_base budgets into the model config, "
                             "replacing the uniform msd_cycle_budget with per-channel "
                             "Tier B budgets. Requires --setup with an MSD-enabled setup "
                             "(use_msd_truncation will be auto-enabled with a warning "
                             "if not). Example: --calibration calibration_MXFP8.json")
    parser.add_argument("--detail-layer", type=int, default=2, metavar="L",
                        help="Transformer layer index for which full per-channel "
                             "MSD performance detail is recorded in the output JSON. "
                             "All other layers only get compact summaries. (default: 2)")
    parser.add_argument("--limit-samples", type=int, default=None, metavar="N",
                        help="Light mode: limit the number of dataset samples to read up front, "
                             "greatly accelerating testing while preserving valid subset metrics.")
    args = parser.parse_args()

    # ── List mode (no GPU needed) ──
    if args.list:
        print(f"{'ID':>3}  {'Tag':<30}  Description")
        print("-" * 75)
        for sid, tag, desc, _ in SETUPS:
            result_file = Path(__file__).parent / f"ppl_results_{tag}.json"
            exists = "  (done)" if result_file.exists() else ""
            print(f"{sid:3d}  {tag:<30}  {desc}{exists}")
        return

    # ── Validate --setup ──
    selected_setup = None
    if args.setup is not None:
        selected_setup = next((s for s in SETUPS if s[0] == args.setup), None)
        if selected_setup is None:
            print(f"Unknown setup ID: {args.setup}. "
                  f"Valid IDs: {[s[0] for s in SETUPS]}. Use --list to see all.")
            return

    # ── Validate --calibration ──
    if args.calibration is not None and args.setup is None:
        print("Error: --calibration requires --setup to select an MSD-enabled setup.")
        return
    cal_data_dict = None
    if args.calibration is not None:
        cal_path = Path(args.calibration)
        if not cal_path.exists():
            print(f"Error: calibration file not found: {cal_path}")
            return
        with open(cal_path) as f:
            cal_json = json.load(f)
        cal_data_dict = cal_json.get("msd_calibration_data")
        if not cal_data_dict:
            print(f"Error: no 'msd_calibration_data' key in {cal_path}")
            return

    restrict_gpus(args.gpus)
    maybe_relaunch_with_torchrun(args.nproc)

    # ── 1. Distributed init ──────────────────────────────────────────────────
    rank, world_size, local_rank, device = init_distributed()
    suppress_warnings(rank)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if is_main(rank):
        print(f"World size: {world_size}  |  Device: {device}  |  dtype: {dtype}")

    # ── 2. Load model ─────────────────────────────────────────────────────
    if is_main(rank):
        print("Loading tokenizer & model ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    model_kwargs: dict = {"local_files_only": True, "dtype": dtype}
    # Each rank loads onto its own GPU -- no device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    model.to(device)
    model.eval()
    reset_peak_memory(device)

    if is_main(rank):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {MODEL_PATH}  |  Params: {num_params/1e6:.1f}M")

    # ── 3. Apply setup config (if --setup given) ─────────────────────────────
    if selected_setup is not None:
        sid, tag, desc, overrides = selected_setup
        reset_to_baseline(model.config)
        apply_config(model.config, overrides)
        reconfigure_mlp_layers(model, device)
        if is_main(rank):
            print(format_config_banner(model.config, setup_id=sid, setup_desc=desc))

    # ── 3b. Inject calibration data (if --calibration given) ─────────────────
    if cal_data_dict is not None:
        # Ensure MSD truncation is enabled
        if not getattr(model.config, "use_msd_truncation", False):
            print("Warning: --calibration given but setup has use_msd_truncation=False. "
                  "Enabling MSD truncation.")
            model.config.use_msd_truncation = True
        model.config.msd_calibration_data = cal_data_dict
        # Invalidate MSD context so it rebuilds with calibrated budgets
        if hasattr(model, "_msd_context"):
            model._msd_context = None
            model._msd_context_config_hash = None
        if is_main(rank):
            n_layers = len(cal_data_dict)
            n_channels = sum(len(v) for v in cal_data_dict.values())
            print(f"Calibration loaded: {args.calibration} ({n_layers} layers, {n_channels} channels)")

    # Determine output filename
    if args.output:
        results_out = args.output
    elif selected_setup is not None:
        tag = selected_setup[1]
        calib_suffix = "_calib" if cal_data_dict is not None else ""
        results_out = f"ppl_results_{tag}{calib_suffix}.json"
    else:
        results_out = RESULTS_OUT

    # ── 4. Load & encode dataset ─────────────────────────────────────────
    ds_name, ds_config, ds_split = DATASET
    if is_main(rank):
        print(f"\nLoading dataset: {ds_name}/{ds_config} ({ds_split}) ...")

    test_data = load_dataset(ds_name, ds_config, split=ds_split)
    total_samples = len(test_data)
    if args.limit_samples is not None:
        if is_main(rank):
            print(f"Light mode enabled: limiting to first {args.limit_samples} samples out of {total_samples} total samples.")
        test_data = test_data.select(range(min(args.limit_samples, total_samples)))
    else:
        if is_main(rank):
            print(f"Processing all {total_samples} samples from the dataset.")

    raw_text  = "\n\n".join(test_data["text"])

    encodings = tokenizer(raw_text, return_tensors="pt")
    seq_len   = encodings.input_ids.size(1)

    num_words = len(raw_text.split())
    num_chars = len(raw_text)
    num_bytes = len(raw_text.encode("utf-8"))

    if is_main(rank):
        print(f"Tokens: {seq_len:,}  |  Words: {num_words:,}  |  Chars: {num_chars:,}  |  Bytes: {num_bytes:,}")

    # ── 5. Pre-compute & shard windows ───────────────────────────────────────
    all_windows = precompute_windows(seq_len, MAX_LENGTH, STRIDE)
    my_windows = all_windows[rank::world_size]
    
    # Pad iterations to prevent NCCL Watchdog Timeout when len(all_windows) % world_size != 0
    max_windows = math.ceil(len(all_windows) / world_size)
    dummy_window = (0, 10, 0)  # (begin_loc, end_loc, trg_len=0) for dummy tasks
    while len(my_windows) < max_windows:
        my_windows.append(dummy_window)

    if is_main(rank):
        print(f"\nTotal windows: {len(all_windows)}  |  Windows per rank: ~{len(my_windows)}")
        print(f"Computing PPL  (max_length={MAX_LENGTH}, stride={STRIDE}) ...")

    barrier()

    # ── 6. Sliding-window PPL (distributed) ──────────────────────────────────
    local_nll_sum  = 0.0
    local_tokens   = 0
    local_chunk_nlls = []

    t_start = time.perf_counter()

    iterator = tqdm(my_windows, desc=f"[rank {rank}] PPL windows", disable=not is_main(rank))
    for win_idx, (begin_loc, end_loc, trg_len) in enumerate(iterator):
        is_dummy = (trg_len == 0)
        
        input_ids  = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        if not is_dummy:
            target_ids[:, :-trg_len] = -100
        else:
            target_ids[:] = -100 # mask all for dummy task

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            avg_nll = outputs.loss.item() if not is_dummy else 0.0

        if not is_dummy:
            local_nll_sum  += avg_nll * trg_len
            local_tokens   += trg_len
            local_chunk_nlls.append(avg_nll)

        # Free tensors and periodically release cached memory to prevent OOM
        del input_ids, target_ids, outputs
        if win_idx % 20 == 19:
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t_start

    # ── 7. Aggregate across ranks ────────────────────────────────────────────
    agg = torch.tensor([local_nll_sum, float(local_tokens)],
                       dtype=torch.float64, device=device)
    all_reduce_sum(agg)
    total_nll_sum = agg[0].item()
    total_tokens  = int(agg[1].item())

    # Gather per-chunk NLLs for reliability stats (all -> rank 0)
    all_chunk_nlls = gather_list(local_chunk_nlls, rank, world_size)

    # ── 7b. Collect MSD performance statistics (rank 0 only) ─────────────────
    msd_perf = None
    if is_main(rank) and hasattr(model, "get_perf_stats"):
        msd_perf = model.get_perf_stats(detail_layer=args.detail_layer)

    # ── 8. Compute & report metrics (rank 0 only) ───────────────────────────
    if is_main(rank):
        mean_nll = total_nll_sum / total_tokens

        token_ppl = math.exp(mean_nll)
        word_ppl  = math.exp(mean_nll * total_tokens / num_words)
        bpb       = (mean_nll * total_tokens / num_bytes) / math.log(2)
        bpc       = (mean_nll * total_tokens / num_chars) / math.log(2)

        throughput = total_tokens / elapsed   # per-rank timing (conservative)

        chunk_arr = np.array(all_chunk_nlls)
        chunk_mean, chunk_std = chunk_arr.mean(), chunk_arr.std()

        mem_str = peak_memory_str(device)

        SEP = "-" * 52
        print(f"\n{SEP}")
        print(f"{'PERPLEXITY EVALUATION RESULTS':^52}")
        print(SEP)
        print(f"  Dataset          : {ds_name}/{ds_config} ({ds_split})")
        print(f"  Model            : {MODEL_PATH}")
        print(f"  Scored tokens    : {total_tokens:,}  (stride={STRIDE})")
        print(f"  GPUs used        : {world_size}")
        print(SEP)
        print(f"  Token Perplexity : {token_ppl:.4f}")
        print(f"  Word  Perplexity : {word_ppl:.4f}")
        print(f"  Bits-per-Byte    : {bpb:.4f}")
        print(f"  Bits-per-Char    : {bpc:.4f}")
        print(f"  Mean NLL (nats)  : {mean_nll:.4f}")
        print(SEP)
        print(f"  Chunks evaluated : {len(all_chunk_nlls)}")
        print(f"  Chunk NLL mean   : {chunk_mean:.4f}")
        print(f"  Chunk NLL std    : {chunk_std:.4f}  (lower = more stable)")
        print(SEP)
        print(f"  Throughput       : {throughput:.1f} tokens/s")
        print(f"  Wall time        : {elapsed:.1f}s")
        print(f"  Peak memory      : {mem_str}")
        print(SEP)

        results = {
            "model": MODEL_PATH,
            "dataset": f"{ds_name}/{ds_config}/{ds_split}",
            "config": {"max_length": MAX_LENGTH, "stride": STRIDE, "dtype": str(dtype),
                       "world_size": world_size,
                       "calibration_file": args.calibration if args.calibration else None},
            "config_snapshot": get_config_snapshot(model.config),
            "metrics": {
                "token_perplexity": round(token_ppl, 4),
                "word_perplexity":  round(word_ppl,  4),
                "bits_per_byte":    round(bpb, 4),
                "bits_per_char":    round(bpc, 4),
                "mean_nll_nats":    round(mean_nll, 4),
            },
            "reliability": {
                "num_chunks":     len(all_chunk_nlls),
                "chunk_nll_mean": round(float(chunk_mean), 4),
                "chunk_nll_std":  round(float(chunk_std),  4),
                "scored_tokens":  total_tokens,
            },
            "performance": {
                "throughput_tokens_per_sec": round(throughput, 1),
                "wall_time_sec":             round(elapsed, 2),
                "peak_memory":               mem_str,
            },
        }

        # ── 8b. Attach MSD performance statistics if available ────────────────
        if msd_perf is not None:
            results["msd_perf_stats"] = msd_perf
            g = msd_perf.get("global", {})
            print(f"\n{'MSD PERFORMANCE STATISTICS':^52}")
            print(SEP)
            print(f"  Layers profiled  : {g.get('num_layers', 0)}")
            print(f"  MAC sparsity     : {g.get('mac_sparsity', 0):.4%}")
            print(f"  Active MACs      : {g.get('active_macs', 0):,} / {g.get('total_macs', 0):,}")
            print(f"  Mean eff. prec.  : {g.get('mean_effective_precision', 0):.2f} digits")
            print(f"  Active eff. prec.: {g.get('active_p_eff_mean', 0):.2f} digits")
            print(f"  Global util.     : {g.get('global_utilization', 0):.4%}")
            print(f"  Zero blocks      : {g.get('zero_block_ratio', 0):.4%}")
            print(f"  Partial blocks   : {g.get('partial_block_ratio', 0):.4%}")
            print(f"  Full blocks      : {g.get('full_block_ratio', 0):.4%}")
            print(f"  Max budget (lat) : {g.get('max_budget', 0):.1f} cycles")
            print(f"  Max total delay  : {g.get('max_total_delay', 0):.1f} cycles")
            print(SEP)

            # ── Per-layer summary table ──
            pl = msd_perf.get("per_layer", {})
            if pl:
                HDR = f"  {'Layer':<40s}  p_eff  util%  mac_sp%  zero_blk%  max_B"
                print(f"\n  Per-layer summary (detail_layer={args.detail_layer}):")
                print(HDR)
                print(f"  {'-'*80}")
                for lname, ldata in pl.items():
                    s = ldata.get("summary", {})
                    bl = s.get("bit_level", {})
                    cl = s.get("channel_level", {})
                    ml = s.get("mac_level", {})
                    bkl = s.get("block_level", {})
                    # Truncate long layer names
                    short = lname if len(lname) <= 40 else lname[:18] + ".." + lname[-18:]
                    p_eff_v = bl.get("p_eff_mean", 0)
                    util_v = cl.get("utilization_mean", 0) * 100
                    mac_sp_v = ml.get("mac_sparsity", 0) * 100
                    zblk_v = bkl.get("zero_block_ratio", 0) * 100
                    max_b_v = cl.get("max_budget", 0)
                    detail_tag = " *" if "channel_detail" in ldata else ""
                    print(f"  {short:<40s}  {p_eff_v:5.1f}  {util_v:5.1f}  {mac_sp_v:6.2f}   {zblk_v:6.2f}  {max_b_v:5.1f}{detail_tag}")
                print(f"  (* = channel detail in JSON for --detail-layer={args.detail_layer})")
            print(SEP)

        out_path = Path(__file__).parent / results_out
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved -> {out_path}")

    # ── 9. Cleanup calibration data from config ─────────────────────────────
    if cal_data_dict is not None:
        model.config.msd_calibration_data = None
    # Reset perf stats after saving
    if hasattr(model, "reset_perf_stats"):
        model.reset_perf_stats()
    cleanup_distributed()


if __name__ == "__main__":
    main()
