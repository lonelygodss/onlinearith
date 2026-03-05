"""
Perplexity (PPL) evaluation script with richer metrics and reliable computation.

Supports **multi-GPU** via torchrun:
    torchrun --nproc_per_node=8 ppltest.py        # 8-GPU parallel
    python ppltest.py                              # single-GPU fallback
    torchrun --nproc_per_node=3 ppltest.py --gpus 0,2,5   # specific GPUs
    python ppltest.py --gpus 3                             # single specific GPU

Predefined setups (same numbering as ppl_batch.py):
    python ppltest.py --list                               # list all setups
    python ppltest.py --setup 6                            # MXFP8 + MSD B=16
    torchrun --nproc_per_node=3 ppltest.py --gpus 0,2,5 --setup 2

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
    restrict_gpus,
)
from ppl_batch import (
    SETUPS,
    _BASELINE_OVERRIDES,
    apply_config,
    reconfigure_mlp_layers,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "../Qwen3-0.6B"
DATASET     = ("wikitext", "wikitext-2-raw-v1", "test")   # (name, config, split)
MAX_LENGTH  = 4096   # max context window fed to the model
STRIDE      = 512    # sliding-window stride (<= MAX_LENGTH)
RESULTS_OUT = "ppl_results_MXFP8_MSD_B16.json"
# ─────────────────────────────────────────────────────────────────────────────


def peak_memory_str(device: torch.device) -> str:
    """Return peak allocated memory as a human-readable string."""
    if device.type == "cuda":
        bytes_ = torch.cuda.max_memory_allocated(device)
        return f"{bytes_ / 1024**3:.2f} GB"
    return "N/A (CPU)"


def reset_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


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
    parser = argparse.ArgumentParser(description="Perplexity evaluation (single setup)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated physical GPU IDs to use, e.g. '0,2,5'. "
                             "Must match --nproc_per_node when using torchrun.")
    parser.add_argument("--setup", type=int, default=None, metavar="ID",
                        help="Use a predefined setup by ID (same numbering as ppl_batch.py). "
                             "See --list for available setups. If omitted, uses config.json as-is.")
    parser.add_argument("--list", action="store_true",
                        help="List all predefined setups and exit")
    parser.add_argument("--output", type=str, default=None, metavar="FILE",
                        help="Override output JSON filename "
                             "(default: auto-generated from setup tag, "
                             "or RESULTS_OUT constant if --setup is not used)")
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

    restrict_gpus(args.gpus)

    # ── 1. Distributed init ──────────────────────────────────────────────────
    rank, world_size, local_rank, device = init_distributed()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if is_main(rank):
        print(f"World size: {world_size}  |  Device: {device}  |  dtype: {dtype}")

    # ── 2. Load model ─────────────────────────────────────────────────────
    if is_main(rank):
        print("Loading tokenizer & model ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    model_kwargs: dict = {"local_files_only": True, "torch_dtype": dtype}
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
        apply_config(model.config, _BASELINE_OVERRIDES)
        apply_config(model.config, overrides)
        reconfigure_mlp_layers(model, device)
        if is_main(rank):
            active_flags = []
            for k in ["use_mxfp8", "use_mxfp6", "use_mxfp4", "use_msd_truncation",
                       "msd_cycle_budget", "msd_deep_pipeline"]:
                v = getattr(model.config, k, None)
                if v is not None and v is not False:
                    active_flags.append(f"{k}={v}")
            print(f"Setup #{sid}: {desc}")
            print(f"Config: {', '.join(active_flags) or '(baseline fp16)'}")

    # Determine output filename
    if args.output:
        results_out = args.output
    elif selected_setup is not None:
        results_out = f"ppl_results_{selected_setup[1]}.json"
    else:
        results_out = RESULTS_OUT

    # ── 4. Load & encode dataset ─────────────────────────────────────────
    ds_name, ds_config, ds_split = DATASET
    if is_main(rank):
        print(f"\nLoading dataset: {ds_name}/{ds_config} ({ds_split}) ...")

    test_data = load_dataset(ds_name, ds_config, split=ds_split)
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
    for begin_loc, end_loc, trg_len in iterator:
        input_ids  = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            avg_nll = outputs.loss.item()

        local_nll_sum  += avg_nll * trg_len
        local_tokens   += trg_len
        local_chunk_nlls.append(avg_nll)

    elapsed = time.perf_counter() - t_start

    # ── 7. Aggregate across ranks ────────────────────────────────────────────
    agg = torch.tensor([local_nll_sum, float(local_tokens)],
                       dtype=torch.float64, device=device)
    all_reduce_sum(agg)
    total_nll_sum = agg[0].item()
    total_tokens  = int(agg[1].item())

    # Gather per-chunk NLLs for reliability stats (all -> rank 0)
    all_chunk_nlls = gather_list(local_chunk_nlls, rank, world_size)

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
                       "world_size": world_size},
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

        out_path = Path(__file__).parent / results_out
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved -> {out_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
