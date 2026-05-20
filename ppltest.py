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
    python ppltest.py --setup 6 --limit-samples 100        # light mode: fast run for testing

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
    clear_mxfp_weight_cache,
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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = default_model_path("Qwen3-8B")
DATASET     = ("wikitext", "wikitext-2-raw-v1", "test")   # (name, config, split)
MAX_LENGTH  = 4096   # max context window fed to the model
STRIDE      = 512    # sliding-window stride (<= MAX_LENGTH)
RESULTS_OUT = "ppl_results_MXFP8_MSD_B16.json"
RESULTS_ROOT = Path("../data/calib-data")

def _load_text_manifest(manifest_path: Path) -> list[str]:
    """
    Load evaluation texts from a manifest file.

    Supported formats:
      - .json   : ["text", ...] or {"texts": [...]} or [{"text": ...}, ...]
      - .jsonl  : one JSON value/object per line (string or {"text": ...})
      - others  : plain text split by blank lines (fallback to non-empty lines)
    """
    suffix = manifest_path.suffix.lower()

    def _normalize(items):
        out = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text = item["text"].strip()
            else:
                continue
            if text:
                out.append(text)
        return out

    if suffix == ".json":
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("texts", [])
        if not isinstance(data, list):
            raise ValueError(f"JSON manifest must be a list or contain 'texts': {manifest_path}")
        texts = _normalize(data)
    elif suffix == ".jsonl":
        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        texts = _normalize(rows)
    else:
        raw = manifest_path.read_text(encoding="utf-8")
        chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
        if len(chunks) <= 1:
            chunks = [c.strip() for c in raw.splitlines() if c.strip()]
        texts = chunks

    if not texts:
        raise ValueError(f"No non-empty texts found in manifest: {manifest_path}")
    return texts


def main():
    # ── 0. Parse args & setup selection ──────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation (single setup)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python ppltest.py --list                                # list all setups
  python ppltest.py --nproc 8 --setup 6                   # MXFP8 + MSD B=16
  python ppltest.py --nproc 4 --gpus 4,5,6,7 --setup 6    # specific GPUs
  python ppltest.py --setup 6 --limit-samples 20          # light mode: fast run
  python ppltest.py --setup 6 --calibration calibration_MXFP8.json
  python ppltest.py --setup 6 --calibration calibration_MXFP8.json --lite  # fast stats

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
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, metavar="DIR",
                        help=f"Local model directory (default: {MODEL_PATH})")
    parser.add_argument("--results-dir", type=str, default=None, metavar="DIR",
                        help=f"Directory for auto-generated result JSON files (default: {RESULTS_ROOT})")
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
                        help="Light mode: limit the number of dataset text samples to process "
                             "for a faster evaluation. Keeps all statistic formulas unchanged.")
    parser.add_argument("--text-manifest", type=str, default=None, metavar="FILE",
                        help="Optional manifest file with evaluation texts. If set, "
                            "bypasses dataset loading and uses these texts directly.")
    parser.add_argument("--lite", action="store_true",
                        help="Lite stats mode: collect only utilization, zero-block%%, "
                             "mean p_eff, and max latency per layer.  Skips expensive "
                             "NAF-width computation and detailed histograms for faster "
                             "MSD inference.  PPL results are identical to full mode.")
    parser.add_argument("--stats", choices=["off", "lite", "full"], default="off",
                        help="MSD performance-statistics mode for PPL runs. "
                             "--lite maps to --stats lite. (default: off)")
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
                        help="Optional path updated atomically with the latest MX/MSD progress event.")
    parser.add_argument("--figure5-layer-cycles", action="store_true",
                        help="Enable Figure 5 layer-cycle profiling in lite stats mode. "
                             "Records per-layer cycle moments from block-serial, "
                             "channel-parallel execution estimates.")
    args = parser.parse_args()
    model_path = args.model_path
    results_root = normalize_output_dir(args.results_dir, RESULTS_ROOT)

    if args.lite:
        args.stats = "lite"
    auto_enabled_lite = args.figure5_layer_cycles and args.stats != "lite"
    if args.figure5_layer_cycles:
        args.stats = "lite"

    # ── List mode (no GPU needed) ──
    if args.list:
        print(f"{'ID':>3}  {'Tag':<30}  Description")
        print("-" * 75)
        list_results_dir = results_root if args.results_dir is not None else Path(__file__).parent
        for sid, tag, desc, _ in SETUPS:
            result_file = list_results_dir / f"ppl_results_{tag}.json"
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

    if not Path(model_path).exists():
        print(f"Error: {describe_missing_model_path(model_path)}")
        cleanup_distributed()
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    model_kwargs: dict = {"local_files_only": True, "dtype": dtype}
    # Each rank loads onto its own GPU -- no device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    reset_peak_memory(device)

    if is_main(rank):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_path}  |  Params: {num_params/1e6:.1f}M")

    # ── 3. Apply setup config (if --setup given) ─────────────────────────────
    if selected_setup is not None:
        sid, tag, desc, overrides = selected_setup
        reset_to_baseline(model.config)
        apply_config(model.config, overrides)
        if args.mx_chunk_target_mib is not None:
            model.config.mxfp_chunk_target_mib = args.mx_chunk_target_mib
            model.config.mxfp_use_chunked_exact = True
        if args.msd_chunk_target_mib is not None:
            model.config.msd_chunk_target_mib = args.msd_chunk_target_mib
        if args.weight_cache_dtype is not None:
            model.config.mxfp_weight_cache_dtype = args.weight_cache_dtype
        reconfigure_mlp_layers(model, device)
        if is_main(rank):
            print(format_config_banner(model.config, setup_id=sid, setup_desc=desc))
    else:
        if args.mx_chunk_target_mib is not None:
            model.config.mxfp_chunk_target_mib = args.mx_chunk_target_mib
            model.config.mxfp_use_chunked_exact = True
        if args.msd_chunk_target_mib is not None:
            model.config.msd_chunk_target_mib = args.msd_chunk_target_mib
        if args.weight_cache_dtype is not None:
            model.config.mxfp_weight_cache_dtype = args.weight_cache_dtype
        clear_mxfp_weight_cache(model)

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

    # ── 3c. Configure stats mode ─────────────────────────────────────────────
    if auto_enabled_lite and is_main(rank):
        print("Figure 5 cycle profiling requested; enabling lite stats mode.")
    if args.stats == "off":
        model.config.msd_perf_stats_enabled = False
        model.config.msd_perf_stats_lite = False
        if is_main(rank):
            print("MSD stats disabled for PPL numerical run.")
    elif args.stats == "lite":
        model.config.msd_perf_stats_enabled = True
        model.config.msd_perf_stats_lite = True
        if is_main(rank):
            print("Lite stats mode: skipping NAF-width & histograms for faster inference.")
    else:
        model.config.msd_perf_stats_enabled = True
        model.config.msd_perf_stats_lite = False
        if is_main(rank):
            print("Full MSD stats mode enabled.")
    if args.figure5_layer_cycles:
        model.config.msd_figure5_layer_cycles = True
        if is_main(rank):
            print("Figure 5 layer-cycle profiling enabled.")
    # Disable stats entirely on non-rank-0 processes (saves compute & memory;
    # rank 0's window sample is representative for aggregate stats).
    if not is_main(rank):
        model.config.msd_perf_stats_enabled = False
    # Invalidate cached MSD context so it picks up the new flags
    if hasattr(model, "_msd_context"):
        model._msd_context = None
        model._msd_context_config_hash = None

    if is_main(rank):
        install_mxfp_progress_hook(
            model,
            interval_sec=args.mxfp_progress_interval_sec,
            progress_file=args.mxfp_progress_file,
            device=device,
            extra={"rank": rank, "setup": selected_setup[0] if selected_setup is not None else None},
        )
    else:
        clear_mxfp_progress_hook(model)

    # Determine output filename
    if args.output:
        results_out = args.output
    elif selected_setup is not None:
        tag = selected_setup[1]
        calib_suffix = "_calib" if cal_data_dict is not None else ""
        results_dir = cal_path.parent if cal_data_dict is not None else results_root
        results_out = results_dir / f"ppl_results_{tag}{calib_suffix}.json"
    else:
        results_out = RESULTS_OUT

    # ── 4. Load & encode dataset/text manifest ───────────────────────────
    manifest_path = None
    if args.text_manifest:
        manifest_path = Path(args.text_manifest).expanduser().resolve()
        if not manifest_path.exists():
            print(f"Error: text manifest not found: {manifest_path}")
            cleanup_distributed()
            return

        try:
            all_samples = _load_text_manifest(manifest_path)
        except Exception as exc:
            print(f"Error: failed to read text manifest: {exc}")
            cleanup_distributed()
            return

        dataset_display = f"manifest ({manifest_path})"
        dataset_label = f"manifest:{manifest_path}"
        if is_main(rank):
            print(f"\nLoading evaluation texts from manifest: {manifest_path}")
    else:
        ds_name, ds_config, ds_split = DATASET
        if is_main(rank):
            print(f"\nLoading dataset: {ds_name}/{ds_config} ({ds_split}) ...")

        test_data = load_dataset(ds_name, ds_config, split=ds_split)
        all_samples = test_data["text"]
        dataset_display = f"{ds_name}/{ds_config} ({ds_split})"
        dataset_label = f"{ds_name}/{ds_config}/{ds_split}"

    total_samples = len(all_samples)
    if args.limit_samples is not None:
        if is_main(rank):
            print(f"\nLight mode enabled: limiting to first {args.limit_samples} samples out of {total_samples} total samples.")
        samples = all_samples[:args.limit_samples]
    else:
        if is_main(rank):
            print(f"\nProcessing all {total_samples} samples from the selected source.")
        samples = all_samples

    raw_text  = "\n\n".join(samples)

    encodings = tokenizer(raw_text, return_tensors="pt")
    seq_len   = encodings.input_ids.size(1)

    num_words = len(raw_text.split())
    num_chars = len(raw_text)
    num_bytes = len(raw_text.encode("utf-8"))

    if is_main(rank):
        print(f"Tokens: {seq_len:,}  |  Words: {num_words:,}  |  Chars: {num_chars:,}  |  Bytes: {num_bytes:,}")

    # ── 5. Pre-compute & shard windows ───────────────────────────────────────
    all_windows = precompute_windows(seq_len, MAX_LENGTH, STRIDE)
    
    # Pad all_windows so length is divisible by world_size to avoid NCCL hangs
    orig_num_windows = len(all_windows)
    pad_len = (world_size - (orig_num_windows % world_size)) % world_size
    if pad_len > 0:
        # Use the last window as a dummy. We add a boolean flag to all tuples.
        dummy_window = all_windows[-1]
        all_windows = [(w[0], w[1], w[2], False) for w in all_windows]
        for _ in range(pad_len):
            all_windows.append((dummy_window[0], dummy_window[1], dummy_window[2], True))
    else:
        all_windows = [(w[0], w[1], w[2], False) for w in all_windows]

    my_windows = all_windows[rank::world_size]

    if is_main(rank):
        print(f"\nTotal windows: {orig_num_windows} (+{pad_len} padded)  |  Windows per rank: {len(my_windows)}")
        print(f"Computing PPL  (max_length={MAX_LENGTH}, stride={STRIDE}) ...")

    barrier()

    # ── 6. Sliding-window PPL (distributed) ──────────────────────────────────
    local_nll_sum  = 0.0
    local_tokens   = 0
    local_chunk_nlls = []
    SYNC_EVERY = 10   # barrier every N windows to cap cumulative rank drift

    t_start = time.perf_counter()

    iterator = tqdm(my_windows, desc=f"[rank {rank}] PPL windows", disable=not is_main(rank))
    for win_idx, (begin_loc, end_loc, trg_len, is_dummy) in enumerate(iterator):
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

        if not is_dummy:
            local_nll_sum, local_tokens = accumulate_weighted_nll(
                local_nll_sum, local_tokens, avg_nll, trg_len
            )
            local_chunk_nlls.append(avg_nll)

        # Free tensors and periodically release cached memory to prevent OOM
        del input_ids, target_ids, loss_kwargs, outputs
        if win_idx % 100 == 99:
            torch.cuda.empty_cache()

        # Periodic barrier: prevent cumulative timing drift across ranks from
        # exceeding the NCCL timeout.  Without this, small per-window timing
        # differences accumulate over hundreds of windows (29+ hours), causing
        # faster ranks to reach the final all_reduce >10 min before slower ones.
        if world_size > 1 and (win_idx + 1) % SYNC_EVERY == 0:
            barrier()

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
    if args.stats != "off" and is_main(rank) and hasattr(model, "get_perf_stats"):
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
        print(f"  Dataset          : {dataset_display}")
        print(f"  Model            : {model_path}")
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
            "model": model_path,
            "dataset": dataset_label,
            "config": {"max_length": MAX_LENGTH, "stride": STRIDE, "dtype": str(dtype),
                       "world_size": world_size,
                       "calibration_file": args.calibration if args.calibration else None,
                       "text_manifest": str(manifest_path) if manifest_path else None,
                       "stats": args.stats},
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
            pl = msd_perf.get("per_layer", {})
            is_lite = args.stats == "lite"

            if is_lite:
                # ── Lite mode output ──
                print(f"\n{'MSD PERFORMANCE STATISTICS (LITE)':^52}")
                print(SEP)
                print(f"  Layers profiled  : {g.get('num_layers', 0)}")
                print(f"  MAC sparsity     : {g.get('mac_sparsity', 0):.4%}")
                print(f"  Mean eff. prec.  : {g.get('mean_effective_precision', 0):.2f} digits")
                print(f"  Global util.     : {g.get('global_utilization', 0):.4%}")
                print(f"  HW lat. overhead : {g.get('hw_latency_overhead', 0):.4%}")
                print(f"  Zero blocks      : {g.get('zero_block_ratio', 0):.4%}")
                print(f"  Max budget       : {g.get('max_budget', 0):.1f} cycles")
                print(f"  Max total delay  : {g.get('max_total_delay', 0):.1f} cycles")
                print(SEP)

                if pl:
                    HDR = f"  {'Layer':<40s}  p_eff  util%  hw_lat%  zero_blk%  avg_B"
                    print(f"\n  Per-layer summary (lite):")
                    print(HDR)
                    print(f"  {'-'*80}")
                    for lname, ldata in pl.items():
                        short = lname if len(lname) <= 40 else lname[:18] + ".." + lname[-18:]
                        p_eff_v = ldata.get("p_eff_mean", 0)
                        util_v = ldata.get("utilization", 0) * 100
                        hw_v = ldata.get("hw_latency_overhead", 0) * 100
                        zblk_v = ldata.get("zero_block_ratio", 0) * 100
                        avg_b_v = ldata.get("avg_max_latency", 0)
                        print(f"  {short:<40s}  {p_eff_v:5.1f}  {util_v:5.1f}  {hw_v:6.2f}   {zblk_v:6.2f}  {avg_b_v:5.1f}")

                    has_figure5 = any("avg_layer_cycle" in ldata for ldata in pl.values())
                    if has_figure5:
                        HDR2 = f"  {'Layer':<40s}  layer_avg  ch_mean  layer_std  ch_std_avg"
                        print(f"\n  Figure 5 cycle summary:")
                        print(HDR2)
                        print(f"  {'-'*88}")
                        for lname, ldata in pl.items():
                            short = lname if len(lname) <= 40 else lname[:18] + ".." + lname[-18:]
                            layer_avg = ldata.get("avg_layer_cycle", 0)
                            ch_mean = ldata.get("avg_mean_channel_cycle", 0)
                            layer_std = ldata.get("layer_cycle_std", 0)
                            ch_std = ldata.get("avg_std_channel_cycle", 0)
                            print(
                                f"  {short:<40s}  {layer_avg:9.3f}  {ch_mean:7.3f}  "
                                f"{layer_std:9.3f}  {ch_std:10.3f}"
                            )
                print(SEP)
            else:
                # ── Full mode output ──
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved -> {out_path}")

    # ── 9. Cleanup calibration data from config ─────────────────────────────
    if cal_data_dict is not None:
        model.config.msd_calibration_data = None
    # Reset perf stats after saving
    if hasattr(model, "reset_perf_stats"):
        model.reset_perf_stats()
    clear_mxfp_progress_hook(model)
    clear_mxfp_weight_cache(model)
    cleanup_distributed()


if __name__ == "__main__":
    main()
