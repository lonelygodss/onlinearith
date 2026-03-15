"""
Offline MSD budget calibration across MXFP formats.

Supports **multi-GPU** — either auto-launched (recommended) or via torchrun:
    python calibrate.py --nproc 4                          # auto-launch 4 GPUs (picks free port)
    python calibrate.py --nproc 4 --gpus 4,5,6,7          # specific GPUs, free port
    python calibrate.py --nproc 2 --only 1 2               # subset on 2 GPUs
    python calibrate.py --setup 1                          # single format, single GPU

    # Manual torchrun (you must pick a free port yourself if 29500 is taken):
    torchrun --nproc_per_node=4 --master-port=29501 calibrate.py
    torchrun --nproc_per_node=2 --master-port=29501 calibrate.py --gpus 0,3

Each GPU loads its own model copy and calibrates its assigned format(s).
Formats are partitioned round-robin across ranks; each rank writes its
result file independently (no inter-rank communication during calibration).

Output files:  calibration_{tag}.json  (e.g. calibration_MXFP8.json)

Usage:
    cd /home/xzj/coding/onlinearith
    source /home/xzj/coding/.venv3_10/bin/activate

    python calibrate.py --list                                # list setups
    python calibrate.py --setup 1                             # single format
    python calibrate.py --nproc 4                             # all 4 formats, auto port
    python calibrate.py --nproc 4 --gpus 4,5,6,7             # specific GPUs
    python calibrate.py --nproc 4 --target-snr 40            # custom SNR
    python calibrate.py --nproc 4 --force                    # re-run existing
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.calibration_msd import calibrate_channel_budgets

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
    apply_config,
    format_config_banner,
    get_active_flags,
    reconfigure_mlp_layers,
    reset_to_baseline,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "../Qwen3-0.6B"
RESULTS_DIR = "../data"
CAL_DATASET = ("wikitext", "wikitext-2-raw-v1", "validation")

# ── Calibration setups (one per MXFP format) ─────────────────────────────────
CAL_SETUPS = [
    (1, "MXFP8",      "MXFP8 (E4M3FN)",
     {"use_mxfp8": True}),
    (2, "MXFP6_E2M3", "MXFP6 E2M3",
     {"use_mxfp6": True, "mxfp6_format": "e2m3"}),
    (3, "MXFP6_E3M2", "MXFP6 E3M2",
     {"use_mxfp6": True, "mxfp6_format": "e3m2"}),
    (4, "MXFP4",      "MXFP4 (E2M1)",
     {"use_mxfp4": True}),
]


def _snr_to_dir_name(target_snr: float) -> str:
    # Keep whole-number SNRs compact (e.g., 30db) and preserve decimals otherwise.
    if float(target_snr).is_integer():
        return f"{int(target_snr)}db"
    return f"{target_snr:g}db"


def main():
    parser = argparse.ArgumentParser(
        description="MSD budget calibration for MXFP formats (multi-GPU via torchrun)"
    )
    parser.add_argument("--list", action="store_true",
                        help="List all calibration setups and exit")
    parser.add_argument("--setup", type=int, default=None, metavar="ID",
                        help="Run a single setup by ID (1-4)")
    parser.add_argument("--only", nargs="+", type=int, metavar="ID",
                        help="Run only these setup IDs (e.g. --only 1 2)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if calibration file already exists")
    parser.add_argument("--nproc", type=int, default=None, metavar="N",
                        help="Number of GPU workers. Auto-launches via torchrun "
                             "with a free port (avoids EADDRINUSE). "
                             "Replaces manual 'torchrun --nproc_per_node=N'.")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated physical GPU IDs, e.g. '4,5,6,7'. "
                             "Must match --nproc (or --nproc_per_node if using torchrun).")
    parser.add_argument("--target-snr", type=float, default=30.0,
                        help="Target SNR in dB (default: 30.0). Higher = more budget.")
    parser.add_argument("--num-texts", type=int, default=20,
                        help="Number of calibration paragraphs (default: 20)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length per calibration sample (default: 512)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for calibration forward passes (default: 4)")
    parser.add_argument("--online-delay", type=int, default=2,
                        help="MSD online delay δ (default: 2)")
    parser.add_argument("--detail-layer", type=int, default=2, metavar="L",
                        help="Transformer layer index for full per-channel "
                             "statistics (default: 2). The 3 MLP projections "
                             "(gate/up/down) of this layer get channel-wise "
                             "detail; all other layers only get compact summaries.")
    args = parser.parse_args()

    output_dir = RESULTS_DIR / "calib-data" / _snr_to_dir_name(args.target_snr)
    output_dir.mkdir(parents=True, exist_ok=True)

    restrict_gpus(args.gpus)
    maybe_relaunch_with_torchrun(args.nproc)

    # ── Distributed init (no NCCL — ranks work independently) ──
    rank, world_size, local_rank, device = init_distributed_lite()
    suppress_warnings(rank)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # ── List mode ──
    if args.list:
        if is_main(rank):
            print(f"\n{'ID':>3}  {'Tag':<15}  Description")
            print("-" * 50)
            for sid, tag, desc, _ in CAL_SETUPS:
                result_file = output_dir / f"calibration_{tag}.json"
                exists = "  (done)" if result_file.exists() else ""
                print(f"{sid:3d}  {tag:<15}  {desc}{exists}")
            print(f"\nOutput dir: {output_dir}")
            print()
        cleanup_distributed()
        return

    # ── Select setups ──
    if args.setup is not None:
        # --setup overrides --only
        selected = next((s for s in CAL_SETUPS if s[0] == args.setup), None)
        if selected is None:
            if is_main(rank):
                print(f"Unknown setup ID: {args.setup}. "
                      f"Valid IDs: {[s[0] for s in CAL_SETUPS]}. Use --list.")
            cleanup_distributed()
            return
        run_setups = [selected]
    elif args.only:
        selected_ids = set(args.only)
        run_setups = [s for s in CAL_SETUPS if s[0] in selected_ids]
        if not run_setups:
            if is_main(rank):
                print(f"No matching setup IDs: {args.only}")
                print(f"Valid IDs: {[s[0] for s in CAL_SETUPS]}")
            cleanup_distributed()
            return
    else:
        run_setups = list(CAL_SETUPS)

    # Partition setups across ranks (round-robin)
    my_setups = run_setups[rank::world_size]

    if is_main(rank):
        print(f"World size: {world_size}  |  Device: {device}  |  dtype: {dtype}")
        print(f"Total setups: {len(run_setups)}  |  Setups on this rank: {len(my_setups)}")
        print(f"Target SNR: {args.target_snr}dB  |  Calibration texts: {args.num_texts}")
        print(f"Detail layer: {args.detail_layer}")
        print()

    if not my_setups:
        print(f"[rank {rank}] No setups assigned (more GPUs than setups). Idle.")
        file_barrier(rank, world_size, RESULTS_DIR)
        cleanup_distributed()
        return

    # ── Load model ──
    if is_main(rank):
        print("Loading tokenizer & model ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=dtype
    )
    model.to(device)
    model.eval()

    if is_main(rank):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {MODEL_PATH}  |  Params: {num_params/1e6:.1f}M")

    # ── Load calibration data ──
    ds_name, ds_config, ds_split = CAL_DATASET
    if is_main(rank):
        print(f"Loading dataset: {ds_name}/{ds_config} ({ds_split}) ...")

    ds = load_dataset(ds_name, ds_config, split=ds_split)
    cal_texts = [t for t in ds["text"] if len(t.strip()) > 100][:args.num_texts]

    if is_main(rank):
        print(f"Calibration texts: {len(cal_texts)} paragraphs "
              f"(max_length={args.max_length}, batch_size={args.batch_size})")
        print()

    # ── Run assigned setups ──
    total_start = time.perf_counter()

    setup_iter = list(my_setups)
    setup_iter = tqdm(
        setup_iter,
        desc=f"[rank {rank}] calibration setups",
        disable=not is_main(rank),
    )

    for i, (sid, tag, desc, overrides) in enumerate(setup_iter):
        result_file = output_dir / f"calibration_{tag}.json"
        print(f"[rank {rank}] {'='*55}")
        print(f"[rank {rank}]   [{i+1}/{len(my_setups)}]  Setup #{sid}: {desc}")
        print(f"[rank {rank}]   Tag: {tag}  ->  {result_file.name}")
        print(f"[rank {rank}] {'='*55}")

        # Skip if exists
        if result_file.exists() and not args.force:
            print(f"[rank {rank}]   Already exists. Use --force to re-run.\n")
            continue

        # Reset to baseline, then apply this setup's MXFP overrides
        reset_to_baseline(model.config)
        apply_config(model.config, overrides)
        reconfigure_mlp_layers(model, device)

        # Show active config
        banner = format_config_banner(model.config, setup_id=sid, setup_desc=desc)
        for line in banner.splitlines():
            print(f"[rank {rank}]   {line}")

        # Run calibration (GPU-accelerated, output-chunked)
        t0 = time.perf_counter()
        calibration_data, layer_summaries, channel_details = calibrate_channel_budgets(
            model, tokenizer, cal_texts,
            target_snr_db=args.target_snr,
            max_length=args.max_length,
            batch_size=args.batch_size,
            online_delay=args.online_delay,
            show_progress=is_main(rank),
            progress_prefix=f"[rank {rank}] ",
            detail_layer=args.detail_layer,
        )
        elapsed = time.perf_counter() - t0

        if not calibration_data:
            print(f"[rank {rank}]   WARNING: No MXFP layers found. Skipping.\n")
            continue

        # Compute global summary from per-layer summaries
        all_budgets = []
        for layer_budgets in calibration_data.values():
            all_budgets.extend(layer_budgets)

        all_snr_means = [s["snr_mean"] for s in layer_summaries.values()
                         if s.get("snr_mean") is not None]
        all_snr_mins = [s["snr_min"] for s in layer_summaries.values()
                        if s.get("snr_min") is not None]
        all_e_combined = [s["e_combined_mean"] for s in layer_summaries.values()]
        all_eff_prec = [s["eff_precision_mean"] for s in layer_summaries.values()]
        all_sig_power = [s["signal_power_db_mean"] for s in layer_summaries.values()
                         if s.get("signal_power_db_mean") is not None]

        result = {
            "format": tag,
            "description": desc,
            "config_overrides": overrides,
            "calibration_params": {
                "target_snr_db": args.target_snr,
                "num_texts": len(cal_texts),
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "online_delay": args.online_delay,
                "detail_layer": args.detail_layer,
            },
            "global_summary": {
                "num_layers": len(calibration_data),
                "total_channels": len(all_budgets),
                "budget_min": min(all_budgets),
                "budget_max": max(all_budgets),
                "budget_mean": round(sum(all_budgets) / len(all_budgets), 2),
                "mean_snr": round(sum(all_snr_means) / len(all_snr_means), 2) if all_snr_means else None,
                "min_snr": round(min(all_snr_mins), 2) if all_snr_mins else None,
                "e_combined_mean": round(sum(all_e_combined) / len(all_e_combined), 2) if all_e_combined else None,
                "eff_precision_mean": round(sum(all_eff_prec) / len(all_eff_prec), 2) if all_eff_prec else None,
                "signal_power_db_mean": round(sum(all_sig_power) / len(all_sig_power), 2) if all_sig_power else None,
                "wall_time_sec": round(elapsed, 2),
            },
            "layer_stats": layer_summaries,
            "channel_detail": {
                "detail_layer": args.detail_layer,
                **channel_details,
            },
            "msd_calibration_data": calibration_data,
        }

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[rank {rank}]   Layers: {len(calibration_data)}  |  "
              f"Budget: [{min(all_budgets):.0f}, {max(all_budgets):.0f}]  |  "
              f"Mean: {sum(all_budgets)/len(all_budgets):.1f}")
        print(f"[rank {rank}]   {elapsed:.1f}s  |  saved -> {result_file.name}\n")

    total_elapsed = time.perf_counter() - total_start

    # ── Wait for all ranks ──
    file_barrier(rank, world_size, RESULTS_DIR)

    # ── Summary (rank 0) ──
    if is_main(rank):
        print()
        print(f"{'='*60}")
        print(f"  CALIBRATION COMPLETE  ({total_elapsed/60:.1f} min, {world_size} GPUs)")
        print(f"{'='*60}")
        print(f"{'ID':>3}  {'Tag':<15}  {'Layers':>7}  {'Budget Range':>14}  {'Mean':>6}  {'Time':>6}")
        print("-" * 60)

        for sid, tag, desc, _ in run_setups:
            result_file = output_dir / f"calibration_{tag}.json"
            if result_file.exists():
                with open(result_file) as f:
                    res = json.load(f)
                s = res.get("global_summary", res.get("summary", {}))
                bmin = s.get("budget_min", "?")
                bmax = s.get("budget_max", "?")
                bmean = s.get("budget_mean", "?")
                nlayers = s.get("num_layers", "?")
                wall = s.get("wall_time_sec", "?")
                bmin_s = f"{bmin:.0f}" if isinstance(bmin, (int, float)) else str(bmin)
                bmax_s = f"{bmax:.0f}" if isinstance(bmax, (int, float)) else str(bmax)
                bmean_s = f"{bmean:.1f}" if isinstance(bmean, (int, float)) else str(bmean)
                wall_s = f"{wall:.0f}s" if isinstance(wall, (int, float)) else str(wall)
                print(f"{sid:3d}  {tag:<15}  {nlayers:>7}  [{bmin_s:>5}, {bmax_s:<5}]  {bmean_s:>6}  {wall_s:>6}")
            else:
                print(f"{sid:3d}  {tag:<15}  {'MISSING':>7}")

        print("-" * 60)
        print(f"\nCalibration files saved in: {output_dir}")
        print(f"To use with PPL tests, copy msd_calibration_data from")
        print(f"calibration_<tag>.json into Qwen3-0.6B/config.json, or")
        print(f"load programmatically with apply_calibration_to_config().")

    cleanup_distributed()


if __name__ == "__main__":
    main()