import argparse
import json
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ONLINEARITH_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = ONLINEARITH_ROOT.parent

if str(ONLINEARITH_ROOT) not in sys.path:
    sys.path.insert(0, str(ONLINEARITH_ROOT))

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
    maybe_relaunch_with_torchrun,
)
from experiment_config import apply_config, reconfigure_mlp_layers, reset_to_baseline
from runtime_paths import describe_missing_model_path, normalize_output_dir

# ── Configuration ──
# (Format 1-4 cover the MXFP evaluations)
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

MODEL_PATH = str((WORKSPACE_ROOT / "Qwen3-0.6B").resolve())
CAL_DATASET = ("wikitext", "wikitext-2-raw-v1", "validation")


def _normalize_output_hook(raw_hook: str) -> str:
    hook = raw_hook.strip()
    if not hook:
        return ""
    hook = re.sub(r"[^A-Za-z0-9._-]+", "_", hook)
    return hook.strip("_")


def _calibration_filename(tag: str, output_hook: str) -> str:
    suffix = f"_{output_hook}" if output_hook else ""
    return f"calibration_base_{tag}{suffix}.pt"


def calibrate_baseline_sparsify(model, tokenizer, calibration_texts, n=2, m=4, max_length=512, batch_size=4):
    device = next(model.parameters()).device
    
    # 1. Register forward pre-hooks to accurately accumulate X norms
    x_norms = {}
    hooks = []
    
    from transformers.models.qwen3.modeling_qwen3 import _MXFPLinearBase
    
    for name, module in model.named_modules():
        if isinstance(module, _MXFPLinearBase):
            x_norms[name] = torch.zeros(module.in_features, dtype=torch.float32, device=device)
            
            def make_hook(layer_name):
                def pre_hook(mod, inputs):
                    x = inputs[0].float() # (..., in_features)
                    # We want sum of squares across N*L elements
                    x_sq = (x ** 2).view(-1, x.shape[-1]).sum(dim=0)
                    x_norms[layer_name] += x_sq
                return pre_hook
                
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
            
    # 2. Run calibration dataset forward passes
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(calibration_texts), batch_size), desc=f"Calibrating X-norms (batch {batch_size})"):
            batch_texts = calibration_texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to(device)
            model(**inputs)
            
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # 3. Compute baseline masks locally
    masks = {}
    for name, module in tqdm(model.named_modules(), desc="Computing baseline metric"):
        if isinstance(module, _MXFPLinearBase):
            # Compute ||X_j||_2
            x_norm_sqrt = torch.sqrt(x_norms[name]) # (Cin)
            
            # Extract weights and quantize them temporarily to the EXACT same MXFP format
            w_2d = module.weight.data.float() # (Cout, Cin)
            w_q, w_scales, pad_len = module._prepare_blocks(w_2d, module.out_features)
            w_quantized = (w_q * w_scales.unsqueeze(-1)).view(module.out_features, -1)
            
            if pad_len > 0:
                w_quantized = w_quantized[:, :-pad_len]
                
            # S_ij = |W_ij| * ||X_j||_2  which is shape (Cout, Cin).
            S = torch.abs(w_quantized) * x_norm_sqrt.unsqueeze(0)
            
            # 4. Perform n:m sparsity over the row (Cout output neurons)
            # m divides Cin
            # We skip channels if they don't match, or pad them temporarily (usually they perfectly match for Qwen models).
            if S.shape[1] % m != 0:
                pad_size = m - (S.shape[1] % m)
                S = torch.nn.functional.pad(S, (0, pad_size), value=float('inf')) # so padded won't be pruned
                
            S_reshaped = S.view(S.shape[0], -1, m)
            
            # Identify the n elements with the SMALLEST scores in each group, and prune them
            _, indices = torch.sort(S_reshaped, dim=-1)
            prune_indices = indices[:, :, :n]
            
            # create mask
            mask_reshaped = torch.ones_like(S_reshaped, dtype=torch.bool)
            # scatter zeros
            mask_reshaped.scatter_(-1, prune_indices, False)
            
            mask = mask_reshaped.view(S.shape[0], -1)
            # Unpad if necessary
            if S.shape[1] != x_norm_sqrt.shape[0]:
                mask = mask[:, :x_norm_sqrt.shape[0]]
                
            # save to mask
            masks[name] = mask.cpu()
            
    return masks


def main():
    parser = argparse.ArgumentParser(description="Baseline sparsity calibration for MXFP formats")
    parser.add_argument("--list", action="store_true", help="List all calibration setups and exit")
    parser.add_argument("--setup", type=int, default=None, metavar="ID", help="Run a single setup by ID (1-4)")
    parser.add_argument("--only", nargs="+", type=int, metavar="ID", help="Run only specific setup IDs (space separated)")
    parser.add_argument("--num-texts", type=int, default=2048, help="Number of paragraphs for calibration")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for forward hook accumulation")
    parser.add_argument("-n", type=int, default=2, help="Sparsify n elements in each group")
    parser.add_argument("-m", type=int, default=4, help="Group size for structured sparsification")
    parser.add_argument("--force", action="store_true", help="Overwrite existing calibration files")
    parser.add_argument("--nproc", type=int, default=1, help="Auto-launch torchrun with N processes")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated list of GPUs to use")
    parser.add_argument(
        "--output-hook",
        type=str,
        default="",
        help="Optional suffix appended to calibration filenames (shared with ppl_batch_base).",
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, metavar="DIR",
                        help=f"Local model directory (default: {MODEL_PATH})")
    parser.add_argument("--results-root", type=str, default=None, metavar="DIR",
                        help=f"Root directory for Wanda baseline outputs (default: {WORKSPACE_ROOT / 'data' / 'wanda_base'})")
    
    args, unparsed = parser.parse_known_args()
    model_path = args.model_path
    results_root = normalize_output_dir(args.results_root, (WORKSPACE_ROOT / "data" / "wanda_base").resolve())
    output_hook = _normalize_output_hook(args.output_hook)
    if args.output_hook.strip() and not output_hook:
        raise SystemExit("Invalid --output-hook: must contain at least one alphanumeric, '.', '_' or '-'.")

    # ── torchrun auto-launcher ──
    restrict_gpus(args.gpus)
    maybe_relaunch_with_torchrun(args.nproc)

    import os
    rank, world_size, local_rank, device = init_distributed_lite()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # ── List mode ──
    if args.list:
        if is_main(rank):
            print(f"\n{'ID':>3}  {'Tag':<15}  Description")
            print("-" * 50)
            for sid, tag, desc, _ in CAL_SETUPS:
                print(f"{sid:3d}  {tag:<15}  {desc}")
            print()
        cleanup_distributed()
        return

    # ── Select setups ──
    if args.setup is not None:
        selected = next((s for s in CAL_SETUPS if s[0] == args.setup), None)
        if selected is None:
            if is_main(rank):
                print(f"Unknown setup ID: {args.setup}")
            cleanup_distributed()
            return
        run_setups = [selected]
    elif args.only:
        selected_ids = set(args.only)
        run_setups = [s for s in CAL_SETUPS if s[0] in selected_ids]
    else:
        run_setups = list(CAL_SETUPS)

    my_setups = run_setups[rank::world_size]

    RESULTS_DIR = results_root / f"{args.n}-{args.m}"
    if is_main(rank):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"World size: {world_size}  |  Device: {device}  |  dtype: {dtype}")
        print(f"Total setups: {len(run_setups)}  |  Setups on this rank: {len(my_setups)}")
        print(f"Struct sparsification: {args.n}:{args.m}")
        if output_hook:
            print(f"Output hook: {output_hook}")
        print(f"Output directory: {RESULTS_DIR}")
        print()

    if not my_setups:
        file_barrier(rank, world_size, RESULTS_DIR)
        cleanup_distributed()
        return

    if is_main(rank):
        print("Loading tokenizer & model ...")

    if not Path(model_path).exists():
        if is_main(rank):
            print(f"Error: {describe_missing_model_path(model_path)}")
        cleanup_distributed()
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=dtype
    )
    model.to(device)

    # ── Load calibration data ──
    ds_name, ds_config, ds_split = CAL_DATASET
    ds = load_dataset(ds_name, ds_config, split=ds_split, trust_remote_code=True)
    cal_texts = [t for t in ds["text"] if len(t.strip()) > 100][:args.num_texts]

    # ── Run assigned setups ──
    for sid, tag, desc, raw_config in my_setups:
        result_file = RESULTS_DIR / _calibration_filename(tag, output_hook)
        if result_file.exists() and not args.force:
            print(f"[rank {rank}] Skipping setup {sid} ({tag}), already calibrated.")
            continue

        print(f"\n[rank {rank}] === Setup {sid}: {tag} ===")
        # Reconfigure to the specific MXFP format
        reset_to_baseline(model.config)
        cfg_diff = apply_config(model.config, raw_config)
        reconfigure_mlp_layers(model,device)

        t0 = time.perf_counter()
        masks = calibrate_baseline_sparsify(
            model, tokenizer, cal_texts, 
            n=args.n, m=args.m,
            max_length=args.max_length, 
            batch_size=args.batch_size
        )
        t_cal = time.perf_counter() - t0
        
        print(f"[rank {rank}] Saving masks to {result_file} (took {t_cal:.1f}s)")
        torch.save(masks, result_file)

    file_barrier(rank, world_size, RESULTS_DIR)
    if is_main(rank):
        print("\nAll baseline calibrations complete.")
    cleanup_distributed()

if __name__ == "__main__":
    main()
