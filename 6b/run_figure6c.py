#!/usr/bin/env python3
"""
Figure 6(c) fixed-horizon ablation runner.

Methods:
  - uniform
  - weight-only
  - combined (reference SNR-min at snr_star)
  - fixed-sum (reference redistribution)

Design goals:
  - Reuse existing 15 dB combined/fixed_sum calibration artifacts.
  - Freeze per-layer total horizon from combined reference.
  - Generate only missing uniform and weight-only calibrations.
  - Evaluate all methods on one held-out dev manifest.
  - Keep runtime dynamic budget adjustment off (setup 6) and verify in outputs.
  - Avoid activation-cache collection for new methods to reduce memory pressure.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import _MXFPLinearBase

SCRIPT_DIR = Path(__file__).resolve().parent
ONLINEARITH_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
if str(ONLINEARITH_ROOT) not in sys.path:
    sys.path.insert(0, str(ONLINEARITH_ROOT))

from experiment_config import apply_config, reconfigure_mlp_layers, reset_to_baseline
from fig6c_utils import (
    allocate_integer_budget_from_scores,
    make_uniform_budgets,
    make_weight_only_scores_from_quantized_weights,
)

MODEL_PATH_DEFAULT = WORKSPACE_ROOT / "Qwen3-0.6B"
OUTPUT_ROOT_DEFAULT = WORKSPACE_ROOT / "6c"
COMBINED_CAL_DEFAULT = WORKSPACE_ROOT / "data" / "calib-data" / "15db" / "calibration_MXFP8.json"
FIXED_SUM_CAL_DEFAULT = WORKSPACE_ROOT / "data" / "calib-data" / "15db" / "calibration_MXFP8_fixed_sum.json"
EVAL_MANIFEST_DEFAULT = WORKSPACE_ROOT / "6b" / "manifests" / "eval_B.json"

DENSE_EVAL_CANDIDATE_DEFAULTS = [
    WORKSPACE_ROOT / "6b" / "evals" / "ppl_dense_B.json",
]
COMBINED_EVAL_CANDIDATE_DEFAULTS = [
    WORKSPACE_ROOT / "data" / "calib-data" / "15db" / "ppl_results_MXFP8_calibration_snr.json",
]
FIXED_SUM_EVAL_CANDIDATE_DEFAULTS = [
    WORKSPACE_ROOT / "data" / "calib-data" / "15db" / "ppl_results_MXFP8_calibration_fix.json",
]

METHOD_ORDER = [
    ("uniform", "uniform"),
    ("weight_only", "weight-only"),
    ("combined", "combined"),
    ("fixed_sum", "fixed-sum"),
]


@dataclass
class MethodSpec:
    key: str
    label: str
    calibration_file: Path
    eval_output_file: Path
    candidate_eval_files: list[Path]


@dataclass
class LayerTarget:
    n_channels: int
    target_sum: int


@dataclass
class EvalJob:
    run_id: str
    calibration_file: Path
    output_file: Path


def _parse_gpu_list(gpu_csv: str | None) -> list[str]:
    if not gpu_csv:
        return []
    ids = [tok.strip() for tok in gpu_csv.split(",") if tok.strip()]
    for tok in ids:
        if not tok.isdigit():
            raise ValueError(f"--gpus expects comma-separated integers, got: {gpu_csv}")
    return ids


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(raw: str | Path, base_dir: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = base_dir / p
    return p.expanduser().resolve()


def _same_path(lhs: str | Path | None, rhs: str | Path, base_dir: Path) -> bool:
    if lhs is None:
        return False
    try:
        lp = _resolve_path(lhs, base_dir)
        rp = _resolve_path(rhs, base_dir)
        return lp == rp
    except Exception:
        return False


def _extract_int_budget_map(calibration_json: dict[str, Any]) -> dict[str, np.ndarray]:
    data = calibration_json.get("msd_calibration_data")
    if not isinstance(data, dict) or not data:
        raise ValueError("calibration JSON is missing non-empty 'msd_calibration_data'")

    out: dict[str, np.ndarray] = {}
    for layer_name, values in data.items():
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"empty budget vector for layer {layer_name}")
        arr_int = np.rint(arr).astype(np.int32)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"non-finite budgets for layer {layer_name}")
        out[layer_name] = arr_int
    return out


def _extract_layer_targets(reference_budget_map: dict[str, np.ndarray]) -> dict[str, LayerTarget]:
    targets: dict[str, LayerTarget] = {}
    for layer_name, budgets in reference_budget_map.items():
        targets[layer_name] = LayerTarget(
            n_channels=int(budgets.size),
            target_sum=int(budgets.sum()),
        )
    return targets


def _build_uniform_budget_map(
    targets: dict[str, LayerTarget],
    budget_range: tuple[int, int],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for layer_name, t in targets.items():
        out[layer_name] = make_uniform_budgets(
            target_sum=t.target_sum,
            n_channels=t.n_channels,
            budget_range=budget_range,
        )
    return out


def _build_weight_only_budget_map(
    targets: dict[str, LayerTarget],
    model_path: Path,
    budget_range: tuple[int, int],
) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    reset_to_baseline(model.config)
    apply_config(model.config, {"use_mxfp8": True})
    reconfigure_mlp_layers(model, device)
    model.eval()

    layer_modules: dict[str, _MXFPLinearBase] = {}
    for name, module in model.named_modules():
        if isinstance(module, _MXFPLinearBase):
            layer_modules[name] = module

    out: dict[str, np.ndarray] = {}
    missing_layers: list[str] = []

    with torch.no_grad():
        for layer_name, target in targets.items():
            local_name = layer_name[len("model."):] if layer_name.startswith("model.") else layer_name
            module = layer_modules.get(local_name) or layer_modules.get(layer_name)
            if module is None:
                missing_layers.append(layer_name)
                continue

            w_2d = module.weight.float()
            w_q, w_scales, _ = module._prepare_blocks(w_2d, module.out_features)
            scores = make_weight_only_scores_from_quantized_weights(w_q, w_scales)

            out[layer_name] = allocate_integer_budget_from_scores(
                scores=scores,
                target_sum=target.target_sum,
                budget_range=budget_range,
            )

            del w_2d, w_q, w_scales

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if missing_layers:
        raise RuntimeError(
            "Missing MXFP8 layer modules for calibration keys: "
            + ", ".join(missing_layers[:8])
            + (" ..." if len(missing_layers) > 8 else "")
        )

    return out


def _validate_budget_map(
    reference_map: dict[str, np.ndarray],
    candidate_map: dict[str, np.ndarray],
    budget_range: tuple[int, int],
) -> None:
    bmin, bmax = budget_range

    ref_keys = set(reference_map.keys())
    cand_keys = set(candidate_map.keys())
    if ref_keys != cand_keys:
        missing = sorted(ref_keys - cand_keys)
        extra = sorted(cand_keys - ref_keys)
        raise ValueError(
            f"candidate budget map keys mismatch; missing={missing[:6]}, extra={extra[:6]}"
        )

    for layer_name, ref in reference_map.items():
        cand = candidate_map[layer_name]
        if cand.shape != ref.shape:
            raise ValueError(
                f"shape mismatch for {layer_name}: ref={ref.shape}, cand={cand.shape}"
            )
        if int(cand.sum()) != int(ref.sum()):
            raise ValueError(
                f"sum mismatch for {layer_name}: ref={int(ref.sum())}, cand={int(cand.sum())}"
            )
        if int(cand.min()) < bmin or int(cand.max()) > bmax:
            raise ValueError(
                f"budget range violation for {layer_name}: "
                f"[{int(cand.min())}, {int(cand.max())}] not in [{bmin}, {bmax}]"
            )


def _build_layer_budget_stats(budget_map: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for layer_name, budgets in budget_map.items():
        vals = budgets.astype(np.float64)
        stats[layer_name] = {
            "budget_mean": round(float(vals.mean()), 4),
            "budget_min": float(vals.min()),
            "budget_max": float(vals.max()),
            "budget_std": round(float(vals.std()), 4),
        }
    return stats


def _build_global_budget_summary(budget_map: dict[str, np.ndarray], wall_time_sec: float) -> dict[str, Any]:
    flat = np.concatenate([v.astype(np.float64) for v in budget_map.values()], axis=0)
    return {
        "num_layers": int(len(budget_map)),
        "total_channels": int(flat.size),
        "budget_min": float(flat.min()),
        "budget_max": float(flat.max()),
        "budget_mean": round(float(flat.mean()), 2),
        "mean_snr": None,
        "min_snr": None,
        "e_combined_mean": None,
        "eff_precision_mean": None,
        "signal_power_db_mean": None,
        "wall_time_sec": round(float(wall_time_sec), 2),
    }


def _to_calibration_payload_lists(budget_map: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        layer_name: [float(v) for v in budgets.tolist()]
        for layer_name, budgets in budget_map.items()
    }


def _build_generated_calibration_json(
    method_key: str,
    method_label: str,
    budget_map: dict[str, np.ndarray],
    reference_json: dict[str, Any],
    source_reference_path: Path,
    snr_star: float,
    freeze_granularity: str,
    generation_time_sec: float,
) -> dict[str, Any]:
    cal_params = dict(reference_json.get("calibration_params", {}))
    cal_params.update(
        {
            "target_snr_db": snr_star,
            "figure": "6c",
            "method": method_key,
            "freeze_granularity": freeze_granularity,
            "reference_calibration": str(source_reference_path),
        }
    )

    result = {
        "format": reference_json.get("format", "MXFP8"),
        "description": reference_json.get("description", "MXFP8 (E4M3FN)"),
        "optimizer": f"fig6c_{method_key}",
        "config_overrides": reference_json.get("config_overrides", {"use_mxfp8": True}),
        "calibration_params": cal_params,
        "global_summary": _build_global_budget_summary(budget_map, wall_time_sec=generation_time_sec),
        "layer_stats": _build_layer_budget_stats(budget_map),
        "msd_calibration_data": _to_calibration_payload_lists(budget_map),
        "figure6c_metadata": {
            "method_label": method_label,
            "freeze_granularity": freeze_granularity,
            "source_reference_calibration": str(source_reference_path),
        },
    }
    return result


def _flatten_budgets(cal_data: dict[str, list[float]]) -> list[float]:
    out: list[float] = []
    for vals in cal_data.values():
        out.extend(vals)
    return out


def _build_dense_cmd(
    python_bin: str,
    eval_manifest: Path,
    output_file: Path,
    gpu: str | None,
    use_lite: bool,
) -> list[str]:
    cmd = [
        python_bin,
        "ppltest.py",
        "--nproc", "1",
        "--setup", "2",
        "--text-manifest", str(eval_manifest),
        "--output", str(output_file),
    ]
    if use_lite:
        cmd.append("--lite")
    if gpu is not None:
        cmd.extend(["--gpus", gpu])
    return cmd


def _build_quant_cmd(
    python_bin: str,
    calibration_file: Path,
    eval_manifest: Path,
    output_file: Path,
    gpu: str | None,
    use_lite: bool,
) -> list[str]:
    cmd = [
        python_bin,
        "ppltest.py",
        "--nproc", "1",
        "--setup", "6",
        "--calibration", str(calibration_file),
        "--text-manifest", str(eval_manifest),
        "--output", str(output_file),
    ]
    if use_lite:
        cmd.append("--lite")
    if gpu is not None:
        cmd.extend(["--gpus", gpu])
    return cmd


def _run_command(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.time() - start

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        if proc.stdout:
            f.write(proc.stdout)
        f.write(f"\nExit code: {proc.returncode}\n")
        f.write(f"Elapsed sec: {elapsed:.2f}\n")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}. See log: {log_path}"
        )


def _is_eval_output_compatible(
    eval_file: Path,
    eval_manifest: Path,
    expected_calibration: Path | None,
    require_dynamic_zero: bool,
) -> tuple[bool, str]:
    if not eval_file.exists():
        return False, "missing output file"

    try:
        payload = _load_json(eval_file)
    except Exception as exc:
        return False, f"failed to parse JSON: {exc}"

    conf = payload.get("config", {})
    conf_manifest = conf.get("text_manifest")
    dataset_name = payload.get("dataset")

    manifest_ok = False
    if isinstance(conf_manifest, str):
        manifest_ok = _same_path(conf_manifest, eval_manifest, base_dir=ONLINEARITH_ROOT)
    if not manifest_ok and isinstance(dataset_name, str) and dataset_name.startswith("manifest:"):
        manifest_ok = _same_path(dataset_name[len("manifest:"):], eval_manifest, base_dir=ONLINEARITH_ROOT)

    if not manifest_ok:
        return False, "manifest mismatch"

    if expected_calibration is not None:
        conf_cal = conf.get("calibration_file")
        if not isinstance(conf_cal, str):
            return False, "missing config.calibration_file"
        if not _same_path(conf_cal, expected_calibration, base_dir=ONLINEARITH_ROOT):
            return False, "calibration mismatch"

    if require_dynamic_zero:
        alpha = payload.get("config_snapshot", {}).get("msd_budget_dynamic_scale")
        try:
            alpha_val = float(alpha)
        except Exception:
            return False, "missing/invalid msd_budget_dynamic_scale"
        if abs(alpha_val) > 1e-12:
            return False, f"msd_budget_dynamic_scale={alpha_val}"

    ppl = payload.get("metrics", {}).get("token_perplexity")
    if not isinstance(ppl, (int, float)):
        return False, "missing token_perplexity"

    return True, "ok"


def _materialize_reused_eval(
    expected_output: Path,
    eval_manifest: Path,
    expected_calibration: Path | None,
    require_dynamic_zero: bool,
    candidate_files: list[Path],
) -> tuple[bool, str]:
    ok, reason = _is_eval_output_compatible(
        expected_output,
        eval_manifest=eval_manifest,
        expected_calibration=expected_calibration,
        require_dynamic_zero=require_dynamic_zero,
    )
    if ok:
        return True, f"reuse existing ({expected_output})"

    for cand in candidate_files:
        ok_c, reason_c = _is_eval_output_compatible(
            cand,
            eval_manifest=eval_manifest,
            expected_calibration=expected_calibration,
            require_dynamic_zero=require_dynamic_zero,
        )
        if not ok_c:
            continue
        expected_output.parent.mkdir(parents=True, exist_ok=True)
        if cand.resolve() != expected_output.resolve():
            shutil.copy2(cand, expected_output)
        return True, f"reused candidate ({cand})"

    return False, reason


def _summarize(
    output_root: Path,
    dense_eval_file: Path,
    methods: list[MethodSpec],
    snr_star: float,
    eval_manifest: Path,
) -> None:
    summary_json = output_root / "figure6c_results.json"
    summary_csv = output_root / "figure6c_results.csv"
    metric_mapping = output_root / "metric_mapping.md"

    dense_payload = _load_json(dense_eval_file)
    ppl_dense = dense_payload.get("metrics", {}).get("token_perplexity")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for method in methods:
        if not method.eval_output_file.exists():
            missing.append(method.key)
            continue

        eval_payload = _load_json(method.eval_output_file)
        cal_payload = _load_json(method.calibration_file)

        ppl_quant = eval_payload.get("metrics", {}).get("token_perplexity")
        ppl_ratio = None
        delta_ppl = None
        if isinstance(ppl_quant, (int, float)) and isinstance(ppl_dense, (int, float)):
            if float(ppl_dense) > 0:
                ppl_ratio = float(ppl_quant) / float(ppl_dense)
            delta_ppl = float(ppl_quant) - float(ppl_dense)

        g = eval_payload.get("msd_perf_stats", {}).get("global", {})
        cal_data = cal_payload.get("msd_calibration_data", {})
        budgets = _flatten_budgets(cal_data) if isinstance(cal_data, dict) else []

        rows.append(
            {
                "method": method.label,
                "method_key": method.key,
                "target_snr": snr_star,
                "ppl_quant": ppl_quant,
                "ppl_dense": ppl_dense,
                "ppl_ratio": ppl_ratio,
                "delta_ppl": delta_ppl,
                "norm_digit_reads": g.get("global_utilization"),
                "block_skip_rate": g.get("zero_block_ratio"),
                "element_skip_rate": g.get("mac_sparsity", g.get("zero_element_percentage")),
                "partial_window_rate": g.get("partial_block_ratio"),
                "mean_H": (sum(budgets) / len(budgets)) if budgets else None,
                "sum_H": sum(budgets) if budgets else None,
                "calibration_file": str(method.calibration_file),
                "eval_file": str(method.eval_output_file),
            }
        )

    summary_obj = {
        "target_snr": snr_star,
        "eval_split_manifest": str(eval_manifest),
        "dense_eval_file": str(dense_eval_file),
        "num_rows": len(rows),
        "missing_methods": missing,
        "rows": rows,
    }
    _write_json(summary_json, summary_obj)

    fieldnames = [
        "method",
        "method_key",
        "target_snr",
        "ppl_quant",
        "ppl_dense",
        "ppl_ratio",
        "delta_ppl",
        "norm_digit_reads",
        "block_skip_rate",
        "element_skip_rate",
        "partial_window_rate",
        "mean_H",
        "sum_H",
        "calibration_file",
        "eval_file",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    metric_mapping.write_text(
        "\n".join(
            [
                "Figure 6(c) metric mapping",
                "",
                "- norm_digit_reads: msd_perf_stats.global.global_utilization",
                "- block_skip_rate: msd_perf_stats.global.zero_block_ratio",
                "- element_skip_rate: msd_perf_stats.global.mac_sparsity (fallback: zero_element_percentage)",
                "- partial_window_rate: msd_perf_stats.global.partial_block_ratio",
                "- mean_H, sum_H: aggregate of calibration JSON msd_calibration_data budgets",
                "- ppl_ratio: ppl_quant / ppl_dense",
                "- delta_ppl: ppl_quant - ppl_dense",
                "",
                "Note: in --lite mode, partial_window_rate may be null if partial_block_ratio",
                "is not emitted by the runtime stats collector.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Figure 6(c) fixed-horizon runner")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT_DEFAULT),
                        help="Artifact root for Figure 6(c) outputs.")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH_DEFAULT),
                        help="Local model path for weight-only budget generation.")
    parser.add_argument("--combined-calibration", type=str, default=str(COMBINED_CAL_DEFAULT),
                        help="Reference combined calibration JSON (SNR-min at snr_star).")
    parser.add_argument("--fixed-sum-calibration", type=str, default=str(FIXED_SUM_CAL_DEFAULT),
                        help="Reference fixed-sum calibration JSON.")
    parser.add_argument("--eval-manifest", type=str, default=str(EVAL_MANIFEST_DEFAULT),
                        help="Held-out dev manifest for Figure 6(c) evaluation.")
    parser.add_argument("--target-snr", type=float, default=15.0,
                        help="Fixed operating point (snr_star) for metadata.")
    parser.add_argument("--budget-min", type=int, default=4,
                        help="Lower bound for channel budgets.")
    parser.add_argument("--budget-max", type=int, default=48,
                        help="Upper bound for channel budgets.")
    parser.add_argument("--freeze-granularity", type=str, default="per_layer", choices=["per_layer"],
                        help="Horizon freeze granularity. Figure 6(c) uses per_layer.")

    parser.add_argument("--python-bin", type=str, default=sys.executable,
                        help="Python executable used for ppltest subprocesses.")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs for eval jobs; each job uses nproc=1.")
    parser.add_argument("--execute", action="store_true",
                        help="Run ppltest evaluations. Without this, only preparation runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print run decisions and command plan without launching eval subprocesses.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate calibrations and rerun evals even if outputs exist.")
    parser.add_argument("--no-lite", action="store_true",
                        help="Disable --lite in ppltest commands.")
    parser.add_argument("--no-reuse-existing-evals", action="store_true",
                        help="Disable compatibility-based reuse of existing eval outputs.")

    parser.add_argument("--dense-eval-candidates", nargs="*", default=[str(p) for p in DENSE_EVAL_CANDIDATE_DEFAULTS],
                        help="Optional dense eval JSON candidates for reuse.")
    parser.add_argument("--combined-eval-candidates", nargs="*", default=[str(p) for p in COMBINED_EVAL_CANDIDATE_DEFAULTS],
                        help="Optional combined-method eval JSON candidates for reuse.")
    parser.add_argument("--fixed-sum-eval-candidates", nargs="*", default=[str(p) for p in FIXED_SUM_EVAL_CANDIDATE_DEFAULTS],
                        help="Optional fixed-sum-method eval JSON candidates for reuse.")

    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    combined_cal_path = Path(args.combined_calibration).expanduser().resolve()
    fixed_sum_cal_path = Path(args.fixed_sum_calibration).expanduser().resolve()
    eval_manifest_path = Path(args.eval_manifest).expanduser().resolve()
    budget_range = (int(args.budget_min), int(args.budget_max))

    reuse_existing_evals = (not args.no_reuse_existing_evals) and (not args.force)
    use_lite = not args.no_lite

    for p in (combined_cal_path, fixed_sum_cal_path, eval_manifest_path, model_path):
        if not p.exists():
            raise FileNotFoundError(f"required path not found: {p}")

    cal_dir = output_root / "calibrations"
    eval_dir = output_root / "evals"
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Load reference calibrations and freeze targets ──────────────────────
    ref_combined_json = _load_json(combined_cal_path)
    ref_combined_map = _extract_int_budget_map(ref_combined_json)
    layer_targets = _extract_layer_targets(ref_combined_map)

    ref_fixed_json = _load_json(fixed_sum_cal_path)
    ref_fixed_map = _extract_int_budget_map(ref_fixed_json)
    _validate_budget_map(ref_combined_map, ref_fixed_map, budget_range)

    freeze_metadata = {
        "freeze_granularity": args.freeze_granularity,
        "num_layers": len(layer_targets),
        "layers": {
            name: {
                "n_channels": t.n_channels,
                "target_sum": t.target_sum,
            }
            for name, t in layer_targets.items()
        },
    }
    _write_json(output_root / "freeze_targets.json", freeze_metadata)

    # ── Generate/reuse uniform and weight-only calibrations ────────────────
    uniform_cal_file = cal_dir / "calibration_MXFP8_fig6c_uniform.json"
    weight_only_cal_file = cal_dir / "calibration_MXFP8_fig6c_weight_only.json"

    if uniform_cal_file.exists() and not args.force:
        uniform_map = _extract_int_budget_map(_load_json(uniform_cal_file))
        _validate_budget_map(ref_combined_map, uniform_map, budget_range)
    else:
        t0 = time.perf_counter()
        uniform_map = _build_uniform_budget_map(layer_targets, budget_range=budget_range)
        _validate_budget_map(ref_combined_map, uniform_map, budget_range)
        uniform_payload = _build_generated_calibration_json(
            method_key="uniform",
            method_label="uniform",
            budget_map=uniform_map,
            reference_json=ref_combined_json,
            source_reference_path=combined_cal_path,
            snr_star=float(args.target_snr),
            freeze_granularity=args.freeze_granularity,
            generation_time_sec=time.perf_counter() - t0,
        )
        _write_json(uniform_cal_file, uniform_payload)

    if weight_only_cal_file.exists() and not args.force:
        weight_map = _extract_int_budget_map(_load_json(weight_only_cal_file))
        _validate_budget_map(ref_combined_map, weight_map, budget_range)
    else:
        t0 = time.perf_counter()
        weight_map = _build_weight_only_budget_map(
            layer_targets,
            model_path=model_path,
            budget_range=budget_range,
        )
        _validate_budget_map(ref_combined_map, weight_map, budget_range)
        weight_payload = _build_generated_calibration_json(
            method_key="weight_only",
            method_label="weight-only",
            budget_map=weight_map,
            reference_json=ref_combined_json,
            source_reference_path=combined_cal_path,
            snr_star=float(args.target_snr),
            freeze_granularity=args.freeze_granularity,
            generation_time_sec=time.perf_counter() - t0,
        )
        _write_json(weight_only_cal_file, weight_payload)

    method_specs: list[MethodSpec] = [
        MethodSpec(
            key="uniform",
            label="uniform",
            calibration_file=uniform_cal_file,
            eval_output_file=eval_dir / "ppl_fig6c_uniform.json",
            candidate_eval_files=[],
        ),
        MethodSpec(
            key="weight_only",
            label="weight-only",
            calibration_file=weight_only_cal_file,
            eval_output_file=eval_dir / "ppl_fig6c_weight_only.json",
            candidate_eval_files=[],
        ),
        MethodSpec(
            key="combined",
            label="combined",
            calibration_file=combined_cal_path,
            eval_output_file=eval_dir / "ppl_fig6c_combined.json",
            candidate_eval_files=[Path(p).expanduser().resolve() for p in args.combined_eval_candidates],
        ),
        MethodSpec(
            key="fixed_sum",
            label="fixed-sum",
            calibration_file=fixed_sum_cal_path,
            eval_output_file=eval_dir / "ppl_fig6c_fixed_sum.json",
            candidate_eval_files=[Path(p).expanduser().resolve() for p in args.fixed_sum_eval_candidates],
        ),
    ]

    dense_eval_output = eval_dir / "ppl_fig6c_dense_evalB.json"
    dense_candidates = [Path(p).expanduser().resolve() for p in args.dense_eval_candidates]

    run_context = {
        "target_snr": float(args.target_snr),
        "eval_manifest": str(eval_manifest_path),
        "freeze_granularity": args.freeze_granularity,
        "budget_range": list(budget_range),
        "reuse_existing_evals": reuse_existing_evals,
        "force": bool(args.force),
        "methods": [
            {
                "key": m.key,
                "label": m.label,
                "calibration_file": str(m.calibration_file),
                "eval_output_file": str(m.eval_output_file),
            }
            for m in method_specs
        ],
    }
    _write_json(output_root / "run_context.json", run_context)

    # ── Reuse existing compatible eval outputs if requested ────────────────
    reuse_notes: list[str] = []
    if reuse_existing_evals:
        reused_dense, note_dense = _materialize_reused_eval(
            expected_output=dense_eval_output,
            eval_manifest=eval_manifest_path,
            expected_calibration=None,
            require_dynamic_zero=False,
            candidate_files=dense_candidates,
        )
        reuse_notes.append(f"dense: {note_dense}")

        for m in method_specs:
            reused, note = _materialize_reused_eval(
                expected_output=m.eval_output_file,
                eval_manifest=eval_manifest_path,
                expected_calibration=m.calibration_file,
                require_dynamic_zero=True,
                candidate_files=m.candidate_eval_files,
            )
            reuse_notes.append(f"{m.key}: {note}")

    _write_json(output_root / "reuse_decisions.json", {"notes": reuse_notes})

    # ── Build execution plan ────────────────────────────────────────────────
    execution_plan: dict[str, Any] = {
        "dense": {
            "output": str(dense_eval_output),
            "needs_run": False,
        },
        "methods": [],
    }

    dense_ok, dense_reason = _is_eval_output_compatible(
        dense_eval_output,
        eval_manifest=eval_manifest_path,
        expected_calibration=None,
        require_dynamic_zero=False,
    )
    dense_needs_run = args.force or (not dense_ok)
    execution_plan["dense"]["needs_run"] = dense_needs_run
    execution_plan["dense"]["compatibility_reason"] = dense_reason

    method_jobs: list[EvalJob] = []
    for m in method_specs:
        ok, reason = _is_eval_output_compatible(
            m.eval_output_file,
            eval_manifest=eval_manifest_path,
            expected_calibration=m.calibration_file,
            require_dynamic_zero=True,
        )
        needs_run = args.force or (not ok)
        execution_plan["methods"].append(
            {
                "method": m.key,
                "output": str(m.eval_output_file),
                "needs_run": needs_run,
                "compatibility_reason": reason,
            }
        )
        if needs_run:
            method_jobs.append(
                EvalJob(
                    run_id=f"fig6c_{m.key}",
                    calibration_file=m.calibration_file,
                    output_file=m.eval_output_file,
                )
            )

    _write_json(output_root / "execution_plan.json", execution_plan)

    if args.dry_run:
        print(f"Dry-run complete. Execution plan written to: {output_root / 'execution_plan.json'}")
        return 0

    if not args.execute:
        print(f"Preparation complete. Execution plan written to: {output_root / 'execution_plan.json'}")
        print("Run again with --execute to launch Figure 6(c) evaluations.")
        return 0

    # ── Execute evals if requested ──────────────────────────────────────────
    gpu_ids = _parse_gpu_list(args.gpus)

    if args.execute:
        if dense_needs_run:
            dense_gpu = gpu_ids[0] if gpu_ids else None
            dense_cmd = _build_dense_cmd(
                python_bin=args.python_bin,
                eval_manifest=eval_manifest_path,
                output_file=dense_eval_output,
                gpu=dense_gpu,
                use_lite=use_lite,
            )
            _run_command(
                dense_cmd,
                cwd=ONLINEARITH_ROOT,
                log_path=logs_dir / "fig6c_dense.log",
            )

        if method_jobs:
            if gpu_ids:
                jobs_by_gpu: dict[str, list[EvalJob]] = {gpu: [] for gpu in gpu_ids}
                for i, job in enumerate(method_jobs):
                    jobs_by_gpu[gpu_ids[i % len(gpu_ids)]].append(job)

                active_workers = [(gpu, jobs) for gpu, jobs in jobs_by_gpu.items() if jobs]

                def _worker(gpu_id: str, jobs: list[EvalJob]) -> list[str]:
                    done: list[str] = []
                    for job in jobs:
                        cmd = _build_quant_cmd(
                            python_bin=args.python_bin,
                            calibration_file=job.calibration_file,
                            eval_manifest=eval_manifest_path,
                            output_file=job.output_file,
                            gpu=gpu_id,
                            use_lite=use_lite,
                        )
                        _run_command(
                            cmd,
                            cwd=ONLINEARITH_ROOT,
                            log_path=logs_dir / f"{job.run_id}.log",
                        )
                        done.append(job.run_id)
                    return done

                with ThreadPoolExecutor(max_workers=len(active_workers)) as ex:
                    futures = {
                        ex.submit(_worker, gpu, jobs): gpu
                        for gpu, jobs in active_workers
                    }
                    for fut in as_completed(futures):
                        fut.result()
            else:
                for job in method_jobs:
                    cmd = _build_quant_cmd(
                        python_bin=args.python_bin,
                        calibration_file=job.calibration_file,
                        eval_manifest=eval_manifest_path,
                        output_file=job.output_file,
                        gpu=None,
                        use_lite=use_lite,
                    )
                    _run_command(
                        cmd,
                        cwd=ONLINEARITH_ROOT,
                        log_path=logs_dir / f"{job.run_id}.log",
                    )

    # ── Post-run checks & summary ───────────────────────────────────────────
    dense_ok, dense_reason = _is_eval_output_compatible(
        dense_eval_output,
        eval_manifest=eval_manifest_path,
        expected_calibration=None,
        require_dynamic_zero=False,
    )
    if not dense_ok:
        raise RuntimeError(f"Dense eval output is not compatible: {dense_reason}")

    for m in method_specs:
        ok, reason = _is_eval_output_compatible(
            m.eval_output_file,
            eval_manifest=eval_manifest_path,
            expected_calibration=m.calibration_file,
            require_dynamic_zero=True,
        )
        if not ok:
            raise RuntimeError(f"Method {m.key} eval output is not compatible: {reason}")

    _summarize(
        output_root=output_root,
        dense_eval_file=dense_eval_output,
        methods=method_specs,
        snr_star=float(args.target_snr),
        eval_manifest=eval_manifest_path,
    )

    print(f"Figure 6(c) run context : {output_root / 'run_context.json'}")
    print(f"Figure 6(c) plan        : {output_root / 'execution_plan.json'}")
    print(f"Figure 6(c) summary JSON: {output_root / 'figure6c_results.json'}")
    print(f"Figure 6(c) summary CSV : {output_root / 'figure6c_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
