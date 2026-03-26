"""
Batch analysis for MSD arrival-window proxy from calibration outputs.

This script is independent from ppltest.py. It scans a target directory for
subfolders named like "10db", "27db", "30db", etc., loads calibration JSON
files, and computes the combined owner-lane budget proxy:

    combined_budget[channel] = max(B_up[channel], B_gate[channel])

for every MLP layer/channel pair. It then exports:
    - JSON summary per calibration file (per file + per layer)
    - CSV tables per calibration file (layer summary + histogram bins)
    - PNG histogram per calibration file

Usage:
  python msd_arrival_hist_batch.py
    python msd_arrival_hist_batch.py --input-dir ../data/calib-data --output-dir ../data/calib-charts/arrival_hist
  python msd_arrival_hist_batch.py --no-plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_DB_DIR_RE = re.compile(r"^\d+(?:\.\d+)?db$", flags=re.IGNORECASE)


@dataclass
class FileResult:
    db_group: str
    file_path: Path
    fmt_tag: str | None
    optimizer: str | None
    num_layer_pairs: int
    num_channels: int
    histogram: dict[str, int]
    stats: dict[str, float]
    per_layer: dict[str, dict[str, object]]
    out_dir: Path
    written_files: list[Path]


def _round_to_int(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "round":
        return np.rint(values).astype(np.int64)
    if mode == "floor":
        return np.floor(values).astype(np.int64)
    if mode == "ceil":
        return np.ceil(values).astype(np.int64)
    raise ValueError(f"Unsupported rounding mode: {mode}")


def _build_histogram(values_int: np.ndarray) -> dict[str, int]:
    if values_int.size == 0:
        return {}
    uniq, counts = np.unique(values_int, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}


def _compute_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p95": math.nan,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def _find_db_dirs(root: Path) -> list[Path]:
    db_dirs: list[Path] = []
    if _DB_DIR_RE.match(root.name):
        db_dirs.append(root)
    db_dirs.extend(p for p in root.rglob("*") if p.is_dir() and _DB_DIR_RE.match(p.name))
    # Deduplicate while preserving deterministic order.
    seen = set()
    out = []
    for p in sorted(db_dirs):
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _iter_calibration_files(db_dir: Path) -> Iterable[Path]:
    yield from sorted(db_dir.glob("calibration_*.json"))


def _extract_combined_budgets(
    msd_data: dict, strict: bool
) -> tuple[np.ndarray, int, dict[str, np.ndarray], list[str]]:
    errors: list[str] = []
    combined_chunks: list[np.ndarray] = []
    per_layer_chunks: dict[str, list[np.ndarray]] = {}
    layer_pairs = 0

    for gate_key, gate_vals in msd_data.items():
        if not gate_key.endswith(".mlp.gate_proj"):
            continue
        up_key = gate_key.replace(".mlp.gate_proj", ".mlp.up_proj")
        if up_key not in msd_data:
            msg = f"Missing paired up_proj for {gate_key}"
            if strict:
                raise ValueError(msg)
            errors.append(msg)
            continue

        gate_arr = np.asarray(gate_vals, dtype=np.float64)
        up_arr = np.asarray(msd_data[up_key], dtype=np.float64)
        if gate_arr.ndim != 1 or up_arr.ndim != 1:
            msg = f"Non-1D calibration arrays for pair {gate_key} / {up_key}"
            if strict:
                raise ValueError(msg)
            errors.append(msg)
            continue

        if gate_arr.size == 0 or up_arr.size == 0:
            msg = f"Empty calibration array for pair {gate_key} / {up_key}"
            if strict:
                raise ValueError(msg)
            errors.append(msg)
            continue

        if gate_arr.size != up_arr.size:
            msg = (
                f"Mismatched channel counts for {gate_key} / {up_key}: "
                f"gate={gate_arr.size}, up={up_arr.size}; using min length"
            )
            if strict:
                raise ValueError(msg)
            errors.append(msg)

        m = min(gate_arr.size, up_arr.size)
        combined = np.maximum(gate_arr[:m], up_arr[:m])
        combined_chunks.append(combined)
        layer_name = gate_key.replace(".mlp.gate_proj", "")
        per_layer_chunks.setdefault(layer_name, []).append(combined)
        layer_pairs += 1

    if not combined_chunks:
        raise ValueError("No gate/up projection pairs found in msd_calibration_data")

    per_layer = {
        layer_name: np.concatenate(chunks, axis=0)
        for layer_name, chunks in per_layer_chunks.items()
    }
    return np.concatenate(combined_chunks, axis=0), layer_pairs, per_layer, errors


def _plot_histogram_from_counts(counts: dict[str, int], title: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    if not counts:
        return

    xs = sorted(int(k) for k in counts.keys())
    ys = [counts[str(x)] for x in xs]

    plt.figure(figsize=(9, 5))
    plt.bar(xs, ys, width=0.8)
    plt.xlabel("Combined budget (cycles), integer bin")
    plt.ylabel("Channel count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def _write_per_file_outputs(
    result: FileResult,
    rounding: str,
    plot_enabled: bool,
    overwrite: bool,
) -> list[Path]:
    out_dir = result.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "arrival_hist.json"
    out_layer_csv = out_dir / "arrival_hist_layer_summary.csv"
    out_bins_csv = out_dir / "arrival_hist_bins.csv"
    out_png = out_dir / "arrival_hist.png"

    if out_json.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {out_json}. Use --overwrite to overwrite per-file outputs."
        )

    payload = {
        "source": {
            "db_group": result.db_group,
            "file": str(result.file_path),
            "format": result.fmt_tag,
            "optimizer": result.optimizer,
        },
        "binning": {
            "mode": "integer_cycles",
            "rounding": rounding,
        },
        "file": {
            "num_layer_pairs": result.num_layer_pairs,
            "num_channels": result.num_channels,
            "histogram": result.histogram,
            "histogram_total": int(sum(result.histogram.values())),
            "stats": result.stats,
        },
        "per_layer": result.per_layer,
    }

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    with open(out_layer_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "num_channels",
            "min",
            "max",
            "mean",
            "std",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
        ])
        for layer_name, layer_data in sorted(result.per_layer.items()):
            s = layer_data["stats"]
            writer.writerow([
                layer_name,
                layer_data["num_channels"],
                s["min"],
                s["max"],
                s["mean"],
                s["std"],
                s["p05"],
                s["p25"],
                s["p50"],
                s["p75"],
                s["p95"],
            ])

    with open(out_bins_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "layer", "bin", "count", "ratio"])

        total = max(1, int(sum(result.histogram.values())))
        for b, c in sorted(result.histogram.items(), key=lambda kv: int(kv[0])):
            writer.writerow(["file", "", b, c, c / total])

        for layer_name, layer_data in sorted(result.per_layer.items()):
            layer_hist = layer_data["histogram"]
            layer_total = max(1, int(layer_data["histogram_total"]))
            for b, c in sorted(layer_hist.items(), key=lambda kv: int(kv[0])):
                writer.writerow(["layer", layer_name, b, c, c / layer_total])

    written = [out_json, out_layer_csv, out_bins_csv]

    if plot_enabled:
        _plot_histogram_from_counts(
            result.histogram,
            title=f"{result.db_group} | {result.file_path.stem} | combined max(up,gate)",
            out_png=out_png,
        )
        written.append(out_png)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch histogram extraction for MSD combined up/gate budgets"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../data/calib-data"),
        help="Root directory containing xxdb calibration subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/calib-charts/arrival_hist"),
        help="Output root that mirrors calib-data hierarchy by SNR and calibration file",
    )
    parser.add_argument(
        "--rounding",
        choices=["round", "floor", "ceil"],
        default="round",
        help="How to map combined budgets to integer cycle bins",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on malformed files instead of skipping",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG generation (JSON/CSV still generated)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    db_dirs = _find_db_dirs(input_dir)
    if not db_dirs:
        raise RuntimeError(f"No xxdb folders found under: {input_dir}")

    results: list[FileResult] = []
    skipped_files: list[dict[str, str]] = []
    parse_warnings: list[dict[str, str]] = []

    # Validate plotting dependency once before per-file writes.
    plot_enabled = not args.no_plot
    if plot_enabled:
        try:
            import matplotlib  # noqa: F401
        except Exception:
            plot_enabled = False
            parse_warnings.append({
                "file": "<plotting>",
                "warning": "matplotlib is unavailable; skipped PNG generation",
            })

    for db_dir in db_dirs:
        db_name = db_dir.name
        for cal_file in _iter_calibration_files(db_dir):
            try:
                with open(cal_file) as f:
                    data = json.load(f)
                msd_data = data.get("msd_calibration_data")
                if not isinstance(msd_data, dict) or not msd_data:
                    raise ValueError("Missing or empty msd_calibration_data")

                combined, n_pairs, per_layer_values, errs = _extract_combined_budgets(msd_data, strict=args.strict)
                for e in errs:
                    parse_warnings.append({"file": str(cal_file), "warning": e})

                combined_int = _round_to_int(combined, args.rounding)
                hist = _build_histogram(combined_int)
                stats = _compute_stats(combined)
                per_layer = {}
                for layer_name, layer_values in sorted(per_layer_values.items()):
                    layer_int = _round_to_int(layer_values, args.rounding)
                    layer_hist = _build_histogram(layer_int)
                    per_layer[layer_name] = {
                        "num_channels": int(layer_values.size),
                        "histogram": layer_hist,
                        "histogram_total": int(sum(layer_hist.values())),
                        "stats": _compute_stats(layer_values),
                    }

                rel_db_path = db_dir.relative_to(input_dir)
                out_dir = output_dir / rel_db_path / cal_file.stem

                result = FileResult(
                    db_group=db_name,
                    file_path=cal_file,
                    fmt_tag=data.get("format"),
                    optimizer=data.get("optimizer"),
                    num_layer_pairs=n_pairs,
                    num_channels=int(combined.size),
                    histogram=hist,
                    stats=stats,
                    per_layer=per_layer,
                    out_dir=out_dir,
                    written_files=[],
                )

                result.written_files = _write_per_file_outputs(
                    result,
                    rounding=args.rounding,
                    plot_enabled=plot_enabled,
                    overwrite=args.overwrite,
                )
                results.append(result)
            except Exception as exc:
                if args.strict:
                    raise
                skipped_files.append({"file": str(cal_file), "error": str(exc)})

    if not results:
        raise RuntimeError("No calibration files were successfully processed.")

    print("MSD arrival histogram batch completed")
    print(f"  Input root       : {input_dir}")
    print(f"  Output dir       : {output_dir}")
    print(f"  Files processed  : {len(results)}")
    print(f"  Files skipped    : {len(skipped_files)}")
    for r in sorted(results, key=lambda x: (x.db_group, x.file_path.name)):
        print(f"  Wrote            : {r.file_path}")
        for out in r.written_files:
            print(f"    -> {out}")
    if skipped_files:
        print("  Skipped files:")
        for item in skipped_files:
            print(f"    - {item['file']}: {item['error']}")
    if parse_warnings:
        print("  Warnings:")
        for item in parse_warnings:
            print(f"    - {item['file']}: {item['warning']}")
    if plot_enabled:
        print("  PNG histograms   : generated per calibration file")
    else:
        print("  PNG histograms   : skipped")


if __name__ == "__main__":
    main()
