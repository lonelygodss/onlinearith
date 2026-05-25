#!/usr/bin/env python3
"""Merge projection-filtered MSD calibration JSON files.

The fixed-sum Qwen3-8B calibration plan runs projection-filtered full-model
jobs in parallel, then combines their disjoint layer metadata. This tool keeps
that merge explicit and reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MERGE_DICT_KEYS = (
    "msd_calibration_data",
    "optimizer_stats",
    "layer_stats",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if "msd_calibration_data" not in data:
        raise SystemExit(f"{path}: missing msd_calibration_data")
    return data


def _same_or_fail(key: str, first: dict[str, Any], other: dict[str, Any], path: Path) -> None:
    if first.get(key) != other.get(key):
        raise SystemExit(f"{path}: {key} does not match the first input")


def _merge_named_dict(
    key: str,
    merged: dict[str, Any],
    incoming: dict[str, Any],
    path: Path,
    allow_overwrite: bool,
) -> None:
    target = merged.setdefault(key, {})
    for name, value in incoming.get(key, {}).items():
        if name in target and target[name] != value and not allow_overwrite:
            raise SystemExit(f"{path}: duplicate conflicting {key} entry: {name}")
        target[name] = value


def _merge_channel_detail(
    merged: dict[str, Any],
    incoming: dict[str, Any],
    path: Path,
    allow_overwrite: bool,
) -> None:
    target = merged.setdefault("channel_detail", {})
    for name, value in incoming.get("channel_detail", {}).items():
        if name in target and target[name] != value and not allow_overwrite:
            raise SystemExit(f"{path}: duplicate conflicting channel_detail entry: {name}")
        target[name] = value


def _recompute_global_summary(data: dict[str, Any], input_summaries: list[dict[str, Any]]) -> None:
    calibration_data = data["msd_calibration_data"]
    layer_stats = data.get("layer_stats", {})
    total_channels = sum(len(budgets) for budgets in calibration_data.values())
    all_budgets = [float(v) for budgets in calibration_data.values() for v in budgets]

    def weighted_mean(field: str) -> float | None:
        num = 0.0
        den = 0
        for name, budgets in calibration_data.items():
            stats = layer_stats.get(name, {})
            if field not in stats:
                continue
            weight = len(budgets)
            num += float(stats[field]) * weight
            den += weight
        return num / den if den else None

    summary: dict[str, Any] = {
        "num_layers": len(calibration_data),
        "total_channels": total_channels,
    }
    if all_budgets:
        summary.update(
            {
                "budget_min": min(all_budgets),
                "budget_max": max(all_budgets),
                "budget_mean": round(sum(all_budgets) / len(all_budgets), 4),
            }
        )

    field_map = {
        "mean_snr": "snr_mean",
        "e_combined_mean": "e_combined_mean",
        "eff_precision_mean": "eff_precision_mean",
        "signal_power_db_mean": "signal_power_db_mean",
    }
    for out_key, layer_key in field_map.items():
        value = weighted_mean(layer_key)
        if value is not None:
            summary[out_key] = round(value, 4)

    min_snr_values = [
        float(stats["snr_min"])
        for name, stats in layer_stats.items()
        if name in calibration_data and "snr_min" in stats
    ]
    if min_snr_values:
        summary["min_snr"] = round(min(min_snr_values), 4)

    wall_time = sum(float(s.get("wall_time_sec", 0.0)) for s in input_summaries)
    if wall_time:
        summary["wall_time_sec"] = round(wall_time, 2)

    data["global_summary"] = summary


def merge_calibrations(paths: list[Path], allow_overwrite: bool) -> dict[str, Any]:
    if not paths:
        raise SystemExit("At least one input calibration is required.")

    loaded = [(path, _load_json(path)) for path in paths]
    first_path, first = loaded[0]
    merged = dict(first)
    for key in MERGE_DICT_KEYS:
        merged[key] = dict(first.get(key, {}))
    merged["channel_detail"] = dict(first.get("channel_detail", {}))

    for path, data in loaded[1:]:
        for key in ("format", "optimizer", "config_overrides"):
            _same_or_fail(key, first, data, path)
        for key in MERGE_DICT_KEYS:
            _merge_named_dict(key, merged, data, path, allow_overwrite)
        _merge_channel_detail(merged, data, path, allow_overwrite)

    merged["merged_from_projection_filters"] = [
        {
            "path": str(path),
            "projection_filter": data.get("calibration_params", {}).get("projection_filter"),
            "num_layers": len(data.get("msd_calibration_data", {})),
            "total_channels": sum(len(v) for v in data.get("msd_calibration_data", {}).values()),
        }
        for path, data in loaded
    ]
    _recompute_global_summary(merged, [data.get("global_summary", {}) for _, data in loaded])
    merged["description"] = (
        f"Merged from {len(paths)} projection-filtered calibration files; "
        f"{merged['global_summary']['num_layers']} layers."
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Input calibration JSON files.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Merged output JSON path.")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow duplicate layer keys, keeping the later input value.",
    )
    args = parser.parse_args()

    merged = merge_calibrations(args.inputs, allow_overwrite=args.allow_overwrite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")

    summary = merged["global_summary"]
    print(
        f"Merged {len(args.inputs)} files -> {args.output} "
        f"({summary['num_layers']} layers, {summary['total_channels']} channels)"
    )


if __name__ == "__main__":
    main()
