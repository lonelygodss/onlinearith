"""
MSD inference performance statistics visualization.

Loads a ppl_results_*.json file (produced by ppltest.py with MSD enabled) and
produces diagnostic charts showing:
  - Layer-wise overview charts (all layers): p_eff, utilization, MAC sparsity,
    block breakdown, max latency
  - Channel-wise detail charts (detail layer only): distributions and
    correlations of per-channel metrics

No model loading required — operates purely on the saved result JSON.

Usage:
    python perf_viz.py ppl_results_MXFP8_MSD_B16.json
    python perf_viz.py ppl_results_MXFP8_MSD_B16_calib.json --output-dir perf_charts/
    python perf_viz.py ppl_results_MXFP8_MSD_B16.json --no-show
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short_layer_name(full_name):
    """'model.layers.2.mlp.gate_proj' -> 'L2.gate'"""
    m = re.match(r"model\.layers\.(\d+)\.mlp\.(\w+)_proj", full_name)
    if m:
        return f"L{m.group(1)}.{m.group(2)}"
    return full_name


def _proj_label(full_name):
    """'model.layers.2.mlp.gate_proj' -> 'gate_proj'"""
    return full_name.rsplit(".", 1)[-1] if "." in full_name else full_name


def _sorted_layer_names(per_layer):
    """Sort layer names by transformer layer index then projection order."""
    order = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
    def key(name):
        m = re.match(r"model\.layers\.(\d+)\.mlp\.(\w+)", name)
        if m:
            return (int(m.group(1)), order.get(m.group(2), 9))
        return (999, name)
    return sorted(per_layer.keys(), key=key)


def _extract_layer_summary(per_layer, names, section, field):
    """Extract a scalar field from the layer summary for each sorted layer."""
    vals = []
    for n in names:
        s = per_layer[n].get("summary", {})
        vals.append(s.get(section, {}).get(field, 0))
    return vals


def _get_detail_layers(per_layer):
    """Return layer names that have channel_detail."""
    return [n for n in per_layer if "channel_detail" in per_layer[n]]


# ── Chart 1: Layer-wise effective precision overview ─────────────────────────

def plot_layer_peff_overview(per_layer, fmt_tag, output_dir):
    """Bar chart: p_eff_mean and active_p_eff_mean per layer."""
    names = _sorted_layer_names(per_layer)
    short = [_short_layer_name(n) for n in names]
    p_eff = _extract_layer_summary(per_layer, names, "bit_level", "p_eff_mean")
    active_p = _extract_layer_summary(per_layer, names, "bit_level", "active_p_eff_mean")

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.22), 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, p_eff, width=w, color="#4c72b0", alpha=0.85, label="p_eff_mean (all)")
    ax.bar(x + w / 2, active_p, width=w, color="#55a868", alpha=0.85, label="active_p_eff_mean")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_ylabel("Effective Precision (digits)")
    ax.set_title(f"{fmt_tag} — Layer-wise Effective Precision")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_layer_peff.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 2: Layer-wise utilization ──────────────────────────────────────────

def plot_layer_utilization(per_layer, fmt_tag, output_dir):
    """Bar chart: utilization_mean per layer."""
    names = _sorted_layer_names(per_layer)
    short = [_short_layer_name(n) for n in names]
    util = _extract_layer_summary(per_layer, names, "channel_level", "utilization_mean")
    util_pct = [v * 100 for v in util]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.22), 5))
    x = np.arange(len(names))
    ax.bar(x, util_pct, width=0.7, color="#4c72b0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_ylabel("Utilization (%)")
    ax.set_title(f"{fmt_tag} — Layer-wise Budget Utilization")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_layer_utilization.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 3: Layer-wise MAC sparsity ─────────────────────────────────────────

def plot_layer_mac_sparsity(per_layer, fmt_tag, output_dir):
    """Bar chart: MAC sparsity per layer."""
    names = _sorted_layer_names(per_layer)
    short = [_short_layer_name(n) for n in names]
    sparsity = _extract_layer_summary(per_layer, names, "mac_level", "mac_sparsity")
    sparsity_pct = [v * 100 for v in sparsity]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.22), 5))
    x = np.arange(len(names))
    ax.bar(x, sparsity_pct, width=0.7, color="#dd8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_ylabel("MAC Sparsity (%)")
    ax.set_title(f"{fmt_tag} — Layer-wise MAC Sparsity")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_layer_mac_sparsity.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 4: Layer-wise block breakdown (stacked) ───────────────────────────

def plot_layer_block_breakdown(per_layer, fmt_tag, output_dir):
    """Stacked bar: zero / partial / full block ratios per layer."""
    names = _sorted_layer_names(per_layer)
    short = [_short_layer_name(n) for n in names]
    zero = _extract_layer_summary(per_layer, names, "block_level", "zero_block_ratio")
    partial = _extract_layer_summary(per_layer, names, "block_level", "partial_block_ratio")
    full = _extract_layer_summary(per_layer, names, "block_level", "full_block_ratio")
    zero_pct = [v * 100 for v in zero]
    partial_pct = [v * 100 for v in partial]
    full_pct = [v * 100 for v in full]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.22), 5))
    x = np.arange(len(names))
    ax.bar(x, zero_pct, width=0.7, color="#c44e52", alpha=0.85, label="Zero blocks")
    ax.bar(x, partial_pct, width=0.7, bottom=zero_pct, color="#dd8452", alpha=0.85, label="Partial blocks")
    bottoms = [z + p for z, p in zip(zero_pct, partial_pct)]
    ax.bar(x, full_pct, width=0.7, bottom=bottoms, color="#55a868", alpha=0.85, label="Full blocks")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_ylabel("Block Fraction (%)")
    ax.set_title(f"{fmt_tag} — Layer-wise Block Activation Breakdown")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_layer_blocks.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 5: Layer-wise max latency indicators ──────────────────────────────

def plot_layer_max_latency(per_layer, fmt_tag, output_dir):
    """Bar chart: max_budget and max_total_delay per layer."""
    names = _sorted_layer_names(per_layer)
    short = [_short_layer_name(n) for n in names]
    max_b = _extract_layer_summary(per_layer, names, "channel_level", "max_budget")
    max_d = _extract_layer_summary(per_layer, names, "channel_level", "max_total_delay")

    # Check if data exists (may be 0 for older JSONs)
    if all(v == 0 for v in max_b) and all(v == 0 for v in max_d):
        print("  Skipped max latency chart (no max_budget/max_total_delay data)")
        return None

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.22), 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, max_b, width=w, color="#4c72b0", alpha=0.85, label="Max budget (cycles)")
    ax.bar(x + w / 2, max_d, width=w, color="#c44e52", alpha=0.85, label="Max total delay (cycles)")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_ylabel("Cycles")
    ax.set_title(f"{fmt_tag} — Layer-wise Max Latency Indicators")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_layer_max_latency.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 6: Channel p_eff distribution ──────────────────────────────────────

def plot_channel_peff_distribution(per_layer, fmt_tag, output_dir):
    """Histogram of per-channel p_eff_mean for each detail-layer projection."""
    detail_names = _get_detail_layers(per_layer)
    if not detail_names:
        print("  Skipped channel p_eff histogram (no channel_detail data)")
        return None

    fig, axes = plt.subplots(1, len(detail_names), figsize=(5 * len(detail_names), 4), sharey=True)
    if len(detail_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, detail_names):
        d = per_layer[pname]["channel_detail"]
        p_eff = np.array(d.get("bit_level", {}).get("p_eff_mean", []))
        if len(p_eff) == 0:
            continue
        ax.hist(p_eff, bins=40, color="#4c72b0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("p_eff_mean")
        ax.set_title(_proj_label(pname))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Channel count")
    detail_layer = _short_layer_name(detail_names[0]).split(".")[0]
    fig.suptitle(f"{fmt_tag} — {detail_layer} Per-Channel Effective Precision", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_channel_peff_hist.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 7: Channel utilization distribution ────────────────────────────────

def plot_channel_utilization_distribution(per_layer, fmt_tag, output_dir):
    """Histogram of per-channel utilization for each detail-layer projection."""
    detail_names = _get_detail_layers(per_layer)
    if not detail_names:
        return None

    fig, axes = plt.subplots(1, len(detail_names), figsize=(5 * len(detail_names), 4), sharey=True)
    if len(detail_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, detail_names):
        d = per_layer[pname]["channel_detail"]
        util = np.array(d.get("channel_level", {}).get("utilization", []))
        if len(util) == 0:
            continue
        ax.hist(util * 100, bins=40, color="#55a868", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Utilization (%)")
        ax.set_title(_proj_label(pname))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Channel count")
    detail_layer = _short_layer_name(detail_names[0]).split(".")[0]
    fig.suptitle(f"{fmt_tag} — {detail_layer} Per-Channel Utilization", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_channel_util_hist.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 8: Channel MAC sparsity distribution ──────────────────────────────

def plot_channel_mac_sparsity_distribution(per_layer, fmt_tag, output_dir):
    """Histogram of per-channel MAC sparsity for each detail-layer projection."""
    detail_names = _get_detail_layers(per_layer)
    if not detail_names:
        return None

    fig, axes = plt.subplots(1, len(detail_names), figsize=(5 * len(detail_names), 4), sharey=True)
    if len(detail_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, detail_names):
        d = per_layer[pname]["channel_detail"]
        sparsity = np.array(d.get("mac_level", {}).get("mac_sparsity", []))
        if len(sparsity) == 0:
            continue
        ax.hist(sparsity * 100, bins=40, color="#dd8452", edgecolor="white", alpha=0.85)
        ax.set_xlabel("MAC Sparsity (%)")
        ax.set_title(_proj_label(pname))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Channel count")
    detail_layer = _short_layer_name(detail_names[0]).split(".")[0]
    fig.suptitle(f"{fmt_tag} — {detail_layer} Per-Channel MAC Sparsity", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_channel_mac_sparsity_hist.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 9: Channel p_eff vs utilization scatter ───────────────────────────

def plot_channel_peff_vs_utilization(per_layer, fmt_tag, output_dir):
    """Scatter: p_eff_mean (x) vs utilization (y) per channel."""
    detail_names = _get_detail_layers(per_layer)
    if not detail_names:
        return None

    fig, axes = plt.subplots(1, len(detail_names), figsize=(5 * len(detail_names), 4), sharey=True)
    if len(detail_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, detail_names):
        d = per_layer[pname]["channel_detail"]
        p_eff = np.array(d.get("bit_level", {}).get("p_eff_mean", []))
        util = np.array(d.get("channel_level", {}).get("utilization", []))
        if len(p_eff) == 0 or len(util) == 0:
            continue
        ax.scatter(p_eff, util * 100, s=4, alpha=0.5, c="#4c72b0")
        ax.set_xlabel("p_eff_mean (digits)")
        ax.set_title(_proj_label(pname))
        ax.grid(linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Utilization (%)")
    detail_layer = _short_layer_name(detail_names[0]).split(".")[0]
    fig.suptitle(f"{fmt_tag} — {detail_layer} Effective Precision vs Utilization", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_channel_peff_vs_util.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 10: Channel max delay distribution ────────────────────────────────

def plot_channel_max_delay_distribution(per_layer, fmt_tag, output_dir):
    """Histogram of per-channel max_total_delay for each detail-layer projection."""
    detail_names = _get_detail_layers(per_layer)
    if not detail_names:
        return None

    # Check if data exists
    has_data = False
    for pname in detail_names:
        d = per_layer[pname]["channel_detail"]
        arr = d.get("channel_level", {}).get("max_total_delay", [])
        if len(arr) > 0 and any(v > 0 for v in arr):
            has_data = True
            break
    if not has_data:
        print("  Skipped channel max delay histogram (no max_total_delay data)")
        return None

    fig, axes = plt.subplots(1, len(detail_names), figsize=(5 * len(detail_names), 4), sharey=True)
    if len(detail_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, detail_names):
        d = per_layer[pname]["channel_detail"]
        max_d = np.array(d.get("channel_level", {}).get("max_total_delay", []))
        max_b = np.array(d.get("channel_level", {}).get("max_budget", []))
        if len(max_d) == 0:
            continue
        ax.hist(max_d, bins=40, color="#c44e52", edgecolor="white", alpha=0.7, label="Max delay")
        if len(max_b) > 0:
            ax.hist(max_b, bins=40, color="#4c72b0", edgecolor="white", alpha=0.5, label="Max budget")
        ax.set_xlabel("Cycles")
        ax.set_title(_proj_label(pname))
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Channel count")
    detail_layer = _short_layer_name(detail_names[0]).split(".")[0]
    fig.suptitle(f"{fmt_tag} — {detail_layer} Per-Channel Max Latency Distribution", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"perf_{fmt_tag}_channel_max_delay_hist.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize MSD inference performance statistics (no model loading required)"
    )
    parser.add_argument("result_json", type=str,
                        help="Path to ppl_results_*.json file with msd_perf_stats")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for chart PNGs (default: same as JSON)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open interactive plot windows")
    args = parser.parse_args()

    # Load result data
    json_path = Path(args.result_json)
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    msd_perf = data.get("msd_perf_stats")
    if not msd_perf:
        print("Error: No msd_perf_stats found in JSON. Was this run with MSD enabled?")
        sys.exit(1)

    global_stats = msd_perf.get("global", {})
    per_layer = msd_perf.get("per_layer", {})

    if not per_layer:
        print("Error: No per_layer data in msd_perf_stats.")
        sys.exit(1)

    # Derive a tag for filenames
    fmt_tag = json_path.stem.replace("ppl_results_", "")

    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    print(f"Performance stats: {fmt_tag} ({global_stats.get('num_layers', '?')} layers)")
    print(f"  MAC sparsity : {global_stats.get('mac_sparsity', 0):.2%}")
    print(f"  Utilization  : {global_stats.get('global_utilization', 0):.2%}")
    print(f"  Max budget   : {global_stats.get('max_budget', 0):.1f} cycles")
    print(f"  Max delay    : {global_stats.get('max_total_delay', 0):.1f} cycles")
    detail_names = _get_detail_layers(per_layer)
    print(f"  Detail layers: {len(detail_names)}")
    print(f"Output: {output_dir}/")
    print()

    # Layer-wise charts (all layers)
    print("Layer-wise charts:")
    plot_layer_peff_overview(per_layer, fmt_tag, output_dir)
    plot_layer_utilization(per_layer, fmt_tag, output_dir)
    plot_layer_mac_sparsity(per_layer, fmt_tag, output_dir)
    plot_layer_block_breakdown(per_layer, fmt_tag, output_dir)
    plot_layer_max_latency(per_layer, fmt_tag, output_dir)

    # Channel-wise detail charts (detail layer only)
    if detail_names:
        print(f"\nChannel-wise detail charts ({len(detail_names)} projections):")
        plot_channel_peff_distribution(per_layer, fmt_tag, output_dir)
        plot_channel_utilization_distribution(per_layer, fmt_tag, output_dir)
        plot_channel_mac_sparsity_distribution(per_layer, fmt_tag, output_dir)
        plot_channel_peff_vs_utilization(per_layer, fmt_tag, output_dir)
        plot_channel_max_delay_distribution(per_layer, fmt_tag, output_dir)
    else:
        print("\nNo channel_detail found — skipping channel-wise charts.")
        print("(Run ppltest.py with --detail-layer to generate channel detail)")

    chart_count = len(list(output_dir.glob(f"perf_{fmt_tag}_*.png")))
    print(f"\nDone. {chart_count} charts saved.")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
