"""
Calibration quality visualization — diagnostic charts for MSD budget calibration.

Loads a calibration_{tag}.json file and produces diagnostic charts showing:
  - Layer-wise budget and SNR overview (all layers)
  - Channel-wise detail for the detail layer (budget distribution, correlations)

No model loading required — operates purely on the saved calibration JSON.

Usage:
    cd /home/xzj/coding/onlinearith
    source /home/xzj/coding/.venv3_10/bin/activate

    python calibration_viz.py calibration_MXFP8.json
    python calibration_viz.py calibration_MXFP8.json --output-dir calib_charts/
    python calibration_viz.py calibration_MXFP8.json --no-show
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short_layer_name(full_name):
    """'model.layers.2.mlp.gate_proj' → 'L2.gate'"""
    m = re.match(r"model\.layers\.(\d+)\.mlp\.(\w+)_proj", full_name)
    if m:
        return f"L{m.group(1)}.{m.group(2)}"
    return full_name


def _proj_label(full_name):
    """'model.layers.2.mlp.gate_proj' → 'gate_proj'"""
    return full_name.rsplit(".", 1)[-1] if "." in full_name else full_name


def _sorted_layer_names(layer_stats):
    """Sort layer names by transformer layer index then projection order."""
    order = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
    def key(name):
        m = re.match(r"model\.layers\.(\d+)\.mlp\.(\w+)", name)
        if m:
            return (int(m.group(1)), order.get(m.group(2), 9))
        return (999, name)
    return sorted(layer_stats.keys(), key=key)


# ── Chart 1: Layer-wise budget overview ──────────────────────────────────────

def plot_layer_budget_overview(layer_stats, fmt_tag, output_dir):
    """Bar chart: mean budget per layer with ±1σ error bars."""
    names = _sorted_layer_names(layer_stats)
    short = [_short_layer_name(n) for n in names]
    means = [layer_stats[n]["budget_mean"] for n in names]
    stds = [layer_stats[n].get("budget_std", 0) for n in names]
    b_min = [layer_stats[n]["budget_min"] for n in names]
    b_max = [layer_stats[n]["budget_max"] for n in names]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.22), 5))
    x = np.arange(len(names))
    yerr_lo = [m - mn for m, mn in zip(means, b_min)]
    yerr_hi = [mx - m for m, mx in zip(means, b_max)]
    ax.bar(x, means, width=0.7, color="#4c72b0", alpha=0.85, label="Mean budget")
    ax.errorbar(x, means, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="gray",
                capsize=2, linewidth=0.8, label="Min–Max range")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_ylabel("Cycle Budget (B)")
    ax.set_title(f"{fmt_tag} — Layer-wise Budget Overview")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = output_dir / f"calib_{fmt_tag}_layer_budget.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 2: Layer-wise SNR overview ─────────────────────────────────────────

def plot_layer_snr_overview(layer_stats, fmt_tag, target_snr, output_dir):
    """Bar chart: mean and min SNR per layer with target line."""
    names = _sorted_layer_names(layer_stats)
    short = [_short_layer_name(n) for n in names]
    snr_means = [layer_stats[n].get("snr_mean") for n in names]
    snr_mins = [layer_stats[n].get("snr_min") for n in names]

    # Filter out None values
    valid = [(i, s, sm, sn) for i, (s, sm, sn) in enumerate(zip(short, snr_means, snr_mins))
             if sm is not None]
    if not valid:
        print("  Skipped layer SNR chart (no SNR data)")
        return None

    idx, short_v, means_v, mins_v = zip(*valid)
    x = np.arange(len(short_v))

    fig, ax = plt.subplots(figsize=(max(14, len(short_v) * 0.22), 5))
    w = 0.35
    ax.bar(x - w/2, means_v, width=w, color="#4c72b0", alpha=0.85, label="Mean SNR")
    ax.bar(x + w/2, mins_v, width=w, color="#dd8452", alpha=0.85, label="Min SNR")
    ax.axhline(target_snr, color="red", linestyle="--", linewidth=1.2, label=f"Target ({target_snr} dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_v, rotation=90, fontsize=6)
    ax.set_ylabel("SNR (dB)")
    ax.set_title(f"{fmt_tag} — Layer-wise SNR Overview")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = output_dir / f"calib_{fmt_tag}_layer_snr.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 3: Detail layer — Budget histogram ────────────────────────────────

def plot_budget_histogram(channel_detail, fmt_tag, detail_layer, output_dir):
    """Histogram of budget values for each of the 3 projections."""
    proj_names = [n for n in channel_detail if n != "detail_layer"]
    if not proj_names:
        print("  Skipped budget histogram (no channel detail)")
        return None

    fig, axes = plt.subplots(1, len(proj_names), figsize=(5 * len(proj_names), 4),
                             sharey=True)
    if len(proj_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, proj_names):
        budgets = channel_detail[pname].get("budget", [])
        if not budgets:
            continue
        budgets = np.array(budgets)
        b_min, b_max = int(budgets.min()), int(budgets.max())
        bins = np.arange(b_min, b_max + 2) - 0.5
        ax.hist(budgets, bins=bins, color="#4c72b0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Budget (B)")
        ax.set_title(_proj_label(pname))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Channel count")
    fig.suptitle(f"{fmt_tag} — Layer {detail_layer} Budget Distribution", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"calib_{fmt_tag}_L{detail_layer}_budget_hist.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 4: Budget vs e_combined scatter ────────────────────────────────────

def plot_budget_vs_ecombined(channel_detail, fmt_tag, detail_layer, output_dir):
    """Scatter: budget (y) vs e_combined_mean (x) per channel."""
    proj_names = [n for n in channel_detail if n != "detail_layer"]
    if not proj_names:
        return None

    fig, axes = plt.subplots(1, len(proj_names), figsize=(5 * len(proj_names), 4),
                             sharey=True)
    if len(proj_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, proj_names):
        d = channel_detail[pname]
        budgets = np.array(d.get("budget", []))
        e_comb = np.array(d.get("e_combined_mean", []))
        if len(budgets) == 0 or len(e_comb) == 0:
            continue
        ax.scatter(e_comb, budgets, s=4, alpha=0.5, c="#4c72b0")
        ax.set_xlabel("e_combined_mean")
        ax.set_title(_proj_label(pname))
        ax.grid(linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Budget (B)")
    fig.suptitle(f"{fmt_tag} — Layer {detail_layer} Budget vs Combined Exponent", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"calib_{fmt_tag}_L{detail_layer}_budget_vs_ecomb.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 5: Budget vs signal_power scatter ──────────────────────────────────

def plot_budget_vs_signal_power(channel_detail, fmt_tag, detail_layer, output_dir):
    """Scatter: budget (y) vs signal_power_db (x) per channel."""
    proj_names = [n for n in channel_detail if n != "detail_layer"]
    if not proj_names:
        return None

    fig, axes = plt.subplots(1, len(proj_names), figsize=(5 * len(proj_names), 4),
                             sharey=True)
    if len(proj_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, proj_names):
        d = channel_detail[pname]
        budgets = np.array(d.get("budget", []))
        sig_pow = np.array(d.get("signal_power_db", []))
        if len(budgets) == 0 or len(sig_pow) == 0:
            continue
        ax.scatter(sig_pow, budgets, s=4, alpha=0.5, c="#55a868")
        ax.set_xlabel("signal_power_db")
        ax.set_title(_proj_label(pname))
        ax.grid(linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Budget (B)")
    fig.suptitle(f"{fmt_tag} — Layer {detail_layer} Budget vs Signal Power", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"calib_{fmt_tag}_L{detail_layer}_budget_vs_sigpow.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 6: SNR distribution ───────────────────────────────────────────────

def plot_snr_distribution(channel_detail, fmt_tag, detail_layer, target_snr, output_dir):
    """Histogram of per-channel SNR with target line."""
    proj_names = [n for n in channel_detail if n != "detail_layer"]
    if not proj_names:
        return None

    fig, axes = plt.subplots(1, len(proj_names), figsize=(5 * len(proj_names), 4),
                             sharey=True)
    if len(proj_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, proj_names):
        snr = np.array(channel_detail[pname].get("snr_at_budget", []))
        if len(snr) == 0:
            continue
        ax.hist(snr, bins=40, color="#dd8452", edgecolor="white", alpha=0.85)
        ax.axvline(target_snr, color="red", linestyle="--", linewidth=1.2,
                   label=f"Target ({target_snr} dB)")
        below = (snr < target_snr).sum()
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"{_proj_label(pname)} ({below} below target)")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Channel count")
    fig.suptitle(f"{fmt_tag} — Layer {detail_layer} SNR Distribution", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"calib_{fmt_tag}_L{detail_layer}_snr_dist.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 7: Effective precision vs budget ───────────────────────────────────

def plot_effprec_vs_budget(channel_detail, fmt_tag, detail_layer, output_dir):
    """Scatter: eff_precision_mean (y) vs budget (x) per channel."""
    proj_names = [n for n in channel_detail if n != "detail_layer"]
    if not proj_names:
        return None

    fig, axes = plt.subplots(1, len(proj_names), figsize=(5 * len(proj_names), 4),
                             sharey=True)
    if len(proj_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, proj_names):
        d = channel_detail[pname]
        budgets = np.array(d.get("budget", []))
        eff_prec = np.array(d.get("eff_precision_mean", []))
        if len(budgets) == 0 or len(eff_prec) == 0:
            continue
        ax.scatter(budgets, eff_prec, s=4, alpha=0.5, c="#c44e52")
        # Reference line y = x (perfect precision = full budget)
        bmin, bmax = budgets.min(), budgets.max()
        ax.plot([bmin, bmax], [bmin, bmax], "k--", alpha=0.3, linewidth=0.8, label="y=x (zero delay)")
        ax.set_xlabel("Budget (B)")
        ax.set_title(_proj_label(pname))
        ax.legend(fontsize=7)
        ax.grid(linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Effective Precision (mean)")
    fig.suptitle(f"{fmt_tag} — Layer {detail_layer} Effective Precision vs Budget", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"calib_{fmt_tag}_L{detail_layer}_effprec_vs_budget.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Chart 8: Inter-delay vs e_combined ───────────────────────────────────────

def plot_interdelay_vs_ecombined(channel_detail, fmt_tag, detail_layer, output_dir):
    """Scatter: inter_delay_mean (y) vs e_combined_mean (x) per channel."""
    proj_names = [n for n in channel_detail if n != "detail_layer"]
    if not proj_names:
        return None

    fig, axes = plt.subplots(1, len(proj_names), figsize=(5 * len(proj_names), 4),
                             sharey=True)
    if len(proj_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, proj_names):
        d = channel_detail[pname]
        e_comb = np.array(d.get("e_combined_mean", []))
        inter_d = np.array(d.get("inter_delay_mean", []))
        if len(e_comb) == 0 or len(inter_d) == 0:
            continue
        ax.scatter(e_comb, inter_d, s=4, alpha=0.5, c="#8172b2")
        ax.set_xlabel("e_combined_mean")
        ax.set_title(_proj_label(pname))
        ax.grid(linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Inter-block Delay (mean)")
    fig.suptitle(f"{fmt_tag} — Layer {detail_layer} Inter-delay vs Combined Exponent",
                 fontsize=12)
    fig.tight_layout()
    path = output_dir / f"calib_{fmt_tag}_L{detail_layer}_interdelay_vs_ecomb.png"
    fig.savefig(path, dpi=200)
    print(f"  Saved: {path.name}")
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize MSD calibration results (no model loading required)"
    )
    parser.add_argument("calibration_json", type=str,
                        help="Path to calibration_{tag}.json file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for chart PNGs (default: same as JSON)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open interactive plot windows")
    args = parser.parse_args()

    # Load calibration data
    json_path = Path(args.calibration_json)
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    fmt_tag = data.get("format", json_path.stem.replace("calibration_", ""))
    target_snr = data.get("calibration_params", {}).get("target_snr_db", 30.0)
    layer_stats = data.get("layer_stats", {})
    channel_detail = data.get("channel_detail", {})
    detail_layer = channel_detail.get("detail_layer",
                                      data.get("calibration_params", {}).get("detail_layer", "?"))

    if not layer_stats:
        print("Error: No layer_stats found in JSON. Was this produced by the updated calibrate.py?")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Calibration: {fmt_tag} ({len(layer_stats)} layers)")
    print(f"Target SNR: {target_snr} dB")
    print(f"Detail layer: {detail_layer}")
    print(f"Output: {output_dir}/")
    print()

    # Layer-wise charts (all layers)
    print("Layer-wise charts:")
    plot_layer_budget_overview(layer_stats, fmt_tag, output_dir)
    plot_layer_snr_overview(layer_stats, fmt_tag, target_snr, output_dir)

    # Channel-wise detail charts (detail layer only)
    proj_keys = [k for k in channel_detail if k != "detail_layer"]
    if proj_keys:
        print(f"\nChannel-wise detail charts (layer {detail_layer}):")
        plot_budget_histogram(channel_detail, fmt_tag, detail_layer, output_dir)
        plot_budget_vs_ecombined(channel_detail, fmt_tag, detail_layer, output_dir)
        plot_budget_vs_signal_power(channel_detail, fmt_tag, detail_layer, output_dir)
        plot_snr_distribution(channel_detail, fmt_tag, detail_layer, target_snr, output_dir)
        plot_effprec_vs_budget(channel_detail, fmt_tag, detail_layer, output_dir)
        plot_interdelay_vs_ecombined(channel_detail, fmt_tag, detail_layer, output_dir)
    else:
        print("\nNo channel_detail found — skipping channel-wise charts.")
        print("(Re-run calibrate.py with the updated code to generate channel detail)")

    print(f"\nDone. {len(list(output_dir.glob(f'calib_{fmt_tag}_*.png')))} charts saved.")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
