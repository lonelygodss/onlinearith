"""
Centralised experiment configuration for MXFP / MSD evaluation.

Single source of truth for:
  - All custom config field names and their baseline (everything-off) defaults
  - Setup definitions (SETUPS) used by ppltest.py, ppl_batch.py
  - MSD default dict and helper
  - Config application, snapshotting, and diffing utilities
  - MLP layer reconfiguration after config changes

Other scripts import from here instead of keeping local copies:
    from experiment_config import (
        BASELINE_CONFIG, SETUPS, apply_config, reset_to_baseline,
        reconfigure_mlp_layers, get_config_snapshot, get_active_flags,
    )
"""

from __future__ import annotations

import torch

# ── All custom MXFP / MSD config field names ────────────────────────────────
# These are the fields added to Qwen3Config beyond the standard HuggingFace
# fields.  Used for snapshotting, resetting, and diffing.

MXFP_MSD_FIELDS: list[str] = [
    # MXFP format selection (mutually exclusive: at most one True)
    "use_mxfp8",
    "mxfp8_block_size",
    "use_mxfp6",
    "mxfp6_block_size",
    "mxfp6_format",         # "e2m3" or "e3m2"
    "use_mxfp4",
    "mxfp4_block_size",
    # Activation-only structured sparsity
    "use_activation_nm_sparsity",
    "activation_nm_n",
    "activation_nm_m",
    # MSD truncation
    "use_msd_truncation",
    "msd_cycle_budget",
    "msd_online_delay",
    "msd_budget_dynamic_scale",
    "msd_budget_dynamic_threshold",
    "msd_budget_dynamic_mode",
    "msd_deep_pipeline",
    "msd_pipeline_precision_loss",
    "msd_calibration_data",
    "msd_chunk_target_mib",
    "msd_figure5_layer_cycles",
]

# ── Baseline config: everything off, all fields at their Qwen3Config defaults
# Applying this dict resets the model to a clean FP16 state.
BASELINE_CONFIG: dict = {
    "use_mxfp8": False,
    "mxfp8_block_size": 32,
    "use_mxfp6": False,
    "mxfp6_block_size": 32,
    "mxfp6_format": "e2m3",
    "use_mxfp4": False,
    "mxfp4_block_size": 32,
    "use_activation_nm_sparsity": False,
    "activation_nm_n": 2,
    "activation_nm_m": 4,
    "use_msd_truncation": False,
    "msd_cycle_budget": 16,
    "msd_online_delay": 2,
    "msd_budget_dynamic_scale": 0.0,
    "msd_budget_dynamic_threshold": 0.0,
    "msd_budget_dynamic_mode": "linear",
    "msd_deep_pipeline": False,
    "msd_pipeline_precision_loss": 2,
    "msd_calibration_data": None,
    "msd_chunk_target_mib": 1536,
    "msd_figure5_layer_cycles": False,
}


# ── MSD convenience defaults & helper ────────────────────────────────────────

_MSD_DEFAULTS: dict = {
    "use_msd_truncation": True,
    "msd_cycle_budget": 16,
    "msd_online_delay": 2,
    "msd_budget_dynamic_scale":0.0,
    "msd_budget_dynamic_threshold": 0.0,
    "msd_budget_dynamic_mode": "linear",
    "msd_deep_pipeline": False,
    "msd_pipeline_precision_loss": 2,
    "msd_calibration_data": None,
    "msd_figure5_layer_cycles": False,
}


def _msd(budget: int = 16, pipeline: bool = False, **extra) -> dict:
    """Return an MSD overrides dict with the given budget and pipeline flag."""
    d = dict(_MSD_DEFAULTS)
    d["msd_cycle_budget"] = budget
    d["msd_deep_pipeline"] = pipeline
    d.update(extra)
    return d


# ── Setup definitions ─────────────────────────────────────────────────────────
# Each setup is (id, tag, description, config_overrides_dict).
# Overrides are applied ON TOP of BASELINE_CONFIG.

SETUPS: list[tuple[int, str, str, dict]] = [
    (1,  "baseline",          "FP16 baseline (no quantization)",
     {"use_mxfp8": False, "use_mxfp6": False, "use_mxfp4": False}),

    (2,  "MXFP8",             "MXFP8 only",
     {"use_mxfp8": True}),

    (3,  "MXFP6_E2M3",       "MXFP6 E2M3 only",
     {"use_mxfp6": True, "mxfp6_format": "e2m3"}),

    (4,  "MXFP6_E3M2",       "MXFP6 E3M2 only",
     {"use_mxfp6": True, "mxfp6_format": "e3m2"}),

    (5,  "MXFP4",             "MXFP4 only",
     {"use_mxfp4": True}),

    (6,  "MXFP8_MSD_B16",    "MXFP8 + MSD B=16",
     {"use_mxfp8": True, **_msd(16)}),

    (7,  "MXFP6_E2M3_MSD_B16", "MXFP6 E2M3 + MSD B=16",
     {"use_mxfp6": True, "mxfp6_format": "e2m3", **_msd(16)}),

    (8,  "MXFP6_E3M2_MSD_B16", "MXFP6 E3M2 + MSD B=16",
     {"use_mxfp6": True, "mxfp6_format": "e3m2", **_msd(16)}),

    (9,  "MXFP4_MSD_B16",    "MXFP4 + MSD B=16",
     {"use_mxfp4": True, **_msd(16)}),

    (10, "MXFP8_MSD_B4",     "MXFP8 + MSD B=4",
     {"use_mxfp8": True, **_msd(4)}),

    (11, "MXFP8_MSD_B8",    "MXFP8 + MSD B=8",
     {"use_mxfp8": True, **_msd(8)}),

    (12, "MXFP8_MSD_B6",    "MXFP8 + MSD B=6",
     {"use_mxfp8": True, **_msd(6)}),

    (13, "MXFP8_MSD_B7",    "MXFP8 + MSD B=7",
     {"use_mxfp8": True, **_msd(7)}),

    (14, "MXFP8_MSD_B9",    "MXFP8 + MSD B=9",
     {"use_mxfp8": True, **_msd(9)}),

    (15, "MXFP8_MSD_B11",     "MXFP8 + MSD B=11",
     {"use_mxfp8": True, **_msd(11)}),

    (16, "MXFP8_MSD_B10",    "MXFP8 + MSD B=10",
     {"use_mxfp8": True, **_msd(10)}),


    (17, "MXFP8_MSD_B18",    "MXFP8 + MSD B=18",
     {"use_mxfp8": True, **_msd(18)}),

    (18, "MXFP4_MSD_B24",    "MXFP4 + MSD B=24",
     {"use_mxfp4": True, **_msd(24)}),

    (19, "MXFP4_MSD_B32",    "MXFP4 + MSD B=32",
     {"use_mxfp4": True, **_msd(32)}),

    (20, "MXFP8_MSD_B16_pipeline", "MXFP8 + MSD B=16 + pipeline",
     {"use_mxfp8": True, **_msd(16, pipeline=True)}),

    (21, "MXFP4_MSD_B16_pipeline", "MXFP4 + MSD B=16 + pipeline",
     {"use_mxfp4": True, **_msd(16, pipeline=True)}),
]


# ── Config application utilities ─────────────────────────────────────────────

def apply_config(config, overrides: dict) -> None:
    """Apply a dict of overrides to a model config (in-place via setattr)."""
    for k, v in overrides.items():
        setattr(config, k, v)


def reset_to_baseline(config) -> None:
    """Reset all MXFP/MSD fields to their baseline (everything-off) values."""
    apply_config(config, BASELINE_CONFIG)


def get_config_snapshot(config) -> dict:
    """
    Return a dict of all MXFP/MSD field values as currently set on *config*.

    Use this to embed a full, reproducible config state in result JSONs.
    Fields that are large (e.g. msd_calibration_data) are summarised.
    """
    snap = {}
    for field in MXFP_MSD_FIELDS:
        val = getattr(config, field, None)
        if field == "msd_calibration_data" and val is not None:
            # Summarise rather than dumping the full per-channel dict
            n_layers = len(val)
            n_channels = sum(len(v) for v in val.values())
            snap[field] = f"<{n_layers} layers, {n_channels} channels>"
        else:
            snap[field] = val
    return snap


def get_config_diff(config) -> dict:
    """Return only fields that differ from BASELINE_CONFIG."""
    diff = {}
    for field in MXFP_MSD_FIELDS:
        val = getattr(config, field, None)
        baseline_val = BASELINE_CONFIG.get(field)
        if val != baseline_val:
            if field == "msd_calibration_data" and val is not None:
                n_layers = len(val)
                n_channels = sum(len(v) for v in val.values())
                diff[field] = f"<{n_layers} layers, {n_channels} channels>"
            else:
                diff[field] = val
    return diff


def get_active_flags(config) -> str:
    """
    Return a concise human-readable string describing the active config.

    Examples:
        "(baseline fp16)"
        "MXFP8"
        "MXFP8 + MSD B=32, delay=2"
        "MXFP4 + MSD B=16 + pipeline"
        "MXFP6 E2M3 + MSD B=16 (calibrated: 84 layers)"
    """
    parts = []

    # Format
    if getattr(config, "use_mxfp8", False):
        parts.append("MXFP8")
    elif getattr(config, "use_mxfp6", False):
        fmt = getattr(config, "mxfp6_format", "e2m3")
        parts.append(f"MXFP6 {fmt.upper()}")
    elif getattr(config, "use_mxfp4", False):
        parts.append("MXFP4")
    else:
        parts.append("FP16 baseline")

    # Activation n:m runtime sparsity
    if getattr(config, "use_activation_nm_sparsity", False):
        n = getattr(config, "activation_nm_n", 2)
        m = getattr(config, "activation_nm_m", 4)
        parts.append(f"ACT N:M {n}:{m}")

    # MSD
    if getattr(config, "use_msd_truncation", False):
        budget = getattr(config, "msd_cycle_budget", 16)
        delay = getattr(config, "msd_online_delay", 2)
        msd_str = f"MSD B={budget}"
        if delay != 2:
            msd_str += f", delay={delay}"
        # Dynamic budget
        scale = getattr(config, "msd_budget_dynamic_scale", 1.0)
        if scale != 1.0:
            msd_str += f", dyn_scale={scale}"
        parts.append(msd_str)

    # Pipeline
    if getattr(config, "msd_deep_pipeline", False):
        ploss = getattr(config, "msd_pipeline_precision_loss", 2)
        parts.append(f"pipeline (ploss={ploss})")

    # Calibration
    cal = getattr(config, "msd_calibration_data", None)
    if cal is not None:
        n_layers = len(cal)
        parts.append(f"calibrated: {n_layers} layers")

    return " + ".join(parts)


def format_config_banner(config, setup_id: int | None = None,
                         setup_desc: str | None = None,
                         calibration_file: str | None = None) -> str:
    """
    Return a multi-line banner string summarising the active config.

    Example output::

        ┌──────────────────────────────────────────────────┐
        │  Setup #14: MXFP8 + MSD B=32                    │
        │  Active : MXFP8 + MSD B=32                      │
        │  Changed: use_mxfp8=True, msd_cycle_budget=32   │
        └──────────────────────────────────────────────────┘
    """
    lines = []
    if setup_id is not None and setup_desc is not None:
        lines.append(f"  Setup #{setup_id}: {setup_desc}")
    lines.append(f"  Active : {get_active_flags(config)}")

    diff = get_config_diff(config)
    if diff:
        diff_str = ", ".join(f"{k}={v}" for k, v in diff.items())
        lines.append(f"  Changed: {diff_str}")
    else:
        lines.append(f"  Changed: (none — baseline)")

    if calibration_file:
        lines.append(f"  Calibration: {calibration_file}")

    # Build box using ASCII for reliable terminal alignment
    max_len = max(len(line) for line in lines)
    w = max_len + 2  # padding
    box = [f"+{'-' * w}+"]
    for line in lines:
        box.append(f"|{line:<{w}}|")
    box.append(f"+{'-' * w}+")
    return "\n".join(box)


# ── MLP layer reconfiguration ────────────────────────────────────────────────

def reconfigure_mlp_layers(model, device: torch.device) -> None:
    """
    Replace every MLP linear layer with the correct type for the current config.

    ``_make_linear()`` is called at model construction and bakes in the linear
    class (MXFP8Linear, MXFP6Linear, MXFP4Linear, or nn.Linear).  Changing
    ``model.config`` later does NOT change the existing layer objects.  This
    function walks all MLP modules and rebuilds the three projections to
    match the current config, sharing the weight tensor so no data is copied.
    """
    from transformers.models.qwen3.modeling_qwen3 import _make_linear, Qwen3MLP

    config = model.config
    for module in model.modules():
        if not isinstance(module, Qwen3MLP):
            continue
        for attr in ("gate_proj", "up_proj", "down_proj"):
            old = getattr(module, attr)
            new = _make_linear(old.in_features, old.out_features, config)
            new.weight = old.weight          # share nn.Parameter (no copy)
            if hasattr(old, "bias_param") and old.bias_param is not None:
                new.bias_param = old.bias_param
            new = new.to(device)
            # Preserve train/eval mode of replaced projections.
            # This is required because new modules default to train mode.
            new.train(old.training)
            setattr(module, attr, new)

    # Invalidate MSD context so it re-walks modules & re-sets layer_name etc.
    if hasattr(model, "_msd_context"):
        model._msd_context = None
        model._msd_context_config_hash = None


# ── Memory helpers ────────────────────────────────────────────────────────────

def peak_memory_str(device: torch.device) -> str:
    """Return peak allocated memory as a human-readable string."""
    if device.type == "cuda":
        bytes_ = torch.cuda.max_memory_allocated(device)
        return f"{bytes_ / 1024**3:.2f} GB"
    return "N/A (CPU)"


def reset_peak_memory(device: torch.device) -> None:
    """Reset peak memory tracking for the given device."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
