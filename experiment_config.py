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

# ── All custom MXFP / MSD config fields ─────────────────────────────────────
# These are the fields added to Qwen3Config beyond standard HuggingFace fields.
# They are used for setup validation, snapshotting, resetting, and diffing.

MXFP_FORMAT_FLAGS: tuple[str, ...] = ("use_mxfp8", "use_mxfp6", "use_mxfp4")
MSD_RUNTIME_STATS_FIELDS: tuple[str, ...] = (
    "msd_perf_stats_enabled",
    "msd_perf_stats_lite",
    "msd_figure5_layer_cycles",
)

CUSTOM_QWEN3_CONFIG_DEFAULTS: dict = {
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
    # Onlinearith experiment default. Qwen3Config's constructor default remains
    # 512 MiB for upstream-compatible standalone model construction.
    "msd_chunk_target_mib": 1536,
    "msd_perf_stats_enabled": True,
    "msd_perf_stats_lite": False,
    "msd_figure5_layer_cycles": False,
}

MXFP_MSD_FIELDS: list[str] = list(CUSTOM_QWEN3_CONFIG_DEFAULTS)

# Applying this dict resets the model to the onlinearith FP16 baseline state.
BASELINE_CONFIG: dict = dict(CUSTOM_QWEN3_CONFIG_DEFAULTS)


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
    "msd_perf_stats_enabled": True,
    "msd_perf_stats_lite": False,
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


def _require_positive_int(name: str, value: object, setup_id: int, tag: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Setup {setup_id}/{tag}: {name} must be a positive integer, got {value!r}")


def validate_setup_definition(setup: tuple[int, str, str, dict]) -> None:
    """Validate one setup tuple against the registered custom config fields."""
    if not (isinstance(setup, tuple) and len(setup) == 4):
        raise ValueError(f"Invalid setup tuple shape: {setup!r}")

    setup_id, tag, description, overrides = setup
    if not isinstance(setup_id, int):
        raise ValueError(f"Setup id must be an int, got {setup_id!r}")
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"Setup {setup_id}: tag must be a non-empty string")
    if not isinstance(description, str) or not description:
        raise ValueError(f"Setup {setup_id}/{tag}: description must be a non-empty string")
    if not isinstance(overrides, dict):
        raise ValueError(f"Setup {setup_id}/{tag}: overrides must be a dict")

    unknown = sorted(set(overrides) - set(BASELINE_CONFIG))
    if unknown:
        raise ValueError(f"Setup {setup_id}/{tag}: unknown override keys {unknown}")

    active_config = dict(BASELINE_CONFIG)
    active_config.update(overrides)
    enabled_formats = [flag for flag in MXFP_FORMAT_FLAGS if bool(active_config.get(flag))]
    if len(enabled_formats) > 1:
        raise ValueError(f"Setup {setup_id}/{tag}: MX format flags are mutually exclusive, got {enabled_formats}")

    if active_config["mxfp6_format"] not in {"e2m3", "e3m2"}:
        raise ValueError(f"Setup {setup_id}/{tag}: mxfp6_format must be 'e2m3' or 'e3m2'")

    for field in ("mxfp8_block_size", "mxfp6_block_size", "mxfp4_block_size"):
        _require_positive_int(field, active_config[field], setup_id, tag)

    if active_config["use_activation_nm_sparsity"]:
        _require_positive_int("activation_nm_n", active_config["activation_nm_n"], setup_id, tag)
        _require_positive_int("activation_nm_m", active_config["activation_nm_m"], setup_id, tag)
        if active_config["activation_nm_n"] > active_config["activation_nm_m"]:
            raise ValueError(f"Setup {setup_id}/{tag}: activation_nm_n must be <= activation_nm_m")

    if active_config["use_msd_truncation"]:
        for field in ("msd_cycle_budget", "msd_online_delay", "msd_pipeline_precision_loss"):
            _require_positive_int(field, active_config[field], setup_id, tag)
    _require_positive_int("msd_chunk_target_mib", active_config["msd_chunk_target_mib"], setup_id, tag)


def validate_all_setups(setups: list[tuple[int, str, str, dict]] = SETUPS) -> None:
    """Validate uniqueness and per-setup override contracts."""
    ids = [setup[0] for setup in setups]
    tags = [setup[1] for setup in setups]
    if len(ids) != len(set(ids)):
        raise ValueError("Setup IDs must be unique")
    if len(tags) != len(set(tags)):
        raise ValueError("Setup tags must be unique")
    for setup in setups:
        validate_setup_definition(setup)


validate_all_setups()


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
