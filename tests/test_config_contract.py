from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RUNTIME_STATS_FIELDS = {
    "msd_perf_stats_enabled",
    "msd_perf_stats_lite",
    "msd_figure5_layer_cycles",
}
MX_FORMAT_FLAGS = ("use_mxfp8", "use_mxfp6", "use_mxfp4")
_CFG = None


def load_experiment_config():
    path = ROOT / "experiment_config.py"
    spec = importlib.util.spec_from_file_location("_experiment_config_under_test", path)
    assert spec is not None and spec.loader is not None, f"Could not load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(ROOT))
        except ValueError:
            pass
    return module


def cfg():
    global _CFG
    if _CFG is None:
        _CFG = load_experiment_config()
    return _CFG


def test_registry_and_baseline_have_same_keys():
    cfg = cfg_module()
    assert set(cfg.MXFP_MSD_FIELDS) == set(cfg.BASELINE_CONFIG)


def test_runtime_stats_fields_are_registered():
    cfg = cfg_module()
    missing = EXPECTED_RUNTIME_STATS_FIELDS - set(cfg.MXFP_MSD_FIELDS)
    assert not missing, f"register runtime stats fields in experiment_config.py: {sorted(missing)}"


def test_setup_ids_and_tags_are_stable_unique():
    cfg = cfg_module()
    ids = [item[0] for item in cfg.SETUPS]
    tags = [item[1] for item in cfg.SETUPS]
    assert len(ids) == len(set(ids))
    assert len(tags) == len(set(tags))
    # Preserve existing public setup IDs unless a separate methodology change approves it.
    assert ids == list(range(1, 22))


def test_setup_override_keys_are_known():
    cfg = cfg_module()
    baseline_keys = set(cfg.BASELINE_CONFIG)
    unknown = []
    for setup_id, tag, _desc, overrides in cfg.SETUPS:
        unknown.extend((setup_id, tag, key) for key in set(overrides) - baseline_keys)
    assert not unknown


def test_mx_format_flags_are_mutually_exclusive():
    cfg = cfg_module()
    conflicts = []
    for setup_id, tag, _desc, overrides in cfg.SETUPS:
        enabled = [flag for flag in MX_FORMAT_FLAGS if bool(overrides.get(flag, cfg.BASELINE_CONFIG.get(flag, False)))]
        if len(enabled) > 1:
            conflicts.append((setup_id, tag, enabled))
    assert not conflicts


def test_mxfp6_format_values_are_explicit():
    cfg = cfg_module()
    bad = []
    for setup_id, tag, _desc, overrides in cfg.SETUPS:
        value = overrides.get("mxfp6_format", cfg.BASELINE_CONFIG.get("mxfp6_format"))
        if value not in {"e2m3", "e3m2"}:
            bad.append((setup_id, tag, value))
    assert not bad


def cfg_module():
    return cfg()


def _run_direct() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    _run_direct()
