import importlib.util
import json
import tempfile
from pathlib import Path


def _load_merge_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "merge_msd_calibrations.py"
    spec = importlib.util.spec_from_file_location("merge_msd_calibrations", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _calibration(layer_name, budgets, projection_filter):
    return {
        "format": "MXFP8",
        "optimizer": "fixed_sum",
        "config_overrides": {"use_mxfp8": True},
        "calibration_params": {"projection_filter": projection_filter},
        "global_summary": {"wall_time_sec": 2.0},
        "optimizer_stats": {layer_name: {"sum_preserved": True}},
        "layer_stats": {
            layer_name: {
                "snr_mean": 31.0,
                "snr_min": 30.0,
                "e_combined_mean": -19.0,
                "eff_precision_mean": 3.0,
                "signal_power_db_mean": 1.0,
            }
        },
        "channel_detail": {"detail_layer": 2},
        "msd_calibration_data": {layer_name: budgets},
    }


def test_merge_projection_filtered_calibrations(tmp_path):
    merge_mod = _load_merge_module()
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(_calibration("layer.gate_proj", [8, 9], "gate_proj")))
    path_b.write_text(json.dumps(_calibration("layer.up_proj", [10], "up_proj")))

    merged = merge_mod.merge_calibrations([path_a, path_b], allow_overwrite=False)

    assert set(merged["msd_calibration_data"]) == {"layer.gate_proj", "layer.up_proj"}
    assert merged["global_summary"]["num_layers"] == 2
    assert merged["global_summary"]["total_channels"] == 3
    assert merged["global_summary"]["budget_mean"] == 9.0
    assert [m["projection_filter"] for m in merged["merged_from_projection_filters"]] == [
        "gate_proj",
        "up_proj",
    ]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_merge_projection_filtered_calibrations(Path(tmp))
    print("ok")
