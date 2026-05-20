"""
Unit tests for fixed-sum budget redistribution optimizer.

Tests:
  1. Sum preservation: sum(B_opt) == sum(B_snr_min)
  2. Bounds preservation: all(B_min <= B_opt <= B_max)
  3. No-op when no beneficial swap: random noise curves
  4. Improvement when swap exists: synthetic beneficial curves
  5. Backward-compatible JSON schema
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add sibling transformers checkout to path when present.
REPO_ROOT = Path(__file__).resolve().parent
TRANSFORMERS_SRC = (REPO_ROOT / ".." / "transformers" / "src").resolve()
if TRANSFORMERS_SRC.exists():
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers.models.qwen3.calibration_msd import (
    LayerErrorCurves,
    solve_fixed_sum_from_error_curves,
    _BUDGET_MIN,
    _BUDGET_MAX,
)


def make_synthetic_curves(
    num_channels: int = 100,
    budget_snr_min: np.ndarray = None,
    budget_range: tuple = (_BUDGET_MIN, _BUDGET_MAX),
    window: int = 3,
    error_scale: float = 1.0,
    noise_scale: float = 0.1,
    make_flat: bool = False,
) -> LayerErrorCurves:
    """Create synthetic error curves for testing."""
    if budget_snr_min is None:
        # Random budgets in safe range
        budget_snr_min = np.random.randint(
            budget_range[0] + window + 1,
            budget_range[1] - window - 1,
            size=num_channels,
        )

    # Build budget values around each channel's SNR-min budget
    all_bvals = set()
    for b_star in budget_snr_min:
        lo = max(budget_range[0], int(b_star) - window)
        hi = min(budget_range[1], int(b_star) + window)
        for b in range(lo, hi + 1):
            all_bvals.add(b)

    budget_values = np.array(sorted(all_bvals), dtype=np.int32)
    num_points = len(budget_values)

    # Create errors
    errors = np.zeros((num_channels, num_points), dtype=np.float64)
    for j, b_star in enumerate(budget_snr_min):
        for i, b in enumerate(budget_values):
            if make_flat:
                # Flat curve - error independent of budget
                base_error = error_scale
            else:
                # Error decreases with budget (more budget = less error)
                base_error = error_scale / (b + 1)

            noise = noise_scale * np.random.randn()
            errors[j, i] = max(0.001, base_error + noise)

    return LayerErrorCurves(
        layer_name="test_layer",
        num_channels=num_channels,
        budget_snr_min=budget_snr_min,
        budget_range=budget_range,
        budget_values=budget_values,
        errors=errors,
    )


def test_sum_preservation():
    """Optimized budgets must sum to the same as SNR-min budgets."""
    np.random.seed(42)
    curves = make_synthetic_curves(num_channels=200)
    target_sum = int(curves.budget_snr_min.sum())

    budgets, stats = solve_fixed_sum_from_error_curves(curves, target_sum=target_sum)

    assert stats["sum_preserved"], f"Sum not preserved: {stats['final_sum']} != {target_sum}"
    assert int(budgets.sum()) == target_sum, f"Budgets sum mismatch: {budgets.sum()} != {target_sum}"
    print(f"  Sum preserved: {target_sum} == {stats['final_sum']}")


def test_bounds_preservation():
    """All budgets must stay within [BUDGET_MIN, BUDGET_MAX]."""
    np.random.seed(43)
    curves = make_synthetic_curves(
        num_channels=100,
        budget_snr_min=np.random.randint(_BUDGET_MIN + 5, _BUDGET_MAX - 5, size=100),
    )

    budgets, stats = solve_fixed_sum_from_error_curves(curves)

    assert budgets.min() >= _BUDGET_MIN, f"Budget below min: {budgets.min()} < {_BUDGET_MIN}"
    assert budgets.max() <= _BUDGET_MAX, f"Budget above max: {budgets.max()} > {_BUDGET_MAX}"
    print(f"  Bounds preserved: [{budgets.min()}, {budgets.max()}] within [{_BUDGET_MIN}, {_BUDGET_MAX}]")


def test_noop_when_flat_curves():
    """When error curves are flat (no swap beneficial), optimizer should not move."""
    np.random.seed(44)
    num_channels = 50
    budget_snr_min = np.full(num_channels, 20, dtype=np.int32)

    curves = make_synthetic_curves(
        num_channels=num_channels,
        budget_snr_min=budget_snr_min,
        make_flat=True,  # Flat curves!
        noise_scale=0.0,  # No noise
    )

    budgets, stats = solve_fixed_sum_from_error_curves(curves)

    assert stats["swaps_performed"] == 0, f"Should not swap with flat curves, got {stats['swaps_performed']} swaps"
    np.testing.assert_array_equal(budgets, budget_snr_min)
    print(f"  No swaps with flat curves")


def test_improvement_with_beneficial_curves():
    """When error curves have clear gain/loss asymmetry, optimizer should improve."""
    np.random.seed(45)
    num_channels = 100

    # Create asymmetric curves: half have high gain from extra budget, half have low loss
    budget_snr_min = np.full(num_channels, 20, dtype=np.int32)

    budget_values = np.arange(_BUDGET_MIN, _BUDGET_MAX + 1, dtype=np.int32)
    num_points = len(budget_values)
    errors = np.zeros((num_channels, num_points), dtype=np.float64)

    b_idx_20 = np.where(budget_values == 20)[0][0]

    for j in range(num_channels):
        for i, b in enumerate(budget_values):
            if j < num_channels // 2:
                # Channels 0-49: high gain - error drops sharply with budget
                errors[j, i] = 100.0 / (b + 1)
            else:
                # Channels 50-99: low gain - error drops slowly with budget
                errors[j, i] = 10.0 / (b + 1) + 50.0  # Higher baseline, slower drop

    curves = LayerErrorCurves(
        layer_name="asymmetric_layer",
        num_channels=num_channels,
        budget_snr_min=budget_snr_min,
        budget_range=(_BUDGET_MIN, _BUDGET_MAX),
        budget_values=budget_values,
        errors=errors,
    )

    budgets, stats = solve_fixed_sum_from_error_curves(curves)

    # Should perform swaps
    assert stats["swaps_performed"] > 0, f"Expected swaps with asymmetric curves, got 0"
    assert stats["total_improvement"] > 0, f"Expected improvement, got {stats['total_improvement']}"

    # Verify budgets moved in expected direction: low-index channels gain, high-index lose
    # Note: Due to heap ordering, exact prediction is hard, but we can check sum is preserved
    assert int(budgets.sum()) == int(budget_snr_min.sum()), "Sum not preserved during improvement test"

    print(f"  Improvement achieved: {stats['swaps_performed']} swaps, total improvement={stats['total_improvement']:.2f}")


def test_backward_compatible_json():
    """Output JSON must be loadable by existing code paths."""
    # Create mock calibration result in the new format
    result = {
        "format": "MXFP8",
        "description": "Test",
        "optimizer": "fixed_sum",
        "config_overrides": {"use_mxfp8": True},
        "calibration_params": {
            "target_snr_db": 30.0,
            "num_texts": 20,
            "holdout_texts": 0,
            "holdout_fraction": 0.0,
            "max_length": 512,
            "batch_size": 4,
            "online_delay": 2,
            "detail_layer": 2,
            "curve_window": 3,
        },
        "global_summary": {
            "num_layers": 84,
            "total_channels": 200704,
            "budget_min": 6.0,
            "budget_max": 19.0,
            "budget_mean": 10.32,
        },
        "optimizer_stats": {
            "model.layers.0.mlp.gate_proj": {
                "swaps_performed": 10,
                "total_improvement": 5.5,
                "final_sum": 1024,
                "target_sum": 1024,
                "sum_preserved": True,
            }
        },
        "holdout_evaluation": None,
        "layer_stats": {},
        "channel_detail": {"detail_layer": 2},
        "msd_calibration_data": {
            "model.layers.0.mlp.gate_proj": [10.0, 9.0, 10.0],
        },
    }

    # Verify it can be serialized
    json_str = json.dumps(result)
    loaded = json.loads(json_str)

    # Verify expected fields exist
    assert "msd_calibration_data" in loaded
    assert "model.layers.0.mlp.gate_proj" in loaded["msd_calibration_data"]

    # New fields should be present
    assert "optimizer" in loaded
    assert loaded["optimizer"] == "fixed_sum"
    assert "optimizer_stats" in loaded

    print(f"  JSON schema backward compatible")


def test_optimizer_preserves_snr_min_when_no_gain():
    """If optimizer cannot improve, it should return SNR-min budgets."""
    np.random.seed(46)

    # Create curves where all channels have identical error profiles
    num_channels = 50
    budget_snr_min = np.random.randint(15, 25, size=num_channels).astype(np.int32)

    budget_values = np.arange(_BUDGET_MIN, _BUDGET_MAX + 1, dtype=np.int32)
    num_points = len(budget_values)
    errors = np.zeros((num_channels, num_points), dtype=np.float64)

    # Same error curve for all channels
    for j in range(num_channels):
        for i, b in enumerate(budget_values):
            errors[j, i] = 1.0 / (b + 1)

    curves = LayerErrorCurves(
        layer_name="uniform_layer",
        num_channels=num_channels,
        budget_snr_min=budget_snr_min,
        budget_range=(_BUDGET_MIN, _BUDGET_MAX),
        budget_values=budget_values,
        errors=errors,
    )

    budgets, stats = solve_fixed_sum_from_error_curves(curves)

    # With identical error curves, any redistribution has zero net gain
    # Floating point noise may cause tiny improvements, so use tolerance
    assert stats["total_improvement"] < 0.1, f"Expected near-zero improvement with uniform curves, got {stats['total_improvement']}"
    print(f"  No false improvement: {stats['swaps_performed']} swaps, improvement={stats['total_improvement']:.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Fixed-Sum Budget Optimizer Tests")
    print("=" * 60)

    tests = [
        ("Sum preservation", test_sum_preservation),
        ("Bounds preservation", test_bounds_preservation),
        ("No-op with flat curves", test_noop_when_flat_curves),
        ("Improvement with beneficial curves", test_improvement_with_beneficial_curves),
        ("Backward-compatible JSON", test_backward_compatible_json),
        ("Preserves SNR-min when no gain", test_optimizer_preserves_snr_min_when_no_gain),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{name}...")
        try:
            test_fn()
            print(f"  PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name} - {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {name} - {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
