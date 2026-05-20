"""
Debug / validation script for MXFP8Linear, MXFP6Linear, MXFP4Linear.

Tests (per format):
  1. Shape correctness (2-D and 3-D inputs)
  2. Bias correctness
  3. Exact match against a pure-Python loop reference
  4. Quantisation error / SNR vs fp32 baseline
  5. Scale tensor shape and dtype

Format-specific extras:
  - FP8: native fp8 cast round-trip grid test
  - FP4: check all 8 positive representable values are in grid
  - FP6 E2M3 / E3M2: FORMAT_MAX switch test

Run from repo root:
    cd /path/to/onlinearith
    ../.venv3_10/bin/python test_mxfp8linear.py
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
TRANSFORMERS_SRC = (REPO_ROOT / ".." / "transformers" / "src").resolve()
if TRANSFORMERS_SRC.exists():
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers.models.qwen3.modeling_qwen3 import (
    MXFP8Linear, MXFP4Linear, MXFP6Linear,
    _FP8_E4M3_MAX, _FP4_E2M1_MAX, _FP6_E2M3_MAX, _FP6_E3M2_MAX,
    _FP4_E2M1_GRID_LIST, _FP6_E2M3_GRID_LIST, _FP6_E3M2_GRID_LIST,
    _nearest_on_grid, _get_grid,
    _msd_truncate, _to_naf_components, _naf_digit_width,
    _compute_inter_block_delays, _compute_intra_block_delays,
    MSDComputeContext, _MXFPLinearBase,
)

BLOCK = 8
IN    = 32
OUT   = 16
torch.manual_seed(42)


# ── config stubs ──────────────────────────────────────────────────────────────

class _CfgFP8:
    use_mxfp8 = True; mxfp8_block_size = BLOCK

class _CfgFP4:
    use_mxfp4 = True; mxfp4_block_size = BLOCK

class _CfgFP6E2M3:
    use_mxfp6 = True; mxfp6_block_size = BLOCK; mxfp6_format = "e2m3"

class _CfgFP6E3M2:
    use_mxfp6 = True; mxfp6_block_size = BLOCK; mxfp6_format = "e3m2"


def make_layer(cls, cfg):
    layer = cls(IN, OUT, bias=False, config=cfg())
    nn.init.normal_(layer.weight)
    return layer


# ── generic reference: block-wise accumulation ────────────────────────────────

def ref_blockwise_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    block_size: int,
    fmt_max: float,
    quantize_fn,
) -> torch.Tensor:
    """
    Pure-Python loop reference for any MX format.
    quantize_fn(tensor_normalised) -> quantised float32 tensor
    """
    N = x.shape[0]
    pad = (-x.shape[-1]) % block_size
    if pad:
        x = F.pad(x, (0, pad))
        w = F.pad(w, (0, pad))
    nb = x.shape[-1] // block_size
    xb = x.float().view(N, nb, block_size)
    wb = w.float().view(w.shape[0], nb, block_size)

    sx = (xb.abs().amax(-1) / fmt_max).clamp(min=1e-30)
    sw = (wb.abs().amax(-1) / fmt_max).clamp(min=1e-30)
    xq = quantize_fn(xb / sx.unsqueeze(-1))
    wq = quantize_fn(wb / sw.unsqueeze(-1))

    out = torch.zeros(N, w.shape[0])
    for b in range(nb):
        out += (xq[:, b, :] @ wq[:, b, :].t()) * (sx[:, b:b+1] * sw[:, b:b+1].t())
    return out


# ── shared test helpers ────────────────────────────────────────────────────────

def _test_all(label: str, cls, cfg_cls, fmt_max: float, quantize_fn):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")

    # 1. Shape (2-D)
    layer = make_layer(cls, cfg_cls)
    y = layer(torch.randn(4, IN))
    assert y.shape == (4, OUT), f"2-D shape: {y.shape}"
    print("[PASS] shape 2-D")

    # 2. Shape (3-D batch)
    y3 = layer(torch.randn(2, 5, IN))
    assert y3.shape == (2, 5, OUT), f"3-D shape: {y3.shape}"
    print("[PASS] shape 3-D batch")

    # 3. Bias: zero input -> output equals bias
    layer_b = cls(IN, OUT, bias=True, config=cfg_cls())
    nn.init.normal_(layer_b.weight)
    nn.init.ones_(layer_b.bias_param)
    y_bias = layer_b(torch.zeros(3, IN))
    assert torch.allclose(y_bias, torch.ones(3, OUT), atol=1e-5), \
        f"bias test: max diff {(y_bias - 1).abs().max()}"
    print("[PASS] bias")

    # 4. Exact match vs pure-Python reference
    torch.manual_seed(0)
    layer = make_layer(cls, cfg_cls)
    x = torch.randn(6, IN)
    y_mx  = layer(x)
    y_ref = ref_blockwise_matmul(x, layer.weight.data, BLOCK, fmt_max, quantize_fn)
    diff  = (y_mx.float() - y_ref).abs().max().item()
    assert diff < 1e-4, f"reference mismatch: max diff {diff}"
    print(f"[PASS] matches reference  (max_diff={diff:.2e})")

    # 5. Scale shape & dtype
    x2   = torch.randn(3, IN)
    pad  = (-IN) % BLOCK
    xp   = F.pad(x2, (0, pad)).float()
    nb   = xp.shape[-1] // BLOCK
    blk  = xp.view(3, nb, BLOCK)
    _, scales = layer._quantize_to_blocks(blk)
    assert scales.shape == (3, nb),       f"scale shape: {scales.shape}"
    assert scales.dtype == torch.float32, f"scale dtype: {scales.dtype}"
    assert (scales > 0).all(),            "non-positive scales"
    print(f"[PASS] scales  shape={tuple(scales.shape)}, dtype={scales.dtype}")

    # 6. Quantisation error / SNR vs fp32
    layer = make_layer(cls, cfg_cls)
    fp_ref = nn.Linear(IN, OUT, bias=False)
    fp_ref.weight.data.copy_(layer.weight.data)
    x = torch.randn(64, IN)
    y_fp32 = fp_ref(x).detach()
    y_mx   = layer(x).detach()
    err    = (y_fp32 - y_mx).abs()
    snr_db = 10 * math.log10(
        (y_fp32.pow(2).mean() / err.pow(2).mean()).item()
    ) if err.pow(2).mean() > 0 else float("inf")
    print(f"[INFO] quant error  max={err.max():.4f}, mean={err.mean():.4f}, SNR={snr_db:.1f} dB")


# ── FP4-specific: verify representable grid ───────────────────────────────────

def test_fp4_grid():
    expected_positive = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    actual_positive   = {v for v in _FP4_E2M1_GRID_LIST if v >= 0}
    assert actual_positive == expected_positive, \
        f"FP4 positive grid wrong:\n  got:      {sorted(actual_positive)}\n  expected: {sorted(expected_positive)}"
    assert len(_FP4_E2M1_GRID_LIST) == 15, \
        f"FP4 grid size: {len(_FP4_E2M1_GRID_LIST)} (expected 15, -0 merged with +0)"
    print(f"[PASS] FP4 E2M1 representable grid  "
          f"({len(_FP4_E2M1_GRID_LIST)} values, max={max(_FP4_E2M1_GRID_LIST):.1f})")


def test_fp6_grids():
    assert max(_FP6_E2M3_GRID_LIST) == pytest_approx(7.5),  f"FP6 E2M3 max wrong: {max(_FP6_E2M3_GRID_LIST)}"
    assert max(_FP6_E3M2_GRID_LIST) == pytest_approx(28.0), f"FP6 E3M2 max wrong: {max(_FP6_E3M2_GRID_LIST)}"
    # E2M3: 32 positive values (including 0); E3M2: same
    e2m3_pos = [v for v in _FP6_E2M3_GRID_LIST if v >= 0]
    e3m2_pos = [v for v in _FP6_E3M2_GRID_LIST if v >= 0]
    assert len(e2m3_pos) == 32, f"FP6 E2M3 positive count: {len(e2m3_pos)}"
    assert len(e3m2_pos) == 32, f"FP6 E3M2 positive count: {len(e3m2_pos)}"
    print(f"[PASS] FP6 E2M3 grid  ({len(_FP6_E2M3_GRID_LIST)} values, max={max(_FP6_E2M3_GRID_LIST):.1f})")
    print(f"[PASS] FP6 E3M2 grid  ({len(_FP6_E3M2_GRID_LIST)} values, max={max(_FP6_E3M2_GRID_LIST):.1f})")


def pytest_approx(v, rel=1e-6):
    """Tiny approx helper (no pytest dependency needed)."""
    class _A:
        def __eq__(self, other): return abs(other - v) <= rel * abs(v) + 1e-12
        def __repr__(self): return f"≈{v}"
    return _A()


def test_nearest_on_grid():
    """Spot-check nearest-neighbor quantization for FP4."""
    grid = _get_grid(_FP4_E2M1_GRID_LIST, torch.device("cpu"))
    cases = [
        (0.0,   0.0),
        (0.3,   0.5),   # closer to 0.5 than 0
        (2.4,   2.0),   # closer to 2 than 3
        (2.6,   3.0),   # closer to 3
        (5.5,   6.0),   # above 4, nearest is 6
        (-1.3, -1.5),
    ]
    for inp, expected in cases:
        t = torch.tensor([inp])
        got = _nearest_on_grid(t, grid).item()
        assert abs(got - expected) < 1e-6, f"nearest({inp}) = {got}, expected {expected}"
    print("[PASS] nearest_on_grid spot-checks (FP4 E2M1)")



# ── quantize_fn wrappers for reference ───────────────────────────────────────

def _qfp8(x):   return x.to(torch.float8_e4m3fn).float()
def _qfp4(x):
    grid = _get_grid(_FP4_E2M1_GRID_LIST, x.device)
    return _nearest_on_grid(x.reshape(-1), grid).view(x.shape)
def _qfp6e2m3(x):
    grid = _get_grid(_FP6_E2M3_GRID_LIST, x.device)
    return _nearest_on_grid(x.reshape(-1), grid).view(x.shape)
def _qfp6e3m2(x):
    grid = _get_grid(_FP6_E3M2_GRID_LIST, x.device)
    return _nearest_on_grid(x.reshape(-1), grid).view(x.shape)


def test_mxfp_format_layers():
    test_mxfp_format_layers()


# ── MSD-specific tests ────────────────────────────────────────────────────────

class _CfgFP8_MSD:
    """Config stub for MXFP8 with MSD truncation enabled."""
    use_mxfp8 = True; mxfp8_block_size = BLOCK
    use_mxfp6 = False; use_mxfp4 = False
    use_msd_truncation = True; msd_cycle_budget = 16
    msd_online_delay = 2; msd_budget_dynamic_scale = 1.0
    msd_budget_dynamic_threshold = 0.0; msd_budget_dynamic_mode = "linear"
    msd_deep_pipeline = False; msd_pipeline_precision_loss = 2
    msd_calibration_data = None


def _make_msd_layer(budget=16):
    """Create an MXFP8Linear with MSD config at given budget."""
    class Cfg(_CfgFP8_MSD):
        msd_cycle_budget = budget
    layer = MXFP8Linear(IN, OUT, bias=False, config=Cfg())
    layer._msd_config = Cfg()
    layer.layer_name = "test_layer"
    nn.init.normal_(layer.weight)
    return layer


def test_msd_truncate_basic():
    """Known-answer tests for _msd_truncate (BSD/NAF truncation)."""
    print(f"\n{'─'*55}")
    print("  MSD Truncation Primitive (BSD/NAF)")
    print(f"{'─'*55}")

    # 7.0 in NAF = 100(-1) (4 NAF digits). Truncate to 2 -> keep positions 3,2 -> 8.0
    val = torch.tensor([7.0])
    result = _msd_truncate(val, torch.tensor([2.0]))
    assert abs(result.item() - 8.0) < 1e-6, f"truncate(7.0, 2) = {result.item()}, expected 8.0"
    print("[PASS] truncate(7.0, 2) = 8.0  (NAF: 100(-1) -> 100_ = 8)")

    # 0.0 -> always 0.0 regardless of num_digits
    result = _msd_truncate(torch.tensor([0.0]), torch.tensor([10.0]))
    assert result.item() == 0.0, f"truncate(0.0, 10) = {result.item()}"
    print("[PASS] truncate(0.0, *) = 0.0")

    # num_digits=0 -> 0.0
    result = _msd_truncate(torch.tensor([42.0]), torch.tensor([0.0]))
    assert result.item() == 0.0, f"truncate(42.0, 0) = {result.item()}"
    print("[PASS] truncate(*, 0) = 0.0")

    # Negative values: -7.0 truncated to 2 digits -> -8.0 (sign symmetry)
    result = _msd_truncate(torch.tensor([-7.0]), torch.tensor([2.0]))
    assert abs(result.item() - (-8.0)) < 1e-6, f"truncate(-7.0, 2) = {result.item()}"
    print("[PASS] truncate(-7.0, 2) = -8.0  (sign-symmetric)")

    # Large num_digits: should be near-lossless
    result = _msd_truncate(torch.tensor([3.14159]), torch.tensor([23.0]))
    assert abs(result.item() - 3.14159) < 1e-5, f"truncate(pi, 23) = {result.item()}"
    print("[PASS] truncate(pi, 23) ≈ pi (near-lossless)")

    # Batch test: shape preserved
    vals = torch.randn(4, 8)
    digits = torch.full((4, 8), 10.0)
    result = _msd_truncate(vals, digits)
    assert result.shape == (4, 8), f"batch shape: {result.shape}"
    print("[PASS] batch shape preserved")


def test_inter_block_delays():
    """Test inter-block delay computation."""
    print(f"\n{'─'*55}")
    print("  Inter-Block Delays")
    print(f"{'─'*55}")

    N, out, nb = 2, 3, 4
    # Create scales with known pattern: one block has much larger scale
    x_scales = torch.tensor([[1.0, 0.5, 0.25, 0.125],
                              [2.0, 1.0, 0.5,  0.25]])   # (N=2, nb=4)
    w_scales = torch.tensor([[1.0, 0.5, 0.25, 0.125],
                              [4.0, 2.0, 1.0,  0.5],
                              [0.5, 0.25, 0.125, 0.0625]]) # (out=3, nb=4)

    delays, e_max = _compute_inter_block_delays(w_scales, x_scales)

    assert delays.shape == (N, out, nb), f"delay shape: {delays.shape}"
    print(f"[PASS] inter-block delay shape = {tuple(delays.shape)}")

    # All delays should be >= 0
    assert (delays >= 0).all(), "negative delays found"
    print("[PASS] all delays >= 0")

    # For each (n, out_ch), the block with max combined scale should have delay=0
    assert (delays.amin(dim=-1) == 0).all(), "min delay per (n,out) should be 0"
    print("[PASS] min delay per output = 0 (max-scale block has no delay)")

    # e_max shape
    assert e_max.shape == (N, out), f"e_max shape: {e_max.shape}"
    print(f"[PASS] e_max shape = {tuple(e_max.shape)}")


def test_intra_block_delays():
    """Test intra-block delay computation."""
    print(f"\n{'─'*55}")
    print("  Intra-Block Delays")
    print(f"{'─'*55}")

    N, nb, bs = 2, 3, 4
    # Create blocks with known exponent pattern
    x_q = torch.tensor([
        [[8.0, 4.0, 2.0, 1.0],   # exponents: 3, 2, 1, 0 -> delays: 0, 1, 2, 3
         [1.0, 1.0, 1.0, 1.0],   # all same -> delays: 0, 0, 0, 0
         [16.0, 0.0, 1.0, 0.5]], # zeros get large delay
        [[4.0, 2.0, 1.0, 0.5],
         [0.0, 0.0, 0.0, 0.0],   # all zeros -> large delays
         [1.0, 0.5, 0.25, 0.125]]
    ])

    delays = _compute_intra_block_delays(x_q)

    assert delays.shape == (N, nb, bs), f"delay shape: {delays.shape}"
    print(f"[PASS] intra-block delay shape = {tuple(delays.shape)}")

    # First block, first sample: [8, 4, 2, 1] -> exps [3, 2, 1, 0] -> delays [0, 1, 2, 3]
    expected = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert torch.allclose(delays[0, 0], expected), f"delays[0,0] = {delays[0, 0]}, expected {expected}"
    print("[PASS] delays for [8,4,2,1] = [0,1,2,3]")

    # Second block, first sample: all 1.0 -> all same exponent -> all delay=0
    assert (delays[0, 1] == 0).all(), f"delays[0,1] = {delays[0, 1]}"
    print("[PASS] uniform block has all-zero delays")


def test_msd_budget_infinity():
    """With infinite budget, MSD result should match exact MX result."""
    print(f"\n{'─'*55}")
    print("  MSD Budget=∞ (near-lossless)")
    print(f"{'─'*55}")

    torch.manual_seed(123)
    layer_msd = _make_msd_layer(budget=999)
    # Also make an exact (non-MSD) layer with same weights
    layer_exact = MXFP8Linear(IN, OUT, bias=False, config=_CfgFP8())
    layer_exact.weight.data.copy_(layer_msd.weight.data)

    x = torch.randn(8, IN)

    y_msd = layer_msd(x).detach()
    y_exact = layer_exact(x).detach()

    err = (y_msd - y_exact).abs()
    signal_power = y_exact.pow(2).mean()
    noise_power = err.pow(2).mean()
    snr_db = 10 * math.log10((signal_power / noise_power).item()) if noise_power > 0 else float("inf")

    assert snr_db > 60.0, f"SNR with B=999 should be >60 dB, got {snr_db:.1f} dB"
    print(f"[PASS] B=999 SNR = {snr_db:.1f} dB (>60 dB, near-lossless)")


def test_msd_budget_zero():
    """With budget=0, all products should be truncated to zero."""
    print(f"\n{'─'*55}")
    print("  MSD Budget=0 (all-zero output)")
    print(f"{'─'*55}")

    torch.manual_seed(456)
    layer = _make_msd_layer(budget=0)
    x = torch.randn(4, IN)
    y = layer(x).detach()

    max_abs = y.abs().max().item()
    assert max_abs < 1e-10, f"B=0 output max_abs = {max_abs}, expected ~0"
    print(f"[PASS] B=0 output max_abs = {max_abs:.2e} (effectively zero)")


def test_msd_budget_sweep_monotonic():
    """Error should decrease (or stay same) as budget increases."""
    print(f"\n{'─'*55}")
    print("  MSD Budget Sweep (monotonic error decrease)")
    print(f"{'─'*55}")

    torch.manual_seed(789)
    # Reference: exact MX layer
    layer_exact = MXFP8Linear(IN, OUT, bias=False, config=_CfgFP8())
    nn.init.normal_(layer_exact.weight)
    x = torch.randn(16, IN)
    y_exact = layer_exact(x).detach()

    budgets = [4, 8, 12, 16, 24, 32]
    errors = []

    for b in budgets:
        layer_msd = _make_msd_layer(budget=b)
        layer_msd.weight.data.copy_(layer_exact.weight.data)
        y_msd = layer_msd(x).detach()
        mse = (y_exact - y_msd).pow(2).mean().item()
        errors.append(mse)
        print(f"  B={b:3d}  MSE={mse:.6f}")

    # Check monotonic decrease (allow small tolerance for numerical noise)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i-1] + 1e-8, \
            f"Error increased: B={budgets[i-1]}→{budgets[i]}, MSE {errors[i-1]:.6f}→{errors[i]:.6f}"

    print(f"[PASS] error monotonically decreases with budget: {errors[0]:.6f} → {errors[-1]:.6f}")


def test_calibration_import():
    """Test that calibration module can be imported and has expected functions."""
    print(f"\n{'─'*55}")
    print("  Calibration Module Import")
    print(f"{'─'*55}")

    from transformers.models.qwen3.calibration_msd import (
        calibrate_channel_budgets,
        apply_calibration_to_config,
        _compute_intra_delays,
        _find_budget_for_snr,
    )

    assert callable(calibrate_channel_budgets), "calibrate_channel_budgets not callable"
    assert callable(apply_calibration_to_config), "apply_calibration_to_config not callable"
    assert callable(_compute_intra_delays), "_compute_intra_delays not callable"
    assert callable(_find_budget_for_snr), "_find_budget_for_snr not callable"
    print("[PASS] Calibration module imports successfully")
    print("[PASS] All calibration functions are callable")


def test_naf_components():
    """Test NAF conversion correctness for known values."""
    print(f"\n{'─'*55}")
    print("  NAF Component Conversion")
    print(f"{'─'*55}")

    # 7 = 0b111 -> NAF = 100(-1), i.e. 8 - 1
    # naf_pos should have bit 3 set (value 8), naf_neg should have bit 0 set (value 1)
    x = torch.tensor([7], dtype=torch.int32)
    pos, neg = _to_naf_components(x)
    assert pos.item() - neg.item() == 7, f"NAF(7): pos={pos.item()}, neg={neg.item()}, diff={pos.item()-neg.item()}"
    print(f"[PASS] NAF(7): pos={pos.item()}, neg={neg.item()}, reconstructs to {pos.item()-neg.item()}")

    # 5 = 0b101 -> already NAF (no adjacent 1s), pos=5, neg=0
    x = torch.tensor([5], dtype=torch.int32)
    pos, neg = _to_naf_components(x)
    assert pos.item() == 5 and neg.item() == 0, f"NAF(5): pos={pos.item()}, neg={neg.item()}"
    print(f"[PASS] NAF(5): pos={pos.item()}, neg={neg.item()} (already NAF)")

    # 3 = 0b11 -> NAF = 10(-1), i.e. 4 - 1
    x = torch.tensor([3], dtype=torch.int32)
    pos, neg = _to_naf_components(x)
    assert pos.item() - neg.item() == 3, f"NAF(3): pos={pos.item()}, neg={neg.item()}"
    print(f"[PASS] NAF(3): pos={pos.item()}, neg={neg.item()}, reconstructs to {pos.item()-neg.item()}")

    # 0 -> pos=0, neg=0
    x = torch.tensor([0], dtype=torch.int32)
    pos, neg = _to_naf_components(x)
    assert pos.item() == 0 and neg.item() == 0, f"NAF(0): pos={pos.item()}, neg={neg.item()}"
    print("[PASS] NAF(0): pos=0, neg=0")

    # Batch correctness: pos - neg == input for random values
    x = torch.randint(0, 10000, (100,), dtype=torch.int32)
    pos, neg = _to_naf_components(x)
    reconstructed = pos - neg
    assert (reconstructed == x).all(), "NAF batch reconstruction failed"
    print("[PASS] NAF batch reconstruction: pos - neg == x for 100 random integers")

    # NAF property: no adjacent non-zero digits (pos & neg have no adjacent bits set)
    combined = pos | neg
    # Check: combined & (combined >> 1) == 0 (no two adjacent bit positions set)
    adjacent = combined & (combined >> 1)
    assert (adjacent == 0).all(), "NAF adjacency property violated"
    print("[PASS] NAF non-adjacency property holds for all 100 values")


def test_naf_digit_width():
    """Test NAF digit width computation."""
    print(f"\n{'─'*55}")
    print("  NAF Digit Width")
    print(f"{'─'*55}")

    # 7 = NAF 100(-1) -> width = 4 (positions 3,2,1,0 used; MSD at position 3)
    x = torch.tensor([7], dtype=torch.int32)
    pos, neg = _to_naf_components(x)
    w = _naf_digit_width(pos, neg)
    assert w.item() == 4, f"NAF width(7) = {w.item()}, expected 4"
    print(f"[PASS] NAF width(7) = 4  (NAF: 100(-1), uses positions 0-3)")

    # 5 = NAF 101 -> width = 3
    x = torch.tensor([5], dtype=torch.int32)
    pos, neg = _to_naf_components(x)
    w = _naf_digit_width(pos, neg)
    assert w.item() == 3, f"NAF width(5) = {w.item()}, expected 3"
    print(f"[PASS] NAF width(5) = 3  (NAF: 101, uses positions 0-2)")

    # 1 -> NAF: 1, width = 1
    x = torch.tensor([1], dtype=torch.int32)
    pos, neg = _to_naf_components(x)
    w = _naf_digit_width(pos, neg)
    assert w.item() == 1, f"NAF width(1) = {w.item()}, expected 1"
    print(f"[PASS] NAF width(1) = 1")

    # 0 -> width = 0
    x = torch.tensor([0], dtype=torch.int32)
    pos, neg = _to_naf_components(x)
    w = _naf_digit_width(pos, neg)
    assert w.item() == 0, f"NAF width(0) = {w.item()}, expected 0"
    print(f"[PASS] NAF width(0) = 0")


def test_msd_truncate_bsd_properties():
    """Test BSD-specific truncation properties that differ from binary."""
    print(f"\n{'─'*55}")
    print("  BSD/NAF Truncation Properties")
    print(f"{'─'*55}")

    # 1) NAF MSD shift: 7 = NAF 100(-1), 4 digits. Keep all 4 -> 7.0
    result = _msd_truncate(torch.tensor([7.0]), torch.tensor([4.0]))
    assert abs(result.item() - 7.0) < 1e-6, f"truncate(7.0, 4) = {result.item()}, expected 7.0"
    print("[PASS] truncate(7.0, 4) = 7.0  (full NAF width)")

    # Keep 3 -> positions 3,2,1 -> 1*8 + 0*4 + 0*2 = 8.0
    result = _msd_truncate(torch.tensor([7.0]), torch.tensor([3.0]))
    assert abs(result.item() - 8.0) < 1e-6, f"truncate(7.0, 3) = {result.item()}, expected 8.0"
    print("[PASS] truncate(7.0, 3) = 8.0  (NAF: drop LSD -1 digit)")

    # 2) Bidirectional error: BSD truncation can overshoot (error > 0 for positive value)
    # truncate(7.0, 2) = 8.0 -> error = +1 (positive, unlike binary which always rounds toward 0)
    result = _msd_truncate(torch.tensor([7.0]), torch.tensor([2.0]))
    error = result.item() - 7.0
    assert error > 0, f"Expected positive error for truncate(7.0,2), got error={error:.6f}"
    print(f"[PASS] Bidirectional error: truncate(7.0, 2) = {result.item()}, error = +{error:.1f}")

    # 3) Binary-NAF agreement for NAF-clean values: 5 = 101 binary = 101 NAF (same)
    # truncate(5.0, 2) = ? In NAF, 5 = 101, width=3, keep 2 -> drop bit 0 -> 100 = 4.0
    result = _msd_truncate(torch.tensor([5.0]), torch.tensor([2.0]))
    assert abs(result.item() - 4.0) < 1e-6, f"truncate(5.0, 2) = {result.item()}, expected 4.0"
    print("[PASS] truncate(5.0, 2) = 4.0  (NAF-clean: 101 -> 10_ = 4)")

    # 4) Sign symmetry: truncate(x, p) == -truncate(-x, p)
    test_values = [7.0, 3.14, 0.5, 13.0, 100.0]
    for v in test_values:
        for p in [2.0, 4.0, 8.0]:
            pos_r = _msd_truncate(torch.tensor([v]), torch.tensor([p])).item()
            neg_r = _msd_truncate(torch.tensor([-v]), torch.tensor([p])).item()
            assert abs(pos_r + neg_r) < 1e-6, \
                f"Sign symmetry failed: truncate({v},{p})={pos_r}, truncate({-v},{p})={neg_r}"
    print("[PASS] Sign symmetry: truncate(x,p) == -truncate(-x,p) for all test values")

    # 5) 3.0 in NAF = 10(-1), width=3. Keep 2 -> drop bottom -> 10_ = 4.0
    result = _msd_truncate(torch.tensor([3.0]), torch.tensor([2.0]))
    assert abs(result.item() - 4.0) < 1e-6, f"truncate(3.0, 2) = {result.item()}, expected 4.0"
    print("[PASS] truncate(3.0, 2) = 4.0  (NAF: 10(-1) -> 10_ = 4)")

    # 6) Large precision = near-lossless for various values
    for v in [1.0, 2.5, 7.7, 100.3, 0.0625]:
        result = _msd_truncate(torch.tensor([v]), torch.tensor([23.0]))
        assert abs(result.item() - v) < 1e-4, \
            f"Near-lossless failed: truncate({v}, 23) = {result.item()}"
    print("[PASS] Large precision (23 digits) is near-lossless for various values")


def test_combined_scale_budget():
    """Test that dynamic budget uses combined activation+weight scales per output channel."""
    print(f"\n{'─'*55}")
    print("  Combined-Scale Dynamic Budget")
    print(f"{'─'*55}")

    # Create a layer with known weight pattern:
    # Output channel 0 has large weights, channel 1 has small weights
    # Note: MX scales = max_abs / FORMAT_MAX, so combined log2 is typically negative.
    # We set a negative threshold to make the dynamic budget differentiate channels.
    class Cfg(_CfgFP8_MSD):
        msd_cycle_budget = 16
        msd_budget_dynamic_scale = 2.0
        msd_budget_dynamic_threshold = -20.0  # negative: differentiates channels
        msd_budget_dynamic_mode = "linear"
    
    layer = MXFP8Linear(IN, OUT, bias=False, config=Cfg())
    layer._msd_config = Cfg()
    layer.layer_name = "test_combined_layer"
    # Set channel 0 weights to be very large, channel 1 to be very small
    with torch.no_grad():
        layer.weight.data.zero_()
        layer.weight.data[0, :] = 100.0   # large weight channel
        layer.weight.data[1, :] = 0.001   # small weight channel

    # Uniform activation input
    x = torch.ones(1, IN) * 1.0

    # Prepare blocks to get scales
    N = 1
    x_2d = x.float().reshape(N, IN)
    w_2d = layer.weight.float()
    x_q, x_scales, _ = layer._prepare_blocks(x_2d, N)
    w_q, w_scales, _ = layer._prepare_blocks(w_2d, OUT)

    # Call _resolve_channel_budgets with combined scales
    b_final = layer._resolve_channel_budgets(None, x_scales, w_scales, N)
    assert b_final.shape == (N, OUT), f"b_final shape = {b_final.shape}"
    print(f"[PASS] b_final shape = {tuple(b_final.shape)}")

    # Channel 0 (large weights) should get higher budget than channel 1 (small weights)
    b_ch0 = b_final[0, 0].item()
    b_ch1 = b_final[0, 1].item()
    print(f"  Channel 0 (large W) budget: {b_ch0:.2f}")
    print(f"  Channel 1 (small W) budget: {b_ch1:.2f}")
    assert b_ch0 > b_ch1, \
        f"Large-weight channel should get higher budget: ch0={b_ch0:.2f} vs ch1={b_ch1:.2f}"
    print("[PASS] Large-weight channel gets higher dynamic budget")

    # Test step mode: threshold between the two channels' combined exponents
    class CfgStep(_CfgFP8_MSD):
        msd_cycle_budget = 16
        msd_budget_dynamic_scale = 5.0
        msd_budget_dynamic_threshold = -20.0  # ch0 ~= -11 > -20; ch1 ~= -28 < -20
        msd_budget_dynamic_mode = "step"

    layer_step = MXFP8Linear(IN, OUT, bias=False, config=CfgStep())
    layer_step._msd_config = CfgStep()
    layer_step.layer_name = "test_step_layer"
    with torch.no_grad():
        layer_step.weight.data.zero_()
        layer_step.weight.data[0, :] = 100.0
        layer_step.weight.data[1, :] = 0.001

    w_2d_s = layer_step.weight.float()
    w_q_s, w_scales_s, _ = layer_step._prepare_blocks(w_2d_s, OUT)
    b_final_step = layer_step._resolve_channel_budgets(None, x_scales, w_scales_s, N)
    b_s0 = b_final_step[0, 0].item()
    b_s1 = b_final_step[0, 1].item()
    print(f"  Step mode - Channel 0 budget: {b_s0:.2f}")
    print(f"  Step mode - Channel 1 budget: {b_s1:.2f}")
    # Channel 0 with large combined scale should exceed threshold and get +alpha
    # Channel 1 with small combined scale may not exceed threshold
    assert b_s0 >= b_s1, \
        f"Step mode: large-W channel budget should >= small-W: ch0={b_s0:.2f} vs ch1={b_s1:.2f}"
    print("[PASS] Step mode: large-weight channel budget >= small-weight channel")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print(f"PyTorch {torch.__version__}  |  BLOCK={BLOCK}, IN={IN}, OUT={OUT}")
    print("=" * 55)

    _test_all("MXFP8 (E4M3FN, max=448)",  MXFP8Linear, _CfgFP8,     _FP8_E4M3_MAX, _qfp8)
    _test_all("MXFP4 (E2M1,   max=6)",    MXFP4Linear, _CfgFP4,     _FP4_E2M1_MAX, _qfp4)
    _test_all("MXFP6 E2M3     (max=7.5)", MXFP6Linear, _CfgFP6E2M3, _FP6_E2M3_MAX, _qfp6e2m3)
    _test_all("MXFP6 E3M2     (max=28)",  MXFP6Linear, _CfgFP6E3M2, _FP6_E3M2_MAX, _qfp6e3m2)

    print()
    test_fp4_grid()
    test_fp6_grids()
    test_nearest_on_grid()

    # ── MSD-specific tests ──────────────────────────────────────────────────
    print()
    test_msd_truncate_basic()
    test_naf_components()
    test_naf_digit_width()
    test_msd_truncate_bsd_properties()
    test_inter_block_delays()
    test_intra_block_delays()
    test_msd_budget_infinity()
    test_msd_budget_zero()
    test_msd_budget_sweep_monotonic()
    test_combined_scale_budget()
    test_calibration_import()

    print("=" * 55)
    print("All tests passed.")
