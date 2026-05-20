#!/usr/bin/env python3
"""
Contract test for the optimized MSD truncation kernel.

This pins _msd_truncate to the previous log2/pow reference formula while still
allowing the implementation to use lower-allocation scale reconstruction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_SRC = (REPO_ROOT / ".." / "transformers" / "src").resolve()
if TRANSFORMERS_SRC.exists() and str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers.models.qwen3.modeling_qwen3 import _msd_truncate


def _reference_msd_truncate_old(value: torch.Tensor, num_digits: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        abs_v = value.abs()
        sign = value.sign()
        mask = (num_digits > 0) & (abs_v > 0)

        msb_pos = torch.floor(torch.log2(abs_v.clamp(min=1e-45)))
        scale_up = torch.pow(2.0, 23.0 - msb_pos)
        x_scaled = torch.round(abs_v * scale_up).to(torch.int32)

        x_h = x_scaled >> 1
        s = x_scaled + x_h
        naf_pos = s & (~x_h)
        naf_neg = x_h & (~s)

        combined = naf_pos | naf_neg
        comb_f = combined.float()
        naf_width = torch.where(
            combined > 0,
            torch.floor(torch.log2(comb_f)).int() + 1,
            torch.zeros_like(combined),
        )

        num_digits_i = num_digits.to(torch.int32) if num_digits.is_floating_point() else num_digits
        drop = (naf_width - num_digits_i).clamp(min=0)
        keep_mask = ~((1 << drop) - 1)

        reconstructed = (naf_pos & keep_mask).float() - (naf_neg & keep_mask).float()
        result = sign * (reconstructed / scale_up)
        return torch.where(mask, result, torch.zeros_like(result))


def test_msd_truncate_matches_log2_pow_reference():
    values = torch.tensor(
        [
            0.0,
            1.0,
            -1.0,
            3.0,
            -7.0,
            31.75,
            -127.5,
            1.0 / 1024.0,
            -3.14159,
            65504.0,
            -0.03125,
            2.0**-20,
        ],
        dtype=torch.float32,
    )
    digits = torch.tensor(
        [
            -2.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            8.0,
            12.0,
            23.0,
            24.0,
            30.0,
        ],
        dtype=torch.float32,
    )

    grid_values = values[:, None].expand(-1, digits.numel()).reshape(-1)
    grid_digits = digits[None, :].expand(values.numel(), -1).reshape(-1)

    actual = _msd_truncate(grid_values, grid_digits)
    expected = _reference_msd_truncate_old(grid_values, grid_digits)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_msd_truncate_matches_log2_pow_reference()
    print("ok")
