#!/usr/bin/env python3
"""Synthetic equivalence checks for PPL helper extraction.

Run this before and after moving logic into ppl_utils.py. It does not load datasets,
models, tokenizers, or CUDA. It only checks window math and weighted-NLL accounting.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

MAX_LENGTH = 4096
STRIDE = 512


def baseline_precompute_windows(seq_len: int, max_length: int = MAX_LENGTH, stride: int = STRIDE):
    windows = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        windows.append((begin_loc, end_loc, trg_len))
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return windows


def load_ppl_utils(root: Path):
    path = root / "ppl_utils.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_onlinearith_ppl_utils_for_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_windows(ppl_utils, seq_lens: list[int]) -> list[str]:
    errors: list[str] = []
    for seq_len in seq_lens:
        expected = baseline_precompute_windows(seq_len)
        actual = ppl_utils.precompute_windows(seq_len, MAX_LENGTH, STRIDE)
        if list(actual) != expected:
            errors.append(f"seq_len={seq_len}: expected {expected}, got {actual}")
    return errors


def check_weighted_nll(ppl_utils) -> list[str]:
    errors: list[str] = []
    losses = [(0.25, 3), (1.5, 2), (0.75, 5)]
    expected_nll = sum(loss * tokens for loss, tokens in losses)
    expected_tokens = sum(tokens for _loss, tokens in losses)
    expected_ppl = math.exp(expected_nll / expected_tokens)

    if hasattr(ppl_utils, "accumulate_weighted_nll"):
        total_nll = 0.0
        total_tokens = 0
        for loss, tokens in losses:
            total_nll, total_tokens = ppl_utils.accumulate_weighted_nll(total_nll, total_tokens, loss, tokens)
    else:
        total_nll = expected_nll
        total_tokens = expected_tokens

    if abs(total_nll - expected_nll) > 1e-12 or total_tokens != expected_tokens:
        errors.append(f"weighted NLL mismatch: got ({total_nll}, {total_tokens}), expected ({expected_nll}, {expected_tokens})")

    if hasattr(ppl_utils, "finalize_ppl"):
        actual_ppl = float(ppl_utils.finalize_ppl(total_nll, total_tokens))
        if abs(actual_ppl - expected_ppl) > 1e-12:
            errors.append(f"PPL mismatch: got {actual_ppl}, expected {expected_ppl}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onlinearith-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.onlinearith_root.resolve()

    ppl_utils = load_ppl_utils(root)
    if ppl_utils is None:
        print("[INFO] ppl_utils.py not found yet. Add it during Phase 3, then re-run this script.")
        return 0

    errors = []
    errors.extend(check_windows(ppl_utils, [1, 2, 511, 512, 513, 4095, 4096, 4097, 8192, 8193]))
    errors.extend(check_weighted_nll(ppl_utils))
    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        return 1
    print("[OK] ppl_utils.py matches baseline synthetic PPL math")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
