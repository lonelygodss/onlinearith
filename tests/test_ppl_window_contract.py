from __future__ import annotations

import importlib.util
import math
from pathlib import Path

MAX_LENGTH = 4096
STRIDE = 512
ROOT = Path(__file__).resolve().parents[1]


def load_ppl_utils():
    path = ROOT / "ppl_utils.py"
    spec = importlib.util.spec_from_file_location("_ppl_utils_under_test", path)
    assert spec is not None and spec.loader is not None, "ppl_utils.py must exist after the PPL helper extraction"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_precompute_windows_matches_current_ppltest_contract():
    ppl_utils = load_ppl_utils()
    for seq_len in [1, 2, 511, 512, 513, 4095, 4096, 4097, 8192, 8193]:
        assert list(ppl_utils.precompute_windows(seq_len, MAX_LENGTH, STRIDE)) == baseline_precompute_windows(seq_len)


def test_precompute_windows_scores_each_token_once_after_context():
    ppl_utils = load_ppl_utils()
    for seq_len in [16, 513, 4097, 12345]:
        windows = list(ppl_utils.precompute_windows(seq_len, MAX_LENGTH, STRIDE))
        assert windows[0][0] == 0
        assert windows[-1][1] == seq_len
        assert sum(trg_len for _begin, _end, trg_len in windows) == seq_len
        assert all(0 < trg_len <= MAX_LENGTH for _begin, _end, trg_len in windows)


def test_weighted_nll_is_not_window_loss_average():
    ppl_utils = load_ppl_utils()
    losses = [(0.2, 3), (1.0, 1), (0.4, 6)]
    expected_nll = sum(loss * tokens for loss, tokens in losses)
    expected_tokens = sum(tokens for _loss, tokens in losses)
    expected_ppl = math.exp(expected_nll / expected_tokens)

    total_nll = 0.0
    total_tokens = 0
    for loss, tokens in losses:
        total_nll, total_tokens = ppl_utils.accumulate_weighted_nll(total_nll, total_tokens, loss, tokens)

    assert total_nll == expected_nll
    assert total_tokens == expected_tokens
    assert ppl_utils.finalize_ppl(total_nll, total_tokens) == expected_ppl


def _run_direct() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    _run_direct()
