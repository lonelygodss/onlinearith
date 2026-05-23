#!/usr/bin/env python3
"""Contract tests for common N:M keep-count semantics."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRANSFORMERS_SRC = (REPO_ROOT / ".." / "transformers" / "src").resolve()
if TRANSFORMERS_SRC.exists() and str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from experiment_config import nm_keep_to_prune_count, validate_nm_keep_ratio
from transformers.models.qwen3.modeling_qwen3 import MXFP8Linear


class Cfg:
    use_msd_truncation = False
    use_activation_nm_sparsity = True
    activation_nm_n = 1
    activation_nm_m = 4
    mxfp8_block_size = 4
    mxfp_use_chunked_exact = True
    mxfp_chunk_target_mib = 1
    mxfp_weight_cache_dtype = "float16"


def test_common_nm_keep_to_prune_count():
    validate_nm_keep_ratio(1, 4)
    validate_nm_keep_ratio(4, 4)
    assert nm_keep_to_prune_count(1, 4) == 3
    assert nm_keep_to_prune_count(2, 4) == 2
    assert nm_keep_to_prune_count(4, 4) == 0

    try:
        validate_nm_keep_ratio(5, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected n > m to be rejected")


def test_activation_nm_keeps_n_largest_values_per_group():
    cfg = Cfg()
    layer = MXFP8Linear(4, 4, bias=False, config=cfg).eval()
    x_q = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])

    sparse = layer._apply_activation_nm_sparsity(x_q)
    expected = torch.tensor([[[0.0, 0.0, 0.0, -4.0]]])
    torch.testing.assert_close(sparse, expected, rtol=0, atol=0)

    cfg.activation_nm_n = 3
    sparse = layer._apply_activation_nm_sparsity(x_q)
    expected = torch.tensor([[[0.0, -2.0, 3.0, -4.0]]])
    torch.testing.assert_close(sparse, expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_common_nm_keep_to_prune_count()
    test_activation_nm_keeps_n_largest_values_per_group()
    print("ok")
