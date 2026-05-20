#!/usr/bin/env python3
"""
Contract tests for exact output-chunked MX-only matmul.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import MethodType

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_SRC = (REPO_ROOT / ".." / "transformers" / "src").resolve()
if TRANSFORMERS_SRC.exists() and str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers.models.qwen3.modeling_qwen3 import MXFP4Linear, MXFP6Linear, MXFP8Linear


class Cfg:
    use_msd_truncation = False
    use_activation_nm_sparsity = False
    mxfp8_block_size = 32
    mxfp6_block_size = 32
    mxfp6_format = "e2m3"
    mxfp4_block_size = 32
    mxfp_use_chunked_exact = True
    mxfp_chunk_target_mib = 1
    mxfp_weight_cache_dtype = "float16"


def old_exact_mx(layer, x_q, x_scales, w_q, w_scales):
    elem_dots = torch.bmm(
        x_q.permute(1, 0, 2).contiguous(),
        w_q.permute(1, 2, 0).contiguous(),
    )
    combined_scales = x_scales.t().unsqueeze(-1) * w_scales.t().unsqueeze(1)
    return (elem_dots * combined_scales).sum(dim=0)


def prepare(layer, x):
    batch_shape = x.shape[:-1]
    N = math.prod(batch_shape) if batch_shape else 1
    x_2d = x.float().reshape(N, layer.in_features)
    x_q, x_scales, _ = layer._prepare_blocks(x_2d, N)
    w_q, w_scales, _ = layer._prepare_blocks(layer.weight.float(), layer.out_features)
    return N, x_q, x_scales, w_q, w_scales


def make_layer(layer_cls, cfg, in_features=96, out_features=80, device="cpu"):
    torch.manual_seed(1234)
    layer = layer_cls(in_features, out_features, bias=False, config=cfg).to(device)
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.15)
    layer.eval()
    return layer


def run_direct_equivalence(layer_cls, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = make_layer(layer_cls, cfg, device=device)
    x = torch.randn(2, 17, layer.in_features, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    N, x_q, x_scales, w_q, w_scales = prepare(layer, x)
    assert hasattr(layer, "_forward_mx_exact_chunked"), "missing _forward_mx_exact_chunked on MXFP layer"
    expected = old_exact_mx(layer, x_q, x_scales, w_q, w_scales)
    got = layer._forward_mx_exact_chunked(x_q, x_scales, w_q, w_scales, N)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_mxfp8_direct_chunked_equivalence():
    run_direct_equivalence(MXFP8Linear, Cfg())


def test_mxfp4_direct_chunked_equivalence():
    run_direct_equivalence(MXFP4Linear, Cfg())


def test_mxfp6_direct_chunked_equivalence():
    cfg = Cfg()
    cfg.mxfp6_format = "e2m3"
    run_direct_equivalence(MXFP6Linear, cfg)


def test_forward_uses_chunked_path_for_mx_only():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Cfg()
    layer = make_layer(MXFP8Linear, cfg, device=device)
    assert hasattr(layer, "_forward_mx_exact_chunked"), "missing _forward_mx_exact_chunked on MXFP layer"

    called = {"n": 0}
    original = layer._forward_mx_exact_chunked

    def wrapped(self, *args, **kwargs):
        called["n"] += 1
        return original(*args, **kwargs)

    layer._forward_mx_exact_chunked = MethodType(wrapped, layer)
    x = torch.randn(1, 33, layer.in_features, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    with torch.inference_mode():
        _ = layer(x)
    assert called["n"] == 1, "MX-only forward() did not route through _forward_mx_exact_chunked"


if __name__ == "__main__":
    test_mxfp8_direct_chunked_equivalence()
    test_mxfp4_direct_chunked_equivalence()
    test_mxfp6_direct_chunked_equivalence()
    test_forward_uses_chunked_path_for_mx_only()
    print("ok")
