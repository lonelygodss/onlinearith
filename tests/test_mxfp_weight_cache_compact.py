#!/usr/bin/env python3
"""
Contract tests for compact MXFP weight caches.
"""
from __future__ import annotations

import sys
from pathlib import Path

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


def make_pair(layer_cls, cfg, device):
    torch.manual_seed(2026)
    a = layer_cls(128, 96, bias=False, config=cfg).to(device).eval()
    b = layer_cls(128, 96, bias=False, config=cfg).to(device).eval()
    with torch.no_grad():
        a.weight.normal_(0, 0.1)
        b.weight.copy_(a.weight)
    return a, b


def assert_compact_cache(layer):
    assert layer._w_cache is not None, "expected _w_cache after forward"
    w_q = layer._w_cache[0]
    assert w_q.dtype != torch.float32, (
        "compact mxfp_weight_cache_dtype should not keep fp32 w_q persistently; "
        "cast chunks back to fp32 only during computation"
    )


def run_cache_case(layer_cls, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compact, reference = make_pair(layer_cls, cfg, device)
    x = torch.randn(2, 19, 128, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)

    cfg.mxfp_weight_cache_dtype = "float16"
    with torch.inference_mode():
        out_compact = compact(x)
    assert_compact_cache(compact)

    cfg_ref = Cfg()
    cfg_ref.mxfp_weight_cache_dtype = "float32"
    reference._msd_config = cfg_ref
    with torch.inference_mode():
        out_ref = reference(x)

    torch.testing.assert_close(out_compact, out_ref, rtol=0, atol=0)


def test_mxfp8_cache_compact_exact():
    run_cache_case(MXFP8Linear, Cfg())


def test_mxfp8_cache_float8_exact():
    cfg = Cfg()
    cfg.mxfp_weight_cache_dtype = "float8"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compact, reference = make_pair(MXFP8Linear, cfg, device)
    x = torch.randn(2, 19, 128, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)

    with torch.inference_mode():
        out_compact = compact(x)
    assert compact._w_cache is not None, "expected _w_cache after forward"
    assert compact._w_cache[0].dtype == torch.float8_e4m3fn

    cfg_ref = Cfg()
    cfg_ref.mxfp_weight_cache_dtype = "float32"
    reference._msd_config = cfg_ref
    with torch.inference_mode():
        out_ref = reference(x)

    torch.testing.assert_close(out_compact, out_ref, rtol=0, atol=0)


def test_mxfp4_cache_compact_exact():
    run_cache_case(MXFP4Linear, Cfg())


def test_mxfp6_cache_compact_exact():
    cfg = Cfg()
    cfg.mxfp6_format = "e2m3"
    run_cache_case(MXFP6Linear, cfg)


if __name__ == "__main__":
    test_mxfp8_cache_compact_exact()
    test_mxfp8_cache_float8_exact()
    test_mxfp4_cache_compact_exact()
    test_mxfp6_cache_compact_exact()
    print("ok")
