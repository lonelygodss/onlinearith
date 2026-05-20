from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_SRC = (ROOT.parent / "transformers" / "src").resolve()
PUBLIC_SYMBOLS = [
    "_MXFPLinearBase",
    "MXFP8Linear",
    "MXFP6Linear",
    "MXFP4Linear",
    "_make_linear",
    "MSDComputeContext",
]


def test_qwen3_custom_symbols_remain_importable_from_modeling_qwen3():
    if not TRANSFORMERS_SRC.exists():
        print(f"[SKIP] sibling transformers checkout not found at {TRANSFORMERS_SRC}")
        return
    sys.path.insert(0, str(TRANSFORMERS_SRC))
    try:
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    finally:
        try:
            sys.path.remove(str(TRANSFORMERS_SRC))
        except ValueError:
            pass
    missing = [name for name in PUBLIC_SYMBOLS if not hasattr(modeling_qwen3, name)]
    assert not missing, f"keep compatibility re-exports in modeling_qwen3.py: {missing}"


def _run_direct() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()


if __name__ == "__main__":
    _run_direct()
