"""Pure helpers for WikiText sliding-window PPL accounting.

These functions mirror the existing PPL runners without importing datasets,
Transformers, or torch at module import time.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def gib(x: int | float) -> float:
    """Convert bytes to GiB."""
    return float(x) / 1024**3


def precompute_windows(seq_len: int, max_length: int = 4096, stride: int = 512) -> list[tuple[int, int, int]]:
    """
    Return ``(begin_loc, end_loc, trg_len)`` windows for weighted PPL scoring.

    ``trg_len`` is the number of newly scored tokens in the window. Earlier
    context tokens must be masked out of the labels by the caller.
    """
    windows: list[tuple[int, int, int]] = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        windows.append((begin_loc, end_loc, trg_len))
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return windows


def mask_context_labels(input_ids, trg_len: int, ignore_index: int = -100):
    """Clone token IDs and mask context labels outside the newly scored tail."""
    labels = input_ids.clone()
    labels[:, :-trg_len] = ignore_index
    return labels


def prepare_tail_logits_loss_kwargs(
    labels,
    trg_len: int,
    *,
    loss_token_chunk_size: int | None = None,
    output_logits: bool = True,
) -> dict:
    """
    Return kwargs that compute the same shifted causal loss using only useful logits.

    Hugging Face's causal loss uses logit position ``i`` to predict label
    position ``i + 1``.  For a PPL window with ``trg_len`` newly scored tail
    labels, only logits from the position immediately before that tail through
    the penultimate input position can affect the loss.  Supplying explicit
    ``shift_labels`` keeps the loss equivalent while avoiding a full
    sequence-length vocabulary projection.
    """
    seq_len = labels.shape[-1]
    keep = min(seq_len - 1, trg_len)
    if keep <= 0:
        return {}
    start = seq_len - keep - 1
    logits_to_keep = labels.new_tensor(list(range(start, start + keep)))
    shift_labels = labels[..., start + 1 : start + 1 + keep].contiguous()
    kwargs = {"logits_to_keep": logits_to_keep, "shift_labels": shift_labels}
    if loss_token_chunk_size is not None:
        kwargs["loss_token_chunk_size"] = int(loss_token_chunk_size)
    if not output_logits:
        kwargs["output_logits"] = False
    return kwargs


def accumulate_weighted_nll(
    total_nll: float,
    total_tokens: int,
    loss: float,
    trg_len: int,
) -> tuple[float, int]:
    """Accumulate average window loss weighted by scored-token count."""
    return total_nll + float(loss) * trg_len, total_tokens + trg_len


def finalize_ppl(total_nll: float, total_tokens: int) -> float:
    """Exponentiate the corpus-level mean NLL once."""
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    return math.exp(total_nll / total_tokens)


def iter_shard(items: Sequence[T], rank: int, world_size: int) -> Iterable[T]:
    """Yield the rank-local strided shard used by the current runners."""
    return items[rank::world_size]


class MxfpProgressReporter:
    """Throttled progress hook for long MXFP/MSD output-channel chunk loops."""

    def __init__(
        self,
        interval_sec: float,
        progress_file: str | None = None,
        device=None,
        extra: dict | None = None,
    ):
        self.interval_sec = max(0.0, float(interval_sec))
        self.progress_file = Path(progress_file) if progress_file else None
        self.device = device
        self.extra = dict(extra or {})
        self.t0 = time.perf_counter()
        self.last_emit = 0.0

    def __call__(self, *, module, phase: str, chunk_idx: int, total_chunks: int, **payload) -> None:
        now = time.perf_counter()
        is_edge = total_chunks > 1 and (chunk_idx == 1 or chunk_idx == total_chunks)
        if not is_edge and self.interval_sec > 0 and (now - self.last_emit) < self.interval_sec:
            return
        self.last_emit = now
        event = {
            "event": "mxfp_progress",
            "elapsed_sec": round(now - self.t0, 3),
            "phase": phase,
            "layer_name": getattr(module, "layer_name", None) or "<unregistered>",
            "class": type(module).__name__,
            "chunk_idx": int(chunk_idx),
            "total_chunks": int(total_chunks),
            "percent": round(100.0 * float(chunk_idx) / max(1, int(total_chunks)), 3),
            **self.extra,
            **payload,
        }
        if self.device is not None and getattr(self.device, "type", None) == "cuda":
            import torch

            event.update(
                {
                    "cuda_alloc_gib": gib(torch.cuda.memory_allocated(self.device)),
                    "cuda_reserved_gib": gib(torch.cuda.memory_reserved(self.device)),
                    "cuda_peak_alloc_gib": gib(torch.cuda.max_memory_allocated(self.device)),
                    "cuda_peak_reserved_gib": gib(torch.cuda.max_memory_reserved(self.device)),
                }
            )
        line = json.dumps(event, sort_keys=True)
        print(f"PROGRESS {line}", flush=True)
        if self.progress_file is not None:
            tmp = self.progress_file.with_suffix(self.progress_file.suffix + ".tmp")
            tmp.write_text(line + "\n", encoding="utf-8")
            tmp.replace(self.progress_file)


def register_mxfp_layer_names(model) -> None:
    """Attach stable module names to MXFP layers for progress and probe records."""
    for name, module in model.named_modules():
        if hasattr(module, "_report_mxfp_progress") and not getattr(module, "layer_name", None):
            module.layer_name = name


def install_mxfp_progress_hook(
    model,
    *,
    interval_sec: float | None,
    progress_file: str | None = None,
    device=None,
    extra: dict | None = None,
) -> None:
    """Install a progress hook on Qwen3 config if progress is enabled."""
    register_mxfp_layer_names(model)
    if interval_sec is None or interval_sec < 0:
        clear_mxfp_progress_hook(model)
        return
    model.config._mxfp_progress_hook = MxfpProgressReporter(
        interval_sec,
        progress_file=progress_file,
        device=device,
        extra=extra,
    )


def clear_mxfp_progress_hook(model) -> None:
    """Remove the MXFP progress hook from a model config, if present."""
    if hasattr(model.config, "_mxfp_progress_hook"):
        delattr(model.config, "_mxfp_progress_hook")
