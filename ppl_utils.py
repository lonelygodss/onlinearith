"""Pure helpers for WikiText sliding-window PPL accounting.

These functions mirror the existing PPL runners without importing datasets,
Transformers, or torch at module import time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


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
