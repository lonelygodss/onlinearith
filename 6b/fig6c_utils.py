#!/usr/bin/env python3
"""Utilities for Figure 6(c) fixed-horizon budget construction."""

from __future__ import annotations

import numpy as np
import torch


def _validate_budget_inputs(target_sum: int, n_channels: int, budget_range: tuple[int, int]) -> tuple[int, int]:
    bmin, bmax = budget_range
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    if bmin > bmax:
        raise ValueError(f"invalid budget_range: {budget_range}")

    min_sum = n_channels * bmin
    max_sum = n_channels * bmax
    if target_sum < min_sum or target_sum > max_sum:
        raise ValueError(
            f"target_sum={target_sum} is infeasible for n_channels={n_channels}, "
            f"budget_range={budget_range} (feasible: [{min_sum}, {max_sum}])"
        )
    return bmin, bmax


def _distribute_units_with_caps(
    weights: np.ndarray,
    capacities: np.ndarray,
    units: int,
) -> np.ndarray:
    """Distribute integer units proportionally to nonnegative weights under per-index caps."""
    if units < 0:
        raise ValueError("units must be nonnegative")

    out = np.zeros_like(capacities, dtype=np.int32)
    caps = capacities.astype(np.int32).copy()
    w = np.clip(weights.astype(np.float64), 0.0, None)

    remaining = int(units)
    while remaining > 0:
        eligible = caps > 0
        if not np.any(eligible):
            raise ValueError("cannot distribute all units due to exhausted capacities")

        w_eff = w.copy()
        w_eff[~eligible] = 0.0
        if float(w_eff.sum()) <= 0.0:
            w_eff = eligible.astype(np.float64)

        raw = (w_eff / float(w_eff.sum())) * remaining
        step = np.floor(raw).astype(np.int32)
        step = np.minimum(step, caps)
        give = int(step.sum())

        if give == 0:
            frac = raw - np.floor(raw)
            priority = frac + (w_eff / (float(np.max(w_eff)) + 1e-12)) * 1e-9
            order = np.argsort(-priority)
            progressed = False
            for idx in order:
                if remaining == 0:
                    break
                if caps[idx] <= 0:
                    continue
                out[idx] += 1
                caps[idx] -= 1
                remaining -= 1
                progressed = True
            if not progressed:
                raise ValueError("failed to make progress while distributing units")
            continue

        out += step
        caps -= step
        remaining -= give

    return out


def repair_integer_budget_sum(
    budgets: np.ndarray,
    target_sum: int,
    budget_range: tuple[int, int],
    scores: np.ndarray | None = None,
) -> np.ndarray:
    """Repair budgets to match target_sum exactly under [bmin, bmax] constraints."""
    bmin, bmax = budget_range
    b = np.rint(np.asarray(budgets, dtype=np.float64)).astype(np.int32)
    b = np.clip(b, bmin, bmax)

    if scores is None:
        s = np.ones_like(b, dtype=np.float64)
    else:
        s = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
        if s.shape != b.shape:
            raise ValueError("scores shape must match budgets shape")

    diff = int(target_sum - int(b.sum()))
    if diff == 0:
        return b

    while diff != 0:
        if diff > 0:
            eligible = np.where(b < bmax)[0]
            if eligible.size == 0:
                raise ValueError("cannot increase budgets to match target_sum")
            order = eligible[np.argsort(-s[eligible])]
            progressed = False
            for idx in order:
                if diff == 0:
                    break
                if b[idx] < bmax:
                    b[idx] += 1
                    diff -= 1
                    progressed = True
            if not progressed:
                raise ValueError("repair loop made no progress while increasing budgets")
        else:
            eligible = np.where(b > bmin)[0]
            if eligible.size == 0:
                raise ValueError("cannot decrease budgets to match target_sum")
            order = eligible[np.argsort(s[eligible])]
            progressed = False
            for idx in order:
                if diff == 0:
                    break
                if b[idx] > bmin:
                    b[idx] -= 1
                    diff += 1
                    progressed = True
            if not progressed:
                raise ValueError("repair loop made no progress while decreasing budgets")

    return b


def allocate_integer_budget_from_scores(
    scores: np.ndarray,
    target_sum: int,
    budget_range: tuple[int, int] = (4, 48),
) -> np.ndarray:
    """
    Allocate integer budgets proportionally to nonnegative channel scores.

    Returns an int32 vector with exact sum(target) and all entries in budget_range.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = s.size
    bmin, bmax = _validate_budget_inputs(int(target_sum), n, budget_range)

    s = np.clip(s, 0.0, None)
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if float(s.sum()) <= 0.0:
        s = np.ones_like(s, dtype=np.float64)

    base = np.full(n, bmin, dtype=np.int32)
    remaining = int(target_sum - int(base.sum()))
    if remaining == 0:
        return base

    capacities = np.full(n, bmax - bmin, dtype=np.int32)
    extra = _distribute_units_with_caps(s, capacities, remaining)
    out = base + extra
    out = repair_integer_budget_sum(out, int(target_sum), budget_range=budget_range, scores=s)
    return out


def make_uniform_budgets(
    target_sum: int,
    n_channels: int,
    budget_range: tuple[int, int] = (4, 48),
) -> np.ndarray:
    """Even budget allocation with exact integer repair under bounds."""
    bmin, bmax = _validate_budget_inputs(int(target_sum), int(n_channels), budget_range)

    base = int(target_sum) // int(n_channels)
    rem = int(target_sum) % int(n_channels)
    budgets = np.full(int(n_channels), base, dtype=np.int32)
    if rem > 0:
        budgets[:rem] += 1
    budgets = np.clip(budgets, bmin, bmax)
    budgets = repair_integer_budget_sum(
        budgets,
        int(target_sum),
        budget_range=budget_range,
        scores=np.ones(int(n_channels), dtype=np.float64),
    )
    return budgets


def make_weight_only_scores_from_quantized_weights(
    w_q: torch.Tensor,
    w_scales: torch.Tensor,
) -> np.ndarray:
    """
    Compute weight-only per-output-channel scores from quantized block tensors.

    Score definition:
        s_c = sum_{b,k} (w_q[c,b,k] * w_scale[c,b])^2
    """
    if w_q.ndim != 3:
        raise ValueError(f"w_q must be 3D (out, nb, bs), got shape={tuple(w_q.shape)}")
    if w_scales.ndim != 2:
        raise ValueError(f"w_scales must be 2D (out, nb), got shape={tuple(w_scales.shape)}")

    w_rec = w_q * w_scales.unsqueeze(-1)
    scores = w_rec.square().sum(dim=(1, 2))
    return scores.detach().float().cpu().numpy().astype(np.float64)
