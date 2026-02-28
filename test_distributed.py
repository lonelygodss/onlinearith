"""
Verification tests for the multi-GPU distributed PPL evaluation infrastructure.

Tests cover:
  1. dist_utils single-GPU fallback (no torchrun env vars)
  2. Window precomputation correctness (trg_len sums to seq_len)
  3. Window sharding correctness (all tokens covered, no overlap)
  4. NCCL all_reduce (requires torchrun, 2+ GPUs)
  5. gather_list across ranks (requires torchrun, 2+ GPUs)
  6. End-to-end PPL window sharding matches sequential baseline

Run modes:
  # Single-process tests only (no GPU needed for most tests):
  python test_distributed.py

  # Full tests including NCCL (needs 2+ GPUs):
  torchrun --nproc_per_node=2 test_distributed.py

  # All 8 GPUs:
  torchrun --nproc_per_node=8 test_distributed.py
"""

import math
import os
import sys

import torch

# Ensure dist_utils is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dist_utils import (
    all_reduce_sum,
    barrier,
    cleanup_distributed,
    gather_list,
    init_distributed,
    is_main,
)
from ppltest import precompute_windows


# ── Test helpers ─────────────────────────────────────────────────────────────

_pass_count = 0
_fail_count = 0
_skip_count = 0


def _report(name, passed, msg=""):
    global _pass_count, _fail_count
    if passed:
        _pass_count += 1
        status = "PASS"
    else:
        _fail_count += 1
        status = "FAIL"
    suffix = f"  ({msg})" if msg else ""
    print(f"  [{status}] {name}{suffix}")


def _skip(name, reason):
    global _skip_count
    _skip_count += 1
    print(f"  [SKIP] {name}  ({reason})")


# ── Non-distributed tests (pure Python, no GPU needed) ──────────────────────

def test_precompute_windows_sum():
    """trg_len across all windows sums exactly to seq_len."""
    for seq_len in [1000, 10000, 287654, 300000]:
        windows = precompute_windows(seq_len, max_length=4096, stride=512)
        total_trg = sum(w[2] for w in windows)
        ok = (total_trg == seq_len)
        _report(f"windows_sum(seq_len={seq_len})",
                ok,
                f"total_trg={total_trg}, expected={seq_len}")


def test_precompute_windows_coverage():
    """Every token position in [0, seq_len) is scored exactly once across all windows."""
    seq_len = 50000
    max_length = 4096
    stride = 512
    windows = precompute_windows(seq_len, max_length, stride)

    scored = [0] * seq_len
    for begin_loc, end_loc, trg_len in windows:
        for pos in range(end_loc - trg_len, end_loc):
            scored[pos] += 1

    all_covered = all(s == 1 for s in scored)
    _report("windows_coverage",
            all_covered,
            f"{seq_len} tokens, {len(windows)} windows, "
            f"min_score={min(scored)}, max_score={max(scored)}")


def test_window_sharding_complete(world_size_sim=8):
    """Round-robin sharding across N ranks covers all windows with no overlap."""
    seq_len = 300000
    all_windows = precompute_windows(seq_len, max_length=4096, stride=512)
    total_windows = len(all_windows)

    seen_indices = set()
    for rank in range(world_size_sim):
        shard = all_windows[rank::world_size_sim]
        for w in shard:
            idx = all_windows.index(w)
            seen_indices.add(idx)

    ok = (len(seen_indices) == total_windows)
    _report(f"window_sharding(world_size={world_size_sim})",
            ok,
            f"covered={len(seen_indices)}/{total_windows}")


def test_window_sharding_nll_sum(world_size_sim=8):
    """Simulated sharded NLL accumulation matches sequential result."""
    seq_len = 300000
    all_windows = precompute_windows(seq_len, max_length=4096, stride=512)

    gt_nll_sum = 0.0
    gt_tokens = 0
    for begin_loc, end_loc, trg_len in all_windows:
        fake_nll = math.log(1.0 + trg_len / 1000.0)
        gt_nll_sum += fake_nll * trg_len
        gt_tokens += trg_len

    total_nll_sum = 0.0
    total_tokens = 0
    for rank in range(world_size_sim):
        shard = all_windows[rank::world_size_sim]
        for begin_loc, end_loc, trg_len in shard:
            fake_nll = math.log(1.0 + trg_len / 1000.0)
            total_nll_sum += fake_nll * trg_len
            total_tokens += trg_len

    ok_sum = abs(total_nll_sum - gt_nll_sum) < 1e-6
    ok_tok = (total_tokens == gt_tokens)
    _report(f"sharded_nll_sum(world_size={world_size_sim})",
            ok_sum and ok_tok,
            f"nll_diff={abs(total_nll_sum - gt_nll_sum):.2e}, "
            f"tokens={total_tokens}=={gt_tokens}")


# ── Distributed tests (require torchrun, init process group once) ────────────

def run_distributed_tests(rank, world_size, device):
    """Run all NCCL-based tests. Process group is already initialised."""

    # ── Test: all_reduce ──
    val = torch.tensor([float(rank + 1), float(rank * 10)],
                       dtype=torch.float64, device=device)
    all_reduce_sum(val)

    expected_0 = world_size * (world_size + 1) / 2
    expected_1 = 10.0 * sum(range(world_size))

    ok = (abs(val[0].item() - expected_0) < 1e-6 and
          abs(val[1].item() - expected_1) < 1e-6)

    if is_main(rank):
        _report("nccl_all_reduce",
                ok,
                f"val={val.tolist()}, expected=[{expected_0}, {expected_1}], "
                f"world_size={world_size}")
    barrier()

    # ── Test: gather_list ──
    local_data = [rank * 100 + i for i in range(3)]
    gathered = gather_list(local_data, rank, world_size)

    if is_main(rank):
        expected_len = 3 * world_size
        expected_set = set()
        for r in range(world_size):
            for i in range(3):
                expected_set.add(r * 100 + i)
        all_present = expected_set == set(gathered)
        _report("gather_list",
                len(gathered) == expected_len and all_present,
                f"gathered_len={len(gathered)}, expected={expected_len}, "
                f"all_present={all_present}")
    barrier()

    # ── Test: simulated PPL aggregate ──
    seq_len = 300000
    all_windows = precompute_windows(seq_len, max_length=4096, stride=512)
    my_windows = all_windows[rank::world_size]

    local_nll_sum = 0.0
    local_tokens = 0
    for begin_loc, end_loc, trg_len in my_windows:
        fake_nll = math.log(1.0 + trg_len / 1000.0)
        local_nll_sum += fake_nll * trg_len
        local_tokens += trg_len

    agg = torch.tensor([local_nll_sum, float(local_tokens)],
                       dtype=torch.float64, device=device)
    all_reduce_sum(agg)
    total_nll_sum = agg[0].item()
    total_tokens = int(agg[1].item())

    gt_nll_sum = 0.0
    gt_tokens = 0
    for begin_loc, end_loc, trg_len in all_windows:
        fake_nll = math.log(1.0 + trg_len / 1000.0)
        gt_nll_sum += fake_nll * trg_len
        gt_tokens += trg_len

    if is_main(rank):
        ok_sum = abs(total_nll_sum - gt_nll_sum) < 1e-4
        ok_tok = (total_tokens == gt_tokens)
        _report("nccl_ppl_aggregate",
                ok_sum and ok_tok,
                f"nll_diff={abs(total_nll_sum - gt_nll_sum):.2e}, "
                f"tokens={total_tokens}=={gt_tokens}, world_size={world_size}")
    barrier()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    distributed_mode = "RANK" in os.environ

    # Initialise distributed (or fallback to single-GPU)
    rank, world_size, local_rank, device = init_distributed()

    if is_main(rank):
        print("=" * 60)
        print("  Distributed PPL Infrastructure Verification Tests")
        print("=" * 60)
        if distributed_mode:
            print(f"  Mode: torchrun (world_size={world_size})")
        else:
            print(f"  Mode: single-process (use torchrun for full NCCL tests)")
        print()

    # ── Non-distributed tests (rank 0 only) ──
    if is_main(rank):
        print("--- Single-process tests ---")
        test_precompute_windows_sum()
        test_precompute_windows_coverage()
        test_window_sharding_complete(world_size_sim=8)
        test_window_sharding_complete(world_size_sim=3)
        test_window_sharding_nll_sum(world_size_sim=8)
        test_window_sharding_nll_sum(world_size_sim=3)
        print()

    # All ranks must reach the barrier before distributed tests
    barrier()

    # ── Distributed tests ──
    if is_main(rank):
        print("--- NCCL distributed tests ---")

    if distributed_mode:
        run_distributed_tests(rank, world_size, device)
    else:
        if is_main(rank):
            _skip("nccl_all_reduce", "not running under torchrun")
            _skip("gather_list", "not running under torchrun")
            _skip("nccl_ppl_aggregate", "not running under torchrun")

    # ── Summary (rank 0) ──
    if is_main(rank):
        print()
        print("-" * 60)
        total = _pass_count + _fail_count + _skip_count
        print(f"  Results: {_pass_count} passed, {_fail_count} failed, "
              f"{_skip_count} skipped (total {total})")
        if _fail_count > 0:
            print("  *** SOME TESTS FAILED ***")
        else:
            print("  All tests passed!")
        print("-" * 60)

    cleanup_distributed()

    if _fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
