"""
Lightweight distributed helpers for torchrun-based multi-GPU evaluation.

When launched via ``torchrun --nproc_per_node=N``, the standard env vars
RANK, WORLD_SIZE, LOCAL_RANK are set automatically.  If they are absent
(plain ``python`` launch), everything falls back to single-GPU mode
transparently — no code changes needed at call sites.

Usage:
    rank, world_size, local_rank, device = init_distributed()
    ...
    cleanup_distributed()
"""

import os

import torch
import torch.distributed as dist


def init_distributed():
    """
    Initialise torch.distributed from torchrun env vars.

    Returns:
        (rank, world_size, local_rank, device)

    If not launched via torchrun, returns (0, 1, 0, "cuda:0") or
    (0, 1, 0, "cpu") and leaves distributed uninitialised.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, local_rank, device

    # Fallback: single-process launch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return 0, 1, 0, device


def cleanup_distributed():
    """Destroy the process group if it was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int = 0) -> bool:
    """True on rank 0 (or always True in single-process mode)."""
    return rank == 0


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place sum-reduce across all ranks.  No-op if distributed is not active.
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def barrier():
    """Synchronisation barrier. No-op if distributed is not active."""
    if dist.is_initialized():
        dist.barrier()


def gather_list(local_list: list, rank: int, world_size: int) -> list:
    """
    Gather Python lists from all ranks onto rank 0.

    Returns the concatenated list on rank 0, empty list on other ranks.
    Uses dist.all_gather_object when distributed is active.
    """
    if not dist.is_initialized() or world_size == 1:
        return local_list

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_list)
    if rank == 0:
        merged = []
        for g in gathered:
            merged.extend(g)
        return merged
    return []
