"""
Lightweight distributed helpers for torchrun-based multi-GPU evaluation.

When launched via ``torchrun --nproc_per_node=N``, the standard env vars
RANK, WORLD_SIZE, LOCAL_RANK are set automatically.  If they are absent
(plain ``python`` launch), everything falls back to single-GPU mode
transparently — no code changes needed at call sites.

Two init modes:
  - ``init_distributed()``      — full NCCL process group (for all_reduce, etc.)
  - ``init_distributed_lite()`` — GPU assignment only, NO NCCL (for independent work)

Auto-launch:
  Scripts can call ``maybe_relaunch_with_torchrun(nproc)`` at startup.
  If the current process is NOT already a torchrun child, it will
  automatically find a free port and re-exec via torchrun.  This avoids
  the common ``EADDRINUSE`` error on port 29500 when multiple torchrun
  jobs share the same machine.

Usage:
    rank, world_size, local_rank, device = init_distributed()
    ...
    cleanup_distributed()
"""

import logging
import os
import re
import socket
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def restrict_gpus(gpu_ids: str | None) -> int:
    """
    Restrict CUDA visibility to specific physical GPUs.

    Must be called **before** any CUDA context is created (i.e. before
    ``init_distributed`` / ``init_distributed_lite`` and before any
    ``torch.cuda.*`` calls).

    Args:
        gpu_ids: Comma-separated physical GPU IDs, e.g. ``"0,2,5"``.
                 If *None*, no restriction is applied.

    Returns:
        Number of GPUs that will be visible (after restriction, or total
        available if *gpu_ids* is None).

    Example::

        restrict_gpus("0,2,5")          # only GPUs 0, 2, 5 visible
        restrict_gpus(None)              # no-op, use all GPUs
    """
    if gpu_ids is None:
        return torch.cuda.device_count()

    # Validate format: comma-separated non-negative integers
    ids = [s.strip() for s in gpu_ids.split(",") if s.strip()]
    for tok in ids:
        if not tok.isdigit():
            raise ValueError(
                f"--gpus: expected comma-separated integers, got {gpu_ids!r}"
            )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    # After setting the env var, PyTorch will see len(ids) devices
    # numbered 0 .. len(ids)-1.
    return len(ids)


def find_free_port(start: int = 29500, end: int = 29999) -> int:
    """
    Find an available TCP port for torchrun rendezvous.

    Scans from *start* upward until a free port is found (or raises
    RuntimeError if the entire range is exhausted).

    Returns:
        An available port number.
    """
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No free port found in range [{start}, {end}]. "
        f"Free some ports or adjust the range."
    )


def maybe_relaunch_with_torchrun(nproc: int | None, extra_env: dict | None = None):
    """
    Auto-launch the current script under torchrun with a free port.

    Call this at the very beginning of ``if __name__ == "__main__":``,
    **before** any CUDA or distributed initialisation.  If the current
    process is already a torchrun child (``RANK`` env var exists) or
    *nproc* is None/0/1, this function returns immediately and the
    script continues normally in single-process mode.

    Otherwise it finds a free port, constructs the torchrun command, and
    replaces the current process via ``os.execvp``.  The script will
    restart as torchrun children with ``RANK``, ``WORLD_SIZE``, etc.
    set automatically.

    Args:
        nproc:     Number of worker processes (GPUs).  If None or ≤ 1,
                   no relaunch happens.
        extra_env: Optional dict of environment variables to set before
                   re-exec (e.g. ``{"CUDA_VISIBLE_DEVICES": "4,5,6,7"}``).

    Example in a script::

        if __name__ == "__main__":
            args = parse_args()          # must include --nproc
            restrict_gpus(args.gpus)     # set CUDA_VISIBLE_DEVICES first
            maybe_relaunch_with_torchrun(args.nproc)
            main(args)                   # reached only by torchrun children
    """
    # Already under torchrun → nothing to do
    if "RANK" in os.environ:
        return

    if nproc is None or nproc <= 1:
        return

    port = find_free_port()

    # Preserve CUDA_VISIBLE_DEVICES if it was already set by restrict_gpus()
    if extra_env:
        os.environ.update(extra_env)

    # Strip --nproc <N> from the forwarded script args so it doesn't clash
    # with torchrun's own --nproc-per-node / --nproc_per_node option.
    script_argv = []
    it = iter(sys.argv[1:])
    for tok in it:
        if tok == "--nproc":
            next(it, None)   # consume the value
        elif tok.startswith("--nproc="):
            pass             # drop --nproc=N
        else:
            script_argv.append(tok)

    # Pre-set OMP_NUM_THREADS to suppress the noisy banner from
    # torch.distributed.run ("Setting OMP_NUM_THREADS ... for each process")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master-port={port}",
        sys.argv[0],         # the script path
    ] + script_argv

    print(f"[auto-launch] torchrun --nproc_per_node={nproc} "
          f"--master-port={port} {sys.argv[0]} {' '.join(script_argv)}")

    os.execvp(sys.executable, cmd)
    # execvp replaces the process — never returns


def init_distributed(timeout_minutes: int = 120):
    """
    Initialise torch.distributed from torchrun env vars.

    Args:
        timeout_minutes: NCCL timeout in minutes.  Default is 120 (2 hours)
            to accommodate long PPL evaluations where cumulative per-window
            timing differences across ranks can exceed PyTorch's default
            10-minute timeout.

    Returns:
        (rank, world_size, local_rank, device)

    If not launched via torchrun, returns (0, 1, 0, "cuda:0") or
    (0, 1, 0, "cpu") and leaves distributed uninitialised.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        n_visible = torch.cuda.device_count()
        if local_rank >= n_visible:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {n_visible} GPU(s) visible. "
                f"If using --gpus, ensure --nproc_per_node matches the number "
                f"of GPU IDs provided."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=timeout_minutes),
        )
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, local_rank, device

    # Fallback: single-process launch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return 0, 1, 0, device


def init_distributed_lite():
    """
    Read torchrun env vars for GPU assignment but do NOT create an NCCL
    process group.  Use this when ranks work completely independently
    (no all_reduce / barrier needed) to avoid NCCL timeout issues.

    Returns:
        (rank, world_size, local_rank, device)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        n_visible = torch.cuda.device_count()
        if local_rank >= n_visible:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {n_visible} GPU(s) visible. "
                f"If using --gpus, ensure --nproc_per_node matches the number "
                f"of GPU IDs provided."
            )
        torch.cuda.set_device(local_rank)
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


def file_barrier(rank: int, world_size: int, marker_dir: Path,
                 poll_interval: float = 5.0):
    """
    Filesystem-based barrier that doesn't require NCCL.

    Each rank writes a ``.done_rank_{rank}`` marker file.  Rank 0 polls
    until all markers exist, then cleans them up.  Other ranks return
    immediately after writing their marker — torchrun keeps the process
    alive until rank 0 exits.

    Safe for arbitrarily long waits (no 10-min NCCL timeout).
    """
    if world_size <= 1:
        return

    marker_dir = Path(marker_dir)
    marker = marker_dir / f".done_rank_{rank}"
    marker.write_text(str(time.time()))

    if rank == 0:
        # Wait for all ranks to signal completion
        expected = {marker_dir / f".done_rank_{r}" for r in range(world_size)}
        while True:
            if all(m.exists() for m in expected):
                break
            time.sleep(poll_interval)
        # Clean up marker files
        for m in expected:
            try:
                m.unlink()
            except FileNotFoundError:
                pass


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


def suppress_warnings(rank: int = 0) -> None:
    """
    Suppress noisy warnings from PyTorch distributed, HuggingFace, and
    tokenizers that clutter multi-GPU terminal output.

    Call this **after** ``init_distributed()`` / ``init_distributed_lite()``
    so that *rank* is known.

    On **all ranks**:
      - PyTorch c10d socket hostname warnings
      - NCCL device-guessing warnings
      - ``barrier(): using the device under current context`` warning
      - ``torch_dtype is deprecated`` deprecation warning
      - ``tie_word_embeddings`` / tied-weights warnings
      - ``Token indices sequence length is longer than`` warnings
      - Tokenizer parallelism fork-safety warning

    On **non-main ranks** (rank > 0), additionally:
      - HuggingFace transformers logging set to ERROR (suppresses model-
        loading progress bars and info messages)
      - Python logging for ``transformers`` set to ERROR
    """
    # ── Tokenizer parallelism (env var must be set before tokenizer import) ──
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ── Python warnings module filters ──
    # PyTorch distributed / c10d
    warnings.filterwarnings("ignore", message=r".*hostname of the client socket cannot be retrieved.*")
    warnings.filterwarnings("ignore", message=r".*Guessing device ID based on global rank.*")
    warnings.filterwarnings("ignore", message=r".*barrier\(\): using the device under current context.*")
    # Transformers
    warnings.filterwarnings("ignore", message=r".*torch_dtype.*is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*tie_word_embeddings.*")
    warnings.filterwarnings("ignore", message=r".*tied weights mapping.*")
    warnings.filterwarnings("ignore", message=r".*Token indices sequence length is longer.*")
    # HF Hub
    warnings.filterwarnings("ignore", message=r".*unauthenticated requests.*")

    # ── C++ level warnings (torch c10d logger writes to stderr directly) ──
    # These are emitted via PyTorch's internal C++ logger, not Python warnings.
    # We suppress them by raising the torch C++ log level.
    os.environ.setdefault("TORCH_CPP_LOG", "ERROR")
    # Specifically silence the NCCL c10d logger
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    # ── Non-main ranks: silence HuggingFace verbosity ──
    if rank > 0:
        try:
            import transformers.utils.logging as hf_logging
            hf_logging.set_verbosity_error()
        except ImportError:
            pass
        # Also suppress Python-level logging from transformers
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
