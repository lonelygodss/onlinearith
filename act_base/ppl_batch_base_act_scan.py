"""
Run activation-only common N:M PPL batches over multiple (n, m) pairs.

Usage:
    cd /path/to/onlinearith
    python act_base/ppl_batch_base_act_scan.py -nm 2:4 1:4 --only 1
    python act_base/ppl_batch_base_act_scan.py -nm 2:4 1:4 --only 1 --gpu 0
    python act_base/ppl_batch_base_act_scan.py --nm "(2,4)" "(1,4)" --force
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ONLINEARITH_ROOT = SCRIPT_DIR.parent
RUNNER = SCRIPT_DIR / "ppl_batch_base_act.py"


def parse_nm(token: str) -> tuple[int, int]:
    """Parse one common N:M token, where N is kept. Separators: ':', ',', '-'."""
    match = re.fullmatch(r"\(?\s*(\d+)\s*[:,-]\s*(\d+)\s*\)?", token.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid N:M pair '{token}'. Use forms like 2:4, 2,4, 2-4, or (2,4)."
        )

    n = int(match.group(1))
    m = int(match.group(2))

    if m <= 0:
        raise argparse.ArgumentTypeError(f"Invalid pair '{token}': m must be > 0.")
    if n < 0:
        raise argparse.ArgumentTypeError(f"Invalid pair '{token}': n must be >= 0.")
    if n > m:
        raise argparse.ArgumentTypeError(f"Invalid pair '{token}': common N:M requires n <= m.")

    return n, m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-GPU sequential sweep of activation-only ppl_batch_base_act over common N:M pairs."
    )
    parser.add_argument(
        "--nm",
        "-nm",
        nargs="+",
        required=True,
        type=parse_nm,
        metavar="N:M",
        help="List of common N:M keep ratios, e.g. --nm 2:4 1:4 (or 2,4 / 2-4 / (2,4)).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        metavar="ID",
        default=[1],
        help="Setup IDs forwarded to runner (default: 1 for MXFP8 only).",
    )
    parser.add_argument("--list", action="store_true", help="Forwarded to runner.")
    parser.add_argument("--force", action="store_true", help="Forwarded to runner.")
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        metavar="N",
        help="Runner process count. This scan mode is single-GPU; only 1 is allowed.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Single physical GPU ID, forwarded as --gpus <id>.",
    )
    parser.add_argument("--gpus", type=str, default=None, help="Single GPU ID string (alternative to --gpu).")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining pairs even if one run fails.",
    )
    args = parser.parse_args()

    if args.nproc != 1:
        raise SystemExit("ERROR: this scan mode is single-GPU only; use --nproc 1.")

    if args.gpus is not None and "," in args.gpus:
        raise SystemExit("ERROR: this scan mode accepts only one GPU ID.")

    selected_gpu = args.gpus
    if selected_gpu is None and args.gpu is not None:
        selected_gpu = str(args.gpu)

    unique_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in args.nm:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    print(f"[scan] Runner: {RUNNER}")
    print(f"[scan] Total unique common N:M pairs: {len(unique_pairs)}")

    summary: list[tuple[int, int, int, float]] = []
    for i, (n, m) in enumerate(unique_pairs, start=1):
        cmd = [
            sys.executable,
            str(RUNNER),
            "-n",
            str(n),
            "-m",
            str(m),
            "--nproc",
            "1",
        ]
        if selected_gpu is not None:
            cmd.extend(["--gpus", selected_gpu])
        if args.force:
            cmd.append("--force")
        if args.list:
            cmd.append("--list")
        cmd.extend(["--only", *[str(sid) for sid in args.only]])

        print()
        print(f"[scan] ({i}/{len(unique_pairs)}) keep {n}:{m} (prune {m - n}:{m})")
        print(f"[scan] Command: {' '.join(cmd)}")

        t0 = time.perf_counter()
        completed = subprocess.run(cmd, cwd=str(ONLINEARITH_ROOT))
        elapsed = time.perf_counter() - t0
        summary.append((n, m, completed.returncode, elapsed))

        if completed.returncode != 0:
            print(f"[scan] FAILED n={n}, m={m} (exit={completed.returncode}, {elapsed:.1f}s)")
            if not args.continue_on_error:
                break
        else:
            print(f"[scan] DONE n={n}, m={m} ({elapsed:.1f}s)")

    print()
    print("[scan] Summary")
    print(f"{'keep':>4}  {'m':>3}  {'exit':>4}  {'time':>8}")
    print("-" * 28)
    any_fail = False
    for n, m, code, elapsed in summary:
        if code != 0:
            any_fail = True
        print(f"{n:4d}  {m:3d}  {code:4d}  {elapsed:7.1f}s")

    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
