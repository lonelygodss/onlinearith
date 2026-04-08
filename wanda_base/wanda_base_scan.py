"""
Single-GPU sequential n:m scan for WANDA-style baseline flow.

For each n:m pair, this script runs two stages in order:
1) calibrate_base.py (produce masks)
2) ppl_batch_base.py (evaluate PPL using those masks)

Usage:
    cd /home/xzj/coding/onlinearith
    python wanda_base/wanda_base_scan.py -nm 2:4 1:4 --only 1 --gpu 0
    python wanda_base/wanda_base_scan.py -nm 2:4 1:4 3:8 --force
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ONLINEARITH_ROOT = SCRIPT_DIR.parent
CALIBRATE_SCRIPT = SCRIPT_DIR / "calibrate_base.py"
PPL_SCRIPT = SCRIPT_DIR / "ppl_batch_base.py"


def parse_nm(token: str) -> tuple[int, int]:
    """Parse one n:m token. Accepted separators: ':', ',', '-' and optional parentheses."""
    match = re.fullmatch(r"\(?\s*(\d+)\s*[:,-]\s*(\d+)\s*\)?", token.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid n:m pair '{token}'. Use forms like 2:4, 2,4, 2-4, or (2,4)."
        )

    n = int(match.group(1))
    m = int(match.group(2))

    if m <= 0:
        raise argparse.ArgumentTypeError(f"Invalid pair '{token}': m must be > 0.")
    if n < 0:
        raise argparse.ArgumentTypeError(f"Invalid pair '{token}': n must be >= 0.")
    if n >= m:
        raise argparse.ArgumentTypeError(f"Invalid pair '{token}': require n < m.")

    return n, m


def _run_stage(cmd: list[str], stage_name: str, cwd: Path) -> tuple[int, float]:
    print(f"[scan] {stage_name} command: {' '.join(cmd)}")
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.perf_counter() - t0
    return completed.returncode, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU sequential n:m scan for wanda_base calibration+PPL flow "
            "(defaults to MXFP8 via --only 1)."
        )
    )
    parser.add_argument(
        "--nm",
        "-nm",
        nargs="+",
        required=True,
        type=parse_nm,
        metavar="N:M",
        help="List of n:m pairs, e.g. -nm 2:4 1:4 (or 2,4 / 2-4 / (2,4)).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        metavar="ID",
        default=[1],
        help="Setup IDs forwarded to both stages (default: 1 for MXFP8 only).",
    )
    parser.add_argument("--force", action="store_true", help="Forwarded to both stages.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Single physical GPU ID, forwarded as --gpus <id>.",
    )
    parser.add_argument("--gpus", type=str, default=None, help="Single GPU ID string (alternative to --gpu).")

    # Calibration-stage forwarding
    parser.add_argument("--num-texts", type=int, default=128, help="Forwarded to calibrate stage.")
    parser.add_argument("--max-length", type=int, default=512, help="Forwarded to calibrate stage.")
    parser.add_argument("--batch-size", type=int, default=4, help="Forwarded to calibrate stage.")

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining n:m pairs even if a stage fails.",
    )
    parser.add_argument("--list", action="store_true", help="Forwarded to both stages.")
    args = parser.parse_args()

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

    print(f"[scan] Calibrate script: {CALIBRATE_SCRIPT}")
    print(f"[scan] PPL script      : {PPL_SCRIPT}")
    print(f"[scan] Total unique n:m pairs: {len(unique_pairs)}")

    summary: list[tuple[int, int, int, float, int, float]] = []

    for i, (n, m) in enumerate(unique_pairs, start=1):
        print()
        print(f"[scan] ({i}/{len(unique_pairs)}) n={n}, m={m}")

        calibrate_cmd = [
            sys.executable,
            str(CALIBRATE_SCRIPT),
            "-n",
            str(n),
            "-m",
            str(m),
            "--nproc",
            "1",
            "--num-texts",
            str(args.num_texts),
            "--max-length",
            str(args.max_length),
            "--batch-size",
            str(args.batch_size),
            "--only",
            *[str(sid) for sid in args.only],
        ]
        if selected_gpu is not None:
            calibrate_cmd.extend(["--gpus", selected_gpu])
        if args.force:
            calibrate_cmd.append("--force")
        if args.list:
            calibrate_cmd.append("--list")

        calib_exit, calib_elapsed = _run_stage(calibrate_cmd, "calibrate", ONLINEARITH_ROOT)
        if calib_exit != 0:
            print(
                f"[scan] FAILED calibrate n={n}, m={m} "
                f"(exit={calib_exit}, {calib_elapsed:.1f}s)"
            )
            summary.append((n, m, calib_exit, calib_elapsed, -1, 0.0))
            if not args.continue_on_error:
                break
            continue

        ppl_cmd = [
            sys.executable,
            str(PPL_SCRIPT),
            "-n",
            str(n),
            "-m",
            str(m),
            "--nproc",
            "1",
            "--only",
            *[str(sid) for sid in args.only],
        ]
        if selected_gpu is not None:
            ppl_cmd.extend(["--gpus", selected_gpu])
        if args.force:
            ppl_cmd.append("--force")
        if args.list:
            ppl_cmd.append("--list")

        ppl_exit, ppl_elapsed = _run_stage(ppl_cmd, "ppl", ONLINEARITH_ROOT)
        summary.append((n, m, calib_exit, calib_elapsed, ppl_exit, ppl_elapsed))

        if ppl_exit != 0:
            print(f"[scan] FAILED ppl n={n}, m={m} (exit={ppl_exit}, {ppl_elapsed:.1f}s)")
            if not args.continue_on_error:
                break
        else:
            print(
                f"[scan] DONE n={n}, m={m} "
                f"(calib {calib_elapsed:.1f}s + ppl {ppl_elapsed:.1f}s)"
            )

    print()
    print("[scan] Summary")
    print(f"{'n':>3}  {'m':>3}  {'cal':>4}  {'t_cal':>8}  {'ppl':>4}  {'t_ppl':>8}")
    print("-" * 46)

    any_fail = False
    for n, m, c_code, c_t, p_code, p_t in summary:
        if c_code != 0 or p_code != 0:
            any_fail = True
        ppl_code_str = "-" if p_code < 0 else str(p_code)
        ppl_time_str = "-" if p_code < 0 else f"{p_t:.1f}s"
        print(f"{n:3d}  {m:3d}  {c_code:4d}  {c_t:7.1f}s  {ppl_code_str:>4}  {ppl_time_str:>8}")

    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
