"""
Single-GPU sequential scan runner for WANDA-style baseline flow.

For each scan point, this script runs two stages in order:
1) calibrate_base.py (produce masks)
2) ppl_batch_base.py (evaluate PPL using those masks)

Supported modes:
- N:M scan mode: iterate over multiple common keep-count N:M pairs.
- calibration-size scan mode: keep one N:M fixed, iterate over multiple
    calibration sizes (--num-texts), using an auto-generated output hook so
    each run reads/writes distinct files.

Usage:
    cd /path/to/onlinearith
    python wanda_base/wanda_base_scan.py -nm 2:4 1:4 --only 1 --gpu 0
    python wanda_base/wanda_base_scan.py -nm 2:4 1:4 3:8 --force
    python wanda_base/wanda_base_scan.py --num-texts-scan 64 128 256 --only 1 --gpu 0 -n 2 -m 4
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


def _parse_positive_int(token: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer: '{token}'") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Value must be > 0, got {value}.")
    return value


def _dedupe_keep_order(values):
    unique = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _append_common_stage_flags(
    cmd: list[str],
    *,
    selected_gpu: str | None,
    force: bool,
    list_mode: bool,
    output_hook: str,
) -> None:
    if selected_gpu is not None:
        cmd.extend(["--gpus", selected_gpu])
    if force:
        cmd.append("--force")
    if list_mode:
        cmd.append("--list")
    if output_hook:
        cmd.extend(["--output-hook", output_hook])


def _run_stage(cmd: list[str], stage_name: str, cwd: Path) -> tuple[int, float]:
    print(f"[scan] {stage_name} command: {' '.join(cmd)}")
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.perf_counter() - t0
    return completed.returncode, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU sequential scan runner for wanda_base calibration+PPL flow "
            "(supports common N:M scan and calibration-size scan; defaults to MXFP8 via --only 1)."
        )
    )
    parser.add_argument(
        "--nm",
        "-nm",
        nargs="+",
        default=None,
        type=parse_nm,
        metavar="N:M",
        help="Common N:M keep-ratio scan list, e.g. -nm 2:4 1:4 (or 2,4 / 2-4 / (2,4)).",
    )
    parser.add_argument(
        "--num-texts-scan",
        nargs="+",
        default=None,
        type=_parse_positive_int,
        metavar="N",
        help="Calibration-size scan mode list (values for calibrate_base --num-texts).",
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

    parser.add_argument("-n", type=int, default=2, help="Fixed n for --num-texts-scan mode.")
    parser.add_argument("-m", type=int, default=4, help="Fixed m for --num-texts-scan mode.")
    parser.add_argument(
        "--output-hook",
        type=str,
        default="",
        help="Optional shared hook appended to output filenames in both stages.",
    )
    parser.add_argument(
        "--output-hook-prefix",
        type=str,
        default="nt",
        help="Prefix for auto hooks in --num-texts-scan mode (default: nt -> nt64, nt128, ...).",
    )

    # Calibration-stage forwarding
    parser.add_argument("--num-texts", type=int, default=2048, help="Forwarded to calibrate stage.")
    parser.add_argument("--max-length", type=int, default=512, help="Forwarded to calibrate stage.")
    parser.add_argument("--batch-size", type=int, default=4, help="Forwarded to calibrate stage.")

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining scan points even if a stage fails.",
    )
    parser.add_argument("--list", action="store_true", help="Forwarded to both stages.")
    args = parser.parse_args()

    has_nm_mode = args.nm is not None
    has_num_text_mode = args.num_texts_scan is not None
    if has_nm_mode == has_num_text_mode:
        raise SystemExit("ERROR: choose exactly one mode: use either --nm ... or --num-texts-scan ...")

    if args.n < 0:
        raise SystemExit("ERROR: -n must be >= 0.")
    if args.m <= 0:
        raise SystemExit("ERROR: -m must be > 0.")
    if args.n > args.m:
        raise SystemExit("ERROR: common N:M requires n <= m.")

    if args.gpus is not None and "," in args.gpus:
        raise SystemExit("ERROR: this scan mode accepts only one GPU ID.")

    selected_gpu = args.gpus
    if selected_gpu is None and args.gpu is not None:
        selected_gpu = str(args.gpu)

    print(f"[scan] Calibrate script: {CALIBRATE_SCRIPT}")
    print(f"[scan] PPL script      : {PPL_SCRIPT}")

    if has_nm_mode:
        unique_pairs = _dedupe_keep_order(args.nm)
        print(f"[scan] Mode: common N:M keep-ratio scan")
        print(f"[scan] Total unique N:M pairs: {len(unique_pairs)}")

        summary: list[tuple[int, int, int, float, int, float]] = []

        for i, (n, m) in enumerate(unique_pairs, start=1):
            print()
            print(f"[scan] ({i}/{len(unique_pairs)}) keep {n}:{m} (prune {m - n}:{m})")

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
            _append_common_stage_flags(
                calibrate_cmd,
                selected_gpu=selected_gpu,
                force=args.force,
                list_mode=args.list,
                output_hook=args.output_hook.strip(),
            )

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
            _append_common_stage_flags(
                ppl_cmd,
                selected_gpu=selected_gpu,
                force=args.force,
                list_mode=args.list,
                output_hook=args.output_hook.strip(),
            )

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
        return

    if len(args.only) != 1:
        raise SystemExit("ERROR: --num-texts-scan mode requires exactly one setup ID via --only.")

    num_texts_values = _dedupe_keep_order(args.num_texts_scan)
    output_hook_prefix = args.output_hook_prefix.strip() or "nt"
    shared_hook = args.output_hook.strip()

    print("[scan] Mode: calibration-size scan")
    print(f"[scan] Fixed N:M: keep {args.n}:{args.m} (prune {args.m - args.n}:{args.m})")
    print(f"[scan] Setup ID : {args.only[0]}")
    print(f"[scan] Num-texts points: {num_texts_values}")
    if shared_hook:
        print(f"[scan] Shared output hook prefix: {shared_hook}")

    size_summary: list[tuple[int, str, int, float, int, float]] = []

    for i, num_texts in enumerate(num_texts_values, start=1):
        auto_hook = f"{output_hook_prefix}{num_texts}"
        run_hook = f"{shared_hook}_{auto_hook}" if shared_hook else auto_hook

        print()
        print(f"[scan] ({i}/{len(num_texts_values)}) num_texts={num_texts}  hook={run_hook}")

        calibrate_cmd = [
            sys.executable,
            str(CALIBRATE_SCRIPT),
            "-n",
            str(args.n),
            "-m",
            str(args.m),
            "--nproc",
            "1",
            "--num-texts",
            str(num_texts),
            "--max-length",
            str(args.max_length),
            "--batch-size",
            str(args.batch_size),
            "--only",
            str(args.only[0]),
        ]
        _append_common_stage_flags(
            calibrate_cmd,
            selected_gpu=selected_gpu,
            force=args.force,
            list_mode=args.list,
            output_hook=run_hook,
        )

        calib_exit, calib_elapsed = _run_stage(calibrate_cmd, "calibrate", ONLINEARITH_ROOT)
        if calib_exit != 0:
            print(
                f"[scan] FAILED calibrate num_texts={num_texts} "
                f"(exit={calib_exit}, {calib_elapsed:.1f}s)"
            )
            size_summary.append((num_texts, run_hook, calib_exit, calib_elapsed, -1, 0.0))
            if not args.continue_on_error:
                break
            continue

        ppl_cmd = [
            sys.executable,
            str(PPL_SCRIPT),
            "-n",
            str(args.n),
            "-m",
            str(args.m),
            "--nproc",
            "1",
            "--only",
            str(args.only[0]),
        ]
        _append_common_stage_flags(
            ppl_cmd,
            selected_gpu=selected_gpu,
            force=args.force,
            list_mode=args.list,
            output_hook=run_hook,
        )

        ppl_exit, ppl_elapsed = _run_stage(ppl_cmd, "ppl", ONLINEARITH_ROOT)
        size_summary.append((num_texts, run_hook, calib_exit, calib_elapsed, ppl_exit, ppl_elapsed))

        if ppl_exit != 0:
            print(
                f"[scan] FAILED ppl num_texts={num_texts} "
                f"(exit={ppl_exit}, {ppl_elapsed:.1f}s)"
            )
            if not args.continue_on_error:
                break
        else:
            print(
                f"[scan] DONE num_texts={num_texts} "
                f"(calib {calib_elapsed:.1f}s + ppl {ppl_elapsed:.1f}s)"
            )

    print()
    print("[scan] Summary")
    print(f"{'texts':>7}  {'hook':<24}  {'cal':>4}  {'t_cal':>8}  {'ppl':>4}  {'t_ppl':>8}")
    print("-" * 74)

    any_fail = False
    for num_texts, run_hook, c_code, c_t, p_code, p_t in size_summary:
        if c_code != 0 or p_code != 0:
            any_fail = True
        ppl_code_str = "-" if p_code < 0 else str(p_code)
        ppl_time_str = "-" if p_code < 0 else f"{p_t:.1f}s"
        print(f"{num_texts:7d}  {run_hook:<24}  {c_code:4d}  {c_t:7.1f}s  {ppl_code_str:>4}  {ppl_time_str:>8}")

    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
