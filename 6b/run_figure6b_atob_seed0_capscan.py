#!/usr/bin/env python3
"""
Cap-scale scan driver for Figure 6(b) A->B, seed 0.

This script does not implement new calibration/evaluation kernels. It only
orchestrates existing infrastructure by calling:
  - run_figure6b_atob_seed0_sweep.py
    - which calls calibrate.py (fixed_sum calibration)
    - and ppltest.py (held-out evaluation)

Primary use:
1) OOM preflight for higher calibration-source caps (e.g., 2x/4x/8x).
2) Optional full finer-grained sweeps on caps that pass preflight.

Example:
  /home/xzj/coding/.venv3_10/bin/python 6b/run_figure6b_atob_seed0_capscan.py \
    --scales 2,4,8 \
    --probe-gpus 0,1,2 \
    --full-gpus 0,1,2,3,4,5,6,7 \
    --execute-full-on-pass
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_SCRIPT = SCRIPT_DIR / "run_figure6b_atob_seed0_sweep.py"
ONLINEARITH_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "6b" / "atob_seed0_capscan"
DEFAULT_SIZE_PERCENTS = "10,15,20,25,30,35,40,50,60,70,80,90,100"


@dataclass
class ProbeResult:
    scale: int
    cap_texts: int
    gpu: str
    output_root: str
    command: list[str]
    return_code: int
    elapsed_sec: float
    oom_detected: bool
    error: str | None
    calibrate_log: str | None


@dataclass
class FullRunResult:
    scale: int
    cap_texts: int
    output_root: str
    command: list[str]
    return_code: int
    elapsed_sec: float
    error: str | None


def _parse_int_list(csv_text: str, name: str) -> list[int]:
    vals: list[int] = []
    for tok in csv_text.split(","):
        t = tok.strip()
        if not t:
            continue
        if not t.isdigit():
            raise ValueError(f"{name} expects comma-separated positive integers, got: {csv_text}")
        vals.append(int(t))
    if not vals:
        raise ValueError(f"{name} is empty")
    return vals


def _parse_gpu_list(csv_text: str) -> list[str]:
    vals: list[str] = []
    for tok in csv_text.split(","):
        t = tok.strip()
        if not t:
            continue
        if not t.isdigit():
            raise ValueError(f"GPU list expects comma-separated integers, got: {csv_text}")
        vals.append(t)
    if not vals:
        raise ValueError("GPU list is empty")
    return vals


def _detect_oom_from_log(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    needles = [
        "cuda out of memory",
        "outofmemoryerror",
        "torch.cuda.outofmemoryerror",
    ]
    return any(n in text for n in needles)


def _find_calibrate_log(output_root: Path) -> Path | None:
    logs_dir = output_root / "logs"
    if not logs_dir.exists():
        return None
    logs = sorted(logs_dir.glob("*_calibrate.log"))
    if not logs:
        return None
    # Preflight-only mode should have exactly one calibrate log.
    return logs[0]


def _run_subprocess(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - t0
    return proc.returncode, elapsed


def _build_sweep_cmd(
    python_bin: str,
    output_root: Path,
    target_snr: float,
    size_percents: str,
    subset_seed: int,
    gpus_csv: str,
    max_cal_texts: int,
    max_cal_tokens: int,
    cal_max_length: int,
    cal_batch_size: int,
    online_delay: int,
    detail_layer: int,
    execute: bool,
    preflight_only: bool,
    no_preflight_largest: bool,
    force: bool,
    no_lite: bool,
    skip_token_count: bool,
    no_live_progress: bool,
) -> list[str]:
    cmd = [
        python_bin,
        str(SWEEP_SCRIPT),
        "--output-root",
        str(output_root),
        "--target-snr",
        str(target_snr),
        "--size-percents",
        size_percents,
        "--subset-seed",
        str(subset_seed),
        "--gpus",
        gpus_csv,
        "--max-cal-texts-per-split",
        str(max_cal_texts),
        "--max-cal-tokens-per-split",
        str(max_cal_tokens),
        "--cal-max-length",
        str(cal_max_length),
        "--cal-batch-size",
        str(cal_batch_size),
        "--online-delay",
        str(online_delay),
        "--detail-layer",
        str(detail_layer),
    ]

    if execute:
        cmd.append("--execute")
    if preflight_only:
        cmd.append("--preflight-only")
    if no_preflight_largest:
        cmd.append("--no-preflight-largest")
    if force:
        cmd.append("--force")
    if no_lite:
        cmd.append("--no-lite")
    if skip_token_count:
        cmd.append("--skip-token-count")
    if no_live_progress:
        cmd.append("--no-live-progress")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="AtoB seed0 cap-scale scan driver")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                        help="Root directory for cap-scan outputs.")
    parser.add_argument("--target-snr", type=float, default=15.0,
                        help="Fixed target SNR in dB (default: 15).")
    parser.add_argument("--size-percents", type=str, default=DEFAULT_SIZE_PERCENTS,
                        help="Finer-grained size grid forwarded to sweep script.")
    parser.add_argument("--subset-seed", type=int, default=0,
                        help="Subset seed (default: 0).")
    parser.add_argument("--base-max-cal-texts", type=int, default=20,
                        help="Base calibration source cap per split before applying scale.")
    parser.add_argument("--max-cal-tokens-per-split", type=int, default=0,
                        help="Optional token cap per split (forwarded).")
    parser.add_argument("--scales", type=str, default="2,4,8",
                        help="Scale multipliers to test, comma-separated (default: 2,4,8).")
    parser.add_argument("--probe-gpus", type=str, default="0,1,2",
                        help="GPUs for parallel preflight probes (one scale per GPU worker).")
    parser.add_argument("--full-gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="GPUs for full sweep runs on passing scales.")
    parser.add_argument("--execute-full-on-pass", action="store_true",
                        help="After preflight, run full sweep for scales that passed.")
    parser.add_argument("--python-bin", type=str, default=sys.executable,
                        help="Python executable for subprocesses.")
    parser.add_argument("--cal-max-length", type=int, default=512)
    parser.add_argument("--cal-batch-size", type=int, default=4)
    parser.add_argument("--online-delay", type=int, default=2)
    parser.add_argument("--detail-layer", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-lite", action="store_true")
    parser.add_argument("--skip-token-count", action="store_true")
    parser.add_argument("--no-live-progress", action="store_true")
    args = parser.parse_args()

    if args.subset_seed != 0:
        print(f"Warning: subset_seed={args.subset_seed}; this flow is intended for seed 0 focus.")

    scales = _parse_int_list(args.scales, "--scales")
    probe_gpus = _parse_gpu_list(args.probe_gpus)
    full_gpus = _parse_gpu_list(args.full_gpus)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    driver_logs = output_root / "driver_logs"
    driver_logs.mkdir(parents=True, exist_ok=True)

    print("Cap-scan configuration:")
    print(f"  scales: {scales}")
    print(f"  base max_cal_texts: {args.base_max_cal_texts}")
    print(f"  size_percents: {args.size_percents}")
    print(f"  probe_gpus: {probe_gpus}")
    print(f"  full_gpus: {full_gpus}")
    print(f"  execute_full_on_pass: {args.execute_full_on_pass}")

    # Phase 1: OOM preflight for each scale (parallel across probe GPUs).
    assignments: list[tuple[int, int, str, Path]] = []
    for i, scale in enumerate(scales):
        cap_texts = max(1, int(args.base_max_cal_texts * scale))
        gpu = probe_gpus[i % len(probe_gpus)]
        cap_root = output_root / f"cap{scale}x"
        assignments.append((scale, cap_texts, gpu, cap_root))

    print("\nPreflight plan:")
    for scale, cap_texts, gpu, cap_root in assignments:
        print(f"  scale={scale}x cap={cap_texts} gpu={gpu} root={cap_root}")

    def _probe_one(scale: int, cap_texts: int, gpu: str, cap_root: Path) -> ProbeResult:
        cmd = _build_sweep_cmd(
            python_bin=args.python_bin,
            output_root=cap_root,
            target_snr=args.target_snr,
            size_percents=args.size_percents,
            subset_seed=args.subset_seed,
            gpus_csv=gpu,
            max_cal_texts=cap_texts,
            max_cal_tokens=args.max_cal_tokens_per_split,
            cal_max_length=args.cal_max_length,
            cal_batch_size=args.cal_batch_size,
            online_delay=args.online_delay,
            detail_layer=args.detail_layer,
            execute=True,
            preflight_only=True,
            no_preflight_largest=False,
            force=args.force,
            no_lite=args.no_lite,
            skip_token_count=args.skip_token_count,
            no_live_progress=args.no_live_progress,
        )

        log_file = driver_logs / f"probe_cap{scale}x.log"
        try:
            rc, elapsed = _run_subprocess(cmd, cwd=ONLINEARITH_ROOT, log_path=log_file)
            cal_log = _find_calibrate_log(cap_root)
            oom = _detect_oom_from_log(cal_log)
            return ProbeResult(
                scale=scale,
                cap_texts=cap_texts,
                gpu=gpu,
                output_root=str(cap_root),
                command=cmd,
                return_code=rc,
                elapsed_sec=elapsed,
                oom_detected=oom,
                error=None,
                calibrate_log=str(cal_log) if cal_log else None,
            )
        except Exception as exc:
            cal_log = _find_calibrate_log(cap_root)
            oom = _detect_oom_from_log(cal_log)
            return ProbeResult(
                scale=scale,
                cap_texts=cap_texts,
                gpu=gpu,
                output_root=str(cap_root),
                command=cmd,
                return_code=1,
                elapsed_sec=0.0,
                oom_detected=oom,
                error=str(exc),
                calibrate_log=str(cal_log) if cal_log else None,
            )

    probe_results: list[ProbeResult] = []
    with ThreadPoolExecutor(max_workers=min(len(assignments), len(probe_gpus))) as executor:
        futures = [
            executor.submit(_probe_one, scale, cap_texts, gpu, cap_root)
            for scale, cap_texts, gpu, cap_root in assignments
        ]
        for fut in as_completed(futures):
            probe_results.append(fut.result())

    probe_results.sort(key=lambda r: r.scale)

    print("\nPreflight results:")
    for r in probe_results:
        status = "PASS" if r.return_code == 0 else "FAIL"
        oom_note = "OOM" if r.oom_detected else "no-OOM-signature"
        print(
            f"  {r.scale}x cap={r.cap_texts} gpu={r.gpu}: {status} ({oom_note}) "
            f"rc={r.return_code} t={r.elapsed_sec:.1f}s"
        )

    _write_obj = {
        "target_snr": args.target_snr,
        "subset_seed": args.subset_seed,
        "size_percents": args.size_percents,
        "base_max_cal_texts": args.base_max_cal_texts,
        "scales": scales,
        "probe_gpus": probe_gpus,
        "results": [asdict(r) for r in probe_results],
    }
    with (output_root / "preflight_capscan_results.json").open("w", encoding="utf-8") as f:
        json.dump(_write_obj, f, indent=2)

    pass_scales = [r for r in probe_results if r.return_code == 0]

    # Phase 2: optional full sweep runs on passing caps.
    full_results: list[FullRunResult] = []
    if args.execute_full_on_pass and pass_scales:
        full_gpu_csv = ",".join(full_gpus)
        print("\nLaunching full sweeps on passing caps:")
        for r in pass_scales:
            cap_root = Path(r.output_root)
            cmd = _build_sweep_cmd(
                python_bin=args.python_bin,
                output_root=cap_root,
                target_snr=args.target_snr,
                size_percents=args.size_percents,
                subset_seed=args.subset_seed,
                gpus_csv=full_gpu_csv,
                max_cal_texts=r.cap_texts,
                max_cal_tokens=args.max_cal_tokens_per_split,
                cal_max_length=args.cal_max_length,
                cal_batch_size=args.cal_batch_size,
                online_delay=args.online_delay,
                detail_layer=args.detail_layer,
                execute=True,
                preflight_only=False,
                no_preflight_largest=True,
                force=args.force,
                no_lite=args.no_lite,
                skip_token_count=args.skip_token_count,
                no_live_progress=args.no_live_progress,
            )
            log_file = driver_logs / f"full_cap{r.scale}x.log"

            print(f"  running scale={r.scale}x cap={r.cap_texts} with gpus={full_gpu_csv}")
            try:
                rc, elapsed = _run_subprocess(cmd, cwd=ONLINEARITH_ROOT, log_path=log_file)
                full_results.append(
                    FullRunResult(
                        scale=r.scale,
                        cap_texts=r.cap_texts,
                        output_root=str(cap_root),
                        command=cmd,
                        return_code=rc,
                        elapsed_sec=elapsed,
                        error=None,
                    )
                )
            except Exception as exc:
                full_results.append(
                    FullRunResult(
                        scale=r.scale,
                        cap_texts=r.cap_texts,
                        output_root=str(cap_root),
                        command=cmd,
                        return_code=1,
                        elapsed_sec=0.0,
                        error=str(exc),
                    )
                )
                break

        with (output_root / "full_capscan_results.json").open("w", encoding="utf-8") as f:
            json.dump({"results": [asdict(r) for r in full_results]}, f, indent=2)

    print("\nArtifacts:")
    print(f"  preflight summary: {output_root / 'preflight_capscan_results.json'}")
    if full_results:
        print(f"  full summary: {output_root / 'full_capscan_results.json'}")
    print(f"  driver logs: {driver_logs}")

    # Return non-zero if any preflight failed (including OOM).
    any_preflight_fail = any(r.return_code != 0 for r in probe_results)
    return 1 if any_preflight_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
