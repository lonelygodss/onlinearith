#!/usr/bin/env python3
"""
Figure 6(b) wider sweep runner (AtoB, seed=0 only).

This script prepares deterministic validation split manifests, builds a wider
calibration-size sweep for A->B only, and can execute all runs by reusing only:
  - calibrate.py (fixed_sum calibration)
  - ppltest.py (held-out evaluation)

No new calibration/evaluation kernels are introduced.

Workflow:
1) Build manifests and run matrix under output root.
2) Optional: run the largest calibration case first (OOM preflight).
3) Finish all quantized calibrations first across available GPUs.
4) After calibration phase completes, run all PPL evaluations.
5) Write summary JSON/CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pty
import random
import select
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ONLINEARITH_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = WORKSPACE_ROOT / "Qwen3-0.6B"

OUTPUT_ROOT_DEFAULT = WORKSPACE_ROOT / "6b" / "atob_seed0_sweep"
CAL_DATASET = ("wikitext", "wikitext-2-raw-v1", "validation")
MIN_TEXT_LEN = 100

DEFAULT_SIZE_PERCENTS = "10,15,20,25,30,35,40,50,60,70,80,90,100"

_PRINT_LOCK = threading.Lock()


@dataclass
class SubsetSpec:
    size_percent: float
    frac: float
    seed: int
    source_count: int
    subset_count: int
    token_count: int | None
    manifest_path: Path
    selected_global_indices: list[int]


@dataclass
class RunSpec:
    run_id: str
    kind: str  # dense | quant
    cal_size_percent: float | None
    cal_size_frac: float | None
    subset_seed: int | None
    cal_size_tokens: int | None
    cal_manifest: Path | None
    eval_manifest: Path
    calibration_file: Path | None
    eval_output_file: Path


def _parse_size_percents(raw: str) -> list[float]:
    vals: list[float] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            p = float(t)
        except ValueError as exc:
            raise ValueError(f"Invalid size percent: {t}") from exc
        if p <= 0.0 or p > 100.0:
            raise ValueError(f"size percent must be in (0, 100], got {p}")
        vals.append(p)

    if not vals:
        raise ValueError("No size percents provided")

    # Keep unique values in ascending order.
    uniq = sorted(set(vals))
    return uniq


def _pct_label(pct: float) -> str:
    text = f"{pct:g}"
    return "p" + text.replace(".", "_")


def _parse_gpu_list(gpu_csv: str | None) -> list[str]:
    if not gpu_csv:
        return []
    ids = [tok.strip() for tok in gpu_csv.split(",") if tok.strip()]
    for tok in ids:
        if not tok.isdigit():
            raise ValueError(f"--gpus expects comma-separated integers, got: {gpu_csv}")
    return ids


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _load_validation_texts() -> tuple[list[str], list[int]]:
    ds_name, ds_config, ds_split = CAL_DATASET
    ds = load_dataset(ds_name, ds_config, split=ds_split)

    texts: list[str] = []
    global_indices: list[int] = []
    for idx, raw in enumerate(ds["text"]):
        txt = raw.strip()
        if len(txt) > MIN_TEXT_LEN:
            texts.append(txt)
            global_indices.append(idx)

    if not texts:
        raise RuntimeError("Filtered validation corpus is empty.")
    return texts, global_indices


def _split_halves(texts: list[str], global_indices: list[int]) -> dict[str, dict[str, Any]]:
    mid = len(texts) // 2
    return {
        "A": {
            "texts": texts[:mid],
            "global_indices": global_indices[:mid],
        },
        "B": {
            "texts": texts[mid:],
            "global_indices": global_indices[mid:],
        },
    }


def _sample_subset(
    source_texts: list[str],
    source_global_indices: list[int],
    frac: float,
    seed: int,
) -> tuple[list[str], list[int]]:
    total = len(source_texts)
    if total == 0:
        return [], []

    if frac >= 1.0:
        local_sel = list(range(total))
    else:
        n = max(1, int(round(total * frac)))
        n = min(n, total)
        rng = random.Random(seed)
        local_sel = sorted(rng.sample(range(total), n))

    texts = [source_texts[i] for i in local_sel]
    global_sel = [source_global_indices[i] for i in local_sel]
    return texts, global_sel


def _count_tokens(tokenizer, texts: list[str], max_length: int) -> int:
    total = 0
    for txt in texts:
        n = len(tokenizer(txt, add_special_tokens=False).input_ids)
        total += min(n, max_length)
    return total


def _cap_split_by_tokens(
    tokenizer,
    texts: list[str],
    global_indices: list[int],
    max_length: int,
    token_cap: int,
) -> tuple[list[str], list[int], int]:
    """
    Keep a deterministic prefix whose truncated token sum stays within token_cap.
    At least one text is kept for non-empty input.
    """
    if token_cap <= 0 or not texts:
        total = _count_tokens(tokenizer, texts, max_length=max_length)
        return texts, global_indices, total

    kept_texts: list[str] = []
    kept_indices: list[int] = []
    running = 0

    for txt, idx in zip(texts, global_indices):
        t = min(len(tokenizer(txt, add_special_tokens=False).input_ids), max_length)
        if kept_texts and running + t > token_cap:
            break
        kept_texts.append(txt)
        kept_indices.append(idx)
        running += t

    if not kept_texts:
        kept_texts.append(texts[0])
        kept_indices.append(global_indices[0])
        running = min(len(tokenizer(texts[0], add_special_tokens=False).input_ids), max_length)

    return kept_texts, kept_indices, running


def _build_manifests(
    output_root: Path,
    size_percents: list[float],
    subset_seed: int,
    max_length: int,
    skip_token_count: bool,
    max_cal_texts_per_split: int,
    max_cal_tokens_per_split: int,
) -> tuple[Path, dict[float, SubsetSpec], dict[str, Any]]:
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    texts, global_indices = _load_validation_texts()
    split_data = _split_halves(texts, global_indices)

    eval_a = manifests_dir / "eval_A.json"
    eval_b = manifests_dir / "eval_B.json"
    _write_json(eval_a, {"texts": split_data["A"]["texts"]})
    _write_json(eval_b, {"texts": split_data["B"]["texts"]})

    tokenizer = None
    need_tokenizer = (not skip_token_count) or (max_cal_tokens_per_split > 0)
    if need_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    a_texts = split_data["A"]["texts"]
    a_indices = split_data["A"]["global_indices"]

    # Keep calibration-source shaping consistent with prior successful 6b flow.
    if max_cal_texts_per_split > 0:
        a_texts = a_texts[:max_cal_texts_per_split]
        a_indices = a_indices[:max_cal_texts_per_split]

    a_token_total = None
    if tokenizer is not None and max_cal_tokens_per_split > 0:
        a_texts, a_indices, a_token_total = _cap_split_by_tokens(
            tokenizer,
            a_texts,
            a_indices,
            max_length=max_length,
            token_cap=max_cal_tokens_per_split,
        )
    elif tokenizer is not None:
        a_token_total = _count_tokens(tokenizer, a_texts, max_length=max_length)

    subset_specs: dict[float, SubsetSpec] = {}
    subset_rows: list[dict[str, Any]] = []

    for pct in size_percents:
        frac = pct / 100.0
        sel_texts, sel_global = _sample_subset(a_texts, a_indices, frac, subset_seed)
        label = _pct_label(pct)
        manifest = manifests_dir / f"cal_A_{label}_seed{subset_seed}.json"
        _write_json(manifest, {"texts": sel_texts})

        token_count = None
        if tokenizer is not None and not skip_token_count:
            token_count = _count_tokens(tokenizer, sel_texts, max_length=max_length)

        spec = SubsetSpec(
            size_percent=pct,
            frac=frac,
            seed=subset_seed,
            source_count=len(a_texts),
            subset_count=len(sel_texts),
            token_count=token_count,
            manifest_path=manifest,
            selected_global_indices=sel_global,
        )
        subset_specs[pct] = spec
        subset_rows.append(
            {
                "size_percent": pct,
                "frac": frac,
                "seed": subset_seed,
                "subset_count": spec.subset_count,
                "token_count": spec.token_count,
                "manifest": str(spec.manifest_path),
                "selected_global_indices": spec.selected_global_indices,
            }
        )

    split_meta = {
        "dataset": "/".join(CAL_DATASET),
        "filter": f"len(text.strip()) > {MIN_TEXT_LEN}",
        "total_filtered_texts": len(texts),
        "split_rule": "deterministic first-half A, second-half B",
        "split_counts": {
            "A": len(split_data["A"]["texts"]),
            "B": len(split_data["B"]["texts"]),
        },
        "eval_manifests": {
            "A": str(eval_a),
            "B": str(eval_b),
        },
        "focus": {
            "direction": "AtoB",
            "subset_seed": subset_seed,
        },
        "calibration_source_A": {
            "count": len(a_texts),
            "token_count_trunc_max_length": a_token_total,
            "max_cal_texts_per_split": max_cal_texts_per_split,
            "max_cal_tokens_per_split": max_cal_tokens_per_split,
        },
        "subsets": subset_rows,
    }

    return eval_b, subset_specs, split_meta


def _build_run_matrix(output_root: Path, eval_b: Path, subset_specs: dict[float, SubsetSpec], subset_seed: int) -> list[RunSpec]:
    cal_dir = output_root / "calibrations"
    eval_dir = output_root / "evals"
    cal_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunSpec] = []

    # Dense reference for held-out B only (A->B focus).
    runs.append(
        RunSpec(
            run_id="dense_B",
            kind="dense",
            cal_size_percent=None,
            cal_size_frac=None,
            subset_seed=None,
            cal_size_tokens=None,
            cal_manifest=None,
            eval_manifest=eval_b,
            calibration_file=None,
            eval_output_file=eval_dir / "ppl_dense_B.json",
        )
    )

    for pct in sorted(subset_specs.keys()):
        spec = subset_specs[pct]
        label = _pct_label(pct)
        run_id = f"AtoB_{label}_seed{subset_seed}"
        runs.append(
            RunSpec(
                run_id=run_id,
                kind="quant",
                cal_size_percent=pct,
                cal_size_frac=(spec.subset_count / spec.source_count) if spec.source_count > 0 else spec.frac,
                subset_seed=subset_seed,
                cal_size_tokens=spec.token_count,
                cal_manifest=spec.manifest_path,
                eval_manifest=eval_b,
                calibration_file=cal_dir / f"calibration_MXFP8_fixed_sum_{run_id}.json",
                eval_output_file=eval_dir / f"ppl_{run_id}.json",
            )
        )

    return runs


def _build_dense_cmd(
    run: RunSpec,
    python_bin: str,
    gpu: str | None,
    use_lite: bool,
    ppl_limit_samples: int,
    eval_default_test: bool,
) -> list[str]:
    cmd = [
        python_bin,
        "ppltest.py",
        "--nproc", "1",
        "--setup", "2",
        "--output", str(run.eval_output_file),
    ]
    if not eval_default_test:
        cmd.extend(["--text-manifest", str(run.eval_manifest)])
    if use_lite:
        cmd.append("--lite")
    if ppl_limit_samples > 0:
        cmd.extend(["--limit-samples", str(ppl_limit_samples)])
    if gpu is not None:
        cmd.extend(["--gpus", gpu])
    return cmd


def _build_cal_cmd(
    run: RunSpec,
    python_bin: str,
    gpu: str | None,
    target_snr: float,
    max_length: int,
    batch_size: int,
    online_delay: int,
    detail_layer: int,
    force: bool,
) -> list[str]:
    assert run.cal_manifest is not None
    assert run.calibration_file is not None
    cmd = [
        python_bin,
        "calibrate.py",
        "--nproc", "1",
        "--setup", "1",
        "--optimizer", "fixed_sum",
        "--target-snr", str(target_snr),
        "--text-manifest", str(run.cal_manifest),
        "--output-dir", str(run.calibration_file.parent),
        "--result-suffix", run.run_id,
        "--max-length", str(max_length),
        "--batch-size", str(batch_size),
        "--online-delay", str(online_delay),
        "--detail-layer", str(detail_layer),
    ]
    if force:
        cmd.append("--force")
    if gpu is not None:
        cmd.extend(["--gpus", gpu])
    return cmd


def _build_quant_eval_cmd(
    run: RunSpec,
    python_bin: str,
    gpu: str | None,
    use_lite: bool,
    ppl_limit_samples: int,
    eval_default_test: bool,
) -> list[str]:
    assert run.calibration_file is not None
    cmd = [
        python_bin,
        "ppltest.py",
        "--nproc", "1",
        "--setup", "6",
        "--calibration", str(run.calibration_file),
        "--output", str(run.eval_output_file),
    ]
    if not eval_default_test:
        cmd.extend(["--text-manifest", str(run.eval_manifest)])
    if use_lite:
        cmd.append("--lite")
    if ppl_limit_samples > 0:
        cmd.extend(["--limit-samples", str(ppl_limit_samples)])
    if gpu is not None:
        cmd.extend(["--gpus", gpu])
    return cmd


def _write_live_bytes(data: bytes) -> None:
    if not data:
        return
    with _PRINT_LOCK:
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except Exception:
            sys.stdout.write(data.decode("utf-8", errors="replace"))
            sys.stdout.flush()


def _run_command(cmd: list[str], cwd: Path, log_path: Path, live_output: bool = True) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    if not live_output:
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write("CMD: " + " ".join(cmd) + "\n\n")
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            elapsed = time.time() - start
            log_f.write(f"\nExit code: {proc.returncode}\n")
            log_f.write(f"Elapsed sec: {elapsed:.2f}\n")
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}. See log: {log_path}")
        return

    master_fd, slave_fd = pty.openpty()
    proc = None
    try:
        with open(log_path, "wb") as log_f:
            header = ("CMD: " + " ".join(cmd) + "\n\n").encode("utf-8", errors="replace")
            log_f.write(header)
            _write_live_bytes(header)

            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
                text=False,
            )
            os.close(slave_fd)
            slave_fd = -1

            while True:
                ready, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in ready:
                    try:
                        chunk = os.read(master_fd, 65536)
                    except OSError:
                        chunk = b""

                    if chunk:
                        log_f.write(chunk)
                        log_f.flush()
                        _write_live_bytes(chunk)
                    elif proc.poll() is not None:
                        break

                if proc.poll() is not None and master_fd not in ready:
                    break

            return_code = proc.wait()
            elapsed = time.time() - start
            tail = f"\nExit code: {return_code}\nElapsed sec: {elapsed:.2f}\n".encode(
                "utf-8", errors="replace"
            )
            log_f.write(tail)
            log_f.flush()
            _write_live_bytes(tail)
    finally:
        if slave_fd != -1:
            try:
                os.close(slave_fd)
            except OSError:
                pass
        try:
            os.close(master_fd)
        except OSError:
            pass

    if proc is not None and proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. See log: {log_path}")


def _execute_calibration_run(
    run: RunSpec,
    run_idx: int,
    total_runs: int,
    python_bin: str,
    gpu: str | None,
    target_snr: float,
    max_length: int,
    batch_size: int,
    online_delay: int,
    detail_layer: int,
    force: bool,
    use_lite: bool,
    ppl_limit_samples: int,
    live_progress: bool,
    logs_dir: Path,
) -> tuple[str, str]:
    print(f"\n=== [CAL {run_idx+1}/{total_runs}] {run.run_id} (gpu={gpu or 'default'}) ===")

    if run.kind != "quant":
        return run.run_id, "skipped"

    assert run.calibration_file is not None
    if run.calibration_file.exists() and not force:
        print(f"Skip: existing calibration output {run.calibration_file}")
        return run.run_id, "skipped"

    cal_cmd = _build_cal_cmd(
        run=run,
        python_bin=python_bin,
        gpu=gpu,
        target_snr=target_snr,
        max_length=max_length,
        batch_size=batch_size,
        online_delay=online_delay,
        detail_layer=detail_layer,
        force=force,
    )
    _run_command(
        cal_cmd,
        cwd=ONLINEARITH_ROOT,
        log_path=logs_dir / f"{run.run_id}_calibrate.log",
        live_output=live_progress,
    )

    return run.run_id, "ok"


def _execute_eval_run(
    run: RunSpec,
    run_idx: int,
    total_runs: int,
    python_bin: str,
    gpu: str | None,
    target_snr: float,
    max_length: int,
    batch_size: int,
    online_delay: int,
    detail_layer: int,
    force: bool,
    use_lite: bool,
    ppl_limit_samples: int,
    eval_default_test: bool,
    live_progress: bool,
    logs_dir: Path,
) -> tuple[str, str]:
    print(f"\n=== [EVAL {run_idx+1}/{total_runs}] {run.run_id} (gpu={gpu or 'default'}) ===")

    if run.kind == "dense":
        if run.eval_output_file.exists() and not force:
            print(f"Skip: existing dense eval output {run.eval_output_file}")
            return run.run_id, "skipped"
        cmd = _build_dense_cmd(
            run,
            python_bin,
            gpu,
            use_lite=use_lite,
            ppl_limit_samples=ppl_limit_samples,
            eval_default_test=eval_default_test,
        )
        _run_command(cmd, cwd=ONLINEARITH_ROOT, log_path=logs_dir / f"{run.run_id}.log", live_output=live_progress)
        return run.run_id, "ok"

    assert run.calibration_file is not None
    if not run.calibration_file.exists():
        raise RuntimeError(
            f"Missing calibration artifact for {run.run_id}: {run.calibration_file}. "
            "Run calibration phase first or remove --run-ids filter mismatch."
        )
    if run.eval_output_file.exists() and not force:
        print(f"Skip: existing quant eval output {run.eval_output_file}")
        return run.run_id, "skipped"

    eval_cmd = _build_quant_eval_cmd(
        run,
        python_bin,
        gpu,
        use_lite=use_lite,
        ppl_limit_samples=ppl_limit_samples,
        eval_default_test=eval_default_test,
    )
    _run_command(
        eval_cmd,
        cwd=ONLINEARITH_ROOT,
        log_path=logs_dir / f"{run.run_id}_eval.log",
        live_output=live_progress,
    )
    return run.run_id, "ok"


def _run_phase_parallel(
    phase_name: str,
    runs: list[RunSpec],
    gpu_ids: list[str],
    worker_fn,
) -> list[tuple[str, str]]:
    if not runs:
        print(f"{phase_name}: no runs selected.")
        return []

    if gpu_ids:
        runs_by_gpu: dict[str, list[tuple[int, RunSpec]]] = {gpu: [] for gpu in gpu_ids}
        for idx, run in enumerate(runs):
            gpu = gpu_ids[idx % len(gpu_ids)]
            runs_by_gpu[gpu].append((idx, run))

        print(f"{phase_name} execution plan:")
        for gpu in gpu_ids:
            assigned = runs_by_gpu[gpu]
            run_ids = [r.run_id for _, r in assigned]
            print(f"  GPU {gpu}: {len(assigned)} run(s) -> {run_ids}")

        active_workers = [(gpu, assigned) for gpu, assigned in runs_by_gpu.items() if assigned]

        def _gpu_worker(gpu_id: str, assignments: list[tuple[int, RunSpec]]) -> list[tuple[str, str]]:
            results: list[tuple[str, str]] = []
            for idx, run in assignments:
                results.append(worker_fn(run, idx, len(runs), gpu_id))
            return results

        phase_status: list[tuple[str, str]] = []
        worker_failures: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=len(active_workers)) as executor:
            futures = {
                executor.submit(_gpu_worker, gpu, assignments): gpu
                for gpu, assignments in active_workers
            }
            for future in as_completed(futures):
                gpu = futures[future]
                try:
                    phase_status.extend(future.result())
                except Exception as exc:
                    worker_failures.append((gpu, str(exc)))

        if worker_failures:
            detail = "; ".join([f"GPU {gpu}: {msg}" for gpu, msg in worker_failures])
            raise RuntimeError(f"{phase_name} failed on one or more workers: {detail}")
        return phase_status

    phase_status: list[tuple[str, str]] = []
    for idx, run in enumerate(runs):
        phase_status.append(worker_fn(run, idx, len(runs), None))
    return phase_status


def _flatten_budgets(cal_data: dict[str, list[float]]) -> list[float]:
    values: list[float] = []
    for layer_vals in cal_data.values():
        values.extend(layer_vals)
    return values


def _summarize_results(output_root: Path, runs: list[RunSpec], target_snr: float) -> None:
    summary_json = output_root / "figure6b_atob_seed0_sweep_results.json"
    summary_csv = output_root / "figure6b_atob_seed0_sweep_results.csv"

    dense_b_ppl = None
    for run in runs:
        if run.kind != "dense":
            continue
        if not run.eval_output_file.exists():
            continue
        with open(run.eval_output_file, "r", encoding="utf-8") as f:
            res = json.load(f)
        ppl = res.get("metrics", {}).get("token_perplexity")
        if isinstance(ppl, (int, float)):
            dense_b_ppl = float(ppl)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for run in runs:
        if run.kind != "quant":
            continue
        assert run.calibration_file is not None
        if not run.calibration_file.exists() or not run.eval_output_file.exists():
            missing.append(run.run_id)
            continue

        with open(run.calibration_file, "r", encoding="utf-8") as f:
            cal_json = json.load(f)
        with open(run.eval_output_file, "r", encoding="utf-8") as f:
            eval_json = json.load(f)

        cal_data = cal_json.get("msd_calibration_data", {})
        budgets = _flatten_budgets(cal_data) if isinstance(cal_data, dict) else []

        ppl_quant = eval_json.get("metrics", {}).get("token_perplexity")
        ppl_ratio = None
        if isinstance(ppl_quant, (int, float)) and isinstance(dense_b_ppl, (int, float)) and dense_b_ppl > 0:
            ppl_ratio = float(ppl_quant) / float(dense_b_ppl)

        g = eval_json.get("msd_perf_stats", {}).get("global", {})
        target = cal_json.get("calibration_params", {}).get("target_snr_db")
        if not isinstance(target, (int, float)):
            target = target_snr

        rows.append(
            {
                "run_id": run.run_id,
                "split_id": "AtoB",
                "cal_size_percent": run.cal_size_percent,
                "cal_size_frac": run.cal_size_frac,
                "subset_seed": run.subset_seed,
                "cal_size_tokens": run.cal_size_tokens,
                "target_snr": target,
                "ppl_quant": ppl_quant,
                "ppl_dense_B": dense_b_ppl,
                "ppl_ratio": ppl_ratio,
                "norm_digit_reads": g.get("global_utilization"),
                "block_skip_rate": g.get("zero_block_ratio"),
                "element_skip_rate": g.get("mac_sparsity", g.get("zero_element_percentage")),
                "partial_window_rate": g.get("partial_block_ratio"),
                "mean_H": (sum(budgets) / len(budgets)) if budgets else None,
                "sum_H": sum(budgets) if budgets else None,
                "calibration_file": str(run.calibration_file),
                "eval_file": str(run.eval_output_file),
            }
        )

    summary = {
        "target_snr": target_snr,
        "focus": "AtoB seed0 wider sweep",
        "dense_reference_B": dense_b_ppl,
        "num_rows": len(rows),
        "missing_quant_runs": missing,
        "rows": rows,
    }
    _write_json(summary_json, summary)

    fieldnames = [
        "run_id",
        "split_id",
        "cal_size_percent",
        "cal_size_frac",
        "subset_seed",
        "cal_size_tokens",
        "target_snr",
        "ppl_quant",
        "ppl_dense_B",
        "ppl_ratio",
        "norm_digit_reads",
        "block_skip_rate",
        "element_skip_rate",
        "partial_window_rate",
        "mean_H",
        "sum_H",
        "calibration_file",
        "eval_file",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _find_largest_quant_run(runs: list[RunSpec]) -> RunSpec:
    quant = [r for r in runs if r.kind == "quant"]
    if not quant:
        raise RuntimeError("No quantized runs found in matrix")

    # Prefer token count if available, otherwise fall back to fraction/percent.
    def key_fn(r: RunSpec) -> tuple[float, float]:
        tok = float(r.cal_size_tokens) if isinstance(r.cal_size_tokens, int) else -1.0
        frac = float(r.cal_size_frac) if isinstance(r.cal_size_frac, float) else -1.0
        return (tok, frac)

    return max(quant, key=key_fn)


def main() -> int:
    parser = argparse.ArgumentParser(description="Figure 6(b) AtoB seed0 wider sweep runner")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT_DEFAULT),
                        help="Artifact root (default: /home/xzj/coding/6b/atob_seed0_sweep)")
    parser.add_argument("--target-snr", type=float, default=15.0,
                        help="Fixed operating point in dB (default: 15).")
    parser.add_argument("--size-percents", type=str, default=DEFAULT_SIZE_PERCENTS,
                        help="Comma-separated calibration size percents for A split (default: wider fine grid).")
    parser.add_argument("--subset-seed", type=int, default=0,
                        help="Subset seed for all sizes (default: 0).")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs; one run uses one GPU (nproc=1).")
    parser.add_argument("--python-bin", type=str, default=sys.executable,
                        help="Python executable for subprocesses (default: current interpreter).")
    parser.add_argument("--cal-max-length", type=int, default=512,
                        help="Calibration max_length forwarded to calibrate.py")
    parser.add_argument("--cal-batch-size", type=int, default=4,
                        help="Calibration batch_size forwarded to calibrate.py")
    parser.add_argument("--online-delay", type=int, default=2,
                        help="MSD online delay forwarded to calibrate.py")
    parser.add_argument("--detail-layer", type=int, default=2,
                        help="Detail layer forwarded to calibrate.py")
    parser.add_argument("--ppl-limit-samples", type=int, default=100,
                        help="Forward --limit-samples to all ppltest calls (default: 100; use <=0 to disable).")
    parser.add_argument("--eval-subdir", type=str, default="evals",
                        help="Subdirectory under output-root for evaluation outputs (default: evals).")
    parser.add_argument("--eval-default-test", action="store_true",
                        help="Evaluate with ppltest default test set (do not pass --text-manifest).")
    parser.add_argument("--execute", action="store_true",
                        help="Execute run commands. Without this flag, only manifests/matrix are prepared.")
    parser.add_argument("--run-ids", nargs="+", default=None,
                        help="Optional subset of run IDs to execute (or summarize).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs already exist.")
    parser.add_argument("--no-lite", action="store_true",
                        help="Disable --lite for ppltest.py runs.")
    parser.add_argument("--skip-token-count", action="store_true",
                        help="Skip tokenizer-based cal_size_tokens counting for faster preparation.")
    parser.add_argument("--max-cal-texts-per-split", type=int, default=20,
                        help="Cap A-split calibration source pool before sampling (default: 20; use <=0 to disable).")
    parser.add_argument("--max-cal-tokens-per-split", type=int, default=0,
                        help="Optional token cap for A-split calibration source after text cap (default: 0 disabled).")
    parser.add_argument("--preflight-largest", action="store_true",
                        help="Enable largest-case-first OOM preflight before calibration phase.")
    parser.add_argument("--no-preflight-largest", action="store_true",
                        help="Legacy compatibility flag: disable largest-case-first OOM preflight.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run only the largest quantized case (and stop). Requires --execute.")
    parser.add_argument("--no-live-progress", action="store_true",
                        help="Disable live subprocess output in terminal (logs are still written).")
    args = parser.parse_args()

    if args.preflight_largest and args.no_preflight_largest:
        print("Cannot set both --preflight-largest and --no-preflight-largest")
        return 2

    if args.subset_seed != 0:
        print(f"Warning: requested subset seed is {args.subset_seed}. This script is intended for seed 0 focus.")

    size_percents = _parse_size_percents(args.size_percents)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    eval_b_manifest, subset_specs, split_meta = _build_manifests(
        output_root=output_root,
        size_percents=size_percents,
        subset_seed=args.subset_seed,
        max_length=args.cal_max_length,
        skip_token_count=args.skip_token_count,
        max_cal_texts_per_split=args.max_cal_texts_per_split,
        max_cal_tokens_per_split=args.max_cal_tokens_per_split,
    )

    eval_dir = output_root / args.eval_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)

    runs = _build_run_matrix(output_root, eval_b_manifest, subset_specs, args.subset_seed)
    for run in runs:
        if run.kind == "dense":
            run.eval_output_file = eval_dir / "ppl_dense_B.json"
        else:
            run.eval_output_file = eval_dir / f"ppl_{run.run_id}.json"
    run_index = {r.run_id: r for r in runs}

    _write_json(output_root / "split_metadata.json", split_meta)
    _write_json(
        output_root / "run_matrix.json",
        {
            "target_snr": args.target_snr,
            "focus": "AtoB seed0 wider sweep",
            "size_percents": size_percents,
            "subset_seed": args.subset_seed,
            "ppl_limit_samples": args.ppl_limit_samples,
            "eval_subdir": args.eval_subdir,
            "eval_default_test": args.eval_default_test,
            "num_runs": len(runs),
            "runs": [
                {
                    "run_id": r.run_id,
                    "kind": r.kind,
                    "cal_size_percent": r.cal_size_percent,
                    "cal_size_frac": r.cal_size_frac,
                    "subset_seed": r.subset_seed,
                    "cal_size_tokens": r.cal_size_tokens,
                    "cal_manifest": str(r.cal_manifest) if r.cal_manifest else None,
                    "eval_manifest": str(r.eval_manifest),
                    "calibration_file": str(r.calibration_file) if r.calibration_file else None,
                    "eval_output_file": str(r.eval_output_file),
                }
                for r in runs
            ],
        },
    )

    selected_runs = runs
    if args.run_ids:
        req = set(args.run_ids)
        missing = sorted(req - set(run_index.keys()))
        if missing:
            print(f"Unknown --run-ids: {missing}")
            return 2
        selected_runs = [r for r in runs if r.run_id in req]

    largest_quant = _find_largest_quant_run(selected_runs)

    print(f"Prepared manifests in: {output_root / 'manifests'}")
    print(f"Prepared run matrix: {output_root / 'run_matrix.json'}")
    print(f"Largest quant run (preflight target): {largest_quant.run_id}")
    print(f"PPL limit-samples: {args.ppl_limit_samples}")
    print(f"Evaluation subdir: {args.eval_subdir}")
    print(f"Evaluation source: {'ppltest default test set' if args.eval_default_test else 'manifest held-out split'}")
    print(f"Selected runs: {[r.run_id for r in selected_runs]}")

    if args.preflight_only and not args.execute:
        print("--preflight-only requires --execute")
        return 2

    if args.execute:
        gpu_ids = _parse_gpu_list(args.gpus)
        use_lite = not args.no_lite
        live_progress = not args.no_live_progress
        logs_dir = output_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # 1) Optional largest-case OOM preflight.
        preflight_enabled = args.preflight_largest or args.preflight_only
        if args.no_preflight_largest and not args.preflight_only:
            preflight_enabled = False
        quant_runs = [r for r in selected_runs if r.kind == "quant"]
        calibration_status: list[tuple[str, str]] = []
        preflight_completed_ids: set[str] = set()

        if preflight_enabled and quant_runs:
            preflight_gpu = gpu_ids[0] if gpu_ids else None
            print(f"\nRunning largest-case calibration preflight first: {largest_quant.run_id} on GPU {preflight_gpu}")
            rid, status = _execute_calibration_run(
                run=largest_quant,
                run_idx=0,
                total_runs=1,
                python_bin=args.python_bin,
                gpu=preflight_gpu,
                target_snr=args.target_snr,
                max_length=args.cal_max_length,
                batch_size=args.cal_batch_size,
                online_delay=args.online_delay,
                detail_layer=args.detail_layer,
                force=args.force,
                use_lite=use_lite,
                ppl_limit_samples=args.ppl_limit_samples,
                live_progress=live_progress,
                logs_dir=logs_dir,
            )
            calibration_status.append((rid, status))
            preflight_completed_ids.add(rid)
            print(f"Preflight result: {rid} -> {status}")

            if args.preflight_only:
                _summarize_results(output_root=output_root, runs=runs, target_snr=args.target_snr)
                print(f"Preflight-only complete. Summary JSON: {output_root / 'figure6b_atob_seed0_sweep_results.json'}")
                return 0
        elif args.preflight_only:
            print("--preflight-only requires preflight to be enabled")
            return 2

        # 2) Calibration phase: complete all quantized calibrations first.
        remaining_cal_runs = [r for r in quant_runs if r.run_id not in preflight_completed_ids]
        calibration_status.extend(
            _run_phase_parallel(
                phase_name="calibration",
                runs=remaining_cal_runs,
                gpu_ids=gpu_ids,
                worker_fn=lambda run, idx, total, gpu: _execute_calibration_run(
                    run=run,
                    run_idx=idx,
                    total_runs=total,
                    python_bin=args.python_bin,
                    gpu=gpu,
                    target_snr=args.target_snr,
                    max_length=args.cal_max_length,
                    batch_size=args.cal_batch_size,
                    online_delay=args.online_delay,
                    detail_layer=args.detail_layer,
                    force=args.force,
                    use_lite=use_lite,
                    ppl_limit_samples=args.ppl_limit_samples,
                    live_progress=live_progress,
                    logs_dir=logs_dir,
                ),
            )
        )

        cal_ok_count = sum(1 for _, status in calibration_status if status == "ok")
        cal_skip_count = sum(1 for _, status in calibration_status if status == "skipped")
        print(f"Calibration phase summary: ok={cal_ok_count}, skipped={cal_skip_count}")

        # 3) Evaluation phase: run dense + quantized ppltest only after calibrations complete.
        eval_status = _run_phase_parallel(
            phase_name="evaluation",
            runs=selected_runs,
            gpu_ids=gpu_ids,
            worker_fn=lambda run, idx, total, gpu: _execute_eval_run(
                run=run,
                run_idx=idx,
                total_runs=total,
                python_bin=args.python_bin,
                gpu=gpu,
                target_snr=args.target_snr,
                max_length=args.cal_max_length,
                batch_size=args.cal_batch_size,
                online_delay=args.online_delay,
                detail_layer=args.detail_layer,
                force=args.force,
                use_lite=use_lite,
                ppl_limit_samples=args.ppl_limit_samples,
                eval_default_test=args.eval_default_test,
                live_progress=live_progress,
                logs_dir=logs_dir,
            ),
        )

        eval_ok_count = sum(1 for _, status in eval_status if status == "ok")
        eval_skip_count = sum(1 for _, status in eval_status if status == "skipped")
        print(f"Evaluation phase summary: ok={eval_ok_count}, skipped={eval_skip_count}")

    _summarize_results(output_root=output_root, runs=runs, target_snr=args.target_snr)
    print(f"Summary JSON: {output_root / 'figure6b_atob_seed0_sweep_results.json'}")
    print(f"Summary CSV : {output_root / 'figure6b_atob_seed0_sweep_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
