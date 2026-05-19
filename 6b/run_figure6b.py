#!/usr/bin/env python3
"""
Figure 6(b) stage-D runner.

This script prepares deterministic validation-set splits, builds the 10-run
Figure 6(b) matrix, and (optionally) executes runs by reusing calibrate.py and
ppltest.py without introducing a new calibration/evaluation process.

Default behavior:
  - Prepare manifests + run matrix under the workspace-level 6b directory
  - Write preflight OOM estimates
  - Do not execute heavy runs unless --execute is provided

Matrix (10 evals total):
  - Dense references: dense_A, dense_B
  - Quantized held-out:
      AtoB small seed0
      AtoB small seed1
      AtoB medium seed0
      AtoB large seed0
      BtoA small seed0
      BtoA small seed1
      BtoA medium seed0
      BtoA large seed0
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

OUTPUT_ROOT_DEFAULT = WORKSPACE_ROOT / "6b"
CAL_DATASET = ("wikitext", "wikitext-2-raw-v1", "validation")
MIN_TEXT_LEN = 100

SIZE_SPECS = [
    ("small", 0.10, [0, 1]),
    ("medium", 0.30, [0]),
    ("large", 1.00, [0]),
]

_PRINT_LOCK = threading.Lock()


@dataclass
class SubsetSpec:
    source_split: str
    size_label: str
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
    split_id: str
    eval_split: str
    cal_split: str | None
    cal_size_label: str | None
    cal_size_frac: float | None
    subset_seed: int | None
    cal_size_tokens: int | None
    cal_manifest: Path | None
    eval_manifest: Path
    calibration_file: Path | None
    eval_output_file: Path


def _parse_gpu_list(gpu_csv: str | None) -> list[str]:
    if not gpu_csv:
        return []
    ids = [tok.strip() for tok in gpu_csv.split(",") if tok.strip()]
    for tok in ids:
        if not tok.isdigit():
            raise ValueError(f"--gpus expects comma-separated integers, got: {gpu_csv}")
    return ids


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
        raise RuntimeError("Filtered validation corpus is empty; cannot build Figure 6(b) splits.")
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
    min_count: int = 1,
) -> tuple[list[str], list[int]]:
    total = len(source_texts)
    if total == 0:
        return [], []

    if frac >= 1.0:
        local_sel = list(range(total))
    else:
        n = max(min_count, int(round(total * frac)))
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
    Deterministically keep a prefix of texts whose truncated-token sum stays
    within token_cap. At least one text is kept when input is non-empty.
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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _build_manifests(
    output_root: Path,
    max_length: int,
    skip_token_count: bool,
    max_cal_texts_per_split: int,
    max_cal_tokens_per_split: int,
    min_small_texts: int,
    min_medium_texts: int,
) -> tuple[dict[str, Path], dict[tuple[str, str, int], SubsetSpec], dict[str, Any]]:
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    texts, global_indices = _load_validation_texts()
    split_data = _split_halves(texts, global_indices)

    tokenizer = None
    need_tokenizer = (not skip_token_count) or (max_cal_tokens_per_split > 0)
    if need_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    eval_manifests: dict[str, Path] = {}
    split_meta: dict[str, Any] = {
        "dataset": "/".join(CAL_DATASET),
        "filter": f"len(text.strip()) > {MIN_TEXT_LEN}",
        "total_filtered_texts": len(texts),
        "total_filtered_tokens_trunc_max_length": None,
        "max_cal_texts_per_split": max_cal_texts_per_split,
        "max_cal_tokens_per_split": max_cal_tokens_per_split,
        "min_small_texts": min_small_texts,
        "min_medium_texts": min_medium_texts,
        "splits": {},
        "calibration_source": {},
        "subsets": [],
    }

    cal_sources: dict[str, dict[str, Any]] = {}

    for split_name in ("A", "B"):
        eval_path = manifests_dir / f"eval_{split_name}.json"
        split_texts = split_data[split_name]["texts"]
        split_indices = split_data[split_name]["global_indices"]
        _write_json(eval_path, {"texts": split_texts})
        eval_manifests[split_name] = eval_path

        split_token_count = None
        if tokenizer is not None:
            split_token_count = _count_tokens(tokenizer, split_texts, max_length=max_length)

        split_meta["splits"][split_name] = {
            "manifest": str(eval_path),
            "count": len(split_texts),
            "token_count_trunc_max_length": split_token_count,
            "global_indices": split_indices,
        }

        cal_source_texts = split_texts
        cal_source_indices = split_indices
        if max_cal_texts_per_split > 0:
            cal_source_texts = cal_source_texts[:max_cal_texts_per_split]
            cal_source_indices = cal_source_indices[:max_cal_texts_per_split]

        if tokenizer is not None and max_cal_tokens_per_split > 0:
            cal_texts, cal_indices, cal_tokens = _cap_split_by_tokens(
                tokenizer,
                cal_source_texts,
                cal_source_indices,
                max_length=max_length,
                token_cap=max_cal_tokens_per_split,
            )
        else:
            cal_texts = cal_source_texts
            cal_indices = cal_source_indices
            cal_tokens = _count_tokens(tokenizer, cal_texts, max_length=max_length) if tokenizer is not None else None

        cal_sources[split_name] = {
            "texts": cal_texts,
            "global_indices": cal_indices,
            "token_count": cal_tokens,
        }
        split_meta["calibration_source"][split_name] = {
            "count": len(cal_texts),
            "token_count_trunc_max_length": cal_tokens,
            "global_indices": cal_indices,
        }

    if tokenizer is not None:
        split_meta["total_filtered_tokens_trunc_max_length"] = sum(
            split_meta["splits"][name]["token_count_trunc_max_length"] for name in ("A", "B")
        )

    subset_specs: dict[tuple[str, str, int], SubsetSpec] = {}

    for source_split in ("A", "B"):
        source_texts = cal_sources[source_split]["texts"]
        source_global = cal_sources[source_split]["global_indices"]

        for size_label, frac, seeds in SIZE_SPECS:
            for seed in seeds:
                min_subset = 1
                if size_label == "small":
                    min_subset = max(1, min_small_texts)
                elif size_label == "medium":
                    min_subset = max(1, min_medium_texts)
                sel_texts, sel_global = _sample_subset(
                    source_texts,
                    source_global,
                    frac,
                    seed,
                    min_count=min_subset,
                )
                manifest = manifests_dir / f"cal_{source_split}_{size_label}_seed{seed}.json"
                _write_json(manifest, {"texts": sel_texts})

                token_count = None
                if tokenizer is not None:
                    token_count = _count_tokens(tokenizer, sel_texts, max_length=max_length)

                spec = SubsetSpec(
                    source_split=source_split,
                    size_label=size_label,
                    frac=frac,
                    seed=seed,
                    source_count=len(source_texts),
                    subset_count=len(sel_texts),
                    token_count=token_count,
                    manifest_path=manifest,
                    selected_global_indices=sel_global,
                )
                subset_specs[(source_split, size_label, seed)] = spec
                split_meta["subsets"].append(
                    {
                        "source_split": source_split,
                        "size_label": size_label,
                        "frac": frac,
                        "seed": seed,
                        "min_subset_count": min_subset,
                        "manifest": str(manifest),
                        "source_count": spec.source_count,
                        "subset_count": spec.subset_count,
                        "token_count": spec.token_count,
                        "selected_global_indices": spec.selected_global_indices,
                    }
                )

    return eval_manifests, subset_specs, split_meta


def _build_run_matrix(
    output_root: Path,
    subset_specs: dict[tuple[str, str, int], SubsetSpec],
    eval_manifests: dict[str, Path],
) -> list[RunSpec]:
    cal_dir = output_root / "calibrations"
    eval_dir = output_root / "evals"
    cal_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunSpec] = []

    # Dense references on each held-out split.
    for split_name in ("A", "B"):
        run_id = f"dense_{split_name}"
        runs.append(
            RunSpec(
                run_id=run_id,
                kind="dense",
                split_id=run_id,
                eval_split=split_name,
                cal_split=None,
                cal_size_label=None,
                cal_size_frac=None,
                subset_seed=None,
                cal_size_tokens=None,
                cal_manifest=None,
                eval_manifest=eval_manifests[split_name],
                calibration_file=None,
                eval_output_file=eval_dir / f"ppl_{run_id}.json",
            )
        )

    # Quantized held-out runs.
    directions = [
        ("A", "B", "AtoB"),
        ("B", "A", "BtoA"),
    ]
    for cal_split, eval_split, split_id in directions:
        for size_label, frac, seeds in SIZE_SPECS:
            for seed in seeds:
                spec = subset_specs[(cal_split, size_label, seed)]
                run_id = f"{split_id}_{size_label}_seed{seed}"
                runs.append(
                    RunSpec(
                        run_id=run_id,
                        kind="quant",
                        split_id=split_id,
                        eval_split=eval_split,
                        cal_split=cal_split,
                        cal_size_label=size_label,
                        cal_size_frac=(spec.subset_count / spec.source_count) if spec.source_count > 0 else frac,
                        subset_seed=seed,
                        cal_size_tokens=spec.token_count,
                        cal_manifest=spec.manifest_path,
                        eval_manifest=eval_manifests[eval_split],
                        calibration_file=cal_dir / f"calibration_MXFP8_fixed_sum_{run_id}.json",
                        eval_output_file=eval_dir / f"ppl_{run_id}.json",
                    )
                )

    return runs


def _build_dense_cmd(run: RunSpec, python_bin: str, gpu: str | None, use_lite: bool) -> list[str]:
    cmd = [
        python_bin,
        "ppltest.py",
        "--nproc", "1",
        "--setup", "2",  # MXFP8 only (dense reference)
        "--text-manifest", str(run.eval_manifest),
        "--output", str(run.eval_output_file),
    ]
    if use_lite:
        cmd.append("--lite")
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
        "--setup", "1",  # MXFP8 calibration setup
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
) -> list[str]:
    assert run.calibration_file is not None

    cmd = [
        python_bin,
        "ppltest.py",
        "--nproc", "1",
        "--setup", "6",  # MXFP8 + MSD B=16, then replaced by calibration data
        "--calibration", str(run.calibration_file),
        "--text-manifest", str(run.eval_manifest),
        "--output", str(run.eval_output_file),
    ]
    if use_lite:
        cmd.append("--lite")
    if gpu is not None:
        cmd.extend(["--gpus", gpu])
    return cmd


def _write_live_bytes(data: bytes) -> None:
    """Thread-safe raw-byte write to stdout for live PTY output."""
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
            raise RuntimeError(
                f"Command failed with exit code {proc.returncode}. See log: {log_path}"
            )
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
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}. See log: {log_path}"
        )


def _execute_one_run(
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
    live_progress: bool,
    logs_dir: Path,
) -> tuple[str, str]:
    """Execute one run end-to-end on a single assigned GPU."""
    print(f"\n=== [{run_idx+1}/{total_runs}] {run.run_id} (gpu={gpu or 'default'}) ===")

    if run.kind == "dense":
        if run.eval_output_file.exists() and not force:
            print(f"Skip: existing dense eval output {run.eval_output_file}")
            return run.run_id, "skipped"
        cmd = _build_dense_cmd(run, python_bin, gpu, use_lite=use_lite)
        _run_command(
            cmd,
            cwd=ONLINEARITH_ROOT,
            log_path=logs_dir / f"{run.run_id}.log",
            live_output=live_progress,
        )
        return run.run_id, "ok"

    assert run.calibration_file is not None
    if run.calibration_file.exists() and run.eval_output_file.exists() and not force:
        print(f"Skip: existing quant outputs {run.calibration_file.name}, {run.eval_output_file.name}")
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

    eval_cmd = _build_quant_eval_cmd(run, python_bin, gpu, use_lite=use_lite)
    _run_command(
        eval_cmd,
        cwd=ONLINEARITH_ROOT,
        log_path=logs_dir / f"{run.run_id}_eval.log",
        live_output=live_progress,
    )
    return run.run_id, "ok"


def _flatten_budgets(cal_data: dict[str, list[float]]) -> list[float]:
    values: list[float] = []
    for layer_vals in cal_data.values():
        values.extend(layer_vals)
    return values


def _summarize_results(output_root: Path, runs: list[RunSpec], target_snr: float) -> None:
    summary_json = output_root / "figure6b_results.json"
    summary_csv = output_root / "figure6b_results.csv"
    mapping_note = output_root / "metric_mapping.md"

    dense_ppl: dict[str, float] = {}
    for run in runs:
        if run.kind != "dense":
            continue
        if not run.eval_output_file.exists():
            continue
        with open(run.eval_output_file, "r", encoding="utf-8") as f:
            res = json.load(f)
        ppl = res.get("metrics", {}).get("token_perplexity")
        if isinstance(ppl, (int, float)):
            dense_ppl[run.eval_split] = float(ppl)

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
        ppl_dense = dense_ppl.get(run.eval_split)
        ppl_ratio = None
        if isinstance(ppl_quant, (int, float)) and isinstance(ppl_dense, (int, float)) and ppl_dense > 0:
            ppl_ratio = float(ppl_quant) / float(ppl_dense)

        g = eval_json.get("msd_perf_stats", {}).get("global", {})

        target = cal_json.get("calibration_params", {}).get("target_snr_db")
        if not isinstance(target, (int, float)):
            target = target_snr

        row = {
            "run_id": run.run_id,
            "split_id": run.split_id,
            "cal_split": run.cal_split,
            "eval_split": run.eval_split,
            "cal_size_label": run.cal_size_label,
            "cal_size_frac": run.cal_size_frac,
            "cal_size_tokens": run.cal_size_tokens,
            "subset_seed": run.subset_seed,
            "target_snr": target,
            "ppl_quant": ppl_quant,
            "ppl_dense": ppl_dense,
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
        rows.append(row)

    summary = {
        "target_snr": target_snr,
        "dense_references": dense_ppl,
        "num_rows": len(rows),
        "missing_quant_runs": missing,
        "rows": rows,
    }
    _write_json(summary_json, summary)

    fieldnames = [
        "run_id",
        "split_id",
        "cal_split",
        "eval_split",
        "cal_size_label",
        "cal_size_frac",
        "cal_size_tokens",
        "subset_seed",
        "target_snr",
        "ppl_quant",
        "ppl_dense",
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

    mapping_note.write_text(
        "\n".join([
            "Figure 6(b) metric mapping",
            "",
            "- norm_digit_reads: msd_perf_stats.global.global_utilization",
            "- block_skip_rate: msd_perf_stats.global.zero_block_ratio",
            "- element_skip_rate: msd_perf_stats.global.mac_sparsity (fallback: zero_element_percentage)",
            "- partial_window_rate: msd_perf_stats.global.partial_block_ratio",
            "- mean_H, sum_H: aggregate of calibration JSON msd_calibration_data budgets",
            "- ppl_ratio: ppl_quant / ppl_dense(eval_split)",
            "",
            "Note: in --lite mode, partial_window_rate may be null if partial_block_ratio",
            "is not emitted by the runtime stats collector.",
        ]),
        encoding="utf-8",
    )


def _write_preflight(
    output_root: Path,
    subset_specs: dict[tuple[str, str, int], SubsetSpec],
    split_meta: dict[str, Any],
    max_length: int,
) -> None:
    # Approximate per-layer cache bytes used by calibration_msd capture:
    # x_q:      N * nb * bs * float32
    # x_scales: N * nb * float32
    # with nb=32, bs=32 -> bytes_per_token = (32*32 + 32) * 4 = 4224
    bytes_per_token = 4224
    mib = 1024 * 1024

    rows = []
    source_token_totals = {
        "A": split_meta.get("calibration_source", {}).get("A", {}).get("token_count_trunc_max_length"),
        "B": split_meta.get("calibration_source", {}).get("B", {}).get("token_count_trunc_max_length"),
    }

    for key, spec in sorted(subset_specs.items()):
        est_mib = None
        token_frac_source = None
        if spec.token_count is not None:
            est_mib = (spec.token_count * bytes_per_token) / mib
            total_source_tokens = source_token_totals.get(spec.source_split)
            if isinstance(total_source_tokens, int) and total_source_tokens > 0:
                token_frac_source = spec.token_count / total_source_tokens

        rows.append(
            {
                "source_split": spec.source_split,
                "size_label": spec.size_label,
                "frac": spec.frac,
                "seed": spec.seed,
                "subset_count": spec.subset_count,
                "cal_size_tokens": spec.token_count,
                "token_fraction_of_source": token_frac_source,
                "est_layer_cache_mib": est_mib,
                "max_length": max_length,
            }
        )

    _write_json(
        output_root / "preflight_oom_estimate.json",
        {
            "bytes_per_token": bytes_per_token,
            "split_token_totals": source_token_totals,
            "rows": rows,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Figure 6(b) stage-D runner")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT_DEFAULT),
                        help="Artifact root (default: workspace_root/6b)")
    parser.add_argument("--target-snr", type=float, default=27.0,
                        help="Fixed gamma* operating point in dB (default: 27).")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs. Each run uses exactly one GPU (nproc=1).")
    parser.add_argument("--python-bin", type=str, default=sys.executable,
                        help="Python executable for subprocesses (default: current interpreter).")
    parser.add_argument("--cal-max-length", type=int, default=512,
                        help="Calibration max_length forwarded to calibrate.py")
    parser.add_argument("--cal-batch-size", type=int, default=4,
                        help="Calibration batch_size forwarded to calibrate.py")
    parser.add_argument("--max-cal-texts-per-split", type=int, default=20,
                        help="Cap calibration source pool to at most this many texts per split "
                            "before 10/30/100 sampling. Use <=0 to disable.")
    parser.add_argument("--max-cal-tokens-per-split", type=int, default=0,
                        help="Optional token cap (after truncation to --cal-max-length) for each "
                            "calibration source split, applied after text cap. Use <=0 to disable.")
    parser.add_argument("--min-small-texts", type=int, default=4,
                        help="Minimum number of texts for small calibration subsets.")
    parser.add_argument("--min-medium-texts", type=int, default=8,
                        help="Minimum number of texts for medium calibration subsets.")
    parser.add_argument("--online-delay", type=int, default=2,
                        help="MSD online delay forwarded to calibrate.py")
    parser.add_argument("--detail-layer", type=int, default=2,
                        help="Detail layer forwarded to calibrate.py")
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
    parser.add_argument("--smoke-large", action="store_true",
                        help="Execute only dense_A, dense_B, and AtoB_large_seed0 when --execute is set.")
    parser.add_argument("--no-live-progress", action="store_true",
                        help="Disable live subprocess output in terminal (logs are still written).")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    eval_manifests, subset_specs, split_meta = _build_manifests(
        output_root=output_root,
        max_length=args.cal_max_length,
        skip_token_count=args.skip_token_count,
        max_cal_texts_per_split=args.max_cal_texts_per_split,
        max_cal_tokens_per_split=args.max_cal_tokens_per_split,
        min_small_texts=args.min_small_texts,
        min_medium_texts=args.min_medium_texts,
    )

    runs = _build_run_matrix(output_root, subset_specs, eval_manifests)
    run_index = {r.run_id: r for r in runs}

    _write_json(output_root / "split_metadata.json", split_meta)
    _write_json(
        output_root / "run_matrix.json",
        {
            "target_snr": args.target_snr,
            "num_runs": len(runs),
            "runs": [
                {
                    "run_id": r.run_id,
                    "kind": r.kind,
                    "split_id": r.split_id,
                    "eval_split": r.eval_split,
                    "cal_split": r.cal_split,
                    "cal_size_label": r.cal_size_label,
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

    _write_preflight(output_root, subset_specs, split_meta, max_length=args.cal_max_length)

    selected_runs = runs
    if args.smoke_large:
        smoke_ids = {"dense_A", "dense_B", "AtoB_large_seed0"}
        selected_runs = [r for r in runs if r.run_id in smoke_ids]

    if args.run_ids:
        req = set(args.run_ids)
        missing = sorted(req - set(run_index.keys()))
        if missing:
            print(f"Unknown --run-ids: {missing}")
            return 2
        selected_runs = [r for r in selected_runs if r.run_id in req]

    print(f"Prepared Figure 6(b) manifests in: {output_root / 'manifests'}")
    print(f"Prepared run matrix: {output_root / 'run_matrix.json'}")
    print(f"Prepared preflight OOM estimate: {output_root / 'preflight_oom_estimate.json'}")
    print(f"Selected runs: {[r.run_id for r in selected_runs]}")

    if args.execute:
        gpu_ids = _parse_gpu_list(args.gpus)
        use_lite = not args.no_lite
        live_progress = not args.no_live_progress
        logs_dir = output_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        indexed_runs = list(enumerate(selected_runs))

        if gpu_ids:
            runs_by_gpu: dict[str, list[tuple[int, RunSpec]]] = {gpu: [] for gpu in gpu_ids}
            for idx, run in indexed_runs:
                gpu = gpu_ids[idx % len(gpu_ids)]
                runs_by_gpu[gpu].append((idx, run))

            print("Parallel execution plan:")
            for gpu in gpu_ids:
                assigned = runs_by_gpu[gpu]
                run_ids = [r.run_id for _, r in assigned]
                print(f"  GPU {gpu}: {len(assigned)} run(s) -> {run_ids}")

            active_workers = [(gpu, assigned) for gpu, assigned in runs_by_gpu.items() if assigned]

            def _gpu_worker(gpu_id: str, assignments: list[tuple[int, RunSpec]]) -> list[tuple[str, str]]:
                results: list[tuple[str, str]] = []
                for idx, run in assignments:
                    results.append(
                        _execute_one_run(
                            run=run,
                            run_idx=idx,
                            total_runs=len(selected_runs),
                            python_bin=args.python_bin,
                            gpu=gpu_id,
                            target_snr=args.target_snr,
                            max_length=args.cal_max_length,
                            batch_size=args.cal_batch_size,
                            online_delay=args.online_delay,
                            detail_layer=args.detail_layer,
                            force=args.force,
                            use_lite=use_lite,
                            live_progress=live_progress,
                            logs_dir=logs_dir,
                        )
                    )
                return results

            worker_failures: list[tuple[str, str]] = []
            status_rows: list[tuple[str, str]] = []

            with ThreadPoolExecutor(max_workers=len(active_workers)) as executor:
                futures = {
                    executor.submit(_gpu_worker, gpu, assignments): gpu
                    for gpu, assignments in active_workers
                }
                for future in as_completed(futures):
                    gpu = futures[future]
                    try:
                        status_rows.extend(future.result())
                    except Exception as exc:
                        worker_failures.append((gpu, str(exc)))

            if worker_failures:
                detail = "; ".join([f"GPU {gpu}: {msg}" for gpu, msg in worker_failures])
                raise RuntimeError(f"One or more GPU workers failed: {detail}")

            ok_count = sum(1 for _, status in status_rows if status == "ok")
            skip_count = sum(1 for _, status in status_rows if status == "skipped")
            print(f"Execution summary: ok={ok_count}, skipped={skip_count}")
        else:
            for idx, run in indexed_runs:
                _execute_one_run(
                    run=run,
                    run_idx=idx,
                    total_runs=len(selected_runs),
                    python_bin=args.python_bin,
                    gpu=None,
                    target_snr=args.target_snr,
                    max_length=args.cal_max_length,
                    batch_size=args.cal_batch_size,
                    online_delay=args.online_delay,
                    detail_layer=args.detail_layer,
                    force=args.force,
                    use_lite=use_lite,
                    live_progress=live_progress,
                    logs_dir=logs_dir,
                )

    _summarize_results(output_root=output_root, runs=runs, target_snr=args.target_snr)
    print(f"Summary JSON: {output_root / 'figure6b_results.json'}")
    print(f"Summary CSV : {output_root / 'figure6b_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
