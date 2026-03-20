# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simulation of a custom Compute-in-Memory (CiM) dot-product hardware unit for LLM inference, using **MSD-first (Most Significant Digit) digit-pipelined arithmetic with BSD (Binary Signed-Digit) representation**. The simulation is implemented as modifications to the Qwen3 model in HuggingFace Transformers, evaluated via perplexity on WikiText-2.

Target model: **Qwen3-0.6B** (at `../Qwen3-0.6B/`).

## Repository Layout

This repo (`onlinearith/`) contains evaluation and calibration scripts. The actual model implementation lives in a sibling directory:

**Modified Transformers source** (at `../transformers/src/transformers/models/qwen3/`):
- `modular_qwen3.py` — Main implementation (edit here; `modeling_qwen3.py` is auto-generated from this)
- `configuration_qwen3.py` — Config with MXFP/MSD fields
- `calibration_msd.py` — Offline budget calibration utility
- `msd_perf_stats.py` — Hierarchical inference performance statistics

**This directory** — evaluation scripts:
- `ppltest.py` — Single-setup PPL evaluation with window-level data parallelism
- `ppl_batch.py` — All-setup batch PPL evaluation with setup-level parallelism
- `calibrate.py` — Multi-format MSD budget calibration (wraps `calibration_msd.py`)
- `calibrate_base.py` — Baseline calibration for sparsification
- `experiment_config.py` — Central config: all setup definitions (`SETUPS`), field defaults (`BASELINE_CONFIG`), config application/snapshotting utilities
- `dist_utils.py` — Lightweight distributed helpers (NCCL init, auto torchrun relaunch, free port finder)
- `test_distributed.py` — Multi-GPU infrastructure verification tests
- `test_mxfp8linear.py` — Unit tests for MXFP8/6/4Linear layers, MSD truncation, NAF conversion
- `calibration_viz.py` — Diagnostic charts from calibration JSON
- `perf_viz.py` — Performance statistics visualization
- `visualization.py` — Benchmark results visualization
- `benchmarktest.py` — MMLU/GSM8K benchmark via lm-eval

## Environment & Running

```bash
cd /home/xzj/coding/onlinearith
source /home/xzj/coding/.venv3_10/bin/activate
```

### Key Commands

```bash
# List all 21 predefined setups
python ppltest.py --list

# Single PPL evaluation (multi-GPU)
python ppltest.py --nproc 8 --setup 6
python ppltest.py --nproc 4 --gpus 4,5,6,7 --setup 6

# Light mode for quick testing
python ppltest.py --setup 6 --limit-samples 100

# Batch PPL across all setups
python ppl_batch.py --nproc 8
python ppl_batch.py --nproc 8 --only 2 6 10

# Calibration (all 4 MXFP formats on 4 GPUs)
python calibrate.py --nproc 4
python calibrate.py --setup 1  # single format

# PPL with calibrated budgets
python ppltest.py --nproc 8 --setup 6 --calibration calibration_MXFP8.json

# Tests
python test_mxfp8linear.py
python test_distributed.py                          # single-process tests
torchrun --nproc_per_node=2 test_distributed.py     # NCCL tests
```

### GPU Selection

`--nproc` must equal the count of `--gpus` IDs. Free port is auto-detected. Use `nvidia-smi` to check idle GPUs on the shared server.

## Architecture Concepts

### MXFP Quantization Formats
Four block-quantized formats (block_size=32): MXFP8 (E4M3), MXFP6 (E2M3 or E3M2), MXFP4 (E2M1). Controlled by `use_mxfp8/6/4` flags in config. Mutually exclusive.

### MSD Truncation (cycle budget system)
The core simulation: MSD-first digit-pipelined dot-product with early termination after B clock cycles. Three-tier budget resolution:
- **Tier A** — Uniform: all channels use `msd_cycle_budget` (default)
- **Tier B** — Calibrated: per-channel budgets from binary search over target SNR (via `calibrate.py`)
- **Tier C** — Dynamic: runtime adjustment based on activation+weight scale exponents (always added on top)

Budget composition: `B_final[n,j] = B_base[j] + ΔB[n,j]`

### Deep Pipeline
This feature is currently abandoned, ignore deep pipeline codes and setups.

### Setup Numbering (shared across scripts)
Defined in `experiment_config.py` `SETUPS` list. 21 setups total:
- #1: FP16 baseline
- #2-5: MXFP only (no MSD)
- #6-9: MXFP + MSD B=16
- #10-14: MXFP8 budget sweep (B=8,12,20,24,32)
- #15-19: MXFP4 budget sweep
- #20-21: Deep pipeline

### Parallelism
- `ppltest.py`: Window-level parallelism (578 windows sharded across GPUs, NCCL all_reduce)
- `ppl_batch.py`: Setup-level parallelism (round-robin, no inter-rank communication)
- `calibrate.py`: Format-level parallelism (round-robin, no NCCL)

### Result Files
- PPL: `ppl_results_{tag}.json` — named from setup tag, `_calib` suffix when using `--calibration`
- Calibration: `calibration_{tag}.json` — contains `msd_calibration_data` for injection into PPL runs
- Batch summary: `ppl_batch_summary.json`

## Key Implementation Details

- `experiment_config.py` is the single source of truth for config fields and setup definitions — import from here, don't duplicate
- `_msd_truncate()` in `modular_qwen3.py` is the core truncation function used at both inference and calibration time
- `MSDComputeContext._active` is a class-level singleton, process-safe under torchrun (separate address spaces)
- Model path: `../Qwen3-0.6B` (relative to this directory)
- `dist_utils.py` is the shared infrastructure for distributed mode
