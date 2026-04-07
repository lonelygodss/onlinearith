# MSD-First Time-Domain Truncated Dot-Product Simulation

Simulation of a custom Compute-in-Memory (CiM) dot-product hardware unit for LLM inference,
using Most Significant Digit (MSD) first, digit-pipelined arithmetic with Binary Signed-Digit (BSD)
representation. Implemented as modifications to the Qwen3 model in the HuggingFace Transformers library.

**Design documents:**
- [260228-MSD-plan-request.md](260228-MSD-plan-request.md) — Hardware specification
- [260228-MSD-plan-reply.md](260228-MSD-plan-reply.md) — Implementation plan
- [260227-proposal_summary.md](260227-proposal_summary.md) — Proposal summary

**Modified source files:**
- `transformers/src/transformers/models/qwen3/modular_qwen3.py` — Main implementation (edit here)
- `transformers/src/transformers/models/qwen3/modeling_qwen3.py` — Auto-generated from modular
- `transformers/src/transformers/models/qwen3/configuration_qwen3.py` — Config with MSD fields
- `transformers/src/transformers/models/qwen3/calibration_msd.py` — Offline budget calibration utility
- `transformers/src/transformers/models/qwen3/msd_perf_stats.py` — Hierarchical inference performance statistics

**Evaluation scripts (multi-GPU via `--nproc` auto-launch or torchrun):**
- `onlinearith/ppltest.py` — Single-setup PPL evaluation with window-level data parallelism (Supports `--limit-samples N` for light mode fast runs)
- `onlinearith/ppl_batch.py` — All-setup batch PPL evaluation with setup-level parallelism
- `onlinearith/dist_utils.py` — Lightweight distributed helpers (NCCL init, all_reduce, barrier)
- `onlinearith/test_distributed.py` — Verification test for multi-GPU infrastructure

---

## 1. Available Setups

All configurations are controlled via `Qwen3-0.6B/config.json`. The table below lists all meaningful test setups.

### Tier 1: Baseline & MX-Only (no MSD)

| # | Setup | Key Config Fields | Description |
|---|-------|-------------------|-------------|
| 1 | FP16 Baseline | `use_mxfp8/6/4: false` | Standard fp16 inference, no quantization |
| 2 | MXFP8 only | `use_mxfp8: true` | MX FP8 E4M3 block quantization (bs=32) |
| 3 | MXFP6 E2M3 only | `use_mxfp6: true, mxfp6_format: "e2m3"` | MX FP6 E2M3 (max=7.5) |
| 4 | MXFP6 E3M2 only | `use_mxfp6: true, mxfp6_format: "e3m2"` | MX FP6 E3M2 (max=28.0) |
| 5 | MXFP4 only | `use_mxfp4: true` | MX FP4 E2M1 (max=6.0) |

### Tier 2: MSD Truncation (default uniform budget)

| # | Setup | Key Config Fields | Description |
|---|-------|-------------------|-------------|
| 6 | MXFP8 + MSD B=16 | `use_mxfp8: true, use_msd_truncation: true, msd_cycle_budget: 16` | Default budget |
| 7 | MXFP6 E2M3 + MSD B=16 | `use_mxfp6: true, ..., use_msd_truncation: true, msd_cycle_budget: 16` | Default budget |
| 8 | MXFP6 E3M2 + MSD B=16 | `use_mxfp6: true, mxfp6_format: "e3m2", use_msd_truncation: true, ...` | Default budget |
| 9 | MXFP4 + MSD B=16 | `use_mxfp4: true, use_msd_truncation: true, msd_cycle_budget: 16` | Default budget |

### Tier 3: Budget Sweep

| # | Setup | Budget Values | Description |
|---|-------|---------------|-------------|
| 10 | MXFP8 + MSD sweep | B = 8, 12, 16, 20, 24, 32 | Characterize PPL vs budget curve |
| 11 | MXFP4 + MSD sweep | B = 8, 12, 16, 20, 24, 32 | Same sweep on lower-precision format |

### Tier 4: Deep Pipeline

| # | Setup | Key Config Fields | Description |
|---|-------|-------------------|-------------|
| 12 | MXFP8 + MSD + Pipeline | `..., msd_deep_pipeline: true, msd_pipeline_precision_loss: 2` | Precision loss through MLP stages |
| 13 | MXFP4 + MSD + Pipeline | `..., msd_deep_pipeline: true, msd_pipeline_precision_loss: 2` | Same on FP4 |

### Tier 5: Calibrated Budget

| # | Setup | Description |
|---|-------|-------------|
| — | Any MSD setup + `--calibration` | Per-channel B_base from calibration (see Section 5). Use `ppltest.py --setup <ID> --calibration <file>`. |

---

## 2. Config.json Reference

### Baseline (FP16, no quantization)

```json
{
  "use_mxfp8": false,
  "use_mxfp6": false,
  "use_mxfp4": false
}
```

### MXFP8 only (no MSD)

```json
{
  "use_mxfp8": true,
  "use_mxfp6": false,
  "use_mxfp4": false
}
```

### MXFP6 E2M3 only

```json
{
  "use_mxfp8": false,
  "use_mxfp6": true,
  "mxfp6_format": "e2m3",
  "use_mxfp4": false
}
```

### MXFP6 E3M2 only

```json
{
  "use_mxfp8": false,
  "use_mxfp6": true,
  "mxfp6_format": "e3m2",
  "use_mxfp4": false
}
```

### MXFP4 only

```json
{
  "use_mxfp8": false,
  "use_mxfp6": false,
  "use_mxfp4": true
}
```

### MXFP8 + MSD (budget=16, default)

```json
{
  "use_mxfp8": true,
  "use_mxfp6": false,
  "use_mxfp4": false,
  "use_msd_truncation": true,
  "msd_cycle_budget": 16,
  "msd_online_delay": 2,
  "msd_budget_dynamic_scale": 0.0,
  "msd_budget_dynamic_threshold": 0.0,
  "msd_budget_dynamic_mode": "linear"
}
```

### MXFP4 + MSD (budget=16, default)

```json
{
  "use_mxfp8": false,
  "use_mxfp6": false,
  "use_mxfp4": true,
  "use_msd_truncation": true,
  "msd_cycle_budget": 16,
  "msd_online_delay": 2
}
```

### MXFP8 + MSD + Deep Pipeline

```json
{
  "use_mxfp8": true,
  "use_mxfp6": false,
  "use_mxfp4": false,
  "use_msd_truncation": true,
  "msd_cycle_budget": 16,
  "msd_online_delay": 2,
  "msd_deep_pipeline": true,
  "msd_pipeline_precision_loss": 2
}
```

### Budget Sweep (change `msd_cycle_budget` for each run)

For a budget sweep, keep all other fields the same and only change `msd_cycle_budget`:

```json
{
  "use_mxfp8": true,
  "use_msd_truncation": true,
  "msd_cycle_budget": 8
}
```

Suggested sweep values: **8, 12, 16, 20, 24, 32**

---

## 3. Budget System Architecture

The cycle budget *B* determines how many clock cycles each output-channel accumulation runs before early termination. The system uses a **three-tier resolution** that combines offline calibration with runtime dynamic adjustment:

### Tier A: Uniform Budget (default)

When no calibration data is loaded, every output channel in every layer uses the same global budget:

> B_base[j] = `msd_cycle_budget`  for all j

This is the simplest mode — set `msd_cycle_budget` in config.json and all channels get that budget. Used in Tier 2 and Tier 3 setups.

### Tier B: Calibrated Budget (per-channel, offline)

When `msd_calibration_data` is present (produced by `calibrate.py`), each output channel gets its own B_base determined by binary search over a target SNR:

> B_base[j] = calibrated_budget[j]  (per layer, per output channel)

**Calibration data scope.** The budget search operates on *all activation samples collected during the calibration forward passes*, concatenated across batches.  Concretely, the total sample count fed to the per-channel search is:

> N_cal = `--num-texts` × (tokens per text after padding/truncation to `--max-length`)

With the defaults (`--num-texts 20`, `--max-length 512`, `--batch-size 4`), this yields N_cal ≈ 10 000 token positions per layer. The search computes SNR statistics **averaged over all N_cal samples**, so more data gives a more representative budget but costs more GPU memory.

To change the calibration scope:
- `--num-texts N` — number of paragraphs drawn from WikiText-2 validation (default 20)
- `--max-length L` — max tokens per paragraph after tokenizer truncation (default 512)
- `--batch-size B` — controls how many texts are processed per forward pass (default 4); does **not** change the total data volume, only the per-pass memory footprint

**Memory implications.**  During the budget binary search, the dominant tensor is the 4D element-wise product `(N_cal, chunk, nb, bs)` where `chunk` is automatically sized to keep the tensor under 256 MiB (the `_CAL_CHUNK_TARGET_BYTES` constant in `calibration_msd.py`). However, the *collected block data* for each layer — `x_q (N_cal, nb, bs)` and `x_scales (N_cal, nb)` — is held on GPU until that layer's search finishes.  For the defaults (N_cal ≈ 10k, nb=32, bs=32, float32), this is about 40 MB per layer.  If you increase `--num-texts` or `--max-length` significantly (e.g., 200 texts × 2048 tokens → N_cal ≈ 400k), the per-layer block data grows to ~1.6 GB and the output-chunked search loop will still stay within the 256 MiB chunk limit, but you may need to lower `--batch-size` to avoid OOM during the forward passes themselves.

**Algorithm** (in `calibration_msd.py`):
1. Run forward passes over calibration data with MSD disabled (exact MXFP mode)
2. Hook into each MXFP layer to collect block-quantized activations (x_q, x_scales) and weights (w_q, w_scales)
3. For each output channel *j*, **binary search** over B ∈ [4, 48] (12 bisection iterations):
   - Maintain `lo` (too small) and `hi` (sufficient) bounds, starting at lo=4, hi=48
   - At each iteration: `mid = floor((lo + hi) / 2)`
   - Compute truncated dot-product at budget=`mid` using the same `_msd_truncate` function used at inference
   - Measure per-channel SNR: SNR_j = 10 · log10(signal_power_j / noise_power_j)  where signal_power = mean(exact²) and noise_power = mean((exact − truncated)²)
   - If SNR_j ≥ `--target-snr` → budget is sufficient → `hi = mid` (search **downward** for a smaller sufficient budget)
   - If SNR_j < `--target-snr` → budget is too small → `lo = mid + 1` (push upward)
   - After 12 iterations, return `hi` — the **smallest budget that meets the SNR target**

   This is a standard bisection that converges on the minimum sufficient budget because `hi` only decreases when the current midpoint already satisfies the SNR constraint. The invariant is: `hi` is always a known-good budget, `lo` is always one above a known-bad budget. After convergence, `hi` is the tightest upper bound, i.e., the minimum budget meeting the target.

4. In a final pass, collect per-channel dynamic range statistics (see Section 5, "Interpreting Results") at the converged budget

Channels with high activation dynamic range or large weight magnitudes get higher budgets; quiet channels get lower budgets — enabling hardware-native budget differentiation.

### Tier C: Dynamic Adjustment (runtime, always-on)

On top of whichever B_base is active (uniform or calibrated), a runtime dynamic delta is added based on the actual combined activation+weight scales observed during inference:

> B_final[n,j] = B_base[j] + ΔB[n,j]

Where:
- E_combined[n,j] = max over blocks *b* of floor(log2(x_scale[n,b] · w_scale[j,b]))  — per (sample, output-channel)
- **Linear mode** (default): ΔB = α · max(0, E_combined − E_threshold)
- **Step mode**: ΔB = α if E_combined > E_threshold, else 0

With the default config (α = 0.0, E_threshold = 0.0), ΔB equals the combined scale exponent, giving extra cycles to high-magnitude channels at runtime. This captures sample-dependent dynamic range that offline calibration cannot predict.

### How the Tiers Compose

```
Tier A (uniform)  ──OR──  Tier B (calibrated)
      │                     │
      └─── B_base[j] ──────┘
          │
        + Tier C (dynamic Δ)  ← always applied
          │
      B_final[n,j]
```

The `_resolve_channel_budgets()` method in `_MXFPLinearBase` implements this composition:
1. Load B_base from calibration data (Tier B) or uniform config (Tier A)
2. Compute delta_b from combined log2 scales (Tier C)
3. Return B_final = B_base + delta_b with shape `(N, out)` — per-sample, per-output-channel

### Config Fields Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_msd_truncation` | bool | false | Master switch for MSD simulation |
| `msd_cycle_budget` | int | 16 | Global default cycle budget B_base (Tier A) |
| `msd_online_delay` | int | 2 | MSD multiplier online delay δ (digits before first valid product) |
| `msd_budget_dynamic_scale` | float | 0.0 | α for Tier C dynamic adjustment |
| `msd_budget_dynamic_threshold` | float | 0.0 | E_threshold for Tier C dynamic adjustment |
| `msd_budget_dynamic_mode` | str | "linear" | `"linear"` or `"step"` for Tier C mode |
| `msd_deep_pipeline` | bool | false | Enable MSD streaming through MLP (gate→silu→×up→down) |
| `msd_pipeline_precision_loss` | int | 2 | Digits of precision lost per pipeline stage |
| `msd_calibration_data` | dict/null | null | Per-layer per-channel calibrated B_base (Tier B) |
| `msd_chunk_target_mib` | int | 512 | Target MiB for largest MSD intermediate tensor per output chunk. Actual peak is ~3x this (multiple coexisting temporaries). Reduce for smaller GPUs; increase for throughput on larger GPUs. |

---

## 4. Running PPL Tests

Both `ppltest.py` and `ppl_batch.py` support **multi-GPU via `--nproc`** (auto-launch) and **single-GPU fallback**.
When running multi-GPU, each GPU loads its own model copy (~1.2 GB for Qwen3-0.6B, well
within 32 GB per RTX 5090) and processes its shard of the work.

### Quick Start (Multi-GPU)

```bash
cd /home/xzj/coding/onlinearith
source /home/xzj/coding/.venv3_10/bin/activate

# Single PPL evaluation — 8 GPUs, ~7-8x speedup
python ppltest.py --nproc 8 --setup 2

# Full 21-config sweep — 8 GPUs, ~3x speedup
python ppl_batch.py --nproc 8

# Subset of configs on 8 GPUs
python ppl_batch.py --nproc 8 --only 2 6 10

# Use only specific GPUs (e.g. GPUs 4,5,6,7 — when 0-3 are busy)
python ppltest.py --nproc 4 --gpus 4,5,6,7 --setup 6
python ppl_batch.py --nproc 4 --gpus 4,5,6,7

# List all predefined setups (works for both scripts)
python ppltest.py --list
python ppl_batch.py --list

# Single-GPU fallback (unchanged behavior, no --nproc needed)
python ppltest.py --setup 1
python ppl_batch.py

# Single specific GPU (e.g. only GPU 3 is free)
python ppltest.py --gpus 3 --setup 6
python ppl_batch.py --gpus 3
```

### GPU Selection

On a shared server, some GPUs may be in use by other users. Use `--gpus` to
restrict which physical GPUs are used:

```bash
# 1. Check which GPUs are idle
nvidia-smi   # or nvtop

# 2. Suppose GPUs 4, 5, 7 are free — use only those
python ppl_batch.py --nproc 3 --gpus 4,5,7

# For single-GPU mode, just specify one GPU
python ppltest.py --gpus 3
```

**Important:** `--nproc` must **equal** the number of GPU IDs in `--gpus`.
A mismatch will produce a clear error message from torchrun.

`--nproc` finds a free rendezvous port automatically, so it is safe to run
multiple jobs in parallel on the same machine without `EADDRINUSE` errors.

The `--gpus` flag sets `CUDA_VISIBLE_DEVICES` internally before any CUDA
context is created. You can also set the environment variable directly if
you prefer:

```bash
# Equivalent to --nproc 3 --gpus 4,5,7
CUDA_VISIBLE_DEVICES=4,5,7 python ppl_batch.py --nproc 3
```

If both `--gpus` and `CUDA_VISIBLE_DEVICES` are provided, `--gpus` takes
precedence (it overwrites the env var).

If you prefer to invoke torchrun manually, you must pick a free port yourself:

```bash
torchrun --nproc_per_node=3 --master-port=29501 ppl_batch.py --gpus 4,5,7
```

### Parallelism Strategy

| Script | Parallelism | How It Works |
|--------|-------------|-------------|
| `ppltest.py` | **Window-level** | 578 sliding windows are sharded round-robin across GPUs. Partial NLL sums are aggregated via NCCL `all_reduce`. Only rank 0 saves the JSON. |
| `ppl_batch.py` | **Setup-level** | 21 config setups are partitioned round-robin across GPUs. Each GPU evaluates its assigned setups independently, writing result files directly. Rank 0 prints the summary table after all ranks finish. |

The `MSDComputeContext._active` class-level singleton is **process-safe** under `--nproc` / torchrun
(each rank is a separate Python process with its own address space).

### Procedure (Single Setup)

`ppltest.py` supports the same 21 predefined setups as `ppl_batch.py`, selected
via `--setup ID`. The setup ID numbering is identical across both scripts.

1. **List available setups:**
   ```bash
   python ppltest.py --list
   ```

2. **Run a setup by ID:**
   ```bash
   # Multi-GPU (recommended)
   python ppltest.py --nproc 8 --setup 6            # MXFP8 + MSD B=16

   # Use specific GPUs only
   python ppltest.py --nproc 4 --gpus 4,5,6,7 --setup 6

   # Single-GPU fallback
   python ppltest.py --setup 6
   ```
   The output file is auto-named `ppl_results_{tag}.json` from the setup tag.
   Use `--output custom_name.json` to override.

3. **With calibrated budgets (--calibration):**
   ```bash
   # First run calibration (see Section 5):
   python calibrate.py --setup 1

   # Then run PPL with calibrated budgets injected:
   python ppltest.py --nproc 8 --setup 6 --calibration calibration_MXFP8.json

   # Output auto-named: ppl_results_MXFP8_MSD_B16_calib.json
   # Override with --output if desired:
   python ppltest.py --setup 6 --calibration calibration_MXFP8.json --output custom.json
   ```
   The `--calibration` flag loads per-channel budgets from a calibration JSON file
   (produced by `calibrate.py`) and injects them into the model config's
   `msd_calibration_data` field. This replaces the uniform `msd_cycle_budget`
   with per-channel Tier B budgets (see Section 3). Requirements:
   - Must be used with `--setup` (not raw config.json mode)
   - The selected setup should have `use_msd_truncation: true` (if not, it will be auto-enabled with a warning)
   - Calibration data is cleared from config after the run completes

4. **Advanced: custom config.json (no --setup):**
   If `--setup` is omitted, `ppltest.py` uses whatever config is in
   `Qwen3-0.6B/config.json` and saves to the hardcoded `RESULTS_OUT` constant.
   ```bash
   # Edit config.json first, then:
   python ppltest.py --nproc 8
   ```

5. **Expected runtime:**
   - Single GPU: ~25-28 min per run (578 windows, ~180 tokens/sec)
   - 8× RTX 5090: ~3-4 min per run (~7-8× speedup)

### Naming Convention for Result Files

Use this pattern: `ppl_results_{FORMAT}[_MSD_B{budget}][_pipeline][_calib].json`

Examples:
- `ppl_results_MXFP8.json` — MXFP8 only (already exists)
- `ppl_results_MXFP8_MSD_B16.json` — MXFP8 + MSD, budget=16
- `ppl_results_MXFP8_MSD_B8.json` — MXFP8 + MSD, budget=8
- `ppl_results_MXFP4_MSD_B16.json` — MXFP4 + MSD, budget=16
- `ppl_results_MXFP8_MSD_B16_pipeline.json` — MXFP8 + MSD + deep pipeline
- `ppl_results_MXFP8_MSD_B16_calib.json` — MXFP8 + MSD + calibrated budgets (via `--calibration`)

### Suggested Test Order

For a complete characterization, run in this order:

1. **Budget sweep on MXFP8** (6 runs):
   - B=8, B=12, B=16, B=20, B=24, B=32
   - This gives the PPL-vs-budget curve for the highest-precision MX format

2. **Default budget across formats** (4 runs):
   - MXFP8+MSD B=16, MXFP6_E2M3+MSD B=16, MXFP6_E3M2+MSD B=16, MXFP4+MSD B=16
   - Compare MSD impact across formats

3. **Deep pipeline** (2 runs):
   - MXFP8+MSD+pipeline B=16, MXFP4+MSD+pipeline B=16
   - Measure pipeline precision loss impact

4. **Calibrated** (1 run after calibration):
   - Run calibration first (see Section 5), then:
   ```bash
   python ppltest.py --nproc 8 --setup 6 --calibration calibration_MXFP8.json
   ```

### Batch Mode (All Setups in One Run)

To run **all setups automatically** without manual config editing, use `ppl_batch.py`.
It loads the model once per GPU, patches config in-memory for each setup, and saves results
to separate JSON files. Existing results are skipped by default (resume-safe).

```bash
cd /home/xzj/coding/onlinearith
source /home/xzj/coding/.venv3_10/bin/activate

# List all available setups (21 total)
python ppl_batch.py --list

# Run ALL setups on 8 GPUs (~1.5 hours vs ~10 hours single-GPU)
python ppl_batch.py --nproc 8

# Run only specific setups by ID
python ppl_batch.py --nproc 8 --only 1 6 10 14

# Run on specific GPUs (e.g. 4,5,6,7 are idle)
python ppl_batch.py --nproc 4 --gpus 4,5,6,7

# Combine GPU selection with setup filtering
python ppl_batch.py --nproc 4 --gpus 4,5,6,7 --only 1 6 10

# Re-run setups even if result files exist
python ppl_batch.py --nproc 8 --force

# Resume after interruption (skips completed ones)
python ppl_batch.py --nproc 8

# Single-GPU fallback (all of the above also work with plain python)
python ppl_batch.py

# Single specific GPU
python ppl_batch.py --gpus 3
```

The 21 setups cover:
- **#1**: FP16 baseline
- **#2-5**: MXFP8/6/4 only (no MSD)
- **#6-9**: MXFP8/6/4 + MSD at default budget B=16
- **#10-14**: MXFP8 budget sweep (B=8,12,20,24,32; B=16 reuses #6)
- **#15-19**: MXFP4 budget sweep (B=8,12,20,24,32; B=16 reuses #9)
- **#20-21**: Deep pipeline (MXFP8, MXFP4)

All result files are named `ppl_results_{tag}.json` and saved in the `onlinearith/` directory.
At the end of each batch run, a consolidated **`ppl_batch_summary.json`** is written by rank 0
containing all metrics, wall times, and run metadata in a single file for easy post-processing.

**Tip:** Run inside `tmux` or `screen` so it survives SSH disconnections:
```bash
tmux new -s ppl
python ppl_batch.py --nproc 4 --gpus 4,5,6,7
# Ctrl+B then D to detach; tmux attach -t ppl to reconnect
```

---

## 5. Running Calibration

Calibration determines optimal per-channel cycle budgets (B_base) based on actual weight/activation statistics. `calibrate.py` supports **multi-GPU via `--nproc`** (like `ppl_batch.py`) and covers all 4 MXFP formats.

### Available Calibration Setups

| ID | Tag | Description |
|----|-----|-------------|
| 1 | `MXFP8` | MXFP8 (E4M3FN) |
| 2 | `MXFP6_E2M3` | MXFP6 E2M3 |
| 3 | `MXFP6_E3M2` | MXFP6 E3M2 |
| 4 | `MXFP4` | MXFP4 (E2M1) |

### Quick Start

```bash
cd /home/xzj/coding/onlinearith
source /home/xzj/coding/.venv3_10/bin/activate

# List calibration setups
python calibrate.py --list

# Single format on single GPU
python calibrate.py --setup 1

# All 4 formats on 4 GPUs (~1 min total)
python calibrate.py --nproc 4

# Specific GPUs
python calibrate.py --nproc 4 --gpus 4,5,6,7

# Subset of formats
python calibrate.py --nproc 2 --gpus 4,5 --only 1 4

# Custom SNR target (higher = more budget = less error)
python calibrate.py --nproc 4 --gpus 4,5,6,7 --target-snr 40

# Re-run even if result files exist
python calibrate.py --nproc 4 --gpus 4,5,6,7 --force

# Single specific GPU
python calibrate.py --gpus 3 --setup 1
```

Output files are named `calibration_{tag}.json` and saved in `../data/calib-data/{snr}db/`.

### Fixed-Sum Budget Redistribution Optimizer

The standard `snr_min` calibration uses binary search to find minimum per-channel budgets meeting a target SNR. The **fixed-sum optimizer** improves on this by redistributing cycles from low-gain channels to high-gain channels while preserving the total hardware cycle budget, reducing total calibration error without increasing cycles.

```bash
# Standard SNR-min calibration (baseline)
python calibrate.py --setup 1

# Fixed-sum redistribution
python calibrate.py --setup 1 --optimizer fixed_sum

# Multi-GPU batch mode (all 4 formats)
python calibrate.py --nproc 4 --optimizer fixed_sum

# With holdout validation (20% held out)
python calibrate.py --setup 1 --optimizer fixed_sum --holdout-fraction 0.2

# Only calibrate gate_proj layers
python calibrate.py --setup 1 --optimizer fixed_sum --projection-filter gate_proj
```

**Output files:**
- SNR-min mode: `calibration_{tag}.json` (e.g., `calibration_MXFP8.json`)
- Fixed-sum mode: `calibration_{tag}_fixed_sum.json` (e.g., `calibration_MXFP8_fixed_sum.json`)

**Key differences:**
- SNR-min: Each channel independently achieves target SNR with minimum cycles
- Fixed-sum: Redistributes cycles to minimize total error while preserving sum(B)

See [fixed_sum_calibration.md](fixed_sum_calibration.md) for detailed algorithm explanation, validation gates, and expected results.

### Parallelism Strategy

| # GPUs | Behavior |
|--------|----------|
| 1 | Runs all selected formats sequentially on one GPU |
| 2-4 | Formats partitioned round-robin across GPUs |
| >4 | Extra GPUs sit idle (only 4 formats available) |

Uses `init_distributed_lite()` — no NCCL, ranks work independently.

### Calibration Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--target-snr` | 30.0 | Target SNR in dB. Higher = more budget = less truncation error. Try 20, 30, 40. |
| `--num-texts` | 20 | Number of calibration paragraphs from WikiText-2 validation set |
| `--max-length` | 512 | Max token length per calibration sample |
| `--batch-size` | 4 | Batch size for calibration forward passes |
| `--online-delay` | 2 | MSD online delay δ (should match inference config) |
| `--detail-layer` | 2 | Transformer layer index for full per-channel statistics. The 3 MLP projections (gate/up/down) of this layer get channel-wise detail; all others get compact summaries only. |
| `--optimizer` | `snr_min` | Optimization mode: `snr_min` (binary search) or `fixed_sum` (redistribution) |
| `--holdout-fraction` | 0.0 | Fraction of texts to hold out for validation (0.0 = no holdout) |
| `--projection-filter` | None | Only calibrate matching projections (e.g., `gate_proj` or `up_proj,down_proj`) |
| `--curve-window` | 3 | Budget window for error curve computation (fixed_sum only) |

### Expected Runtime

- Single GPU: ~2-5 min per format (depends on `--num-texts` and `--max-length`)
- 4× RTX 5090: ~2-5 min total (all 4 formats in parallel)

### Using Calibration Results with PPL Tests

Use the `--calibration` flag in `ppltest.py` to inject calibrated budgets directly — no manual `config.json` editing needed:

```bash
# Run calibration first (produces calibration_MXFP8.json):
python calibrate.py --setup 1

# Then PPL test with calibrated budgets:
python ppltest.py --nproc 8 --setup 6 --calibration calibration_MXFP8.json

# Output: ppl_results_MXFP8_MSD_B16_calib.json
# The _calib suffix is added automatically when --calibration is used

# With fixed-sum calibration:
python calibrate.py --setup 1 --optimizer fixed_sum
python ppltest.py --nproc 8 --setup 6 --calibration calibration_MXFP8_fixed_sum.json
```

`--calibration` loads the `msd_calibration_data` from the specified JSON file and overrides the setup's uniform budget with per-channel Tier B budgets (see Section 3). The calibration data is automatically cleared from config after the run.

**Note:** `ppl_batch.py` does not support calibration — use `ppltest.py` for calibrated runs.

### Interpreting Results

Each `calibration_{tag}.json` has four main sections:

1. **`global_summary`** — aggregate statistics across all layers
2. **`layer_stats`** — compact per-layer summary (scalars only, ~15 fields per layer)
3. **`channel_detail`** — full per-channel arrays for one specific layer (the detail layer)
4. **`msd_calibration_data`** — per-channel budgets for all layers (used by `ppltest.py --calibration`)

#### Output JSON Structure

```json
{
  "format": "MXFP8",
  "calibration_params": {
    "target_snr_db": 30.0,
    "num_texts": 20,
    "max_length": 512,
    "batch_size": 4,
    "online_delay": 2,
    "detail_layer": 2
  },
  "global_summary": {
    "num_layers": 84,
    "total_channels": 200704,
    "budget_min": 4,
    "budget_max": 32,
    "budget_mean": 16.5,
    "mean_snr": 32.5,
    "min_snr": 30.0,
    "e_combined_mean": -18.1,
    "eff_precision_mean": 10.5,
    "signal_power_db_mean": 0.6,
    "wall_time_sec": 120.0
  },
  "layer_stats": {
    "model.layers.0.mlp.gate_proj": {
      "budget_mean": 7.2, "budget_min": 4, "budget_max": 12,
      "budget_std": 1.5, "budget_p25": 6, "budget_p50": 7, "budget_p75": 8,
      "budget_histogram": {"4": 10, "5": 50, "6": 300, "7": 800, "8": 400},
      "frac_at_min_budget": 0.004, "frac_at_max_budget": 0.0,
      "snr_mean": 32.5, "snr_min": 30.0,
      "e_combined_mean": -18.1, "e_combined_std": 1.2,
      "e_combined_range": [-22.0, -12.0],
      "inter_delay_mean": 2.1, "intra_delay_mean": 2.03,
      "eff_precision_mean": 0.9, "eff_precision_min": 0.0,
      "signal_power_db_mean": 0.6, "signal_power_db_range": [-10.0, 5.0]
    }
  },
  "channel_detail": {
    "detail_layer": 2,
    "model.layers.2.mlp.gate_proj": {
      "budget": [7, 8, 6, ...],
      "snr_at_budget": [31.2, 30.5, 32.0, ...],
      "e_combined_mean": [-17.8, -19.2, ...],
      "e_combined_std": [1.1, 1.3, ...],
      "inter_delay_mean": [2.0, 2.3, ...],
      "eff_precision_mean": [1.2, 0.8, ...],
      "signal_power_db": [-12.3, -8.1, ...]
    },
    "model.layers.2.mlp.up_proj": { "..." : "..." },
    "model.layers.2.mlp.down_proj": { "..." : "..." }
  },
  "msd_calibration_data": {
    "model.layers.0.mlp.gate_proj": [7, 8, 6, ...],
    "...": "...(all 84 layers)"
  }
}
```

#### Layer-wise Summary Statistics (`layer_stats`)

Each layer gets a compact dict of scalar statistics. These are sufficient to understand
cross-layer trends without storing per-channel arrays.

**Budget distribution:**

| Stat | Calculation | Purpose |
|------|-------------|---------|
| `budget_mean` | `mean(B[j])` over all output channels | Average budget assigned to this layer |
| `budget_min` / `budget_max` | `min(B[j])` / `max(B[j])` | Budget extremes |
| `budget_std` | `std(B[j])` | Budget spread — large std means diverse channel requirements |
| `budget_p25` / `budget_p50` / `budget_p75` | 25th, 50th, 75th percentile of `{B[j]}` | Distribution shape (skew detection) |
| `budget_histogram` | Count of channels at each discrete budget value, as `{"B": count}` | Full distribution in compact form |
| `frac_at_min_budget` | Fraction of channels where `B[j] = 4` (search lower bound) | Detects saturation at minimum — these channels may not need as much budget as they got |
| `frac_at_max_budget` | Fraction of channels where `B[j] = 48` (search upper bound) | Detects saturation at maximum — these channels may need *more* budget than the search range allows |

**SNR validation:**

| Stat | Calculation | Purpose |
|------|-------------|---------|
| `snr_mean` | `mean(SNR[j])` over all output channels | Average calibration quality |
| `snr_min` | `min(SNR[j])` | Worst-case channel — should be >= `target_snr_db` |

Where:

> SNR[j] = 10 · log10( mean_n(exact[j]²) / mean_n((exact[j] − truncated[j])²) )

is the signal-to-noise ratio at the converged budget for output channel `j`. Here `exact[j]` is the full-precision block-scaled dot-product result and `truncated[j]` is the result after MSD truncation at budget `B[j]`. The expectation `mean_n` averages over all N_cal calibration token positions.

**Combined exponent:**

| Stat | Calculation | Purpose |
|------|-------------|---------|
| `e_combined_mean` | `mean_j( mean_n( E_combined[n,j] ) )` | Mean dynamic range across channels |
| `e_combined_std` | Mean of per-channel standard deviations of `E_combined` | Temporal variability of dynamic range |
| `e_combined_range` | `[min(E_combined), max(E_combined)]` over all (n, j) | Global dynamic range envelope |

Where:

> E_combined[n,j] = max over blocks b of floor(log2(x_scale[n,b]) + log2(w_scale[j,b]))

is the combined activation+weight scale exponent for sample `n`, output channel `j`. The max is taken over blocks `b`, giving the dominant block's magnitude. This is the primary driver of budget assignment — channels with higher `E_combined` produce larger intermediate results and need more precision digits to maintain accuracy.

**Delays:**

| Stat | Calculation | Purpose |
|------|-------------|---------|
| `inter_delay_mean` | `mean` over all (j, n, b) of `inter_delay[n,j,b]` | Average alignment cost from block scale differences |
| `intra_delay_mean` | `mean` over all (n, b, k) of `intra_delay[n,b,k]` | Average element-level exponent spread (scalar, same for all channels) |

Where:
- `inter_delay[n,j,b]`: the inter-block alignment delay, where each block's combined scale may differ and the MSD pipeline must align all blocks to the dominant block, costing delay cycles for smaller blocks:

  > inter_delay[n,j,b] = E_max[n,j] − floor(log2(x_scale[n,b] · w_scale[j,b]))

- `intra_delay[n,b,k]`: the intra-block delay from element-level activation exponent differences within each block; elements with smaller magnitudes start producing significant digits later:

  > intra_delay[n,b,k] = e_max[n,b] − floor(log2(|x_q[n,b,k]|))

**Effective precision:**

| Stat | Calculation | Purpose |
|------|-------------|---------|
| `eff_precision_mean` | `mean` over all (j, n, b, k) of `p_eff[n,j,b,k]` | Average useful precision after delay overhead |
| `eff_precision_min` | `min` over all (j, n, b, k) of `p_eff[n,j,b,k]` | Worst-case precision across all elements |

Where:

> p_eff[n,j,b,k] = max(0, B[j] − inter_delay[n,j,b] − intra_delay[n,b,k] − δ)

is the effective precision — the number of BSD digits actually computed for element `(n,j,b,k)`. Here `δ` is the MSD online delay (default 2). The effective precision represents the *useful computation cycles* remaining after accounting for all three sources of delay. A low `eff_precision_mean` relative to `budget_mean` indicates that delays consume most of the budget.

**Signal power:**

| Stat | Calculation | Purpose |
|------|-------------|---------|
| `signal_power_db_mean` | `mean_j( 10 · log10( mean_n(exact[j]²) ) )` | Mean signal magnitude scale |
| `signal_power_db_range` | `[min, max]` of per-channel signal power in dB | Signal magnitude spread |

This shows the intrinsic magnitude of the exact dot-product results. Channels with high signal power have large outputs and typically correlate with higher budgets since the absolute truncation error must be proportionally small to maintain the SNR target.

#### Channel-wise Detail Statistics (`channel_detail`)

Full per-channel arrays are collected only for the 3 MLP projections (gate/up/down) of the
`--detail-layer` (default: layer 2). This provides detailed diagnostic data without bloating
the JSON (the previous format stored per-channel arrays for all 84 layers, producing ~2M-line files).

| Stat | Shape | Description |
|------|-------|-------------|
| `budget` | `(out,)` | Calibrated `B[j]` per channel (same as `msd_calibration_data` for this layer) |
| `snr_at_budget` | `(out,)` | Actual SNR (dB) at converged budget — primary validation metric |
| `e_combined_mean` | `(out,)` | Per-channel mean `E_combined` — main budget driver |
| `e_combined_std` | `(out,)` | Per-channel temporal variability of `E_combined` |
| `inter_delay_mean` | `(out,)` | Per-channel mean inter-block alignment delay |
| `eff_precision_mean` | `(out,)` | Per-channel mean effective precision |
| `signal_power_db` | `(out,)` | Per-channel signal power in dB |

These arrays enable scatter-plot visualization of channel-level correlations using
`calibration_viz.py` (see below).

### Visualizing Calibration Results

Use `calibration_viz.py` to produce diagnostic charts from any calibration JSON:

```bash
python calibration_viz.py calibration_MXFP8.json
python calibration_viz.py calibration_MXFP8.json --output-dir calib_charts/
python calibration_viz.py calibration_MXFP8.json --no-show
```

Charts produced (saved as PNGs):

| Chart | Scope | What it shows |
|-------|-------|---------------|
| Layer budget overview | All layers | Mean budget per layer with min-max range bars |
| Layer SNR overview | All layers | Mean and min SNR per layer with target line |
| Budget histogram | Detail layer | Distribution of discrete budget values per projection |
| Budget vs e_combined | Detail layer | How combined exponent drives budget assignment |
| Budget vs signal power | Detail layer | How signal magnitude relates to budget |
| SNR distribution | Detail layer | Per-channel SNR histogram with target line |
| Eff. precision vs budget | Detail layer | How much budget translates to useful precision |
| Inter-delay vs e_combined | Detail layer | Alignment cost vs dynamic range |

### To Remove Calibration

When using `--calibration`, cleanup is automatic. For manual config.json usage,
set `msd_calibration_data` to `null` to revert to uniform budgets:
```json
{
  "msd_calibration_data": null
}
```

---

## 5b. Inference Performance Statistics (MSD Hardware Model)

When MSD truncation is active during PPL evaluation, hierarchical performance statistics
are automatically collected and saved in the result JSON under `msd_perf_stats`. These
statistics reflect **actual runtime behaviour** during inference (not calibration-time
properties) and are designed to feed future hardware simulation:

- **Latency** = f(total cycles, budget allocation)
- **Energy** = f(MAC sparsity, block-level gating, effective precision)

The statistics capture both sources of early termination in the MSD-first architecture:

1. **Whole-block early termination** — blocks where ALL partial products have p_eff=0 can be
   entirely gated off in hardware (zero power consumption for that block)
2. **Within-block partial skipping** — individual elements within an active block may still
   have p_eff=0 or reduced precision, saving energy at the bit level

### Hierarchy

The statistics cover five levels, from fine to coarse:

| Level | Granularity | What it measures |
|-------|-------------|------------------|
| **Bit level** | per-element | Effective precision (p_eff) distribution — how many BSD digits each partial product actually computed, both overall and conditional on the element being active |
| **Block level** | per-block (n,j,b) | Zero / partial / full block activation counts, plus mean active element fraction within partial blocks |
| **Channel level** | per-output-channel | Total budget cycles vs effective cycles consumed, utilization ratio |
| **MAC level** | per-element | Total / active / skipped multiply-accumulate operations — the primary energy metric |
| **Global** | whole-model | Aggregate sparsity ratios, mean precision, block breakdown, total MAC counts |

### Layer Summaries vs Channel Detail

To keep result files manageable (the previous format stored per-channel arrays for every layer,
producing multi-megabyte JSON), the statistics use a **two-tier output pattern** matching the
calibration system (Section 5):

- **Layer summaries** (compact scalars) are produced for **every** MXFP layer
- **Per-channel detail arrays** are produced only for layers belonging to a single transformer
  layer index, controlled by `--detail-layer` (default: 2)

Use `--detail-layer` in `ppltest.py` to select which layer gets full channel-level detail:

```bash
python ppltest.py --nproc 8 --setup 6 --detail-layer 2
```

Layers matching `model.layers.<detail_layer>.` (e.g., `model.layers.2.mlp.gate_proj`,
`model.layers.2.mlp.up_proj`, `model.layers.2.mlp.down_proj`) get both `summary` and
`channel_detail` in the output JSON. All other layers get only `summary`.

### Output Structure

In the PPL result JSON (`ppl_results_*.json`):

```json
{
  "msd_perf_stats": {
    "global": {
      "num_layers": 84,
      "total_macs": 123456789,
      "active_macs": 111111111,
      "mac_sparsity": 0.1000,
      "mean_effective_precision": 12.30,
      "active_p_eff_mean": 13.67,
      "total_budget_cycles": 98765432.0,
      "effective_cycles": 74074074.0,
      "global_utilization": 0.7500,
      "total_blocks": 1234567,
      "zero_blocks": 1234,
      "zero_block_ratio": 0.001,
      "partial_blocks": 234567,
      "partial_block_ratio": 0.190,
      "full_blocks": 998766,
      "full_block_ratio": 0.809
    },
    "per_layer": {
      "model.layers.0.mlp.gate_proj": {
        "summary": {
          "bit_level": {
            "p_eff_mean": 12.30,
            "p_eff_std": 4.50,
            "active_p_eff_mean": 13.67,
            "p_eff_histogram": {
              "bin_labels": ["0", "1-4", "5-8", "9-12", "13-16", "17-24", "25-32", "33+"],
              "counts": [12345, 5000, 8000, 20000, 30000, 15000, 5000, 1000]
            }
          },
          "block_level": {
            "total_blocks": 50000,
            "zero_block_ratio": 0.001,
            "partial_block_ratio": 0.190,
            "full_block_ratio": 0.809,
            "partial_block_mean_active_frac": 0.625
          },
          "channel_level": {
            "budget_mean": 16.50,
            "effective_cycles_total": 800000.0,
            "utilization_mean": 0.7500
          },
          "mac_level": {
            "total_macs": 1600000,
            "active_macs": 1440000,
            "mac_sparsity": 0.1000
          }
        }
      },
      "model.layers.2.mlp.gate_proj": {
        "summary": { "...same structure as above..." },
        "channel_detail": {
          "bit_level": {
            "p_eff_mean": [12.3, 11.8, 13.1, "..."],
            "p_eff_std": [4.5, 4.2, 4.8, "..."],
            "active_p_eff_mean": [13.7, 12.9, 14.0, "..."],
            "p_eff_histogram": {
              "bin_labels": ["0", "1-4", "5-8", "..."],
              "counts": [[12, 5, 8, "..."], ["..."]]
            }
          },
          "block_level": {
            "zero_block_count": [10, 5, 15, "..."],
            "partial_block_count": [200, 180, 220, "..."],
            "full_block_count": [800, 820, 780, "..."],
            "zero_block_ratio": [0.01, 0.005, 0.015, "..."],
            "partial_block_ratio": [0.20, 0.18, 0.22, "..."],
            "full_block_ratio": [0.79, 0.815, 0.765, "..."],
            "partial_block_active_frac": [0.63, 0.58, 0.71, "..."]
          },
          "channel_level": {
            "total_budget_cycles": [16500.0, 16200.0, "..."],
            "effective_cycles": [12375.0, 12150.0, "..."],
            "skipped_cycles": [4125.0, 4050.0, "..."],
            "utilization": [0.75, 0.74, "..."]
          },
          "mac_level": {
            "total_elements": [6000, 6000, "..."],
            "zero_elements": [600, 720, "..."],
            "mac_sparsity": [0.10, 0.12, "..."]
          }
        }
      }
    }
  }
}
```

Note the structural difference: `model.layers.0.mlp.gate_proj` has only `summary` (compact
scalars), while `model.layers.2.mlp.gate_proj` (matching `--detail-layer 2`) also has
`channel_detail` with per-output-channel arrays.

### Console Output

During `ppltest.py`, a compact summary is printed to the console:

```
          MSD PERFORMANCE STATISTICS
----------------------------------------------------
  Layers profiled  : 84
  MAC sparsity     : 10.00%
  Active MACs      : 111,111,111 / 123,456,789
  Mean eff. prec.  : 12.30 digits
  Active eff. prec.: 13.67 digits
  Global util.     : 75.00%
  Zero blocks      : 0.10%
  Partial blocks   : 19.00%
  Full blocks      : 80.90%
----------------------------------------------------

  Per-layer summary (detail_layer=2):
  Layer                                     p_eff  util%  mac_sp%  zero_blk%
  -------------------------------------------------------------------------
  model.layers.0.mlp.gate_proj               12.3   75.0   10.00     0.10
  model.layers.0.mlp.up_proj                 11.8   72.5   12.30     0.15
  ...                                          ...   ...     ...      ...
  model.layers.2.mlp.gate_proj               12.5   74.2   10.50     0.12 *
  model.layers.2.mlp.up_proj                 12.1   73.8   11.20     0.14 *
  model.layers.2.mlp.down_proj               13.0   76.1    9.80     0.08 *
  ...                                          ...   ...     ...      ...
  (* = channel detail in JSON for --detail-layer=2)
----------------------------------------------------
```

### Interpreting the Statistics

The statistics are organized around two complementary energy-saving mechanisms:

**Early termination case 1: Whole-block gating (block level)**

When ALL elements in a block (n,j,b) have p_eff=0, the entire block can be power-gated
in hardware — no computation, no switching activity, maximum energy savings per block.

- `zero_block_ratio` — fraction of blocks that are entirely gated off
- `full_block_ratio` — fraction of blocks where every element computes (no gating)
- `partial_block_ratio` — fraction of blocks with mixed activity

**Early termination case 2: Within-block partial skipping (bit + MAC level)**

Even within an active block, individual elements may have p_eff=0 (skipped) or reduced
precision (fewer BSD digits computed). This provides fine-grained energy savings.

- `mac_sparsity` — fraction of all element-level MACs that are completely skipped
  (p_eff=0). Computed as: mac_sparsity = zero_elements / total_elements. This is the
  primary energy saving metric at element granularity.
- `active_macs` — count of elements where p_eff > 0 (actual work done).
  active_macs = total_elements - zero_elements.
- `active_p_eff_mean` — mean effective precision *only for active elements* (where
  p_eff > 0). This separates the "skipped entirely" savings from the "computed but
  with fewer digits" savings. For energy modelling: skipped elements cost zero,
  active elements cost proportional to active_p_eff_mean.
- `partial_block_mean_active_frac` — within partial blocks, the average fraction of
  elements that are active (between 0 and 1). A value of 0.5 means partial blocks
  have half their elements active on average. This metric captures the fine-grained
  sparsity structure within mixed-activity blocks.

**Precision distribution (bit level)**

- `p_eff_mean` — mean effective precision across ALL elements (including zeros).
  Relates to average computation cost per element.
- `p_eff_std` — standard deviation of effective precision.
- `p_eff_histogram` — distribution of effective precision values across 8 bins:
  [0], [1-4], [5-8], [9-12], [13-16], [17-24], [25-32], [33+]. Peaks at "0"
  indicate aggressive early termination; peaks at high bins indicate the budget
  is being fully utilized.

**Cycle accounting (channel level)**

- `budget_mean` — mean cycle budget assigned per dot-product (averaged across
  all output channels and samples for a layer).
- `utilization_mean` — ratio of effective cycles (sum of all p_eff values) to
  total budget cycles allocated. Lower utilization = more early termination =
  greater energy and latency savings.
- `effective_cycles_total` — total "work done" in cycle units across all
  samples and channels for this layer.

**Energy hierarchy summary:**

| Saving mechanism | Metric | Granularity | Hardware mapping |
|-----------------|--------|-------------|------------------|
| Block gating | zero_block_ratio | Block (n,j,b) | Power-gate entire block's MAC array |
| Element skipping | mac_sparsity | Element (n,j,b,k) | Skip individual MAC operation |
| Precision reduction | active_p_eff_mean | Element (active only) | Fewer digit cycles per active MAC |
| Partial block savings | partial_block_mean_active_frac | Block (partial only) | Fraction of MACs active in mixed blocks |
| Cycle utilization | utilization_mean | Channel (j) | Ratio of useful-to-allocated cycles |

### Layer Summary Statistics

Each layer gets a compact dict of scalar statistics under `"summary"`, organized into four
hierarchy levels. These are sufficient to understand cross-layer trends without storing
per-channel arrays.

**Bit level (`summary.bit_level`):**

| Stat | Calculation | Purpose |
|------|-------------|--------|
| `p_eff_mean` | Mean of p_eff over all elements and channels | Average computation cost per element |
| `p_eff_std` | Mean of per-channel p_eff standard deviations | Precision spread |
| `active_p_eff_mean` | Mean of p_eff only for elements where p_eff > 0 | Cost per active element (excludes skipped) |
| `p_eff_histogram` | Element counts in 8 precision bins, aggregated across channels | Distribution shape for this layer |

Where:

> p_eff[n,j,b,k] = max(0, B_final[n,j] - inter_delay[n,j,b] - intra_delay[n,b,k] - online_delay)

is the effective precision (BSD digits computed) for element (n,j,b,k).

The conditional metric `active_p_eff_mean` is useful because it separates two distinct
energy mechanisms: elements with p_eff=0 are *completely skipped* (zero cost), while
elements with p_eff > 0 have a cost proportional to their effective precision. Knowing
both the skip rate (mac_sparsity) and the per-active-element cost (active_p_eff_mean)
enables a more accurate energy model than using the overall mean alone.

**Block level (`summary.block_level`):**

| Stat | Calculation | Purpose |
|------|-------------|--------|
| `total_blocks` | Total blocks processed by this layer | Denominator for ratios |
| `zero_block_ratio` | Fraction of blocks where ALL elements have p_eff=0 | Fully-gatable blocks |
| `partial_block_ratio` | Fraction of blocks with mixed p_eff (some zero, some active) | Blocks with partial savings |
| `full_block_ratio` | Fraction of blocks where ALL elements have p_eff > 0 | Blocks with no element-level gating |
| `partial_block_mean_active_frac` | Mean of (active_count / block_size) across partial blocks | Fine-grained sparsity in mixed blocks |

Block classification:
- A block (n,j,b) with `block_size` elements is **zero** if all `block_size` elements
  have p_eff=0 — the block is entirely skippable in hardware.
- A block is **full** if all `block_size` elements have p_eff > 0 — every element
  must be computed.
- A block is **partial** if some elements have p_eff=0 and others don't — the
  block must be activated, but individual element-level gating can save energy.

The `partial_block_mean_active_frac` metric captures how much work remains within
partial blocks. A value of 0.3 means on average only 30% of elements within partial
blocks are active, so 70% of the energy in those blocks can be saved via element-level
gating. This is the fine-grained complement to the coarse block-level zero_block_ratio.

**Channel level (`summary.channel_level`):**

| Stat | Calculation | Purpose |
|------|-------------|--------|
| `budget_mean` | total_budget_sum / total_blocks -- average budget per dot-product | Cycle allocation efficiency |
| `effective_cycles_total` | Sum of all p_eff values across all elements | Total "work done" for this layer |
| `utilization_mean` | Mean across channels of (effective_cycles / total_budget) | How much of the allocated budget was used |
| `max_budget` | Max of b_final[n,j] across all samples and channels | Worst-case cycles allocated (layer latency indicator) |
| `max_total_delay` | Max of (inter_delay + intra_delay + online_delay) across all (n,j,b,k) | Worst-case start delay before computation begins |

Utilization reflects the combined effect of all delay sources and early termination.
A utilization of 0.6 means 40% of allocated cycles were wasted on delays or early
termination — representing 40% energy savings from the budget's perspective.

**MAC level (`summary.mac_level`):**

| Stat | Calculation | Purpose |
|------|-------------|--------|
| `total_macs` | Total element-level MACs (= total_elements = N * nb * bs * out) | Maximum possible computation |
| `active_macs` | Elements where p_eff > 0 (= total_elements - zero_elements) | Actual computation performed |
| `mac_sparsity` | zero_elements / total_elements | Fraction of MACs completely skipped |

MAC sparsity is the most direct energy metric: each skipped MAC consumes zero switching
energy. The total energy saving from MAC-level sparsity is proportional to
mac_sparsity * energy_per_MAC.

### Channel Detail Statistics

Full per-channel arrays under `"channel_detail"` are collected only for the MLP projections
(gate/up/down) of the `--detail-layer` (default: layer 2). This provides diagnostic data
for channel-level analysis without bloating the JSON.

The channel detail contains the same four hierarchy levels as the summary, but with
per-output-channel arrays instead of scalar aggregates:

| Section | Arrays | Shape |
|---------|--------|-------|
| `bit_level` | p_eff_mean, p_eff_std, active_p_eff_mean, p_eff_histogram | (out,) or (out, 8) |
| `block_level` | zero/partial/full_block_count, ratios, partial_block_active_frac | (out,) |
| `channel_level` | total_budget_cycles, effective_cycles, skipped_cycles, utilization, max_budget, max_total_delay | (out,) |
| `mac_level` | total_elements, zero_elements, mac_sparsity | (out,) |

These arrays enable scatter-plot visualization of channel-level correlations (e.g.,
MAC sparsity vs utilization, p_eff_mean vs budget).

### Visualizing Performance Statistics

Use `perf_viz.py` to produce diagnostic charts from any PPL result JSON that contains
`msd_perf_stats`:

```bash
python perf_viz.py ppl_results_MXFP8_MSD_B16.json
python perf_viz.py ppl_results_MXFP8_MSD_B16_calib.json --output-dir perf_charts/
python perf_viz.py ppl_results_MXFP8_MSD_B16.json --no-show
```

Charts produced (saved as PNGs):

| Chart | Scope | What it shows |
|-------|-------|---------------|
| Layer p_eff overview | All layers | Mean and active effective precision per layer |
| Layer utilization | All layers | Budget utilization percentage per layer |
| Layer MAC sparsity | All layers | Fraction of skipped MACs per layer |
| Layer block breakdown | All layers | Stacked zero/partial/full block ratios |
| Layer max latency | All layers | Max budget and max total delay per layer (latency indicators) |
| Channel p_eff histogram | Detail layer | Distribution of per-channel effective precision |
| Channel utilization histogram | Detail layer | Distribution of per-channel utilization |
| Channel MAC sparsity histogram | Detail layer | Distribution of per-channel MAC sparsity |
| Channel p_eff vs utilization | Detail layer | Correlation between precision and utilization |
| Channel max delay histogram | Detail layer | Distribution of per-channel max delay and max budget |

### Design Intent

These statistics serve a different purpose from the calibration statistics (Section 5):

| Aspect | Calibration Stats (Section 5) | Inference Perf Stats (Section 5b) |
|--------|-------------------------------|-----------------------------------|
| **Phase** | Offline budget search | Runtime inference |
| **Purpose** | Validate budget quality (SNR target met?) | Hardware simulation (latency, energy) |
| **Perspective** | Algorithm quality | Hardware behaviour |
| **Key metrics** | snr_at_budget (dB) | mac_sparsity, utilization, zero_block_ratio |
| **Conversion** | Budget -> SNR | Statistics -> latency/energy (future) |
| **Layer detail** | `--detail-layer` for calibration | `--detail-layer` for ppltest |

The conversion from these raw statistics to actual latency (ns) and energy (pJ) is left for
future work and will depend on the specific hardware implementation (clock frequency, voltage,
gate-level power model). The comprehensive hierarchy (bit -> block -> channel -> MAC -> global)
is designed so that energy models at any granularity can be built from these statistics:

- **Coarse model:** energy ~ (1 - mac_sparsity) * total_macs * energy_per_MAC
- **Medium model:** energy ~ sum over layers of (active_macs * active_p_eff_mean * energy_per_digit)
- **Fine model:** per-block energy = f(block_type, active_count, element_precisions)

---

## 6. Running Unit Tests

```bash
cd /home/xzjnew/coding
source .venv_310/bin/activate
python onlinearith/test_mxfp8linear.py
```

This runs:
- **MXFP format tests:** Shape, bias, reference match, SNR, grids for MXFP8/6/4
- **BSD/NAF truncation tests:** NAF component conversion, digit width, known-answer BSD truncation, bidirectional error, sign symmetry
- **MSD system tests:** Delay computation (inter/intra-block), B=∞ lossless, B=0 zero output, monotonic error decrease
- **Combined-scale budget tests:** Per-output-channel budget differentiation with combined activation+weight scales, linear and step modes
- **Calibration import test:** Verifies calibration module loads correctly

---

## 7. Existing PPL Results (Baseline)

These results were collected on Qwen3-0.6B (1024 hidden, 3072 intermediate, 28 layers) with WikiText-2-raw-v1 test split, max_length=4096, stride=512, fp16.

| Setup | Token PPL | Word PPL | BPB | BPC | Mean NLL |
|-------|-----------|----------|-----|-----|----------|
| MXFP8 only | 17.28 | 34.23 | 0.948 | 0.950 | 2.850 |
| MXFP6 E2M3 only | 17.51 | 34.80 | 0.953 | 0.954 | 2.863 |
| MXFP6 E3M2 only | 17.95 | 35.89 | 0.961 | 0.963 | 2.888 |
| MXFP4 only | 20.68 | 42.78 | 1.008 | 1.010 | 3.029 |

**Key observations:**
- MXFP8 is nearest to fp16 baseline
- MXFP6 E2M3 only marginally worse than MXFP8
- MXFP6 E3M2 slightly worse (larger exponent range trades mantissa precision)
- MXFP4 has noticeable degradation (~20% PPL increase vs MXFP8)

**Note:** No baseline fp16 PPL result was saved previously. You should run a baseline (all `use_mxfp*: false`) as the first test if not already done.

---

## 8. Architecture Overview

### Data Flow (MSD Mode)

```
Input Token IDs
      │
      ▼
  Embedding
      │
      ▼
  ┌─────────────────────────────┐
  │  Decoder Layer (×28)         │
  │                              │
  │  ┌── Self-Attention ──┐     │  ← Standard nn.Linear (no MSD)
  │  │  q/k/v/o_proj      │     │
  │  └────────────────────┘     │
  │           │                  │
  │  ┌── MLP (MSD-aware) ──┐   │
  │  │                      │   │
  │  │  gate_proj ─→ SiLU   │   │  ← MXFP + MSD truncated dot-product
  │  │     ×                 │   │  ← Optional deep pipeline
  │  │  up_proj ─────┘       │   │
  │  │     │                 │   │
  │  │  down_proj            │   │
  │  └──────────────────────┘   │
  └─────────────────────────────┘
      │
      ▼
  LM Head → Logits
```

### MSD Truncation Algorithm

For each MLP linear layer `out[n,j] = Σ_b scale_x[n,b] · scale_w[j,b] · dot(x_q[n,b,:], w_q[j,b,:])`:

1. **Inter-block delays:** E_i = floor(log2(x_scale[b] · w_scale[b])), delay = E_max − E_i
2. **Intra-block delays:** Per-element activation exponent differences within each block
3. **Budget resolution:** B_final = B_base + α · max(0, E_combined − E_threshold), where E_combined = max_b(floor(log2(x_scale[n,b] · w_scale[j,b]))) is computed per (sample, output-channel) using the combined activation+weight scales
4. **Effective precision:** P = max(0, B_final − inter_delay − intra_delay − online_delay)
5. **Truncation (BSD/NAF):** Each product is truncated to P most significant BSD digits using Non-Adjacent Form (NAF) representation. NAF is the canonical, minimum-weight BSD encoding computed via:
   - `x_h = x >> 1; s = x + x_h; naf_pos = s & ~x_h; naf_neg = x_h & ~s`
   - NAF can shift the MSD position +1 vs binary (e.g. 7 = `111` binary but `100(-1)` NAF = 4 digit positions), making truncation error bidirectional
6. **Accumulation:** Truncated products are summed, then scaled by shared block scales

### Deep Pipeline (Optional)

When `msd_deep_pipeline=true`, precision is tracked through the MLP stages:
- gate_proj output → SiLU (−precision_loss digits) → element-wise multiply (−online_delay) → down_proj
- This models the physical constraint that MSD digit streams lose precision at each pipeline stage
- All truncation operations (including SiLU and element-wise multiply) use BSD/NAF truncation

---

## 9. Known Limitations

1. **Attention not covered:** Only MLP projections (gate/up/down_proj) use MXFP + MSD. Attention projections (q/k/v/o_proj) remain standard nn.Linear. This is by design — attention is a separable concern.

2. **Deep pipeline uses scalar budget proxy:** The pipeline precision tracking uses the global `msd_cycle_budget` as a scalar proxy rather than per-element effective precision from the actual truncated dot-product. This simplification is intentional and may be refined based on PPL results.

3. **Not thread-safe (but process-safe):** The `MSDComputeContext` uses a class-level singleton pattern (`_active`). This would break under concurrent forward passes within a single process (e.g., `nn.DataParallel`). However, it is safe under `torchrun` / DDP because each rank is a separate process with its own class variable.

4. **Memory for large models:** The MSD path creates a `(N, out, nb, bs)` tensor for per-element truncation. This is now **output-chunked** with a configurable target (`msd_chunk_target_mib`, default 512 MiB). The actual peak memory per chunk is roughly 3x this value due to multiple coexisting intermediates. Reduce `msd_chunk_target_mib` in config.json if encountering OOM on smaller GPUs.

5. **MLP-only scope:** The simulation covers the `gate_proj → SiLU → ×up_proj → down_proj` pattern. Other operations (LayerNorm, residual adds, softmax) use standard precision.

6. **NAF as BSD reference:** The BSD truncation simulation uses Non-Adjacent Form (NAF) — the canonical minimum-weight BSD encoding. The actual hardware digit stream during online arithmetic may differ from NAF due to computation order and intermediate residuals. NAF gives a deterministic, reproducible, and mathematically well-defined truncation model that captures the key BSD property (carry absorption and MSD position shift).

7. **Combined-scale budget memory:** The `_resolve_channel_budgets` method creates a `(N, out, nb)` intermediate tensor for computing per-(sample, output-channel) combined log2 scales. For PPL evaluation (N=4096, out=3072, nb=32) this is ~1.5 GB; acceptable but worth noting for very large batch sizes.

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `ppltest.py` | Perplexity evaluation — single setup, multi-GPU via `--nproc` |
| `ppl_batch.py` | Perplexity evaluation — all 21 setups, multi-GPU via `--nproc` |
| `dist_utils.py` | Distributed helpers (NCCL init, all_reduce, barrier, gather) |
| `test_distributed.py` | Verification tests for multi-GPU infrastructure |
| `test_mxfp8linear.py` | Unit tests for MXFP layers and MSD truncation |
| `qwen3test.py` | Quick generation sanity check |
| `benchmarktest.py` | lm-eval harness (MMLU, GSM8K) |
| `visualization.py` | Chart generation from benchmark JSON results |
| `calibrate.py` | Offline MSD budget calibration, multi-GPU via `--nproc` |
| `calibration_viz.py` | Calibration quality visualization (diagnostic charts from JSON) |
| `perf_viz.py` | MSD inference performance statistics visualization (charts from PPL result JSON) |
| `ppl_results_*.json` | Saved PPL evaluation results (one per setup) |
| `ppl_batch_summary.json` | Consolidated batch summary with all metrics |

## Baseline N:M Model Sparsification
To carry out N:M baseline structured sparsification, we dynamically benchmark masks utilizing offline activation norms and weight magnitudes. The entire process simulates the exact MXFP quantization prior to mask generation and effectively directly modifies model parameters before inference.

Read the detailed mathematical foundation here: [Baseline Sparsify](baseline_sparsify.md).

```bash
# 1. Run Baseline Sparsity Calibration 
# Creates baseline boolean masks scaling $N=2$ elements strictly across $M=4$ grouped metrics
python calibrate_base.py --nproc 4 -n 2 -m 4

# 2. Evaluate Baseline Target Over PPL Batches
# Computes the generated outputs accurately running entirely via standard MX calculation logic
python ppl_batch_base.py --nproc 4 -n 2 -m 4
```

### Activation-only Runtime N:M (No Calibration Files)

This mode applies n:m sparsity to activations dynamically at inference time.
It does not use `calibrate_base.py` and does not load `calibration_base_*.pt` masks.

```bash
# Runtime activation-only n:m sparsity over MXFP setups
python act_base/ppl_batch_base_act.py --nproc 4 -n 2 -m 4

# Sweep multiple n:m pairs with one command
python act_base/ppl_batch_base_act_scan.py --nm 2:4 1:4 --nproc 4
```

Results are saved under `~/coding/data/act_base/{n}-{m}/`.
