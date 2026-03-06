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

**Evaluation scripts (multi-GPU via torchrun):**
- `onlinearith/ppltest.py` — Single-setup PPL evaluation with window-level data parallelism
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
| 14 | MXFP8 + MSD + Calibrated | Per-channel B_base from calibration (see Section 5) |

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
  "msd_budget_dynamic_scale": 1.0,
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

## 3. MSD Config Fields Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_msd_truncation` | bool | false | Master switch for MSD simulation |
| `msd_cycle_budget` | int | 16 | Global default cycle budget B_base |
| `msd_online_delay` | int | 2 | MSD multiplier online delay δ (digits before first valid product) |
| `msd_budget_dynamic_scale` | float | 1.0 | α in B_final = B_base + α·max(0, E_combined − E_threshold), where E_combined is the max combined log2 scale of activation and weight per (sample, output-channel) |
| `msd_budget_dynamic_threshold` | float | 0.0 | E_threshold for dynamic combined-scale budget override |
| `msd_budget_dynamic_mode` | str | "linear" | "linear" or "step" for dynamic budget override mode |
| `msd_deep_pipeline` | bool | false | Enable MSD streaming through MLP (gate→silu→×up→down) |
| `msd_pipeline_precision_loss` | int | 2 | Digits of precision lost per pipeline stage |
| `msd_calibration_data` | dict/null | null | Per-layer per-channel calibrated B_base values |

---

## 4. Running PPL Tests

Both `ppltest.py` and `ppl_batch.py` support **multi-GPU via torchrun** and **single-GPU fallback**.
When launched with `torchrun`, each GPU loads its own model copy (~1.2 GB for Qwen3-0.6B, well
within 32 GB per RTX 5090) and processes its shard of the work.

### Quick Start (Multi-GPU)

```bash
cd /home/xzj/coding/onlinearith
source /home/xzj/coding/.venv3_10/bin/activate

# Single PPL evaluation — 8 GPUs, ~7-8x speedup
torchrun --nproc_per_node=8 ppltest.py --setup 2

# Full 21-config sweep — 8 GPUs, ~3x speedup
torchrun --nproc_per_node=8 ppl_batch.py

# Subset of configs on 8 GPUs
torchrun --nproc_per_node=8 ppl_batch.py --only 2 6 10

# Use only specific GPUs (e.g. GPUs 0, 2, 5 — when others are busy)
torchrun --nproc_per_node=3 ppltest.py --gpus 0,2,5 --setup 6
torchrun --nproc_per_node=3 ppl_batch.py --gpus 0,2,5

# List all predefined setups (works for both scripts)
python ppltest.py --list
python ppl_batch.py --list

# Single-GPU fallback (unchanged behavior, no torchrun needed)
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

# 2. Suppose GPUs 0, 3, 7 are free — use only those
torchrun --nproc_per_node=3 ppl_batch.py --gpus 0,3,7

# For single-GPU mode, just specify one GPU
python ppltest.py --gpus 3
```

**Important:** When using `torchrun`, `--nproc_per_node` must **equal** the
number of GPU IDs in `--gpus`. A mismatch will produce a clear error message.

The `--gpus` flag sets `CUDA_VISIBLE_DEVICES` internally before any CUDA
context is created. You can also set the environment variable directly if
you prefer:

```bash
# Equivalent to --gpus 0,3,7
CUDA_VISIBLE_DEVICES=0,3,7 torchrun --nproc_per_node=3 ppl_batch.py
```

If both `--gpus` and `CUDA_VISIBLE_DEVICES` are provided, `--gpus` takes
precedence (it overwrites the env var).

### Parallelism Strategy

| Script | Parallelism | How It Works |
|--------|-------------|-------------|
| `ppltest.py` | **Window-level** | 578 sliding windows are sharded round-robin across GPUs. Partial NLL sums are aggregated via NCCL `all_reduce`. Only rank 0 saves the JSON. |
| `ppl_batch.py` | **Setup-level** | 21 config setups are partitioned round-robin across GPUs. Each GPU evaluates its assigned setups independently, writing result files directly. Rank 0 prints the summary table after all ranks finish. |

The `MSDComputeContext._active` class-level singleton is **process-safe** under torchrun
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
   torchrun --nproc_per_node=8 ppltest.py --setup 6      # MXFP8 + MSD B=16

   # Use specific GPUs only
   torchrun --nproc_per_node=3 ppltest.py --gpus 0,2,5 --setup 6

   # Single-GPU fallback
   python ppltest.py --setup 6
   ```
   The output file is auto-named `ppl_results_{tag}.json` from the setup tag.
   Use `--output custom_name.json` to override.

3. **Advanced: custom config.json (no --setup):**
   If `--setup` is omitted, `ppltest.py` uses whatever config is in
   `Qwen3-0.6B/config.json` and saves to the hardcoded `RESULTS_OUT` constant.
   ```bash
   # Edit config.json first, then:
   torchrun --nproc_per_node=8 ppltest.py
   ```

4. **Expected runtime:**
   - Single GPU: ~25-28 min per run (578 windows, ~180 tokens/sec)
   - 8× RTX 5090: ~3-4 min per run (~7-8× speedup)

### Naming Convention for Result Files

Use this pattern: `ppl_results_{FORMAT}[_MSD_B{budget}][_pipeline][_calibrated].json`

Examples:
- `ppl_results_MXFP8.json` — MXFP8 only (already exists)
- `ppl_results_MXFP8_MSD_B16.json` — MXFP8 + MSD, budget=16
- `ppl_results_MXFP8_MSD_B8.json` — MXFP8 + MSD, budget=8
- `ppl_results_MXFP4_MSD_B16.json` — MXFP4 + MSD, budget=16
- `ppl_results_MXFP8_MSD_B16_pipeline.json` — MXFP8 + MSD + deep pipeline
- `ppl_results_MXFP8_MSD_calibrated.json` — MXFP8 + calibrated budgets

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
   - Run calibration first (see Section 5), then PPL test

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
torchrun --nproc_per_node=8 ppl_batch.py

# Run only specific setups by ID
torchrun --nproc_per_node=8 ppl_batch.py --only 1 6 10 14

# Run on specific GPUs (e.g. 0, 2, 5 are idle)
torchrun --nproc_per_node=3 ppl_batch.py --gpus 0,2,5

# Combine GPU selection with setup filtering
torchrun --nproc_per_node=3 ppl_batch.py --gpus 0,2,5 --only 1 6 10

# Re-run setups even if result files exist
torchrun --nproc_per_node=8 ppl_batch.py --force

# Resume after interruption (skips completed ones)
torchrun --nproc_per_node=8 ppl_batch.py

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
torchrun --nproc_per_node=3 ppl_batch.py --gpus 0,2,5
# Ctrl+B then D to detach; tmux attach -t ppl to reconnect
```

---

## 5. Running Calibration

Calibration determines optimal per-channel cycle budgets (B_base) based on actual weight/activation statistics. `calibrate.py` supports **multi-GPU via torchrun** (like `ppl_batch.py`) and covers all 4 MXFP formats.

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
torchrun --nproc_per_node=4 calibrate.py

# Specific GPUs
torchrun --nproc_per_node=2 calibrate.py --gpus 0,3

# Subset of formats
torchrun --nproc_per_node=2 calibrate.py --only 1 4

# Custom SNR target (higher = more budget = less error)
torchrun --nproc_per_node=4 calibrate.py --target-snr 40

# Re-run even if result files exist
torchrun --nproc_per_node=4 calibrate.py --force

# Single specific GPU
python calibrate.py --gpus 3 --setup 1
```

Output files are named `calibration_{tag}.json` and saved in `onlinearith/`.

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

### Expected Runtime

- Single GPU: ~2-5 min per format (depends on `--num-texts` and `--max-length`)
- 4× RTX 5090: ~2-5 min total (all 4 formats in parallel)

### Using Calibration Results with PPL Tests

After calibration, copy the `msd_calibration_data` field from the calibration JSON into `Qwen3-0.6B/config.json`:

```bash
# Example: use MXFP8 calibration with PPL test
python -c "
import json
with open('calibration_MXFP8.json') as f:
    cal = json.load(f)
with open('../Qwen3-0.6B/config.json') as f:
    cfg = json.load(f)
cfg['msd_calibration_data'] = cal['msd_calibration_data']
cfg['use_msd_truncation'] = True
cfg['use_mxfp8'] = True
with open('../Qwen3-0.6B/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Done')
"

# Then run PPL test (setup 14 = MXFP8 + MSD + Calibrated)
torchrun --nproc_per_node=8 ppltest.py --setup 14
```

### Interpreting Results

Each `calibration_{tag}.json` contains:

```json
{
  "format": "MXFP8",
  "summary": {
    "num_layers": 84,
    "budget_min": 8,
    "budget_max": 32,
    "budget_mean": 16.5,
    "wall_time_sec": 120.0
  },
  "msd_calibration_data": {
    "model.layers.0.mlp.gate_proj": [12, 14, 16, ...],
    "model.layers.0.mlp.up_proj": [10, 12, 14, ...],
    "model.layers.0.mlp.down_proj": [14, 16, 18, ...],
    ...
  }
}
```

Channels with large activation outliers get higher budgets; channels with small, uniform-scale data get lower budgets — enabling hardware-native unstructured sparsity.

### To Remove Calibration

Set `msd_calibration_data` to `null` in config.json to revert to uniform budgets:
```json
{
  "msd_calibration_data": null
}
```

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

4. **Memory for large models:** The MSD path creates a `(N, out, nb, bs)` tensor for per-element truncation. This is now **output-chunked** to stay within a 2 GiB budget, which solved the original 48 GB OOM on Qwen3-0.6B. Larger models should also work within the chunk limits.

5. **MLP-only scope:** The simulation covers the `gate_proj → SiLU → ×up_proj → down_proj` pattern. Other operations (LayerNorm, residual adds, softmax) use standard precision.

6. **NAF as BSD reference:** The BSD truncation simulation uses Non-Adjacent Form (NAF) — the canonical minimum-weight BSD encoding. The actual hardware digit stream during online arithmetic may differ from NAF due to computation order and intermediate residuals. NAF gives a deterministic, reproducible, and mathematically well-defined truncation model that captures the key BSD property (carry absorption and MSD position shift).

7. **Combined-scale budget memory:** The `_resolve_channel_budgets` method creates a `(N, out, nb)` intermediate tensor for computing per-(sample, output-channel) combined log2 scales. For PPL evaluation (N=4096, out=3072, nb=32) this is ~1.5 GB; acceptable but worth noting for very large batch sizes.

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `ppltest.py` | Perplexity evaluation — single setup, multi-GPU via torchrun |
| `ppl_batch.py` | Perplexity evaluation — all 21 setups, multi-GPU via torchrun |
| `dist_utils.py` | Distributed helpers (NCCL init, all_reduce, barrier, gather) |
| `test_distributed.py` | Verification tests for multi-GPU infrastructure |
| `test_mxfp8linear.py` | Unit tests for MXFP layers and MSD truncation |
| `qwen3test.py` | Quick generation sanity check |
| `benchmarktest.py` | lm-eval harness (MMLU, GSM8K) |
| `visualization.py` | Chart generation from benchmark JSON results |
| `calibrate.py` | Offline MSD budget calibration, multi-GPU via torchrun |
| `ppl_results_*.json` | Saved PPL evaluation results (one per setup) |
| `ppl_batch_summary.json` | Consolidated batch summary with all metrics |
