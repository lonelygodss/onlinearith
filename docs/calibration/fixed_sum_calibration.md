# Fixed-Sum Budget Redistribution Calibration

## Overview

The **fixed-sum optimizer** is an advanced MSD budget calibration method that redistributes cycle budgets across output channels to minimize total calibration error while preserving the total hardware cycle budget.

## Motivation

The standard `snr_min` calibration uses binary search to find the minimum per-channel budget that meets a target SNR (e.g., 30 dB). While this ensures each channel independently achieves its SNR target with minimal cycles, it does not account for the **asymmetry in error sensitivity** across channels:

- Some channels have **high marginal gain**: reducing their budget by 1 cycle causes large error increase
- Other channels have **low marginal loss**: increasing their budget by 1 cycle provides small error reduction

The fixed-sum optimizer exploits this asymmetry by moving cycles from low-gain channels (donors) to high-gain channels (receivers), reducing total layer error without changing the hardware cycle budget.

## Algorithm

### Stage 1: Capture Block Data
```python
caches = collect_layer_block_cache(model, tokenizer, texts)
```
Runs forward passes with MSD disabled (exact MX quantization), hooks into each MXFP linear layer, and captures:
- Quantized activation blocks `(x_q, x_scales)` per batch
- Quantized weight blocks `(w_q, w_scales)` once per layer (fixes duplication bug)

### Stage 2a: Solve SNR-Min Budgets
```python
budgets_snr_min, summary, detail = solve_min_snr_budgets_from_cache(cache)
```
Binary search (12 iterations) to find per-channel budgets meeting target SNR. This provides the **anchor budget vector** `B*` and defines the target sum `S = sum(B*)`.

### Stage 2b: Build Error Curves
```python
curves = build_error_curves_from_cache(cache, budgets_snr_min, window=3)
```
For each channel `j`, evaluates the squared error `E_j(b) = ||Y_exact[:,j] - Y_hat[:,j; b]||²` at budget values in `[B*_j - window, B*_j + window]`, clamped to `[4, 48]`.

### Stage 2c: Fixed-Sum Redistribution
```python
budgets_opt, stats = solve_fixed_sum_from_error_curves(curves, target_sum=S)
```

**Greedy heap-based swap algorithm:**

1. Initialize `B = B*` (SNR-min budgets), `S = sum(B*)`
2. For each channel `j`, compute:
   - **Donor loss**: `L_j = E_j(B_j - 1) - E_j(B_j)` (error increase from donating 1 cycle)
   - **Receiver gain**: `G_j = E_j(B_j) - E_j(B_j + 1)` (error decrease from receiving 1 cycle)
3. Build two heaps:
   - Donor heap: min-heap on `L_j` (pop smallest loss)
   - Receiver heap: max-heap on `G_j` (pop largest gain)
4. While heaps non-empty:
   - Peek best donor `j_d` (smallest loss) and best receiver `j_r` (largest gain)
   - If `G_r > L_d`: swap 1 cycle from `j_d` to `j_r`, update heaps
   - Else: stop (no beneficial swap remains)
5. Return optimized budgets `B_opt` with `sum(B_opt) == S`

**Constraints enforced:**
- `sum(B) == S` (preserved exactly)
- `4 <= B_j <= 48` (hardware limits)
- `B_j` integer

### Stage 3: Holdout Validation (Optional)
```python
holdout_caches = collect_layer_block_cache(model, tokenizer, holdout_texts)
holdout_eval = evaluate_budget_vector_from_cache(holdout_cache, budgets_opt)
```
Evaluates optimized budgets on held-out texts to verify generalization.

## Usage

### Basic Usage
```bash
# Standard SNR-min calibration (baseline)
python calibrate.py --setup 1

# Fixed-sum redistribution
python calibrate.py --setup 1 --optimizer fixed_sum

# Multi-GPU batch mode (all 4 formats)
python calibrate.py --nproc 4 --optimizer fixed_sum
```

### Advanced Options
```bash
# With holdout validation (20% held out)
python calibrate.py --setup 1 --optimizer fixed_sum --holdout-fraction 0.2

# Only calibrate gate_proj layers
python calibrate.py --setup 1 --optimizer fixed_sum --projection-filter gate_proj

# Wider error curve window (default: 3)
python calibrate.py --setup 1 --optimizer fixed_sum --curve-window 5

# Save full error curves for debugging (large files)
python calibrate.py --setup 1 --optimizer fixed_sum --save-curve-detail
```

## Output Files

Results are saved in `../data/calib-data/{snr}db/`:

- **SNR-min mode**: `calibration_{tag}.json` (e.g., `calibration_MXFP8.json`)
- **Fixed-sum mode**: `calibration_{tag}_fixed_sum.json` (e.g., `calibration_MXFP8_fixed_sum.json`)

### JSON Schema

The output JSON extends the standard schema with:

```json
{
  "optimizer": "fixed_sum",
  "calibration_params": {
    "holdout_texts": 4,
    "holdout_fraction": 0.2,
    "curve_window": 3
  },
  "optimizer_stats": {
    "model.layers.0.mlp.gate_proj": {
      "swaps_performed": 42,
      "total_improvement": 12.5,
      "final_sum": 2048,
      "target_sum": 2048,
      "sum_preserved": true
    }
  },
  "holdout_evaluation": {
    "model.layers.0.mlp.gate_proj": {
      "layer_snr_mean": 30.2,
      "layer_snr_min": 29.8,
      "total_error": 0.045,
      "budget_sum": 2048.0,
      "budget_mean": 10.2
    }
  },
  "msd_calibration_data": { ... }
}
```

The `msd_calibration_data` field remains **backward-compatible** with existing inference code.

## Validation Gates

### Gate A: Held-Out Calibration Error
Split texts into train/holdout (e.g., 80/20). Optimize on train split, evaluate on holdout. Accept if:
- `total_error_holdout(fixed_sum) <= total_error_holdout(snr_min)`

### Gate B: Short PPL + Runtime Metrics
```bash
python ppltest.py --setup 6 --calibration calibration_MXFP8_fixed_sum.json --limit-samples 100
```
Compare against `calibration_MXFP8.json`. Accept if:
- PPL is non-inferior
- Runtime metrics (`utilization`, `mac_sparsity`, `max_total_delay`) do not worsen

## Expected Results

- **Swaps performed**: 10-100 per layer (depends on error curve asymmetry)
- **Total improvement**: Positive (error reduction on train set)
- **Sum preservation**: Exact (verified by unit tests)
- **Holdout error**: Equal or lower than SNR-min
- **Inference runtime**: Identical (same total cycle budget)

## Implementation Details

### Key Files
- `calibration_msd.py`: Core implementation (staged API, heap optimizer)
- `calibrate.py`: Driver script with CLI integration
- `test_fixed_sum_optimizer.py`: Unit tests

### Memory Efficiency
- Uses same GPU-chunked computation as SNR-min (256 MiB target)
- Error curves computed only in local window (default: ±3 cycles)
- Weights stored once per layer (fixes duplication bug)

### Computational Cost
- **Stage 1 (capture)**: Same as SNR-min (forward passes with exact MX)
- **Stage 2a (SNR-min)**: Same as SNR-min (binary search)
- **Stage 2b (curves)**: ~7 evaluations per channel (window=3 → 7 budget points)
- **Stage 2c (redistribution)**: O(k log n) where k = swaps, n = channels (~100 ms)

Total overhead: ~2-3x SNR-min runtime for curve building.
