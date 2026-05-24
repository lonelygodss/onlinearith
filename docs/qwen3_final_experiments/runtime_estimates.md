# Single-Setup Runtime Estimates

This is a living estimate for the focused Qwen3 model family. It intentionally
tracks one representative setup per path. Target sweeps are deferred. Multi-GPU
model-sharded estimates should be added after explicit sharding is validated.

## Models

Local model directories currently present under `../`:

| Model path | Layers | Hidden | MLP intermediate | Relative MLP projection work |
|---|---:|---:|---:|---:|
| `../Qwen3-0.6B` | 28 | 1024 | 3072 | 1.00x |
| `../Qwen3-1.7B` | 28 | 2048 | 6144 | 4.00x |
| `../Qwen3-4B` | 36 | 2560 | 9728 | 10.18x |
| `../Qwen3-8B` | 36 | 4096 | 12288 | 20.57x |

The estimates below are single-GPU only until the multi-GPU section is filled
with direct-CUDA evidence. `ppltest.py --nproc` is not model sharding; it
replicates the model and shards windows.

## Representative Setups

| Path | Representative setup | Why this point |
|---|---|---|
| MX baseline | `ppltest.py --setup 2` | Dense MXFP8 reference. |
| Calibrated MSD | fixed-sum calibrated MXFP8 MSD at target-SNR 30 dB | Figure 4 plots fixed-sum 30 dB at `plot_norm_digit_read = 0.87`, close to dense digit-read work. |
| WANDA | common `2:4` N:M | 50% kept weights, 50% structured sparsity. |
| Activation N:M | common `2:4` N:M | 50% kept activations, 50% runtime activation sparsity. |

For WANDA and activation baselines, use common N:M notation: `2:4` means keep 2
values per group of 4. Internally this prunes `(4 - 2):4`.

For MSD, do not use `msd_perf_stats.global.global_utilization` as the
cross-method equivalent sparsity/work coordinate. Figure 4 uses
`plot_norm_digit_read = mean_effective_precision / 3.0`. Fixed-sum target-SNR
30 dB is close to dense work (`0.87`), and 35 dB is closer still (`0.95`).

## Existing MSD Equivalent-Work Evidence

The equivalent sparsity/work coordinate used for Figure 4 is
`plot_norm_digit_read`. The original extractor in the figure repo history,
`../figure` commit `d60ad9c`, `figure4/extract_figure4_data.py`, defines:

```text
DENSE_DIGIT_READS = 3.0
norm_read_uncap = uncap_e_p_eff / 3.0
norm_read_cap = cap_e_eff / 3.0
```

Here `uncap_e_p_eff` and `cap_e_eff` are
`msd_perf_stats.global.mean_effective_precision` from the relevant PPL JSONs.
For fixed-sum and uniform MSD rows, `plot_norm_digit_read` uses capped
`cap_e_eff / 3.0` when the cap100 file exists, otherwise uncapped
`uncap_e_p_eff / 3.0`. The current
`../figure/figure4/prepare_figure4_plot_data.py` then carries this raw
`plot_norm_digit_read` into the plot x-axis. For WANDA and activation N:M rows,
the historical extractor used kept fraction `(m - n) / m` from the old
prune-count notation.

Current prepared Figure 4 data includes these fixed-sum points. The implied
effective precision is `plot_norm_digit_read * 3.0`.

| Target SNR | `plot_norm_digit_read` | Implied `mean_effective_precision` |
|---:|---:|---:|
| 15 dB | 0.4133 | 1.2399 |
| 18 dB | 0.5170 | 1.5511 |
| 20 dB | 0.5858 | 1.7575 |
| 25 dB | 0.7268 | 2.1803 |
| 30 dB | 0.8700 | 2.6100 |
| 35 dB | 0.9500 | 2.8500 |

Therefore fixed-sum 30 dB should be treated as close to dense equivalent
digit-read work, not as a 50% MSD point.

## Historical Runtime-Stat Artifacts

Former 0.6B calibrated fixed-sum PPL artifacts under
`../data/calib-data/<xdb>/ppl_results_MXFP8_fix_{cap,time}.json` were generated
with commands shaped like:

```bash
python ppltest.py --nproc 1 --setup 6 \
  --calibration ../data/calib-data/27db/calibration_MXFP8_fixed_sum.json \
  --lite \
  --output ../data/calib-data/27db/ppl_results_MXFP8_fix_time.json \
  --limit-samples 100 \
  --figure5-layer-cycles \
  --gpus 6
```

Those generated files were later removed from version control by:

```text
e0a38ec908233cbbf21c56262e5b79963a212660 Clean up repo docs and remove obsolete scripts
```

Do not restore those files. For consistency, regenerate any comparable pilot
with the same target behavior: `--setup 6`, fixed-sum calibration JSON, explicit
`--output`, and a 100-sample lite-stat PPL pass.

The April 9 digit-cap fix applied before those runs. The relevant Transformers
commit is:

```text
faab82a9b3 digit cap fix
```

Current code still contains the same lite-stat p_eff cap path
(`_resolve_lite_p_eff_cap` plus `MSDPerfAccumulator(..., lite_p_eff_cap=...)`).
Current `ppltest.py` also preserves the old CLI behavior: `--lite` maps to
`--stats lite`, and `--figure5-layer-cycles` forces lite stats while adding the
Figure 5 cycle summaries.

The standard maintained command for new fixed-sum MSD timing/utilization probes
is:

```bash
python ppltest.py --nproc 1 --setup 6 \
  --calibration ../data/calib-data/27db/calibration_MXFP8_fixed_sum.json \
  --msd-utilization-mode \
  --output ../data/calib-data/27db/ppl_results_MXFP8_fix_time.json \
  --gpus 6
```

`--msd-utilization-mode` sets `--stats lite`, defaults `--limit-samples` to 100
when omitted, and does not collect Figure 5 cycle moments. Add
`--figure5-layer-cycles` only when debugging Figure 5 cycle accounting; it is no
longer part of the standard utilization/timing probe.

Observed fixed-sum MXFP8 runtime `global_utilization` from those artifacts:

| Target SNR | Output variants checked | Runtime `global_utilization` |
|---:|---|---:|
| 12 dB | `_fix_cap`, `_fix_time` | 0.121 |
| 15 dB | `_fix_cap`, `_fix_time` | 0.149 |
| 20 dB | `_fix_cap`, `_fix_time` | 0.190 |
| 25 dB | `_fix_cap`, `_fix_time` | 0.215 |
| 30 dB | `_fix_time` | 0.225 |

These values are runtime diagnostics. They are not the equivalent sparsity/work
axis used in Figure 4 and should not drive the representative MSD setup choice.

Output convention for new compatibility checks:

- Calibration metadata: use `calibrate.py --optimizer fixed_sum --target-snr 30`.
- PPL timing/utilization: use explicit
  `ppltest.py --msd-utilization-mode --output ..._fix_time.json`.
- If a cap-vs-time comparison is needed, write the paired output explicitly as
  `..._fix_cap.json` and `..._fix_time.json`; do not rely on implicit `_calib`
  output names.

## Single-Setup Runtime Estimate

One row is one model. Each cell is one representative setup on one GPU. For
calibrated MSD, "calibration" means one fixed-sum calibration at target-SNR
30 dB, and "PPL" means one calibrated MSD PPL run using that metadata.

| Model | MXFP8 PPL | Fixed-sum MSD calibration | Fixed-sum MSD PPL, current runtime | WANDA 2:4 calibration + PPL | Activation 2:4 PPL |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 5-20 min | 1.2 h measured | about 2 h | 10-30 min | 5-20 min |
| Qwen3-1.7B | 10-30 min | 4.7 h estimated | about 8 h | 20-60 min | 10-30 min |
| Qwen3-4B | 25-60 min | 11.9 h estimated | about 20 h | 1-2 h | 25-60 min |
| Qwen3-8B | 0.6-1.0 h | 24 h estimated | about 40 h | 1.3-2 h | 0.6-0.8 h |

Basis:

- Full WikiText-2 PPL is about 299k scored tokens.
- Qwen3-8B MXFP8 setup 2 prefix80 measured 4144 tokens in 31.97s on one GPU
  and 31.80s with explicit sequential model sharding across two active visible
  CUDA devices. This extrapolates to about 0.64 h for full PPL either way; the
  sharded benefit in this probe is per-GPU memory, not throughput.
- Qwen3-8B calibrated/uniform MSD prefix80 measured 4144 tokens in about 1999s,
  which extrapolates to about 40 h for full PPL at the current runtime.
- Qwen3-8B WANDA and activation prefix80 measured about 32-33s on the same 4144
  tokens, extrapolating to about 0.65-0.75 h for full PPL. WANDA includes an
  additional offline mask calibration estimate.
- Qwen3-0.6B fixed-sum calibration with 20 texts x 512 measured about 70 min per
  target. Larger-model calibration estimates scale by relative MLP projection
  work and should be replaced by direct timings.

## Multi-GPU Estimate Status

Model-sharded MXFP8 has prefix-level correctness and timing evidence. MSD,
WANDA, and activation N:M sharded timings are still pending.

| Model | Path | Sharding mode | GPUs | Evidence | Wall-time estimate |
|---|---|---|---:|---|---:|
| Qwen3-8B | MXFP8 PPL | sequential | 2 active of 4 visible | non-final prefix80, exact PPL parity with single GPU; peak alloc 19.8733 + 9.3105 GiB | about 0.64 h, non-final extrapolation |
| Qwen3-8B | Fixed-sum MSD 30 dB PPL | pending | pending | not validated | pending |
| Qwen3-8B | WANDA 2:4 | pending | pending | not validated | pending |
| Qwen3-8B | Activation N:M 2:4 | pending | pending | not validated | pending |

## Update Rules

- Update this file after each accepted optimization or full direct-CUDA timing
  run.
- Keep smoke/prefix/`--limit-samples` runs out of this table unless explicitly
  labeled as non-final evidence.
- Record cache dtype, chunk sizes, stats mode, compile flag, target-SNR,
  Figure 4 `plot_norm_digit_read` when available, and observed runtime
  `global_utilization` with every new MSD timing.
- Add multi-GPU wall-time estimates only after explicit model sharding or job
  packing is validated and the output metadata records that execution mode.
