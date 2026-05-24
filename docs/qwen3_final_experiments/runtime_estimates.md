# Single-Setup Runtime Estimates

This is a living estimate for the focused Qwen3 model family. It intentionally
tracks one representative setup per path. Target sweeps are deferred.

## Models

Local model directories currently present under `../`:

| Model path | Layers | Hidden | MLP intermediate | Relative MLP projection work |
|---|---:|---:|---:|---:|
| `../Qwen3-0.6B` | 28 | 1024 | 3072 | 1.00x |
| `../Qwen3-1.7B` | 28 | 2048 | 6144 | 4.00x |
| `../Qwen3-4B` | 36 | 2560 | 9728 | 10.18x |
| `../Qwen3-8B` | 36 | 4096 | 12288 | 20.57x |

`ppltest.py --nproc` is not model sharding; it replicates the full model and
shards PPL windows. That makes it a valid final wall-time accelerator when each
selected GPU can fit a full replica. `ppltest.py --device-map` is the separate
single-process model-sharding path and should be treated as memory relief unless
fresh direct-CUDA evidence shows throughput improvement.

Current `ppltest.py --nproc` runs disable MSD stats on nonzero ranks. Use
multi-rank output for PPL quality and wall time, but do not report its
`msd_perf_stats` as a full-dataset aggregate unless rank-level stats aggregation
is added. Until then, collect work/accounting metrics in a separate
single-process utilization run or another explicitly documented probe.

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
| Qwen3-8B | about 2.6 h single GPU; about 0.67 h validated on 4 full replicas; about 0.34 h theoretical on 8 if launch is fixed | 24 h estimated single job; faster with projection-filtered task parallel jobs | about 160 h single GPU; about 45 h validated on 4 full replicas with float8 cache; about 23 h theoretical on 8 if launch is fixed | about 2.6 h PPL plus 1.3-2 h mask calibration | about 2.7 h single GPU; about 0.3 h on 8 full replicas if launch is fixed |

Basis:

- Full WikiText-2 PPL tokenizes to 299,078 tokens and 578 forward windows at
  `MAX_LENGTH=4096`, `STRIDE=512`.
- Runtime should be estimated by forward windows, not scored tokens. The first
  window scores 4096 tokens, but almost all later windows still feed a full
  4096-token context while scoring only 512 new tokens.
- Qwen3-8B MXFP8 setup 2 prefix80 measured 4144 tokens in 31.97s on one GPU
  and 31.80s with explicit sequential model sharding across two active visible
  CUDA devices. This prefix has two long forward windows, so the single-GPU
  full-PPL estimate is about 2.6 h by window count; the sharded benefit in this
  probe is per-GPU memory, not throughput.
- Qwen3-8B MXFP8 setup 2 with `ppltest.py --nproc 2 --gpus 4,5 --stats off`
  on the same prefix80 slice matched single-GPU PPL exactly at recorded
  precision and reduced wall time from 31.97s to 17.02s.
- Qwen3-8B MXFP8 setup 2 with `ppltest.py --nproc 4 --gpus 0,1,2,3
  --limit-samples 120 --stats off` completed a prefix120 slice with eight PPL
  windows in 33.2s. That is about 16.6s per assigned window, or about 0.67 h
  for the full 578-window run on four workers. An eight-worker launch on GPUs
  0-7 failed with rank-0 `SIGKILL` during model loading/materialization before
  evaluation; the eight-worker estimate remains theoretical until that startup
  issue is fixed.
- Qwen3-8B calibrated/uniform MSD prefix80 measured 4144 tokens in about 1999s,
  or about 1000s per long forward window. The full-PPL estimate is therefore
  about 160 h on one GPU.
- Qwen3-8B fixed-sum target-SNR 30 dB with `ppltest.py --nproc 2 --gpus 4,5
  --stats off --compile-msd-truncate --weight-cache-dtype float8` matched the
  prior prefix PPL exactly at recorded precision and completed in 1120.41s. A
  default float16-cache `--nproc 2` run OOMed on rank 1, so the current
  multi-GPU MSD recipe should include `--weight-cache-dtype float8`.
- Qwen3-8B fixed-sum target-SNR 30 dB with `ppltest.py --nproc 4 --gpus
  0,1,2,3 --limit-samples 120 --stats off --compile-msd-truncate
  --weight-cache-dtype float8` completed eight PPL windows in 2239.0s, again
  about 1120s per assigned window. Scaling to 145 assigned windows per worker
  for the full 578-window run gives about 45.1 h on four workers. Eight workers
  would be about 22.7 h by assigned-window scaling, but that remains
  theoretical until the eight-replica launch/load SIGKILL is resolved.
- Qwen3-8B sequential model-sharded MSD prefix80 measured 4144 tokens in
  2120.60s for uniform setup 6 and 2111.13s for fixed-sum 30 dB. These
  extrapolate by window count to about 170.2 h and 169.5 h for full PPL.
  Sequential sharding did not improve runtime, but reduced the largest recorded
  per-device allocation
  from about 28.03 GB single-GPU peak memory to 26.6621 GiB on the busiest
  visible CUDA device, with 16.1289 GiB on the second active device.
- Qwen3-8B WANDA and activation prefix80 measured about 32-33s on the same 4144
  tokens, extrapolating by window count to about 2.6-2.7 h for full PPL on one
  GPU and about 0.3 h on eight full replicas. WANDA includes an
  additional offline mask calibration estimate.
- Qwen3-0.6B fixed-sum calibration with 20 texts x 512 measured about 70 min per
  target. Larger-model calibration estimates scale by relative MLP projection
  work and should be replaced by direct timings.

## Multi-GPU Estimate Status

Model-sharded MXFP8 and MSD have prefix-level correctness and timing evidence,
but current sequential model sharding is memory relief rather than a speedup.
The final PPL acceleration path should be full-replica data parallel window
sharding with `ppltest.py --nproc` when full replicas fit. Qwen3-8B is
validated up to four workers; eight-worker full-replica startup currently needs
separate launch/load debugging.

| Model | Path | Execution mode | GPUs | Evidence | Wall-time estimate |
|---|---|---|---:|---|---:|
| Qwen3-8B | MXFP8 PPL | `--nproc` full-replica window sharding | 4 validated; 8 attempted but loading SIGKILLed | prefix80 exact PPL parity; prefix120 completed in 33.2s on four workers | about 0.67 h validated on 4 workers; about 0.34 h theoretical on 8 |
| Qwen3-8B | Fixed-sum MSD 30 dB PPL | `--nproc` full-replica window sharding with float8 cache | 4 validated; 8 pending startup fix | prefix80 exact PPL parity; default float16 cache OOMed; prefix120 float8 cache completed in 2239.0s on four workers | about 45.1 h validated on 4 workers; about 22.7 h theoretical on 8 |
| Qwen3-8B | WANDA 2:4 PPL | `--nproc` full-replica window sharding | 4 should be validated before final run; 8 blocked by startup fix | direct Qwen3-8B `--nproc` timing pending; single-GPU prefix has 15.93s/window | about 0.67 h on 4 workers if MXFP8-like scaling holds; about 0.3 h theoretical on 8 |
| Qwen3-8B | Activation N:M 2:4 PPL | `--nproc` full-replica window sharding | 4 should be validated before final run; 8 blocked by startup fix | direct Qwen3-8B `--nproc` timing pending; single-GPU prefix has 16.52s/window | about 0.69 h on 4 workers if MXFP8-like scaling holds; about 0.3 h theoretical on 8 |
| Qwen3-8B | MXFP8 PPL | `--device-map sequential` | 2 active of 4 visible | non-final prefix80, exact PPL parity with single GPU; peak alloc 19.8733 + 9.3105 GiB | about 2.6 h; memory relief only |
| Qwen3-8B | Uniform MSD setup 6 PPL | `--device-map sequential` | 2 active of 4 visible | non-final prefix80, exact PPL parity with historical single GPU; peak alloc 26.6621 + 16.1289 GiB | about 170 h; memory relief only |
| Qwen3-8B | Fixed-sum MSD 30 dB PPL | `--device-map sequential` | 2 active of 4 visible | non-final prefix80, exact PPL parity with historical single GPU; peak alloc 26.6621 + 16.1289 GiB | about 169 h; memory relief only |

## Update Rules

- Update this file after each accepted optimization or full direct-CUDA timing
  run.
- Keep smoke/prefix/`--limit-samples` runs out of this table unless explicitly
  labeled as non-final evidence.
- Record cache dtype, chunk sizes, stats mode, compile flag, target-SNR,
  Figure 4 `plot_norm_digit_read` when available, and observed runtime
  `global_utilization` with every new MSD timing.
- Add multi-GPU wall-time estimates only after explicit model sharding,
  full-replica PPL window sharding, or job packing is validated and the output
  metadata records that execution mode.
