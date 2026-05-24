# Qwen3 OOM/Performance Evidence Log

This file keeps measurement details out of the always-read session context.
Only direct-CUDA runs with valid CUDA memory fields count as performance/OOM
evidence.

## Validated Qwen3-8B OOM Evidence

- Setup 2 probe, seq_len 4096: status ok; peak_alloc 27.6147 GiB;
  peak_reserved 28.2090 GiB; reserved_headroom 3.1477 GiB;
  `mxfp_weight_cache.total_gib=10.7578`.
- Setup 6 with `--compile-msd-truncate`, seq_len 4096 and float16 cache:
  status ok; elapsed 1059.95s; loss 0.01120709; peak_alloc 27.8387 GiB;
  peak_reserved 28.7617 GiB; reserved_headroom 2.5950 GiB;
  `mxfp_weight_cache.total_gib=10.7578`.
- Setup 6 with `--compile-msd-truncate`, seq_len 4096 and native MXFP8
  `--weight-cache-dtype float8`: status ok; elapsed 1060.33s;
  loss 0.01120709; peak_alloc 22.7762 GiB; peak_reserved 23.8984 GiB;
  reserved_headroom 7.4583 GiB; cache total 5.6953 GiB.

Interpretation: float8 cache preserves loss/runtime for MXFP8 and saves about
5.06 GiB allocated and 4.86 GiB reserved versus the float16 cache.

## Prefix/Smoke PPL Evidence

- Setup 6 `ppltest --limit-samples=2 --compile-msd-truncate`: smoke only,
  scored 8 tokens; token PPL 423.0598; mean NLL 6.0475; peak memory 27.08 GB.
- Setup 6 `ppltest --limit-samples=80 --compile-msd-truncate`: non-final
  prefix; scored 4,144 tokens across two windows including a 4096-token context
  window; token PPL 8.8708; mean NLL 2.1828; peak memory 28.03 GB;
  wall time 2000.6s.
- Fixed-sum calibrated MSD using staged merged full-MLP metadata,
  `--limit-samples=80`: scored 4,144 tokens; token PPL 8.8632;
  mean NLL 2.1819; peak memory 28.03 GB; peak_reserved 28.7617 GiB;
  wall time 1998.8s.
- WANDA Qwen3-8B prefix80, common 2:4 mask: token PPL 15.6914;
  mean NLL 2.7531; peak memory 27.61 GB; wall time 31.85s.
- Runtime activation N:M Qwen3-8B prefix80, common 2:4: token PPL 11.3428;
  mean NLL 2.4286; peak memory 27.61 GB; wall time 33.03s.

Interpretation: single-GPU OOM feasibility is shown across MX-only, uniform MSD,
fixed-sum calibrated MSD, WANDA, and activation N:M. Calibrated/uniform MSD
runtime remains the main bottleneck.

## Model-Sharded PPL Smoke Evidence

2026-05-24 direct-CUDA tiny smokes validated the explicit single-process
`ppltest.py --device-map sequential` path on Qwen3-0.6B with physical GPUs
4,5,6,7 visible. These are correctness smokes only, not timing estimates.

Common command shape:

```bash
../.venv3_10/bin/python ppltest.py \
  --model-path ../Qwen3-0.6B \
  --gpus 4,5,6,7 \
  --device-map sequential \
  --max-memory 0:1GiB,1:4GiB,2:4GiB,3:4GiB \
  --text-manifest README.md \
  --limit-samples 2 \
  --stats off \
  --mxfp-progress-interval-sec -1
```

The equivalent non-sharded comparisons used `--gpus 4` and
`--device-map none`. The sequential sharded map placed layers on visible CUDA
devices 0 and 1.

| Setup | Path | Scored tokens | Token PPL delta | Mean NLL delta | Sharded CUDA peak allocation |
|---:|---|---:|---:|---:|---|
| 2 | MXFP8 | 22 | 0.0 | 0.0 | cuda:0 1.7282 GiB; cuda:1 0.2849 GiB |
| 6 | MXFP8 + uniform MSD B=16 | 22 | 0.0 | 0.0 | cuda:0 4.2715 GiB; cuda:1 2.8278 GiB |

Two sharded bugs were fixed during this validation:

- Tensor `logits_to_keep` can be moved by Accelerate to the input device, so
  `Qwen3ForCausalLM.forward()` now moves tensor slice indices to
  `hidden_states.device` immediately before indexing.
- In chunked tail-logits loss, `lm_head` can live on a different device from
  final hidden states. The loss accumulator now follows the actual
  `lm_head`/cross-entropy output device.

2026-05-24 direct-CUDA prefix validation on Qwen3-8B setup 2 covered a
`--limit-samples 80` WikiText-2 prefix with 4,144 tokens and two PPL windows,
including one full 4,096-token context window. The single-GPU reference used
physical GPU 4. The sharded comparison used physical GPUs 4,5,6,7 visible with
`--device-map sequential --max-memory 0:12GiB,1:12GiB,2:12GiB,3:12GiB`; the
resolved map placed layers on visible CUDA devices 0 and 1.

| Model | Setup | Path | Scored tokens | Token PPL | Mean NLL | Wall time | CUDA peak allocation |
|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-8B | 2 | single GPU, `--device-map none` | 4,144 | 8.8440 | 2.1797 | 31.97s | cuda:0 27.6147 GiB |
| Qwen3-8B | 2 | sequential sharded, visible GPUs 4-7 | 4,144 | 8.8440 | 2.1797 | 31.80s | cuda:0 19.8733 GiB; cuda:1 9.3105 GiB; cuda:2/3 0.0 GiB |

Interpretation: explicit single-process model sharding now has prefix-level
correctness evidence on Qwen3-8B MXFP8 with an actual 4,096-token context
window. The initial sequential placement lowered per-GPU memory but did not
materially change MXFP8 wall time on this two-window prefix. Next validation:
repeat the prefix comparison for setup 6 or calibrated fixed-sum MSD before
accepting sharded MSD wall-time estimates.

## Fixed-Sum Calibration Evidence

- Targeted Qwen3-8B calibration smoke on one projection completed:
  `projection-filter model.layers.0.mlp.gate_proj`, `num-texts=1`,
  `max-length=64`, target SNR 10; total_channels 12,288; budget range [4, 9];
  budget_mean 6.09; mean_snr 12.84 dB; wall_time 6.68s.
- Gate/up/down projection-subset fixed-sum calibrations completed with
  `--weight-cache-dtype none`; merged staged metadata covered 108 MLP projection
  entries and 1,032,192 channels.
- Broad projection-filtered calibration can still accumulate unrelated MXFP
  forward caches if a persistent cache is enabled. Use
  `--weight-cache-dtype none` for broad calibration capture until that runtime
  is further refined.

## Figure 4 Equivalent-Work Evidence

The equivalent sparsity/work coordinate for Figure 4 is
`plot_norm_digit_read`. The original calculation is in the figure repo history:
`../figure` commit `d60ad9c`, `figure4/extract_figure4_data.py`.

```text
plot_norm_digit_read = mean_effective_precision / 3.0
```

For fixed-sum and uniform rows, the extractor uses capped
`cap_e_eff / 3.0` when cap100 data exists, otherwise uncapped
`uncap_e_p_eff / 3.0`. The current
`../figure/figure4/prepare_figure4_plot_data.py` carries this value into the
plot x-axis. Current `figure4_plot_data.csv` has the following fixed-sum
points; the implied effective precision is `plot_norm_digit_read * 3.0`.

| Fixed-sum target SNR | `plot_norm_digit_read` | Implied `mean_effective_precision` |
|---:|---:|---:|
| 15 dB | 0.4133 | 1.2399 |
| 18 dB | 0.5170 | 1.5511 |
| 20 dB | 0.5858 | 1.7575 |
| 25 dB | 0.7268 | 2.1803 |
| 30 dB | 0.8700 | 2.6100 |
| 35 dB | 0.9500 | 2.8500 |

Fixed-sum target-SNR 30 dB is therefore close to dense digit-read work. Do not
use runtime `msd_perf_stats.global.global_utilization` as the Figure 4
equivalent sparsity/work axis.

## Old Runtime-Stat Artifacts

Former 0.6B fixed-sum PPL artifacts under
`../data/calib-data/<xdb>/ppl_results_MXFP8_fix_{cap,time}.json` were generated
with the old explicit command shape:

```bash
python ppltest.py --nproc 1 --setup 6 \
  --calibration ../data/calib-data/27db/calibration_MXFP8_fixed_sum.json \
  --lite \
  --output ../data/calib-data/27db/ppl_results_MXFP8_fix_time.json \
  --limit-samples 100 \
  --figure5-layer-cycles \
  --gpus 6
```

Those files were removed from version control by:

```text
e0a38ec908233cbbf21c56262e5b79963a212660 Clean up repo docs and remove obsolete scripts
```

Do not restore them. Current standard fixed-sum MSD timing/utilization probes
should use:

```bash
python ppltest.py --nproc 1 --setup 6 \
  --calibration <calibration_MXFP8_fixed_sum.json> \
  --msd-utilization-mode \
  --output <ppl_results_MXFP8_fix_time.json> \
  --gpus <id>
```

`--msd-utilization-mode` is the maintained 100-sample lite-stat mode. Add
`--figure5-layer-cycles` only when debugging Figure 5 cycle accounting.

Observed old fixed-sum MXFP8 runtime `global_utilization`:

| Target SNR | Output variants checked | Runtime `global_utilization` |
|---:|---|---:|
| 12 dB | `_fix_cap`, `_fix_time` | 0.121 |
| 15 dB | `_fix_cap`, `_fix_time` | 0.149 |
| 20 dB | `_fix_cap`, `_fix_time` | 0.190 |
| 25 dB | `_fix_cap`, `_fix_time` | 0.215 |
| 30 dB | `_fix_time` | 0.225 |

These are runtime diagnostics only. They are useful to record for performance
accounting, but they are not equivalent sparsity/work values.

## Invalid Evidence

- Ignore 2026-05-21 sandbox progress files without `cuda_*` fields. They were
  CPU or CUDA-invisible runs.
- Generated probe JSON/log files are not committed; rely on summarized,
  dated evidence in this file and fresh direct-CUDA reruns.
