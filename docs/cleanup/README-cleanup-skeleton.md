# onlinearith

Experiment drivers for MXFP/MSD Qwen3 simulation and temporal significance scheduling studies.

This repository is the driver/evaluation side of the project. The modified model code lives in the sibling Transformers fork.

## Active source layout

```text
onlinearith/
  ppltest.py              # single-setup PPL evaluation
  ppl_batch.py            # batch PPL evaluation over setup IDs
  calibrate.py            # MXFP/MSD calibration driver
  calibrate_base.py       # structured n:m baseline calibration
  experiment_config.py    # setup IDs and custom config defaults
  dist_utils.py           # torchrun/NCCL helpers
  AGENTS.md              # Codex repository guidance

../transformers/src/transformers/models/qwen3/
  modeling_qwen3.py      # current operational Qwen3 implementation
  configuration_qwen3.py # custom config fields
  calibration_msd.py     # calibration implementation
  msd_perf_stats.py      # performance-statistics implementation
```

`modeling_qwen3.py` is the operational source for the current simulation. Do not regenerate from `modular_qwen3.py` unless the task is explicitly about the modular converter and the generated diff is reviewed carefully.

## Setup

Expected local layout:

```text
/path/to/workspace/
  onlinearith/
  transformers/
  Qwen3-0.6B/
  data/
```

Typical shell setup:

```bash
cd /path/to/workspace/onlinearith
source /path/to/venv/bin/activate
export PYTHONPATH="$(pwd)/../transformers/src:${PYTHONPATH}"
```

The scripts default to local model paths such as `../Qwen3-0.6B` and use `local_files_only=True`. Keep model downloads and large data outside this git repo.

## Common commands

List experiment setups:

```bash
python ppltest.py --list
python ppl_batch.py --list
python calibrate.py --list
```

Single-setup PPL smoke tests:

```bash
python ppltest.py --setup 1 --limit-samples 2
python ppltest.py --setup 6 --lite --limit-samples 2
```

Full PPL example:

```bash
python ppltest.py --setup 6 --lite
```

Batch PPL examples:

```bash
python ppl_batch.py --only 1 2 6
python ppl_batch.py --nproc 8 --only 2 6 10
```

Calibration examples:

```bash
python calibrate.py --setup 1
python calibrate.py --setup 1 --optimizer fixed_sum
python calibrate.py --nproc 4 --only 1 2
```

Validation scripts:

```bash
python test_mxfp8linear.py
python test_fixed_sum_optimizer.py
python test_distributed.py
```

## PPL methodology invariants

Final PPL uses:

- dataset: `wikitext/wikitext-2-raw-v1/test`
- `MAX_LENGTH = 4096`
- `STRIDE = 512`
- context labels masked with `-100`
- weighted NLL accumulation: sum `loss * trg_len`, divide by total scored tokens

`--limit-samples` is only a smoke-test shortcut. Do not use it for final paper numbers.

## Distributed behavior

Current `--nproc` mode is data parallel:

- `ppltest.py` shards windows across ranks.
- `ppl_batch.py` shards setup IDs across ranks.
- each rank loads a full model copy.

This improves throughput for small models but does not reduce per-GPU memory. Qwen3-8B memory fixes should use explicit memory-neutral simulator changes and, later if needed, model sharding. Do not treat `--nproc 8` as model parallelism.

## Active setup source

Setup IDs and config overrides are defined in `experiment_config.py`. Do not duplicate the setup table in other scripts or docs. Update README summaries only after changing `experiment_config.py` intentionally.

## Archived or secondary material

- Deep-pipeline notes are archived/abandoned unless explicitly requested.
- Long calibration and baseline notes live under `docs/`.
- Generated results, plots, caches, model weights, and data outputs should not be committed.

## Next planned implementation work

After cleanup, the next focused implementation task is:

1. Add an exact output-chunked MX-only baseline path so the MXFP baseline does not allocate full `(num_blocks, tokens, out_features)` intermediates.
2. Make lite/full performance-statistics behavior explicit so PPL calculation is unchanged while expensive stats are skipped when requested.

Do not solve OOM by changing the PPL methodology.
