# onlinearith

Experiment drivers for MXFP/MSD Qwen3 simulation and temporal significance scheduling studies.

This repository contains evaluation, calibration, distributed helper, plotting, and lightweight validation scripts. The modified Qwen3 model code lives in the sibling Transformers checkout, normally at:

```text
../transformers/src/transformers/models/qwen3/
```

## Active Source Layout

- `ppltest.py`: single-setup WikiText-2 PPL evaluation.
- `ppl_batch.py`: batch PPL runner across setup IDs.
- `calibrate.py`: MXFP/MSD budget calibration driver.
- `calibrate_base.py`: structured n:m baseline mask calibration.
- `experiment_config.py`: setup IDs, setup tags, baseline config fields, and config application helpers. Treat this as the source of truth for setup definitions.
- `dist_utils.py`: shared distributed helpers.
- `test_mxfp8linear.py`, `test_fixed_sum_optimizer.py`, `test_distributed.py`: lightweight validation scripts.
- `../transformers/src/transformers/models/qwen3/modeling_qwen3.py`: current operational Qwen3 implementation. Patch this file for current MXFP/MSD runtime work unless modular-converter work is explicitly requested.
- `../transformers/src/transformers/models/qwen3/modular_qwen3.py`: reference/modular source only. Do not regenerate `modeling_qwen3.py` from it during cleanup.

## Setup

Expected local layout:

```text
coding/
  .venv3_10/
  onlinearith/
  transformers/
  Qwen3-0.6B/
```

Use the parent-directory virtualenv:

```bash
cd /path/to/onlinearith
source ../.venv3_10/bin/activate
export PYTHONPATH="$(pwd)/../transformers/src:${PYTHONPATH}"
```

Commands in this README can also be run explicitly as `../.venv3_10/bin/python ...`.

## Common Commands

```bash
python ppltest.py --list
python ppltest.py --setup 6 --lite --limit-samples 2
python ppl_batch.py --list
python calibrate.py --list
python test_mxfp8linear.py
python test_fixed_sum_optimizer.py
```

Multi-process examples:

```bash
python ppltest.py --nproc 4 --gpus 4,5,6,7 --setup 6
python ppl_batch.py --nproc 8 --only 2 6 10
torchrun --nproc_per_node=2 test_distributed.py
```

## PPL Methodology Invariants

Full PPL uses WikiText-2 raw test, `MAX_LENGTH = 4096`, `STRIDE = 512`, masked context labels, and weighted NLL accumulation (`loss * trg_len` divided by total scored tokens). Do not average window losses directly.

`--limit-samples` is only a smoke-test shortcut. Results produced with sample limits are not final PPL numbers.

## Distributed Note

Current `--nproc` behavior is data-parallel execution with full model replicas. Each rank loads a complete model copy. This can improve throughput for smaller runs, but it is not model sharding and does not solve Qwen3-8B per-GPU OOM by itself.

## Documentation

- Cleanup status and quality gates: `docs/codex/`, `tools/`, `tests/`, and `scripts/run_repo_quality_gate.sh`
- Baseline notes: `docs/baselines/`
- Calibration notes: `docs/calibration/`
- Developer notes, including modular converter details: `docs/dev/`
- Archived/obsolete material: `docs/archive/`

Deep-pipeline material is archived/abandoned unless explicitly requested. Existing setup IDs are preserved for compatibility, but deep pipeline is not part of the immediate OOM-fix path.

## Next Planned Work

After cleanup, the focused implementation task is:

1. Add an exact output-chunked MX-only baseline path in `_MXFPLinearBase` without changing old MX math.
2. Separate lite/full statistics overhead from PPL numerical behavior so logits, labels, loss, and PPL stay unchanged.

Do not change `MAX_LENGTH`, `STRIDE`, dataset split, tokenizer behavior, calibration semantics, setup IDs, or result schemas as part of cleanup.
