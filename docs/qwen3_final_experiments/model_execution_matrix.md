# Model Execution Matrix

This file settles the per-model execution policy before final runs. It is a
recipe matrix, not a new methodology. PPL math, datasets, tokenizer behavior,
window constants, labels, and weighted NLL accumulation remain unchanged.
Concrete Qwen3-8B launch commands live in `final_run_commands.md`.

## Global Defaults

Use these defaults unless a model/path row below overrides them.

- PPL numerical runs: `--stats off`.
- MSD PPL runs: `--compile-msd-truncate`.
- Full-replica acceleration: `ppltest.py --nproc`, which shards PPL windows and
  loads one complete model replica per worker.
- Baseline-runner single-setup acceleration: `wanda_base/ppl_batch_base.py` and
  `act_base/ppl_batch_base_act.py` require `--window-shard` with `--nproc`;
  their default `--nproc` behavior shards setup IDs, not PPL windows.
- Model sharding: `ppltest.py --device-map ...`, single process only, memory
  relief only unless direct-CUDA timing proves speedup.
- Fixed-sum calibration: projection-filtered task-parallel full-model jobs,
  then merge metadata; do not use model sharding as the default calibration
  trick.
- Broad calibration capture: `--weight-cache-dtype none` to avoid persistent
  cache accumulation during projection-filtered runs.
- Qwen3-8B multi-rank loading: use `--load-stagger-sec 8` when launching eight
  full replicas.

## Final PPL Recipes

| Model | MXFP8 PPL | Fixed-sum MSD 30 dB PPL | WANDA 2:4 PPL | Activation N:M 2:4 PPL |
|---|---|---|---|---|
| Qwen3-0.6B | Single GPU is acceptable; use `--nproc` only for turnaround. | Single GPU or job-packed `--nproc`; default float16 cache is acceptable unless memory evidence says otherwise. | Single GPU or job-packed `--nproc`. | Single GPU or job-packed `--nproc`. |
| Qwen3-1.7B | Prefer `--nproc 8` after one prefix validation. | Prefer `--nproc 8` after one prefix validation; keep default float16 cache unless a prefix run shows memory pressure. | Prefer `--nproc 8` after one prefix validation. | Prefer `--nproc 8` after one prefix validation. |
| Qwen3-4B | Prefer `--nproc 8` after one prefix validation. | Prefer `--nproc 8` after one prefix validation; consider float8 cache only if default float16 cache is near OOM. | Prefer `--nproc 8` after one prefix validation. | Prefer `--nproc 8` after one prefix validation. |
| Qwen3-8B | Use `--nproc 8 --load-stagger-sec 8`. | Use `--nproc 8 --load-stagger-sec 8 --weight-cache-dtype float8`. | Use baseline runner `--nproc 8 --window-shard --load-stagger-sec 8` with the Qwen3-8B-shaped mask. | Use baseline runner `--nproc 8 --window-shard --load-stagger-sec 8`. |

## Validation Gates

Before committing to a final full PPL command for any model/path combination:

1. Verify direct CUDA: `../.venv3_10/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'`.
2. Run a prefix that includes at least one 4096-token context window.
3. Confirm scored tokens and PPL match the corresponding single-process or
   smaller-worker reference for the same prefix.
4. Confirm output JSON records CUDA memory fields and the intended execution
   mode (`world_size`, `device_map`, cache dtype, calibration file, and
   `load_stagger_sec` when used).
5. For MSD, confirm the chosen cache dtype. Qwen3-8B fixed-sum MSD currently
   requires `--weight-cache-dtype float8` for full-replica multi-GPU PPL.

## Current Settled Choices

- Qwen3-8B MXFP8: eight full replicas are validated with
  `--load-stagger-sec 8`; use this for final MXFP8 PPL.
- Qwen3-8B fixed-sum MSD: eight full replicas are validated with
  `--load-stagger-sec 8 --weight-cache-dtype float8`; use this for final
  fixed-sum MSD PPL.
- Qwen3-8B WANDA 2:4: eight full replicas are validated through
  `wanda_base/ppl_batch_base.py --window-shard --load-stagger-sec 8`; use a
  Qwen3-8B-shaped mask, not the committed 0.6B masks under
  `../data/wanda_base/2-4`.
- Qwen3-8B activation N:M 2:4: eight full replicas are validated through
  `act_base/ppl_batch_base_act.py --window-shard --load-stagger-sec 8`.
- Qwen3-8B model sharding: sequential `--device-map` is correctness-validated
  but slower for the tested prefix; keep it as memory relief, not final speed.
- Smaller models should not inherit Qwen3-8B-only tricks blindly. Use default
  float16 cache unless a model-specific prefix run shows memory pressure, and
  add `--load-stagger-sec` only when launch/load contention appears.
