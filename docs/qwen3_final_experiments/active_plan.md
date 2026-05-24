# Active Qwen3 Final Experiment Plan

This file is the concise active plan. Historical implementation details live
under `docs/qwen3_final_experiments/references/`.

## Scope

Prepare the final experiment execution path and wall-time estimates for the
focused Qwen3 model family.

Representative paths:

- MXFP8 baseline: `ppltest.py --setup 2`
- Fixed-sum calibrated MSD: `calibrate.py --optimizer fixed_sum --target-snr 30`
  plus `ppltest.py --setup 6 --calibration <fixed_sum.json>`
- WANDA structured baseline: common keep-count `2:4`
- Runtime activation N:M baseline: common keep-count `2:4`

## Invariants

- Preserve PPL methodology: WikiText-2 raw test split, `MAX_LENGTH=4096`,
  `STRIDE=512`, masked context labels, and weighted NLL accumulation.
- Preserve setup IDs, result JSON schemas, calibration JSON schemas, tokenizer
  behavior, and calibration semantics.
- For MSD equivalent-work comparisons, use
  `plot_norm_digit_read = mean_effective_precision / 3.0`.
- `--limit-samples` and `--msd-utilization-mode` are non-final probes unless
  explicitly labeled otherwise.
- `ppltest.py --nproc` is data-parallel window sharding with one full model
  replica per process. It is valid for final PPL wall-time acceleration when
  each selected GPU can fit a full replica, but it is not model sharding and
  is not an OOM solution.
- Different model sizes may need different execution recipes. Before a full
  final run, settle the model/path-specific recipe in
  `model_execution_matrix.md` and validate it on a prefix with a full
  4096-token context window.

## Current Work

1. Use full-replica data parallelism (`ppltest.py --nproc`) as the final PPL
   acceleration path when replicas fit. Qwen3-8B MXFP8 and fixed-sum 30 dB MSD
   are prefix-validated on eight replicas when using `--load-stagger-sec 8`.
   For Qwen3-8B MSD, include `--weight-cache-dtype float8`; the default float16
   persistent cache OOMed in a two-worker fixed-sum prefix run.
2. Use baseline-runner window sharding for the representative WANDA and
   activation N:M baselines. `wanda_base/ppl_batch_base.py` and
   `act_base/ppl_batch_base_act.py` need `--window-shard` with `--nproc` for a
   single final setup; their default `--nproc` behavior shards setup IDs.
   Qwen3-8B WANDA 2:4 and activation N:M 2:4 are prefix-validated on eight
   replicas with `--window-shard --load-stagger-sec 8`.
3. Treat current `--device-map sequential` placement as memory relief only.
   Do not claim model-parallel speedup unless `balanced` or a manual placement
   policy beats single-GPU and data-parallel timing with direct-CUDA evidence.
4. Do not treat MSD stats from a current `--nproc` run as full-dataset work
   aggregates: nonzero ranks disable MSD stats. Use `--nproc` for PPL quality
   and wall time, and use a separate single-process utilization/accounting run
   or add rank-level stats aggregation before reporting aggregate work metrics.
5. For fixed-sum calibration, prefer task parallelism over model sharding:
   run projection-filtered full-replica jobs on separate GPUs, then merge the
   resulting metadata with the established staged-calibration workflow.
6. Update `docs/qwen3_final_experiments/runtime_estimates.md` with measured
   single-GPU and multi-GPU wall-time estimates as each representative path is
   validated.
7. Keep generated calibration/result artifacts out of commits unless explicitly
   requested.

## Sharding Guardrails

- Keep the two multi-GPU modes distinct:
  `--nproc` means data-parallel PPL window sharding with one full model replica
  per process; `--device-map` means single-process layer placement across GPUs.
- Do not combine model sharding with `--nproc` in the current runner.
- Use `--nproc` for final PPL acceleration only when direct CUDA is visible and
  every selected GPU has enough memory for a full model replica.
- Do not silently use `device_map="auto"` as a default. Sharding must be
  opt-in and visible in output metadata.
- Use visible-device IDs in `--max-memory`, after `--gpus` has narrowed
  `CUDA_VISIBLE_DEVICES`; for example `--gpus 4,5,6,7 --max-memory
  0:30GiB,1:30GiB,2:30GiB,3:30GiB`.
- Keep custom Qwen3 MX/MSD modules on the device of the layer they replace.
- Input tensors should enter on the model's input embedding device; layer
  dispatch should then follow the sharded model layout.
- Validate loss equality on tiny windows before trusting timing.

Detailed design notes: `references/multigpu_sharding_plan.md`.

## Cheap Contracts

```bash
../.venv3_10/bin/python tests/test_msd_truncate_equivalence.py
../.venv3_10/bin/python tests/test_msd_stats_off_equivalence.py
../.venv3_10/bin/python tests/test_ppl_device_map_utils.py
../.venv3_10/bin/python tests/test_mx_exact_chunked.py
../.venv3_10/bin/python tests/test_mxfp_weight_cache_compact.py
../.venv3_10/bin/python tests/test_ppl_tail_logits_loss.py
../.venv3_10/bin/python tests/test_nm_keep_semantics.py
../.venv3_10/bin/python test_fixed_sum_optimizer.py
../.venv3_10/bin/python ppltest.py --list
../.venv3_10/bin/python ppl_batch.py --list
../.venv3_10/bin/python calibrate.py --list
```
