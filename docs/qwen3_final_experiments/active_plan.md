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
  replica per process. Explicit model sharding must be a separate mode.

## Current Work

1. Repeat Qwen3-8B prefix validation for the slow MSD path, starting with setup
   6 and then fixed-sum target-SNR 30 dB if calibrated metadata is available.
2. Extend the same explicit placement discipline to calibration only after PPL
   sharding is validated.
3. Update `docs/qwen3_final_experiments/runtime_estimates.md` with measured
   single-GPU and multi-GPU wall-time estimates as each representative path is
   validated.
4. Keep generated calibration/result artifacts out of commits unless explicitly
   requested.

## Sharding Guardrails

- Start with one process controlling a sharded model; do not combine initial
  model sharding with `--nproc`.
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
