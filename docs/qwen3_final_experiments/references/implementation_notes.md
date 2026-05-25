# Qwen3 OOM/Performance Implementation Notes

This file preserves details that are useful for debugging but too large for the
standard session handoff.

## Completed Core Work

- Exact output-chunked MX-only path is implemented in `_MXFPLinearBase`.
- Compact MXFP weight cache is implemented:
  - `float32`: historical cache behavior.
  - `float16`: compact cache for MXFP4/6/8.
  - `float8`: native `torch.float8_e4m3fn` cache for MXFP8 only.
  - `none`: no persistent quantized-weight cache.
- `clear_mxfp_weight_cache(model)` exists in `experiment_config.py` and is used
  by runners after setup transitions and probes.
- PPL/probe runtime flags are implemented:
  - `--stats {off,lite,full}`
  - `--mx-chunk-target-mib`
  - `--msd-chunk-target-mib`
  - `--weight-cache-dtype`
  - `--compile-msd-truncate`
  - `--msd-utilization-mode` for standard fixed-sum MSD timing/utilization
    probes.
- `ppltest.py` has an explicit single-process model-sharding entry point:
  - `--device-map {none,auto,sequential,balanced}`
  - `--max-memory 0:30GiB,1:30GiB,...`
  - sharded modes are rejected with `--nproc > 1` or torchrun world size > 1;
  - result JSON records `device_map`, `max_memory`, visible CUDA devices,
    input device, resolved Hugging Face device map, and per-visible-device
    CUDA peak memory.
- Tail-logits chunked loss is implemented and covered by tests.
- Tail-logits index tensors are created on CPU in `ppl_utils.py`. In
  `Qwen3ForCausalLM.forward()`, tensor slice indices are moved to
  `hidden_states.device` immediately before indexing because Accelerate may
  move kwargs to the input device in a sharded model.
- Chunked tail-logits loss accumulates on the actual cross-entropy output
  device. This matters when `lm_head` is on a different CUDA device from final
  hidden states.
- `reconfigure_mlp_layers(model, device=None)` preserves each replaced MLP
  projection's existing weight device. Use this for explicitly sharded PPL
  models so MX/MSD replacement layers do not collapse back to one GPU.
- MXFP/MSD progress events include `module_device` and report CUDA memory for
  the active module device when available.
- MSD inference computes total delay once, extracts `max_delay_chunk` when
  stats are enabled, then transforms that tensor in-place into `p_eff`. This
  avoids materializing both `total_delay` and `p_eff` as full 4D tensors in
  stats-off and stats-lite modes. Chunk-local tensors such as `combined_e`,
  `w_q_c`, `b_final_c`, and inter-block delays are also freed earlier before
  the truncation call.
- `_forward_msd_truncated` allocates its output buffer with `torch.empty`
  instead of `torch.zeros`; every output-channel slice is assigned exactly once
  by the chunk loop.
- `_forward_mx_exact_chunked` and `_forward_msd_truncated` cache the optional
  progress hook before entering the chunk loop and only build progress payloads
  when the hook is installed.
- Shared MXFP/MSD progress reporting is wired into probe, `ppltest.py`,
  `ppl_batch.py`, and the ladder script.
- `--gpus` is applied before torch import in probe, `ppltest.py`, `ppl_batch.py`,
  and parity baseline runners so physical GPU selection is honored before CUDA
  state is cached.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set by default before torch
  import in probe, `ppltest.py`, `ppl_batch.py`, and parity baseline runners.
- PPL always forces `use_cache=False`.

## Sibling Transformers Notes

Active files live in `../transformers/src/transformers/models/qwen3/`:

- `modeling_qwen3.py`: operational implementation. Treat this as authoritative.
- `configuration_qwen3.py`: custom MXFP/MSD config fields.
- `calibration_msd.py`: calibration implementation imported by `calibrate.py`.
- `msd_perf_stats.py`: performance-statistics accumulator.

Important implementation details:

- MSD truncation uses frexp/ldexp scale reconstruction and avoids the old final
  zero-mask allocation.
- NAF-width extraction uses frexp-style exponent logic.
- The old April 9 digit-cap fix is present through `_resolve_lite_p_eff_cap`
  and `MSDPerfAccumulator(..., lite_p_eff_cap=...)`.
- Activation N:M uses common keep-count notation: keep N values per M group,
  internally prune `(M - N)`.
- `float8` weight cache is only valid for `MXFP8Linear`; non-MXFP8 paths should
  use `float16`, `float32`, or `none`.
- With explicit `ppltest.py --device-map`, Accelerate can place `lm_head` and
  final layers on different visible CUDA devices. Keep tensor index movement
  and loss-device handling local to `Qwen3ForCausalLM.forward()` rather than
  depending on caller-side tensor placement.

## Baseline Parity Status

The following paths have comparable runner hygiene:

- MX-only and uniform MSD in `ppltest.py` / `ppl_batch.py`.
- Fixed-sum calibrated MSD:
  - calibration generation has chunk/cache/GPU/compile/projection-filter
    controls;
  - `tools/merge_msd_calibrations.py` merges disjoint projection-filtered
    calibration JSONs into one PPL-ready `msd_calibration_data` map;
  - `ppltest.py --calibration` injects metadata into the optimized MSD runtime.
- WANDA:
  - `wanda_base/calibrate_base.py`
  - `wanda_base/ppl_batch_base.py`
- Runtime activation N:M:
  - `act_base/ppl_batch_base_act.py`

Parity controls include early `--gpus`, allocator default, `use_cache=False`,
shared PPL utilities, tail-logits loss, MX chunk/cache controls, progress hooks,
cache cleanup, and `--limit-samples` smoke support.

## Historical Initial Plan

The initial OOM diagnosis was:

1. The exact MX path materialized tensors shaped `(num_blocks, tokens,
   out_features)`. For Qwen3-8B gate/up projections this is about 24 GiB for a
   single fp32 tensor.
2. Persistent quantized-weight cache stored all `w_q` tensors as fp32, adding
   tens of GiB after enough layers had been visited.
3. Full MSD stats and large chunk targets produced excessive temporary memory.

The core fixes were output-channel chunking, compact/bounded weight caches,
stats controls, `use_cache=False`, progress reporting, and calibration/baseline
runner parity.

## Live Contract Tests

Run the cheapest relevant checks after repo or runner edits:

```bash
../.venv3_10/bin/python tests/test_msd_truncate_equivalence.py
../.venv3_10/bin/python tests/test_msd_stats_off_equivalence.py
../.venv3_10/bin/python tests/test_ppl_device_map_utils.py
../.venv3_10/bin/python tests/test_mx_exact_chunked.py
../.venv3_10/bin/python tests/test_mxfp_weight_cache_compact.py
../.venv3_10/bin/python tests/test_ppl_tail_logits_loss.py
../.venv3_10/bin/python tests/test_nm_keep_semantics.py
../.venv3_10/bin/python test_mxfp8linear.py
../.venv3_10/bin/python test_fixed_sum_optimizer.py
../.venv3_10/bin/python ppltest.py --list
../.venv3_10/bin/python ppl_batch.py --list
../.venv3_10/bin/python calibrate.py --list
```
