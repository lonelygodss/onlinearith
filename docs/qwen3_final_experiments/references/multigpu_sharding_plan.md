# Multi-GPU Sharding Preparation

This is a read-when-needed design note for the next iteration. Keep the active
handoff concise; update this file with implementation details as they emerge.

## Goal

Add and validate explicit multi-GPU execution modes for final Qwen3 experiments.
There are two separate modes:

- full-replica data parallel PPL window sharding with `ppltest.py --nproc`,
  which accelerates wall time when each GPU can fit a complete model;
- single-process model sharding with `ppltest.py --device-map`, which reduces
  per-GPU memory but has not shown throughput improvement in current evidence.

The first target is PPL. Calibration sharding should be considered after PPL
sharding is validated, because calibration has additional activation-capture and
projection-filter behavior.

## Non-Goals

- Do not change PPL math, dataset, tokenizer, window constants, labels, or loss
  weighting.
- Do not use `ppltest.py --nproc` as model sharding. It launches multiple full
  model replicas and shards windows. It is acceptable as the final PPL
  acceleration method when full replicas fit.
- For Qwen3-8B MSD full-replica multi-GPU PPL, use `--weight-cache-dtype
  float8` unless newer evidence supersedes it. A two-worker fixed-sum prefix
  run with the default float16 persistent cache OOMed, while the float8-cache
  run preserved PPL and completed.
- Qwen3-8B full-replica `--nproc` is validated up to four workers for MXFP8
  and fixed-sum 30 dB MSD. An eight-worker MXFP8 launch on GPUs 0-7 SIGKILLed
  during model loading/materialization before evaluation, so validate launch
  scale separately from PPL correctness before using eight workers for final
  estimates.
- Do not report current `--nproc` `msd_perf_stats` as a full-dataset aggregate:
  nonzero ranks disable MSD stats. Add explicit stats aggregation or use a
  separate single-process accounting run for work metrics.
- Do not make `device_map="auto"` an implicit default. Sharding must be opt-in
  and visible in result metadata.
- Do not combine initial model sharding with `--nproc`; validate single-process
  model sharding first.

## Proposed CLI Shape

`ppltest.py` now has opt-in model placement controls:

```text
--device-map {none,auto,sequential,balanced}
--max-memory 0:30GiB,1:30GiB,...
```

Defaults:

- `--device-map none`: current behavior.
- If a sharded mode is selected, `ppltest.py` does not call `model.to(device)`.
- `--device-map` is rejected with `--nproc > 1` or torchrun world size > 1.
- `--max-memory` uses visible CUDA ordinals after `--gpus` filtering. For
  physical GPUs 4,5,6,7, use `--gpus 4,5,6,7 --max-memory
  0:30GiB,1:30GiB,2:30GiB,3:30GiB`.
- Output JSON records `device_map`, `max_memory`, visible CUDA devices, input
  device, and the resolved Hugging Face device map when available.

## Implementation Checklist

1. Load the model with explicit placement only when `--device-map != none`.
2. Ensure `reconfigure_mlp_layers()` creates replacement MX/MSD modules on the
   same device as the original layer weights. Done for PPL through
   `reconfigure_mlp_layers(model, device=None)`.
3. Ensure persistent MXFP weight caches remain layer-local and device-local.
4. Place `input_ids` on the input embedding device for sharded forwards. Done
   for PPL through `model_input_device()`.
5. Keep tail-logits loss behavior unchanged; labels must follow the logits or
   model output device used by the loss path.
6. Ensure progress and memory reporting identify devices clearly. Progress
   events now include the active module device.
7. Add output metadata for sharding mode and per-device memory when available.
   Done for PPL output JSON.

## Validation Ladder

Use direct CUDA only.

1. Verify CUDA visibility:

```bash
../.venv3_10/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

2. Run non-sharded tiny PPL smoke and capture the loss/output metadata.
3. Run sharded tiny PPL smoke with the same setup and `--limit-samples 2`.
4. Compare scored tokens and token PPL within expected deterministic tolerance.
5. Run a prefix smoke large enough to include a 4096-token context window.
6. Only after correctness is established, add wall-time estimates to
   `docs/qwen3_final_experiments/runtime_estimates.md`.

## Open Questions

- Whether manual sequential placement is preferable to automatic placement for
  the custom Qwen3 modules.
- Whether calibration should use the same sharded loader or a separate
  projection-subset capture workflow.
- Whether `--device-map balanced` or manual placement can beat single-GPU
  runtime. Current sequential evidence says model sharding is memory relief
  only; `--nproc` full-replica window sharding is the preferred final speed path
  when memory allows it.
