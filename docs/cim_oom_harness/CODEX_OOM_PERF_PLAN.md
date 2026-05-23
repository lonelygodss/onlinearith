# Codex implementation plan: Qwen3-8B PPL OOM and performance

This is an implementation brief for the `onlinearith` + sibling `transformers` checkout. The goal is to make Qwen3-8B PPL run on one RTX 5090 32 GB GPU for MX-only and MSD setups, then use the eight GPUs only for independent jobs/setups, not as data parallel replicas for one setup.

## Non-negotiable correctness invariants

1. Do not change PPL methodology: WikiText-2 raw test, `MAX_LENGTH=4096`, `STRIDE=512`, masked context labels, and weighted NLL accumulation stay unchanged.
2. Exact MX-only baseline math must be unchanged. Chunking may split the output channel dimension, but must not change block quantization, scale multiplication, or the order of reduction over blocks for a given output channel.
3. MSD math must stay unchanged. Only memory scheduling, stats collection, cache dtype/storage, and launch/evaluation plumbing should change.
4. Do not use `ppltest.py --nproc 8` as an OOM fix for Qwen3-8B. That path replicates the full model per GPU and only shards windows.
5. Keep old behavior available behind explicit flags or config values so existing small-model comparisons can be reproduced.

## Diagnosis to verify first

The current exact MX path allocates tensors shaped `(num_blocks, N, out)`. For Qwen3-8B, `hidden_size=4096`, `intermediate_size=12288`, and `MAX_LENGTH=4096`, so for gate/up projections `num_blocks=4096/32=128`, `N=4096`, `out=12288`. One fp32 `(128,4096,12288)` tensor is about 24 GiB. The current code creates `elem_dots`, `combined_scales`, and a product before summing, so setup 2 can OOM even before MSD is used.

The second big issue is persistent weight quantization cache. Current `_MXFPLinearBase.forward` stores `_w_cache=(w_q,w_scales,w_pad)`. With Qwen3-8B MLP weights, caching all `w_q` tensors as fp32 can add about 22 GiB on top of the original model weights. This must be compacted or made bounded; otherwise chunking alone may still OOM after enough layers have been cached.

The third issue is stats and MSD chunk sizing. MSD is already output-chunked, but `onlinearith` default `msd_chunk_target_mib=1536` is aggressive for 8B, especially when full stats compute `naf_width` and keep several 4D tensors alive. For PPL numbers, default stats should be off or lightweight.

## Phase 1: exact output-chunked MX-only baseline

Patch `../transformers/src/transformers/models/qwen3/modeling_qwen3.py` in `_MXFPLinearBase`.

Add config fields to Qwen3Config and to `onlinearith/experiment_config.py` defaults:

```python
"mxfp_use_chunked_exact": True,
"mxfp_chunk_target_mib": 256,
"mxfp_weight_cache_dtype": "float16",  # choices: "float32", "float16", "float8", "none"
```

Implement `_forward_mx_exact_chunked(self, x_q, x_scales, w_q, w_scales, N)` with identical math to the current non-MSD path, but chunk `j0:j1` over output channels:

```python
nb = x_q.shape[1]
out = w_q.shape[0]
target_bytes = int(getattr(cfg, "mxfp_chunk_target_mib", 256)) * 1024**2
chunk = max(1, target_bytes // (4 * N * nb))
chunk = min(chunk, out)
result = torch.empty((N, out), dtype=torch.float32, device=x_q.device)
x_bmm = x_q.permute(1, 0, 2).contiguous()  # (nb, N, bs)
x_scales_t = x_scales.t().unsqueeze(-1)    # (nb, N, 1)
for j0 in range(0, out, chunk):
    j1 = min(j0 + chunk, out)
    w_q_c = w_q[j0:j1]
    if w_q_c.dtype != torch.float32:
        w_q_c = w_q_c.float()
    dots = torch.bmm(x_bmm, w_q_c.permute(1, 2, 0).contiguous())  # (nb, N, c)
    scales = x_scales_t * w_scales[j0:j1].t().unsqueeze(1)        # (nb, N, c)
    result[:, j0:j1] = (dots * scales).sum(dim=0)
    del dots, scales, w_q_c
return result
```

Then replace the current non-MSD branch:

```python
result = self._forward_mx_exact_chunked(x_q, x_scales, w_q, w_scales, N)
```

Keep a fallback config (`mxfp_use_chunked_exact=False`) to run the old non-chunked block for small tests, but do not use it for Qwen3-8B.

Acceptance: `tests/test_mx_exact_chunked.py` passes. For a small deterministic layer, chunked output must match the old path exactly or within `atol=0, rtol=0` for fp32 results before final dtype cast. If exact equality fails only due to bmm kernel nondeterminism, document and use a very tight tolerance, but first try to preserve bitwise equality.

## Phase 2: compact or bounded weight cache

Current cache key only checks `data_ptr`; improve it to include weight `_version`, device, block size, and MX format. Implement `_get_quantized_weight_cache()` rather than inlining cache logic in `forward`.

Rules:

1. `mxfp_weight_cache_dtype="float32"`: old behavior for debugging small models.
2. `mxfp_weight_cache_dtype="float16"`: store `w_q` as fp16, store `w_scales` as fp32, and cast `w_q` chunks back to fp32 in chunked MX/MSD math. OCP MXFP4/6/8 representable normalized values should be exactly representable in fp16; validate with tests.
3. `mxfp_weight_cache_dtype="float8"`: MXFP8-only mode that stores `w_q` as native `torch.float8_e4m3fn`, keeps `w_scales` as fp32, and casts chunks back to fp32 for compute. This is the preferred Qwen3-8B cache mode for MXFP8 setups once exactness is validated.
4. `mxfp_weight_cache_dtype="none"`: no persistent cache; recompute per forward for lowest memory, useful for debugging OOM.

Add `clear_mxfp_weight_cache(model)` in `onlinearith/experiment_config.py` and call it after setup changes, before/after batch setup transitions, and after OOM probe runs. It should walk modules and set `_w_cache = None`, `_w_cache_key = None`, `_w_cache_data_ptr = None`, then run `gc.collect()` and `torch.cuda.empty_cache()` if CUDA is available.

Acceptance: `tests/test_mxfp_weight_cache_compact.py` passes, and `tools/probe_mxfp_memory.py --setup 2 --seq-len 4096` no longer shows a monotonic multi-GB cache increase across layers.

## Phase 3: PPL runner memory controls

Patch `ppltest.py` and `ppl_batch.py`.

Add flags:

```text
--stats {off,lite,full}      default: off for PPL numerical runs
--mx-chunk-target-mib N      forwards to model.config.mxfp_chunk_target_mib
--msd-chunk-target-mib N     forwards to model.config.msd_chunk_target_mib
--weight-cache-dtype {float16,float32,float8,none}
```

Backward compatibility:

* `--lite` remains accepted and maps to `--stats lite`.
* `--figure5-layer-cycles` implies `--stats lite`.
* If `--stats off`, set `model.config.msd_perf_stats_enabled=False` and skip `get_perf_stats()`.
* If `--stats full`, keep old full stats behavior.

In the PPL loop:

* Set `model.config.use_cache = False` after loading.
* Pass `use_cache=False` to `model(...)` if accepted.
* Prefer `torch.inference_mode()` over `torch.no_grad()` for evaluation, with a fallback if any custom code requires normal no-grad tensor behavior.
* Keep `target_ids = mask_context_labels(input_ids, trg_len)` unchanged.

Do not implement loss/logit slicing in this phase unless you also add a correctness test that compares per-window loss with the old full-logits path for many `trg_len` values. Logit slicing is a good later optimization but has easy off-by-one traps because CausalLM loss shifts logits and labels.

## Phase 4: optional model parallel path for models larger than one GPU

Only after single-GPU Qwen3-8B works, add an optional single-process model-sharded mode:

```text
--device-map {none,auto,sequential}
--max-memory 0:30GiB,1:30GiB,...
```

Implementation notes:

* If `device_map != none`, do not call `model.to(device)`.
* `reconfigure_mlp_layers()` must place new projections on `old.weight.device`, not a global device.
* PPL `input_ids` should be placed on the model input embedding device; let HF/Accelerate dispatch layer activations.
* Do not combine this with `--nproc` initially.

## Suggested acceptance ladder

Run from `onlinearith` with `PYTHONPATH=../transformers/src:$PYTHONPATH`.

0. Verify direct CUDA visibility in the exact command environment:
   `python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'`.
   Expected output on this machine is `True 8`. If it reports `False 0`, do not
   run GPU performance/OOM probes there. Valid probe progress includes `cuda_*`
   memory fields and should be visible in `nvtop`.
1. `python tests/test_mx_exact_chunked.py`
2. `python tests/test_mxfp_weight_cache_compact.py`
3. `python tools/probe_mxfp_memory.py --model-path ../Qwen3-8B --setup 1 --seq-len 256 --stats off`
4. `python tools/probe_mxfp_memory.py --model-path ../Qwen3-8B --setup 2 --seq-len 4096 --mx-chunk-target-mib 256 --weight-cache-dtype float8 --stats off`
5. `python tools/probe_mxfp_memory.py --model-path ../Qwen3-8B --setup 6 --seq-len 4096 --mx-chunk-target-mib 256 --msd-chunk-target-mib 256 --weight-cache-dtype float8 --stats off`
6. `python ppltest.py --model-path ../Qwen3-8B --setup 2 --gpus 0 --stats off --mx-chunk-target-mib 256 --weight-cache-dtype float8 --limit-samples 2`
7. Full setup 2 on one GPU.
8. Setup 6 with `--stats off`, then `--stats lite` only when collecting figure stats.

Do not proceed to full 8B PPL until the probe reports peak allocated and reserved memory with at least 2-4 GiB headroom on a 32 GiB GPU.

## Phase 5: parity for all paper-critical paths

The OOM/performance iteration is not complete until every important evaluation
path has the same runner hygiene and memory scheduling level as the MX baseline.
The paper-critical paths are:

1. MX-only baselines: setup 2/3/4/5 in `ppltest.py` and `ppl_batch.py`.
2. Uniform MSD baselines: setup 6+ without `--calibration`, especially setup 6
   for MXFP8 + uniform B=16.
3. Fixed-sum calibrated MSD: `calibrate.py --optimizer fixed_sum --target-snr ...`
   produces per-channel MLP budget metadata in `msd_calibration_data`; later
   `ppltest.py --setup <MSD setup> --calibration <calibration_..._fixed_sum.json>`
   consumes this metadata for calibrated MSD inference. This is a first-class
   method, not an optional calibration smoke.
4. WANDA structured sparsity baseline: `wanda_base/calibrate_base.py` plus
   `wanda_base/ppl_batch_base.py`.
5. Runtime activation n:m baseline: `act_base/ppl_batch_base_act.py`.

For both baseline sparsity paths, use common N:M notation: `N:M` means keep
`N` values per group of `M`. The implementation prunes `(M - N):M` internally.
Historical result directories generated before this convention fix may need
manual relabeling before comparison.

Current status at the start of this phase:

* MX-only and uniform MSD paths have the main Qwen3-8B memory controls:
  early `--gpus`, allocator default, chunk/cache controls, `use_cache=False`,
  tail-logits loss, stats controls, progress reporting, and compile toggle.
* `calibrate.py --optimizer fixed_sum` has the calibration-side controls and a
  targeted 8B layer smoke. `ppltest.py --calibration ...` uses the optimized MSD
  runtime once metadata is injected, and a small Qwen3-8B calibrated-MSD smoke
  has passed with generated fixed-sum metadata. An all-`gate_proj` fixed-sum
  calibration smoke also passed when calibration capture used
  `--weight-cache-dtype none`; the same run OOMed with the persistent float16
  forward weight cache because projection-filtered hooks do not stop unrelated
  MXFP forward caches from accumulating. Staged all-`up_proj` and all-`down_proj`
  calibration smokes also passed with cache disabled, and a merged full-MLP
  metadata file built from gate/up/down subsets completed a calibrated-MSD PPL
  smoke plus a longer non-final prefix80 PPL run. It still needs final
  calibrated metrics and, if required, a true single-run all-MLP calibration
  capture.
* WANDA and activation baseline runner parity is implemented in
  `wanda_base/calibrate_base.py`, `wanda_base/ppl_batch_base.py`, and
  `act_base/ppl_batch_base_act.py`. Small Qwen3-8B smokes and non-final
  prefix80 measurements have passed; final measurements remain.

Implementation work for this phase:

1. Refactor baseline PPL loops to reuse shared PPL utilities from `ppl_utils.py`:
   `mask_context_labels`, `prepare_tail_logits_loss_kwargs`,
   `accumulate_weighted_nll`, and `precompute_windows`, preserving
   `MAX_LENGTH=4096`, `STRIDE=512`, weighted NLL, and masked-context labels.
2. Add early `--gpus` parsing and `PYTORCH_ALLOC_CONF=expandable_segments:True`
   before importing torch in `wanda_base/calibrate_base.py`,
   `wanda_base/ppl_batch_base.py`, and `act_base/ppl_batch_base_act.py`.
3. Add `--mx-chunk-target-mib` and `--weight-cache-dtype` to WANDA calibration,
   WANDA PPL, and activation PPL; apply them before `reconfigure_mlp_layers()`
   and clear the MXFP weight cache after each setup.
4. Force `model.config.use_cache=False` and pass `use_cache=False` where model
   calls accept it in all baseline calibration/evaluation scripts.
5. Add `--limit-samples` and optional progress files to WANDA/activation PPL so
   the same non-final smoke discipline used by `ppltest.py` can be applied.
6. Validate calibrated-MSD with a real generated fixed-sum calibration file:
   first projection-filtered, then broader MLP subsets, then full 8B only after
   memory and runtime are understood. For broad calibration capture, prefer
   `--weight-cache-dtype none` until the calibration runtime can disable
   unrelated persistent MXFP forward caches automatically. Projection-subset
   metadata can be merged for staged full-MLP inference smoke tests because
   `msd_calibration_data` is keyed by projection module name.
7. Validate WANDA and activation baselines with Qwen3-8B smoke runs on direct
   CUDA after the runner changes. Do not treat sandbox CPU/CUDA-invisible logs
   as evidence.

Acceptance additions:

* `calibrate.py --optimizer fixed_sum --target-snr ...` can generate
  per-channel metadata for a staged Qwen3-8B subset without materializing
  unrelated projections.
* `ppltest.py --setup 6 --calibration <fixed_sum.json> --compile-msd-truncate`
  completes at least a non-final Qwen3-8B smoke and reports the calibration file
  in output JSON.
* Gate/up/down projection-subset fixed-sum metadata can be used individually and
  merged into one full-MLP staged metadata JSON for calibrated-MSD inference
  smoke testing.
* WANDA PPL and activation n:m PPL scripts use the same loss/window semantics and
  memory controls as `ppltest.py`.
* WANDA and activation baseline smoke runs on Qwen3-8B include valid `cuda_*` or
  peak memory evidence from a direct-CUDA environment.

Phase 5 status after the latest iteration:

* Single-GPU OOM feasibility is now demonstrated across the paper-critical
  paths: MX-only, uniform MSD setup 6, fixed-sum calibrated MSD, WANDA
  structured sparsity, and runtime activation n:m.
* Fixed-sum calibrated MSD using the staged merged full-MLP `target-snr`
  metadata completed `--limit-samples 80`: scored_tokens=4,144,
  token_ppl=8.8632, mean_nll=2.1819, peak_memory=28.03 GB,
  peak_reserved=28.7617 GiB, wall_time=1998.8s.
* WANDA prefix80 completed with token_ppl=15.6914, mean_nll=2.7531,
  peak_memory=27.61 GB, wall_time=31.85s. The runner expects the mask under
  `--results-root/<n>-<m>/`; this run used a symlink to the existing smoke mask.
* Runtime activation n:m prefix80 completed with token_ppl=11.3428,
  mean_nll=2.4286, peak_memory=27.61 GB, wall_time=33.03s.
* Native MXFP8 float8 cache is implemented and validated as exact against the
  float32 cache path. Setup 6 seq_len=4096 with `--weight-cache-dtype float8`
  completed with loss=0.01120709, peak_alloc=22.7762 GiB,
  peak_reserved=23.8984 GiB, reserved_headroom=7.4583 GiB, and cache total
  5.6953 GiB. The previous float16-cache probe had the same loss with
  peak_alloc=27.8387 GiB, peak_reserved=28.7617 GiB, and cache total
  10.7578 GiB.
* The current stage is done for OOM and runner-hygiene parity. Further work
  should prioritize calibrated-MSD runtime: the fixed-sum prefix80 path is about
  60x slower than WANDA/activation on the same prefix while staying within the
  memory budget.
* The focused Qwen3 experiment-family time estimate is tracked in
  `docs/experiments_time_estimates.md`. Update it after each accepted runtime
  optimization or full direct-CUDA timing run.
