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
"mxfp_weight_cache_dtype": "float16",  # choices: "float32", "float16", "none"
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
3. `mxfp_weight_cache_dtype="none"`: no persistent cache; recompute per forward for lowest memory, useful for debugging OOM.

Add `clear_mxfp_weight_cache(model)` in `onlinearith/experiment_config.py` and call it after setup changes, before/after batch setup transitions, and after OOM probe runs. It should walk modules and set `_w_cache = None`, `_w_cache_key = None`, `_w_cache_data_ptr = None`, then run `gc.collect()` and `torch.cuda.empty_cache()` if CUDA is available.

Acceptance: `tests/test_mxfp_weight_cache_compact.py` passes, and `tools/probe_mxfp_memory.py --setup 2 --seq-len 4096` no longer shows a monotonic multi-GB cache increase across layers.

## Phase 3: PPL runner memory controls

Patch `ppltest.py` and `ppl_batch.py`.

Add flags:

```text
--stats {off,lite,full}      default: off for PPL numerical runs
--mx-chunk-target-mib N      forwards to model.config.mxfp_chunk_target_mib
--msd-chunk-target-mib N     forwards to model.config.msd_chunk_target_mib
--weight-cache-dtype {float16,float32,none}
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

1. `python tests/test_mx_exact_chunked.py`
2. `python tests/test_mxfp_weight_cache_compact.py`
3. `python tools/probe_mxfp_memory.py --model-path ../Qwen3-8B --setup 1 --seq-len 256 --stats off`
4. `python tools/probe_mxfp_memory.py --model-path ../Qwen3-8B --setup 2 --seq-len 4096 --mx-chunk-target-mib 256 --weight-cache-dtype float16 --stats off`
5. `python tools/probe_mxfp_memory.py --model-path ../Qwen3-8B --setup 6 --seq-len 4096 --mx-chunk-target-mib 256 --msd-chunk-target-mib 256 --weight-cache-dtype float16 --stats off`
6. `python ppltest.py --model-path ../Qwen3-8B --setup 2 --gpus 0 --stats off --mx-chunk-target-mib 256 --weight-cache-dtype float16 --limit-samples 2`
7. Full setup 2 on one GPU.
8. Setup 6 with `--stats off`, then `--stats lite` only when collecting figure stats.

Do not proceed to full 8B PPL until the probe reports peak allocated and reserved memory with at least 2-4 GiB headroom on a 32 GiB GPU.
