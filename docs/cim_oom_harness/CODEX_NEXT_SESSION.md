# Qwen3-8B OOM/performance handoff

Use this prompt for the next Codex session:

```text
Continue the Qwen3-8B OOM/performance iteration in /home/xzj/coding/onlinearith.

Read and follow:
- AGENTS.md
- docs/cim_oom_harness/CODEX_PROMPT.md
- docs/cim_oom_harness/CODEX_OOM_PERF_PLAN.md

Hard invariants:
- Preserve PPL methodology: WikiText-2 raw test, MAX_LENGTH=4096, STRIDE=512,
  masked context labels, weighted NLL accumulation.
- Do not implement model sharding/device_map until single-GPU setup 2 and setup 6
  are proven.
- Do not change dataset, tokenizer, labels, loss weighting, MAX_LENGTH, or STRIDE.
- Treat fixed-sum calibrated MSD, uniform MSD, MX-only, WANDA structured
  sparsity, and activation n:m as paper-critical paths. Do not call the overall
  OOM/perf iteration complete until these paths have comparable runner hygiene
  and smoke evidence.

GPU visibility rule:
- Before any Qwen3-8B probe/PPL/calibration/timing run, execute:
  ../.venv3_10/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
- Expected output is True and 8 devices. If it reports False/0, do not run GPU
  performance work in that environment.
- Valid GPU progress has cuda_alloc_gib/cuda_peak_* fields and should be visible
  in nvtop. Ignore sandbox progress files without cuda_* fields.
- Current GPU snapshot after the latest prefix80 runs: GPUs 0-3 are idle
  (~15 MiB, 0% util); GPUs 4-7 are busy (~15.2 GiB each).

Current implementation:
- Exact output-chunked MX-only path is implemented.
- Compact MXFP weight cache is implemented.
- PPL/probe flags are implemented: --stats, --mx-chunk-target-mib,
  --msd-chunk-target-mib, --weight-cache-dtype. `float8` is supported for
  MXFP8 weight caches and is the recommended Qwen3-8B MXFP8 cache mode; use
  `float16` or `none` for non-MXFP8 paths.
- Tail-logits chunked loss is implemented and covered by tests.
- Shared MXFP/MSD progress reporting is wired into probe, ppltest, ppl_batch,
  and the ladder script.
- Sibling Transformers modeling_qwen3.py has the optimized MSD truncation path:
  frexp/ldexp scale reconstruction, no final zero-mask allocation, and NAF-width
  frexp exponent extraction.
- `--gpus` is applied before torch import in probe, ppltest, and ppl_batch so
  requested physical GPUs are honored before CUDA state is cached.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set by default before torch
  import in probe, ppltest, and ppl_batch.
- `--compile-msd-truncate` is implemented for probe, ppltest, and ppl_batch.
  It sets `config.msd_compile_truncate=True` and uses
  `torch.compile(_msd_truncate, fullgraph=True, mode="reduce-overhead")` only
  when explicitly requested on CUDA. The default remains false.
- Parent venv now has setuptools installed, so torch.compile is no longer
  blocked by ModuleNotFoundError.
- `calibrate.py` is now wired into the same Qwen3-8B runtime controls:
  `--gpus` is applied before torch import, `PYTORCH_ALLOC_CONF` defaults to
  `expandable_segments:True`, model `use_cache` is forced false, and the CLI
  accepts `--mx-chunk-target-mib`, `--cal-chunk-target-mib`,
  `--weight-cache-dtype`, and `--compile-msd-truncate`.
- Calibration capture now applies `--projection-filter` before hook
  registration and quantized-weight capture, so targeted 8B calibration probes
  do not materialize every MLP projection's quantized weights.
- Sibling Transformers `calibration_msd.py` has calibration-only runtime
  controls, compile-aware `_msd_truncate` scheduling, and frexp-based
  intra-block delay extraction that preserves the old `floor(log2(abs(x)))`
  delay semantics.

Path parity status:
- MX-only baselines and uniform MSD setup 6 are memory-feasible on one 32 GB GPU
  for the validated Qwen3-8B probes/smokes below.
- Fixed-sum calibrated MSD is optimized for runner mechanics: calibration
  generation has the new controls, `ppltest.py --calibration` injects metadata
  into the optimized MSD runtime, and Qwen3-8B PPL smokes using generated
  fixed-sum metadata have passed for one-layer, all-gate, all-up, all-down, and
  merged full-MLP staged metadata. A longer non-final prefix80 run using the
  merged full-MLP staged metadata also passed. Missing: final calibrated metrics
  and, if needed, a true single-run all-MLP calibration capture.
- WANDA structured sparsity baseline runner parity is implemented for
  `wanda_base/calibrate_base.py` and `wanda_base/ppl_batch_base.py`: early
  `--gpus`, allocator default before torch import, `use_cache=False`, shared
  PPL window/loss utilities, tail-logits loss, MX chunk/cache controls,
  progress hook support, cache cleanup, and `--limit-samples` smoke support.
- Runtime activation n:m baseline runner parity is implemented for
  `act_base/ppl_batch_base_act.py` with the same PPL/runtime controls as WANDA.
  WANDA and activation prefix80 runs have passed; final measurements remain.
- Baseline sparsity notation has been corrected to common N:M keep-count
  semantics. `-n 2 -m 4` means keep 2 of 4; internally the old prune-count
  behavior is `(m - n):m`. This matters for asymmetric sweeps such as `1:4`.
- A living single-setup estimate table for the focused model/config family is
  in `docs/experiments_time_estimates.md`. It includes Qwen3-0.6B, 1.7B, 4B,
  and 8B, one representative 50% point per path, and current single-GPU runtime
  estimates. Sweeps and multi-GPU estimates are intentionally deferred.
- For MSD "50% sparsity", use measured
  `msd_perf_stats.global.global_utilization ~= 0.5`, not a fixed target-SNR.
  Existing 0.6B fixed-sum `_fix_cap`/`_fix_time` artifacts only reach about
  0.225 utilization at 30 dB, so a small target-finding pilot is still needed.
  Those old files were produced with `ppltest.py --setup 6 --calibration ...
  --lite --output ..._fix_time.json --limit-samples 100
  --figure5-layer-cycles --gpus <id>` and were later removed by commit
  `e0a38ec908233cbbf21c56262e5b79963a212660`; do not restore them. For new
  target-finding pilots use the cleaned standard mode:
  `ppltest.py --setup 6 --calibration ... --msd-utilization-mode --output
  ..._fix_time.json --gpus <id>`. Add `--figure5-layer-cycles` only when
  explicitly debugging Figure 5 cycle accounting.

Valid GPU measurements:
- Setup 2 probe, seq_len=4096: status ok; peak_alloc=27.6147 GiB;
  peak_reserved=28.2090 GiB; reserved_headroom=3.1477 GiB;
  meets_min_headroom=true; mxfp_weight_cache.total_gib=10.7578.
- Setup 2 ppltest --limit-samples=2 completed on one GPU.
- Setup 6 baseline before the NAF-width frexp change:
  layer 0 gate/up/down completed in about 51.2s / 102.1s / 153.9s;
  bounded timeout reached layers.1.mlp.up_proj chunk 1830 / 3072;
  peak_reserved about 19.3418 GiB.
- Setup 6 with kept NAF-width frexp change:
  layer 0 gate/up/down completed in about 45.8s / 91.3s / 137.6s;
  bounded timeout reached layers.1.mlp.up_proj at 100%, then entered
  layers.1.mlp.down_proj; peak_reserved about 19.3418 GiB.
- Setup 6 chunk-size trials:
  MSD_CHUNK=2048 OOMed immediately, peak_reserved=30.533 GiB,
  reserved_headroom=0.823 GiB.
  MSD_CHUNK=1024 avoided OOM in a bounded run but did not materially improve
  runtime versus 256 MiB.
- frexp-width plus in-place stats-off p_eff construction was slower on GPU
  (layer 0 about 46.9s / 93.6s / 141.0s) and was reverted.
- Setup 6 with `--compile-msd-truncate`, seq_len=4096: status ok;
  elapsed=1059.95s; loss=0.01120709; peak_alloc=27.8387 GiB;
  peak_reserved=28.7617 GiB; reserved_headroom=2.5950 GiB;
  meets_min_headroom=true; mxfp_weight_cache.total_gib=10.7578.
- Setup 6 with native MXFP8 `--weight-cache-dtype float8`,
  `--compile-msd-truncate`, seq_len=4096: status ok; elapsed=1060.33s;
  loss=0.01120709; peak_alloc=22.7762 GiB; peak_reserved=23.8984 GiB;
  reserved_headroom=7.4583 GiB; meets_min_headroom=true;
  mxfp_weight_cache.total_gib=5.6953, with 5.0625 GiB stored as
  `torch.float8_e4m3fn` and 0.6328 GiB fp32 scales. This has the same loss and
  essentially the same runtime as the float16-cache probe, while saving about
  5.06 GiB allocated and 4.86 GiB reserved.
- Setup 6 `ppltest --limit-samples=2 --compile-msd-truncate` completed on one
  GPU. It is a smoke only: the first two WikiText test samples score 8 tokens.
  Token PPL=423.0598, mean NLL=6.0475, peak memory=27.08 GB.
- Setup 6 `ppltest --limit-samples=80 --compile-msd-truncate` completed on one
  GPU. This prefix has 4,144 tokens and evaluates two windows, including a full
  4,096-token context window. Token PPL=8.8708, mean NLL=2.1828, scored
  tokens=4,144, peak memory=28.03 GB, wall time=2000.6s.
- Targeted Qwen3-8B calibration smoke completed on GPU 5:
  `calibrate.py --model-path ../Qwen3-8B --gpus 5 --setup 1 --optimizer fixed_sum
  --projection-filter model.layers.0.mlp.gate_proj --num-texts 1 --max-length 64
  --batch-size 1 --target-snr 10 --curve-window 1 --mx-chunk-target-mib 256
  --cal-chunk-target-mib 64 --weight-cache-dtype float16 --compile-msd-truncate`.
  It captured/solved exactly one layer, total_channels=12,288, budget range
  [4, 9], budget_mean=6.09, mean_snr=12.84 dB, wall_time=6.68s. Output was
  written under `/tmp/onlinearith_calib_smoke/` and should not be committed.
- Calibrated-MSD Qwen3-8B PPL smoke completed on GPU 1 using the targeted
  fixed-sum metadata above:
  `ppltest.py --setup 6 --calibration /tmp/onlinearith_calib_smoke/calibration_MXFP8_fixed_sum_qwen8b_layer0_smoke.json
  --limit-samples 2 --stats off --mx-chunk-target-mib 256 --msd-chunk-target-mib 256
  --weight-cache-dtype float16 --compile-msd-truncate`. It scored 8 tokens,
  token PPL=446.5720, mean NLL=6.1016, peak memory=27.08 GB, wall time=5.53s.
  This is a smoke only because the calibration file covers one projection.
- Broader fixed-sum calibration with `--projection-filter gate_proj` initially
  OOMed on one 32 GB GPU when `--weight-cache-dtype float16` was enabled. The
  projection filter correctly limits calibration hooks and metadata capture, but
  the normal MXFP forward path can still persist quantized weights for every MLP
  projection during the calibration pass. For broader calibration capture, use
  `--weight-cache-dtype none` unless/until calibration disables unrelated
  persistent forward caches automatically.
- All-gate fixed-sum calibration completed on GPU 0 with cache disabled:
  `calibrate.py --model-path ../Qwen3-8B --gpus 0 --setup 1 --optimizer fixed_sum
  --projection-filter gate_proj --num-texts 1 --max-length 64 --batch-size 1
  --target-snr 30 --curve-window 1 --mx-chunk-target-mib 256
  --cal-chunk-target-mib 64 --weight-cache-dtype none --compile-msd-truncate`.
  Output:
  `/tmp/onlinearith_calib_gate_smoke/calibration_MXFP8_fixed_sum_qwen8b_gate_snr30_smoke_nocache.json`.
  It captured/solved 36 gate projections, total_channels=442,368, budget range
  [6, 16], budget_mean=10.2, wall_time=105.4s. This is still a staged smoke
  because up/down projections fall back to uniform/default budgets.
- Calibrated-MSD Qwen3-8B PPL smoke completed on GPU 0 using the all-gate
  fixed-sum metadata:
  `ppltest.py --setup 6 --calibration /tmp/onlinearith_calib_gate_smoke/calibration_MXFP8_fixed_sum_qwen8b_gate_snr30_smoke_nocache.json
  --limit-samples 2 --stats off --mx-chunk-target-mib 256 --msd-chunk-target-mib 256
  --weight-cache-dtype float16 --compile-msd-truncate`. It scored 8 tokens,
  token PPL=356.8912, mean NLL=5.8774, peak memory=27.08 GB, wall time=5.3s.
- All-up fixed-sum calibration completed on GPU 0 with cache disabled:
  `/tmp/onlinearith_calib_up_smoke/calibration_MXFP8_fixed_sum_qwen8b_up_snr30_smoke_nocache.json`.
  It captured/solved 36 up projections, total_channels=442,368, budget range
  [5, 17], budget_mean=11.1, wall_time=105.2s.
- All-down fixed-sum calibration completed on GPU 1 with cache disabled:
  `/tmp/onlinearith_calib_down_smoke/calibration_MXFP8_fixed_sum_qwen8b_down_snr30_smoke_nocache.json`.
  It captured/solved 36 down projections, total_channels=147,456, budget range
  [7, 21], budget_mean=12.6, wall_time=98.7s.
- Calibrated-MSD Qwen3-8B PPL smokes completed using the all-up and all-down
  fixed-sum metadata:
  all-up scored 8 tokens, token PPL=406.7382, mean NLL=6.0082, peak memory
  27.08 GB, wall time=5.4s; all-down scored 8 tokens, token PPL=452.6000,
  mean NLL=6.1150, peak memory=27.08 GB, wall time=5.2s.
- A staged merged full-MLP fixed-sum metadata file was built from the all-gate,
  all-up, and all-down projection-subset files:
  `/tmp/onlinearith_calib_mlp_merged_smoke/calibration_MXFP8_fixed_sum_qwen8b_mlp_merged_snr30_smoke_nocache.json`.
  It covers 108 MLP projection entries, total_channels=1,032,192, budget range
  [5, 21], budget_mean=10.9369. This is merged from independent projection
  captures, not a single all-projection capture.
- Calibrated-MSD Qwen3-8B PPL smoke completed on GPU 0 using the merged
  full-MLP staged metadata:
  `ppltest.py --setup 6 --calibration /tmp/onlinearith_calib_mlp_merged_smoke/calibration_MXFP8_fixed_sum_qwen8b_mlp_merged_snr30_smoke_nocache.json
  --limit-samples 2 --stats off --mx-chunk-target-mib 256 --msd-chunk-target-mib 256
  --weight-cache-dtype float16 --compile-msd-truncate`. It scored 8 tokens,
  token PPL=362.5553, mean NLL=5.8932, peak memory=27.08 GB, wall time=5.9s.
- Calibrated-MSD Qwen3-8B prefix80 run completed on GPU 0 using the merged
  full-MLP staged metadata:
  `ppltest.py --setup 6 --calibration /tmp/onlinearith_calib_mlp_merged_smoke/calibration_MXFP8_fixed_sum_qwen8b_mlp_merged_snr30_smoke_nocache.json
  --limit-samples 80 --stats off --mx-chunk-target-mib 256 --msd-chunk-target-mib 256
  --weight-cache-dtype float16 --compile-msd-truncate`. This is non-final, but
  it evaluates two windows including a full 4,096-token context window. It
  scored 4,144 tokens, token PPL=8.8632, mean NLL=2.1819, peak memory=28.03 GB,
  peak_reserved during progress=28.7617 GiB, throughput=2.1 tokens/s, wall
  time=1998.8s. Output:
  `/tmp/onlinearith_calibrated_msd_mlp_merged_snr30_ppl_prefix80.json`.
- Runtime activation n:m Qwen3-8B smoke completed on GPU 0:
  `act_base/ppl_batch_base_act.py --model-path ../Qwen3-8B --gpus 0 --only 1
  -n 2 -m 4 --limit-samples 2 --mx-chunk-target-mib 256 --weight-cache-dtype float16`.
  It scored 8 tokens, token PPL=831.3688, mean NLL=6.7231, peak memory=26.54 GB,
  wall time=1.23s, with valid `cuda_*` MX progress fields.
- Runtime activation n:m Qwen3-8B prefix80 run completed on GPU 2:
  `act_base/ppl_batch_base_act.py --model-path ../Qwen3-8B --gpus 2 --only 1
  -n 2 -m 4 --limit-samples 80 --mx-chunk-target-mib 256
  --weight-cache-dtype float16 --results-root /tmp/onlinearith_act_prefix80 --force`.
  It scored the same prefix as the calibrated-MSD prefix80 run, token
  PPL=11.3428, mean NLL=2.4286, peak memory=27.61 GB, wall time=33.03s, summary
  `/tmp/onlinearith_act_prefix80/2-4/ppl_batch_base_act_summary.json`.
- WANDA Qwen3-8B smoke completed on GPU 2. First,
  `wanda_base/calibrate_base.py --model-path ../Qwen3-8B --gpus 2 --setup 1
  -n 2 -m 4 --num-texts 1 --max-length 64 --batch-size 1 --mx-chunk-target-mib 256
  --weight-cache-dtype float16 --output-hook qwen8b_smoke` generated
  `/tmp/onlinearith_wanda_smoke/2-4/calibration_base_MXFP8_qwen8b_smoke.pt`.
  Then `wanda_base/ppl_batch_base.py --model-path ../Qwen3-8B --gpus 2 --only 1
  -n 2 -m 4 --limit-samples 2 --mx-chunk-target-mib 256 --weight-cache-dtype float16
  --output-hook qwen8b_smoke` scored 8 tokens, token PPL=576.8480,
  mean NLL=6.3576, peak memory=26.54 GB, wall time=0.95s, with valid `cuda_*`
  MX progress fields.
- WANDA Qwen3-8B prefix80 run completed on GPU 1. The WANDA PPL runner expects
  the mask under its `--results-root`/`n-m` directory; a symlink was used from
  `/tmp/onlinearith_wanda_prefix80/2-4/calibration_base_MXFP8_qwen8b_smoke_prefix80.pt`
  to the existing smoke mask
  `/tmp/onlinearith_wanda_smoke/2-4/calibration_base_MXFP8_qwen8b_smoke.pt`.
  Then `wanda_base/ppl_batch_base.py --model-path ../Qwen3-8B --gpus 1 --only 1
  -n 2 -m 4 --limit-samples 80 --mx-chunk-target-mib 256
  --weight-cache-dtype float16 --results-root /tmp/onlinearith_wanda_prefix80
  --output-hook qwen8b_smoke_prefix80 --force` completed. Token PPL=15.6914,
  mean NLL=2.7531, peak memory=27.61 GB, wall time=31.85s, summary
  `/tmp/onlinearith_wanda_prefix80/2-4/ppl_batch_base_summary_qwen8b_smoke_prefix80.json`.

Invalid/non-source-of-truth artifacts:
- Ignore 2026-05-21 sandbox progress files without cuda_* fields. They were CPU
  or CUDA-invisible runs and are not GPU evidence.
- Generated probe JSON/log files are not committed; rely on this handoff for
  summarized measurements.

Recommended next steps:
1. Verify CUDA visibility outside the sandbox.
2. Run the cheap contracts:
   ../.venv3_10/bin/python tests/test_msd_truncate_equivalence.py
   ../.venv3_10/bin/python tests/test_mx_exact_chunked.py
   ../.venv3_10/bin/python tests/test_mxfp_weight_cache_compact.py
   ../.venv3_10/bin/python tests/test_ppl_tail_logits_loss.py
   ../.venv3_10/bin/python test_mxfp8linear.py
   ../.venv3_10/bin/python test_fixed_sum_optimizer.py
   ../.venv3_10/bin/python calibrate.py --list
   ../.venv3_10/bin/python tests/test_nm_keep_semantics.py
3. Treat the current stage as done for single-GPU OOM feasibility across the
   paper-critical paths: MX-only, uniform MSD, fixed-sum calibrated MSD,
   WANDA, and activation n:m have direct-CUDA Qwen3-8B evidence. The fixed-sum
   calibrated path is functionally aligned but remains the major runtime
   outlier.
4. Use `--weight-cache-dtype float8` for Qwen3-8B MXFP8 setup 2/setup 6 probes
   and PPL runs when the goal is memory headroom. Use `float16` for MXFP4/6 or
   mixed batch sweeps that include non-MXFP8 setups, and use `none` for broad
   calibration capture if unrelated forward caches would otherwise accumulate.
5. For calibrated fixed-sum MSD, either run final metrics with generated
   metadata or attempt a true single-run all-MLP calibration capture with
   `--weight-cache-dtype none` if exact capture equivalence is required. Treat
   `target-snr` fixed-sum metadata as the main calibrated method.
6. For WANDA and activation baselines, proceed from prefix80 to final
   measurements with the same runner flags. Keep any `--limit-samples` runs
   clearly marked non-final.
7. Full setup 6 seq_len=4096 probe completes in about 17.7 minutes with
   compile enabled; a two-window setup 6 PPL smoke takes about 33.3 minutes.
8. If optimizing further, keep MSD math unchanged and benchmark only with direct
   CUDA.
   Current conservative setup 6 chunking at seq_len=4096 uses gate/up chunk 4
   and down chunk 1 because the temporary (N, chunk, nb, bs) tensor is large.
9. Consider a calibration-specific runtime improvement that disables unrelated
   persistent MXFP forward weight caches during capture while preserving selected
   calibration metadata. This would make broad projection-filtered fixed-sum
   calibration less dependent on manually choosing `--weight-cache-dtype none`.
10. Main further-improvement direction: speed up calibrated MSD inference. The
   prefix80 calibrated-MSD run took about 1999s versus about 32-33s for WANDA
   and activation n:m on the same prefix. Focus on output-chunk scheduling,
   reducing per-chunk temporary work, and compile/fusion opportunities while
   preserving the existing MSD truncation math and PPL invariants.
```
