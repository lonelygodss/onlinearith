# Qwen3 Final Experiment Handoff

Use this short prompt for the next Codex session:

```text
Continue final Qwen3 experiment preparation in /home/xzj/coding/onlinearith.

Read first:
- AGENTS.md
- docs/qwen3_final_experiments/codex_prompt.md
- docs/qwen3_final_experiments/active_plan.md
- docs/qwen3_final_experiments/runtime_estimates.md

Read only when needed:
- docs/qwen3_final_experiments/references/evidence_log.md
- docs/qwen3_final_experiments/references/implementation_notes.md
- docs/qwen3_final_experiments/references/multigpu_sharding_plan.md

Current principles:
- Preserve PPL methodology: WikiText-2 raw test, MAX_LENGTH=4096, STRIDE=512,
  masked context labels, and weighted NLL accumulation.
- Treat MXFP8 baseline, fixed-sum calibrated MSD at target-SNR 30 dB, WANDA
  2:4, and activation N:M 2:4 as the representative single-setup experiment
  family until an explicit sweep is added.
- For MSD equivalent work, use Figure 4
  `plot_norm_digit_read = mean_effective_precision / 3.0`; do not substitute
  runtime `global_utilization`.
- `ppltest.py --nproc` is data-parallel window sharding and replicates the full
  model. It is valid for final PPL wall-time acceleration when every selected
  GPU can fit a full replica, but it is not model sharding and should not be
  used as the multi-GPU OOM solution.
- `ppltest.py --device-map {auto,sequential,balanced}` is the explicit
  single-process model-sharding path. It records sharding metadata in the
  result JSON and must be validated with direct CUDA before timing estimates
  are accepted.

Current status:
- Qwen3-8B setup 2 MXFP8 has direct-CUDA prefix80 validation with one full
  4096-token context window. Sequential model sharding matched single-GPU
  scored tokens and PPL exactly at recorded precision.
- Qwen3-8B setup 6 uniform MSD and fixed-sum target-SNR 30 dB also have
  direct-CUDA prefix80 validation with exact scored-token and PPL parity at
  recorded precision.
- Qwen3-8B `ppltest.py --nproc` full-replica data-parallel PPL is now
  prefix-validated for setup 2 MXFP8 and fixed-sum target-SNR 30 dB up to four
  workers. MXFP8 prefix80 matched prior PPL and ran in 17.02s on two workers
  versus 31.97s single-GPU; MXFP8 prefix120 ran in 33.2s on four workers.
  Fixed-sum prefix80 matched prior PPL and ran in 1120.41s on two workers with
  `--weight-cache-dtype float8`; fixed-sum prefix120 ran in 2239.0s on four
  workers with the same float8 cache.
- `ppltest.py --load-stagger-sec S` is now available and records
  `load_stagger_sec` in output metadata. It sleeps `local_rank*S` seconds before
  tokenizer/model loading, reducing transient host RAM and disk I/O spikes
  without changing PPL math.
- Qwen3-8B MXFP8 prefix120 is now validated on eight full replicas with
  `--load-stagger-sec 8`: PPL 9.8221, mean NLL 2.2846, wall 17.03s. This
  replaces the prior failed unstaggered eight-worker launch as the MXFP8 final
  speed recipe.
- Qwen3-8B fixed-sum target-SNR 30 dB prefix120 is now validated on eight full
  replicas with `--load-stagger-sec 8 --weight-cache-dtype float8`: PPL 9.8648,
  mean NLL 2.2890, wall 1120.87s. This promotes the final fixed-sum MSD PPL
  recipe from four replicas to eight replicas.
- Qwen3-8B WANDA 2:4 and activation N:M 2:4 now have explicit baseline-runner
  window sharding. Use `--window-shard` with `--nproc`; default baseline-runner
  `--nproc` shards setup IDs and does not accelerate a single selected setup.
  WANDA prefix120 on eight replicas with `--window-shard --load-stagger-sec 8`
  gave PPL 18.2443, mean NLL 2.9039, wall 17.13s. Activation N:M prefix120 on
  eight replicas with the same execution mode gave PPL 13.0141, mean NLL 2.5660,
  wall 18.26s.
- For Qwen3-8B WANDA final PPL, use a Qwen3-8B-shaped mask. The committed
  `../data/wanda_base/2-4/calibration_base_MXFP8.pt` is shaped for Qwen3-0.6B
  and fails on Qwen3-8B.
- The same fixed-sum `--nproc 2` run with the default float16 persistent weight
  cache OOMed on rank 1. Use `--weight-cache-dtype float8` for Qwen3-8B MSD
  full-replica multi-GPU PPL unless a newer memory fix supersedes this.
- An unstaggered eight-worker MXFP8 launch on GPUs 0-7 failed before evaluation
  with rank-0 `SIGKILL` during model loading/materialization and produced no
  output JSON. Use `--load-stagger-sec 8` for Qwen3-8B eight-replica launches.
- WikiText-2 test tokenization is 299,078 tokens and 578 PPL forward windows
  at `MAX_LENGTH=4096`, `STRIDE=512`. Runtime estimates should scale from
  forward-window count, not from scored tokens, because almost all windows feed
  a full context while scoring only 512 new tokens.
- The tested sequential map used physical GPUs 4-7 as visible devices but
  placed layers on visible CUDA 0 and 1 only. It lowered per-GPU memory but did
  not improve wall time; the fixed-sum sharded prefix was 2111.13s versus
  1998.8s for the historical single-GPU prefix.

Next iteration:
1. Prepare final full-run commands for the representative four-path Qwen3-8B
   family: MXFP8, fixed-sum target-SNR 30 dB MSD, WANDA 2:4, and activation
   N:M 2:4. Use `model_execution_matrix.md` for per-path flags.
2. For WANDA final PPL, either promote/generate a Qwen3-8B-shaped 2:4 mask in
   the intended results root or pass a compatible `--output-hook`; do not use
   the committed 0.6B-shaped mask.
3. Treat `--device-map sequential` as memory relief only unless `balanced` or
   manual placement shows direct-CUDA speedup over single-GPU and `--nproc`.
4. Remember that current `--nproc` disables MSD stats on nonzero ranks; use it
   for PPL quality and wall time, not as an aggregate work-stats source unless
   stats aggregation is added.
5. Keep `--device-map` single-process and separate from `--nproc`; use
   visible-device IDs in `--max-memory` after `--gpus` filtering.
6. For fixed-sum calibration, use projection-filtered task parallel full-model
   jobs before considering model-sharded calibration.
7. Use `model_execution_matrix.md` to keep model-specific tricks explicit:
   smaller models should not inherit Qwen3-8B-only float8 cache or load-stagger
   settings unless their own prefix validation shows they need them.
```

Do not add run logs here. Put measurements in `references/evidence_log.md` and
implementation history in `references/implementation_notes.md`.
