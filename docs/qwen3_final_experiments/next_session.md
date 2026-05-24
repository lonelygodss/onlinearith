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
- The same fixed-sum `--nproc 2` run with the default float16 persistent weight
  cache OOMed on rank 1. Use `--weight-cache-dtype float8` for Qwen3-8B MSD
  full-replica multi-GPU PPL unless a newer memory fix supersedes this.
- An eight-worker MXFP8 launch on GPUs 0-7 failed before evaluation with rank-0
  `SIGKILL` during model loading/materialization and produced no output JSON.
  Treat four full replicas as the currently validated final-run ceiling until
  the eight-replica startup/load issue is fixed.
- WikiText-2 test tokenization is 299,078 tokens and 578 PPL forward windows
  at `MAX_LENGTH=4096`, `STRIDE=512`. Runtime estimates should scale from
  forward-window count, not from scored tokens, because almost all windows feed
  a full context while scoring only 512 new tokens.
- The tested sequential map used physical GPUs 4-7 as visible devices but
  placed layers on visible CUDA 0 and 1 only. It lowered per-GPU memory but did
  not improve wall time; the fixed-sum sharded prefix was 2111.13s versus
  1998.8s for the historical single-GPU prefix.

Next iteration:
1. Decide whether to investigate the eight-replica loading SIGKILL or proceed
   with four validated full replicas for final PPL wall-time runs. For MSD,
   include `--weight-cache-dtype float8`.
2. Treat `--device-map sequential` as memory relief only unless `balanced` or
   manual placement shows direct-CUDA speedup over single-GPU and `--nproc`.
3. Remember that current `--nproc` disables MSD stats on nonzero ranks; use it
   for PPL quality and wall time, not as an aggregate work-stats source unless
   stats aggregation is added.
4. Keep `--device-map` single-process and separate from `--nproc`; use
   visible-device IDs in `--max-memory` after `--gpus` filtering.
5. For fixed-sum calibration, use projection-filtered task parallel full-model
   jobs before considering model-sharded calibration.
```

Do not add run logs here. Put measurements in `references/evidence_log.md` and
implementation history in `references/implementation_notes.md`.
