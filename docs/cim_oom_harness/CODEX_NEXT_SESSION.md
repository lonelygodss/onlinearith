# Qwen3-8B OOM/performance handoff

Use this prompt for the next Codex session:

```text
Continue the Qwen3-8B OOM/performance iteration in /home/xzj/coding/onlinearith.

Follow:
- docs/cim_oom_harness/CODEX_PROMPT.md
- docs/cim_oom_harness/CODEX_OOM_PERF_PLAN.md
- AGENTS.md invariants from the repo root

Current status:
- PPL methodology is preserved: WikiText-2 raw test, MAX_LENGTH=4096, STRIDE=512,
  masked context labels, weighted NLL accumulation.
- Do not implement model sharding/device_map until single-GPU setup 2 and setup 6
  are proven.
- Exact output-chunked MX-only path, compact MXFP weight cache, PPL memory flags,
  and tail-logits chunked loss are already implemented.
- Shared MXFP/MSD progress reporting is now wired into:
  - tools/probe_mxfp_memory.py
  - ppltest.py
  - ppl_batch.py
  - scripts/run_qwen8b_oom_ladder.sh
- Long runs print `PROGRESS {json}` and update latest-event progress files such as
  probe_setup6_progress.json and ppl_setup6_progress.json.
- Sibling Transformers `modeling_qwen3.py` has an optimized `_msd_truncate`
  implementation that replaces log2/pow scale tensors with `torch.frexp` /
  `torch.ldexp` reconstruction and removes the final zeroing mask by clamping
  requested digit counts to nonnegative values before truncation.

Validation already run:
- ../.venv3_10/bin/python -m py_compile ppl_utils.py ppltest.py ppl_batch.py tools/probe_mxfp_memory.py
- ../.venv3_10/bin/python tests/test_mx_exact_chunked.py
- ../.venv3_10/bin/python tests/test_mxfp_weight_cache_compact.py
- ../.venv3_10/bin/python tests/test_ppl_tail_logits_loss.py
- ../.venv3_10/bin/python tests/test_msd_truncate_equivalence.py
- ../.venv3_10/bin/python test_mxfp8linear.py
- ../.venv3_10/bin/python ppltest.py --list
- ../.venv3_10/bin/python ppl_batch.py --list
- GPU 0 setup 2 probe completed:
  status=ok, peak_alloc=27.6147 GiB, peak_reserved=28.2090 GiB,
  reserved_headroom=3.1477 GiB, meets_min_headroom=true,
  mxfp_weight_cache.total_gib=10.7578.
- GPU 0 setup 2 ppltest --limit-samples=2 completed.
- GPU 0 setup 6 probe was intentionally bounded with timeout after 240s.
  It did not OOM; it progressed through layer 0 MLP and into layers.1.mlp.up_proj.
  Last observed peak_reserved was about 19.2441 GiB.
- Setup 6 MSD chunk-size experiments:
  - `MSD_CHUNK=2048` OOMed immediately in the first gate chunk:
    peak_reserved=30.533 GiB, reserved_headroom=0.823 GiB.
  - `MSD_CHUNK=1024` did not OOM in a bounded 180s run but did not materially
    improve first-layer runtime versus the default 256 MiB target.
  - Larger chunk scheduling is therefore not the main setup 6 bottleneck; the
    per-element MSD truncation work dominates.
- Setup 6 optimized `_msd_truncate` bounded run at the default 256 MiB target:
  - first layer gate/up/down completed in about 51.2s / 102.2s / 153.9s;
  - peak_reserved was about 18.951 GiB;
  - run was intentionally timed out after entering `layers.1.mlp.gate_proj`.
- Valid GPU rerun on 2026-05-21 confirmed direct CUDA access (`True`, 8 devices)
  and reproduced the kept-path setup 6 baseline:
  - `probe_setup6_gpu_rerun_progress.json`;
  - layer 0 gate/up/down completed in about 51.2s / 102.1s / 153.9s;
  - bounded timeout reached `layers.1.mlp.up_proj` at chunk 1830 / 3072;
  - peak_reserved reached about 19.3418 GiB.
- A valid GPU trial replacing NAF-width `floor(log2(combined)) + 1` with
  `torch.frexp` exponent extraction is now kept in sibling Transformers:
  - `probe_setup6_gpu_frexp_width_progress.json`;
  - layer 0 gate/up/down completed in about 45.8s / 91.3s / 137.6s;
  - bounded timeout reached `layers.1.mlp.up_proj` at 100%, then entered
    `layers.1.mlp.down_proj`;
  - peak_reserved stayed about 19.3418 GiB.
- A valid GPU trial combining frexp-width with in-place stats-off `p_eff`
  construction was slower and was reverted:
  - `probe_setup6_gpu_frexp_inplace_progress.json`;
  - layer 0 gate/up/down completed in about 46.9s / 93.6s / 141.0s;
  - peak_reserved was not improved enough to justify the slowdown.
- Important correction from 2026-05-21: the bounded probes run in that Codex
  sandbox did not have CUDA visibility (`torch.cuda.is_available() == False`,
  `device_count == 0`, and progress JSON had no `cuda_*` fields). Do not use
  `probe_setup6_current_progress.json`, `probe_setup6_frexp_width_progress.json`,
  or `probe_setup6_inplace_peff_progress.json` as GPU performance evidence.
  GPU commands must be run outside the sandbox / with direct CUDA access.
- `torch.compile(_msd_truncate)` was tried but is blocked in the current venv by
  `ModuleNotFoundError: No module named 'setuptools'` from Inductor.

Next recommended work:
1. First verify CUDA visibility in the command environment:
   `../.venv3_10/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'`.
   Expect `True` and 8 devices. If false, do not run probe/PPL performance jobs.
2. Inspect valid GPU progress files (`probe_setup6_gpu_frexp_width_progress.json`,
   `probe_setup6_gpu_rerun_progress.json`), not the 2026-05-21 sandbox progress
   files without `cuda_*` fields.
3. Decide whether to spend the full multi-hour setup 6 probe/PPL acceptance run,
   or continue optimizing MSD scheduling first.
4. If optimizing, focus on MSD performance without changing MSD math or PPL
   methodology. Current setup 6 chunking at seq_len=4096 is very conservative:
   gate/up use chunk_size=4 and down uses chunk_size=1 because the temporary
   `(N, chunk, nb, bs)` tensor is large.
5. Keep `PYTORCH_ALLOC_CONF=expandable_segments:True`, `MAX_LENGTH=4096`,
   `STRIDE=512`, dataset/tokenizer/labels/loss weighting unchanged.
```
