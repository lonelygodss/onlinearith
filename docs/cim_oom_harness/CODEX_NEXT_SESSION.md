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

Validation already run:
- ../.venv3_10/bin/python -m py_compile ppl_utils.py ppltest.py ppl_batch.py tools/probe_mxfp_memory.py
- ../.venv3_10/bin/python tests/test_mx_exact_chunked.py
- ../.venv3_10/bin/python tests/test_mxfp_weight_cache_compact.py
- ../.venv3_10/bin/python tests/test_ppl_tail_logits_loss.py
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

Next recommended work:
1. Inspect probe_setup6_progress.json to confirm the latest bounded-run state.
2. Decide whether to spend the full multi-hour setup 6 probe/PPL acceptance run,
   or continue optimizing MSD scheduling first.
3. If optimizing, focus on MSD performance without changing MSD math or PPL
   methodology. Current setup 6 chunking at seq_len=4096 is very conservative:
   gate/up use chunk_size=4 and down uses chunk_size=1 because the temporary
   `(N, chunk, nb, bs)` tensor is large.
4. Keep `PYTORCH_ALLOC_CONF=expandable_segments:True`, `MAX_LENGTH=4096`,
   `STRIDE=512`, dataset/tokenizer/labels/loss weighting unchanged.
```

