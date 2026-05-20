# Prompt to give Codex

You are working in a local checkout with sibling repos:

```text
coding/
  onlinearith/
  transformers/
  Qwen3-8B/
```

Read `onlinearith/CODEX_OOM_PERF_PLAN.md`. Implement the plan in small commits. Do not change PPL methodology (`MAX_LENGTH`, `STRIDE`, dataset, masked-label semantics, weighted NLL). Prioritize:

1. exact output-chunked MX-only path in `_MXFPLinearBase`;
2. compact `mxfp_weight_cache_dtype=float16` cache with fp32 compute;
3. PPL flags for stats/chunk/cache control;
4. memory probe and acceptance ladder.

Use these tests as contracts:

```bash
python tests/test_mx_exact_chunked.py
python tests/test_mxfp_weight_cache_compact.py
```

Then run:

```bash
bash scripts/run_qwen8b_oom_ladder.sh
```

Acceptance criteria:

* setup 2, seq_len 4096 probe completes on one 32 GB GPU with at least 2 GiB headroom;
* setup 6, seq_len 4096 probe completes with `--stats off`;
* `ppltest.py --setup 2 --limit-samples 2` and `--setup 6 --limit-samples 2` complete on a single GPU;
* small-layer exact MX test passes without changing old MX math.
