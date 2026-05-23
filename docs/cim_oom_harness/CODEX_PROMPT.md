# Prompt to give Codex

You are working in a local checkout with sibling repos:

```text
coding/
  onlinearith/
  transformers/
  Qwen3-8B/
```

Read `onlinearith/docs/cim_oom_harness/CODEX_OOM_PERF_PLAN.md`. Implement the plan in small commits. Do not change PPL methodology (`MAX_LENGTH`, `STRIDE`, dataset, masked-label semantics, weighted NLL). Prioritize:

1. exact output-chunked MX-only path in `_MXFPLinearBase`;
2. compact MXFP weight cache, using `mxfp_weight_cache_dtype=float8` for
   MXFP8 Qwen3-8B runs and `float16`/`none` for non-MXFP8 paths;
3. PPL flags for stats/chunk/cache control;
4. memory probe and acceptance ladder.
5. calibration runner parity with the same GPU, chunk, cache, compile, and
   projection-filter controls.
6. parity for paper-critical baselines: fixed-sum calibrated MSD, uniform MSD,
   MX-only, WANDA structured sparsity, and runtime activation n:m.

Use these tests as contracts:

```bash
python tests/test_mx_exact_chunked.py
python tests/test_mxfp_weight_cache_compact.py
python test_fixed_sum_optimizer.py
```

Then run:

```bash
bash scripts/run_qwen8b_oom_ladder.sh
```

Before any Qwen3-8B probe, PPL smoke test, calibration run, or timing comparison, verify that the command environment has direct CUDA visibility:

```bash
../.venv3_10/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Expected output on this machine is `True 8`. If it prints `False 0`, do not use that environment for GPU performance/OOM conclusions. Valid GPU progress logs include `cuda_*` memory fields and should be visible in `nvtop`.

Acceptance criteria:

* setup 2, seq_len 4096 probe completes on one 32 GB GPU with at least 2 GiB headroom;
* setup 6, seq_len 4096 probe completes with `--stats off`;
* setup 6, seq_len 4096 with `--weight-cache-dtype float8` completes with
  unchanged loss versus the float16 cache and substantially more headroom;
* `ppltest.py --setup 2 --limit-samples 2` and `--setup 6 --limit-samples 2` complete on a single GPU;
* a targeted 8B calibration smoke with `--projection-filter` captures only the requested projection and completes on one GPU;
* broader fixed-sum calibration subsets use `--weight-cache-dtype none` for now, because projection-filtered hooks do not prevent unrelated persistent MXFP forward caches from accumulating;
* gate/up/down fixed-sum subset metadata can be merged into a staged full-MLP calibration JSON because `msd_calibration_data` is keyed by projection module name;
* fixed-sum calibrated MSD PPL uses generated `msd_calibration_data` from `calibrate.py --optimizer fixed_sum --target-snr ...` and completes a Qwen3-8B smoke;
* WANDA and activation n:m baseline runners use the same GPU visibility, allocator, chunk/cache, `use_cache=False`, and PPL loss/window semantics as `ppltest.py`;
* prefix80 Qwen3-8B measurements exist for staged full-MLP fixed-sum calibrated MSD, WANDA, and activation n:m; OOM feasibility is established, but fixed-sum calibrated MSD runtime is the main remaining optimization target;
* small-layer exact MX test passes without changing old MX math.
