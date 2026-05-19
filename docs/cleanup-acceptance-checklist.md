# Cleanup acceptance checklist

Use this checklist before merging the repository cleanup. It is designed to catch accidental methodology changes before the later Qwen3-8B OOM fixes.

## Agent migration

- [ ] `AGENTS.md` exists at the repo root.
- [ ] `CLAUDE.md` is removed from the repo root or archived under `docs/archive/` as obsolete.
- [ ] Codex can summarize the loaded instructions:

```bash
codex --ask-for-approval never "Summarize the current repository instructions."
```

## Repository hygiene

- [ ] `.gitignore` covers Python caches, logs, generated PPL/calibration outputs, plots, model artifacts, and local data/model directories.
- [ ] `__pycache__/` is not tracked.
- [ ] `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, and `.ipynb_checkpoints/` are not tracked.
- [ ] `ppl_batch_summary.json` is not tracked as source.
- [ ] Generated `ppl_results_*.json`, `calibration_*.json`, `*.pt`, and benchmark result JSON files are ignored unless deliberately committed as fixtures.

## Documentation coherence

- [ ] README says `onlinearith` is the experiment driver repo.
- [ ] README says the modified Qwen3 code lives in the sibling `transformers` repo.
- [ ] README identifies `modeling_qwen3.py` as the current operational implementation.
- [ ] README identifies `experiment_config.py` as the setup/config source of truth.
- [ ] README warns that current `--nproc` means data-parallel full model replicas, not model sharding.
- [ ] README marks `--limit-samples` as smoke-test only.
- [ ] Deep-pipeline notes are archived or clearly marked abandoned.

## Source and test path sanity

- [ ] No active source file contains personal absolute paths such as `/home/xzjnew/`.
- [ ] Tests use a relative sibling `../transformers/src` path or documented `PYTHONPATH`.
- [ ] Tests import from `transformers.models.qwen3.modeling_qwen3` by default, not `modular_qwen3.py`, unless the test specifically targets the modular converter.

## PPL methodology invariants

- [ ] `MAX_LENGTH` remains `4096` in active PPL scripts.
- [ ] `STRIDE` remains `512` in active PPL scripts.
- [ ] Evaluation dataset remains `wikitext/wikitext-2-raw-v1/test`.
- [ ] Calibration dataset remains `wikitext/wikitext-2-raw-v1/validation`.
- [ ] Context labels are still masked with `-100`.
- [ ] PPL accumulation still uses `loss * trg_len` divided by total scored tokens.
- [ ] Setup IDs and tags in `experiment_config.py` are unchanged unless explicitly approved.
- [ ] Result JSON schema and default filenames are unchanged unless backwards-compatible aliases are added.

## Quick commands

Run these if the local environment has dependencies available:

```bash
python ppltest.py --list
python ppl_batch.py --list
python calibrate.py --list
python test_mxfp8linear.py
python test_fixed_sum_optimizer.py
python test_distributed.py
```

Optional smoke PPL checks:

```bash
python ppltest.py --setup 1 --limit-samples 2
python ppltest.py --setup 6 --lite --limit-samples 2
```

A cleanup patch may still be acceptable if local model/dataset dependencies are missing, but Codex must report the exact failing command and the missing dependency/path.

## Explicit non-goals confirmed

- [ ] The cleanup patch does not implement exact MX output chunking yet.
- [ ] The cleanup patch does not change lite/full stats behavior yet, except documentation.
- [ ] The cleanup patch does not add `device_map="auto"` or model sharding.
- [ ] The cleanup patch does not force `use_cache=False` or `logits_to_keep` yet.
- [ ] The cleanup patch does not reduce sequence length, stride, dataset size, tokenizer behavior, or quantization format to avoid OOM.

## Ready for next task

- [ ] Codex can understand where to patch Fix 1 later: `_MXFPLinearBase` in `../transformers/src/transformers/models/qwen3/modeling_qwen3.py`.
- [ ] Codex can understand where to patch Fix 5 later: stats/lite behavior in `modeling_qwen3.py`, `msd_perf_stats.py`, and PPL script flags.
- [ ] The next task can start from a fresh Codex session using the handoff prompt in `docs/cleanup/codex-cleanup-playbook.md`.
