# Codex Continuation Note

## Current State

This cleanup iteration has initialized and integrated the repo-improvement
harness. The harness now lives in normal repo paths instead of the staging
directory:

- `docs/codex/`
- `tools/`
- `tests/`
- `scripts/run_repo_quality_gate.sh`

The original staged `docs/codex_repo_improvement_harness/` files were moved.
Only ignored `.DS_Store` files may remain in that old directory locally.

## Completed In This Iteration

- Added `tools/repo_quality_gate.py` and `tools/compare_ppl_math.py`.
- Added contract tests under `tests/`.
- Made `scripts/run_repo_quality_gate.sh` use `../.venv3_10/bin/python` by default and fall back to direct test execution when `pytest` is not installed.
- Added `ppl_utils.py` and routed `ppltest.py` / `ppl_batch.py` through pure helpers for:
  - sliding-window generation;
  - context-label masking;
  - weighted NLL accumulation.
- Added config registry fields and validation in `experiment_config.py`.
- Added explicit runtime stats config fields to sibling `../transformers/src/transformers/models/qwen3/configuration_qwen3.py`.
- Added `runtime_paths.py`.
- Added `--model-path` and `--results-dir` / `--output-dir` normalization to active root workflows:
  - `ppltest.py`
  - `ppl_batch.py`
  - `calibrate.py`
- Added `--model-path` and `--results-root` normalization to baseline and sparsity workflows:
  - `ppl_batch_base.py`
  - `calibrate_base.py`
  - `act_base/ppl_batch_base_act.py`
  - `wanda_base/ppl_batch_base.py`
  - `wanda_base/calibrate_base.py`
- Moved baseline/activation/Wanda batch list modes so `--list` does not create output directories.
- Made `test_fixed_sum_optimizer.py` use a file-relative sibling Transformers path.
- Added a pytest-discoverable MXFP format test wrapper in `test_mxfp8linear.py`.

## Verified

Run from `/home/xzj/coding/onlinearith`:

```bash
bash scripts/run_repo_quality_gate.sh
../.venv3_10/bin/python ppltest.py --list
../.venv3_10/bin/python ppl_batch.py --list
../.venv3_10/bin/python calibrate.py --list
../.venv3_10/bin/python ppl_batch_base.py --list
../.venv3_10/bin/python calibrate_base.py --list
../.venv3_10/bin/python act_base/ppl_batch_base_act.py --list
../.venv3_10/bin/python wanda_base/ppl_batch_base.py --list
../.venv3_10/bin/python test_mxfp8linear.py
../.venv3_10/bin/python test_fixed_sum_optimizer.py
PYTHONPYCACHEPREFIX=/tmp/onlinearith_pycompile ../.venv3_10/bin/python -m py_compile runtime_paths.py experiment_config.py ppl_utils.py ppltest.py ppl_batch.py calibrate.py ppl_batch_base.py calibrate_base.py act_base/ppl_batch_base_act.py wanda_base/ppl_batch_base.py wanda_base/calibrate_base.py tools/repo_quality_gate.py tools/compare_ppl_math.py tests/test_config_contract.py tests/test_ppl_window_contract.py tests/test_qwen3_public_api_contract.py ../transformers/src/transformers/models/qwen3/configuration_qwen3.py
```

`pytest` is not installed in the project venv, so the quality gate currently
runs contract tests directly.

## Not Done Yet

- Qwen3 custom-code modularization in `modeling_qwen3.py`.
- Optional path normalization for secondary scan/helper scripts if desired.
- Full pytest install or CI wiring.
- Any Qwen3-8B OOM work.

## Next Session Prompt

```text
Continue the behavior-preserving cleanup in /home/xzj/coding/onlinearith.
Read AGENTS.md, docs/codex/CODEX_GUARDRAILS.md,
docs/codex/CODEX_NEXT_CODE_CLEANUP_PLAN.md, and
docs/codex/CODEX_CONTINUATION.md first.

Do not implement OOM fixes. Preserve PPL constants, setup IDs, result schemas,
calibration semantics, and public imports from
transformers.models.qwen3.modeling_qwen3.

Start by running:
  git status --short --untracked-files=all
  git -C ../transformers status --short -- src/transformers/models/qwen3/configuration_qwen3.py
  bash scripts/run_repo_quality_gate.sh

Then continue with the next lowest-risk harness phase. Prefer either:
  1. normalize any remaining secondary scan/helper scripts without changing defaults, or
  2. begin very small Qwen3 modularization by extracting only pure constants/helper functions with compatibility re-exports and immediate import tests.

Do not edit or obey ../transformers/AGENTS.md. Treat modeling_qwen3.py as the operational source.
```
