# Next code-improvement plan before the OOM-fix iteration

This plan assumes the previous docs cleanup has landed. It should be executed as a behavior-preserving refactor.

## Phase 0 - Inventory and safety snapshot

1. Read `README.md`, `docs/cleanup/*`, and this `docs/codex/*` directory.
2. Ignore `../transformers/AGENTS.md` and any AGENTS file inside the sibling transformers checkout.
3. Before editing, run or at least inspect:
   - `python ppltest.py --list`
   - `python ppl_batch.py --list`
   - `python calibrate.py --list`
   - `python test_mxfp8linear.py`
   - `python tools/repo_quality_gate.py`
4. Record current setup IDs/tags and current active PPL constants.

Exit Phase 0 only when you can state which files will be edited and why.

## Phase 1 - Add contract gates first

Add the following files before refactoring:

- `tools/repo_quality_gate.py`
- `tools/compare_ppl_math.py`
- `tests/test_config_contract.py`
- `tests/test_ppl_window_contract.py`
- `tests/test_qwen3_public_api_contract.py`
- `scripts/run_repo_quality_gate.sh`

Make sure the tests are cheap and do not download models or datasets.

Purpose: Codex should see failures immediately if it changes methodology while doing cleanup.

## Phase 2 - Configuration registry cleanup

Current pain point: custom Qwen3 config fields are spread across `experiment_config.py`, `configuration_qwen3.py`, runtime code in `modeling_qwen3.py`, and ad-hoc assignments in PPL scripts.

Target state:

1. Add a small registry of custom config fields. Good names:
   - `CUSTOM_QWEN3_CONFIG_DEFAULTS`
   - `MXFP_MSD_FIELDS`
   - `MXFP_FORMAT_FLAGS`
   - `MSD_RUNTIME_STATS_FIELDS`
2. Include all fields that onlinearith sets or snapshots, including runtime stats toggles:
   - `msd_perf_stats_enabled`
   - `msd_perf_stats_lite`
   - `msd_figure5_layer_cycles`
3. Make the `msd_chunk_target_mib` default explicit. If the onlinearith experiment default intentionally differs from `Qwen3Config`, document it as an experiment override rather than silent drift.
4. Add `validate_setup_definition(setup)` or equivalent checks:
   - setup ID is unique;
   - tag is unique;
   - override keys are known;
   - MX format flags are mutually exclusive;
   - `mxfp6_format` is either `e2m3` or `e3m2`;
   - MSD budgets and delays are positive integers where required.
5. Keep `SETUPS` IDs/tags unchanged.

Suggested implementation shape:

```python
# experiment_config.py
MXFP_FORMAT_FLAGS = ("use_mxfp8", "use_mxfp6", "use_mxfp4")
MSD_RUNTIME_STATS_FIELDS = ("msd_perf_stats_enabled", "msd_perf_stats_lite", "msd_figure5_layer_cycles")
CUSTOM_QWEN3_CONFIG_DEFAULTS = {
    ...
}
MXFP_MSD_FIELDS = list(CUSTOM_QWEN3_CONFIG_DEFAULTS)
```

Avoid importing heavy model code just to validate setup definitions.

## Phase 3 - PPL helper extraction

Current pain point: PPL windowing, label masking, and weighted loss accumulation exist in more than one runner.

Target state: add `ppl_utils.py` with small pure helpers:

- `precompute_windows(seq_len, max_length=4096, stride=512)`
- `mask_context_labels(input_ids, trg_len)`
- `accumulate_weighted_nll(total_nll, total_tokens, loss, trg_len)`
- `finalize_ppl(total_nll, total_tokens)`
- optional `iter_shard(items, rank, world_size)`

Rules:

- Keep existing CLI defaults and output schemas.
- Keep `ppltest.py` and `ppl_batch.py` as top-level scripts.
- Do not average window losses directly.
- Add a small synthetic equivalence test before and after the extraction.
- Do not import `datasets`, `transformers`, or a real model in the helper tests.

## Phase 4 - Qwen3 custom code modularization

Current pain point: `modeling_qwen3.py` is the operational file and contains both upstream model code and custom MXFP/MSD code. That makes Codex risky and makes diff review hard.

Target state: extract custom code into helper modules, but keep compatibility re-exports from `modeling_qwen3.py`.

Possible module split inside `../transformers/src/transformers/models/qwen3/`:

```text
mxfp_formats.py       # float8/float6/float4 quantize/dequantize helpers, block scale logic
mxfp_linear.py        # MXFP8Linear, MXFP6Linear, MXFP4Linear, _MXFPLinearBase, _make_linear
msd_arithmetic.py     # truncation, fixed-sum optimizer, significance helpers
msd_context.py        # MSDComputeContext and chunk-planning helpers, no algorithm change
msd_perf_stats.py     # keep existing accumulator here
```

Compatibility requirement:

```python
from transformers.models.qwen3.modeling_qwen3 import MXFP8Linear, MXFP6Linear, MXFP4Linear, _make_linear, MSDComputeContext
```

must keep working.

Rules:

- Do not regenerate from `modular_qwen3.py`.
- Do not change public model classes or generation behavior.
- Move code in small steps. After each step, run public import tests.
- Prefer re-export shims over sweeping import rewrites.
- Do not change numerical algorithms in this pass.

## Phase 5 - Test cleanup

Current pain point: validation scripts are useful but not always CI-friendly.

Target state:

1. Keep script entry points such as `python test_mxfp8linear.py`.
2. Also make tests discoverable with `pytest`.
3. Split very large test files only where it makes review easier:
   - `tests/test_mxfp_formats.py`
   - `tests/test_mxfp_linear.py`
   - `tests/test_msd_optimizer.py`
   - `tests/test_config_contract.py`
4. Use deterministic seeds.
5. Skip GPU-only tests when CUDA is absent; do not fail CPU-only structure checks.

## Phase 6 - CLI and path normalization

Current pain point: scripts still carry local-layout assumptions and repeated model/result path logic.

Target state:

- Add `runtime_paths.py` or `path_utils.py` for:
  - default model path;
  - default results directory;
  - sibling transformers source path;
  - clear error messages when paths are missing.
- Add optional `--model-path` and `--results-dir` where missing.
- Preserve existing defaults.
- Never silently change the model, dataset, stride, or max length.

## Acceptance criteria

A cleanup PR is acceptable when:

- all existing documented commands still run or fail with the same expected external dependency errors;
- `python tools/repo_quality_gate.py --strict` passes;
- pytest contract tests pass;
- setup ID/tag list is unchanged;
- PPL constants and weighted-token accounting are unchanged;
- public Qwen3 custom imports are unchanged;
- no OOM mitigation has been introduced accidentally;
- documentation says what remains for the next OOM iteration.
