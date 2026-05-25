# Qwen3 Final Experiment Docs

This directory is the lightweight entry point for the Qwen3 final experiment
iteration. Keep always-read files short. Put long evidence, command history, and
historical implementation notes under `references/`.

## Always-Read

- `next_session.md`: short handoff prompt and current next steps.
- `codex_prompt.md`: durable constraints and development rules for this effort.
- `active_plan.md`: current active plan, not a full history.
- `runtime_estimates.md`: single-setup runtime estimates for the focused Qwen3
  model family, plus pending multi-GPU estimates.
- `model_execution_matrix.md`: per-model execution recipes and validation gates
  before final runs.
- `final_run_commands.md`: concrete Qwen3-8B calibration and final PPL command
  sheet for the representative four-path family.

## Read When Needed

- `references/evidence_log.md`: direct-CUDA measurements and old artifact notes.
- `references/implementation_notes.md`: implementation history and design details
  that do not need to be reread every session.
- `references/multigpu_sharding_plan.md`: explicit model-sharding design notes
  for the next iteration.

## Live Files

The live scripts and tests are at repo root. Do not keep duplicate executable
copies under `docs/`.

- `scripts/run_qwen8b_oom_ladder.sh`
- `tools/probe_mxfp_memory.py`
- `tools/merge_msd_calibrations.py`
- `tests/test_mx_exact_chunked.py`
- `tests/test_mxfp_weight_cache_compact.py`
- `tests/test_msd_truncate_equivalence.py`
- `tests/test_ppl_device_map_utils.py`
- `tests/test_ppl_tail_logits_loss.py`
- `tests/test_nm_keep_semantics.py`
- `tests/test_merge_msd_calibrations.py`
