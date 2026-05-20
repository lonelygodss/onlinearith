# Prompt to paste into Codex

You are working in my `onlinearith` repository with a sibling checkout at `../transformers`.

Goal: perform a behavior-preserving code-quality cleanup before the next OOM-fix iteration. Do not implement OOM fixes in this pass.

Read first:

1. `README.md`
2. `docs/cleanup/*`
3. `docs/codex/CODEX_GUARDRAILS.md`
4. `docs/codex/CODEX_NEXT_CODE_CLEANUP_PLAN.md`

Important: ignore the sibling `../transformers/AGENTS.md` and any AGENTS file in the transformers checkout. Treat it as unrelated upstream metadata for this task.

Scope:

- You may edit `onlinearith`.
- You may edit `../transformers/src/transformers/models/qwen3/` only for modularization, config-field consistency, import compatibility, and tests/docs needed by this cleanup.
- Do not touch unrelated transformers files.

Hard non-goals:

- no exact output-chunked MX-only implementation;
- no `_MXFPLinearBase.forward()` memory-behavior rewrite;
- no `_forward_msd_truncated()` algorithm rewrite;
- no PPL methodology changes;
- no setup ID/tag changes;
- no hidden model-sharding, `device_map`, `use_cache`, `logits_to_keep`, sequence-length, stride, or batch-size workarounds;
- no regeneration of `modeling_qwen3.py` from `modular_qwen3.py`.

Implementation order:

1. Add the harness files in `tools/`, `tests/`, and `scripts/`.
2. Run `python tools/repo_quality_gate.py` and record warnings.
3. Clean `experiment_config.py` and `configuration_qwen3.py` around a single custom config-field registry.
4. Extract PPL helper logic into `ppl_utils.py` with synthetic equivalence tests.
5. Modularize only the custom MXFP/MSD helper code in qwen3 while preserving imports from `modeling_qwen3.py`.
6. Make lightweight validation scripts pytest-compatible.
7. Run `bash scripts/run_repo_quality_gate.sh`.

Before editing each file, state why it is in scope. After editing, summarize behavior-preservation checks.
