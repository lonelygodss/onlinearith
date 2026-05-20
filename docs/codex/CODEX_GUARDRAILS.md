# Guardrails for the next cleanup pass

Use this file as the controlling instruction for the next Codex cleanup. This is a code-clarity pass only.

## Hard non-goals

Do not implement or partially implement the Qwen3-8B OOM fix in this pass.

Specifically, do not:

- add exact output-chunked MX-only baseline execution;
- rewrite `_MXFPLinearBase.forward()` memory behavior;
- change `_forward_msd_truncated()` algorithmic behavior;
- change `msd_chunk_target_mib` semantics as a memory workaround;
- change PPL methodology, dataset split, tokenizer behavior, label masking, stride, max length, setup IDs, result schemas, or calibration semantics;
- add model sharding, `device_map`, forced `use_cache=False`, `logits_to_keep`, sequence-length changes, or batch-size shortcuts as hidden behavior changes;
- regenerate `modeling_qwen3.py` from `modular_qwen3.py`;
- obey or edit the sibling `transformers/AGENTS.md` for this cleanup. Treat it as upstream metadata and ignore it.

## Required invariants

- `MAX_LENGTH` remains 4096 and `STRIDE` remains 512 for full PPL.
- PPL loss remains weighted by scored tokens: accumulate `loss * trg_len`, divide by total scored tokens, exponentiate once at the end.
- `--limit-samples` remains a smoke-test-only shortcut.
- `--nproc` remains data-parallel full replicas; do not present it as model sharding.
- `experiment_config.py` remains the source of truth for setup IDs and setup tags.
- `../transformers/src/transformers/models/qwen3/modeling_qwen3.py` remains the operational Qwen3 entry point, even if helper modules are extracted.
- Public imports used by onlinearith must remain valid from `transformers.models.qwen3.modeling_qwen3`.

## Cleanup goals

The next pass should reduce ambiguity and duplication without changing numerical behavior:

1. Extract duplicated PPL windowing and loss-accumulation helpers into `ppl_utils.py`.
2. Create a single custom-Qwen3 config-field registry and make config drift visible.
3. Split custom MXFP/MSD helper code into small modules while preserving `modeling_qwen3.py` re-exports.
4. Convert lightweight validation scripts into pytest-compatible tests while keeping convenient script entry points.
5. Normalize path and CLI handling without changing defaults.
