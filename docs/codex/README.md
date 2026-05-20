# Codex Repo-Improvement Harness

These files steer behavior-preserving cleanup passes for `onlinearith` and the
sibling `../transformers/src/transformers/models/qwen3/` implementation before
any CUDA OOM work.

Integrated placement:

```text
onlinearith/
  docs/codex/CODEX_NEXT_CODE_CLEANUP_PLAN.md
  docs/codex/CODEX_GUARDRAILS.md
  docs/codex/CODEX_PROMPT_CODE_CLEANUP.md
  tools/repo_quality_gate.py
  tools/compare_ppl_math.py
  tests/test_config_contract.py
  tests/test_ppl_window_contract.py
  tests/test_qwen3_public_api_contract.py
  scripts/run_repo_quality_gate.sh
```

Run after Codex changes:

```bash
python tools/repo_quality_gate.py --strict
python -m pytest -q tests/test_config_contract.py tests/test_ppl_window_contract.py tests/test_qwen3_public_api_contract.py
bash scripts/run_repo_quality_gate.sh
```

The tests intentionally avoid loading a real Qwen3 model. They are contract and structure checks, not PPL benchmarks.
