# Codex Quality Gates

Reusable repo checks live in normal repo paths:

```text
onlinearith/
  tools/repo_quality_gate.py
  tools/compare_ppl_math.py
  tests/test_config_contract.py
  tests/test_ppl_window_contract.py
  tests/test_qwen3_public_api_contract.py
  scripts/run_repo_quality_gate.sh
```

Verification:

```bash
bash scripts/run_repo_quality_gate.sh
```

The contract tests intentionally avoid loading a real Qwen3 model. They are
contract and structure checks, not PPL benchmarks. If `pytest` is not installed
in the project environment, `run_repo_quality_gate.sh` runs them directly.
