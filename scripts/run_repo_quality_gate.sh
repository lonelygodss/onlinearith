#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "../.venv3_10/bin/python" ]]; then
  PYTHON_BIN="../.venv3_10/bin/python"
else
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" tools/repo_quality_gate.py --strict
"${PYTHON_BIN}" tools/compare_ppl_math.py

if ! "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
  echo "[INFO] pytest is not installed for ${PYTHON_BIN}; running contract tests directly."
  "${PYTHON_BIN}" tests/test_config_contract.py
  "${PYTHON_BIN}" tests/test_ppl_window_contract.py
  "${PYTHON_BIN}" tests/test_qwen3_public_api_contract.py
  exit 0
fi

"${PYTHON_BIN}" -m pytest -q \
  tests/test_config_contract.py \
  tests/test_ppl_window_contract.py \
  tests/test_qwen3_public_api_contract.py
