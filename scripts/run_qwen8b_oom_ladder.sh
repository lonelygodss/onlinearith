#!/usr/bin/env bash
set -euo pipefail

# Run from the onlinearith repo root.
# These commands exercise the flags described in docs/cim_oom_harness/CODEX_OOM_PERF_PLAN.md.

export PYTHONPATH="$(pwd)/../transformers/src:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
MODEL_PATH="${MODEL_PATH:-../Qwen3-8B}"
GPU="${GPU:-0}"
MX_CHUNK="${MX_CHUNK:-256}"
MSD_CHUNK="${MSD_CHUNK:-256}"
CACHE_DTYPE="${CACHE_DTYPE:-float8}"
MIN_HEADROOM_GIB="${MIN_HEADROOM_GIB:-2}"
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-30}"
if [[ -z "${PYTHON:-}" && -x "../.venv3_10/bin/python" ]]; then
  PYTHON="../.venv3_10/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

"$PYTHON" tests/test_mx_exact_chunked.py
"$PYTHON" tests/test_mxfp_weight_cache_compact.py
"$PYTHON" tests/test_ppl_tail_logits_loss.py

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the Qwen3-8B OOM ladder; no CUDA device is visible.")
PY

"$PYTHON" tools/probe_mxfp_memory.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 1 --seq-len 256 --stats off \
  --progress-interval-sec "$PROGRESS_INTERVAL_SEC" --progress-file probe_setup1_progress.json \
  --output probe_setup1_seq256.json

"$PYTHON" tools/probe_mxfp_memory.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 2 --seq-len 4096 \
  --mx-chunk-target-mib "$MX_CHUNK" --weight-cache-dtype "$CACHE_DTYPE" --stats off \
  --progress-interval-sec "$PROGRESS_INTERVAL_SEC" --progress-file probe_setup2_progress.json \
  --min-headroom-gib "$MIN_HEADROOM_GIB" \
  --output probe_setup2_seq4096.json

"$PYTHON" tools/probe_mxfp_memory.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 6 --seq-len 4096 \
  --mx-chunk-target-mib "$MX_CHUNK" --msd-chunk-target-mib "$MSD_CHUNK" \
  --weight-cache-dtype "$CACHE_DTYPE" --stats off \
  --progress-interval-sec "$PROGRESS_INTERVAL_SEC" --progress-file probe_setup6_progress.json \
  --min-headroom-gib "$MIN_HEADROOM_GIB" \
  --output probe_setup6_seq4096.json

"$PYTHON" ppltest.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 2 --limit-samples 2 \
  --stats off --mx-chunk-target-mib "$MX_CHUNK" --weight-cache-dtype "$CACHE_DTYPE" \
  --mxfp-progress-interval-sec "$PROGRESS_INTERVAL_SEC" --mxfp-progress-file ppl_setup2_progress.json

"$PYTHON" ppltest.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 6 --limit-samples 2 \
  --stats off --mx-chunk-target-mib "$MX_CHUNK" --msd-chunk-target-mib "$MSD_CHUNK" \
  --weight-cache-dtype "$CACHE_DTYPE" \
  --mxfp-progress-interval-sec "$PROGRESS_INTERVAL_SEC" --mxfp-progress-file ppl_setup6_progress.json
