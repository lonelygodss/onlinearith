#!/usr/bin/env bash
set -euo pipefail

# Copy to onlinearith/scripts/run_qwen8b_oom_ladder.sh and run from onlinearith.
# These commands assume Codex has implemented the flags described in CODEX_OOM_PERF_PLAN.md.

export PYTHONPATH="$(pwd)/../transformers/src:${PYTHONPATH:-}"
MODEL_PATH="${MODEL_PATH:-../Qwen3-8B}"
GPU="${GPU:-0}"
MX_CHUNK="${MX_CHUNK:-256}"
MSD_CHUNK="${MSD_CHUNK:-256}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"

python tests/test_mx_exact_chunked.py
python tests/test_mxfp_weight_cache_compact.py

python tools/probe_mxfp_memory.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 1 --seq-len 256 --stats off \
  --output probe_setup1_seq256.json

python tools/probe_mxfp_memory.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 2 --seq-len 4096 \
  --mx-chunk-target-mib "$MX_CHUNK" --weight-cache-dtype "$CACHE_DTYPE" --stats off \
  --output probe_setup2_seq4096.json

python tools/probe_mxfp_memory.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 6 --seq-len 4096 \
  --mx-chunk-target-mib "$MX_CHUNK" --msd-chunk-target-mib "$MSD_CHUNK" \
  --weight-cache-dtype "$CACHE_DTYPE" --stats off \
  --output probe_setup6_seq4096.json

python ppltest.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 2 --limit-samples 2 \
  --stats off --mx-chunk-target-mib "$MX_CHUNK" --weight-cache-dtype "$CACHE_DTYPE"

python ppltest.py \
  --model-path "$MODEL_PATH" --gpus "$GPU" --setup 6 --limit-samples 2 \
  --stats off --mx-chunk-target-mib "$MX_CHUNK" --msd-chunk-target-mib "$MSD_CHUNK" \
  --weight-cache-dtype "$CACHE_DTYPE"
