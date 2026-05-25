# Final Run Commands

These are the settled commands for the representative Qwen3-8B experiment
family. They preserve the existing PPL method: WikiText-2 raw test,
`MAX_LENGTH=4096`, `STRIDE=512`, masked context labels, and weighted NLL
accumulation.

Run the CUDA visibility check before launching final work:

```bash
../.venv3_10/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Expected on the current machine: `True 8`.

## Paths

Use a shell variable block like this to avoid mixing smoke artifacts with final
outputs:

```bash
MODEL=../Qwen3-8B
GPUS=0,1,2,3,4,5,6,7
FINAL_ROOT=../data/qwen3_final_experiments/qwen3_8b
MSD_DIR=$FINAL_ROOT/calib_fixed_sum_30db
MSD_CAL=$MSD_DIR/calibration_MXFP8_fixed_sum_qwen8b_final_merged.json
WANDA_ROOT=../data/wanda_base
WANDA_HOOK=qwen8b_final
ACT_ROOT=$FINAL_ROOT/act_base
```

Do not use `../data/calib-data/30db/calibration_MXFP8_fixed_sum.json` or
`../data/wanda_base/2-4/calibration_base_MXFP8.pt` for Qwen3-8B final runs;
those existing files are shaped for smaller-model work.

## Calibration Prerequisites

### Fixed-Sum MSD

Run projection-filtered full-model calibration jobs, preferably one projection
family per GPU. Use `--weight-cache-dtype none` for broad calibration capture.

```bash
../.venv3_10/bin/python calibrate.py \
  --model-path "$MODEL" \
  --setup 1 \
  --optimizer fixed_sum \
  --target-snr 30 \
  --projection-filter gate_proj \
  --num-texts 20 \
  --max-length 512 \
  --batch-size 4 \
  --result-suffix qwen8b_final_gate \
  --output-dir "$MSD_DIR" \
  --mx-chunk-target-mib 256 \
  --cal-chunk-target-mib 64 \
  --weight-cache-dtype none \
  --compile-msd-truncate \
  --gpus 0

../.venv3_10/bin/python calibrate.py \
  --model-path "$MODEL" \
  --setup 1 \
  --optimizer fixed_sum \
  --target-snr 30 \
  --projection-filter up_proj \
  --num-texts 20 \
  --max-length 512 \
  --batch-size 4 \
  --result-suffix qwen8b_final_up \
  --output-dir "$MSD_DIR" \
  --mx-chunk-target-mib 256 \
  --cal-chunk-target-mib 64 \
  --weight-cache-dtype none \
  --compile-msd-truncate \
  --gpus 1

../.venv3_10/bin/python calibrate.py \
  --model-path "$MODEL" \
  --setup 1 \
  --optimizer fixed_sum \
  --target-snr 30 \
  --projection-filter down_proj \
  --num-texts 20 \
  --max-length 512 \
  --batch-size 4 \
  --result-suffix qwen8b_final_down \
  --output-dir "$MSD_DIR" \
  --mx-chunk-target-mib 256 \
  --cal-chunk-target-mib 64 \
  --weight-cache-dtype none \
  --compile-msd-truncate \
  --gpus 2
```

Merge the three disjoint projection outputs:

```bash
../.venv3_10/bin/python tools/merge_msd_calibrations.py \
  "$MSD_DIR/calibration_MXFP8_fixed_sum_qwen8b_final_gate.json" \
  "$MSD_DIR/calibration_MXFP8_fixed_sum_qwen8b_final_up.json" \
  "$MSD_DIR/calibration_MXFP8_fixed_sum_qwen8b_final_down.json" \
  --output "$MSD_CAL"
```

### WANDA 2:4 Mask

Generate a Qwen3-8B-shaped WANDA mask with a suffix, so it cannot collide with
the existing smaller-model mask:

```bash
../.venv3_10/bin/python wanda_base/calibrate_base.py \
  --model-path "$MODEL" \
  --results-root "$WANDA_ROOT" \
  -n 2 -m 4 \
  --setup 1 \
  --num-texts 2048 \
  --max-length 512 \
  --batch-size 4 \
  --output-hook "$WANDA_HOOK" \
  --mx-chunk-target-mib 256 \
  --weight-cache-dtype none \
  --gpus 0
```

This should create:

```text
../data/wanda_base/2-4/calibration_base_MXFP8_qwen8b_final.pt
```

## Final PPL Runs

### MXFP8 Baseline

```bash
../.venv3_10/bin/python ppltest.py \
  --model-path "$MODEL" \
  --setup 2 \
  --nproc 8 \
  --gpus "$GPUS" \
  --stats off \
  --load-stagger-sec 8 \
  --output "$FINAL_ROOT/ppl_results_MXFP8_qwen8b_final.json" \
  --mxfp-progress-interval-sec -1
```

### Fixed-Sum MSD 30 dB

```bash
../.venv3_10/bin/python ppltest.py \
  --model-path "$MODEL" \
  --setup 6 \
  --calibration "$MSD_CAL" \
  --nproc 8 \
  --gpus "$GPUS" \
  --stats off \
  --compile-msd-truncate \
  --weight-cache-dtype float8 \
  --load-stagger-sec 8 \
  --output "$FINAL_ROOT/ppl_results_MXFP8_fixed_sum30_qwen8b_final.json" \
  --mxfp-progress-interval-sec -1
```

### WANDA 2:4

Use `--window-shard`; default baseline-runner `--nproc` shards setup IDs and
does not accelerate a single selected setup.

```bash
../.venv3_10/bin/python wanda_base/ppl_batch_base.py \
  --model-path "$MODEL" \
  --results-root "$WANDA_ROOT" \
  -n 2 -m 4 \
  --only 1 \
  --output-hook "$WANDA_HOOK" \
  --nproc 8 \
  --gpus "$GPUS" \
  --window-shard \
  --load-stagger-sec 8 \
  --mxfp-progress-interval-sec -1
```

### Activation N:M 2:4

```bash
../.venv3_10/bin/python act_base/ppl_batch_base_act.py \
  --model-path "$MODEL" \
  --results-root "$ACT_ROOT" \
  -n 2 -m 4 \
  --only 1 \
  --nproc 8 \
  --gpus "$GPUS" \
  --window-shard \
  --load-stagger-sec 8 \
  --mxfp-progress-interval-sec -1
```

## Expected Wall Times

- MXFP8 PPL: about 0.34 h on eight workers.
- Fixed-sum MSD 30 dB PPL: about 22.7 h on eight workers.
- WANDA 2:4 PPL: about 0.35 h on eight workers, after mask calibration.
- Activation N:M 2:4 PPL: about 0.37 h on eight workers.
