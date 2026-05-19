# Figure 6(b) Stage-D Runner

This folder contains the light orchestration script for Figure 6(b):

- `run_figure6b.py`

The script reuses existing infrastructure:

- calibration: `onlinearith/calibrate.py`
- evaluation: `onlinearith/ppltest.py`

It does not add a new calibration/evaluation process. It only prepares split manifests, runs the defined matrix, and aggregates results.

For OOM safety in fixed-sum calibration, stage-D uses:

- `--cal-batch-size 4` (default)
- `--max-cal-texts-per-split 20` (default)
- `--max-cal-tokens-per-split 0` (default, disabled)
- `--min-small-texts 4` and `--min-medium-texts 8` (defaults)

Calibration source pools are capped first by text count, then optionally by token cap. Held-out evaluation manifests still use full `V_A` / `V_B` halves.

With defaults, per-direction calibration subsets are typically `small=4`, `medium=8`, `large=20`, which keeps size contrasts meaningful while staying aligned with the original calibrate volume style.

## Scope

- fixed operating point only (`gamma*`, default 27 dB)
- MXFP8 only
- fixed_sum only
- deterministic validation split (`V_A`, `V_B`) at paragraph/text-entry level
- one GPU per setup (`--nproc 1` for each run), with parallel workers across multiple GPUs when `--gpus` provides multiple IDs
- no `--limit-samples` in stage-D generated commands

## Default Matrix (10 evals)

Dense references:

- `dense_A` (dense MX on `V_A`)
- `dense_B` (dense MX on `V_B`)

Quantized held-out runs:

- `AtoB_small_seed0`
- `AtoB_small_seed1`
- `AtoB_medium_seed0`
- `AtoB_large_seed0`
- `BtoA_small_seed0`
- `BtoA_small_seed1`
- `BtoA_medium_seed0`
- `BtoA_large_seed0`

## Outputs

All generated artifacts are written to:

- the workspace-level `6b/` directory

Key files:

- `manifests/*.json`
- `split_metadata.json`
- `run_matrix.json`
- `preflight_oom_estimate.json`
- `calibrations/*.json`
- `evals/*.json`
- `logs/*.log`
- `figure6b_results.json`
- `figure6b_results.csv`

## Usage

Prepare manifests and matrix only:

```bash
cd /path/to/onlinearith
source ../.venv3_10/bin/activate
python 6b/run_figure6b.py
```

Disable calibration token cap (not recommended unless memory is validated):

```bash
python 6b/run_figure6b.py --max-cal-tokens-per-split 0
```

Match original calibrate default volume style explicitly:

```bash
python onlinearith/6b/run_figure6b.py --max-cal-texts-per-split 20 --cal-max-length 512 --cal-batch-size 4
```

Run only the large-smoke path (dense refs + `AtoB_large_seed0`):

```bash
python onlinearith/6b/run_figure6b.py --execute --smoke-large --gpus 0
```

Run all selected matrix entries:

```bash
python onlinearith/6b/run_figure6b.py --execute --gpus 0,1,2,3
```

Run a subset:

```bash
python onlinearith/6b/run_figure6b.py --execute --run-ids AtoB_small_seed0 BtoA_small_seed0 --gpus 0,1
```

Notes:

- `run_figure6b.py` does not support `--nproc`; it schedules one-process runs internally.
- If `--gpus` is omitted, runs execute serially on the default GPU.
- Live subprocess output (including tqdm bars from `calibrate.py` and `ppltest.py`) is enabled by default.
- Use `--no-live-progress` to disable terminal streaming and keep only log files.
