# AGENTS.md

## Project purpose

This repository contains the experiment drivers, calibration scripts, visualization helpers, and lightweight tests for a CIM / online-arithmetic LLM simulation project. The model implementation itself lives in a sibling fork of Hugging Face Transformers, normally at:

```text
../transformers/src/transformers/models/qwen3/
```

The current paper vocabulary should stay consistent with the drafting notes:

- paper-level principle: temporal significance scheduling
- algorithmic object: local execution windows on aligned contribution streams
- hardware realization: metadata-first two-plane micro-tile with channel-parallel, block-serial execution

Do not rewrite the method as generic sparsity, generic quantization, or generic pruning. The main quality/work metric is executed-digit ratio; read ratios and latency/accounting metrics are secondary hardware-accounting metrics.

## Repository map

Active onlinearith files:

- `ppltest.py`: single-setup WikiText-2 PPL evaluation. It shards sliding windows across ranks when `--nproc` is used.
- `ppl_batch.py`: batch PPL runner over setup IDs. It shards setups across ranks.
- `calibrate.py`: MXFP/MSD calibration driver, including `snr_min` and `fixed_sum` modes.
- `calibrate_base.py`: structured n:m baseline mask calibration.
- `experiment_config.py`: single source of truth for setup IDs, baseline config fields, config snapshots, and MLP reconfiguration.
- `dist_utils.py`: torchrun/NCCL and lite distributed helpers.
- `test_mxfp8linear.py`, `test_fixed_sum_optimizer.py`, `test_distributed.py`: validation scripts. Modernize these before relying on them for major changes.
- `perf_viz.py`, `calibration_viz.py`, `visualization.py`: plotting and diagnostic helpers.

Active modified Transformers files:

- `modeling_qwen3.py`: current operational Qwen3 implementation. Treat this as authoritative unless the user explicitly asks for modular-converter work.
- `configuration_qwen3.py`: custom MXFP/MSD config fields.
- `calibration_msd.py`: calibration implementation imported by `calibrate.py`.
- `msd_perf_stats.py`: performance-statistics accumulator.
- `modular_qwen3.py`: do not edit or regenerate from this during cleanup unless explicitly requested. If the converter is used later, compare the generated `modeling_qwen3.py` carefully and reapply documented manual fixes.

## Environment

Use the parent-directory virtualenv for local commands:

```bash
source ../.venv3_10/bin/activate
```

If the shell is not activated, run commands explicitly through:

```bash
../.venv3_10/bin/python <script>.py
```

The expected local Transformers source is `../transformers/src`. Prefer `PYTHONPATH="$(pwd)/../transformers/src:${PYTHONPATH}"` or a script-local relative bootstrap over any absolute machine-specific path.

## Cleanup scope and non-goals

The immediate task is repository cleanup and migration from Claude Code guidance to Codex guidance. Do not implement the Qwen3-8B OOM fixes during cleanup unless the user explicitly asks.

Allowed cleanup work:

- Replace `CLAUDE.md` with this `AGENTS.md` and remove stale Claude-specific guidance after verification.
- Expand `.gitignore` for caches, generated results, calibration outputs, plots, temporary logs, and model/output artifacts.
- Remove tracked generated artifacts such as `__pycache__/` and transient result JSON files from git tracking.
- Move old notes into `docs/` or `docs/archive/` without changing code behavior.
- Make README and docs reflect the current active scripts, setup IDs, and authoritative source files.
- Remove hardcoded local absolute paths from tests and helper scripts.
- Rename or relocate tests only with thin compatibility wrappers or clear command updates.

Non-goals during cleanup:

- Do not reduce `MAX_LENGTH`, increase `STRIDE`, truncate the dataset, change the tokenizer, or change loss weighting to avoid OOM.
- Do not change setup IDs, result JSON schema, calibration JSON schema, or default file names unless you add backward-compatible aliases.
- Do not change MX quantization math, MSD truncation math, calibration semantics, or PPL window semantics.
- Do not delete deep-pipeline code paths unless the user explicitly approves. It is acceptable to label them archived/abandoned and exclude them from default workflows.
- Do not silently switch from single-GPU data parallel behavior to `device_map="auto"` during cleanup. That belongs to the later OOM-fix task.

## PPL invariants

PPL correctness is more important than speed. Preserve these unless the user explicitly requests a methodology change:

- Dataset: `wikitext`, `wikitext-2-raw-v1`, `test` for evaluation.
- Calibration split: `wikitext`, `wikitext-2-raw-v1`, `validation` for calibration.
- Evaluation window constants: `MAX_LENGTH = 4096`, `STRIDE = 512`.
- Labels for context tokens must be set to `-100`.
- Accumulate `loss * trg_len` and divide by total scored tokens. Do not average window losses directly.
- `--limit-samples` is only for quick tests and must be clearly marked as a non-final run.
- `--lite` may reduce stats overhead, but it must not change logits, labels, loss, or PPL.

## Known 8B OOM context for future work

The current cleanup should prepare for, but not implement, these later changes:

1. Add an exact output-chunked MX-only baseline path in `_MXFPLinearBase.forward()` so the non-MSD MX baseline does not materialize the full `(num_blocks, tokens, out_features)` tensor.
2. Keep `--lite` or stats-disable controls separate from numerical PPL. Lite stats should only skip expensive performance-statistics details.
3. Force `use_cache=False` for PPL in the later OOM fix.
4. Use `logits_to_keep=trg_len+1` plus sliced labels in the later OOM fix only after verifying loss equality on small windows.
5. For Qwen3-8B, avoid 8 data-parallel full model replicas. Model sharding is a later functional change, not cleanup.

## Coding conventions

- Prefer small, reviewable commits/patches.
- Keep script entry points stable. If moving code into helper modules, leave root-level wrappers unless the user agrees to command changes.
- Import shared config from `experiment_config.py`; do not duplicate setup tables in scripts.
- Avoid machine-specific paths such as `/home/xzjnew/coding/...`. Use relative paths, environment variables, or CLI arguments.
- Avoid broad exception swallowing. Print explicit errors for missing model paths, calibration files, or unsupported setup combinations.
- Keep `local_files_only=True` for model/tokenizer loading unless the user asks for download behavior.
- Do not commit generated outputs, cache directories, model weights, calibration dumps, plots, or benchmark JSON unless the user explicitly asks.

## Verification commands

Run the cheapest relevant checks after cleanup edits:

```bash
../.venv3_10/bin/python ppltest.py --list
../.venv3_10/bin/python ppl_batch.py --list
../.venv3_10/bin/python calibrate.py --list
../.venv3_10/bin/python test_mxfp8linear.py
../.venv3_10/bin/python test_fixed_sum_optimizer.py
```

For a smoke PPL check, use a clearly non-final run:

```bash
python ppltest.py --setup 1 --limit-samples 2
python ppltest.py --setup 6 --lite --limit-samples 2
```

For distributed infrastructure only:

```bash
python test_distributed.py
torchrun --nproc_per_node=2 test_distributed.py
```

Do not run full PPL or full calibration by default during cleanup. They are long jobs and not needed to validate file organization or docs.

## Done definition for cleanup

Cleanup is done when:

- `AGENTS.md` is present at repo root and `CLAUDE.md` is removed or explicitly marked obsolete.
- README states the current authoritative code path and active workflows.
- `.gitignore` covers caches, generated results, calibration outputs, plots, and local model/data artifacts.
- Tracked generated files are removed from git tracking.
- Active scripts still run with their old commands.
- Tests no longer rely on hardcoded personal absolute paths.
- No PPL methodology constants or setup IDs changed unintentionally.
- The repository is ready for the next task: exact chunked MX baseline and lite/full stats separation.
