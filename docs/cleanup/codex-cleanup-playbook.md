# Codex cleanup playbook for onlinearith

Use this file as the execution guide after placing `AGENTS.md` in the repository root. The cleanup should be done in small phases. Do not implement the Qwen3-8B OOM fixes in this pass.

## Initial Codex prompt

Paste this into Codex from the `onlinearith` repo root:

```text
Read AGENTS.md and docs/cleanup/repo-cleanup-analysis.md first. Clean the repository only; do not implement the Qwen3-8B OOM fixes yet. Preserve PPL methodology, setup IDs, result schemas, and active root commands. Start by inspecting git status, setup listings, README/CLAUDE conflicts, and tracked generated files. Then propose a small phased cleanup patch. Do not move Python entry-point scripts unless you add compatibility wrappers and I approve.
```

## Phase 0: inventory and safety

Ask Codex to run:

```bash
git status --short
git ls-files | sort | sed -n '1,200p'
python ppltest.py --list
python ppl_batch.py --list
python calibrate.py --list
```

Expected decisions:

- If the working tree has unrelated uncommitted changes, Codex should stop and summarize them.
- If active list commands fail because local dependencies are unavailable, Codex should still continue docs/hygiene cleanup but record the failure.
- Codex should not run full PPL or full calibration during cleanup.

## Phase 1: add Codex guidance and ignore generated noise

Expected patch:

1. Add root `AGENTS.md`.
2. Expand `.gitignore`.
3. Remove generated artifacts from tracking only, not from the lab archive:

```bash
git rm -r --cached __pycache__ || true
git rm --cached ppl_batch_summary.json || true
rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
```

4. If `ppl_batch_summary.json` contains a useful historical result, copy it to an ignored path before untracking:

```bash
mkdir -p ../data/archive/onlinearith
cp -n ppl_batch_summary.json ../data/archive/onlinearith/ppl_batch_summary_legacy.json || true
```

Acceptance:

```bash
git status --short
python ppltest.py --list
python calibrate.py --list
```

No Python behavior should change in this phase.

## Phase 2: resolve documentation contradictions

Expected patch:

1. Replace or archive `CLAUDE.md`.
   - Preferred: move to `docs/archive/CLAUDE.old.md` or delete after confirming `AGENTS.md` covers the needed information.
   - Do not leave both as active root guidance.
2. Rewrite README to state:
   - `onlinearith` holds drivers and evaluation scripts.
   - `../transformers/src/transformers/models/qwen3/modeling_qwen3.py` is the current operational Qwen3 implementation.
   - `experiment_config.py` is the setup source of truth.
   - `--nproc` currently means data-parallel full model replicas.
   - `--limit-samples` is a non-final smoke-test shortcut.
   - Deep pipeline is archived/abandoned unless explicitly requested.
3. Move docs:

```bash
mkdir -p docs/baselines docs/calibration docs/dev docs/archive/old-plan-files
mv baseline_sparsify.md docs/baselines/ 2>/dev/null || true
mv fixed_sum_calibration.md docs/calibration/ 2>/dev/null || true
mv modular_converter_guide.md docs/dev/ 2>/dev/null || true
mv deep-pipeline.txt docs/archive/deep-pipeline-abandoned.md 2>/dev/null || true
mv plan_files/* docs/archive/old-plan-files/ 2>/dev/null || true
rmdir plan_files 2>/dev/null || true
```

4. Keep relative links working from README.

Acceptance:

```bash
python ppltest.py --list
python ppl_batch.py --list
grep -R "modular_qwen3.py.*Main implementation" -n README.md docs || true
grep -R "CLAUDE.md" -n . --exclude-dir=.git || true
```

The first grep should not show README claiming `modular_qwen3.py` is the active main implementation. The second grep may show archive references only.

## Phase 3: remove hardcoded personal paths from tests

Expected patch:

1. In `test_mxfp8linear.py`, remove personal absolute path insertion such as:

```python
sys.path.insert(0, "/home/xzjnew/coding/transformers/src")
```

2. Prefer a relative bootstrap near the top:

```python
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
TRANSFORMERS_SRC = (REPO_ROOT / ".." / "transformers" / "src").resolve()
if TRANSFORMERS_SRC.exists():
    sys.path.insert(0, str(TRANSFORMERS_SRC))
```

If the file later moves under `tests/`, use `Path(__file__).resolve().parents[1]` instead.

3. Import from the operational implementation unless a test specifically targets converter parity:

```python
from transformers.models.qwen3.modeling_qwen3 import (...)
```

not from `modular_qwen3.py` by default.

Acceptance:

```bash
python test_mxfp8linear.py
python test_fixed_sum_optimizer.py
```

If tests fail because the local sibling transformers repo is absent, Codex should report that and avoid inventing a different import path.

## Phase 4: optional README replacement skeleton

Use this skeleton if rewriting README from scratch:

```markdown
# onlinearith

Experiment drivers for MXFP/MSD Qwen3 simulation and temporal significance scheduling studies.

## Active source layout

- `onlinearith/`: PPL, calibration, distributed helpers, result visualization.
- `../transformers/src/transformers/models/qwen3/modeling_qwen3.py`: current operational Qwen3 implementation.
- `experiment_config.py`: setup IDs and custom config fields.

## Setup

```bash
cd /path/to/onlinearith
source /path/to/venv/bin/activate
export PYTHONPATH="$(pwd)/../transformers/src:${PYTHONPATH}"
```

## Common commands

```bash
python ppltest.py --list
python ppltest.py --setup 6 --lite --limit-samples 2
python ppl_batch.py --list
python calibrate.py --list
python test_mxfp8linear.py
```

## PPL methodology invariants

Full PPL uses WikiText-2 raw test, `MAX_LENGTH=4096`, `STRIDE=512`, masked context labels, and weighted NLL accumulation. `--limit-samples` is only for smoke tests.

## Distributed note

Current `--nproc` runs data-parallel workers. Each rank loads a full model copy. This is not model sharding and will not solve Qwen3-8B per-GPU OOM by itself.

## Next planned work

After cleanup, implement exact output-chunked MX baseline and separate lite/full stats overhead from PPL calculation.
```

## Phase 5: optional PPL helper extraction, only after approval

This phase touches behavior-sensitive code. Do not do it automatically.

Potential extraction target: `ppl_utils.py` with:

- `precompute_windows(seq_len, max_length, stride)`
- `make_ppl_labels(input_ids, trg_len)`
- `accumulate_window_loss(loss, trg_len)`
- metric formatting helpers

Required equivalence test before accepting:

1. Run old code on `--setup 1 --limit-samples 2` and save metrics.
2. Run refactored code with the same command.
3. Compare `scored_tokens`, `mean_nll_nats`, and `token_perplexity` exactly or within formatting roundoff.

## Cleanup acceptance checklist

Before considering cleanup complete, verify:

- [ ] `AGENTS.md` exists at repo root.
- [ ] `CLAUDE.md` is removed from root or archived as obsolete.
- [ ] README no longer contradicts the operational source-of-truth decision.
- [ ] `.gitignore` covers caches and generated experiment outputs.
- [ ] `__pycache__/` is not tracked.
- [ ] `ppl_batch_summary.json` is not tracked as source.
- [ ] No active file contains `/home/xzjnew/` or another personal hardcoded source path.
- [ ] `python ppltest.py --list` works or the failure is documented as an environment issue.
- [ ] `python calibrate.py --list` works or the failure is documented as an environment issue.
- [ ] No cleanup patch changes `MAX_LENGTH`, `STRIDE`, dataset split, setup IDs, or weighted NLL logic.
- [ ] The diff does not implement the OOM fixes yet.

## Handoff prompt for the later OOM-fix task

After cleanup is merged, use a new Codex session with a focused prompt like:

```text
Now implement Fix 1 and Fix 5 only. Add an exact output-chunked MX-only baseline path in _MXFPLinearBase that preserves old MX math, and make lite/full stats behavior explicit so PPL logits/loss are unchanged. Add small validation tests comparing old unchunked vs chunked MX on small tensors and full vs lite PPL loss on a tiny window. Do not change MAX_LENGTH, STRIDE, dataset, tokenizer, or calibration semantics.
```
