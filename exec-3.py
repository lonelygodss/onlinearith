#!/usr/bin/env python3
"""Run commands defined in this file with optional common prefix/suffix.

Edit COMMON_PREFIX, COMMON_SUFFIX, and COMMANDS below, then run:
    python auto_exec.py
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Sequence


# Applied to every command unless that command sets its own prefix/suffix.
COMMON_PREFIX = ""
COMMON_SUFFIX = ""

# Safety toggles.
DRY_RUN = False
STOP_ON_ERROR = True


def _merge_command_parts(prefix: str, command: str, suffix: str) -> str:
    """Join prefix/command/suffix with single spaces."""
    pieces = []
    if prefix and prefix.strip():
        pieces.append(prefix.strip())
    pieces.append(command.strip())
    if suffix and suffix.strip():
        pieces.append(suffix.strip())
    return " ".join(pieces)


@dataclass(frozen=True)
class CommandSpec:
    command: str
    prefix: str | None = None
    suffix: str | None = None

    def render(self) -> str:
        prefix = COMMON_PREFIX if self.prefix is None else self.prefix
        suffix = COMMON_SUFFIX if self.suffix is None else self.suffix
        return _merge_command_parts(prefix, self.command, suffix)


# Add commands here.
# Each entry uses COMMON_PREFIX/COMMON_SUFFIX unless overridden.
COMMANDS: Sequence[CommandSpec] = (
    CommandSpec("echo hello"),
    CommandSpec("python ppltest.py --nproc 1 --setup 6 --calibration ../data/calib-data/12db/calibration_MXFP8_fixed_sum.json --lite --output ../data/calib-data/12db/ppl_results_MXFP8_fix_time.json --limit-samples 100 --figure5-layer-cycles --gpus 0"),
    CommandSpec('python ppltest.py --nproc 1 --setup 6 --calibration ../data/calib-data/14db/calibration_MXFP8_fixed_sum.json --lite --output ../data/calib-data/14db/ppl_results_MXFP8_fix_time.json --limit-samples 100 --figure5-layer-cycles --gpus 0'),
    CommandSpec('python ppltest.py --nproc 1 --setup 6 --calibration ../data/calib-data/15db/calibration_MXFP8_fixed_sum.json --lite --output ../data/calib-data/15db/ppl_results_MXFP8_fix_time.json --limit-samples 100 --figure5-layer-cycles --gpus 0')
    # CommandSpec("python eval.py", suffix="--batch_size 8"),
)


def run_commands(commands: Sequence[CommandSpec]) -> int:
    total = len(commands)
    for index, spec in enumerate(commands, start=1):
        full_command = spec.render()
        print(f"[{index}/{total}] {full_command}")

        if DRY_RUN:
            continue

        result = subprocess.run(full_command, shell=True, text=True)
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}: {full_command}")
            if STOP_ON_ERROR:
                return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(run_commands(COMMANDS))
