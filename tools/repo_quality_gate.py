#!/usr/bin/env python3
"""Cheap repository-quality gate for onlinearith cleanup passes.

This script is intentionally static/lightweight. It does not load a model, download
datasets, or run PPL. It checks contracts that should remain true while refactoring.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

EXPECTED_PPL_CONSTANTS = {"MAX_LENGTH": 4096, "STRIDE": 512}
EXPECTED_RUNTIME_STATS_FIELDS = {
    "msd_perf_stats_enabled",
    "msd_perf_stats_lite",
    "msd_figure5_layer_cycles",
}
MX_FORMAT_FLAGS = ("use_mxfp8", "use_mxfp6", "use_mxfp4")
PUBLIC_QWEN3_SYMBOLS = (
    "_MXFPLinearBase",
    "MXFP8Linear",
    "MXFP6Linear",
    "MXFP4Linear",
    "_make_linear",
    "MSDComputeContext",
)
SUSPICIOUS_PPL_WORKAROUNDS = (
    "device_map",
    "logits_to_keep",
    "use_cache=False",
    "use_cache = False",
)


@dataclass
class Finding:
    level: str
    message: str


class Gate:
    def __init__(self, strict: bool) -> None:
        self.strict = strict
        self.findings: list[Finding] = []

    def ok(self, message: str) -> None:
        self.findings.append(Finding("OK", message))

    def warn(self, message: str) -> None:
        self.findings.append(Finding("WARN", message))

    def fail(self, message: str) -> None:
        self.findings.append(Finding("FAIL", message))

    def warn_or_fail(self, message: str) -> None:
        if self.strict:
            self.fail(message)
        else:
            self.warn(message)

    def exit_code(self) -> int:
        return 1 if any(f.level == "FAIL" for f in self.findings) else 0

    def print(self) -> None:
        for f in self.findings:
            print(f"[{f.level}] {f.message}")


def parse_assign_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out


def load_experiment_config(root: Path):
    path = root / "experiment_config.py"
    spec = importlib.util.spec_from_file_location("_onlinearith_experiment_config_for_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    return module


def iter_python_files(root: Path) -> Iterable[Path]:
    ignored_parts = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
    for path in root.rglob("*.py"):
        if any(part in ignored_parts for part in path.parts):
            continue
        yield path


def python_without_comments(text: str) -> str:
    """Return Python source with comments removed for token-level string scans."""
    out: list[str] = []
    for token in tokenize.generate_tokens(StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            out.append(token.string)
    return " ".join(out)


def check_ppl_constants(gate: Gate, root: Path) -> None:
    for rel in ("ppltest.py", "ppl_batch.py"):
        path = root / rel
        if not path.exists():
            gate.warn_or_fail(f"Missing {rel}; expected top-level PPL runner to remain present")
            continue
        values = parse_assign_constants(path)
        for name, expected in EXPECTED_PPL_CONSTANTS.items():
            actual = values.get(name)
            if actual == expected:
                gate.ok(f"{rel}: {name}={expected}")
            else:
                gate.fail(f"{rel}: expected {name}={expected}, found {actual!r}")


def check_experiment_config(gate: Gate, root: Path) -> None:
    path = root / "experiment_config.py"
    if not path.exists():
        gate.fail("Missing experiment_config.py")
        return
    try:
        cfg = load_experiment_config(root)
    except Exception as exc:
        gate.warn_or_fail(f"Could not import experiment_config.py: {exc}")
        return

    baseline = getattr(cfg, "BASELINE_CONFIG", None)
    fields = getattr(cfg, "MXFP_MSD_FIELDS", None)
    setups = getattr(cfg, "SETUPS", None)
    if not isinstance(baseline, dict) or not isinstance(fields, list) or not isinstance(setups, list):
        gate.fail("experiment_config.py must expose BASELINE_CONFIG, MXFP_MSD_FIELDS, and SETUPS")
        return

    baseline_keys = set(baseline)
    field_keys = set(fields)
    if baseline_keys == field_keys:
        gate.ok("BASELINE_CONFIG keys match MXFP_MSD_FIELDS")
    else:
        missing_from_baseline = sorted(field_keys - baseline_keys)
        missing_from_fields = sorted(baseline_keys - field_keys)
        gate.warn_or_fail(
            "Config registry drift: "
            f"fields not in baseline={missing_from_baseline}, baseline not in fields={missing_from_fields}"
        )

    missing_runtime = sorted(EXPECTED_RUNTIME_STATS_FIELDS - field_keys)
    if missing_runtime:
        gate.warn_or_fail(f"Runtime stats fields are not fully registered: missing {missing_runtime}")
    else:
        gate.ok("Runtime stats fields are registered")

    ids: list[int] = []
    tags: list[str] = []
    unknown_override_keys: list[str] = []
    format_conflicts: list[str] = []
    for setup in setups:
        if not (isinstance(setup, tuple) and len(setup) == 4):
            gate.fail(f"Invalid setup tuple shape: {setup!r}")
            continue
        setup_id, tag, _desc, overrides = setup
        ids.append(setup_id)
        tags.append(tag)
        if not isinstance(overrides, dict):
            gate.fail(f"Setup {setup_id}/{tag} overrides must be a dict")
            continue
        unknown_override_keys.extend(sorted(set(overrides) - baseline_keys))
        enabled = [flag for flag in MX_FORMAT_FLAGS if bool(overrides.get(flag, baseline.get(flag, False)))]
        if len(enabled) > 1:
            format_conflicts.append(f"{setup_id}:{tag}:{enabled}")
        fmt = overrides.get("mxfp6_format", baseline.get("mxfp6_format"))
        if fmt not in {"e2m3", "e3m2"}:
            gate.fail(f"Setup {setup_id}/{tag}: unsupported mxfp6_format={fmt!r}")

    if len(ids) == len(set(ids)):
        gate.ok(f"Setup IDs are unique ({len(ids)} setups)")
    else:
        gate.fail("Duplicate setup IDs found")
    if len(tags) == len(set(tags)):
        gate.ok("Setup tags are unique")
    else:
        gate.fail("Duplicate setup tags found")
    if unknown_override_keys:
        gate.fail(f"Setup overrides contain unknown keys: {sorted(set(unknown_override_keys))}")
    else:
        gate.ok("All setup override keys are registered")
    if format_conflicts:
        gate.fail(f"More than one MX format enabled in setup(s): {format_conflicts}")
    else:
        gate.ok("MX format flags are mutually exclusive across setups")


def check_transformers_qwen3(gate: Gate, transformers_root: Path) -> None:
    qwen3 = transformers_root / "src" / "transformers" / "models" / "qwen3"
    modeling = qwen3 / "modeling_qwen3.py"
    config = qwen3 / "configuration_qwen3.py"
    if not qwen3.exists():
        gate.warn_or_fail(f"Sibling qwen3 directory not found: {qwen3}")
        return
    if not modeling.exists():
        gate.fail(f"Missing {modeling}")
    else:
        text = modeling.read_text(encoding="utf-8")
        missing_symbols = [name for name in PUBLIC_QWEN3_SYMBOLS if name not in text]
        if missing_symbols:
            gate.warn_or_fail(f"Public qwen3 symbols not visible in modeling_qwen3.py: {missing_symbols}")
        else:
            gate.ok("Public qwen3 custom symbols remain visible in modeling_qwen3.py")
    if not config.exists():
        gate.fail(f"Missing {config}")
    else:
        text = config.read_text(encoding="utf-8")
        missing_runtime = sorted(EXPECTED_RUNTIME_STATS_FIELDS - set(re.findall(r"['\"](msd_[A-Za-z0-9_]+)['\"]", text)))
        # String search fallback catches attribute names without quotes.
        missing_runtime = [name for name in EXPECTED_RUNTIME_STATS_FIELDS if name not in text]
        if missing_runtime:
            gate.warn_or_fail(f"Qwen3Config does not visibly define runtime stats fields: {missing_runtime}")
        else:
            gate.ok("Qwen3Config visibly defines runtime stats fields")


def check_personal_paths(gate: Gate, root: Path) -> None:
    offenders: list[str] = []
    home_prefix = "/" + "home" + "/"
    users_prefix = "/" + "Users" + "/"
    pattern = re.compile(
        rf"({re.escape(home_prefix)}[^\s'\"]+|{re.escape(users_prefix)}[^\s'\"]+)"
    )
    for path in iter_python_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            offenders.append(str(path.relative_to(root)))
    if offenders:
        gate.warn_or_fail(f"Hard-coded personal absolute paths found in Python files: {offenders}")
    else:
        gate.ok("No hard-coded personal absolute paths found in onlinearith Python files")


def check_no_hidden_ppl_workarounds(gate: Gate, root: Path) -> None:
    offenders: list[str] = []
    for rel in ("ppltest.py", "ppl_batch.py"):
        path = root / rel
        if not path.exists():
            continue
        text = python_without_comments(path.read_text(encoding="utf-8", errors="ignore"))
        for token in SUSPICIOUS_PPL_WORKAROUNDS:
            if token in text:
                offenders.append(f"{rel}:{token}")
    if offenders:
        gate.warn_or_fail(
            "PPL runners contain possible hidden OOM/methodology workaround tokens: "
            f"{offenders}. Verify these are intentional before merging cleanup."
        )
    else:
        gate.ok("No suspicious hidden PPL workaround tokens found in PPL runners")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onlinearith-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--transformers-root", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Turn cleanup warnings into failures.")
    args = parser.parse_args()

    root = args.onlinearith_root.resolve()
    transformers_root = (args.transformers_root or (root.parent / "transformers")).resolve()

    gate = Gate(strict=args.strict)
    gate.ok(f"onlinearith root: {root}")
    gate.ok(f"transformers root: {transformers_root}")

    check_ppl_constants(gate, root)
    check_experiment_config(gate, root)
    check_transformers_qwen3(gate, transformers_root)
    check_personal_paths(gate, root)
    check_no_hidden_ppl_workarounds(gate, root)

    gate.print()
    return gate.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
