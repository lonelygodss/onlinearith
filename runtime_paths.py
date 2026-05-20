"""Runtime path defaults and CLI normalization helpers."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent
TRANSFORMERS_SRC = WORKSPACE_ROOT / "transformers" / "src"


def default_model_path(model_dir_name: str) -> str:
    """Return the historical relative model path used by root scripts."""
    return f"../{model_dir_name}"


def default_data_dir() -> Path:
    """Return the historical shared data directory."""
    return Path("../data")


def normalize_output_dir(path: str | Path | None, default: Path) -> Path:
    """Expand a user-provided output directory or keep the script default."""
    if path is None:
        return default
    return Path(path).expanduser().resolve()


def describe_missing_model_path(model_path: str | Path) -> str:
    """Build a clear local-files-only error for missing model directories."""
    return (
        f"Model path not found: {model_path}. "
        "These scripts use local_files_only=True; pass --model-path or place the model at the default path."
    )
