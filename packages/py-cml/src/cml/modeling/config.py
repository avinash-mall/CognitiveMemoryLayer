"""Shared config helpers for cml.modeling."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path | None:
    """Detect repo root by scanning parent directories for marker files."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        markers = (
            candidate / "docker" / "docker-compose.yml",
            candidate / "evaluation" / "locomo_plus",
            candidate / "packages" / "models" / "model_pipeline.toml",
        )
        if all(marker.exists() for marker in markers):
            return candidate
    return None


def default_model_config_path(start: Path | None = None) -> Path:
    root = find_repo_root(start)
    if root is None:
        raise FileNotFoundError(
            "Could not detect repository root. Pass --config explicitly (e.g. packages/models/model_pipeline.toml)."
        )
    return root / "packages" / "models" / "model_pipeline.toml"
