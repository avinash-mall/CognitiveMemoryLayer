"""Shared helpers for evaluation modules."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path | None:
    """Detect repo root by scanning parents for known project markers."""
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


def load_repo_dotenv(repo_root: Path) -> None:
    """Load repo .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
