"""Hatch build plugin: set package version from .env (VERSION) or env var or VERSION file."""

import os
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


def get_version(root: Path | None = None) -> str:
    """Return project version (same logic as the metadata hook). Use for badges/scripts.
    If root is None, uses the directory containing this file (repo root).
    """
    if root is None:
        root = Path(__file__).resolve().parent
    return (
        _read_version_from_env_file(root)
        or os.environ.get("VERSION")
        or _read_version_from_version_file(root)
        or "0.0.0"
    )


def _read_version_from_env_file(root: Path) -> str | None:
    """Read VERSION from .env in project root. Returns None if missing or unset."""
    env_file = root / ".env"
    if not env_file.is_file():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "VERSION":
            return value.strip().strip('"').strip("'") or None
    return None


def _read_version_from_version_file(root: Path) -> str | None:
    """Read first line of VERSION file in project root. Used as fallback when .env is absent."""
    version_file = root / "VERSION"
    if not version_file.is_file():
        return None
    return version_file.read_text().strip() or None


class VersionFromEnvMetadataHook(MetadataHookInterface):
    """Set project version from .env (VERSION=...), then env var VERSION, then VERSION file."""

    def update(self, metadata: dict) -> None:
        root = Path(self.root)
        metadata["version"] = get_version(root)
