"""Hatch build plugin: set package version from .env (VERSION) or env var or VERSION file."""

from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


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
        import os

        root = Path(self.root)
        version = (
            _read_version_from_env_file(root)
            or os.environ.get("VERSION")
            or _read_version_from_version_file(root)
            or "0.0.0"
        )
        metadata["version"] = version
