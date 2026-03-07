"""Shared helpers for repo-local developer scripts."""

from __future__ import annotations

import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API_PREFIX = "/api/v1"

_ENV_LOADED = False


def load_repo_env() -> None:
    """Load the repo-root .env once when python-dotenv is available."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except ImportError:
            pass
    _ENV_LOADED = True


def normalize_bool_env(name: str, *, default: str = "false") -> None:
    """Coerce noisy truthy/falsy env values to a stable boolean string."""
    value = os.environ.get(name)
    if value is None:
        os.environ[name] = default
        return
    normalized = value.strip().lower()
    if normalized in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        os.environ[name] = "true" if normalized in {"1", "true", "yes", "on"} else "false"
        return
    os.environ[name] = default


def normalize_cml_base_url(raw: str | None, *, default: str = "http://localhost:8000") -> str:
    """Return the root CML base URL without a trailing /api/v1 suffix."""
    value = (raw or default).strip().rstrip("/")
    if not value:
        return default
    if value.endswith(API_PREFIX):
        trimmed = value[: -len(API_PREFIX)].rstrip("/")
        return trimmed or default
    return value


def build_api_url(base_url: str | None, path: str) -> str:
    """Build a concrete API URL for an endpoint path like /memory/read."""
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{normalize_cml_base_url(base_url)}{API_PREFIX}{normalized_path}"


def default_tenant_id(prefix: str) -> str:
    """Generate a per-run tenant id for smoke probes."""
    safe_prefix = prefix.strip().replace(" ", "-") or "probe"
    return f"{safe_prefix}-{int(time.time())}"
