"""Shared helpers for runnable examples."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from cml import CMLConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_NON_INTERACTIVE_ENV = "CML_EXAMPLE_NON_INTERACTIVE"
EXAMPLE_INPUTS_ENV = "CML_EXAMPLE_INPUTS"

_ENV_LOADED = False


def _normalize_boolean_env(name: str) -> None:
    value = os.environ.get(name)
    if value is None:
        return
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        return
    os.environ[name] = "false"


def load_repo_env() -> None:
    """Load the repo-root .env file once when python-dotenv is available."""
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
    _normalize_boolean_env("DEBUG")
    _ENV_LOADED = True


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def print_header(title: str) -> None:
    line = "=" * max(24, len(title) + 4)
    print(line)
    print(title)
    print(line)


def normalize_base_url(base_url: str, *, api_path: bool) -> str:
    """Normalize a CML base URL with or without the /api/v1 suffix."""
    value = base_url.strip().rstrip("/")
    if not value:
        fail("Set CML_BASE_URL in the repo root .env before running this example.")

    api_suffix = "/api/v1"
    if api_path:
        if value.endswith(api_suffix):
            return value
        return f"{value}{api_suffix}"

    if value.endswith(api_suffix):
        return value[: -len(api_suffix)] or value
    return value


def get_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def require_env(name: str, *, hint: str | None = None) -> str:
    value = get_env(name)
    if value is None:
        suffix = f" {hint}" if hint else ""
        fail(f"Set {name} in the repo root .env before running this example.{suffix}")
    return value


def api_base_url() -> str:
    return normalize_base_url(require_env("CML_BASE_URL"), api_path=True)


def sdk_base_url() -> str:
    return normalize_base_url(require_env("CML_BASE_URL"), api_path=False)


def build_cml_config(*, timeout: float = 30.0, require_admin_key: bool = False) -> CMLConfig:
    """Build a CMLConfig from the shared example environment."""
    from cml import CMLConfig

    return CMLConfig(
        api_key=require_env("CML_API_KEY"),
        base_url=sdk_base_url(),
        admin_api_key=require_env("CML_ADMIN_API_KEY")
        if require_admin_key
        else get_env("CML_ADMIN_API_KEY"),
        timeout=timeout,
        tenant_id=get_env("CML_TENANT_ID") or "default",
    )


def api_headers(*, use_admin_key: bool = False) -> dict[str, str]:
    api_key = require_env("CML_ADMIN_API_KEY") if use_admin_key else require_env("CML_API_KEY")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
    if use_admin_key:
        headers["X-Requested-With"] = "XMLHttpRequest"
    tenant_id = get_env("CML_TENANT_ID")
    if tenant_id:
        headers["X-Tenant-ID"] = tenant_id
    return headers


def is_non_interactive() -> bool:
    value = os.environ.get(EXAMPLE_NON_INTERACTIVE_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def scripted_inputs(default_inputs: Iterable[str] | None = None) -> list[str]:
    raw = os.environ.get(EXAMPLE_INPUTS_ENV, "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [line for line in raw.splitlines() if line.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        fail(f"{EXAMPLE_INPUTS_ENV} must be a JSON list or newline-delimited string.")
    return [str(item) for item in (default_inputs or [])]


def iter_user_inputs(prompt: str, *, default_inputs: Iterable[str] | None = None) -> Iterator[str]:
    """Yield inputs interactively or from the non-interactive contract."""
    if is_non_interactive():
        for item in scripted_inputs(default_inputs):
            print(f"{prompt}{item}")
            yield item
        return

    while True:
        yield input(prompt).strip()


def openai_settings() -> dict[str, str | None]:
    """Resolve OpenAI-compatible model settings for example scripts."""
    model = get_env("LLM_INTERNAL__MODEL") or get_env("OPENAI_MODEL")
    if not model:
        fail("Set OPENAI_MODEL or LLM_INTERNAL__MODEL in the repo root .env.")

    internal_base_url = get_env("LLM_INTERNAL__BASE_URL")
    if internal_base_url:
        return {
            "model": model,
            "base_url": internal_base_url,
            "api_key": get_env("LLM_INTERNAL__API_KEY") or get_env("OPENAI_API_KEY") or "dummy",
        }

    return {
        "model": model,
        "base_url": None,
        "api_key": require_env("OPENAI_API_KEY"),
    }


def anthropic_settings() -> dict[str, str]:
    return {
        "model": require_env("ANTHROPIC_MODEL"),
        "api_key": require_env("ANTHROPIC_API_KEY"),
    }


def explain_connection_failure() -> str:
    return (
        "Verify CML_API_KEY, CML_BASE_URL, and that the API server is running. "
        "See examples/README.md for local setup."
    )


load_repo_env()
