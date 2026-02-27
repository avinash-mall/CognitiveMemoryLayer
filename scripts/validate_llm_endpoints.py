"""Fail-fast LLM endpoint reachability checks for container startup."""

from __future__ import annotations

import os
import urllib.error
import urllib.request


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _log(msg: str) -> None:
    print(f"[llm-validate] {msg}", flush=True)


def _should_skip() -> bool:
    if not _as_bool("LLM_STARTUP_VALIDATION_ENABLED", True):
        _log("skipped: LLM_STARTUP_VALIDATION_ENABLED is false")
        return True
    if _as_bool("GITHUB_ACTIONS", False):
        _log("skipped: running in GitHub Actions")
        return True
    if _as_bool("CI", False) and not _as_bool("LLM_STARTUP_VALIDATE_IN_CI", False):
        _log("skipped: running in CI and LLM_STARTUP_VALIDATE_IN_CI is false")
        return True
    if not _as_bool("FEATURES__USE_LLM_ENABLED", False):
        _log("skipped: FEATURES__USE_LLM_ENABLED is false")
        return True
    return False


def _normalize_base_url(raw: str) -> str:
    return raw.strip().rstrip("/")


def _candidate_urls(provider: str, base_url: str) -> list[str]:
    candidates: list[str] = []
    if base_url.endswith("/v1"):
        candidates.append(f"{base_url}/models")
    else:
        candidates.append(f"{base_url}/v1/models")
        candidates.append(f"{base_url}/models")
    if provider == "ollama":
        root = base_url[:-3] if base_url.endswith("/v1") else base_url
        candidates.append(f"{root}/api/tags")
    seen: set[str] = set()
    ordered: list[str] = []
    for url in candidates:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _reachable(url: str, timeout_seconds: float) -> tuple[bool, str]:
    req = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return True, f"{resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 401, 403, 404, 405}:
            return True, f"{exc.code}"
        return False, f"http_error:{exc.code}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    if _should_skip():
        return 0

    provider = (os.getenv("LLM_INTERNAL__PROVIDER") or "openai").strip().lower()
    base_url = _normalize_base_url(os.getenv("LLM_INTERNAL__BASE_URL", ""))
    timeout_seconds = float(os.getenv("LLM_STARTUP_VALIDATION_TIMEOUT_SEC", "3"))

    providers_requiring_base = {"ollama", "openai_compatible", "vllm", "sglang"}
    if provider in providers_requiring_base and not base_url:
        _log(f"failed: provider '{provider}' requires LLM_INTERNAL__BASE_URL")
        return 1
    if not base_url:
        _log(f"skipped: provider '{provider}' has no explicit local base URL")
        return 0

    urls = _candidate_urls(provider, base_url)
    failures: list[str] = []
    for url in urls:
        ok, detail = _reachable(url, timeout_seconds)
        if ok:
            _log(f"reachable: {url} (status={detail})")
            return 0
        failures.append(f"{url} -> {detail}")

    _log("failed: no reachable LLM endpoint")
    for item in failures:
        _log(f"attempt: {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
