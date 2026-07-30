"""Unit tests for the Packages Audit implementation (PKG-01 through PKG-09)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# PR-02 (PKG-04): Retry config.max_retry_delay wiring
# ---------------------------------------------------------------------------


class TestRetryMaxDelayWiring:
    """Verify that config.max_retry_delay caps backoff."""

    def test_sync_sleep_respects_max_delay(self):
        from cml.transport.retry import _sleep_with_backoff

        actual_sleeps: list[float] = []
        with patch("cml.transport.retry.time.sleep", side_effect=lambda d: actual_sleeps.append(d)):
            _sleep_with_backoff(attempt=10, base_delay=1.0, max_delay=5.0)

        assert all(d <= 5.0 for d in actual_sleeps)

    def test_async_sleep_respects_max_delay(self):
        from cml.transport.retry import _async_sleep_with_backoff

        actual_sleeps: list[float] = []

        async def fake_sleep(d: float):
            actual_sleeps.append(d)

        with patch("cml.transport.retry.asyncio.sleep", side_effect=fake_sleep):
            asyncio.run(_async_sleep_with_backoff(attempt=10, base_delay=1.0, max_delay=5.0))

        assert all(d <= 5.0 for d in actual_sleeps)

    def test_retry_sync_reads_config_max_retry_delay(self):
        from cml.config import CMLConfig
        from cml.exceptions import ServerError
        from cml.transport.retry import retry_sync

        config = CMLConfig(
            api_key="test",
            base_url="http://localhost:8000",
            max_retries=1,
            retry_delay=1.0,
            max_retry_delay=3.0,
        )
        actual_sleeps: list[float] = []
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ServerError("boom")

        with (
            patch("cml.transport.retry.time.sleep", side_effect=lambda d: actual_sleeps.append(d)),
            pytest.raises(ServerError),
        ):
            retry_sync(config, failing_func)

        assert all(d <= 3.0 for d in actual_sleeps)


# ---------------------------------------------------------------------------
# PR-02 (PKG-05): Export/import fidelity
# ---------------------------------------------------------------------------


class TestExportImportFidelity:
    """Verify that import preserves fields beyond text and metadata."""

    def test_roundtrip_preserves_fields(self, tmp_path: Path):
        export_path = tmp_path / "export.jsonl"
        records = [
            {
                "id": "rec-001",
                "text": "User prefers dark mode",
                "type": "preference",
                "confidence": 0.95,
                "timestamp": "2026-01-15T10:00:00+00:00",
                "context_tags": ["settings", "ui"],
                "namespace": "global",
                "source_session_id": "sess-42",
                "metadata": {"key": "value"},
            }
        ]
        with open(export_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        write_calls: list[dict[str, Any]] = []

        class FakeTarget:
            def write(self, text, **kwargs):
                write_calls.append({"text": text, **kwargs})

        from cml.embedded_utils import import_memories_async

        # The import helper accepts concrete client types; for this unit test we
        # only need an object with a compatible `write()` method.
        target = cast("Any", FakeTarget())
        count = asyncio.run(import_memories_async(target, str(export_path)))

        assert count == 1
        call = write_calls[0]
        assert call["text"] == "User prefers dark mode"
        assert call["memory_type"] == "preference"
        assert call.get("context_tags") == ["settings", "ui"]
        assert call.get("namespace") == "global"
        assert call.get("session_id") == "sess-42"
        assert call["metadata"].get("_imported_confidence") == 0.95


# ---------------------------------------------------------------------------
# PR-04: Eval CLI explicit path requirements
# ---------------------------------------------------------------------------


class TestEvalCliPaths:
    """Verify eval CLI errors clearly when repo root is not detected."""

    def test_default_repo_root_returns_none_outside_repo(self, tmp_path: Path):
        from cml.eval.config import find_repo_root

        result = find_repo_root(tmp_path)
        assert result is None

    def test_require_path_raises_on_none(self):
        from cml.eval.cli import _require_path

        with pytest.raises(SystemExit):
            _require_path(None, "repo-root")


# ---------------------------------------------------------------------------
# PR-05 (PKG-09): Client parity
# ---------------------------------------------------------------------------


class TestClientParity:
    """Verify sync and async clients expose matching public APIs."""

    def test_public_method_parity(self):
        from cml.async_client import AsyncCognitiveMemoryLayer
        from cml.client import CognitiveMemoryLayer

        skip = {
            "__aenter__",
            "__aexit__",
            "__enter__",
            "__exit__",
            "__del__",
            "__init__",
            "close",
            "_ensure_same_loop",
            "_loop",
        }

        sync_methods = {
            name
            for name in dir(CognitiveMemoryLayer)
            if not name.startswith("_") and callable(getattr(CognitiveMemoryLayer, name, None))
        }
        async_methods = {
            name
            for name in dir(AsyncCognitiveMemoryLayer)
            if not name.startswith("_") and callable(getattr(AsyncCognitiveMemoryLayer, name, None))
        }

        sync_only = sync_methods - async_methods - skip
        async_only = async_methods - sync_methods - skip

        assert not sync_only, f"Sync-only methods: {sync_only}"
        assert not async_only, f"Async-only methods: {async_only}"


# ---------------------------------------------------------------------------
# PR-05 (PKG-09): Shared _endpoints.py
# ---------------------------------------------------------------------------


class TestSharedEndpoints:
    """Verify shared endpoint helpers produce correct payloads."""

    def test_build_write_body(self):
        from cml._endpoints import build_write_body

        body = build_write_body(
            "test content",
            context_tags=["tag1"],
            namespace="ns1",
        )
        assert body["content"] == "test content"
        assert body["context_tags"] == ["tag1"]
        assert body["namespace"] == "ns1"

    def test_build_read_body(self):
        from cml._endpoints import build_read_body

        body = build_read_body("search query", max_results=5)
        assert body["query"] == "search query"
        assert body["max_results"] == 5

    def test_eval_mode_headers(self):
        from cml._endpoints import eval_mode_headers

        assert eval_mode_headers(True) == {"X-Eval-Mode": "true"}
        assert eval_mode_headers(False) is None


# ---------------------------------------------------------------------------
# PR-05 (PKG-01): Packaging structure
# ---------------------------------------------------------------------------


class TestPackagingStructure:
    """Verify pyproject.toml dependency split."""

    def test_base_deps_exclude_server(self):
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")
        with open(pyproject, "rb") as f:
            config = tomllib.load(f)
        base_deps = [d.lower() for d in config["project"]["dependencies"]]
        base_text = " ".join(base_deps)
        assert "fastapi" not in base_text
        assert "uvicorn" not in base_text
        assert "neo4j" not in base_text
        assert "redis" not in base_text
        assert "celery" not in base_text

    def test_server_extras_exist(self):
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")
        with open(pyproject, "rb") as f:
            config = tomllib.load(f)
        optional = config["project"]["optional-dependencies"]
        assert "server" in optional
        server_text = " ".join(d.lower() for d in optional["server"])
        assert "fastapi" in server_text

    def test_wheel_does_not_package_repo_src(self):
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")
        with open(pyproject, "rb") as f:
            config = tomllib.load(f)
        wheel_cfg = config["tool"]["hatch"]["build"]["targets"]["wheel"]
        packages = [str(pkg) for pkg in wheel_cfg.get("packages", [])]
        assert "src" not in packages
        assert "packages/py-cml/src/cml" in packages
