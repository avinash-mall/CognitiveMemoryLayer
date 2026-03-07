from __future__ import annotations

import runpy
from types import SimpleNamespace
from unittest.mock import MagicMock

import redis.asyncio as redis
import structlog

from src.storage.redis import get_redis_client
from src.utils import logging_config


def test_get_redis_client_uses_settings_url(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(redis, "from_url", lambda url: (url, sentinel))
    monkeypatch.setattr(
        "src.storage.redis.get_settings",
        lambda: SimpleNamespace(database=SimpleNamespace(redis_url="redis://cache:6379/1")),
    )

    assert get_redis_client() == ("redis://cache:6379/1", sentinel)


def test_configure_logging_uses_json_renderer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(structlog, "configure", _capture)

    logging_config.configure_logging(log_level="debug", json_output=True)

    assert any(type(proc).__name__ == "JSONRenderer" for proc in captured["processors"])


def test_configure_logging_uses_console_renderer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(structlog, "configure", _capture)

    logging_config.configure_logging(log_level="info", json_output=False)

    assert any(type(proc).__name__ == "ConsoleRenderer" for proc in captured["processors"])


def test_get_logger_delegates_to_structlog(monkeypatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr(structlog, "get_logger", fake)

    logging_config.get_logger("cml.test")

    fake.assert_called_once_with("cml.test")


def test_main_runs_uvicorn_with_debug_reload(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: SimpleNamespace(debug=True),
    )
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs)))

    runpy.run_module("src.main", run_name="__main__")

    args, kwargs = calls[0]
    assert args == ("src.api.app:app",)
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000
    assert kwargs["reload"] is True
