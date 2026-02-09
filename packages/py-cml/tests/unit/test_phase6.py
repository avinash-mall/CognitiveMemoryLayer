"""Unit tests for Phase 6 developer experience."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer, configure_logging
from cml.config import CMLConfig
from cml.exceptions import AuthenticationError, CMLError
from cml.models import ReadResponse, StatsResponse, WriteResponse
from cml.utils.deprecation import deprecated
from cml.utils.serialization import CMLJSONEncoder, serialize_for_api


# ---- Exceptions: suggestion and request_id ----


def test_cml_error_has_suggestion_and_request_id() -> None:
    err = AuthenticationError(
        "Invalid API key",
        status_code=401,
        request_id="req-123",
        suggestion="Set CML_API_KEY env var or pass api_key= to constructor",
    )
    assert err.message == "Invalid API key"
    assert err.status_code == 401
    assert err.request_id == "req-123"
    assert err.suggestion == "Set CML_API_KEY env var or pass api_key= to constructor"
    assert "Suggestion:" in str(err)
    assert "Request ID:" in str(err)


def test_cml_error_repr() -> None:
    err = CMLError("Something failed", status_code=500)
    r = repr(err)
    assert "CMLError" in r
    assert "Something failed" in r
    assert "500" in r


# ---- configure_logging ----


def test_configure_logging_sets_level_and_handler() -> None:
    import logging

    from cml.utils.logging import logger

    configure_logging("DEBUG")
    assert logger.level == logging.DEBUG
    configure_logging("WARNING")
    assert logger.level == logging.WARNING


# ---- read_safe ----


def test_read_safe_returns_empty_on_connection_error() -> None:
    from cml.exceptions import ConnectionError

    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client.read = MagicMock(side_effect=ConnectionError("Failed to connect"))  # type: ignore[method-assign]
    result = client.read_safe("query")
    assert isinstance(result, ReadResponse)
    assert result.query == "query"
    assert result.total_count == 0
    assert result.memories == []
    assert result.elapsed_ms == 0.0


def test_read_safe_returns_empty_on_timeout() -> None:
    from cml.exceptions import TimeoutError

    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client.read = MagicMock(side_effect=TimeoutError("Timed out"))  # type: ignore[method-assign]
    result = client.read_safe("q")
    assert result.query == "q"
    assert result.total_count == 0


# ---- Response __str__ ----


def test_write_response_str() -> None:
    r = WriteResponse(success=True, memory_id=uuid4(), chunks_created=2, message="ok")
    s = str(r)
    assert "WriteResponse" in s
    assert "success=True" in s
    assert "chunks=2" in s


def test_read_response_str() -> None:
    from cml.models import MemoryItem

    r = ReadResponse(
        query="test",
        memories=[
            MemoryItem(
                id=uuid4(),
                text="short",
                type="fact",
                confidence=0.9,
                relevance=0.8,
                timestamp=datetime.now(),
            )
        ],
        total_count=1,
        elapsed_ms=10.0,
    )
    s = str(r)
    assert "Query:" in s
    assert "1 results" in s
    assert "[fact]" in s
    assert "short" in s


def test_stats_response_str() -> None:
    r = StatsResponse(
        total_memories=100,
        active_memories=80,
        silent_memories=10,
        archived_memories=10,
        by_type={},
        avg_confidence=0.8,
        avg_importance=0.7,
        estimated_size_mb=0.1,
    )
    s = str(r)
    assert "Memory Stats" in s
    assert "100 total" in s
    assert "80 active" in s


# ---- serialize_for_api ----


def test_serialize_for_api_uuid_datetime_none() -> None:
    uid = uuid4()
    dt = datetime(2025, 1, 1, 12, 0, 0)
    data = {"id": uid, "created": dt, "skip": None, "nested": {"a": uid}}
    out = serialize_for_api(data)
    assert out["id"] == str(uid)
    assert out["created"] == dt.isoformat()
    assert "skip" not in out
    assert out["nested"]["a"] == str(uid)


def test_cml_json_encoder_uuid_datetime() -> None:
    import json

    uid = uuid4()
    dt = datetime(2025, 1, 1, 12, 0, 0)
    s = json.dumps({"id": uid, "ts": dt}, cls=CMLJSONEncoder)
    assert str(uid) in s
    assert dt.isoformat() in s


# ---- Session context manager ----


def test_session_context_manager_injects_session_id() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client.create_session = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(session_id="sess-123")
    )
    client.write = MagicMock(  # type: ignore[method-assign]
        return_value=WriteResponse(success=True, memory_id=uuid4(), chunks_created=1)
    )
    with client.session(name="onboarding") as sess:
        assert sess.session_id == "sess-123"
        sess.write("Hello")
    client.write.assert_called_once()
    call_kw = client.write.call_args[1]
    assert call_kw["session_id"] == "sess-123"


# ---- Deprecation ----


@deprecated("new_func()", "2.0.0")
def _deprecated_func() -> str:
    return "ok"


def test_deprecated_emits_warning() -> None:
    with pytest.warns(DeprecationWarning, match="_deprecated_func is deprecated"):
        result = _deprecated_func()
    assert result == "ok"


# ---- Thread safety: set_tenant with lock ----


def test_set_tenant_thread_safe() -> None:
    import threading

    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000", tenant_id="t0")
    client = CognitiveMemoryLayer(config=config)
    client._transport.close = MagicMock()  # type: ignore[method-assign]
    results: list[str] = []

    def set_and_read(tenant: str) -> None:
        client.set_tenant(tenant)
        results.append(client.tenant_id)

    t1 = threading.Thread(target=set_and_read, args=("t1",))
    t2 = threading.Thread(target=set_and_read, args=("t2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(results) == 2
    assert set(results) == {"t1", "t2"}


# ---- Async event loop consistency ----


@pytest.mark.asyncio
async def test_async_client_raises_when_used_in_different_loop() -> None:
    import queue
    import threading

    result: queue.Queue = queue.Queue()

    def create_in_thread() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def create_client() -> AsyncCognitiveMemoryLayer:
            return AsyncCognitiveMemoryLayer(
                api_key="sk-test",
                base_url="http://localhost:8000",
            )

        client = loop.run_until_complete(create_client())
        result.put((client, loop))

    t = threading.Thread(target=create_in_thread)
    t.start()
    t.join()
    client, other_loop = result.get()
    try:
        with pytest.raises(RuntimeError, match="same event loop"):
            await client.health()
    finally:
        other_loop.close()
