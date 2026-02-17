"""Unit tests for sync/async client memory operations (write, read, update, forget) with mocked transport."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig
from cml.models import (
    ForgetResponse,
    ReadResponse,
    UpdateResponse,
    WriteResponse,
)

# ---- Sync client ----


def test_sync_write_returns_write_response(cml_config: CMLConfig) -> None:
    """Sync client write() returns WriteResponse."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    mid = str(uuid4())
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "success": True,
            "memory_id": mid,
            "chunks_created": 1,
            "message": "ok",
        }
    )
    result = client.write("hello")
    assert isinstance(result, WriteResponse)
    assert result.success is True
    assert str(result.memory_id) == mid
    client._transport.request.assert_called_once()
    call = client._transport.request.call_args
    assert call[0] == ("POST", "/memory/write")
    assert call[1]["json"]["content"] == "hello"


def test_sync_read_returns_read_response(cml_config: CMLConfig) -> None:
    """Sync client read() returns ReadResponse."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "query": "prefs",
            "memories": [],
            "total_count": 0,
            "elapsed_ms": 1.0,
        }
    )
    result = client.read("prefs")
    assert isinstance(result, ReadResponse)
    assert result.query == "prefs"
    assert result.total_count == 0
    call = client._transport.request.call_args
    assert call[0] == ("POST", "/memory/read")
    assert call[1]["json"]["query"] == "prefs"


def test_sync_read_passes_user_timezone_when_provided(cml_config: CMLConfig) -> None:
    """Sync client read() includes user_timezone in request body when provided."""
    client = CognitiveMemoryLayer(config=cml_config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "query": "today?",
            "memories": [],
            "total_count": 0,
            "elapsed_ms": 1.0,
        }
    )
    client.read("today?", user_timezone="America/New_York")
    call = client._transport.request.call_args
    assert call[1]["json"]["user_timezone"] == "America/New_York"


def test_sync_turn_passes_user_timezone_when_provided(cml_config: CMLConfig) -> None:
    """Sync client turn() includes user_timezone in request body when provided."""
    client = CognitiveMemoryLayer(config=cml_config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "memory_context": "",
            "memories_retrieved": 0,
            "memories_stored": 0,
            "reconsolidation_applied": False,
        }
    )
    client.turn("Hi", user_timezone="Europe/London")
    call = client._transport.request.call_args
    assert call[1]["json"]["user_timezone"] == "Europe/London"


def test_sync_forget_raises_without_selector(cml_config: CMLConfig) -> None:
    """Sync client forget() raises ValueError when no memory_ids, query, or before."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    with pytest.raises(ValueError, match="At least one of memory_ids, query, or before"):
        client.forget()


def test_sync_delete_all_raises_without_confirm(cml_config: CMLConfig) -> None:
    """Sync client delete_all() raises ValueError when confirm is not True."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    with pytest.raises(ValueError, match="confirm=True"):
        client.delete_all()
    with pytest.raises(ValueError, match="confirm=True"):
        client.delete_all(confirm=False)


def test_sync_delete_all_returns_affected_count_when_confirmed(cml_config: CMLConfig) -> None:
    """Sync client delete_all(confirm=True) returns affected_count from response."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"affected_count": 42}
    )
    result = client.delete_all(confirm=True)
    assert result == 42
    client._transport.request.assert_called_once_with("DELETE", "/memory/all", use_admin_key=True)


def test_sync_get_context_returns_str(cml_config: CMLConfig) -> None:
    """Sync client get_context() returns formatted context string."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "query": "x",
            "memories": [],
            "llm_context": "## Memory\n(none)",
            "total_count": 0,
            "elapsed_ms": 0.5,
        }
    )
    result = client.get_context("x")
    assert isinstance(result, str)
    assert result == "## Memory\n(none)"


def test_sync_remember_delegates_to_write(cml_config: CMLConfig) -> None:
    """Sync client remember() delegates to write()."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"success": True, "chunks_created": 0, "message": ""}
    )
    result = client.remember("something")
    assert isinstance(result, WriteResponse)
    assert result.success is True
    client._transport.request.assert_called_once()
    assert client._transport.request.call_args[1]["json"]["content"] == "something"


def test_sync_search_delegates_to_read(cml_config: CMLConfig) -> None:
    """Sync client search() delegates to read()."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "query": "q",
            "memories": [],
            "total_count": 0,
            "elapsed_ms": 0.0,
        }
    )
    result = client.search("q")
    assert isinstance(result, ReadResponse)
    assert result.query == "q"


# ---- Async client ----


@pytest.mark.asyncio
async def test_async_write_returns_write_response(cml_config: CMLConfig) -> None:
    """Async client write() returns WriteResponse."""
    config = cml_config
    client = AsyncCognitiveMemoryLayer(config=config)
    mid = str(uuid4())
    client._transport.request = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "success": True,
            "memory_id": mid,
            "chunks_created": 1,
            "message": "ok",
        }
    )
    result = await client.write("hello")
    assert isinstance(result, WriteResponse)
    assert result.success is True
    client._transport.request.assert_called_once()
    call = client._transport.request.call_args
    assert call[0] == ("POST", "/memory/write")
    assert call[1]["json"]["content"] == "hello"


@pytest.mark.asyncio
async def test_async_read_returns_read_response(cml_config: CMLConfig) -> None:
    """Async client read() returns ReadResponse."""
    config = cml_config
    client = AsyncCognitiveMemoryLayer(config=config)
    client._transport.request = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "query": "prefs",
            "memories": [],
            "total_count": 0,
            "elapsed_ms": 1.0,
        }
    )
    result = await client.read("prefs")
    assert isinstance(result, ReadResponse)
    assert result.query == "prefs"


@pytest.mark.asyncio
async def test_async_forget_raises_without_selector(cml_config: CMLConfig) -> None:
    """Async client forget() raises ValueError when no selector provided."""
    config = cml_config
    client = AsyncCognitiveMemoryLayer(config=config)
    with pytest.raises(ValueError, match="At least one of memory_ids, query, or before"):
        await client.forget()


@pytest.mark.asyncio
async def test_async_delete_all_raises_without_confirm(cml_config: CMLConfig) -> None:
    """Async client delete_all() raises ValueError when confirm is not True."""
    config = cml_config
    client = AsyncCognitiveMemoryLayer(config=config)
    with pytest.raises(ValueError, match="confirm=True"):
        await client.delete_all(confirm=False)


@pytest.mark.asyncio
async def test_async_get_context_returns_str(cml_config: CMLConfig) -> None:
    """Async client get_context() returns formatted context string."""
    config = cml_config
    client = AsyncCognitiveMemoryLayer(config=config)
    client._transport.request = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "query": "x",
            "memories": [],
            "llm_context": "## Memory\n(none)",
            "total_count": 0,
            "elapsed_ms": 0.5,
        }
    )
    result = await client.get_context("x")
    assert isinstance(result, str)
    assert result == "## Memory\n(none)"


# ---- Sync update / forget / read params ----


def test_sync_update_returns_update_response(cml_config: CMLConfig) -> None:
    """Sync client update() returns UpdateResponse."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    mid = uuid4()
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"success": True, "memory_id": str(mid), "version": 2, "message": "ok"}
    )
    result = client.update(memory_id=mid, text="Updated")
    assert isinstance(result, UpdateResponse)
    assert result.success is True
    assert result.version == 2
    call = client._transport.request.call_args
    assert call[0] == ("POST", "/memory/update")
    assert call[1]["json"]["memory_id"] == str(mid)
    assert call[1]["json"]["text"] == "Updated"


def test_sync_forget_with_memory_ids(cml_config: CMLConfig) -> None:
    """Sync client forget(memory_ids=[...]) sends POST /memory/forget with serialized UUIDs."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    mid = uuid4()
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"success": True, "affected_count": 1, "message": "ok"}
    )
    result = client.forget(memory_ids=[mid])
    assert isinstance(result, ForgetResponse)
    assert result.affected_count == 1
    call = client._transport.request.call_args
    assert call[0] == ("POST", "/memory/forget")
    assert call[1]["json"]["memory_ids"] == [str(mid)]


def test_sync_forget_with_query(cml_config: CMLConfig) -> None:
    """Sync client forget(query=...) sends query in body."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"success": True, "affected_count": 3, "message": "ok"}
    )
    result = client.forget(query="obsolete")
    assert result.affected_count == 3
    assert client._transport.request.call_args[1]["json"]["query"] == "obsolete"


def test_sync_read_passes_response_format_and_since_until(cml_config: CMLConfig) -> None:
    """Sync client read() passes response_format, since, until in request body."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"query": "q", "memories": [], "total_count": 0, "elapsed_ms": 0.0}
    )
    since = datetime(2025, 1, 1, 0, 0, 0)
    until = datetime(2025, 12, 31, 23, 59, 59)
    client.read("q", response_format="llm_context", since=since, until=until)
    call = client._transport.request.call_args
    body = call[1]["json"]
    assert body.get("format") == "llm_context" or body.get("response_format") == "llm_context"
    assert "since" in body
    assert "until" in body


def test_sync_get_context_returns_empty_when_llm_context_missing(cml_config: CMLConfig) -> None:
    """get_context() returns empty string when llm_context is None/omitted."""
    config = cml_config
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"query": "x", "memories": [], "total_count": 0, "elapsed_ms": 0.0}
    )
    result = client.get_context("x")
    assert result == ""


@pytest.mark.asyncio
async def test_async_update_returns_update_response(cml_config: CMLConfig) -> None:
    """Async client update() returns UpdateResponse."""
    config = cml_config
    client = AsyncCognitiveMemoryLayer(config=config)
    mid = uuid4()
    client._transport.request = AsyncMock(  # type: ignore[method-assign]
        return_value={"success": True, "memory_id": str(mid), "version": 2, "message": "ok"}
    )
    result = await client.update(memory_id=mid, text="Updated")
    assert isinstance(result, UpdateResponse)
    assert result.version == 2


@pytest.mark.asyncio
async def test_async_forget_with_before(cml_config: CMLConfig) -> None:
    """Async client forget(before=...) sends before in body."""
    config = cml_config
    client = AsyncCognitiveMemoryLayer(config=config)
    client._transport.request = AsyncMock(  # type: ignore[method-assign]
        return_value={"success": True, "affected_count": 0, "message": "ok"}
    )
    before = datetime(2024, 6, 1, 0, 0, 0)
    await client.forget(before=before)
    call = client._transport.request.call_args
    assert call[1]["json"]["before"] == before.isoformat()
