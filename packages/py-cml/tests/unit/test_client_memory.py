"""Unit tests for client memory operations with mocked transport."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig
from cml.models import ReadResponse, WriteResponse

# ---- Sync client ----


def test_sync_write_returns_write_response() -> None:
    """Sync client write() returns WriteResponse."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
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


def test_sync_read_returns_read_response() -> None:
    """Sync client read() returns ReadResponse."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
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


def test_sync_forget_raises_without_selector() -> None:
    """Sync client forget() raises ValueError when no memory_ids, query, or before."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    with pytest.raises(ValueError, match="At least one of memory_ids, query, or before"):
        client.forget()


def test_sync_delete_all_raises_without_confirm() -> None:
    """Sync client delete_all() raises ValueError when confirm is not True."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    with pytest.raises(ValueError, match="confirm=True"):
        client.delete_all()
    with pytest.raises(ValueError, match="confirm=True"):
        client.delete_all(confirm=False)


def test_sync_delete_all_returns_affected_count_when_confirmed() -> None:
    """Sync client delete_all(confirm=True) returns affected_count from response."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"affected_count": 42}
    )
    result = client.delete_all(confirm=True)
    assert result == 42
    client._transport.request.assert_called_once_with("DELETE", "/memory/all", use_admin_key=True)


def test_sync_get_context_returns_str() -> None:
    """Sync client get_context() returns formatted context string."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
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


def test_sync_remember_delegates_to_write() -> None:
    """Sync client remember() delegates to write()."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"success": True, "chunks_created": 0, "message": ""}
    )
    result = client.remember("something")
    assert isinstance(result, WriteResponse)
    assert result.success is True
    client._transport.request.assert_called_once()
    assert client._transport.request.call_args[1]["json"]["content"] == "something"


def test_sync_search_delegates_to_read() -> None:
    """Sync client search() delegates to read()."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
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
async def test_async_write_returns_write_response() -> None:
    """Async client write() returns WriteResponse."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
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
async def test_async_read_returns_read_response() -> None:
    """Async client read() returns ReadResponse."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
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
async def test_async_forget_raises_without_selector() -> None:
    """Async client forget() raises ValueError when no selector provided."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = AsyncCognitiveMemoryLayer(config=config)
    with pytest.raises(ValueError, match="At least one of memory_ids, query, or before"):
        await client.forget()


@pytest.mark.asyncio
async def test_async_delete_all_raises_without_confirm() -> None:
    """Async client delete_all() raises ValueError when confirm is not True."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = AsyncCognitiveMemoryLayer(config=config)
    with pytest.raises(ValueError, match="confirm=True"):
        await client.delete_all(confirm=False)


@pytest.mark.asyncio
async def test_async_get_context_returns_str() -> None:
    """Async client get_context() returns formatted context string."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
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
