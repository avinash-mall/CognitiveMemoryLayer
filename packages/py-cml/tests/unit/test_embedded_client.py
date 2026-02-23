"""Unit tests for embedded CML client (config and mocked orchestrator)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cml.embedded import EmbeddedCognitiveMemoryLayer
from cml.embedded_config import EmbeddedConfig


def test_embedded_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """EmbeddedConfig has expected defaults."""
    for key in (
        "EMBEDDING_INTERNAL__PROVIDER",
        "EMBEDDING_INTERNAL__MODEL",
        "EMBEDDING_INTERNAL__DIMENSIONS",
        "EMBEDDING_INTERNAL__BASE_URL",
        "EMBEDDING_INTERNAL__LOCAL_MODEL",
        "LLM_INTERNAL__PROVIDER",
        "LLM_INTERNAL__MODEL",
        "LLM_INTERNAL__BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    config = EmbeddedConfig()
    assert config.storage_mode == "lite"
    assert config.tenant_id == "default"
    assert config.database.database_url.startswith("sqlite")
    assert config.embedding.provider == "local"
    assert config.llm.provider == "openai"


@pytest.mark.asyncio
async def test_embedded_write_returns_write_response_when_initialized() -> None:
    """When orchestrator is set, write() returns WriteResponse."""
    config = EmbeddedConfig(tenant_id="test-tenant")
    client = EmbeddedCognitiveMemoryLayer(config=config)
    mid = uuid4()
    client._orchestrator = MagicMock()
    client._orchestrator.write = AsyncMock(
        return_value={
            "memory_id": mid,
            "chunks_created": 1,
            "message": "ok",
        }
    )
    client._initialized = True
    result = await client.write("hello")
    assert result.success is True
    assert result.memory_id == mid
    assert result.chunks_created == 1
    client._orchestrator.write.assert_called_once()
    call_kw = client._orchestrator.write.call_args[1]
    assert call_kw["tenant_id"] == "test-tenant"
    assert call_kw["content"] == "hello"


@pytest.mark.asyncio
async def test_embedded_read_passes_filters_and_user_timezone() -> None:
    """Embedded read() passes memory_types, since, until, and user_timezone to orchestrator.read()."""
    from cml.models.enums import MemoryType

    config = EmbeddedConfig(tenant_id="t1")
    client = EmbeddedCognitiveMemoryLayer(config=config)
    client._orchestrator = MagicMock()
    client._orchestrator.read = AsyncMock(
        return_value=MagicMock(
            all_memories=[],
            facts=[],
            preferences=[],
            recent_episodes=[],
        )
    )
    client._initialized = True
    since = datetime(2025, 1, 1, tzinfo=UTC)
    until = datetime(2025, 1, 2, tzinfo=UTC)
    await client.read(
        "q",
        memory_types=[MemoryType.PREFERENCE],
        since=since,
        until=until,
        user_timezone="America/New_York",
    )
    client._orchestrator.read.assert_called_once()
    call_kw = client._orchestrator.read.call_args[1]
    assert call_kw["tenant_id"] == "t1"
    assert call_kw["query"] == "q"
    assert call_kw["memory_types"] == ["preference"]
    assert call_kw["since"] == since
    assert call_kw["until"] == until
    assert call_kw["user_timezone"] == "America/New_York"


@pytest.mark.asyncio
async def test_embedded_ensure_initialized_raises() -> None:
    """write() raises if not initialized."""
    client = EmbeddedCognitiveMemoryLayer()
    with pytest.raises(RuntimeError, match="not initialized"):
        await client.write("x")
