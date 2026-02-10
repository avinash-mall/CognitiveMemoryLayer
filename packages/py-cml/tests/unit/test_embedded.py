"""Unit tests for embedded client (mocked orchestrator, no engine required)."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cml.embedded import EmbeddedCognitiveMemoryLayer
from cml.embedded_config import EmbeddedConfig


def test_embedded_config_defaults() -> None:
    """EmbeddedConfig has expected defaults."""
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
async def test_embedded_ensure_initialized_raises() -> None:
    """write() raises if not initialized."""
    client = EmbeddedCognitiveMemoryLayer()
    with pytest.raises(RuntimeError, match="not initialized"):
        await client.write("x")
