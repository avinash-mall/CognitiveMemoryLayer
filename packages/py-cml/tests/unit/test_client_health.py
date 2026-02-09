"""Tests for client health() with mocked transport."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer, HealthResponse
from cml.config import CMLConfig


def test_sync_health_returns_health_response() -> None:
    """Sync client health() returns HealthResponse from transport."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"status": "healthy", "version": "1.0.0", "components": {}}
    )
    result = client.health()
    assert isinstance(result, HealthResponse)
    assert result.status == "healthy"
    assert result.version == "1.0.0"
    client._transport.request.assert_called_once_with("GET", "/health")


def test_sync_context_manager_closes_transport() -> None:
    """Sync client context manager closes transport on exit."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.close = MagicMock()  # type: ignore[method-assign]
    with client:
        pass
    client._transport.close.assert_called_once()


@pytest.mark.asyncio
async def test_async_health_returns_health_response() -> None:
    """Async client health() returns HealthResponse from transport."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = AsyncCognitiveMemoryLayer(config=config)
    client._transport.request = AsyncMock(  # type: ignore[method-assign]
        return_value={"status": "ok", "components": {}}
    )
    result = await client.health()
    assert isinstance(result, HealthResponse)
    assert result.status == "ok"
    client._transport.request.assert_called_once_with("GET", "/health")


@pytest.mark.asyncio
async def test_async_context_manager_closes_transport() -> None:
    """Async client context manager closes transport on exit."""
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = AsyncCognitiveMemoryLayer(config=config)
    client._transport.close = AsyncMock()  # type: ignore[method-assign]
    async with client:
        pass
    client._transport.close.assert_called_once()
