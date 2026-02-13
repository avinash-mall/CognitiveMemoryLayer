"""Embedded lifecycle tests (context manager, close, ensure_initialized)."""

from __future__ import annotations

import pytest


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_context_manager_enters_exits():
    """async with enters and exits cleanly."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    async with EmbeddedCognitiveMemoryLayer() as m:
        assert m._initialized


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_close_after_use():
    """Explicit initialize then close."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    m = EmbeddedCognitiveMemoryLayer()
    await m.initialize()
    assert m._initialized
    await m.close()
    assert not m._initialized


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_ensure_initialized_raises_before_init():
    """write() before initialize() raises RuntimeError."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    client = EmbeddedCognitiveMemoryLayer()
    with pytest.raises(RuntimeError, match="not initialized"):
        await client.write("x")
