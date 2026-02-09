"""Embedded lifecycle tests (context manager, close, ensure_initialized)."""

from __future__ import annotations

import pytest


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_context_manager_enters_exits():
    """async with enters and exits cleanly."""
    try:
        from cml.embedded import EmbeddedCognitiveMemoryLayer
    except ImportError:
        pytest.skip("Embedded not installed")
    try:
        async with EmbeddedCognitiveMemoryLayer() as m:
            assert m._initialized
    except (ImportError, NotImplementedError) as e:
        pytest.skip(f"Embedded engine not available: {e}")


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_close_after_use():
    """Explicit initialize then close."""
    try:
        from cml.embedded import EmbeddedCognitiveMemoryLayer
    except ImportError:
        pytest.skip("Embedded not installed")
    try:
        m = EmbeddedCognitiveMemoryLayer()
        await m.initialize()
        assert m._initialized
        await m.close()
        assert not m._initialized
    except (ImportError, NotImplementedError) as e:
        pytest.skip(f"Embedded engine not available: {e}")


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_ensure_initialized_raises_before_init():
    """write() before initialize() raises RuntimeError."""
    try:
        from cml.embedded import EmbeddedCognitiveMemoryLayer
    except ImportError:
        pytest.skip("Embedded not installed")
    client = EmbeddedCognitiveMemoryLayer()
    with pytest.raises(RuntimeError, match="not initialized"):
        await client.write("x")
