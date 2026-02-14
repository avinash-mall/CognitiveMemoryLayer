"""Embedded lifecycle tests (context manager, close, ensure_initialized)."""

from __future__ import annotations

import pytest


def _skip_if_embedding_unavailable(exc: BaseException) -> None:
    """Skip when embedding model cannot be loaded (e.g. HFValidationError, network)."""
    pytest.skip(f"Embedding unavailable: {type(exc).__name__}: {exc}")


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_context_manager_enters_exits():
    """async with enters and exits cleanly."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    try:
        async with EmbeddedCognitiveMemoryLayer() as m:
            assert m._initialized
    except Exception as e:
        if "repo id" in str(e).lower() or "embed" in str(e).lower() or "model" in str(e).lower():
            _skip_if_embedding_unavailable(e)
        raise


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_close_after_use():
    """Explicit initialize then close."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    m = EmbeddedCognitiveMemoryLayer()
    try:
        await m.initialize()
        assert m._initialized
        await m.close()
        assert not m._initialized
    except Exception as e:
        if "repo id" in str(e).lower() or "embed" in str(e).lower() or "model" in str(e).lower():
            _skip_if_embedding_unavailable(e)
        raise


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_ensure_initialized_raises_before_init():
    """write() before initialize() raises RuntimeError."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    client = EmbeddedCognitiveMemoryLayer()
    with pytest.raises(RuntimeError, match="not initialized"):
        await client.write("x")
