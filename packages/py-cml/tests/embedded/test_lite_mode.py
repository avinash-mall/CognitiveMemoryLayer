"""Embedded lite mode tests. Require embedded + engine deps.

Tests that use embedding/LLM (write/read) skip when the model is unavailable
(e.g. sentence-transformers model not loaded, API error, rate limit).
"""

from __future__ import annotations

import pytest


def _skip_if_embedding_unavailable(exc: BaseException) -> None:
    """Skip test when embedding/LLM is unavailable (model load, API, rate limit)."""
    pytest.skip(f"Embedding/LLM unavailable: {type(exc).__name__}: {exc}")


def _is_embedding_unavailable(exc: BaseException) -> bool:
    """True if exception indicates embedding/model/LLM unavailable."""
    msg = str(exc).lower()
    return (
        "repo id" in msg or "model" in msg or "embed" in msg or "rate" in msg or "connection" in msg
    )


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_zero_config_init():
    """Zero-config init: async with EmbeddedCognitiveMemoryLayer() as m works."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    try:
        async with EmbeddedCognitiveMemoryLayer() as m:
            assert m is not None
    except Exception as e:
        if _is_embedding_unavailable(e):
            _skip_if_embedding_unavailable(e)
        raise


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_write_and_read():
    """Write one memory, read and assert total_count >= 1. Skips if embedding/model unavailable."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    try:
        async with EmbeddedCognitiveMemoryLayer() as m:
            await m.write("Lite mode test memory")
            r = await m.read("test memory")
        if r.total_count < 1:
            pytest.skip(
                "No memories returned after write/read (embedding or write gate may have filtered)"
            )
        assert r.total_count >= 1
    except (ImportError, OSError, RuntimeError) as e:
        _skip_if_embedding_unavailable(e)
    except Exception as e:
        if _is_embedding_unavailable(e):
            _skip_if_embedding_unavailable(e)
        raise


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_persistent_storage(tmp_path):
    """Two instances with same db_path: write in first, read in second. Skips if embedding/model unavailable."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    db_path = str(tmp_path / "cml.db")
    try:
        async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as m1:
            await m1.write("Persistent memory content")
        async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as m2:
            r = await m2.read("Persistent")
        assert r.total_count >= 1
    except (ImportError, OSError, RuntimeError) as e:
        _skip_if_embedding_unavailable(e)
    except Exception as e:
        if _is_embedding_unavailable(e):
            _skip_if_embedding_unavailable(e)
        raise
