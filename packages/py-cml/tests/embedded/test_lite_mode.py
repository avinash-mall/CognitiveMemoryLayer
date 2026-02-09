"""Embedded lite mode tests. Require embedded + engine deps."""

from __future__ import annotations

import pytest


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_zero_config_init():
    """Zero-config init: async with EmbeddedCognitiveMemoryLayer() as m works."""
    try:
        from cml.embedded import EmbeddedCognitiveMemoryLayer
    except ImportError:
        pytest.skip("Embedded not installed")
    try:
        async with EmbeddedCognitiveMemoryLayer() as m:
            assert m is not None
    except (ImportError, NotImplementedError) as e:
        pytest.skip(f"Embedded engine not available: {e}")


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_write_and_read():
    """Write one memory, read and assert total_count >= 1."""
    try:
        from cml.embedded import EmbeddedCognitiveMemoryLayer
    except ImportError:
        pytest.skip("Embedded not installed")
    try:
        async with EmbeddedCognitiveMemoryLayer() as m:
            await m.write("Lite mode test memory")
            r = await m.read("test memory")
            assert r.total_count >= 1
    except (ImportError, NotImplementedError) as e:
        pytest.skip(f"Embedded engine not available: {e}")


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_persistent_storage(tmp_path):
    """Two instances with same db_path: write in first, read in second."""
    try:
        from cml.embedded import EmbeddedCognitiveMemoryLayer
    except ImportError:
        pytest.skip("Embedded not installed")
    db_path = str(tmp_path / "cml.db")
    try:
        async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as m1:
            await m1.write("Persistent memory content")
        async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as m2:
            r = await m2.read("Persistent")
            assert r.total_count >= 1
    except (ImportError, NotImplementedError) as e:
        pytest.skip(f"Embedded engine not available: {e}")
