"""Embedded lite mode tests."""

from __future__ import annotations

import pytest


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_zero_config_init():
    """Zero-config init: async with EmbeddedCognitiveMemoryLayer() as m works."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    async with EmbeddedCognitiveMemoryLayer() as m:
        assert m is not None


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_lite_mode_write_and_read_roundtrip():
    """Write one memory, then read it back."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    async with EmbeddedCognitiveMemoryLayer() as m:
        await m.write("Lite mode test memory")
        r = await m.read("Lite mode test memory")
    assert r.total_count >= 1


@pytest.mark.embedded
@pytest.mark.asyncio
async def test_persistent_storage(tmp_path):
    """Two instances with same db_path: write in first, read in second."""
    # Use tempdir (TMPDIR or /tmp); sqlite fails in some Docker tmp_path locations
    import tempfile
    from pathlib import Path

    from cml.embedded import EmbeddedCognitiveMemoryLayer

    base = Path(tempfile.mkdtemp(prefix="cml_embed_"))
    db_path = str(base / "cml.db")
    async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as m1:
        await m1.write("Persistent memory content for sqlite roundtrip")
    async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as m2:
        r = await m2.read("Persistent memory content for sqlite roundtrip")
    assert r.total_count >= 1
