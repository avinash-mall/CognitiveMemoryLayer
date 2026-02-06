"""Integration test: full short-term ingest flow."""

import pytest

from src.memory.short_term import ShortTermMemory, ShortTermMemoryConfig


@pytest.mark.asyncio
async def test_full_ingest_flow():
    """Ingest turns -> get encodable chunks -> clear."""
    config = ShortTermMemoryConfig(
        use_fast_chunker=True,
        min_salience_for_encoding=0.4,
        working_max_chunks=10,
    )
    stm = ShortTermMemory(config=config)

    # First turn: preference + fact
    r1 = await stm.ingest_turn(
        "tenant-a",
        "user-a",
        "I prefer vegetarian food. My name is Alex.",
        turn_id="turn-1",
        role="user",
    )
    assert r1["tokens_buffered"] > 0
    assert r1["chunks_created"] >= 1
    encodable = await stm.get_encodable_chunks("tenant-a", "user-a")
    assert len(encodable) >= 1
    high_salience = [c for c in encodable if c.salience >= 0.5]
    assert len(high_salience) >= 1

    # Second turn
    r2 = await stm.ingest_turn(
        "tenant-a",
        "user-a",
        "What's the weather?",
        turn_id="turn-2",
        role="user",
    )
    assert r2["chunks_created"] >= 1

    # Context includes both
    ctx = await stm.get_immediate_context(
        "tenant-a",
        "user-a",
        include_sensory=True,
        max_working_chunks=10,
    )
    assert ctx["working_memory"]
    assert "vegetarian" in ctx["recent_text"] or "Alex" in ctx["recent_text"]

    # Stats
    stats = await stm.working.get_stats("tenant-a", "user-a")
    assert stats["chunk_count"] >= 2
    assert stats["turn_count"] == 2

    # Clear
    await stm.clear("tenant-a", "user-a")
    encodable_after = await stm.get_encodable_chunks("tenant-a", "user-a")
    assert len(encodable_after) == 0
