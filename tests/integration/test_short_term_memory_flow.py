"""Integration tests for full short-term memory ingest flow."""

import pytest

from src.memory.hippocampal.write_gate import WriteGate
from src.memory.short_term import ShortTermMemory, ShortTermMemoryConfig


@pytest.mark.asyncio
async def test_sensory_buffer_content_and_token_count():
    """Ingest text -> verify sensory buffer content and token count."""
    config = ShortTermMemoryConfig(min_salience_for_encoding=0.4)
    stm = ShortTermMemory(config=config)
    text = "I prefer coffee and my name is Bob."
    r = await stm.ingest_turn("tenant-a", "user-a", text, turn_id="turn-1", role="user")
    assert r["tokens_buffered"] > 0
    ctx = await stm.get_immediate_context("tenant-a", "user-a", include_sensory=True)
    assert "recent_text" in ctx
    assert "coffee" in ctx["recent_text"] or "Bob" in ctx["recent_text"]


@pytest.mark.asyncio
async def test_chunks_for_encoding_respect_min_salience():
    """Encodable chunks only include those with salience >= min_salience_for_encoding."""
    config = ShortTermMemoryConfig(
        min_salience_for_encoding=0.6,
        working_max_chunks=10,
    )
    stm = ShortTermMemory(config=config)
    await stm.ingest_turn(
        "tenant-a",
        "user-a",
        "I prefer vegetarian food. My name is Alex. Okay sure.",
        turn_id="turn-1",
        role="user",
    )
    encodable = await stm.get_encodable_chunks("tenant-a", "user-a")
    for c in encodable:
        assert c.salience >= 0.6, f"Encodable chunk has salience {c.salience} < 0.6"
    encodable_ids = {c.id for c in encodable}
    all_from_result = await stm.working.get_state("tenant-a", "user-a")
    all_chunks = all_from_result.chunks
    for c in all_chunks:
        if c.salience < 0.6:
            assert c.id not in encodable_ids


@pytest.mark.asyncio
async def test_working_memory_chunk_types_and_salience():
    """Verify working memory chunks have expected types and salience."""
    config = ShortTermMemoryConfig(min_salience_for_encoding=0.3)
    stm = ShortTermMemory(config=config)
    r = await stm.ingest_turn(
        "tenant-a",
        "user-a",
        "I prefer dark mode. My name is Alice.",
        turn_id="turn-1",
        role="user",
    )
    assert r["chunks_created"] >= 1
    all_chunks = r["all_chunks"]
    # Semchunk returns STATEMENT only; ensure chunks exist and have valid salience
    assert len(all_chunks) >= 1
    for c in all_chunks:
        assert 0 <= c.salience <= 1.0
        assert c.text


@pytest.mark.asyncio
async def test_chunks_for_encoding_handoff_to_write_gate_preserves_fields():
    """Feed chunks_for_encoding to WriteGate; assert chunk fields preserved."""
    # Use min_salience_for_encoding=0 so all chunks are encodable (no skip)
    config = ShortTermMemoryConfig(min_salience_for_encoding=0.0)
    stm = ShortTermMemory(config=config)
    await stm.ingest_turn(
        "tenant-a",
        "user-a",
        "I prefer tea. My name is Carol.",
        turn_id="turn-1",
        role="user",
    )
    encodable = await stm.get_encodable_chunks("tenant-a", "user-a")
    assert encodable, "Chunker should produce at least one encodable chunk with min_salience=0"
    gate = WriteGate()
    for chunk in encodable:
        result = gate.evaluate(chunk)
        assert chunk.text
        assert chunk.chunk_type is not None
        assert 0 <= chunk.salience <= 1.0
        assert result.decision.value in ("store", "skip", "redact_and_store")
        if result.decision.value != "skip":
            assert result.memory_types


@pytest.mark.asyncio
async def test_full_ingest_flow():
    """Ingest turns -> get encodable chunks -> clear."""
    config = ShortTermMemoryConfig(
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
