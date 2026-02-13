"""Unit tests for Phase 2: sensory buffer and working memory."""

import pytest

from src.memory.sensory.buffer import (
    SensoryBuffer,
    SensoryBufferConfig,
)
from src.memory.sensory.manager import SensoryBufferManager
from src.memory.short_term import ShortTermMemory, ShortTermMemoryConfig
from src.memory.working.chunker import RuleBasedChunker
from src.memory.working.models import ChunkType, SemanticChunk, WorkingMemoryState


class TestSensoryBuffer:
    """Sensory buffer operations."""

    @pytest.mark.asyncio
    async def test_ingest_returns_token_count(self):
        config = SensoryBufferConfig(max_tokens=100, decay_seconds=60.0)
        buf = SensoryBuffer(config)
        n = await buf.ingest("hello world", turn_id="t1", role="user")
        assert n == 2
        assert buf.size == 2

    @pytest.mark.asyncio
    async def test_get_text(self):
        buf = SensoryBuffer(SensoryBufferConfig(max_tokens=100))
        await buf.ingest("one two three")
        assert await buf.get_text() == "one two three"
        assert await buf.get_text(max_tokens=2) == "one two"

    @pytest.mark.asyncio
    async def test_clear(self):
        buf = SensoryBuffer()
        await buf.ingest("foo bar")
        await buf.clear()
        assert buf.is_empty
        assert await buf.get_text() == ""

    @pytest.mark.asyncio
    async def test_capacity_enforced(self):
        config = SensoryBufferConfig(max_tokens=3, decay_seconds=60.0)
        buf = SensoryBuffer(config)
        await buf.ingest("a b c d e")
        assert buf.size == 3
        text = await buf.get_text()
        assert len(text.split()) == 3


class TestSensoryBufferManager:
    """Per-user sensory buffer manager."""

    @pytest.mark.asyncio
    async def test_ingest_and_get_recent_text(self):
        manager = SensoryBufferManager()
        await manager.ingest("t1", "u1", "hello world", role="user")
        text = await manager.get_recent_text("t1", "u1")
        assert "hello" in text and "world" in text

    @pytest.mark.asyncio
    async def test_clear_user(self):
        manager = SensoryBufferManager()
        await manager.ingest("t1", "u1", "foo")
        await manager.clear_user("t1", "u1")
        text = await manager.get_recent_text("t1", "u1")
        assert text == ""


class TestRuleBasedChunker:
    """Rule-based chunker."""

    def test_preference_markers(self):
        chunker = RuleBasedChunker()
        chunks = chunker.chunk("I prefer dark mode", role="user")
        assert len(chunks) >= 1
        pref = [c for c in chunks if c.chunk_type == ChunkType.PREFERENCE]
        assert len(pref) >= 1
        assert pref[0].salience >= 0.7

    def test_fact_markers(self):
        chunker = RuleBasedChunker()
        chunks = chunker.chunk("My name is Alice.")
        assert any(c.chunk_type == ChunkType.FACT for c in chunks)

    def test_question(self):
        chunker = RuleBasedChunker()
        chunks = chunker.chunk("What time is it?")
        assert any(c.chunk_type == ChunkType.QUESTION for c in chunks)


class TestWorkingMemoryState:
    """WorkingMemoryState capacity and salience."""

    def test_add_chunk_evicts_low_salience(self):
        state = WorkingMemoryState("t1", "u1", max_chunks=2)
        state.add_chunk(
            SemanticChunk(id="1", text="a", chunk_type=ChunkType.STATEMENT, salience=0.9)
        )
        state.add_chunk(
            SemanticChunk(id="2", text="b", chunk_type=ChunkType.STATEMENT, salience=0.3)
        )
        state.add_chunk(
            SemanticChunk(id="3", text="c", chunk_type=ChunkType.STATEMENT, salience=0.8)
        )
        assert len(state.chunks) == 2
        ids = [c.id for c in state.chunks]
        assert "2" not in ids  # lowest salience evicted
        assert "1" in ids
        assert "3" in ids


class TestShortTermMemory:
    """Short-term memory facade."""

    @pytest.mark.asyncio
    async def test_ingest_turn_returns_chunks(self):
        config = ShortTermMemoryConfig(use_fast_chunker=True)
        stm = ShortTermMemory(config=config)
        result = await stm.ingest_turn(
            "t1", "u1", "I prefer coffee and my name is Bob.", turn_id="turn-1", role="user"
        )
        assert result["tokens_buffered"] > 0
        assert result["chunks_created"] >= 1
        assert "all_chunks" in result
        assert isinstance(result["chunks_for_encoding"], list)

    @pytest.mark.asyncio
    async def test_get_immediate_context(self):
        config = ShortTermMemoryConfig(use_fast_chunker=True)
        stm = ShortTermMemory(config=config)
        await stm.ingest_turn("t1", "u1", "Hello world.")
        ctx = await stm.get_immediate_context("t1", "u1", include_sensory=True)
        assert "working_memory" in ctx
        assert "recent_text" in ctx

    @pytest.mark.asyncio
    async def test_clear(self):
        config = ShortTermMemoryConfig(use_fast_chunker=True)
        stm = ShortTermMemory(config=config)
        await stm.ingest_turn("t1", "u1", "Some text.")
        await stm.clear("t1", "u1")
        ctx = await stm.get_immediate_context("t1", "u1")
        assert ctx["working_memory"] == ""
        assert ctx["recent_text"] == ""
