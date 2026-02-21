"""Unit tests for SemchunkChunker (semchunk + Hugging Face tokenizer)."""

from datetime import UTC, datetime

from src.memory.working.chunker import SemchunkChunker
from src.memory.working.models import ChunkType, SemanticChunk


class TestSemchunkChunker:
    """SemchunkChunker maps semchunk output to SemanticChunk."""

    def test_chunk_empty_text_returns_empty_list(self):
        chunker = SemchunkChunker(
            tokenizer_id="google/flan-t5-base",
            chunk_size=500,
            overlap_percent=0.15,
        )
        result = chunker.chunk("")
        assert result == []

    def test_chunk_maps_strings_to_semantic_chunks(self):
        chunker = SemchunkChunker(
            tokenizer_id="google/flan-t5-base",
            chunk_size=512,
            overlap_percent=0.15,
        )
        result = chunker.chunk(
            "Hello world. This is a test.",
            turn_id="t1",
            role="user",
            timestamp=datetime.now(UTC),
        )
        assert len(result) >= 1
        for c in result:
            assert isinstance(c, SemanticChunk)
            assert c.chunk_type == ChunkType.STATEMENT
            assert c.salience == 0.5
            assert c.confidence == 0.8
            assert c.source_turn_id == "t1"
            assert c.source_role == "user"
        texts = [c.text for c in result]
        assert "Hello" in " ".join(texts) or "world" in " ".join(texts)

    def test_chunk_single_word_fallback(self):
        chunker = SemchunkChunker(
            tokenizer_id="google/flan-t5-base",
            chunk_size=500,
            overlap_percent=0.15,
        )
        result = chunker.chunk("Hi.", turn_id="t1", role="user")
        assert len(result) >= 1
        assert result[0].text
        assert result[0].chunk_type == ChunkType.STATEMENT
