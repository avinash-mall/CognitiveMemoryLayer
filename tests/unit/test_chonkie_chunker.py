"""Unit tests for Chonkie semantic chunking adapter and working memory integration."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.memory.working.chunker import (
    ChonkieChunkerAdapter,
    ChonkieUnavailableError,
    _get_chonkie_semantic_chunker,
)
from src.memory.working.manager import WorkingMemoryManager
from src.memory.working.models import ChunkType, SemanticChunk


def _fake_chonkie_chunk(text: str, start: int = 0, end: int | None = None):
    """Minimal Chonkie-like chunk with .text, .start_index, .end_index, .token_count."""
    c = MagicMock()
    c.text = text
    c.start_index = start
    c.end_index = end if end is not None else start + len(text)
    c.token_count = len(text.split())
    return c


class TestChonkieUnavailableError:
    """ChonkieUnavailableError when chonkie[semantic] is not installed."""

    def test_get_chonkie_raises_when_import_fails(self):
        import builtins

        real_import = builtins.__import__

        def fail_chonkie_only(name, *args, **kwargs):
            if name == "chonkie":
                raise ImportError("No module named 'chonkie'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fail_chonkie_only),
            pytest.raises(ChonkieUnavailableError) as exc_info,
        ):
            _get_chonkie_semantic_chunker()
        assert "chonkie" in str(exc_info.value).lower() or "install" in str(exc_info.value).lower()


class TestChonkieChunkerAdapter:
    """ChonkieChunkerAdapter maps Chonkie output to SemanticChunk (no LLM/rule)."""

    def test_chunk_empty_text_returns_empty_list(self):
        with patch("src.memory.working.chunker._get_chonkie_semantic_chunker") as get_chonkie:
            get_chonkie.return_value.chunk.return_value = []
            adapter = ChonkieChunkerAdapter()
            result = adapter.chunk("")
            assert result == []

    def test_chunk_maps_chonkie_output_to_semantic_chunks(self):
        raw = [
            _fake_chonkie_chunk("First segment.", 0, 14),
            _fake_chonkie_chunk("Second segment.", 15, 30),
        ]
        with patch("src.memory.working.chunker._get_chonkie_semantic_chunker") as get_chonkie:
            mock_chonkie = MagicMock()
            mock_chonkie.chunk.return_value = raw
            get_chonkie.return_value = mock_chonkie
            adapter = ChonkieChunkerAdapter()
            result = adapter.chunk(
                "First segment. Second segment.",
                turn_id="t1",
                role="user",
                timestamp=datetime.now(datetime.UTC),
            )
        assert len(result) == 2
        for i, c in enumerate(result):
            assert isinstance(c, SemanticChunk)
            assert c.chunk_type == ChunkType.STATEMENT
            assert c.salience == 0.5
            assert c.confidence == 0.8
            assert c.source_turn_id == "t1"
            assert c.source_role == "user"
        assert result[0].text == "First segment."
        assert result[1].text == "Second segment."
        assert len(result[0].id) == 16
        assert result[0].id != result[1].id

    def test_chunk_fallback_single_chunk_when_chonkie_returns_empty(self):
        with patch("src.memory.working.chunker._get_chonkie_semantic_chunker") as get_chonkie:
            get_chonkie.return_value.chunk.return_value = []
            adapter = ChonkieChunkerAdapter()
            result = adapter.chunk("Some text.", turn_id="t1", role="user")
        assert len(result) == 1
        assert result[0].text == "Some text."
        assert result[0].chunk_type == ChunkType.STATEMENT


class TestWorkingMemoryManagerChonkiePath:
    """WorkingMemoryManager uses Chonkie when use_chonkie_for_large_text and threshold met."""

    @pytest.mark.asyncio
    async def test_process_input_uses_chonkie_when_enabled_and_above_threshold(self):
        fake_chunks = [
            SemanticChunk(
                id="c1",
                text="Part one.",
                chunk_type=ChunkType.STATEMENT,
                salience=0.5,
                confidence=0.8,
                timestamp=datetime.now(datetime.UTC),
            ),
        ]
        with patch("src.memory.working.manager.ChonkieChunkerAdapter") as adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.chunk.return_value = fake_chunks
            adapter_class.return_value = mock_adapter
            manager = WorkingMemoryManager(
                use_fast_chunker=True,
                use_chonkie_for_large_text=True,
                large_text_threshold_chars=0,
            )
            result = await manager.process_input(
                "tenant", "scope", "Hello world.", turn_id="t1", role="user"
            )
        adapter_class.assert_called_once()
        mock_adapter.chunk.assert_called_once()
        call_kw = mock_adapter.chunk.call_args[1]
        assert call_kw.get("turn_id") == "t1"
        assert call_kw.get("role") == "user"
        assert result == fake_chunks

    @pytest.mark.asyncio
    async def test_process_input_uses_default_chunker_when_chonkie_disabled(self):
        manager = WorkingMemoryManager(
            use_fast_chunker=True,
            use_chonkie_for_large_text=False,
            large_text_threshold_chars=0,
        )
        result = await manager.process_input(
            "tenant", "scope", "I prefer tea.", turn_id="t1", role="user"
        )
        assert len(result) >= 1
        assert any(c.chunk_type == ChunkType.PREFERENCE for c in result)

    @pytest.mark.asyncio
    async def test_process_input_falls_back_when_chonkie_unavailable(self):
        with patch(
            "src.memory.working.manager.ChonkieChunkerAdapter",
            side_effect=ChonkieUnavailableError("not installed"),
        ):
            manager = WorkingMemoryManager(
                use_fast_chunker=True,
                use_chonkie_for_large_text=True,
                large_text_threshold_chars=0,
            )
            result = await manager.process_input(
                "tenant", "scope", "Fallback text.", turn_id="t1", role="user"
            )
        assert len(result) >= 1
        assert any("Fallback" in c.text for c in result)
