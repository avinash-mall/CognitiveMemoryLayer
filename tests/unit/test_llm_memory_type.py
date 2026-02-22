"""Unit tests for use_llm_memory_type feature (LLM-set memory type in Unified Extractor)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.extraction.unified_write_extractor import UnifiedWritePathExtractor
from src.memory.working.models import ChunkType, SemanticChunk


def _chunk(text: str) -> SemanticChunk:
    return SemanticChunk(
        id=str(uuid4()),
        text=text,
        chunk_type=ChunkType.STATEMENT,
        salience=0.5,
        timestamp=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_unified_extractor_returns_memory_type():
    """Unified extractor returns memory_type when LLM outputs it."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [],
            "facts": [],
            "salience": 0.6,
            "importance": 0.7,
            "memory_type": "preference",
        }
    )
    extractor = UnifiedWritePathExtractor(mock_llm)
    chunk = _chunk("I prefer dark mode")
    result = await extractor.extract(chunk)
    assert result.memory_type == "preference"


@pytest.mark.asyncio
async def test_unified_extractor_handles_invalid_memory_type():
    """Invalid or unknown memory_type is set to None (fallback to gate/constraint)."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [],
            "facts": [],
            "salience": 0.5,
            "importance": 0.5,
            "memory_type": "invalid_type",
        }
    )
    extractor = UnifiedWritePathExtractor(mock_llm)
    chunk = _chunk("Some text")
    result = await extractor.extract(chunk)
    assert result.memory_type is None


@pytest.mark.asyncio
async def test_unified_extractor_handles_missing_memory_type():
    """Missing memory_type in LLM response yields None."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [],
            "facts": [],
            "salience": 0.5,
            "importance": 0.5,
        }
    )
    extractor = UnifiedWritePathExtractor(mock_llm)
    chunk = _chunk("Some text")
    result = await extractor.extract(chunk)
    assert result.memory_type is None


def test_parse_result_validates_memory_type():
    """_parse_result validates memory_type against allowed values."""
    extractor = UnifiedWritePathExtractor(None)
    chunk = _chunk("I like Python")
    result = extractor._parse_result(
        {
            "entities": [],
            "relations": [],
            "constraints": [],
            "facts": [],
            "salience": 0.6,
            "importance": 0.6,
            "memory_type": "semantic_fact",
        },
        chunk,
    )
    assert result.memory_type == "semantic_fact"


def test_parse_result_rejects_invalid_memory_type():
    """_parse_result rejects invalid memory_type."""
    extractor = UnifiedWritePathExtractor(None)
    chunk = _chunk("Test")
    result = extractor._parse_result(
        {
            "entities": [],
            "relations": [],
            "constraints": [],
            "facts": [],
            "salience": 0.5,
            "importance": 0.5,
            "memory_type": "not_a_valid_type",
        },
        chunk,
    )
    assert result.memory_type is None
