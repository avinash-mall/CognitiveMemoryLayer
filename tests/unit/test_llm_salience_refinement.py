"""Unit tests for use_llm_salience_refinement feature flag."""

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
async def test_unified_extractor_returns_salience():
    """Unified extractor returns salience from LLM."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [],
            "facts": [],
            "salience": 0.9,
            "importance": 0.7,
        }
    )
    extractor = UnifiedWritePathExtractor(mock_llm)
    chunk = _chunk("This is very important to me")
    result = await extractor.extract(chunk)
    assert result.salience == 0.9
