"""Unit tests for use_llm_write_gate_importance feature flag."""

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
async def test_unified_extractor_returns_importance():
    """Unified extractor returns importance for WriteGate override."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [],
            "facts": [],
            "salience": 0.6,
            "importance": 0.95,
        }
    )
    extractor = UnifiedWritePathExtractor(mock_llm)
    chunk = _chunk("I must never share my password")
    result = await extractor.extract(chunk)
    assert result.importance == 0.95
