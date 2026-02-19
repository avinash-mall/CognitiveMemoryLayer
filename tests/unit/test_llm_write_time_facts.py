"""Unit tests for LLM write-time facts (unified path and feature flag)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.extraction.unified_write_extractor import UnifiedWritePathExtractor
from src.extraction.write_time_facts import WriteTimeFactExtractor
from src.memory.neocortical.schemas import FactCategory
from src.memory.working.models import ChunkType, SemanticChunk


def _chunk(text: str, chunk_type: ChunkType = ChunkType.PREFERENCE) -> SemanticChunk:
    return SemanticChunk(
        id=str(uuid4()),
        text=text,
        chunk_type=chunk_type,
        salience=0.7,
        timestamp=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_unified_extractor_returns_facts_with_mock_llm():
    """Unified extractor returns valid facts from LLM response."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [],
            "facts": [
                {
                    "key": "user:preference:cuisine",
                    "category": "preference",
                    "predicate": "cuisine",
                    "value": "vegetarian",
                    "confidence": 0.8,
                }
            ],
            "salience": 0.7,
            "importance": 0.6,
        }
    )
    extractor = UnifiedWritePathExtractor(mock_llm)
    chunk = _chunk("I prefer vegetarian food")
    result = await extractor.extract(chunk)
    assert len(result.facts) == 1
    assert result.facts[0].key == "user:preference:cuisine"
    assert result.facts[0].category == FactCategory.PREFERENCE
    assert result.facts[0].value == "vegetarian"


def test_rule_based_write_time_fact_extractor_still_works():
    """Rule-based WriteTimeFactExtractor produces valid facts (flag=false path)."""
    extractor = WriteTimeFactExtractor()
    chunk = _chunk("I prefer Italian food")
    facts = extractor.extract(chunk)
    assert len(facts) >= 1
    assert any(f.category == FactCategory.PREFERENCE for f in facts)
