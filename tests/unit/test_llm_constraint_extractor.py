"""Unit tests for LLM constraint extractor (unified path and feature flag)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.extraction.constraint_extractor import ConstraintExtractor, ConstraintObject
from src.extraction.unified_write_extractor import UnifiedWritePathExtractor
from src.memory.working.models import ChunkType, SemanticChunk


def _chunk(text: str, chunk_type: ChunkType = ChunkType.STATEMENT) -> SemanticChunk:
    return SemanticChunk(
        id=str(uuid4()),
        text=text,
        chunk_type=chunk_type,
        salience=0.7,
        timestamp=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_unified_extractor_returns_constraints_with_mock_llm():
    """Unified extractor returns valid constraints from LLM response."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [
                {
                    "constraint_type": "goal",
                    "subject": "user",
                    "description": "I want to eat healthier",
                    "scope": [],
                    "confidence": 0.85,
                }
            ],
            "facts": [],
            "salience": 0.8,
            "importance": 0.7,
        }
    )
    extractor = UnifiedWritePathExtractor(mock_llm)
    chunk = _chunk("I want to eat healthier")
    result = await extractor.extract(chunk)
    assert len(result.constraints) == 1
    assert result.constraints[0].constraint_type == "goal"
    assert result.constraints[0].subject == "user"
    assert result.constraints[0].description == "I want to eat healthier"
    assert result.constraints[0].confidence == 0.85


@pytest.mark.asyncio
async def test_unified_extractor_empty_text_returns_defaults():
    """Unified extractor returns defaults for empty text."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock()
    extractor = UnifiedWritePathExtractor(mock_llm)
    chunk = _chunk("")
    result = await extractor.extract(chunk)
    assert result.constraints == []
    # salience falls back to chunk.salience (0.7) or 0.5 when empty
    assert 0.5 <= result.salience <= 1.0
    assert result.importance == 0.5


def test_rule_based_constraint_extractor_still_works():
    """Rule-based ConstraintExtractor produces valid constraints (flag=false path)."""
    extractor = ConstraintExtractor()
    chunk = _chunk("I'm trying to save money for a vacation")
    constraints = extractor.extract(chunk)
    assert len(constraints) >= 1
    assert any(c.constraint_type == "goal" for c in constraints)


def test_constraint_fact_key_preserved():
    """ConstraintExtractor.constraint_fact_key is used for deduplication."""
    c = ConstraintObject(
        constraint_type="goal",
        subject="user",
        description="test",
        scope=[],
        confidence=0.8,
    )
    key = ConstraintExtractor.constraint_fact_key(c)
    assert key.startswith("user:goal:")
    assert len(key) > 10
