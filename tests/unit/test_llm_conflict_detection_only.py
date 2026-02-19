"""Unit tests for use_llm_conflict_detection_only feature flag."""

import pytest

from src.core.schemas import MemoryRecord, Provenance
from src.reconsolidation.conflict_detector import ConflictDetector
from src.utils.llm import LLMClient


@pytest.fixture
def sample_memory_record():
    """Sample memory record for conflict detection."""
    from datetime import datetime

    from src.core.enums import MemorySource, MemoryStatus, MemoryType

    return MemoryRecord(
        id="00000000-0000-0000-0000-000000000001",
        tenant_id="test",
        context_tags=[],
        type=MemoryType.EPISODIC_EVENT,
        text="I prefer coffee in the morning",
        confidence=0.8,
        importance=0.7,
        timestamp=datetime.now(),
        written_at=datetime.now(),
        access_count=0,
        status=MemoryStatus.ACTIVE,
        provenance=Provenance(source=MemorySource.AGENT_INFERRED),
    )


@pytest.mark.asyncio
async def test_use_llm_conflict_detection_only_skips_fast_path(monkeypatch, sample_memory_record):
    """When use_llm_conflict_detection_only=true, ConflictDetector skips fast path."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.complete_json = AsyncMock(
        return_value={
            "conflict_type": "TEMPORAL_CHANGE",
            "confidence": 0.9,
            "reasoning": "Preference changed",
            "is_superseding": True,
        }
    )
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: type(
            "S",
            (),
            {"features": type("F", (), {"use_llm_conflict_detection_only": True})()},
        )(),
    )
    detector = ConflictDetector(llm_client=mock_llm)
    await detector.detect(sample_memory_record, "Actually I prefer tea now")
    # Should have used LLM
    mock_llm.complete_json.assert_called()
