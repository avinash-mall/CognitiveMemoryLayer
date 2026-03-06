"""Unit tests for consolidation worker guardrails and lineage references."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.consolidation.clusterer import EpisodeCluster
from src.consolidation.summarizer import ExtractedGist
from src.consolidation.worker import ConsolidationWorker
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance


def _record(text: str, mem_type: MemoryType, confidence: float = 0.8) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id="t",
        context_tags=[],
        type=mem_type,
        text=text,
        confidence=confidence,
        importance=0.7,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=datetime.now(UTC),
        metadata={},
    )


@pytest.mark.asyncio
async def test_guardrails_replace_generic_gist_for_mixed_cluster():
    episodic = MagicMock()
    fact_store = MagicMock()
    neocortical = MagicMock()
    neocortical.facts = fact_store
    worker = ConsolidationWorker(
        episodic_store=episodic, neocortical_store=neocortical, llm_client=None
    )

    ep1 = _record("I never eat shellfish due to allergies.", MemoryType.CONSTRAINT, confidence=0.95)
    ep2 = _record("I went running in the morning.", MemoryType.EPISODIC_EVENT, confidence=0.7)
    cluster = EpisodeCluster(
        cluster_id=0, episodes=[ep1, ep2], common_entities=[], dominant_type="constraint"
    )
    gist = ExtractedGist(
        text="General conversation about various topics",
        gist_type="summary",
        confidence=0.8,
        supporting_episode_ids=[str(ep1.id), str(ep2.id)],
    )

    out = worker._apply_gist_guardrails([gist], [cluster])
    assert len(out) == 1
    assert "various topics" not in out[0].text.lower()
    assert "shellfish" in out[0].text.lower()


@pytest.mark.asyncio
async def test_consolidation_adds_lineage_refs_when_superseding(monkeypatch):
    episodic = MagicMock()
    episodic.deactivate_constraints_by_key = AsyncMock()
    fact_store = MagicMock()
    fact_store.get_facts_by_category = AsyncMock(
        return_value=[
            SimpleNamespace(key="user:goal:oldscope", value="Save 1000", context_tags=["nyc"])
        ]
    )
    fact_store.invalidate_fact = AsyncMock()

    neocortical = MagicMock()
    neocortical.facts = fact_store
    neocortical.store_fact = AsyncMock()

    worker = ConsolidationWorker(
        episodic_store=episodic, neocortical_store=neocortical, llm_client=None
    )

    constraint_episode = _record("I now want to save 2000", MemoryType.CONSTRAINT, confidence=0.9)
    constraint_episode.metadata = {
        "constraints": [
            {
                "constraint_type": "goal",
                "subject": "user",
                "description": "Save 2000",
                "scope": ["new york city"],
                "confidence": 0.9,
            }
        ]
    }

    worker.sampler.sample = AsyncMock(return_value=[constraint_episode])

    async def _always_supersedes(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        "src.extraction.constraint_extractor.ConstraintExtractor.detect_supersession",
        _always_supersedes,
    )

    await worker.consolidate("t", "u")

    assert neocortical.store_fact.await_count >= 1
    evidence_ids = neocortical.store_fact.await_args_list[0].kwargs["evidence_ids"]
    assert any(item.startswith("semantic_key:") for item in evidence_ids)
    assert any(item.startswith("episodic_constraint_key:") for item in evidence_ids)
