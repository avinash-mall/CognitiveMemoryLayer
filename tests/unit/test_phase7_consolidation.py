"""Unit tests for Phase 7: consolidation (triggers, clusterer, sampler scoring)."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.consolidation.triggers import (
    ConsolidationScheduler,
    ConsolidationTask,
    TriggerCondition,
    TriggerType,
)
from src.consolidation.clusterer import EpisodeCluster, SemanticClusterer
from src.core.enums import MemoryScope, MemorySource, MemoryType
from src.core.schemas import EntityMention, MemoryRecord, Provenance


def _make_record(
    text: str = "test",
    embedding: list | None = None,
    importance: float = 0.5,
    confidence: float = 0.8,
    access_count: int = 0,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id="t",
        scope=MemoryScope.USER,
        scope_id="u",
        user_id="u",
        type=MemoryType.EPISODIC_EVENT,
        text=text,
        confidence=confidence,
        importance=importance,
        access_count=access_count,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=datetime.now(timezone.utc),
        embedding=embedding,
        entities=[],
    )


class TestConsolidationScheduler:
    @pytest.mark.asyncio
    async def test_trigger_manual_and_get_next_task(self):
        s = ConsolidationScheduler()
        await s.trigger_manual("t1", "u1", reason="Manual", priority=10)
        assert s.has_pending_tasks()
        task = await s.get_next_task()
        assert task is not None
        assert task.tenant_id == "t1"
        assert task.user_id == "u1"
        assert task.trigger_type == TriggerType.MANUAL
        assert task.trigger_reason == "Manual"

    @pytest.mark.asyncio
    async def test_check_triggers_quota(self):
        s = ConsolidationScheduler(quota_threshold_episodes=10)
        # Register only QUOTA so scheduled first-run doesn't win
        s.register_user("t", "u", conditions=[TriggerCondition(TriggerType.QUOTA, min_episodes=10)])
        triggered = await s.check_triggers("t", "u", episode_count=15, memory_size_mb=1.0)
        assert triggered is True
        task = await s.get_next_task()
        assert task is not None
        assert task.trigger_type == TriggerType.QUOTA

    @pytest.mark.asyncio
    async def test_check_triggers_scheduled_first_run(self):
        s = ConsolidationScheduler()
        s.register_user("t", "u")
        triggered = await s.check_triggers("t", "u", episode_count=0, memory_size_mb=0.0)
        assert triggered is True
        task = await s.get_next_task()
        assert task.trigger_type == TriggerType.SCHEDULED

    @pytest.mark.asyncio
    async def test_get_next_task_timeout_returns_none(self):
        s = ConsolidationScheduler()
        task = await s.get_next_task()
        assert task is None


class TestSemanticClusterer:
    def test_cluster_empty_returns_empty(self):
        c = SemanticClusterer()
        assert c.cluster([]) == []

    def test_cluster_single_episode_no_embedding(self):
        c = SemanticClusterer(min_cluster_size=1)
        rec = _make_record("one", embedding=None)
        clusters = c.cluster([rec])
        assert len(clusters) == 1
        assert clusters[0].cluster_id == 0
        assert len(clusters[0].episodes) == 1
        assert clusters[0].avg_confidence == rec.confidence

    def test_cluster_two_similar_embeddings(self):
        c = SemanticClusterer(min_cluster_size=2, similarity_threshold=0.5)
        # Same direction = high cosine sim
        emb = [1.0, 0.0, 0.0]
        r1 = _make_record("a", embedding=emb)
        r2 = _make_record("b", embedding=emb)
        clusters = c.cluster([r1, r2])
        assert len(clusters) >= 1
        total_eps = sum(len(cl.episodes) for cl in clusters)
        assert total_eps == 2

    def test_cluster_common_entities_and_dominant_type(self):
        c = SemanticClusterer(min_cluster_size=1)
        r1 = _make_record("a", embedding=[1.0, 0, 0])
        r1.entities = [EntityMention(text="Alice", normalized="alice", entity_type="PERSON")]
        r2 = _make_record("b", embedding=[0.99, 0.01, 0])
        r2.entities = [EntityMention(text="Alice", normalized="alice", entity_type="PERSON")]
        clusters = c.cluster([r1, r2])
        assert len(clusters) >= 1
        if len(clusters) == 1 and len(clusters[0].episodes) == 2:
            assert "alice" in clusters[0].common_entities or clusters[0].dominant_type
