"""Unit tests for consolidation migrator and episode sampler."""

from datetime import datetime, timezone
from uuid import uuid4

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.consolidation.migrator import ConsolidationMigrator, MigrationResult
from src.consolidation.sampler import EpisodeSampler, SamplingConfig
from src.consolidation.schema_aligner import AlignmentResult
from src.consolidation.summarizer import ExtractedGist
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance


def _make_memory_record(
    text: str = "episode",
    importance: float = 0.5,
    confidence: float = 0.8,
    access_count: int = 0,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id="t1",
        context_tags=[],
        type=MemoryType.EPISODIC_EVENT,
        text=text,
        confidence=confidence,
        importance=importance,
        access_count=access_count,
        timestamp=datetime.now(timezone.utc),
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
    )


def _make_gist(
    text: str = "User prefers X",
    supporting_ids: list | None = None,
    key: str | None = "user:preference:food",
) -> ExtractedGist:
    return ExtractedGist(
        text=text,
        gist_type="preference",
        confidence=0.9,
        supporting_episode_ids=supporting_ids or [],
        key=key,
        subject="user",
        predicate="food",
        value="vegetarian",
    )


class TestConsolidationMigrator:
    """ConsolidationMigrator with mocked stores."""

    @pytest.mark.asyncio
    async def test_migrate_returns_migration_result(self):
        mock_semantic = MagicMock()
        mock_semantic.store_fact = AsyncMock()
        mock_episodic = MagicMock()
        mock_episodic.get_by_id = AsyncMock(return_value=None)
        mock_episodic.update = AsyncMock()

        gist = _make_gist(supporting_ids=[])
        alignment = AlignmentResult(
            gist=gist,
            can_integrate_rapidly=False,
            integration_key=None,
            suggested_schema={"key": "user:preference:food", "category": "preference"},
        )
        migrator = ConsolidationMigrator(neocortical=mock_semantic, episodic_store=mock_episodic)
        result = await migrator.migrate(
            tenant_id="t1",
            user_id="u1",
            alignments=[alignment],
            mark_episodes_consolidated=False,
        )
        assert isinstance(result, MigrationResult)
        assert result.gists_processed == 1
        assert result.facts_created == 1
        assert result.facts_updated == 0
        assert result.episodes_marked == 0
        assert result.errors == []
        mock_semantic.store_fact.assert_called_once()

    @pytest.mark.asyncio
    async def test_migrate_updates_existing_fact_when_can_integrate_rapidly(self):
        mock_semantic = MagicMock()
        mock_semantic.store_fact = AsyncMock()
        mock_episodic = MagicMock()
        mock_episodic.get_by_id = AsyncMock(return_value=None)
        mock_episodic.update = AsyncMock()

        gist = _make_gist(key="user:preference:food")
        alignment = AlignmentResult(
            gist=gist,
            can_integrate_rapidly=True,
            integration_key="user:preference:food",
        )
        migrator = ConsolidationMigrator(neocortical=mock_semantic, episodic_store=mock_episodic)
        result = await migrator.migrate(
            tenant_id="t1",
            user_id="u1",
            alignments=[alignment],
            mark_episodes_consolidated=False,
        )
        assert result.facts_updated == 1
        assert result.facts_created == 0
        mock_semantic.store_fact.assert_called_once_with(
            tenant_id="t1",
            key="user:preference:food",
            value="vegetarian",
            confidence=0.9,
            evidence_ids=[],
        )


class TestEpisodeSampler:
    """EpisodeSampler with mocked store."""

    @pytest.mark.asyncio
    async def test_sample_returns_records_up_to_max(self):
        mock_store = MagicMock()
        records = [
            _make_memory_record("a", importance=0.9, access_count=5),
            _make_memory_record("b", importance=0.7, access_count=2),
            _make_memory_record("c", importance=0.5),
        ]
        mock_store.scan = AsyncMock(return_value=records)

        sampler = EpisodeSampler(store=mock_store, config=SamplingConfig(max_episodes=200))
        result = await sampler.sample(tenant_id="t1", user_id="u1", max_episodes=2)
        assert len(result) == 2
        assert all(isinstance(r, MemoryRecord) for r in result)
        mock_store.scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_sample_respects_min_importance_and_confidence(self):
        mock_store = MagicMock()
        low = _make_memory_record("low", importance=0.1, confidence=0.1)
        high = _make_memory_record("high", importance=0.9, confidence=0.9)
        mock_store.scan = AsyncMock(return_value=[low, high])

        config = SamplingConfig(min_importance=0.5, min_confidence=0.5)
        sampler = EpisodeSampler(store=mock_store, config=config)
        result = await sampler.sample(tenant_id="t1", user_id="u1", max_episodes=10)
        assert len(result) == 1
        assert result[0].text == "high"

    @pytest.mark.asyncio
    async def test_sample_excludes_consolidated_when_requested(self):
        mock_store = MagicMock()
        consolidated = _make_memory_record("cons")
        consolidated.metadata = {"consolidated": True}
        active = _make_memory_record("active")
        mock_store.scan = AsyncMock(return_value=[consolidated, active])

        sampler = EpisodeSampler(store=mock_store)
        result = await sampler.sample(
            tenant_id="t1",
            user_id="u1",
            max_episodes=10,
            exclude_consolidated=True,
        )
        assert len(result) == 1
        assert result[0].text == "active"
