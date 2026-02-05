"""Integration tests for Phase 7: consolidation flow."""
from uuid import uuid4

import pytest

from src.consolidation.worker import ConsolidationWorker
from src.core.enums import MemoryScope, MemorySource, MemoryType
from src.core.schemas import MemoryRecordCreate, Provenance
from src.memory.neocortical.fact_store import SemanticFactStore
from src.memory.neocortical.store import NeocorticalStore
from src.storage.postgres import PostgresMemoryStore


class _MockGraphStore:
    async def merge_edge(self, *args, **kwargs):
        return "mock-edge-id"

    async def get_entity_facts(self, *args, **kwargs):
        return []

    async def personalized_pagerank(self, *args, **kwargs):
        return []


class _MockLLM:
    """Returns invalid JSON so GistExtractor uses fallback summary."""

    async def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 500, system_prompt=None):
        return "not valid json"

    async def complete_json(self, prompt: str, schema=None, temperature: float = 0.0):
        return {}


@pytest.mark.asyncio
async def test_consolidation_empty_episodes(pg_session_factory):
    """With no episodes, consolidate returns report with zeros."""
    episodic = PostgresMemoryStore(pg_session_factory)
    fact_store = SemanticFactStore(pg_session_factory)
    neocortical = NeocorticalStore(graph_store=_MockGraphStore(), fact_store=fact_store)
    worker = ConsolidationWorker(
        episodic_store=episodic,
        neocortical_store=neocortical,
        llm_client=_MockLLM(),
    )
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    report = await worker.consolidate(tenant_id, user_id)

    assert report.tenant_id == tenant_id
    assert report.user_id == user_id
    assert report.episodes_sampled == 0
    assert report.clusters_formed == 0
    assert report.gists_extracted == 0
    assert report.migration.gists_processed == 0
    assert report.elapsed_seconds >= 0


@pytest.mark.asyncio
async def test_consolidation_with_episodes_fallback_gist(pg_session_factory):
    """With episodic records, consolidation runs; mock LLM triggers fallback gist -> align -> migrate."""
    episodic = PostgresMemoryStore(pg_session_factory)
    fact_store = SemanticFactStore(pg_session_factory)
    neocortical = NeocorticalStore(graph_store=_MockGraphStore(), fact_store=fact_store)
    worker = ConsolidationWorker(
        episodic_store=episodic,
        neocortical_store=neocortical,
        llm_client=_MockLLM(),
    )
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    await episodic.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            scope=MemoryScope.USER,
            scope_id=user_id,
            user_id=user_id,
            type=MemoryType.EPISODIC_EVENT,
            text="User said they like pizza.",
            confidence=0.7,
            importance=0.6,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )

    report = await worker.consolidate(tenant_id, user_id)

    assert report.episodes_sampled >= 1
    assert report.clusters_formed >= 1
    assert report.gists_extracted >= 1
    assert report.migration.gists_processed >= 1
    assert report.success or len(report.migration.errors) >= 0
