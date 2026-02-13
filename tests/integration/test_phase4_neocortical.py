"""Integration tests for Phase 4: NeocorticalStore (fact store + mock graph)."""

from uuid import uuid4

import pytest

from src.memory.neocortical.fact_store import SemanticFactStore
from src.memory.neocortical.store import NeocorticalStore


class _MockGraphStore:
    """Minimal mock so NeocorticalStore runs without Neo4j."""

    async def merge_edge(self, *args, **kwargs):
        return "mock-edge-id"

    async def get_entity_facts(self, *args, **kwargs):
        return []

    async def personalized_pagerank(self, *args, **kwargs):
        return []


@pytest.mark.asyncio
async def test_neocortical_store_fact_and_profile(pg_session_factory):
    fact_store = SemanticFactStore(pg_session_factory)
    graph_store = _MockGraphStore()
    store = NeocorticalStore(graph_store=graph_store, fact_store=fact_store)

    tenant_id = f"t-{uuid4().hex[:8]}"

    fact = await store.store_fact(tenant_id, "user:identity:name", "Diana", confidence=0.85)
    assert fact.value == "Diana"

    got = await store.get_fact(tenant_id, "user:identity:name")
    assert got is not None
    assert got.value == "Diana"

    profile = await store.get_tenant_profile(tenant_id)
    assert isinstance(profile, dict)


@pytest.mark.asyncio
async def test_neocortical_text_search(pg_session_factory):
    fact_store = SemanticFactStore(pg_session_factory)
    graph_store = _MockGraphStore()
    store = NeocorticalStore(graph_store=graph_store, fact_store=fact_store)

    tenant_id = f"t-{uuid4().hex[:8]}"
    await store.store_fact(tenant_id, "user:location:current_city", "Berlin")

    results = await store.text_search(tenant_id, "Berlin", limit=5)
    assert isinstance(results, list)
    if results:
        assert results[0]["type"] == "fact"
