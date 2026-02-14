"""Unit tests for no-op storage implementations."""

from src.memory.neocortical.schemas import FactCategory
from src.storage.noop_stores import NoOpFactStore, NoOpGraphStore


class TestNoOpGraphStore:
    """NoOpGraphStore returns fixed ids and empty lists."""

    async def test_merge_node_returns_fixed_id(self):
        store = NoOpGraphStore()
        out = await store.merge_node(
            tenant_id="t1",
            scope_id="s1",
            entity="e1",
            entity_type="PERSON",
        )
        assert out == "noop-node"

    async def test_merge_edge_returns_fixed_id(self):
        store = NoOpGraphStore()
        out = await store.merge_edge(
            tenant_id="t1",
            scope_id="s1",
            subject="a",
            predicate="knows",
            object="b",
        )
        assert out == "noop-edge"

    async def test_get_neighbors_returns_empty_list(self):
        store = NoOpGraphStore()
        out = await store.get_neighbors(
            tenant_id="t1",
            scope_id="s1",
            entity="e1",
            max_depth=2,
        )
        assert out == []

    async def test_personalized_pagerank_returns_empty_list(self):
        store = NoOpGraphStore()
        out = await store.personalized_pagerank(
            tenant_id="t1",
            scope_id="s1",
            seed_entities=["e1"],
            top_k=10,
        )
        assert out == []

    async def test_get_entity_facts_returns_empty_list(self):
        store = NoOpGraphStore()
        out = await store.get_entity_facts(
            tenant_id="t1",
            scope_id="s1",
            entity="e1",
        )
        assert out == []

    async def test_get_entity_facts_batch_returns_empty_dict(self):
        store = NoOpGraphStore()
        out = await store.get_entity_facts_batch(
            tenant_id="t1",
            scope_id="s1",
            entity_names=["e1", "e2"],
        )
        assert out == {}


class TestNoOpFactStore:
    """NoOpFactStore no-ops and returns empty/None or stub fact."""

    async def test_upsert_fact_returns_stub_semantic_fact(self):
        store = NoOpFactStore()
        out = await store.upsert_fact(
            tenant_id="t1",
            key="user:preference",
            value="dark mode",
            confidence=0.9,
        )
        assert out.id == "noop-fact"
        assert out.tenant_id == "t1"
        assert out.key == "user:preference"
        assert out.category == FactCategory.CUSTOM
        assert out.value == "dark mode"
        assert out.confidence == 0.9

    async def test_get_fact_returns_none(self):
        store = NoOpFactStore()
        out = await store.get_fact(tenant_id="t1", key="any")
        assert out is None

    async def test_get_facts_by_category_returns_empty_list(self):
        store = NoOpFactStore()
        out = await store.get_facts_by_category(
            tenant_id="t1",
            category=FactCategory.PREFERENCE,
        )
        assert out == []

    async def test_get_tenant_profile_returns_empty_dict(self):
        store = NoOpFactStore()
        out = await store.get_tenant_profile(tenant_id="t1")
        assert out == {}

    async def test_search_facts_returns_empty_list(self):
        store = NoOpFactStore()
        out = await store.search_facts(tenant_id="t1", query="test")
        assert out == []

    async def test_search_facts_batch_returns_empty_dict(self):
        store = NoOpFactStore()
        out = await store.search_facts_batch(
            tenant_id="t1",
            entity_names=["Alice", "Bob"],
            limit_per_entity=5,
        )
        assert out == {}
