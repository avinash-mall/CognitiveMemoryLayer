"""Integration tests for NeocorticalStore (fact store and mock graph)."""

from typing import Any
from uuid import uuid4

import pytest

from src.core.schemas import Relation
from src.memory.neocortical.fact_store import SemanticFactStore
from src.memory.neocortical.store import NeocorticalStore


class _MockGraphStore:
    """Minimal mock so NeocorticalStore runs without Neo4j.

    Records every merge_edge / merge_node call for assertion.
    """

    def __init__(self) -> None:
        self.edges: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []

    async def merge_edge(self, *args: Any, **kwargs: Any) -> str:
        self.edges.append({"args": args, "kwargs": kwargs})
        return f"mock-edge-{len(self.edges)}"

    async def merge_node(self, *args: Any, **kwargs: Any) -> str:
        self.nodes.append({"args": args, "kwargs": kwargs})
        return f"mock-node-{len(self.nodes)}"

    async def get_entity_facts(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def personalized_pagerank(self, *args: Any, **kwargs: Any) -> list[Any]:
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


@pytest.mark.asyncio
async def test_neocortical_store_fact_get_fact_text_search_flow(pg_session_factory):
    """Flow: store_fact -> get_fact and text_search; assert key, value, confidence."""
    fact_store = SemanticFactStore(pg_session_factory)
    graph_store = _MockGraphStore()
    store = NeocorticalStore(graph_store=graph_store, fact_store=fact_store)
    tenant_id = f"t-{uuid4().hex[:8]}"

    key = "user:preference:cuisine"
    value = "Italian"
    confidence = 0.92
    await store.store_fact(tenant_id, key, value, confidence=confidence)

    got = await store.get_fact(tenant_id, key)
    assert got is not None
    assert got.key == key
    assert got.value == value
    assert got.confidence == confidence

    search_results = await store.text_search(tenant_id, "cuisine", limit=5)
    assert isinstance(search_results, list)
    found = next((r for r in search_results if r.get("key") == key), None)
    assert found is not None
    assert found["value"] == value
    assert found["confidence"] == confidence
    assert found["type"] == "fact"


# ---------------------------------------------------------------------------
# Graph relation sync tests (validates the code path used by orchestrator)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_relation_creates_graph_edge(pg_session_factory):
    """store_relation() should call graph.merge_edge with subject/predicate/object."""
    fact_store = SemanticFactStore(pg_session_factory)
    graph_store = _MockGraphStore()
    store = NeocorticalStore(graph_store=graph_store, fact_store=fact_store)

    tenant_id = f"t-{uuid4().hex[:8]}"
    relation = Relation(subject="Alice", predicate="knows", object="Bob", confidence=0.9)

    edge_id = await store.store_relation(tenant_id, relation, evidence_ids=["ev-1"])

    assert edge_id.startswith("mock-edge-")
    assert len(graph_store.edges) == 1

    call = graph_store.edges[0]
    # Positional: tenant_id, scope_id, subject=, predicate=, object=, properties=
    assert call["kwargs"]["subject"] == "Alice"
    assert call["kwargs"]["predicate"] == "knows"
    assert call["kwargs"]["object"] == "Bob"
    assert call["kwargs"]["properties"]["confidence"] == 0.9
    assert "ev-1" in call["kwargs"]["properties"]["evidence_ids"]


@pytest.mark.asyncio
async def test_store_relations_batch_creates_multiple_edges(pg_session_factory):
    """store_relations_batch() should create one edge per relation."""
    fact_store = SemanticFactStore(pg_session_factory)
    graph_store = _MockGraphStore()
    store = NeocorticalStore(graph_store=graph_store, fact_store=fact_store)

    tenant_id = f"t-{uuid4().hex[:8]}"
    relations = [
        Relation(subject="user", predicate="lives_in", object="Paris", confidence=0.85),
        Relation(subject="user", predicate="works_at", object="ACME", confidence=0.7),
        Relation(subject="Paris", predicate="located_in", object="France", confidence=1.0),
    ]

    edge_ids = await store.store_relations_batch(tenant_id, relations, evidence_ids=["mem-42"])

    assert len(edge_ids) == 3
    assert len(graph_store.edges) == 3

    subjects = [e["kwargs"]["subject"] for e in graph_store.edges]
    assert "user" in subjects
    assert "Paris" in subjects


@pytest.mark.asyncio
async def test_store_fact_syncs_relation_like_key_to_graph(pg_session_factory):
    """store_fact() with a colon-containing key and string value should
    also create an edge via _sync_fact_to_graph."""
    fact_store = SemanticFactStore(pg_session_factory)
    graph_store = _MockGraphStore()
    store = NeocorticalStore(graph_store=graph_store, fact_store=fact_store)

    tenant_id = f"t-{uuid4().hex[:8]}"

    # Key contains ":" and value is str → should trigger graph sync
    fact = await store.store_fact(tenant_id, "user:location:city", "Tokyo", confidence=0.9)

    assert fact.value == "Tokyo"
    # merge_edge should have been called once (by _sync_fact_to_graph)
    assert len(graph_store.edges) == 1


@pytest.mark.asyncio
async def test_store_fact_no_graph_sync_for_non_relation_key(pg_session_factory):
    """store_fact() whose key has no colon should NOT sync to graph."""
    fact_store = SemanticFactStore(pg_session_factory)
    graph_store = _MockGraphStore()
    store = NeocorticalStore(graph_store=graph_store, fact_store=fact_store)

    tenant_id = f"t-{uuid4().hex[:8]}"

    # Key without colon → no graph sync
    fact = await store.store_fact(tenant_id, "favorite_color", "blue", confidence=0.8)

    assert fact.value == "blue"
    assert len(graph_store.edges) == 0


# ---------------------------------------------------------------------------
# Backfill logic integration test (Postgres entities/relations → graph sync)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_syncs_entities_and_relations_from_postgres(pg_session_factory):
    """Simulate backfill: insert a memory with entities+relations in Postgres,
    then read it back and sync to a mock graph — mirrors what
    scripts/backfill_neo4j.py does."""
    from src.core.enums import MemorySource, MemoryType
    from src.core.schemas import EntityMention, MemoryRecordCreate, Provenance
    from src.storage.postgres import PostgresMemoryStore

    store = PostgresMemoryStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    # Create a record that already has entities and relations stored as JSON
    entities = [
        EntityMention(text="Rome", normalized="Rome", entity_type="LOCATION"),
        EntityMention(text="Italy", normalized="Italy", entity_type="LOCATION"),
    ]
    relations_data = [
        Relation(subject="Rome", predicate="capital_of", object="Italy", confidence=0.95),
    ]

    rec = await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text="Rome is the capital of Italy.",
            entities=entities,
            relations=relations_data,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )

    # Read back from Postgres to confirm JSON is stored
    fetched = await store.get_by_id(rec.id)
    assert fetched is not None
    assert len(fetched.entities) == 2
    assert len(fetched.relations) == 1

    # Now simulate backfill: iterate entities/relations and push to mock graph
    graph_store = _MockGraphStore()
    fact_store = SemanticFactStore(pg_session_factory)
    neocortical = NeocorticalStore(graph_store=graph_store, fact_store=fact_store)

    for ent in fetched.entities:
        await graph_store.merge_node(
            tenant_id=tenant_id,
            scope_id=tenant_id,
            entity=ent.normalized,
            entity_type=ent.entity_type,
        )
    for rel in fetched.relations:
        await neocortical.store_relation(
            tenant_id=tenant_id,
            relation=rel,
            evidence_ids=[str(fetched.id)],
        )

    # Assert nodes and edges were created
    assert len(graph_store.nodes) == 2
    assert len(graph_store.edges) == 1
    assert graph_store.nodes[0]["kwargs"]["entity"] == "Rome"
    assert graph_store.nodes[1]["kwargs"]["entity"] == "Italy"
    assert graph_store.edges[0]["kwargs"]["subject"] == "Rome"
    assert graph_store.edges[0]["kwargs"]["object"] == "Italy"
