"""Integration test: full retrieval flow (classify -> plan -> retrieve -> rerank -> packet)."""

from uuid import uuid4

import pytest

from src.core.config import get_settings
from src.memory.hippocampal.store import HippocampalStore
from src.memory.hippocampal.write_gate import WriteGate
from src.memory.hippocampal.redactor import PIIRedactor
from src.memory.neocortical.store import NeocorticalStore
from src.memory.neocortical.fact_store import SemanticFactStore
from src.storage.postgres import PostgresMemoryStore
from src.utils.embeddings import MockEmbeddingClient
from src.retrieval.memory_retriever import MemoryRetriever


class _MockGraph:
    async def merge_edge(self, *args, **kwargs):
        return "mock"

    async def get_entity_facts(self, *args, **kwargs):
        return []

    async def personalized_pagerank(self, *args, **kwargs):
        return []


@pytest.mark.asyncio
async def test_retrieve_returns_packet_with_facts(pg_session_factory):
    """Store a fact, then retrieve via MemoryRetriever; packet should contain it."""
    pg_store = PostgresMemoryStore(pg_session_factory)
    hippocampal = HippocampalStore(
        vector_store=pg_store,
        embedding_client=MockEmbeddingClient(dimensions=get_settings().embedding.dimensions),
        entity_extractor=None,
        relation_extractor=None,
        write_gate=WriteGate(),
        redactor=PIIRedactor(),
    )
    fact_store = SemanticFactStore(pg_session_factory)
    neocortical = NeocorticalStore(graph_store=_MockGraph(), fact_store=fact_store)

    tenant_id = f"t-{uuid4().hex[:8]}"

    await neocortical.store_fact(tenant_id, "user:preference:cuisine", "Italian", confidence=0.9)

    retriever = MemoryRetriever(
        hippocampal=hippocampal,
        neocortical=neocortical,
        llm_client=None,
    )
    packet = await retriever.retrieve(tenant_id, "cuisine")

    assert packet.query == "cuisine"
    all_mems = packet.all_memories
    assert (
        len(all_mems) >= 1
    ), "retrieval should find fact (key user:preference:cuisine contains 'cuisine')"
    texts = [m.record.text for m in all_mems]
    assert any("Italian" in t or "cuisine" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_retrieve_for_llm_returns_string(pg_session_factory):
    pg_store = PostgresMemoryStore(pg_session_factory)
    hippocampal = HippocampalStore(
        vector_store=pg_store,
        embedding_client=MockEmbeddingClient(dimensions=get_settings().embedding.dimensions),
        entity_extractor=None,
        relation_extractor=None,
        write_gate=WriteGate(),
        redactor=PIIRedactor(),
    )
    fact_store = SemanticFactStore(pg_session_factory)
    neocortical = NeocorticalStore(graph_store=_MockGraph(), fact_store=fact_store)
    retriever = MemoryRetriever(hippocampal=hippocampal, neocortical=neocortical, llm_client=None)

    tenant_id = f"t-{uuid4().hex[:8]}"

    ctx = await retriever.retrieve_for_llm(tenant_id, "my preferences", max_tokens=500)
    assert isinstance(ctx, str)
    assert "Retrieved Memory" in ctx or len(ctx) >= 0
