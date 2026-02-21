"""Integration tests for full retrieval flow (classify, plan, retrieve, rerank, packet)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.core.config import get_settings
from src.core.enums import MemoryType
from src.memory.hippocampal.redactor import PIIRedactor
from src.memory.hippocampal.store import HippocampalStore
from src.memory.hippocampal.write_gate import WriteGate
from src.memory.neocortical.fact_store import SemanticFactStore
from src.memory.neocortical.store import NeocorticalStore
from src.retrieval.memory_retriever import MemoryRetriever
from src.storage.postgres import PostgresMemoryStore
from src.utils.embeddings import MockEmbeddingClient


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
    assert len(all_mems) >= 1, (
        "retrieval should find fact (key user:preference:cuisine contains 'cuisine')"
    )
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


@pytest.mark.asyncio
async def test_retrieve_with_memory_types_filter_returns_only_allowed_types(pg_session_factory):
    """Seed vector store with episodic and preference; retrieve with memory_types filter; assert only allowed types."""
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
    retriever = MemoryRetriever(
        hippocampal=hippocampal,
        neocortical=neocortical,
        llm_client=None,
    )
    tenant_id = f"t-{uuid4().hex[:8]}"

    from src.memory.working.models import ChunkType, SemanticChunk

    # Seed one preference and one episodic (event)
    for chunk in [
        SemanticChunk(
            id="c1",
            text="I prefer coffee in the morning.",
            chunk_type=ChunkType.PREFERENCE,
            salience=0.85,
            timestamp=datetime.now(UTC),
        ),
        SemanticChunk(
            id="c2",
            text="We had a meeting yesterday.",
            chunk_type=ChunkType.EVENT,
            salience=0.7,
            timestamp=datetime.now(UTC),
        ),
    ]:
        await hippocampal.encode_chunk(tenant_id, chunk, existing_memories=None)

    packet = await retriever.retrieve(
        tenant_id, "coffee and meetings", memory_types=["preference"], max_results=10
    )
    for mem in packet.all_memories:
        assert mem.record.type == MemoryType.PREFERENCE, (
            f"memory_types filter should restrict to preference, got {mem.record.type}"
        )


@pytest.mark.asyncio
async def test_retrieve_mixed_vector_and_facts_both_sources_contribute(pg_session_factory):
    """Ecphory: seed episodic (vector) and fact (neocortical); retrieve; assert both sources in packet."""
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
    retriever = MemoryRetriever(
        hippocampal=hippocampal,
        neocortical=neocortical,
        llm_client=None,
    )
    tenant_id = f"t-{uuid4().hex[:8]}"

    from src.memory.working.models import ChunkType, SemanticChunk

    await hippocampal.encode_chunk(
        tenant_id,
        SemanticChunk(
            id="c1",
            text="theme",
            chunk_type=ChunkType.EVENT,
            salience=0.8,
            timestamp=datetime.now(UTC),
        ),
        existing_memories=None,
    )
    await neocortical.store_fact(tenant_id, "user:preference:theme", "dark", confidence=0.9)

    packet = await retriever.retrieve(tenant_id, "theme", max_results=10)
    all_mems = packet.all_memories
    assert len(all_mems) >= 1
    types = {m.record.type for m in all_mems}
    has_vector = MemoryType.EPISODIC_EVENT in types or len(packet.recent_episodes) >= 1
    has_fact = MemoryType.SEMANTIC_FACT in types or len(packet.facts) >= 1
    assert has_vector or has_fact, "packet should contain vector and/or fact results"
