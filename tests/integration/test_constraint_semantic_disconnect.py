"""Integration test: constraint retrieval under semantic disconnect.

When the trigger query has no lexical overlap with the cue (e.g. "What restaurant?"
vs "I never eat shellfish"), the fact-based constraint path should still surface
the constraint.
"""

from uuid import uuid4

import pytest

from src.core.config import get_embedding_dimensions
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
async def test_constraint_surfaces_despite_semantic_disconnect(pg_session_factory):
    """Store 'I never eat shellfish because I'm allergic' as policy constraint;
    query 'What restaurant should we go to?' (no lexical overlap);
    assert constraint appears in packet.
    """
    pg_store = PostgresMemoryStore(pg_session_factory)
    hippocampal = HippocampalStore(
        vector_store=pg_store,
        embedding_client=MockEmbeddingClient(
            dimensions=get_embedding_dimensions()
        ),
        entity_extractor=None,
        relation_extractor=None,
        write_gate=WriteGate(),
        redactor=PIIRedactor(),
    )
    fact_store = SemanticFactStore(pg_session_factory)
    neocortical = NeocorticalStore(graph_store=_MockGraph(), fact_store=fact_store)

    tenant_id = f"t-{uuid4().hex[:8]}"
    constraint_text = "I never eat shellfish because I'm allergic."

    # Store constraint as semantic fact (POLICY category)
    await neocortical.store_fact(
        tenant_id,
        key="user:policy:shellfish_allergy",
        value=constraint_text,
        confidence=0.9,
    )

    retriever = MemoryRetriever(
        hippocampal=hippocampal,
        neocortical=neocortical,
        llm_client=None,
    )
    # Query has no lexical overlap with constraint (restaurant vs shellfish/allergic)
    packet = await retriever.retrieve(tenant_id, "What restaurant should we go to?")

    # Constraint should appear via fact-based constraint retrieval (all cognitive categories)
    all_texts = [m.record.text for m in packet.all_memories]
    constraint_found = any("shellfish" in t and "allergic" in t for t in all_texts) or any(
        constraint_text in t for t in all_texts
    )
    assert constraint_found, (
        f"Constraint '{constraint_text}' should appear for restaurant query "
        f"despite semantic disconnect. Got: {all_texts}"
    )
    # Should be in constraints section
    assert (
        len(packet.constraints) >= 1
    ), "Packet should have at least one constraint from fact lookup"
    constraint_mem = next(
        (
            c
            for c in packet.constraints
            if "shellfish" in c.record.text or "allergic" in c.record.text
        ),
        None,
    )
    assert (
        constraint_mem is not None
    ), f"Constraints should include shellfish allergy. Got: {[c.record.text for c in packet.constraints]}"
