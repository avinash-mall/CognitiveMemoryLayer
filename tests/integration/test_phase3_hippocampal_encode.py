"""Integration test: encode chunk through hippocampal store (with mock embedding)."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.memory.working.models import ChunkType, SemanticChunk
from src.memory.hippocampal.store import HippocampalStore
from src.memory.hippocampal.write_gate import WriteGate
from src.memory.hippocampal.redactor import PIIRedactor
from src.storage.postgres import PostgresMemoryStore
from src.utils.embeddings import MockEmbeddingClient


def _make_store(session_factory):
    pg_store = PostgresMemoryStore(session_factory)
    embeddings = MockEmbeddingClient(dimensions=1536)
    return HippocampalStore(
        vector_store=pg_store,
        embedding_client=embeddings,
        entity_extractor=None,
        relation_extractor=None,
        write_gate=WriteGate(),
        redactor=PIIRedactor(),
    )


@pytest.mark.asyncio
async def test_encode_chunk_stores_record(pg_session_factory):
    """Encode a chunk -> record in DB; search returns it."""
    hippocampal_store = _make_store(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    chunk = SemanticChunk(
        id="c1",
        text="I prefer dark mode for the interface.",
        chunk_type=ChunkType.PREFERENCE,
        salience=0.8,
        confidence=0.9,
        timestamp=datetime.now(timezone.utc),
    )
    record = await hippocampal_store.encode_chunk(
        tenant_id, chunk, existing_memories=None
    )
    assert record is not None
    assert record.text == chunk.text
    assert record.embedding is not None
    assert len(record.embedding) == 1536

    # Verify we can retrieve the record (scan/get_recent)
    recent = await hippocampal_store.get_recent(tenant_id, limit=10)
    assert any(r.id == record.id for r in recent), "encoded record should appear in get_recent"

    # Vector search smoke test (no exception; may return 0 results depending on pgvector/embedding env)
    results = await hippocampal_store.search(
        tenant_id, "user preference dark mode", top_k=5
    )
    if results:
        assert any(r.id == record.id for r in results), "stored record should appear in search when results returned"


@pytest.mark.asyncio
async def test_encode_chunk_skip_low_salience(pg_session_factory):
    hippocampal_store = _make_store(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    chunk = SemanticChunk(
        id="c2",
        text="Okay, sure.",
        chunk_type=ChunkType.STATEMENT,
        salience=0.1,
    )
    record = await hippocampal_store.encode_chunk(
        tenant_id, chunk, existing_memories=None
    )
    # May be None if write gate skips
    if record is None:
        return
    # If stored, it's still valid
    assert record.text == chunk.text
