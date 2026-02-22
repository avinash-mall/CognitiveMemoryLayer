"""Integration tests for unified write path (LLM extraction)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.core.config import get_settings
from src.core.enums import MemoryType
from src.extraction.unified_write_extractor import UnifiedWritePathExtractor
from src.memory.hippocampal.redactor import PIIRedactor
from src.memory.hippocampal.store import HippocampalStore
from src.memory.hippocampal.write_gate import WriteGate
from src.memory.working.models import ChunkType, SemanticChunk
from src.storage.postgres import PostgresMemoryStore
from src.utils.embeddings import MockEmbeddingClient


def _make_store_with_unified(session_factory):
    """Create HippocampalStore with mock UnifiedWritePathExtractor."""
    pg_store = PostgresMemoryStore(session_factory)
    dims = get_settings().embedding.dimensions
    embeddings = MockEmbeddingClient(dimensions=dims)
    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [
                {
                    "constraint_type": "goal",
                    "subject": "user",
                    "description": "I want to eat healthier",
                    "scope": [],
                    "confidence": 0.85,
                }
            ],
            "facts": [
                {
                    "key": "user:preference:cuisine",
                    "category": "preference",
                    "predicate": "cuisine",
                    "value": "vegetarian",
                    "confidence": 0.8,
                }
            ],
            "salience": 0.8,
            "importance": 0.7,
            "pii_spans": [],
            "contains_secrets": False,
        }
    )
    unified = UnifiedWritePathExtractor(mock_llm)
    return HippocampalStore(
        vector_store=pg_store,
        embedding_client=embeddings,
        entity_extractor=None,
        relation_extractor=None,
        write_gate=WriteGate(),
        redactor=PIIRedactor(),
        unified_extractor=unified,
    )


@pytest.mark.asyncio
async def test_encode_batch_with_unified_extractor_uses_llm_results(
    pg_session_factory, monkeypatch
):
    """When unified extractor is present and flags enabled, encode_batch uses LLM results."""
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: type(
            "S",
            (),
            {
                "features": type(
                    "F",
                    (),
                    {
                        "use_llm_constraint_extractor": True,
                        "use_llm_write_time_facts": True,
                        "use_llm_salience_refinement": True,
                        "use_llm_pii_redaction": False,
                        "use_llm_write_gate_importance": True,
                        "use_llm_memory_type": True,
                        "use_llm_confidence": True,
                        "use_llm_context_tags": True,
                        "use_llm_decay_rate": True,
                    },
                )()
            },
        )(),
    )
    store = _make_store_with_unified(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    chunks = [
        SemanticChunk(
            id="c1",
            text="I prefer vegetarian food and want to eat healthier",
            chunk_type=ChunkType.PREFERENCE,
            salience=0.7,
            confidence=0.9,
            timestamp=datetime.now(UTC),
        )
    ]
    results, _gate_results, unified_results = await store.encode_batch(
        tenant_id, chunks, return_gate_results=True
    )
    assert len(results) >= 1
    assert unified_results is not None
    assert len(unified_results) >= 1
    if unified_results[0]:
        assert len(unified_results[0].constraints) >= 1
        assert unified_results[0].importance == 0.7


def _make_store_with_llm_fields_mock(session_factory, confidence, context_tags, decay_rate):
    """Create HippocampalStore with mock returning confidence, context_tags, decay_rate."""
    pg_store = PostgresMemoryStore(session_factory)
    dims = get_settings().embedding.dimensions
    embeddings = MockEmbeddingClient(dimensions=dims)
    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [],
            "facts": [],
            "entities": [],
            "relations": [],
            "salience": 0.6,
            "importance": 0.6,
            "pii_spans": [],
            "contains_secrets": False,
            "confidence": confidence,
            "context_tags": context_tags,
            "decay_rate": decay_rate,
        }
    )
    unified = UnifiedWritePathExtractor(mock_llm)
    return HippocampalStore(
        vector_store=pg_store,
        embedding_client=embeddings,
        entity_extractor=None,
        relation_extractor=None,
        write_gate=WriteGate(),
        redactor=PIIRedactor(),
        unified_extractor=unified,
    )


@pytest.mark.asyncio
async def test_encode_batch_uses_llm_confidence_context_tags_decay_rate(
    pg_session_factory, monkeypatch
):
    """When use_llm_* flags are True, store uses LLM confidence, context_tags, decay_rate."""
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: type(
            "S",
            (),
            {
                "features": type(
                    "F",
                    (),
                    {
                        "use_llm_constraint_extractor": True,
                        "use_llm_write_time_facts": False,
                        "use_llm_salience_refinement": True,
                        "use_llm_pii_redaction": False,
                        "use_llm_write_gate_importance": True,
                        "use_llm_memory_type": True,
                        "use_llm_confidence": True,
                        "use_llm_context_tags": True,
                        "use_llm_decay_rate": True,
                    },
                )()
            },
        )(),
    )
    store = _make_store_with_llm_fields_mock(
        pg_session_factory,
        confidence=0.85,
        context_tags=["personal", "dietary"],
        decay_rate=0.1,
    )
    tenant_id = f"t-{uuid4().hex[:8]}"
    chunks = [
        SemanticChunk(
            id="c1",
            text="I prefer vegan food for health",
            chunk_type=ChunkType.PREFERENCE,
            salience=0.7,
            confidence=0.5,
            timestamp=datetime.now(UTC),
        )
    ]
    results, _, _ = await store.encode_batch(tenant_id, chunks)
    assert len(results) >= 1
    record = results[0]
    assert record.confidence == 0.85
    assert set(record.context_tags) >= {"personal", "dietary"}
    assert record.decay_rate == 0.1


def _make_store_with_memory_type_mock(session_factory, memory_type: str):
    """Create HippocampalStore with mock that returns specific memory_type."""
    pg_store = PostgresMemoryStore(session_factory)
    dims = get_settings().embedding.dimensions
    embeddings = MockEmbeddingClient(dimensions=dims)
    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(
        return_value={
            "constraints": [],
            "facts": [],
            "salience": 0.6,
            "importance": 0.6,
            "memory_type": memory_type,
            "pii_spans": [],
            "contains_secrets": False,
        }
    )
    unified = UnifiedWritePathExtractor(mock_llm)
    return HippocampalStore(
        vector_store=pg_store,
        embedding_client=embeddings,
        entity_extractor=None,
        relation_extractor=None,
        write_gate=WriteGate(),
        redactor=PIIRedactor(),
        unified_extractor=unified,
    )


@pytest.mark.asyncio
async def test_encode_batch_uses_llm_memory_type(pg_session_factory, monkeypatch):
    """When use_llm_memory_type is True and unified extractor returns memory_type, store uses it."""
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: type(
            "S",
            (),
            {
                "features": type(
                    "F",
                    (),
                    {
                        "use_llm_constraint_extractor": True,
                        "use_llm_write_time_facts": False,
                        "use_llm_salience_refinement": True,
                        "use_llm_pii_redaction": False,
                        "use_llm_write_gate_importance": True,
                        "use_llm_memory_type": True,
                        "use_llm_confidence": True,
                        "use_llm_context_tags": True,
                        "use_llm_decay_rate": True,
                    },
                )()
            },
        )(),
    )
    store = _make_store_with_memory_type_mock(pg_session_factory, "preference")
    tenant_id = f"t-{uuid4().hex[:8]}"
    chunks = [
        SemanticChunk(
            id="c1",
            text="I prefer dark mode for my IDE",
            chunk_type=ChunkType.STATEMENT,
            salience=0.7,
            confidence=0.9,
            timestamp=datetime.now(UTC),
        )
    ]
    results, _, _ = await store.encode_batch(tenant_id, chunks)
    assert len(results) >= 1
    assert results[0].type == MemoryType.PREFERENCE


@pytest.mark.asyncio
async def test_encode_batch_fallback_when_llm_memory_type_disabled(pg_session_factory, monkeypatch):
    """When use_llm_memory_type is False, store ignores LLM memory_type and uses gate/constraint."""
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: type(
            "S",
            (),
            {
                "features": type(
                    "F",
                    (),
                    {
                        "use_llm_constraint_extractor": True,
                        "use_llm_write_time_facts": False,
                        "use_llm_salience_refinement": True,
                        "use_llm_pii_redaction": False,
                        "use_llm_write_gate_importance": True,
                        "use_llm_memory_type": False,
                        "use_llm_confidence": True,
                        "use_llm_context_tags": True,
                        "use_llm_decay_rate": True,
                    },
                )()
            },
        )(),
    )
    store = _make_store_with_memory_type_mock(pg_session_factory, "preference")
    tenant_id = f"t-{uuid4().hex[:8]}"
    chunks = [
        SemanticChunk(
            id="c1",
            text="Some episodic event happened",
            chunk_type=ChunkType.STATEMENT,
            salience=0.6,
            confidence=0.8,
            timestamp=datetime.now(UTC),
        )
    ]
    results, _, _ = await store.encode_batch(tenant_id, chunks)
    assert len(results) >= 1
    assert results[0].type == MemoryType.EPISODIC_EVENT
