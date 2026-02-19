"""Integration tests for unified write path (LLM extraction)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.core.config import get_settings
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
