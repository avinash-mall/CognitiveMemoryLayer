"""Prospective indexing: batched, tagged, and off unless asked for.

The feature was flagged on by default and had never executed — it lived only in
``encode_chunk``, which had no production caller. 0 prospective records existed across
245,386 real ones. Now that it runs, the things worth pinning are that it costs one
embedding call per *write* rather than per record, and that what it produces is marked
as agent-authored.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.enums import MemorySource
from src.memory.hippocampal.store import HippocampalStore
from src.memory.working.models import ChunkType, SemanticChunk
from src.utils.embeddings import EmbeddingResult


def _emb(n: int) -> list[EmbeddingResult]:
    return [
        EmbeddingResult(embedding=[0.1] * 8, model="m", dimensions=8, tokens_used=5)
        for _ in range(n)
    ]


def _store() -> tuple[HippocampalStore, AsyncMock, MagicMock]:
    vector_store = AsyncMock()
    vector_store.scan = AsyncMock(return_value=[])
    vector_store.scan_texts_for_gate = AsyncMock(return_value=[])
    # Must return a real-ish record: the batched helper reads context_tags/timestamp off
    # it, and MemoryRecordCreate rejects MagicMock values.
    vector_store.upsert = AsyncMock(
        side_effect=lambda r: SimpleNamespace(
            id=uuid4(),
            text=r.text,
            tenant_id=r.tenant_id,
            context_tags=list(r.context_tags or []),
            source_session_id=r.source_session_id,
            agent_id=r.agent_id,
            namespace=r.namespace,
            timestamp=r.timestamp,
            confidence=r.confidence,
            importance=r.importance,
            metadata=dict(r.metadata or {}),
        )
    )
    embeddings = MagicMock()
    embeddings.embed_batch = AsyncMock(side_effect=lambda texts: _emb(len(texts)))
    return (
        HippocampalStore(vector_store=vector_store, embedding_client=embeddings),
        (vector_store),
        embeddings,
    )


def _chunk(text: str) -> SemanticChunk:
    return SemanticChunk(id=str(uuid4()), text=text, chunk_type=ChunkType.STATEMENT, salience=0.9)


def _unified(implications: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        prospective_implications=implications,
        entities=[],
        relations=[],
        pii_spans=[],
        memory_type=None,
        importance=None,
        confidence=None,
        decay_rate=None,
        context_tags=None,
        speaker=None,
        event_date=None,
        causal_chain=None,
    )


class TestBatching:
    @pytest.mark.asyncio
    async def test_one_embed_call_for_all_implications_not_one_per_record(self, monkeypatch):
        """The per-record form issued N small embedding calls for an N-chunk write."""
        store, vector_store, embeddings = _store()
        monkeypatch.setattr(
            "src.core.config.get_settings",
            lambda: SimpleNamespace(
                features=SimpleNamespace(
                    prospective_indexing_enabled=True,
                    prospective_index_count=4,
                    use_llm_enabled=False,
                    temporal_resolution_enabled=False,
                ),
                performance=SimpleNamespace(resolved_gate_workers=lambda: 2),
            ),
        )

        await store.encode_batch(
            "t1",
            [_chunk("chunk one"), _chunk("chunk two")],
            unified_results=[_unified(["a", "b"]), _unified(["c", "d"])],
        )

        # exactly two: one for the chunk texts, one for every implication in the write
        assert embeddings.embed_batch.call_count == 2
        assert len(embeddings.embed_batch.call_args_list[1][0][0]) == 4
        assert vector_store.upsert.call_count == 6  # 2 real + 4 implications

    @pytest.mark.asyncio
    async def test_implications_are_tagged_as_agent_authored(self, monkeypatch):
        store, vector_store, _ = _store()
        monkeypatch.setattr(
            "src.core.config.get_settings",
            lambda: SimpleNamespace(
                features=SimpleNamespace(
                    prospective_indexing_enabled=True,
                    prospective_index_count=4,
                    use_llm_enabled=False,
                    temporal_resolution_enabled=False,
                ),
                performance=SimpleNamespace(resolved_gate_workers=lambda: 2),
            ),
        )

        await store.encode_batch(
            "t1", [_chunk("real statement")], unified_results=[_unified(["an implication"])]
        )

        written = [c[0][0] for c in vector_store.upsert.call_args_list]
        synthetic = [r for r in written if r.metadata.get("is_prospective_index")]
        assert len(synthetic) == 1
        assert synthetic[0].provenance.source == MemorySource.AGENT_INFERRED
        assert synthetic[0].key.startswith("prospective:")
        # Never presented as confidently as the statement it was derived from.
        assert synthetic[0].confidence < written[0].confidence


class TestDisabledByDefault:
    @pytest.mark.asyncio
    async def test_nothing_synthetic_is_written_when_the_flag_is_off(self, monkeypatch):
        store, vector_store, embeddings = _store()
        monkeypatch.setattr(
            "src.core.config.get_settings",
            lambda: SimpleNamespace(
                features=SimpleNamespace(
                    prospective_indexing_enabled=False,
                    prospective_index_count=4,
                    use_llm_enabled=False,
                    temporal_resolution_enabled=False,
                ),
                performance=SimpleNamespace(resolved_gate_workers=lambda: 2),
            ),
        )

        await store.encode_batch(
            "t1", [_chunk("real statement")], unified_results=[_unified(["an implication"])]
        )

        assert embeddings.embed_batch.call_count == 1
        written = [c[0][0] for c in vector_store.upsert.call_args_list]
        assert not any(r.metadata.get("is_prospective_index") for r in written)
