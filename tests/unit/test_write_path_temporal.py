"""Temporal resolution must run on the live write path.

It used to live only in ``HippocampalStore.encode_chunk``, which had no production
caller — so across 245,386 real records only 117 carried ``event_date``, and those were
test residue. ``packet_builder`` renders ``event_date`` on the Recent Events section, so
without this the model had no absolute date to answer "when did X happen" from.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.memory.hippocampal.store import HippocampalStore
from src.memory.working.models import ChunkType, SemanticChunk
from src.utils.embeddings import EmbeddingResult


def _store(n_embeddings: int) -> tuple[HippocampalStore, AsyncMock]:
    vector_store = AsyncMock()
    vector_store.scan = AsyncMock(return_value=[])
    vector_store.scan_texts_for_gate = AsyncMock(return_value=[])
    vector_store.upsert = AsyncMock(side_effect=lambda r: MagicMock(id=uuid4(), text=r.text))

    embeddings = MagicMock()
    embeddings.embed_batch = AsyncMock(
        return_value=[
            EmbeddingResult(embedding=[0.1] * 8, model="m", dimensions=8, tokens_used=5)
            for _ in range(n_embeddings)
        ]
    )
    return HippocampalStore(vector_store=vector_store, embedding_client=embeddings), vector_store


def _chunk(text: str, when: datetime) -> SemanticChunk:
    return SemanticChunk(
        id=str(uuid4()),
        text=text,
        chunk_type=ChunkType.EVENT,
        salience=0.9,
        timestamp=when,
    )


class TestEventDateOnTheLivePath:
    @pytest.mark.asyncio
    async def test_relative_reference_resolves_against_the_chunk_timestamp(self):
        session_day = datetime(2026, 3, 14, 9, 0, tzinfo=UTC)
        store, vector_store = _store(1)

        await store.encode_batch("t1", [_chunk("We shipped the release yesterday.", session_day)])

        written = vector_store.upsert.call_args[0][0]
        assert "event_date" in written.metadata, "event_date must reach the stored record"
        assert written.metadata["event_date"].startswith(
            (session_day - timedelta(days=1)).date().isoformat()
        )

    @pytest.mark.asyncio
    async def test_text_without_a_temporal_reference_gets_no_event_date(self):
        store, vector_store = _store(1)

        await store.encode_batch(
            "t1", [_chunk("The interface uses a monospace font.", datetime.now(UTC))]
        )

        written = vector_store.upsert.call_args[0][0]
        assert "event_date" not in written.metadata

    @pytest.mark.asyncio
    async def test_each_chunk_resolves_against_its_own_timestamp(self):
        """One flat loop over the batch — an off-by-one here would date-stamp a record
        from a neighbour's conversation."""
        early = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
        late = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
        store, vector_store = _store(2)

        await store.encode_batch(
            "t1",
            [
                _chunk("We met yesterday to plan the migration.", early),
                _chunk("The incident happened yesterday during deploy.", late),
            ],
        )

        by_text = {
            call[0][0].text: call[0][0].metadata.get("event_date")
            for call in vector_store.upsert.call_args_list
        }
        assert by_text["We met yesterday to plan the migration."].startswith("2026-01-09")
        assert by_text["The incident happened yesterday during deploy."].startswith("2026-06-19")

    @pytest.mark.asyncio
    async def test_caller_supplied_event_date_wins_over_the_regex(self):
        """Precedence is deliberate: regex goes in before the request_metadata merge."""
        store, vector_store = _store(1)

        await store.encode_batch(
            "t1",
            [_chunk("We shipped it yesterday.", datetime(2026, 3, 14, tzinfo=UTC))],
            request_metadata={"event_date": "1999-12-31"},
        )

        written = vector_store.upsert.call_args[0][0]
        assert written.metadata["event_date"] == "1999-12-31"


class TestRegexBeatsTheModel:
    """A date resolved against the record's own timestamp is right by construction;
    the LLM's is a guess. Letting the model win put a wrong date on 13.4% of records
    in a LoCoMo subset run -- every one the same hallucinated day -- and took the
    temporal category from 0.32 to 0.13."""

    @pytest.mark.asyncio
    async def test_llm_date_does_not_override_a_resolved_one(self):
        from types import SimpleNamespace

        store, vector_store = _store(1)
        session_day = datetime(2026, 3, 14, tzinfo=UTC)
        unified = SimpleNamespace(
            event_date="2023-10-24",
            speaker=None,
            entities=[],
            relations=[],
            pii_spans=[],
            memory_type=None,
            importance=None,
            confidence=None,
            decay_rate=None,
            context_tags=None,
            causal_chain=None,
            prospective_implications=None,
        )

        await store.encode_batch(
            "t1", [_chunk("We shipped it yesterday.", session_day)], unified_results=[unified]
        )

        written = vector_store.upsert.call_args[0][0]
        assert written.metadata["event_date"].startswith("2026-03-13")

    @pytest.mark.asyncio
    async def test_llm_date_is_used_when_the_regex_finds_nothing(self):
        from types import SimpleNamespace

        store, vector_store = _store(1)
        unified = SimpleNamespace(
            event_date="2023-10-24",
            speaker=None,
            entities=[],
            relations=[],
            pii_spans=[],
            memory_type=None,
            importance=None,
            confidence=None,
            decay_rate=None,
            context_tags=None,
            causal_chain=None,
            prospective_implications=None,
        )

        await store.encode_batch(
            "t1",
            [_chunk("The interface uses a monospace font.", datetime.now(UTC))],
            unified_results=[unified],
        )

        written = vector_store.upsert.call_args[0][0]
        assert written.metadata["event_date"] == "2023-10-24"
