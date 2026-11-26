"""Unit tests for MemoryRetriever explicit memory_types and since/until filtering."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.retrieval.memory_retriever import MemoryRetriever


def _rec(text: str, mtype: MemoryType, ts: datetime) -> RetrievedMemory:
    record = MemoryRecord(
        id=uuid4(),
        tenant_id="t1",
        context_tags=[],
        type=mtype,
        text=text,
        timestamp=ts,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
    )
    return RetrievedMemory(record=record, relevance_score=0.9, retrieval_source="vector")


def _make_retriever(candidates: list[RetrievedMemory]) -> MemoryRetriever:
    mr = MemoryRetriever.__new__(MemoryRetriever)
    mr.classifier = MagicMock()
    mr.classifier.classify = AsyncMock(
        return_value=SimpleNamespace(
            original_query="test",
            entities=[],
            user_timezone=None,
        )
    )
    mr.planner = MagicMock()
    mr.planner.plan = MagicMock(
        return_value=SimpleNamespace(
            steps=[],
            analysis=SimpleNamespace(
                original_query="test",
                metadata={},
                intent=SimpleNamespace(value="general"),
            ),
        )
    )
    mr.compute_intrinsic_spec = MagicMock(return_value=SimpleNamespace(to_dict=lambda: {}))
    mr.retriever = MagicMock()
    mr.retriever.retrieve = AsyncMock(return_value=list(candidates))
    mr.retriever.hippocampal = MagicMock()
    mr.retriever.hippocampal.embeddings = MagicMock()
    mr.retriever.hippocampal.embeddings.embed = AsyncMock(
        return_value=SimpleNamespace(embedding=[0.1] * 8)
    )
    mr._hyde_generator = None
    mr.short_term = None
    mr.dmn_store = None
    mr.procedural_engine = None
    mr.reranker = MagicMock()
    mr.reranker.rerank = AsyncMock(side_effect=lambda res, *_a, **_kw: res)
    mr.packet_builder = MagicMock()
    mr.packet_builder.build = MagicMock(
        side_effect=lambda res, query="": SimpleNamespace(
            query=query,
            all_memories=res,
            retrieval_meta={},
            open_questions=[],
            warnings=[],
        )
    )
    mr._sparse_bm25_augment = AsyncMock(return_value=[])
    mr._hypergraph_ppr_augment = AsyncMock(return_value=[])
    mr._resolve_proustian_and_conway_parents = AsyncMock(side_effect=lambda res, **_kw: res)
    return mr


@pytest.mark.asyncio
async def test_retrieve_enforces_memory_types_across_all_pathways() -> None:
    """Explicit memory_types filters out non-matching items from merged pathways."""
    c_const = _rec("Never schedule before 10am", MemoryType.CONSTRAINT, datetime(2025, 1, 1))
    c_fact = _rec("Dog is Biscuit", MemoryType.SEMANTIC_FACT, datetime(2025, 1, 1))
    c_ep = _rec("Visited Iceland", MemoryType.EPISODIC_EVENT, datetime(2025, 1, 1))
    mr = _make_retriever([c_const, c_fact, c_ep])

    packet = await mr.retrieve("t1", "schedule", memory_types=["constraint"])
    assert [m.record.type for m in packet.all_memories] == [MemoryType.CONSTRAINT]


@pytest.mark.asyncio
async def test_retrieve_enforces_since_and_until_across_all_pathways() -> None:
    """Explicit since/until filters out out-of-range records (naive or aware)."""
    old = _rec("In 2019 I visited Iceland", MemoryType.EPISODIC_EVENT, datetime(2019, 6, 15))
    mid = _rec("In 2024 I moved to Tokyo", MemoryType.EPISODIC_EVENT, datetime(2024, 6, 15))
    future = _rec("In 2026 I bought a bike", MemoryType.EPISODIC_EVENT, datetime(2026, 6, 15))
    mr = _make_retriever([old, mid, future])

    packet = await mr.retrieve(
        "t1",
        "events",
        since=datetime(2023, 1, 1, tzinfo=UTC),
        until=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert [m.record.text for m in packet.all_memories] == ["In 2024 I moved to Tokyo"]
