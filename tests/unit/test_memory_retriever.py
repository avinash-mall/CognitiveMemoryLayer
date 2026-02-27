"""Unit tests for MemoryRetriever (embedding reuse, etc.)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.schemas import MemoryPacket
from src.retrieval.planner import RetrievalPlan, RetrievalSource, RetrievalStep
from src.retrieval.query_types import QueryAnalysis, QueryIntent
from src.utils.embeddings import EmbeddingResult


@pytest.mark.asyncio
async def test_retrieve_embeds_query_once(mock_embeddings):
    """MemoryRetriever.retrieve() calls embeddings.embed exactly once per query,
    passing the embedding to HybridRetriever to avoid duplicate embedding calls.
    """
    from src.retrieval.memory_retriever import MemoryRetriever

    # Ensure mock returns proper EmbeddingResult
    mock_embeddings.embed = AsyncMock(
        return_value=EmbeddingResult(
            embedding=[0.1] * 1536,
            model="test",
            dimensions=1536,
            tokens_used=10,
        )
    )

    hippocampal = MagicMock()
    hippocampal.embeddings = mock_embeddings
    hippocampal.search = AsyncMock(return_value=[])

    neocortical = MagicMock()
    neocortical.facts = MagicMock()
    neocortical.facts.get_facts_by_category = AsyncMock(return_value=[])
    neocortical.get_fact = AsyncMock(return_value=None)
    neocortical.text_search = AsyncMock(return_value=[])

    retriever = MemoryRetriever(
        hippocampal=hippocampal,
        neocortical=neocortical,
        llm_client=None,
    )

    # Decision-style query triggers both VECTOR and CONSTRAINTS steps
    await retriever.retrieve("tenant-1", "Should I order the lobster?")

    # Embeddings.embed must be called exactly once
    assert mock_embeddings.embed.await_count == 1
    mock_embeddings.embed.assert_called_once_with("Should I order the lobster?")


@pytest.mark.asyncio
async def test_retrieve_propagates_retrieval_meta(mock_embeddings):
    """MemoryRetriever copies retriever diagnostics into MemoryPacket.retrieval_meta."""
    from src.retrieval.memory_retriever import MemoryRetriever

    mock_embeddings.embed = AsyncMock(
        return_value=EmbeddingResult(
            embedding=[0.1] * 1536,
            model="test",
            dimensions=1536,
            tokens_used=10,
        )
    )
    hippocampal = MagicMock()
    hippocampal.embeddings = mock_embeddings
    neocortical = MagicMock()

    retriever = MemoryRetriever(hippocampal=hippocampal, neocortical=neocortical, llm_client=None)

    analysis = QueryAnalysis(
        original_query="q",
        intent=QueryIntent.GENERAL_QUESTION,
        confidence=0.8,
    )
    plan = RetrievalPlan(
        query="q",
        analysis=analysis,
        steps=[RetrievalStep(source=RetrievalSource.VECTOR, query="q")],
    )
    retriever.classifier = MagicMock()
    retriever.classifier.classify = AsyncMock(return_value=analysis)
    retriever.planner = MagicMock()
    retriever.planner.plan = MagicMock(return_value=plan)

    async def _mock_retrieve(*_args, **_kwargs):
        plan.analysis.metadata["retrieval_meta"] = {
            "sources_completed": ["vector"],
            "sources_timed_out": [],
            "total_elapsed_ms": 12.3,
        }
        return []

    retriever.retriever = MagicMock()
    retriever.retriever.hippocampal = MagicMock()
    retriever.retriever.hippocampal.embeddings = MagicMock()
    retriever.retriever.hippocampal.embeddings.embed = AsyncMock(
        return_value=EmbeddingResult(
            embedding=[0.1] * 1536,
            model="test",
            dimensions=1536,
            tokens_used=10,
        )
    )
    retriever.retriever.retrieve = AsyncMock(side_effect=_mock_retrieve)
    retriever.reranker = MagicMock()
    retriever.reranker.rerank = AsyncMock(return_value=[])
    retriever.packet_builder = MagicMock()
    retriever.packet_builder.build = MagicMock(return_value=MemoryPacket(query="q"))

    packet = await retriever.retrieve("tenant-1", "q")
    assert packet.retrieval_meta == {
        "sources_completed": ["vector"],
        "sources_timed_out": [],
        "total_elapsed_ms": 12.3,
    }


@pytest.mark.asyncio
async def test_retrieve_session_scope_prunes_sources_and_applies_filter(mock_embeddings):
    """Session-scoped retrieval keeps VECTOR/CONSTRAINTS only and adds source_session_id filter."""
    from src.retrieval.memory_retriever import MemoryRetriever

    mock_embeddings.embed = AsyncMock(
        return_value=EmbeddingResult(
            embedding=[0.1] * 1536,
            model="test",
            dimensions=1536,
            tokens_used=10,
        )
    )
    hippocampal = MagicMock()
    hippocampal.embeddings = mock_embeddings
    neocortical = MagicMock()
    retriever = MemoryRetriever(hippocampal=hippocampal, neocortical=neocortical, llm_client=None)

    analysis = QueryAnalysis(
        original_query="q",
        intent=QueryIntent.GENERAL_QUESTION,
        confidence=0.8,
    )
    plan = RetrievalPlan(
        query="q",
        analysis=analysis,
        steps=[
            RetrievalStep(source=RetrievalSource.FACTS, query="q"),
            RetrievalStep(source=RetrievalSource.VECTOR, query="q"),
            RetrievalStep(source=RetrievalSource.GRAPH, query="q"),
            RetrievalStep(source=RetrievalSource.CONSTRAINTS, query="q"),
        ],
        parallel_steps=[[0, 1, 2, 3]],
    )

    retriever.classifier = MagicMock()
    retriever.classifier.classify = AsyncMock(return_value=analysis)
    retriever.planner = MagicMock()
    retriever.planner.plan = MagicMock(return_value=plan)

    captured = {}

    async def _capture_retrieve(_tenant, passed_plan, **kwargs):
        captured["plan"] = passed_plan
        return []

    retriever.retriever = MagicMock()
    retriever.retriever.hippocampal = MagicMock()
    retriever.retriever.hippocampal.embeddings = MagicMock()
    retriever.retriever.hippocampal.embeddings.embed = AsyncMock(
        return_value=EmbeddingResult(
            embedding=[0.1] * 1536,
            model="test",
            dimensions=1536,
            tokens_used=10,
        )
    )
    retriever.retriever.retrieve = AsyncMock(side_effect=_capture_retrieve)
    retriever.reranker = MagicMock()
    retriever.reranker.rerank = AsyncMock(return_value=[])
    retriever.packet_builder = MagicMock()
    retriever.packet_builder.build = MagicMock(return_value=MemoryPacket(query="q"))

    await retriever.retrieve("tenant-1", "q", source_session_id="session-123")

    assert "plan" in captured
    filtered_sources = [s.source for s in captured["plan"].steps]
    assert filtered_sources == [RetrievalSource.VECTOR, RetrievalSource.CONSTRAINTS]
    for step in captured["plan"].steps:
        assert step.time_filter is not None
        assert step.time_filter["source_session_id"] == "session-123"
