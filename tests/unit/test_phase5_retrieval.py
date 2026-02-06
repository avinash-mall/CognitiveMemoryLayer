"""Unit tests for Phase 5: retrieval (classifier, planner, reranker, packet builder)."""
from datetime import datetime, timezone

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.retrieval.query_types import QueryAnalysis, QueryIntent
from src.retrieval.classifier import QueryClassifier
from src.retrieval.planner import RetrievalPlanner, RetrievalSource
from src.retrieval.reranker import MemoryReranker, RerankerConfig
from src.retrieval.packet_builder import MemoryPacketBuilder


class TestQueryClassifier:
    @pytest.mark.asyncio
    async def test_fast_preference_lookup(self):
        classifier = QueryClassifier(llm_client=None)
        result = await classifier.classify("What do I like for food?")
        assert result.intent == QueryIntent.PREFERENCE_LOOKUP
        assert result.confidence > 0.8
        assert "facts" in result.suggested_sources

    @pytest.mark.asyncio
    async def test_fast_identity_lookup(self):
        classifier = QueryClassifier(llm_client=None)
        result = await classifier.classify("What is my name?")
        assert result.intent == QueryIntent.IDENTITY_LOOKUP
        assert "facts" in result.suggested_sources

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self):
        classifier = QueryClassifier(llm_client=None)
        result = await classifier.classify("xyz random query abc")
        assert result.intent in (QueryIntent.GENERAL_QUESTION, QueryIntent.UNKNOWN)
        assert result.suggested_top_k == 10


class TestRetrievalPlanner:
    def test_plan_preference_lookup(self):
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="my favorite cuisine",
            intent=QueryIntent.PREFERENCE_LOOKUP,
            confidence=0.9,
            entities=["cuisine"],
            suggested_sources=["facts"],
            suggested_top_k=3,
        )
        plan = planner.plan(analysis)
        assert len(plan.steps) >= 1
        assert any(s.source == RetrievalSource.FACTS for s in plan.steps)
        assert plan.steps[0].key == "user:preference:cuisine"

    def test_plan_general_question(self):
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="tell me about the project",
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.7,
            suggested_sources=["vector", "facts"],
            suggested_top_k=10,
        )
        plan = planner.plan(analysis)
        assert any(s.source == RetrievalSource.VECTOR for s in plan.steps)
        assert any(s.source == RetrievalSource.FACTS for s in plan.steps)


class TestMemoryReranker:
    def _make_memory(self, text: str, relevance: float = 0.8, confidence: float = 0.9):
        return RetrievedMemory(
            record=MemoryRecord(
                tenant_id="t",
                context_tags=[],
                type=MemoryType.EPISODIC_EVENT,
                text=text,
                provenance=Provenance(source=MemorySource.AGENT_INFERRED),
                timestamp=datetime.now(timezone.utc),
                confidence=confidence,
            ),
            relevance_score=relevance,
            retrieval_source="vector",
        )

    def test_rerank_orders_by_score(self):
        reranker = MemoryReranker()
        mems = [
            self._make_memory("low", relevance=0.3, confidence=0.5),
            self._make_memory("high", relevance=0.9, confidence=0.9),
        ]
        result = reranker.rerank(mems, "query", max_results=2)
        assert len(result) == 2
        assert result[0].record.text == "high"

    def test_rerank_respects_max_results(self):
        reranker = MemoryReranker()
        mems = [self._make_memory(f"text{i}") for i in range(5)]
        result = reranker.rerank(mems, "query", max_results=2)
        assert len(result) == 2


class TestMemoryPacketBuilder:
    def _make_retrieved(self, text: str, mem_type: MemoryType = MemoryType.EPISODIC_EVENT):
        return RetrievedMemory(
            record=MemoryRecord(
                tenant_id="t",
                context_tags=[],
                type=mem_type,
                text=text,
                provenance=Provenance(source=MemorySource.AGENT_INFERRED),
                timestamp=datetime.now(timezone.utc),
                confidence=0.8,
            ),
            relevance_score=0.8,
            retrieval_source="vector",
        )

    def test_build_categorizes_by_type(self):
        builder = MemoryPacketBuilder()
        memories = [
            self._make_retrieved("episode one", MemoryType.EPISODIC_EVENT),
            self._make_retrieved("a fact", MemoryType.SEMANTIC_FACT),
            self._make_retrieved("a preference", MemoryType.PREFERENCE),
        ]
        packet = builder.build(memories, "query")
        assert len(packet.recent_episodes) >= 1
        assert len(packet.facts) >= 1
        assert len(packet.preferences) >= 1

    def test_to_llm_context_markdown(self):
        builder = MemoryPacketBuilder()
        memories = [self._make_retrieved("User likes coffee", MemoryType.PREFERENCE)]
        packet = builder.build(memories, "query")
        ctx = builder.to_llm_context(packet, max_tokens=500, format="markdown")
        assert "Retrieved Memory" in ctx
        assert "coffee" in ctx
