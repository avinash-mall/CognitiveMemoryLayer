"""Unit tests for retrieval (classifier, planner, reranker, packet builder)."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.retrieval.classifier import QueryClassifier
from src.retrieval.packet_builder import MemoryPacketBuilder
from src.retrieval.planner import RetrievalPlanner, RetrievalSource
from src.retrieval.query_types import QueryAnalysis, QueryIntent
from src.retrieval.reranker import MemoryReranker


class TestQueryClassifier:
    class _StubModelPack:
        def __init__(self, intent: str | None = None, dimension: str | None = None):
            self.available = True
            self._intent = intent
            self._dimension = dimension

        def predict_single(self, task: str, text: str):
            if task == "query_intent" and self._intent is not None:
                return SimpleNamespace(label=self._intent, confidence=0.91)
            if task == "constraint_dimension" and self._dimension is not None:
                return SimpleNamespace(label=self._dimension, confidence=0.86)
            return None

    @pytest.mark.asyncio
    async def test_modelpack_preference_lookup(self):
        classifier = QueryClassifier(
            llm_client=None,
            modelpack=self._StubModelPack(intent="preference_lookup"),
        )
        result = await classifier.classify("What do I like for food?")
        assert result.intent in (QueryIntent.PREFERENCE_LOOKUP, QueryIntent.GENERAL_QUESTION)
        assert "facts" in result.suggested_sources

    @pytest.mark.asyncio
    async def test_modelpack_identity_lookup(self):
        classifier = QueryClassifier(
            llm_client=None,
            modelpack=self._StubModelPack(intent="identity_lookup"),
        )
        result = await classifier.classify("What is my name?")
        assert result.intent in (QueryIntent.IDENTITY_LOOKUP, QueryIntent.GENERAL_QUESTION)
        assert "facts" in result.suggested_sources

    @pytest.mark.asyncio
    async def test_classifier_returns_general_or_unknown_for_random_query_without_llm(self):
        classifier = QueryClassifier(llm_client=None)
        result = await classifier.classify("xyz random query abc")
        assert result.intent in (QueryIntent.GENERAL_QUESTION, QueryIntent.UNKNOWN)
        assert result.suggested_top_k == 10

    @pytest.mark.asyncio
    async def test_decision_query_defaults_to_policy_dimension(self):
        classifier = QueryClassifier(
            llm_client=None,
            modelpack=self._StubModelPack(intent="constraint_check", dimension="policy"),
        )
        result = await classifier.classify("Should I try this new restaurant?")
        assert result.is_decision_query is True
        assert "policy" in (result.constraint_dimensions or [])


class TestRetrievalPlanner:
    def test_planner_preference_lookup_produces_facts_step(self):
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

    def test_planner_general_question_includes_vector_and_facts_steps(self):
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
                timestamp=datetime.now(UTC),
                confidence=confidence,
            ),
            relevance_score=relevance,
            retrieval_source="vector",
        )

    @pytest.mark.asyncio
    async def test_rerank_orders_by_score(self):
        reranker = MemoryReranker()
        mems = [
            self._make_memory("low", relevance=0.3, confidence=0.5),
            self._make_memory("high", relevance=0.9, confidence=0.9),
        ]
        result = await reranker.rerank(mems, "query", max_results=2)
        assert len(result) == 2
        assert result[0].record.text == "high"

    @pytest.mark.asyncio
    async def test_rerank_respects_max_results(self):
        reranker = MemoryReranker()
        mems = [self._make_memory(f"text{i}") for i in range(5)]
        result = await reranker.rerank(mems, "query", max_results=2)
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
                timestamp=datetime.now(UTC),
                confidence=0.8,
            ),
            relevance_score=0.8,
            retrieval_source="vector",
        )

    def test_packet_builder_categorizes_by_memory_type(self):
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

    def test_packet_builder_to_llm_context_includes_markdown(self):
        builder = MemoryPacketBuilder()
        memories = [self._make_retrieved("User likes coffee", MemoryType.PREFERENCE)]
        packet = builder.build(memories, "query")
        ctx = builder.to_llm_context(packet, max_tokens=500, format="markdown")
        assert "Retrieved Memory" in ctx
        assert "coffee" in ctx
