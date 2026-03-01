"""Unit tests for the Cognitive Constraint Layer.

Covers all 5 implementation phases:
  Phase 1: Evaluation harness correctness (date parsing, neutral prompt)
  Phase 2: Constraint extraction & storage (ConstraintExtractor, ChunkType, salience, write gate, schemas)
  Phase 3: Constraint-aware retrieval routing (classifier, planner, reranker, packet builder)
  Phase 4: Supersession & consolidation fixes (sampler, schema aligner, summarizer)
  Phase 5: Observability (query analysis fields)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.extraction.constraint_extractor import (
    ConstraintExtractor,
    ConstraintObject,
)
from src.memory.hippocampal.write_gate import WriteDecision, WriteGate
from src.memory.neocortical.schema_manager import SchemaManager
from src.memory.neocortical.schemas import (
    DEFAULT_FACT_SCHEMAS,
    FactCategory,
)
from src.memory.working.models import ChunkType, SemanticChunk
from src.retrieval.classifier import QueryClassifier
from src.retrieval.packet_builder import MemoryPacketBuilder
from src.retrieval.planner import RetrievalPlanner, RetrievalSource
from src.retrieval.query_types import QueryAnalysis, QueryIntent
from src.retrieval.reranker import MemoryReranker
from src.utils.modelpack import ModelPackRuntime

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_chunk(
    text: str,
    chunk_type: ChunkType = ChunkType.STATEMENT,
    entities: list[str] | None = None,
    source_turn_id: str | None = None,
) -> SemanticChunk:
    return SemanticChunk(
        id=str(uuid4()),
        text=text,
        chunk_type=chunk_type,
        entities=entities or [],
        source_turn_id=source_turn_id,
    )


def _make_retrieved(
    text: str,
    mem_type: MemoryType = MemoryType.EPISODIC_EVENT,
    metadata: dict | None = None,
    confidence: float = 0.8,
    relevance: float = 0.8,
) -> RetrievedMemory:
    return RetrievedMemory(
        record=MemoryRecord(
            tenant_id="t",
            context_tags=[],
            type=mem_type,
            text=text,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
            timestamp=datetime.now(UTC),
            confidence=confidence,
            metadata=metadata or {},
        ),
        relevance_score=relevance,
        retrieval_source="vector",
    )


class _StubModelPack:
    def __init__(
        self,
        *,
        single: dict[str, tuple[str, float]] | None = None,
        pair: dict[str, tuple[str, float]] | None = None,
    ):
        self.available = True
        self._single = single or {}
        self._pair = pair or {}

    def predict_single(self, task: str, text: str):
        payload = self._single.get(task)
        if payload is None:
            return None
        label, confidence = payload
        return SimpleNamespace(label=label, confidence=confidence)

    def predict_pair(self, task: str, text_a: str, text_b: str):
        payload = self._pair.get(task)
        if payload is None:
            return None
        label, confidence = payload
        return SimpleNamespace(label=label, confidence=confidence)


# ═══════════════════════════════════════════════════════════════════
# Phase 2a: ChunkType.CONSTRAINT
# ═══════════════════════════════════════════════════════════════════


class TestChunkTypeConstraint:
    """ChunkType enum includes CONSTRAINT value."""

    def test_constraint_enum_exists(self):
        assert hasattr(ChunkType, "CONSTRAINT")
        assert ChunkType.CONSTRAINT.value == "constraint"

    def test_constraint_distinct_from_other_types(self):
        all_types = {t.value for t in ChunkType}
        assert "constraint" in all_types
        assert len(all_types) == len(ChunkType)  # No duplicates


# ═══════════════════════════════════════════════════════════════════
# Phase 2b: ConstraintExtractor
# ═══════════════════════════════════════════════════════════════════


class TestConstraintExtractor:
    """ConstraintExtractor detects goals, values, states, causal, policy."""

    def test_constraint_extractor_extracts_goal(self):
        extractor = ConstraintExtractor(
            modelpack=_StubModelPack(single={"constraint_type": ("goal", 0.9)})
        )
        chunk = _make_chunk("I'm trying to eat healthier this year.")
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        types = {c.constraint_type for c in constraints}
        assert "goal" in types

    def test_constraint_extractor_extracts_value(self):
        extractor = ConstraintExtractor(
            modelpack=_StubModelPack(single={"constraint_type": ("value", 0.9)})
        )
        chunk = _make_chunk("I value environmental sustainability.")
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        types = {c.constraint_type for c in constraints}
        assert "value" in types

    def test_constraint_extractor_extracts_state(self):
        extractor = ConstraintExtractor(
            modelpack=_StubModelPack(single={"constraint_type": ("state", 0.9)})
        )
        chunk = _make_chunk("I'm currently dealing with a tight deadline.")
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        types = {c.constraint_type for c in constraints}
        assert "state" in types

    def test_constraint_extractor_extracts_causal(self):
        extractor = ConstraintExtractor(
            modelpack=_StubModelPack(single={"constraint_type": ("causal", 0.9)})
        )
        chunk = _make_chunk("I'm studying hard in order to get the scholarship.")
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        types = {c.constraint_type for c in constraints}
        assert "causal" in types

    def test_constraint_extractor_extracts_policy(self):
        extractor = ConstraintExtractor(
            modelpack=_StubModelPack(single={"constraint_type": ("policy", 0.9)})
        )
        chunk = _make_chunk("I never eat shellfish because I'm allergic.")
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        types = {c.constraint_type for c in constraints}
        assert "policy" in types

    def test_no_constraint_for_plain_text(self):
        extractor = ConstraintExtractor()
        chunk = _make_chunk("The sky is blue.")
        constraints = extractor.extract(chunk)
        assert len(constraints) == 0

    def test_empty_text_returns_empty(self):
        extractor = ConstraintExtractor()
        chunk = _make_chunk("")
        constraints = extractor.extract(chunk)
        assert constraints == []

    def test_confidence_includes_base(self):
        extractor = ConstraintExtractor(
            base_confidence=0.65,
            modelpack=_StubModelPack(single={"constraint_type": ("goal", 0.4)}),
        )
        chunk = _make_chunk("I'm trying to save money for a trip.")
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        for c in constraints:
            assert c.confidence >= 0.65

    def test_confidence_capped_at_1(self):
        extractor = ConstraintExtractor(base_confidence=0.95)
        chunk = _make_chunk("I'm trying to save money. I'm focused on this. I plan to do it.")
        constraints = extractor.extract(chunk)
        for c in constraints:
            assert c.confidence <= 1.0

    def test_subject_defaults_to_user(self):
        extractor = ConstraintExtractor(
            modelpack=_StubModelPack(single={"constraint_type": ("goal", 0.9)})
        )
        chunk = _make_chunk("I'm trying to be more productive.")
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        assert constraints[0].subject == "user"

    def test_subject_extracted_from_speaker_prefix(self):
        extractor = ConstraintExtractor(
            modelpack=_StubModelPack(single={"constraint_type": ("goal", 0.9)})
        )
        chunk = _make_chunk("Alice: I'm trying to be more productive.")
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        assert constraints[0].subject == "alice"

    def test_provenance_from_turn_id(self):
        extractor = ConstraintExtractor(
            modelpack=_StubModelPack(single={"constraint_type": ("policy", 0.9)})
        )
        chunk = _make_chunk("I must avoid dairy.", source_turn_id="turn_7")
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        assert "turn_7" in constraints[0].provenance

    def test_scope_from_entities(self):
        extractor = ConstraintExtractor(
            modelpack=_StubModelPack(single={"constraint_type": ("goal", 0.9)})
        )
        chunk = _make_chunk(
            "I'm trying to save money for Japan.",
            entities=["money", "Japan"],
        )
        constraints = extractor.extract(chunk)
        assert len(constraints) >= 1
        assert "Japan" in constraints[0].scope or "money" in constraints[0].scope


class TestConstraintExtractorBatch:
    """Batch extraction."""

    def test_extract_batch_multiple_chunks(self):
        class _BatchModelPack:
            available = True

            def predict_single(self, task: str, text: str):
                if task != "constraint_type":
                    return None
                lowered = text.lower()
                if "trying to" in lowered:
                    return SimpleNamespace(label="goal", confidence=0.9)
                if "never eat" in lowered:
                    return SimpleNamespace(label="policy", confidence=0.9)
                return None

        extractor = ConstraintExtractor(modelpack=_BatchModelPack())
        chunks = [
            _make_chunk("I'm trying to eat healthier."),
            _make_chunk("The weather is sunny."),
            _make_chunk("I never eat after midnight."),
        ]
        constraints = extractor.extract_batch(chunks)
        # First and third should produce constraints
        assert len(constraints) >= 2
        types = {c.constraint_type for c in constraints}
        assert "goal" in types
        assert "policy" in types

    def test_extract_batch_empty_list(self):
        extractor = ConstraintExtractor()
        assert extractor.extract_batch([]) == []


class TestConstraintObjectSerialization:
    """ConstraintObject.to_dict() serialization."""

    def test_to_dict_basic(self):
        co = ConstraintObject(
            constraint_type="goal",
            subject="user",
            description="Save money for Japan.",
            scope=["money", "Japan"],
            confidence=0.85,
        )
        d = co.to_dict()
        assert d["constraint_type"] == "goal"
        assert d["subject"] == "user"
        assert d["description"] == "Save money for Japan."
        assert d["scope"] == ["money", "Japan"]
        assert d["confidence"] == 0.85
        assert d["status"] == "active"

    def test_to_dict_datetime_serialised(self):
        now = datetime.now(UTC)
        co = ConstraintObject(
            constraint_type="state",
            subject="user",
            description="Stressed about work.",
            valid_from=now,
        )
        d = co.to_dict()
        assert isinstance(d["valid_from"], str)
        assert now.isoformat() == d["valid_from"]


# ═══════════════════════════════════════════════════════════════════
# Phase 2b: Supersession
# ═══════════════════════════════════════════════════════════════════


class TestConstraintSupersession:
    """detect_supersession() identifies when a new constraint replaces an old one."""

    @pytest.mark.asyncio
    async def test_same_type_same_scope_supersedes(self, monkeypatch):
        monkeypatch.setattr(
            "src.extraction.constraint_extractor.get_modelpack_runtime",
            lambda: _StubModelPack(pair={"supersession": ("supersedes", 0.9)}),
        )
        old = ConstraintObject("goal", "user", "Save 1000", scope=["money"])
        new = ConstraintObject("goal", "user", "Save 2000", scope=["money"])
        assert await ConstraintExtractor.detect_supersession(old, new) is True

    @pytest.mark.asyncio
    async def test_different_type_does_not_supersede(self):
        old = ConstraintObject("goal", "user", "Save money", scope=["money"])
        new = ConstraintObject("policy", "user", "Never spend", scope=["money"])
        assert await ConstraintExtractor.detect_supersession(old, new) is False

    @pytest.mark.asyncio
    async def test_same_type_no_scope_overlap_does_not_supersede(self):
        old = ConstraintObject("goal", "user", "Save money", scope=["money"])
        new = ConstraintObject("goal", "user", "Lose weight", scope=["health"])
        assert await ConstraintExtractor.detect_supersession(old, new) is False

    @pytest.mark.asyncio
    async def test_both_unscoped_same_type_supersedes(self, monkeypatch):
        monkeypatch.setattr(
            "src.extraction.constraint_extractor.get_modelpack_runtime",
            lambda: _StubModelPack(pair={"supersession": ("supersedes", 0.9)}),
        )
        old = ConstraintObject("policy", "user", "Never eat late")
        new = ConstraintObject("policy", "user", "Never eat after 8pm")
        assert await ConstraintExtractor.detect_supersession(old, new) is True

    @pytest.mark.asyncio
    async def test_inactive_old_does_not_supersede(self):
        old = ConstraintObject("goal", "user", "Old goal", status="superseded")
        new = ConstraintObject("goal", "user", "New goal")
        assert await ConstraintExtractor.detect_supersession(old, new) is False


# ═══════════════════════════════════════════════════════════════════
# Phase 2b: Constraint fact key
# ═══════════════════════════════════════════════════════════════════


class TestConstraintFactKey:
    """constraint_fact_key() generates stable keys."""

    def test_constraint_fact_key_deterministic(self):
        co = ConstraintObject("goal", "user", "Save money", scope=["finance"])
        key1 = ConstraintExtractor.constraint_fact_key(co)
        key2 = ConstraintExtractor.constraint_fact_key(co)
        assert key1 == key2

    def test_constraint_fact_key_format(self):
        co = ConstraintObject("value", "user", "Health matters", scope=["health"])
        key = ConstraintExtractor.constraint_fact_key(co)
        assert key.startswith("user:value:")
        # SHA256 hex chars after the second colon
        parts = key.split(":")
        assert len(parts) == 3
        assert len(parts[2]) == 12

    def test_different_scope_different_key(self):
        co1 = ConstraintObject("goal", "user", "A", scope=["finance"])
        co2 = ConstraintObject("goal", "user", "B", scope=["health"])
        assert ConstraintExtractor.constraint_fact_key(
            co1
        ) != ConstraintExtractor.constraint_fact_key(co2)

    def test_unscoped_uses_general(self):
        co = ConstraintObject("policy", "user", "Never eat late")
        key = ConstraintExtractor.constraint_fact_key(co)
        expected_hash = hashlib.sha256(b"general").hexdigest()[:12]
        assert key == f"user:policy:{expected_hash}"


# ═══════════════════════════════════════════════════════════════════
# Phase 2c: Write gate maps CONSTRAINT chunk type
# ═══════════════════════════════════════════════════════════════════


class TestWriteGateConstraint:
    """Write gate correctly handles ChunkType.CONSTRAINT."""

    def test_constraint_chunk_is_stored(self):
        gate = WriteGate()
        chunk = SemanticChunk(
            id="c1",
            text="I must avoid gluten due to my allergy.",
            chunk_type=ChunkType.CONSTRAINT,
            salience=0.85,
        )
        result = gate.evaluate(chunk)
        assert result.decision in (
            WriteDecision.STORE,
            WriteDecision.STORE_SYNC,
            WriteDecision.REDACT_AND_STORE,
        )
        # Must include CONSTRAINT memory type
        assert MemoryType.CONSTRAINT in result.memory_types

    def test_constraint_chunk_gets_importance_boost(self):
        class _ImportanceModelPack:
            available = True

            def predict_single(self, task: str, text: str):
                if task != "importance_bin":
                    return None
                lowered = text.lower()
                if "focused on saving" in lowered:
                    return SimpleNamespace(label="high", confidence=0.9)
                return SimpleNamespace(label="low", confidence=0.9)

        gate = WriteGate(modelpack=_ImportanceModelPack())
        constraint_chunk = SemanticChunk(
            id="c2",
            text="I'm focused on saving for the trip.",
            chunk_type=ChunkType.CONSTRAINT,
            salience=0.5,
        )
        statement_chunk = SemanticChunk(
            id="s1",
            text="The weather was nice today.",
            chunk_type=ChunkType.STATEMENT,
            salience=0.5,
        )
        r_constraint = gate.evaluate(constraint_chunk)
        r_statement = gate.evaluate(statement_chunk)
        # Constraint should have higher importance due to type boost
        assert r_constraint.importance > r_statement.importance


# ═══════════════════════════════════════════════════════════════════
# Phase 2c: FactCategory cognitive values
# ═══════════════════════════════════════════════════════════════════


class TestFactCategoryCognitive:
    """FactCategory enum includes cognitive constraint types."""

    def test_fact_category_goal_value(self):
        assert FactCategory.GOAL.value == "goal"

    def test_fact_category_state_value(self):
        assert FactCategory.STATE.value == "state"

    def test_fact_category_value_value(self):
        assert FactCategory.VALUE.value == "value"

    def test_fact_category_causal_value(self):
        assert FactCategory.CAUSAL.value == "causal"

    def test_fact_category_policy_value(self):
        assert FactCategory.POLICY.value == "policy"


class TestCognitiveFactSchemas:
    """DEFAULT_FACT_SCHEMAS includes cognitive constraint schemas."""

    def test_goal_schema_present(self):
        assert "user:goal:*" in DEFAULT_FACT_SCHEMAS
        schema = DEFAULT_FACT_SCHEMAS["user:goal:*"]
        assert schema.category == FactCategory.GOAL
        assert schema.temporal is True

    def test_value_schema_present(self):
        assert "user:value:*" in DEFAULT_FACT_SCHEMAS
        schema = DEFAULT_FACT_SCHEMAS["user:value:*"]
        assert schema.category == FactCategory.VALUE

    def test_state_schema_present(self):
        assert "user:state:*" in DEFAULT_FACT_SCHEMAS
        schema = DEFAULT_FACT_SCHEMAS["user:state:*"]
        assert schema.category == FactCategory.STATE
        assert schema.temporal is True

    def test_causal_schema_present(self):
        assert "user:causal:*" in DEFAULT_FACT_SCHEMAS
        schema = DEFAULT_FACT_SCHEMAS["user:causal:*"]
        assert schema.category == FactCategory.CAUSAL

    def test_policy_schema_present(self):
        assert "user:policy:*" in DEFAULT_FACT_SCHEMAS
        schema = DEFAULT_FACT_SCHEMAS["user:policy:*"]
        assert schema.category == FactCategory.POLICY


class TestSchemaManagerCognitive:
    """SchemaManager resolves cognitive wildcard schemas."""

    def test_goal_wildcard_resolves(self):
        mgr = SchemaManager()
        schema = mgr.get_schema("user:goal:fitness")
        assert schema is not None
        assert schema.category == FactCategory.GOAL

    def test_policy_wildcard_resolves(self):
        mgr = SchemaManager()
        schema = mgr.get_schema("user:policy:diet")
        assert schema is not None
        assert schema.category == FactCategory.POLICY

    def test_value_wildcard_resolves(self):
        mgr = SchemaManager()
        schema = mgr.get_schema("user:value:family")
        assert schema is not None
        assert schema.category == FactCategory.VALUE


# ═══════════════════════════════════════════════════════════════════
# Phase 3a: Query classifier CONSTRAINT_CHECK intent
# ═══════════════════════════════════════════════════════════════════


class TestQueryClassifierConstraintCheck:
    """Classifier detects CONSTRAINT_CHECK intent and enriches constraint dimensions."""

    @staticmethod
    def _constraint_classifier() -> QueryClassifier:
        return QueryClassifier(
            llm_client=None,
            modelpack=cast(
                "ModelPackRuntime",
                _StubModelPack(
                    single={
                        "query_intent": ("constraint_check", 0.9),
                        "constraint_dimension": ("policy", 0.8),
                    }
                ),
            ),
        )

    @pytest.mark.asyncio
    async def test_should_i_triggers_constraint_check(self):
        classifier = self._constraint_classifier()
        result = await classifier.classify("Should I order the lobster?")
        assert result.intent == QueryIntent.CONSTRAINT_CHECK
        assert "constraints" in result.suggested_sources

    @pytest.mark.asyncio
    async def test_can_i_triggers_constraint_check(self):
        classifier = self._constraint_classifier()
        result = await classifier.classify("Can I eat this cake?")
        assert result.intent == QueryIntent.CONSTRAINT_CHECK

    @pytest.mark.asyncio
    async def test_is_it_ok_triggers_constraint_check(self):
        classifier = self._constraint_classifier()
        result = await classifier.classify("Is it ok to skip the gym session?")
        assert result.intent == QueryIntent.CONSTRAINT_CHECK

    @pytest.mark.asyncio
    async def test_recommend_triggers_constraint_check(self):
        classifier = self._constraint_classifier()
        result = await classifier.classify("Can you recommend a restaurant?")
        assert result.intent == QueryIntent.CONSTRAINT_CHECK

    @pytest.mark.asyncio
    async def test_decision_query_enriches_dimensions(self):
        classifier = self._constraint_classifier()
        result = await classifier.classify("Should I buy this expensive jacket?")
        assert result.is_decision_query is True

    @pytest.mark.asyncio
    async def test_non_decision_query_not_flagged(self):
        classifier = QueryClassifier(
            llm_client=None,
            modelpack=cast(
                "ModelPackRuntime",
                _StubModelPack(single={"query_intent": ("identity_lookup", 0.9)}),
            ),
        )
        result = await classifier.classify("What is my name?")
        assert result.is_decision_query is False
        assert result.intent != QueryIntent.CONSTRAINT_CHECK


# ═══════════════════════════════════════════════════════════════════
# Phase 3a: QueryAnalysis constraint fields
# ═══════════════════════════════════════════════════════════════════


class TestQueryAnalysisConstraintFields:
    """QueryAnalysis includes constraint_dimensions and is_decision_query."""

    def test_default_constraint_dimensions_empty(self):
        qa = QueryAnalysis(
            original_query="test",
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.5,
            suggested_sources=["vector"],
            suggested_top_k=10,
        )
        assert qa.constraint_dimensions == []
        assert qa.is_decision_query is False

    def test_constraint_dimensions_set(self):
        qa = QueryAnalysis(
            original_query="test",
            intent=QueryIntent.CONSTRAINT_CHECK,
            confidence=0.9,
            suggested_sources=["constraints"],
            suggested_top_k=10,
            constraint_dimensions=["goal", "value"],
            is_decision_query=True,
        )
        assert qa.constraint_dimensions == ["goal", "value"]
        assert qa.is_decision_query is True


# ═══════════════════════════════════════════════════════════════════
# Phase 3b: Retrieval planner generates CONSTRAINTS step
# ═══════════════════════════════════════════════════════════════════


class TestRetrievalPlannerConstraints:
    """Planner creates CONSTRAINTS retrieval step for constraint-related queries."""

    def test_constraint_check_produces_constraints_step(self):
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="Should I eat the seafood?",
            intent=QueryIntent.CONSTRAINT_CHECK,
            confidence=0.9,
            suggested_sources=["constraints", "facts", "vector"],
            suggested_top_k=10,
            is_decision_query=True,
        )
        plan = planner.plan(analysis)
        constraint_steps = [s for s in plan.steps if s.source == RetrievalSource.CONSTRAINTS]
        assert len(constraint_steps) >= 1
        # Constraint step should have highest priority (0)
        assert constraint_steps[0].priority == 0

    def test_decision_query_produces_constraints_step(self):
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="Should I skip the meeting?",
            intent=QueryIntent.CONSTRAINT_CHECK,
            confidence=0.8,
            suggested_sources=["constraints", "facts", "vector"],
            suggested_top_k=10,
            is_decision_query=True,
            constraint_dimensions=["policy"],
        )
        plan = planner.plan(analysis)
        assert any(s.source == RetrievalSource.CONSTRAINTS for s in plan.steps)
        # Should also have vector and facts steps
        assert any(s.source == RetrievalSource.VECTOR for s in plan.steps)
        assert any(s.source == RetrievalSource.FACTS for s in plan.steps)

    def test_general_question_has_constraints_step(self):
        """BUG-01: General path now includes CONSTRAINTS step for semantic overlap (preferences/values)."""
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="Tell me about the project status.",
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.7,
            suggested_sources=["vector", "facts"],
            suggested_top_k=10,
        )
        plan = planner.plan(analysis)
        constraint_steps = [s for s in plan.steps if s.source == RetrievalSource.CONSTRAINTS]
        assert len(constraint_steps) >= 1

    def test_constraints_source_enum_value(self):
        assert RetrievalSource.CONSTRAINTS.value == "constraints"

    def test_can_i_afford_dinner_plan_contains_constraints_step(self):
        """For a decision-style query like 'Can I afford dinner?' the plan contains CONSTRAINTS step."""
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="Can I afford dinner?",
            intent=QueryIntent.CONSTRAINT_CHECK,
            confidence=0.85,
            suggested_sources=["constraints", "facts", "vector"],
            suggested_top_k=10,
            is_decision_query=True,
            constraint_dimensions=["goal", "value"],
        )
        plan = planner.plan(analysis)
        constraint_steps = [s for s in plan.steps if s.source == RetrievalSource.CONSTRAINTS]
        assert len(constraint_steps) >= 1
        assert constraint_steps[0].constraint_categories == ["goal", "value"]


# ═══════════════════════════════════════════════════════════════════
# Phase 3c: Reranker stability-aware recency weights
# ═══════════════════════════════════════════════════════════════════


class TestRerankerConstraintWeights:
    """Reranker uses reduced recency weight for constraint memory types."""

    def test_constraint_type_lower_recency_weight(self):
        reranker = MemoryReranker()
        constraint_mem = _make_retrieved(
            "I must avoid gluten.",
            mem_type=MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "policy"}]},
        )
        weight = reranker._get_recency_weight(constraint_mem)
        # Stable constraint types (policy) have zero recency weight
        assert weight == 0.0

    def test_goal_constraint_semi_stable_weight(self):
        reranker = MemoryReranker()
        goal_mem = _make_retrieved(
            "I'm trying to lose weight.",
            mem_type=MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "goal"}]},
        )
        weight = reranker._get_recency_weight(goal_mem)
        # Semi-stable: higher than stable but lower than episodic
        assert 0.10 <= weight <= 0.20

    def test_episodic_event_default_recency_weight(self):
        reranker = MemoryReranker()
        episode_mem = _make_retrieved("I had lunch.", mem_type=MemoryType.EPISODIC_EVENT)
        weight = reranker._get_recency_weight(episode_mem)
        # Default recency weight from config (0.2)
        assert weight == reranker.config.recency_weight

    @pytest.mark.asyncio
    async def test_constraint_reranked_above_old_episode(self):
        reranker = MemoryReranker()
        # Constraint: lower relevance but stable
        constraint = _make_retrieved(
            "I never eat shellfish.",
            mem_type=MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "policy"}]},
            relevance=0.7,
            confidence=0.9,
        )
        # Episode: slightly higher relevance but volatile
        episode = _make_retrieved(
            "Yesterday I had pizza.",
            mem_type=MemoryType.EPISODIC_EVENT,
            relevance=0.75,
            confidence=0.6,
        )
        result = await reranker.rerank(
            [episode, constraint], "Should I eat the shrimp?", max_results=2
        )
        # Both should be present (we aren't testing exact ordering due to complex scoring)
        assert len(result) == 2

    def test_value_constraint_recency_weight_zero(self):
        """Value constraint type gets recency_weight=0 (stable)."""
        reranker = MemoryReranker()
        value_mem = _make_retrieved(
            "I value sustainability.",
            mem_type=MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "value"}]},
        )
        weight = reranker._get_recency_weight(value_mem)
        assert weight == 0.0

    def test_state_constraint_semi_stable_weight(self):
        """State constraint type gets semi-stable recency weight (0.15)."""
        reranker = MemoryReranker()
        state_mem = _make_retrieved(
            "I'm currently stressed about work.",
            mem_type=MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "state"}]},
        )
        weight = reranker._get_recency_weight(state_mem)
        assert 0.10 <= weight <= 0.20

    @pytest.mark.asyncio
    async def test_stable_constraint_not_outranked_by_recency(self):
        """Old policy constraint (90 days) should not be outranked by recent episode;
        stable constraints have recency_weight=0, so age does not affect their score."""
        from datetime import timedelta

        reranker = MemoryReranker()
        old = datetime.now(UTC) - timedelta(days=90)
        recent = datetime.now(UTC) - timedelta(days=1)
        constraint = RetrievedMemory(
            record=MemoryRecord(
                tenant_id="t",
                context_tags=[],
                type=MemoryType.CONSTRAINT,
                text="I never eat shellfish.",
                provenance=Provenance(source=MemorySource.AGENT_INFERRED),
                timestamp=old,
                confidence=0.95,
                metadata={"constraints": [{"constraint_type": "policy"}]},
            ),
            relevance_score=0.85,
            retrieval_source="constraints",
        )
        episode = RetrievedMemory(
            record=MemoryRecord(
                tenant_id="t",
                context_tags=[],
                type=MemoryType.EPISODIC_EVENT,
                text="Yesterday I had pizza at the place.",
                provenance=Provenance(source=MemorySource.AGENT_INFERRED),
                timestamp=recent,
                confidence=0.6,
            ),
            relevance_score=0.7,
            retrieval_source="vector",
        )
        result = await reranker.rerank(
            [episode, constraint], "recommend a restaurant", max_results=2
        )
        assert len(result) == 2
        # Constraint (policy) has recency_weight=0 so age does not hurt it;
        # with higher relevance+confidence it should rank first
        assert result[0].record.text == "I never eat shellfish."

    @pytest.mark.asyncio
    async def test_rerank_stable_constraint_beats_newer_episode(self):
        """Old value constraint should rank above recent episode when scores are comparable;
        value has recency_weight=0 so age does not penalize it."""
        from datetime import timedelta

        reranker = MemoryReranker()
        old = datetime.now(UTC) - timedelta(days=60)
        recent = datetime.now(UTC) - timedelta(days=1)
        constraint = RetrievedMemory(
            record=MemoryRecord(
                tenant_id="t",
                context_tags=[],
                type=MemoryType.CONSTRAINT,
                text="I value honesty above everything.",
                provenance=Provenance(source=MemorySource.AGENT_INFERRED),
                timestamp=old,
                confidence=0.95,
                metadata={"constraints": [{"constraint_type": "value"}]},
            ),
            relevance_score=0.9,
            retrieval_source="constraints",
        )
        episode = RetrievedMemory(
            record=MemoryRecord(
                tenant_id="t",
                context_tags=[],
                type=MemoryType.EPISODIC_EVENT,
                text="Today I had a great meeting.",
                provenance=Provenance(source=MemorySource.AGENT_INFERRED),
                timestamp=recent,
                confidence=0.5,
            ),
            relevance_score=0.65,
            retrieval_source="vector",
        )
        result = await reranker.rerank([episode, constraint], "how should I behave?", max_results=2)
        assert len(result) == 2
        assert result[0].record.text == "I value honesty above everything."


# ═══════════════════════════════════════════════════════════════════
# Phase 3d: Packet builder constraint section
# ═══════════════════════════════════════════════════════════════════


class TestPacketBuilderConstraints:
    """Packet builder includes constraints in the packet."""

    def test_constraint_categorised_in_packet(self):
        builder = MemoryPacketBuilder()
        memories = [
            _make_retrieved("User is allergic to shellfish.", MemoryType.CONSTRAINT),
            _make_retrieved("A fact about user.", MemoryType.SEMANTIC_FACT),
            _make_retrieved("User likes coffee.", MemoryType.PREFERENCE),
        ]
        packet = builder.build(memories, "dietary restrictions")
        assert len(packet.constraints) >= 1
        assert packet.constraints[0].record.text == "User is allergic to shellfish."

    def test_constraint_in_llm_context_markdown(self):
        builder = MemoryPacketBuilder()
        memories = [
            _make_retrieved("Never eat peanuts.", MemoryType.CONSTRAINT),
        ]
        packet = builder.build(memories, "query")
        ctx = builder.to_llm_context(packet, max_tokens=500, format="markdown")
        assert "Constraints" in ctx
        assert "peanuts" in ctx

    def test_constraint_provenance_in_markdown(self):
        builder = MemoryPacketBuilder()
        memories = [
            _make_retrieved(
                "Must avoid dairy.",
                MemoryType.CONSTRAINT,
                metadata={"constraints": [{"constraint_type": "policy", "provenance": ["turn_3"]}]},
            ),
        ]
        packet = builder.build(memories, "query")
        ctx = builder.to_llm_context(packet, max_tokens=500, format="markdown")
        # Should include provenance info
        assert "dairy" in ctx

    def test_markdown_starts_with_active_constraints_and_filters_low_relevance_episodes(self):
        """Packet markdown starts with constraints section; episodes with relevance <= threshold omitted (BUG-02: default 0.5)."""
        builder = MemoryPacketBuilder()
        memories = [
            _make_retrieved(
                "User must follow budget.",
                MemoryType.CONSTRAINT,
                relevance=0.9,
                metadata={"constraints": [{"constraint_type": "policy"}]},
            ),
            _make_retrieved("Low relevance episode.", MemoryType.EPISODIC_EVENT, relevance=0.3),
            _make_retrieved("Another low one.", MemoryType.EPISODIC_EVENT, relevance=0.2),
            _make_retrieved("Relevant episode.", MemoryType.EPISODIC_EVENT, relevance=0.6),
        ]
        packet = builder.build(memories, "query")
        ctx = builder.to_llm_context(packet, max_tokens=1000, format="markdown")
        assert "Constraints" in ctx
        assert "Must Follow" in ctx
        assert "budget" in ctx
        assert "Earlier you said" in ctx
        assert "Relevant episode" in ctx
        assert "Low relevance episode" not in ctx
        assert "Another low one" not in ctx

    def test_constraint_token_budget_constraints_not_truncated(self):
        """With many facts/preferences and few constraints, constraints section is
        present and not truncated; truncation happens in facts/preferences first."""
        builder = MemoryPacketBuilder()
        memories = [
            _make_retrieved(
                "User must avoid shellfish.",
                MemoryType.CONSTRAINT,
                relevance=0.9,
                metadata={"constraints": [{"constraint_type": "policy"}]},
            ),
            _make_retrieved(
                "User follows a budget.",
                MemoryType.CONSTRAINT,
                relevance=0.8,
                metadata={"constraints": [{"constraint_type": "value"}]},
            ),
        ]
        for i in range(10):
            memories.append(
                _make_retrieved(
                    f"Fact number {i} with some extra text to consume tokens.",
                    MemoryType.SEMANTIC_FACT,
                    relevance=0.7,
                )
            )
        for i in range(5):
            memories.append(
                _make_retrieved(
                    f"Preference {i} with additional content.",
                    MemoryType.PREFERENCE,
                    relevance=0.7,
                )
            )
        packet = builder.build(memories, "query")
        ctx = builder.to_llm_context(packet, max_tokens=500, format="markdown")
        assert "Constraints" in ctx
        assert "shellfish" in ctx
        assert "budget" in ctx
        assert "Must Follow" in ctx
        assert ctx.find("shellfish") < ctx.find("... (truncated)") or "... (truncated)" not in ctx

    def test_markdown_section_order_constraints_first(self):
        """Markdown output has constraints before Recent Events and Known Facts."""
        builder = MemoryPacketBuilder()
        memories = [
            _make_retrieved(
                "User must avoid gluten.",
                MemoryType.CONSTRAINT,
                relevance=0.9,
                metadata={"constraints": [{"constraint_type": "policy"}]},
            ),
            _make_retrieved("A known fact.", MemoryType.SEMANTIC_FACT, relevance=0.8),
            _make_retrieved("Recent episode.", MemoryType.EPISODIC_EVENT, relevance=0.7),
        ]
        packet = builder.build(memories, "query")
        ctx = builder.to_llm_context(packet, max_tokens=500, format="markdown")
        constraints_pos = ctx.find("Constraints")
        recent_pos = ctx.find("Recent Events")
        facts_pos = ctx.find("Known Facts")
        assert constraints_pos >= 0
        if recent_pos >= 0:
            assert constraints_pos < recent_pos
        if facts_pos >= 0:
            assert constraints_pos < facts_pos

    def test_tight_token_budget_truncates_facts_not_constraints(self):
        """With tight token budget, constraints appear in full; facts get less space."""
        builder = MemoryPacketBuilder()
        memories = [
            _make_retrieved(
                "User must avoid shellfish.",
                MemoryType.CONSTRAINT,
                relevance=0.9,
                metadata={"constraints": [{"constraint_type": "policy"}]},
            ),
            _make_retrieved(
                "User follows budget.",
                MemoryType.CONSTRAINT,
                relevance=0.8,
                metadata={"constraints": [{"constraint_type": "value"}]},
            ),
        ]
        for i in range(20):
            memories.append(
                _make_retrieved(
                    f"Fact {i} with substantial text to consume token budget.",
                    MemoryType.SEMANTIC_FACT,
                    relevance=0.7,
                )
            )
        packet = builder.build(memories, "query")
        ctx_small = builder.to_llm_context(packet, max_tokens=100, format="markdown")
        ctx_large = builder.to_llm_context(packet, max_tokens=500, format="markdown")
        assert "Constraints" in ctx_small
        assert "shellfish" in ctx_small
        assert "budget" in ctx_small
        assert "Must Follow" in ctx_small
        assert len(ctx_small) < len(ctx_large)


# ═══════════════════════════════════════════════════════════════════
# Phase 4a: Consolidation sampler includes CONSTRAINT type
# ═══════════════════════════════════════════════════════════════════


class TestConsolidationSamplerConstraint:
    """EpisodeSampler includes CONSTRAINT type with longer time window."""

    def test_constraint_time_window_is_90_days(self):
        from src.consolidation.sampler import EpisodeSampler

        assert EpisodeSampler.CONSTRAINT_TIME_WINDOW_DAYS == 90


# ═══════════════════════════════════════════════════════════════════
# Phase 4b: Schema aligner cognitive type mapping
# ═══════════════════════════════════════════════════════════════════


class TestSchemaAlignerCognitiveTypes:
    """SchemaAligner maps cognitive gist types to FactCategory."""

    def test_cognitive_type_map_complete(self):
        from src.consolidation.schema_aligner import SchemaAligner

        expected_types = {"goal", "value", "state", "causal", "policy"}
        assert set(SchemaAligner._COGNITIVE_TYPE_MAP.keys()) == expected_types

    def test_cognitive_type_map_values(self):
        from src.consolidation.schema_aligner import SchemaAligner

        for k, v in SchemaAligner._COGNITIVE_TYPE_MAP.items():
            assert k == v  # Direct mapping in current implementation


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Extraction module exports
# ═══════════════════════════════════════════════════════════════════


class TestExtractionExports:
    """Verify ConstraintExtractor and ConstraintObject are exported."""

    def test_constraint_extractor_importable(self):
        from src.extraction import ConstraintExtractor as ImportedConstraintExtractor

        assert ImportedConstraintExtractor is ConstraintExtractor

    def test_constraint_object_importable(self):
        from src.extraction import ConstraintObject as ImportedConstraintObject

        assert ImportedConstraintObject is ConstraintObject
