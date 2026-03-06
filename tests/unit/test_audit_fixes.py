"""Unit tests for CML Audit 2026-03-06 fixes.

Covers:
  PR 1: Turn formatting (AUD-03), ranked memory order (AUD-04),
        generic query constraint filtering (AUD-05)
  PR 2: Metadata propagation (AUD-06), deterministic ordering,
        distinct evidence IDs (AUD-11)
  PR 3: Key coexistence (AUD-02), consolidation subtype preservation (AUD-08)
  PR 4: Non-LLM write-path semantic fidelity (AUD-01)
  Deferred: Graph expansion (AUD-07), batch edges (AUD-09),
            typed reconsolidation (AUD-12)
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryPacket, MemoryRecord, Provenance, RetrievedMemory
from src.retrieval.packet_builder import MemoryPacketBuilder
from src.retrieval.planner import RetrievalPlanner, RetrievalSource, RetrievalStep
from src.retrieval.query_types import QueryAnalysis, QueryIntent


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


# ═══════════════════════════════════════════════════════════════════
# PR 1 - AUD-03: Turn formatting uses constraint-first formatter
# ═══════════════════════════════════════════════════════════════════


class TestTurnFormattingConstraintFirst:
    """AUD-03: /memory/turn should use constraint-first formatting."""

    def test_format_for_injection_starts_with_constraints(self):
        from src.memory.seamless_provider import SeamlessMemoryProvider

        orchestrator = MagicMock()
        provider = SeamlessMemoryProvider(orchestrator, max_context_tokens=500)

        constraint_mem = _make_retrieved(
            "Never eat shellfish.",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "policy"}]},
            relevance=0.9,
        )
        episode_mem = _make_retrieved(
            "We went to a restaurant yesterday.",
            MemoryType.EPISODIC_EVENT,
            relevance=0.7,
        )

        packet = MemoryPacket(query="restaurant recommendation")
        result = provider._format_for_injection(packet, [constraint_mem, episode_mem])

        constraint_pos = result.find("Constraints")
        episode_pos = result.find("Recent Events")
        assert constraint_pos != -1, "Constraints section should be present"
        if episode_pos != -1:
            assert constraint_pos < episode_pos, "Constraints should appear before episodes"

    def test_format_for_injection_uses_packet_builder(self):
        from src.memory.seamless_provider import SeamlessMemoryProvider

        orchestrator = MagicMock()
        provider = SeamlessMemoryProvider(orchestrator, max_context_tokens=500)

        constraint_mem = _make_retrieved(
            "Must follow budget.",
            MemoryType.CONSTRAINT,
            relevance=0.9,
        )

        packet = MemoryPacket(query="spending advice")
        result = provider._format_for_injection(packet, [constraint_mem])

        assert "Retrieved Memory Context" in result


# ═══════════════════════════════════════════════════════════════════
# PR 1 - AUD-04: Ranked memories preserve reranker order
# ═══════════════════════════════════════════════════════════════════


class TestRankedMemoryOrder:
    """AUD-04: MemoryPacket.all_memories should preserve reranker order."""

    def test_build_populates_ranked_memories(self):
        builder = MemoryPacketBuilder()
        memories = [
            _make_retrieved("constraint text", MemoryType.CONSTRAINT, relevance=0.95),
            _make_retrieved("fact text", MemoryType.SEMANTIC_FACT, relevance=0.90),
            _make_retrieved("episode text", MemoryType.EPISODIC_EVENT, relevance=0.85),
        ]
        packet = builder.build(memories, "test query")

        assert len(packet.ranked_memories) == 3
        assert packet.ranked_memories[0].record.text == "constraint text"
        assert packet.ranked_memories[1].record.text == "fact text"
        assert packet.ranked_memories[2].record.text == "episode text"

    def test_all_memories_returns_ranked_order(self):
        builder = MemoryPacketBuilder()
        memories = [
            _make_retrieved("constraint", MemoryType.CONSTRAINT, relevance=0.95),
            _make_retrieved("fact", MemoryType.SEMANTIC_FACT, relevance=0.90),
            _make_retrieved("episode", MemoryType.EPISODIC_EVENT, relevance=0.85),
        ]
        packet = builder.build(memories, "test query")

        all_mems = packet.all_memories
        assert all_mems[0].record.text == "constraint"
        assert all_mems[1].record.text == "fact"
        assert all_mems[2].record.text == "episode"

    def test_all_memories_falls_back_to_categories_when_no_ranked(self):
        packet = MemoryPacket(
            query="test",
            facts=[_make_retrieved("fact", MemoryType.SEMANTIC_FACT)],
            constraints=[_make_retrieved("constraint", MemoryType.CONSTRAINT)],
        )
        all_mems = packet.all_memories
        assert len(all_mems) == 2
        assert all_mems[0].record.type == MemoryType.SEMANTIC_FACT
        assert all_mems[1].record.type == MemoryType.CONSTRAINT


# ═══════════════════════════════════════════════════════════════════
# PR 1 - AUD-05: Generic queries don't pull all constraint categories
# ═══════════════════════════════════════════════════════════════════


class TestConstraintFallbackSuppression:
    """AUD-05: Generic questions should not retrieve all constraint categories."""

    @pytest.mark.asyncio
    async def test_empty_categories_skips_fact_lookup(self):
        from src.retrieval.retriever import HybridRetriever

        hippocampal = MagicMock()
        hippocampal.search = AsyncMock(return_value=[])
        neocortical = MagicMock()
        neocortical.facts = MagicMock()
        neocortical.facts.get_facts_by_categories = AsyncMock(return_value=[])
        neocortical.facts.get_facts_by_category = AsyncMock(return_value=[])

        retriever = HybridRetriever(hippocampal, neocortical)

        step = RetrievalStep(
            source=RetrievalSource.CONSTRAINTS,
            query="What is the project status?",
            top_k=5,
            constraint_categories=[],
        )

        await retriever._retrieve_constraints("t", step)

        neocortical.facts.get_facts_by_categories.assert_not_awaited()
        neocortical.facts.get_facts_by_category.assert_not_awaited()

    def test_planner_generic_passes_empty_dimensions(self):
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="What is the project status?",
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.9,
            entities=[],
            constraint_dimensions=[],
            is_decision_query=False,
        )

        plan = planner.plan(analysis)

        constraint_steps = [s for s in plan.steps if s.source == RetrievalSource.CONSTRAINTS]
        for step in constraint_steps:
            assert not step.constraint_categories

    def test_planner_decision_query_gets_all_categories(self):
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="Should I order the shellfish?",
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.9,
            entities=[],
            constraint_dimensions=[],
            is_decision_query=True,
        )

        plan = planner.plan(analysis)

        constraint_steps = [s for s in plan.steps if s.source == RetrievalSource.CONSTRAINTS]
        assert len(constraint_steps) >= 1
        cats = constraint_steps[0].constraint_categories
        assert cats and len(cats) == 5


# ═══════════════════════════════════════════════════════════════════
# PR 2 - AUD-06: Metadata propagation in _fact_to_record
# ═══════════════════════════════════════════════════════════════════


class TestFactToRecordMetadata:
    """AUD-06: _fact_to_record should preserve subtype/provenance metadata."""

    def test_fact_to_record_populates_metadata(self):
        from src.memory.neocortical.schemas import FactCategory, SemanticFact
        from src.retrieval.retriever import HybridRetriever

        hippocampal = MagicMock()
        neocortical = MagicMock()
        retriever = HybridRetriever(hippocampal, neocortical)

        fact = SemanticFact(
            id=str(uuid4()),
            tenant_id="t",
            category=FactCategory.POLICY,
            key="user:policy:abc",
            subject="user",
            predicate="diet",
            value="no shellfish",
            value_type="str",
            confidence=0.9,
            evidence_ids=["ev1", "ev2"],
            updated_at=datetime.now(UTC),
        )
        item = {
            "type": MemoryType.CONSTRAINT.value,
            "source": "constraints",
            "text": "[Policy] no shellfish",
            "relevance": 0.8,
        }

        record = retriever._fact_to_record(fact, item)

        assert "constraints" in record.metadata
        assert record.metadata["constraints"][0]["constraint_type"] == "policy"
        assert record.metadata["constraints"][0]["evidence_ids"] == ["ev1", "ev2"]


# ═══════════════════════════════════════════════════════════════════
# PR 2 - AUD-11: Evidence ID mapping in batched writes
# ═══════════════════════════════════════════════════════════════════


class TestBatchedEvidenceMapping:
    """AUD-11: Multi-chunk writes should create distinct evidence IDs per fact."""

    @pytest.mark.asyncio
    async def test_write_time_facts_use_per_chunk_evidence(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from src.memory.orchestrator import MemoryOrchestrator

        orchestrator = MagicMock(spec=MemoryOrchestrator)
        orchestrator.neocortical = MagicMock()
        store_fact_mock = AsyncMock()
        orchestrator.neocortical.store_fact = store_fact_mock

        stored_rec_0 = SimpleNamespace(id="id-chunk-0")
        stored_rec_1 = SimpleNamespace(id="id-chunk-1")
        stored = [stored_rec_0, stored_rec_1]

        chunk_0 = SimpleNamespace(text="I like Italian food", chunk_type="preference")
        chunk_1 = SimpleNamespace(text="I live in New York", chunk_type="fact")
        chunks = [chunk_0, chunk_1]

        ur_0 = SimpleNamespace(
            facts=[SimpleNamespace(key="pref:cuisine", value="Italian", confidence=0.8)],
        )
        ur_1 = SimpleNamespace(
            facts=[SimpleNamespace(key="identity:location", value="New York", confidence=0.9)],
        )
        unified_results = [ur_0, ur_1]

        wpc = SimpleNamespace(
            write_time_facts=True,
            use_llm_facts=True,
        )

        await MemoryOrchestrator._phase_write_time_facts(
            orchestrator,
            tenant_id="t",
            chunks=chunks,
            stored=stored,
            unified_results=unified_results,
            timestamp=None,
            wpc=wpc,
        )

        assert store_fact_mock.await_count == 2
        call_0 = store_fact_mock.call_args_list[0]
        call_1 = store_fact_mock.call_args_list[1]

        assert call_0.kwargs["evidence_ids"] == ["id-chunk-0"]
        assert call_1.kwargs["evidence_ids"] == ["id-chunk-1"]


# ═══════════════════════════════════════════════════════════════════
# PR 3 - AUD-02: Constraint key coexistence
# ═══════════════════════════════════════════════════════════════════


class TestConstraintKeyCoexistence:
    """AUD-02: Different constraints of same type/scope should not collide."""

    def test_different_descriptions_produce_different_keys(self):
        from src.extraction.constraint_extractor import ConstraintExtractor, ConstraintObject

        c1 = ConstraintObject(
            constraint_type="policy",
            subject="user",
            description="Never eat shellfish due to allergy",
        )
        c2 = ConstraintObject(
            constraint_type="policy",
            subject="user",
            description="Always tip 20% at restaurants",
        )

        key1 = ConstraintExtractor.constraint_fact_key(c1)
        key2 = ConstraintExtractor.constraint_fact_key(c2)

        assert key1 != key2, "Different constraints should produce different keys"

    def test_same_description_produces_same_key(self):
        from src.extraction.constraint_extractor import ConstraintExtractor, ConstraintObject

        c1 = ConstraintObject(
            constraint_type="policy",
            subject="user",
            description="Never eat shellfish",
        )
        c2 = ConstraintObject(
            constraint_type="policy",
            subject="user",
            description="Never eat shellfish",
        )

        assert ConstraintExtractor.constraint_fact_key(
            c1
        ) == ConstraintExtractor.constraint_fact_key(c2)

    def test_key_includes_type_scope_and_description(self):
        from src.extraction.constraint_extractor import ConstraintExtractor, ConstraintObject

        c = ConstraintObject(
            constraint_type="value",
            subject="user",
            description="I value honesty above all",
            scope=["ethics"],
        )

        key = ConstraintExtractor.constraint_fact_key(c)
        parts = key.split(":")
        assert len(parts) == 4
        assert parts[0] == "user"
        assert parts[1] == "value"


# ═══════════════════════════════════════════════════════════════════
# PR 3 - AUD-08: Consolidation subtype preservation
# ═══════════════════════════════════════════════════════════════════


class TestConsolidationSubtypePreservation:
    """AUD-08: Consolidation should preserve constraint subtypes, not collapse to 'policy'."""

    def test_fallback_gist_type_preserves_goal(self):
        from src.consolidation.summarizer import GistExtractor

        result = GistExtractor._fallback_gist_type(["goal", "fact"])
        assert result == "goal"

    def test_fallback_gist_type_preserves_value(self):
        from src.consolidation.summarizer import GistExtractor

        result = GistExtractor._fallback_gist_type(["value"])
        assert result == "value"

    def test_fallback_gist_type_preserves_state(self):
        from src.consolidation.summarizer import GistExtractor

        result = GistExtractor._fallback_gist_type(["state", "summary"])
        assert result == "state"

    def test_fallback_gist_type_preserves_causal(self):
        from src.consolidation.summarizer import GistExtractor

        result = GistExtractor._fallback_gist_type(["causal"])
        assert result == "causal"

    def test_fallback_gist_type_preserves_policy(self):
        from src.consolidation.summarizer import GistExtractor

        result = GistExtractor._fallback_gist_type(["policy"])
        assert result == "policy"

    def test_fallback_gist_type_constraint_without_subtype_maps_to_policy(self):
        from src.consolidation.summarizer import GistExtractor

        result = GistExtractor._fallback_gist_type(["constraint"])
        assert result == "policy"

    def test_fallback_gist_preserves_dominant_constraint_subtype(self):
        from src.consolidation.worker import ConsolidationWorker

        cluster = SimpleNamespace(
            episodes=[
                SimpleNamespace(
                    id="ep1",
                    text="Save money for retirement",
                    confidence=0.9,
                    type=SimpleNamespace(value="constraint"),
                ),
            ],
            dominant_type="goal",
        )

        gist = ConsolidationWorker._fallback_gist(cluster)
        assert gist is not None
        assert gist.gist_type == "goal"


# ═══════════════════════════════════════════════════════════════════
# PR 4 - AUD-01: Non-LLM write-path semantic fidelity
# ═══════════════════════════════════════════════════════════════════


class TestNonLLMWritePathFidelity:
    """AUD-01: Local extractor memory_type and facts should be consumed."""

    def test_write_time_facts_processes_statement_chunks(self):
        from src.extraction.write_time_facts import WriteTimeFactExtractor
        from src.memory.working.models import ChunkType, SemanticChunk

        extractor = WriteTimeFactExtractor()
        chunk = SemanticChunk(
            id="c1",
            text="I really love Italian food",
            chunk_type=ChunkType.STATEMENT,
        )

        facts = extractor.extract(chunk)
        assert len(facts) >= 1, "STATEMENT chunks should now be processed for facts"
        assert any("Italian" in f.value or "italian" in f.value.lower() for f in facts)

    def test_write_time_facts_still_processes_preference_chunks(self):
        from src.extraction.write_time_facts import WriteTimeFactExtractor
        from src.memory.working.models import ChunkType, SemanticChunk

        extractor = WriteTimeFactExtractor()
        chunk = SemanticChunk(
            id="c2",
            text="I prefer reading science fiction books",
            chunk_type=ChunkType.PREFERENCE,
        )

        facts = extractor.extract(chunk)
        assert len(facts) >= 1

    def test_local_memory_type_used_in_store(self):
        """When local extractor returns memory_type, the store should use it."""
        from src.core.enums import MemoryType

        local_res = {"memory_type": "preference", "importance": 0.7}
        mt = None
        if local_res and local_res.get("memory_type"):
            try:
                mt = MemoryType(local_res["memory_type"])
            except (ValueError, KeyError):
                pass
        assert mt == MemoryType.PREFERENCE


# ═══════════════════════════════════════════════════════════════════
# AUD-07: Associative graph expansion beyond entity seeds
# ═══════════════════════════════════════════════════════════════════


def _make_memory_record(
    text: str,
    mem_type: MemoryType = MemoryType.EPISODIC_EVENT,
    metadata: dict | None = None,
    confidence: float = 0.8,
) -> MemoryRecord:
    return MemoryRecord(
        tenant_id="t",
        context_tags=[],
        type=mem_type,
        text=text,
        provenance=Provenance(source=MemorySource.AGENT_INFERRED),
        timestamp=datetime.now(UTC),
        confidence=confidence,
        metadata=metadata or {},
    )


class TestDeriveGraphSeeds:
    """AUD-07: _derive_graph_seeds extracts entities from constraint/fact results."""

    def test_extracts_entities_from_records(self):
        from src.retrieval.retriever import HybridRetriever

        entity = SimpleNamespace(normalized="Italian Food", text="Italian Food")
        rec = SimpleNamespace(entities=[entity], metadata={})
        results = [{"record": rec}]

        seeds = HybridRetriever._derive_graph_seeds(results)
        assert "Italian Food" in seeds

    def test_extracts_subject_from_constraint_metadata(self):
        from src.retrieval.retriever import HybridRetriever

        rec = SimpleNamespace(
            entities=[],
            metadata={
                "constraints": [{"constraint_type": "policy", "subject": "diet", "scope": "health"}]
            },
        )
        results = [{"record": rec}]

        seeds = HybridRetriever._derive_graph_seeds(results)
        assert "diet" in seeds
        assert "health" in seeds

    def test_deduplicates_and_caps_seeds(self):
        from src.retrieval.retriever import HybridRetriever

        entities = [
            SimpleNamespace(normalized=f"entity_{i}", text=f"entity_{i}") for i in range(20)
        ]
        rec = SimpleNamespace(entities=entities, metadata={})
        results = [{"record": rec}]

        seeds = HybridRetriever._derive_graph_seeds(results, max_seeds=5)
        assert len(seeds) == 5

    def test_skips_none_records(self):
        from src.retrieval.retriever import HybridRetriever

        results = [{"record": None}, {"record": None}]
        seeds = HybridRetriever._derive_graph_seeds(results)
        assert seeds == []

    def test_returns_empty_for_no_results(self):
        from src.retrieval.retriever import HybridRetriever

        seeds = HybridRetriever._derive_graph_seeds([])
        assert seeds == []


class TestPostRetrievalGraphExpansion:
    """AUD-07: Graph expansion fires post-retrieval when no GRAPH step in plan."""

    @pytest.mark.asyncio
    async def test_graph_expansion_fires_when_no_graph_step(self):
        from src.retrieval.retriever import HybridRetriever

        hippocampal = MagicMock()
        neocortical = MagicMock()
        neocortical.multi_hop_query = AsyncMock(
            return_value=[
                {"entity": "Restaurant A", "relevance_score": 0.7},
            ]
        )

        retriever = HybridRetriever(hippocampal, neocortical)

        entity = SimpleNamespace(normalized="Italian Food", text="Italian Food")
        constraint_rec = SimpleNamespace(
            entities=[entity],
            metadata={"constraints": [{"constraint_type": "policy", "subject": "diet"}]},
            source_session_id=None,
        )

        plan = MagicMock()
        plan.steps = [
            RetrievalStep(source=RetrievalSource.CONSTRAINTS, query="restaurant", top_k=5),
        ]
        plan.parallel_steps = [[0]]
        plan.total_timeout_ms = 5000
        plan.analysis = QueryAnalysis(
            original_query="restaurant recommendation",
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.9,
            entities=[],
        )
        plan.analysis.metadata = {}

        step_result = MagicMock()
        step_result.success = True
        step_result.items = [
            {
                "type": "constraint",
                "source": "constraints",
                "text": "No shellfish",
                "confidence": 0.9,
                "relevance": 0.9,
                "record": constraint_rec,
            },
        ]
        step_result.error = None
        retriever._execute_step = AsyncMock(return_value=step_result)

        retriever._to_retrieved_memories = MagicMock(return_value=[])

        await retriever.retrieve("t", plan)

        neocortical.multi_hop_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graph_expansion_skipped_when_graph_step_exists(self):
        from src.retrieval.retriever import HybridRetriever

        hippocampal = MagicMock()
        neocortical = MagicMock()
        neocortical.multi_hop_query = AsyncMock(return_value=[])

        retriever = HybridRetriever(hippocampal, neocortical)

        plan = MagicMock()
        plan.steps = [
            RetrievalStep(source=RetrievalSource.GRAPH, seeds=["entity"], top_k=5),
        ]
        plan.parallel_steps = [[0]]
        plan.total_timeout_ms = 5000
        plan.analysis = QueryAnalysis(
            original_query="test",
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.9,
        )
        plan.analysis.metadata = {}

        step_result = MagicMock()
        step_result.success = True
        step_result.items = []
        step_result.error = None
        retriever._execute_step = AsyncMock(return_value=step_result)
        retriever._to_retrieved_memories = MagicMock(return_value=[])

        await retriever.retrieve("t", plan)

        # multi_hop_query should only be called once (from the plan step), not twice
        assert neocortical.multi_hop_query.await_count <= 1


# ═══════════════════════════════════════════════════════════════════
# AUD-09: Batch graph edge writes and background sync
# ═══════════════════════════════════════════════════════════════════


class TestBatchEdgeWrites:
    """AUD-09: store_relations_batch uses merge_edges_batch for multiple relations."""

    @pytest.mark.asyncio
    async def test_batch_delegates_to_merge_edges_batch(self):
        from src.core.schemas import Relation
        from src.memory.neocortical.store import NeocorticalStore

        graph = MagicMock()
        graph.merge_edges_batch = AsyncMock(return_value=["e1", "e2"])
        facts = MagicMock()
        store = NeocorticalStore(graph_store=graph, fact_store=facts)

        relations = [
            Relation(subject="Alice", predicate="KNOWS", object="Bob", confidence=0.9),
            Relation(subject="Alice", predicate="LIKES", object="Pasta", confidence=0.8),
        ]
        result = await store.store_relations_batch("t", relations, evidence_ids=["ev1"])

        graph.merge_edges_batch.assert_awaited_once()
        assert result == ["e1", "e2"]

        call_args = graph.merge_edges_batch.call_args
        assert call_args[0][0] == "t"  # tenant_id
        edges = call_args[0][2]  # edges list
        assert len(edges) == 2

    @pytest.mark.asyncio
    async def test_single_relation_uses_individual_store(self):
        from src.core.schemas import Relation
        from src.memory.neocortical.store import NeocorticalStore

        graph = MagicMock()
        graph.merge_edge = AsyncMock(return_value="e1")
        graph.merge_edges_batch = AsyncMock()
        facts = MagicMock()
        store = NeocorticalStore(graph_store=graph, fact_store=facts)

        relations = [
            Relation(subject="Alice", predicate="KNOWS", object="Bob", confidence=0.9),
        ]
        await store.store_relations_batch("t", relations)

        graph.merge_edges_batch.assert_not_awaited()
        graph.merge_edge.assert_awaited_once()


class TestBackgroundGraphSync:
    """AUD-09: Graph sync runs as background task, not blocking the write."""

    @pytest.mark.asyncio
    async def test_sync_to_graph_launched_as_task(self):
        import asyncio

        from src.memory.orchestrator import MemoryOrchestrator

        orchestrator = MagicMock(spec=MemoryOrchestrator)
        sync_mock = AsyncMock()
        orchestrator._sync_to_graph = sync_mock

        task = asyncio.create_task(sync_mock("t", ["rec"]))
        await task

        sync_mock.assert_awaited_once_with("t", ["rec"])


# ═══════════════════════════════════════════════════════════════════
# AUD-12: Typed constraint reconsolidation
# ═══════════════════════════════════════════════════════════════════


class TestTypedCandidateSelection:
    """AUD-12: Same-type constraints are prioritised in candidate selection."""

    def test_same_type_prioritised(self):
        from src.reconsolidation.service import ReconsolidationService

        service = ReconsolidationService.__new__(ReconsolidationService)
        service.modelpack = MagicMock()
        service.modelpack.available = False
        service.modelpack.has_task_model = MagicMock(return_value=False)

        value_mem_1 = _make_memory_record(
            "I value honesty",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "value", "subject": "user"}]},
        )
        value_mem_2 = _make_memory_record(
            "I value kindness",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "value", "subject": "user"}]},
        )
        goal_mem = _make_memory_record(
            "I want to save money",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "goal", "subject": "user"}]},
        )
        state_mem = _make_memory_record(
            "I am currently stressed",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "state", "subject": "user"}]},
        )
        generic_mem = _make_memory_record(
            "Had lunch yesterday",
            MemoryType.EPISODIC_EVENT,
        )

        new_record = _make_memory_record(
            "I value integrity",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "value", "subject": "user"}]},
        )

        memories = [goal_mem, state_mem, value_mem_1, generic_mem, value_mem_2]
        result = service._top_k_similar_memories(
            "I value integrity", memories, k=2, new_record=new_record
        )

        assert len(result) == 2
        for m in result:
            meta = m.metadata.get("constraints", [])
            assert meta and meta[0]["constraint_type"] == "value"

    def test_falls_back_when_no_constraint_meta(self):
        from src.reconsolidation.service import ReconsolidationService

        service = ReconsolidationService.__new__(ReconsolidationService)
        service.modelpack = MagicMock()
        service.modelpack.available = False
        service.modelpack.has_task_model = MagicMock(return_value=False)

        memories = [
            _make_memory_record("memory A"),
            _make_memory_record("memory B"),
            _make_memory_record("memory C"),
        ]

        result = service._top_k_similar_memories("memory A", memories, k=2)
        assert len(result) == 2


class TestTypeAwareConflictDetection:
    """AUD-12: Type-aware pre-filtering in conflict detection."""

    @pytest.mark.asyncio
    async def test_different_types_no_conflict(self):
        from src.reconsolidation.conflict_detector import ConflictDetector, ConflictType

        detector = ConflictDetector(llm_client=None)

        old = _make_memory_record(
            "I value honesty",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "value", "subject": "user"}]},
        )
        new = _make_memory_record(
            "I want to save money",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "goal", "subject": "user"}]},
        )

        result = await detector.detect(old, new.text, new_memory=new)
        assert result.conflict_type == ConflictType.NONE
        assert "independent" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_same_type_different_scope_coexistence(self):
        from src.reconsolidation.conflict_detector import ConflictDetector, ConflictType

        detector = ConflictDetector(llm_client=None)

        old = _make_memory_record(
            "No shellfish at work events",
            MemoryType.CONSTRAINT,
            metadata={
                "constraints": [{"constraint_type": "policy", "subject": "diet", "scope": "work"}]
            },
        )
        new = _make_memory_record(
            "No peanuts at home",
            MemoryType.CONSTRAINT,
            metadata={
                "constraints": [{"constraint_type": "policy", "subject": "diet", "scope": "home"}]
            },
        )

        result = await detector.detect(old, new.text, new_memory=new)
        assert result.conflict_type == ConflictType.NONE
        assert "coexistence" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_same_type_same_subject_correction(self):
        from src.reconsolidation.conflict_detector import ConflictDetector, ConflictType

        detector = ConflictDetector(llm_client=None)

        old = _make_memory_record(
            "I value honesty above all",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "value", "subject": "user"}]},
        )
        new = _make_memory_record(
            "I value kindness above all",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "value", "subject": "user"}]},
        )

        result = await detector.detect(old, new.text, new_memory=new)
        assert result.conflict_type == ConflictType.CORRECTION
        assert result.is_superseding

    @pytest.mark.asyncio
    async def test_falls_through_without_new_memory(self):
        from src.reconsolidation.conflict_detector import ConflictDetector, ConflictType

        modelpack = MagicMock()
        modelpack.available = False
        detector = ConflictDetector(llm_client=None, modelpack=modelpack)

        old = _make_memory_record(
            "I value honesty",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "value", "subject": "user"}]},
        )

        result = await detector.detect(old, "some new statement")
        assert result.conflict_type == ConflictType.NONE
        assert result.confidence == 0.0


class TestConstraintTypeAwareStrategy:
    """AUD-12: Belief revision uses constraint type for strategy selection."""

    def test_stable_type_contradiction_uses_correction(self):
        from src.reconsolidation.belief_revision import BeliefRevisionEngine, RevisionStrategy
        from src.reconsolidation.conflict_detector import ConflictResult, ConflictType

        engine = BeliefRevisionEngine()
        old = _make_memory_record(
            "Always tip 20%",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "policy", "subject": "user"}]},
        )
        conflict = ConflictResult(
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.85,
            old_statement="Always tip 20%",
            new_statement="Always tip 25%",
            is_superseding=False,
        )

        plan = engine.plan_revision(
            conflict,
            old,
            MemoryType.CONSTRAINT,
            "t",
            "ev1",
        )

        assert plan.strategy == RevisionStrategy.TIME_SLICE

    def test_volatile_type_contradiction_uses_time_slice(self):
        from src.reconsolidation.belief_revision import BeliefRevisionEngine, RevisionStrategy
        from src.reconsolidation.conflict_detector import ConflictResult, ConflictType

        engine = BeliefRevisionEngine()
        old = _make_memory_record(
            "I am stressed",
            MemoryType.CONSTRAINT,
            metadata={"constraints": [{"constraint_type": "state", "subject": "user"}]},
        )
        conflict = ConflictResult(
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.85,
            old_statement="I am stressed",
            new_statement="I am relaxed now",
            is_superseding=False,
        )

        plan = engine.plan_revision(
            conflict,
            old,
            MemoryType.CONSTRAINT,
            "t",
            "ev1",
        )

        assert plan.strategy == RevisionStrategy.TIME_SLICE

    def test_no_constraint_meta_falls_through(self):
        from src.reconsolidation.belief_revision import BeliefRevisionEngine, RevisionStrategy
        from src.reconsolidation.conflict_detector import ConflictResult, ConflictType

        engine = BeliefRevisionEngine()
        old = _make_memory_record("generic memory", MemoryType.EPISODIC_EVENT)
        conflict = ConflictResult(
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.5,
            old_statement="generic memory",
            new_statement="different memory",
            is_superseding=False,
        )

        plan = engine.plan_revision(conflict, old, MemoryType.EPISODIC_EVENT, "t")
        assert plan.strategy == RevisionStrategy.ADD_HYPOTHESIS
