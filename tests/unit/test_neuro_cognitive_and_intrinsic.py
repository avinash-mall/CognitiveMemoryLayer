"""Unit tests for the restored and upgraded Neuro-Cognitive & ActiveCML subsystems.

Covers:
1. Amygdala Somatic Markers (Damasio valence/arousal, flashbulb retention, affective resonance)
2. Prefrontal Baddeley Working Memory (active entity focus, coreference binding, zero-latency read overlay)
3. Thalamic/Amygdala Dual-Process Triage Gate (surprise index & System-1 fast-path vs System-2 LLM triage)
4. Rostral PFC (BA10) Prospective Memory (System-1 causal triggers & parent-episode ecphory)
5. Default Mode Network (DMN / mPFC) Theory-of-Mind mentalizing synthesis
6. CA1 Comparator & Orbitofrontal Reality Monitoring (false-premise / counterfactual detection)
7. Synaptic Co-Activation Resonance & Multiple-Trace Detail Shield in MemoryReranker
8. Sharp-Wave Ripple (SWR) Cross-Episode Link Induction in ConsolidationWorker
9. DB-backed Labile Reconsolidation & Belief-Evolution lineage
10. ActiveCML Intrinsic Memory (I-2 Logit Bias, I-3 Concurrency-Isolated Activation Steering Hooks,
    I-4 Synaptic KV-Prefix Hierarchy, I-5 Controller & Gating Unit, I-8 Dynamic LoRA Router)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.consolidation.clusterer import EpisodeCluster
from src.consolidation.worker import ConsolidationWorker
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import EntityMention, MemoryPacket, MemoryRecord, Provenance, RetrievedMemory
from src.intrinsic import (
    ActivationSteeringEngine,
    DynamicLoRARouter,
    IntrinsicController,
    IntrinsicDepth,
    LogitBiasSynthesizer,
    SynapticKVManager,
)
from src.memory.hippocampal.write_gate import WriteGate
from src.memory.prospective import ProspectiveMemoryEngine
from src.memory.short_term import ShortTermMemory
from src.memory.somatic import SomaticMarkerAnalyzer
from src.memory.working.models import ChunkType, SemanticChunk, WorkingMemoryState
from src.reconsolidation.conflict_detector import ConflictResult, ConflictType
from src.reconsolidation.service import ReconsolidationService
from src.retrieval.packet_builder import MemoryPacketBuilder
from src.retrieval.reality_monitor import CA1RealityMonitor
from src.retrieval.reranker import MemoryReranker
from src.retrieval.retriever import HybridRetriever


def _make_record(
    text: str,
    *,
    mem_type: MemoryType = MemoryType.EPISODIC_EVENT,
    confidence: float = 0.85,
    embedding: list[float] | None = None,
    entities: list[EntityMention] | None = None,
    metadata: dict | None = None,
    key: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id="tenant-neuro",
        context_tags=[],
        type=mem_type,
        text=text,
        key=key,
        confidence=confidence,
        importance=0.7,
        embedding=embedding or [0.25, 0.5, 0.75, 0.1],
        entities=entities or [],
        metadata=metadata or {},
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=datetime.now(UTC),
    )


class TestAmygdalaSomaticMarkers:
    def test_high_arousal_negative_event_slows_decay_and_boosts_importance(self):
        marker = SomaticMarkerAnalyzer.analyze(
            "I am extremely stressed and terrified about my emergency surgery tomorrow!"
        )
        assert marker.valence < -0.5
        assert marker.arousal >= 0.7
        assert marker.dominant_emotion in ("fear", "stress")
        assert marker.retention_modifier < 0.65
        assert marker.importance_boost > 0.1

    def test_positive_milestone_tagged_joy_or_excitement(self):
        marker = SomaticMarkerAnalyzer.analyze(
            "I am so thrilled and overjoyed that I got promoted and engaged!"
        )
        assert marker.valence > 0.6
        assert marker.arousal >= 0.7
        assert marker.dominant_emotion in ("joy", "excitement")

    def test_affective_resonance_matches_state_congruent_query(self):
        marker = SomaticMarkerAnalyzer.analyze("I am overwhelmed and stressed by the deadline.")
        meta = {"somatic_marker": marker.to_dict()}
        boost = SomaticMarkerAnalyzer.query_affective_resonance(
            "What is causing stress and worry right now?", meta
        )
        assert boost > 0.05
        assert SomaticMarkerAnalyzer.query_affective_resonance("What is 2 + 2?", meta) == 0.0


class TestBaddeleyWorkingMemoryAndOverlay:
    def test_episodic_buffer_binds_coreference_entities_across_turns(self):
        state = WorkingMemoryState(tenant_id="t1", user_id="u1", max_chunks=7)
        c1 = SemanticChunk(
            id="c1",
            text="Alice moved to Seattle for her job at Anthropic.",
            chunk_type=ChunkType.STATEMENT,
        )
        state.add_chunk(c1)
        assert "Alice" in state.active_entities
        assert "Seattle" in state.active_entities

        c2 = SemanticChunk(
            id="c2",
            text="She really loves living there and biking to work.",
            chunk_type=ChunkType.STATEMENT,
        )
        state.add_chunk(c2)
        focus = c2.metadata.get("working_memory_focus", [])
        assert "Alice" in focus or "Seattle" in focus
        assert any(e in c2.entities for e in ("Alice", "Seattle"))

    @pytest.mark.asyncio
    async def test_short_term_memory_zero_latency_search(self):
        stm = ShortTermMemory()
        await stm.ingest_turn("t1", "s1", "Melanie adopted a golden retriever puppy named Sunny.")
        hits = stm.search_working_memory("t1", "What puppy did Melanie adopt?")
        assert len(hits) >= 1
        chunk, score = hits[0]
        assert "Sunny" in chunk.text
        assert score >= 0.25


class TestThalamicSurpriseTriage:
    def test_high_surprise_vs_routine_greeting(self):
        gate = WriteGate()
        routine = SemanticChunk(
            id="1", text="Okay sounds good thanks", chunk_type=ChunkType.STATEMENT, salience=0.2
        )
        high_surprise = SemanticChunk(
            id="2",
            text="Actually I broke my ankle in Paris in 2024 and I am allergic to peanuts!",
            chunk_type=ChunkType.CONSTRAINT,
            salience=0.9,
        )
        s_low = gate.compute_surprise_index(
            routine, existing_memories=[{"text": "Okay sounds good thanks"}]
        )
        s_high = gate.compute_surprise_index(high_surprise, existing_memories=[])
        assert s_high >= 0.55
        assert s_low < 0.25


class TestBA10ProspectiveMemory:
    def test_system1_causal_implication_induction(self):
        triggers = ProspectiveMemoryEngine.induce_triggers(
            "I broke my ankle playing soccer last weekend and I am allergic to shellfish."
        )
        assert len(triggers) >= 2
        types = {t.trigger_type for t in triggers}
        assert "causal_constraint" in types
        strings = ProspectiveMemoryEngine.induce_implication_strings(
            "I broke my ankle playing soccer last weekend."
        )
        assert any("strenuous physical activities" in s for s in strings)

    @pytest.mark.asyncio
    async def test_prospective_hit_resolves_parent_episode_at_retrieval(self):
        parent_id = uuid4()
        parent_rec = _make_record("I fractured my knee skiing in Aspen.")
        parent_rec.id = parent_id

        prospective_rec = _make_record(
            "Physical limitation (fractured my knee): cannot run marathon.",
            metadata={"is_prospective_index": True, "parent_id": str(parent_id)},
        )

        hippocampal = MagicMock()
        hippocampal.store = MagicMock()
        hippocampal.store.get_by_ids_batch = AsyncMock(return_value=[parent_rec])
        neocortical = MagicMock()

        retriever = HybridRetriever(hippocampal, neocortical)
        all_results = [
            {
                "type": MemoryType.EPISODIC_EVENT.value,
                "source": "vector",
                "text": prospective_rec.text,
                "relevance": 0.78,
                "record": prospective_rec,
            }
        ]
        await retriever._resolve_prospective_parents("tenant-neuro", all_results)
        sources = {r["source"] for r in all_results}
        assert "prospective_parent" in sources
        assert any("fractured my knee skiing in Aspen" in r["text"] for r in all_results)


class TestCA1RealityMonitoringAndDMN:
    def test_ca1_detects_false_premise_presupposition(self):
        m1 = RetrievedMemory(
            record=_make_record("Melanie went camping at the beach with Caroline in June."),
            relevance_score=0.82,
            retrieval_source="vector",
        )
        m2 = RetrievedMemory(
            record=_make_record("Melanie loves painting sunrises and hiking."),
            relevance_score=0.71,
            retrieval_source="vector",
        )
        verdict = CA1RealityMonitor.inspect(
            "When did Melanie go camping in Yosemite with Caroline?", [m1, m2]
        )
        assert verdict.has_mismatch is True
        assert "Yosemite" in verdict.unverified_presuppositions
        assert "Melanie" in verdict.grounded_anchors
        assert any("Yosemite" in w for w in verdict.warnings)

    def test_dmn_synthesizes_theory_of_mind_context_for_mentalizing_query(self):
        somatic = SomaticMarkerAnalyzer.analyze(
            "I am super stressed and overwhelmed about my tenure deadline."
        )
        ep = RetrievedMemory(
            record=_make_record(
                "I am super stressed and overwhelmed about my tenure deadline.",
                metadata={"speaker": "Alice", "somatic_marker": somatic.to_dict()},
            ),
            relevance_score=0.85,
            retrieval_source="vector",
        )
        pref = RetrievedMemory(
            record=_make_record(
                "Alice loves calming jasmine tea and pottery classes.",
                mem_type=MemoryType.PREFERENCE,
                metadata={"speaker": "Alice"},
            ),
            relevance_score=0.80,
            retrieval_source="vector",
        )
        builder = MemoryPacketBuilder()
        query = "What gift would Alice appreciate to help with how she is feeling?"
        packet = builder.build([ep, pref], query)
        md = builder.to_llm_context(packet, max_tokens=2000)
        assert "## Theory-of-Mind & Psychological Context" in md
        assert "Alice" in md


class TestRerankerCoActivationAndMultipleTraceShield:
    @pytest.mark.asyncio
    async def test_coactivation_resonance_boosts_multi_prong_hit(self):
        single_hit = RetrievedMemory(
            record=_make_record("Single pathway memory", confidence=0.8),
            relevance_score=0.75,
            retrieval_source="vector",
        )
        converging_hit = RetrievedMemory(
            record=_make_record(
                "Multi-pathway memory",
                confidence=0.8,
                metadata={"_converging_sources": ["graph", "vector"], "_coactivation_count": 2},
            ),
            relevance_score=0.75,
            retrieval_source="vector",
        )
        reranker = MemoryReranker()
        ranked, breakdowns = await reranker.rerank_with_breakdown(
            [single_hit, converging_hit], "query"
        )
        assert ranked[0].record.text == "Multi-pathway memory"
        assert "coactivation_resonance_x2" in breakdowns[0]["notes"]

    @pytest.mark.asyncio
    async def test_multiple_trace_shield_preserves_episode_with_query_specific_details(self):
        gist = RetrievedMemory(
            record=_make_record(
                "prefers vegetarian food",
                mem_type=MemoryType.SEMANTIC_FACT,
                key="user:preference:diet",
            ),
            relevance_score=0.70,
            retrieval_source="facts",
        )
        episode = RetrievedMemory(
            record=_make_record(
                "I ordered the vegetarian platter at GreenZen in Kyoto",
                metadata={
                    "consolidated": True,
                    "consolidated_into_fact_key": "user:preference:diet",
                },
            ),
            relevance_score=0.88,
            retrieval_source="vector",
        )
        reranker = MemoryReranker()
        _, breakdown = await reranker.rerank_with_breakdown(
            [episode, gist], "What restaurant in Kyoto did I order the platter at?"
        )
        by_text = {b["text"]: b for b in breakdown}
        ep_notes = by_text["I ordered the vegetarian platter at GreenZen in Kyoto"]["notes"]
        assert "multiple_trace_detail_preserved" in ep_notes
        assert breakdown[0]["text"] == "I ordered the vegetarian platter at GreenZen in Kyoto"


class TestSWRSleepReplayAndLabileReconsolidation:
    @pytest.mark.asyncio
    async def test_swr_induces_cross_episode_graph_relations(self):
        episodic = MagicMock()
        neocortical = MagicMock()
        neocortical.facts = MagicMock()
        neocortical.store_relations_batch = AsyncMock(return_value=["e1"])

        worker = ConsolidationWorker(
            episodic_store=episodic, neocortical_store=neocortical, llm_client=None
        )
        ep1 = _make_record(
            "Alice joined AcmeCorp.",
            entities=[EntityMention(text="Alice", normalized="Alice", entity_type="PERSON")],
        )
        ep2 = _make_record(
            "AcmeCorp opened an office in Austin.",
            entities=[EntityMention(text="Austin", normalized="Austin", entity_type="LOCATION")],
        )
        cluster = EpisodeCluster(
            cluster_id=1, episodes=[ep1, ep2], common_entities=[], dominant_type="episodic_event"
        )
        induced = await worker._induce_swr_cross_episode_links("tenant-neuro", [cluster])
        assert induced == 1
        neocortical.store_relations_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reconsolidation_toggles_db_labile_and_records_belief_evolution(self):
        store = MagicMock()
        store.update = AsyncMock(return_value=None)
        store.upsert = AsyncMock(
            side_effect=lambda r: SimpleNamespace(id=uuid4(), metadata=r.metadata)
        )
        store.get_by_id = AsyncMock(return_value=None)

        svc = ReconsolidationService(memory_store=store, llm_client=None)
        svc.conflict_detector.detect = AsyncMock(
            return_value=ConflictResult(
                conflict_type=ConflictType.CORRECTION,
                confidence=0.95,
                old_statement="I live in Boston.",
                new_statement="I live in Denver now.",
                is_superseding=True,
                reasoning="Moved cities",
            )
        )
        old_mem = _make_record("I live in Boston.", mem_type=MemoryType.SEMANTIC_FACT)
        res = await svc.process_turn(
            tenant_id="t1",
            scope_id="u1",
            turn_id="turn-9",
            user_message="Actually, I live in Denver now.",
            assistant_response="Updated your city to Denver.",
            retrieved_memories=[old_mem],
        )
        assert res.conflicts_found == 1
        # Verify DB labile=True and labile=False were called on store.update
        labile_calls = [
            c.args[1]
            for c in store.update.call_args_list
            if len(c.args) > 1 and "labile" in c.args[1]
        ]
        assert {"labile": True} in labile_calls
        assert {"labile": False} in labile_calls
        # Verify belief_evolution metadata attached to new record
        upserted_rec = store.upsert.call_args.args[0]
        assert "belief_evolution" in upserted_rec.metadata
        assert upserted_rec.metadata["belief_evolution"]["previous_belief"] == "I live in Boston."


class TestActiveCMLIntrinsicSubsystem:
    def test_controller_synthesizes_all_four_intrinsic_depths(self):
        rec1 = _make_record(
            "Melanie visited Rome in 2023.",
            embedding=[0.4, -0.2, 0.8, 0.1],
            entities=[
                EntityMention(text="Melanie", normalized="melanie", entity_type="PERSON"),
                EntityMention(text="Rome", normalized="rome", entity_type="LOCATION"),
            ],
            key="fact:melanie:trip",
        )
        rec2 = _make_record(
            "Melanie loves oil painting.",
            mem_type=MemoryType.PREFERENCE,
            embedding=[0.3, 0.1, 0.7, 0.2],
            key="pref:melanie:hobby",
        )
        packet = MemoryPacket(
            query="When did Melanie visit Paris with her painting group?",
            facts=[RetrievedMemory(record=rec1, relevance_score=0.88, retrieval_source="facts")],
            preferences=[
                RetrievedMemory(record=rec2, relevance_score=0.80, retrieval_source="vector")
            ],
            ranked_memories=[
                RetrievedMemory(record=rec1, relevance_score=0.88, retrieval_source="facts"),
                RetrievedMemory(record=rec2, relevance_score=0.80, retrieval_source="vector"),
            ],
        )

        controller = IntrinsicController(
            logit_synthesizer=LogitBiasSynthesizer(),
            activation_engine=ActivationSteeringEngine(default_layers=[0, 1]),
            synaptic_kv=SynapticKVManager(),
            lora_router=DynamicLoRARouter(),
        )
        spec = controller.compute_spec(
            packet, tenant_id="t1", intent="temporal_query", target_hidden_dim=8
        )
        assert spec.enabled is True
        assert spec.gate_strength > 0.3
        assert IntrinsicDepth.LOGIT.value in spec.selected_depths
        assert IntrinsicDepth.ACTIVATION.value in spec.selected_depths
        assert IntrinsicDepth.SYNAPTIC_KV.value in spec.selected_depths
        assert IntrinsicDepth.WEIGHT_LORA.value in spec.selected_depths
        assert "Melanie" in spec.boosted_terms or "Rome" in spec.boosted_terms
        # False premise "Paris" detected by CA1 Reality Monitor -> negatively biased!
        assert "Paris" in spec.suppressed_terms
        assert any(v < 0 for v in spec.logit_bias.values())
        assert len(spec.steering_vector) == 8
        assert spec.canonical_kv_prefix is not None
        assert "<synaptic_memory_kv_prefix>" in spec.canonical_kv_prefix
        assert spec.lora_adapter_id == "cml-lora-temporal-v1"
        openai_kwargs = spec.to_openai_extra_kwargs()
        assert "logit_bias" in openai_kwargs

    @pytest.mark.asyncio
    async def test_activation_steering_hook_is_concurrency_isolated(self):
        torch = pytest.importorskip("torch")

        class DummyLayer(torch.nn.Module):
            def forward(self, x):
                return (x,)

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([DummyLayer(), DummyLayer()])

            def forward(self, x):
                out = x
                for layer in self.layers:
                    out = layer(out)[0]
                return out

        model = DummyModel()
        engine = ActivationSteeringEngine(default_layers=[0])
        x = torch.zeros((1, 2, 4), dtype=torch.float32)

        async def _steered_call():
            with engine.steer_model(model, [1.0, 0.0, 0.0, 0.0], scale=0.5, layer_indices=[0]):
                await asyncio.sleep(0.01)
                return model(x)

        async def _unsteered_call():
            await asyncio.sleep(0.005)
            return model(x)

        steered_out, unsteered_out = await asyncio.gather(_steered_call(), _unsteered_call())
        assert float(steered_out[0, 0, 0]) == pytest.approx(0.5, rel=1e-3)
        assert float(unsteered_out.abs().sum()) == 0.0


class TestBjorkDualStrengthEcphory:
    @pytest.mark.asyncio
    async def test_dormant_silent_memory_reactivates_only_on_high_similarity_cue(self):
        from contextlib import asynccontextmanager

        from src.core.enums import MemoryStatus
        from src.storage.postgres import PostgresMemoryStore

        dormant_model = SimpleNamespace(
            id=uuid4(),
            tenant_id="tenant-neuro",
            context_tags=[],
            source_session_id=None,
            agent_id=None,
            namespace=None,
            type=MemoryType.EPISODIC_EVENT.value,
            text="I buried the spare house key under the blue flowerpot.",
            key=None,
            embedding=[0.5, 0.5, 0.5, 0.5],
            entities=[],
            relations=[],
            meta={"silenced_at": "2026-01-01T00:00:00+00:00"},
            timestamp=datetime.now(UTC),
            written_at=datetime.now(UTC),
            valid_from=None,
            valid_to=None,
            confidence=0.8,
            importance=0.6,
            access_count=1,
            last_accessed_at=None,
            decay_rate=0.01,
            status=MemoryStatus.SILENT.value,
            labile=False,
            provenance={"source": MemorySource.USER_EXPLICIT.value, "evidence_refs": []},
            version=1,
            supersedes_id=None,
            content_hash="abc",
        )

        mock_session = MagicMock()
        # 1. Weak cue (similarity 0.55 < 0.75) -> dormant memory stays hidden
        mock_session.execute = AsyncMock(return_value=[(dormant_model, 0.55)])
        mock_session.commit = AsyncMock()

        @asynccontextmanager
        async def _session_factory():
            yield mock_session

        store = PostgresMemoryStore(_session_factory)
        weak_results = await store.vector_search("tenant-neuro", [0.5, 0.5, 0.5, 0.5], top_k=5)
        assert len(weak_results) == 0
        assert dormant_model.status == MemoryStatus.SILENT.value

        # 2. Specific cue (similarity 0.88 >= 0.75) -> triggers Ecphory reactivation!
        mock_session.execute = AsyncMock(return_value=[(dormant_model, 0.88)])
        strong_results = await store.vector_search("tenant-neuro", [0.5, 0.5, 0.5, 0.5], top_k=5)
        assert len(strong_results) == 1
        assert dormant_model.status == MemoryStatus.ACTIVE.value
        assert "ecphory_reactivated_at" in dormant_model.meta
        mock_session.commit.assert_awaited_once()


def test_feature_flags_use_llm_enabled_by_default(monkeypatch):
    monkeypatch.delenv("FEATURES__USE_LLM_ENABLED", raising=False)
    from src.core.config import FeatureFlags

    flags = FeatureFlags()
    assert flags.use_llm_enabled is True


@pytest.mark.asyncio
async def test_orchestrator_create_graceful_fallback_without_cloud_api_key(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import src.memory.orchestrator as orch_mod
    from src.core.config import FeatureFlags, LLMInternalSettings, get_settings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    real_settings = get_settings()
    patched = real_settings.model_copy(
        update={
            "features": FeatureFlags(use_llm_enabled=True, cached_embeddings_enabled=False),
            "llm_internal": LLMInternalSettings(
                provider="openai",
                model="gpt-4o-mini",
                api_key=None,
                base_url=None,
                extra_body=None,
            ),
        }
    )
    monkeypatch.setattr("src.core.config.get_settings", lambda: patched)
    monkeypatch.setattr(orch_mod, "get_embedding_client", lambda: MagicMock())
    monkeypatch.setattr("src.storage.neo4j.initialize_graph_schema", AsyncMock())

    db_manager = SimpleNamespace(pg_session=MagicMock(), neo4j_driver=MagicMock(), redis=None)
    orch = await orch_mod.MemoryOrchestrator.create(db_manager)
    assert orch.reconsolidation.fact_extractor is None
    assert orch.hippocampal.unified_extractor is None


def test_ca1_skips_hypothetical_and_matches_nickname_prefix():
    from src.retrieval.reality_monitor import CA1RealityMonitor

    memories = [
        _make_record("Caroline: I collect classic children's books and educational stories."),
        _make_record("Melanie: 5 years married already! Time flies."),
    ]
    # Hypothetical common-sense query with unmentioned proper noun ("Dr. Seuss") must NOT trigger alert
    v_hypo = CA1RealityMonitor.inspect(
        "Would Caroline likely have Dr. Seuss books on her bookshelf?", memories
    )
    assert v_hypo.has_mismatch is False
    assert v_hypo.warnings == []

    # Nickname prefix ("Mel" -> "Melanie") must NOT trigger alert
    v_nick = CA1RealityMonitor.inspect("How long have Mel and her husband been married?", memories)
    assert v_nick.has_mismatch is False
    assert v_nick.warnings == []


@pytest.mark.asyncio
async def test_cognitive_hyde_generates_three_cues():
    from unittest.mock import AsyncMock

    from src.retrieval.hyde import HyDEGenerator

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        "1. Saying no to extra commitments reduced my daily stress.\n"
        "2. Setting firm personal boundaries prevents burnout.\n"
        "3. Overcommitting to volunteer work always leaves me exhausted."
    )
    gen = HyDEGenerator(mock_llm)
    cues = await gen.generate_cognitive_cues(
        "A: I ended up volunteering for that project, and now I'm totally overwhelmed."
    )
    assert len(cues) == 3
    assert "Saying no" in cues[0]


def test_ca1_detects_speaker_swap_from_metadata_and_skips_cognitive_cues():
    from src.retrieval.reality_monitor import CA1RealityMonitor

    # Real Postgres records store speaker in metadata and/or "Speaker said, ..."
    memories = [
        _make_record(
            'Melanie said, "I ran a charity race for mental health last weekend!"',
            metadata={"speaker": "Melanie"},
        ),
        _make_record(
            'Melanie said, "That charity race taught me self-care is really important."',
            metadata={"speaker": "Melanie"},
        ),
    ]
    v_swap = CA1RealityMonitor.inspect(
        "What did Caroline realize after her charity race?", memories
    )
    assert v_swap.has_mismatch is True
    assert any("speaker_swap:Caroline->Melanie" in c for c in v_swap.contradictions)

    # Cognitive cue dialogue statements must never trigger CA1 warnings
    v_cog = CA1RealityMonitor.inspect(
        "A: I ended up volunteering for that project, and now I'm totally overwhelmed.",
        memories,
    )
    assert v_cog.has_mismatch is False
    assert v_cog.warnings == []


@pytest.mark.asyncio
async def test_bridge_query_and_temporal_chronological_ordering():
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock

    from src.retrieval.hyde import HyDEGenerator
    from src.retrieval.packet_builder import MemoryPacketBuilder

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "Caroline home country Sweden"
    gen = HyDEGenerator(mock_llm)
    bq = await gen.generate_bridge_query(
        "Where did Caroline move from 4 years ago?",
        ["Caroline: I moved from my home country 4 years ago."],
    )
    assert bq == "Caroline home country Sweden"

    mock_llm.complete.return_value = "NONE"
    assert await gen.generate_bridge_query("What color?", ["It is blue."]) is None

    # Chronological sorting on temporal queries
    late_rec = _make_record("Later event")
    late_rec.timestamp = datetime(2023, 6, 15, tzinfo=UTC)
    early_rec = _make_record("Earlier event")
    early_rec.timestamp = datetime(2023, 5, 1, tzinfo=UTC)
    late = RetrievedMemory(
        record=late_rec,
        relevance_score=0.95,
        retrieval_source="vector",
    )
    early = RetrievedMemory(
        record=early_rec,
        relevance_score=0.80,
        retrieval_source="vector",
    )
    builder = MemoryPacketBuilder()
    pkt = builder.build([late, early], "When did the events happen?")
    md = builder.to_llm_context(pkt)
    assert "Chronological Order" in md
    assert md.index("Earlier event") < md.index("Later event")


@pytest.mark.asyncio
async def test_cognitive_cue_selector_returns_valid_indices():
    from unittest.mock import AsyncMock

    from src.retrieval.hyde import HyDEGenerator

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "3, 1"
    gen = HyDEGenerator(mock_llm)
    chosen = await gen.select_cognitive_cues(
        "A: I booked a hotel right next to the conference.",
        ["Random chat", "Weather talk", "Running out of gas made me never go below 1/4 tank."],
    )
    assert chosen == [2, 0]
