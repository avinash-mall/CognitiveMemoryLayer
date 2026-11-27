"""Unit tests for LoCoMo / LoCoMo-Plus root-cause fixes across ingestion, consolidation, retrieval, and QA."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from cml.eval.locomo import (
    _extract_answer,
    _parse_input_prompt_into_turns,
    _strip_cognitive_trigger_turns,
    format_cognitive_trigger_and_prompt,
)
from src.consolidation.windowed_consolidator import WindowedMicroSleepConsolidator
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.memory.hippocampal.store import HippocampalStore
from src.memory.neocortical.store import NeocorticalStore
from src.memory.working.models import ChunkType, SemanticChunk
from src.retrieval.augmenters import RetrievalAugmentersMixin
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.packet_builder import MemoryPacketBuilder
from src.retrieval.reality_monitor import CA1RealityMonitor
from src.retrieval.reranker import MemoryReranker
from src.storage.noop_stores import NoOpFactStore, NoOpGraphStore
from src.utils.embeddings import MockEmbeddingClient


def _rec(
    text: str,
    *,
    speaker: str = "",
    ts: datetime | None = None,
    source: MemorySource = MemorySource.USER_EXPLICIT,
    mem_type: MemoryType = MemoryType.EPISODIC_EVENT,
    extra_meta: dict | None = None,
) -> MemoryRecord:
    meta = dict(extra_meta or {})
    if speaker:
        meta["speaker"] = speaker
    return MemoryRecord(
        tenant_id="t1",
        context_tags=["conversation"],
        type=mem_type,
        text=text,
        metadata=meta,
        provenance=Provenance(source=source),
        timestamp=ts or datetime(2023, 7, 15, 14, 0, tzinfo=UTC),
        confidence=0.85,
    )


class TestEvalIngestionAndPromptFixes:
    def test_preserves_blip_photo_captions_and_multiline_turns(self):
        raw_prompt = (
            "DATE: 02:30 PM on 15 July, 2023\n"
            "CONVERSATION:\n"
            'Melanie said, "5 years already! Time flies!" and shared a photo of a bride and groom at their wedding.\n'
            'Nate said, "Congrats Joanna! How was it on the big screen?\n'
            '[shares a photo holding a videogame controller]" and shared a photo of a box with a controller inside.\n'
            "\nQuestion: How long have Mel and her husband been married?"
        )
        turns = _parse_input_prompt_into_turns(raw_prompt)
        assert len(turns) == 2
        assert "[Shared photo: a photo of a bride and groom at their wedding]" in turns[0][1]
        assert "Nate: Congrats Joanna!" in turns[1][1]
        assert "[Shared photo: a photo of a box with a controller inside]" in turns[1][1]

    def test_strips_cognitive_trigger_self_contamination_and_maps_speakers(self):
        sample = {
            "category": "Cognitive",
            "speaker_a": "Calvin",
            "speaker_b": "Dave",
            "input_prompt": (
                'Calvin said, "Ever since I broke my ankle falling off that electric scooter, I sold it."\n'
                'Dave said, "That sounds like a smart change."\n'
                'Calvin said, "I booked that trip to Lisbon right in the center so I can get around easily."'
            ),
            "trigger": "A: I booked that trip to Lisbon right in the center so I can get around easily.",
            "turn_timestamps": [
                "2023-08-01T10:00:00",
                "2023-08-01T10:00:00",
                "2023-11-24T10:00:00",
            ],
        }
        turns = _parse_input_prompt_into_turns(sample["input_prompt"])
        assert len(turns) == 3
        stripped, stripped_ts = _strip_cognitive_trigger_turns(
            sample, turns, sample["turn_timestamps"]
        )
        assert len(stripped) == 2
        assert len(stripped_ts) == 2
        assert all("Lisbon" not in c for _, c, _ in stripped)

        read_q, full_prompt = format_cognitive_trigger_and_prompt(sample, "# Context")
        assert read_q.startswith("Calvin: I booked that trip to Lisbon")
        assert "as Dave replying to Calvin" in full_prompt
        assert "Final Answer:" in full_prompt

    def test_extract_answer_always_strips_structured_cot_before_final_answer(self):
        # Even when Reasoning is shorter than Final Answer (>0.8 ratio that previously failed)
        raw = (
            "Reasoning: Found 3 camping locations across July and August.\n"
            "Final Answer: Melanie has camped at the beach, in the mountains, and in the forest with her family."
        )
        assert (
            _extract_answer(raw)
            == "Melanie has camped at the beach, in the mountains, and in the forest with her family."
        )


class TestProvenanceAndConsolidationFixes:
    @pytest.mark.asyncio
    async def test_user_chunks_get_user_explicit_provenance_not_inferred(self):
        mem_store = AsyncMock()
        mem_store.scan_texts_for_gate = AsyncMock(return_value=[])
        mem_store.upsert = AsyncMock(
            side_effect=lambda rc: _rec(rc.text, source=rc.provenance.source)
        )

        hstore = HippocampalStore(
            vector_store=mem_store,
            embedding_client=MockEmbeddingClient(dimensions=768),
        )
        user_chunk = SemanticChunk(
            id="c1",
            text="Caroline: I moved from Sweden 4 years ago.",
            chunk_type=ChunkType.STATEMENT,
            source_turn_id="turn_1",
            source_role="user",
            salience=0.8,
            timestamp=datetime(2023, 6, 9, tzinfo=UTC),
        )
        assistant_chunk = SemanticChunk(
            id="c2",
            text="Assistant summary note.",
            chunk_type=ChunkType.STATEMENT,
            source_turn_id="turn_2",
            source_role="assistant",
            salience=0.8,
            timestamp=datetime(2023, 6, 9, tzinfo=UTC),
        )
        stored, _, _, _ = await hstore.encode_batch("t1", [user_chunk, assistant_chunk])
        assert len(stored) == 2
        assert stored[0].provenance.source == MemorySource.USER_EXPLICIT
        assert stored[1].provenance.source == MemorySource.AGENT_INFERRED

    @pytest.mark.asyncio
    async def test_windowed_consolidator_preserves_multi_valued_facts_without_collision(self):
        recorded_keys: list[tuple[str, str]] = []

        class _CapturingFactStore(NoOpFactStore):
            async def upsert_fact(self, tenant_id, key, value, *args, **kwargs):
                recorded_keys.append((key, str(value)))
                return await super().upsert_fact(tenant_id, key, value, *args, **kwargs)

        ep_store = AsyncMock()
        ep1 = _rec(
            "Melanie: I love pottery and camping!",
            speaker="Melanie",
            extra_meta={"turn_idx": 0, "session_idx": 1},
        )
        ep_store.scan = AsyncMock(return_value=[ep1])
        ep_store.upsert = AsyncMock()
        ep_store.update = AsyncMock()

        mock_llm = AsyncMock()
        mock_llm.model = "test-multi-fact-model"
        mock_llm.complete = AsyncMock(
            return_value='{"facts": ['
            '{"subject": "Melanie", "predicate": "hobby", "value": "Pottery classes", "category": "preference", "turn_indices": [0]},'
            '{"subject": "Melanie", "predicate": "hobby", "value": "Family beach camping", "category": "preference", "turn_indices": [0]},'
            '{"subject": "Melanie", "predicate": "marital_status", "value": "Married 5 years", "category": "relationship", "turn_indices": [0]}'
            '], "hyperedges": [], "proustian_pointers": [], "chapter_gist": {}, "persona_deltas": []}'
        )

        neo = NeocorticalStore(NoOpGraphStore(), _CapturingFactStore())
        consolidator = WindowedMicroSleepConsolidator(
            episodic_store=ep_store,
            neocortical_store=neo,
            embedding_client=MockEmbeddingClient(dimensions=768),
            llm_client=mock_llm,
        )
        stats = await consolidator.consolidate_tenant("t1")
        assert stats["facts_stored"] == 3
        keys_only = [k for k, _ in recorded_keys]
        # Both multi-valued hobbies survive under distinct keys, while marital_status uses singleton key
        assert len([k for k in keys_only if k.startswith("melanie:preference:hobby_")]) == 2
        assert "melanie:relationship:marital_status" in keys_only


class TestTwoStageRetrievalAndRealityMonitoringFixes:
    @pytest.mark.asyncio
    async def test_hyde_judged_delivery_and_multi_angle_expansion(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(
            return_value="electric scooter ankle accident\nsold scooter walk to work\nmobility injury vacation\nI broke my ankle on a scooter."
        )
        mock_llm.complete_json = AsyncMock(
            return_value={
                "selected": [2, 1],
                "grounding": "supported",
                "swap_note": "",
                "bridge_query": "Caroline home country grandma",
            }
        )
        gen = HyDEGenerator(mock_llm)
        angles = await gen.expand_multi_angle_queries(
            "Calvin: I booked a central Lisbon hotel so I can get around.",
            is_cognitive=True,
        )
        assert len(angles) == 4

        verdict = await gen.evaluate_candidates_and_grounding(
            "Where did Caroline move from?",
            [
                "[2023-06-09] Caroline: Moved from my home country 4 years ago.",
                "[2023-06-27] Caroline: My grandma in Sweden gave me this necklace.",
            ],
        )
        assert verdict["selected_indices"] == [1, 0]
        assert verdict["grounding"] == "supported"
        assert verdict["bridge_query"] == "Caroline home country grandma"

    def test_llm_grounding_suppresses_false_presupposition_and_avoids_duplicate_speaker_prefix(
        self,
    ):
        m1 = RetrievedMemory(
            record=_rec(
                "Caroline: I moved from Sweden 4 years ago.",
                speaker="Caroline",
                ts=datetime(2023, 7, 15, tzinfo=UTC),
            ),
            relevance_score=0.88,
            retrieval_source="vector",
        )
        # Calendar month "July" matches timestamp 2023-07-15 -> zero CA1 mismatch
        v_cal = CA1RealityMonitor.inspect("What did Caroline say in July 2023?", [m1])
        assert v_cal.has_mismatch is False

        # Even if query contains an unmentioned proper noun, llm_grounding="supported" suppresses false warning
        builder = MemoryPacketBuilder()
        pkt = builder.build(
            [m1],
            "What did Caroline say about Stockholm?",
            llm_grounding={"grounding": "supported"},
        )
        assert pkt.warnings == []
        md = builder.to_llm_context(pkt)
        assert "(Caroline) Caroline:" not in md
        assert "- Caroline: I moved from Sweden 4 years ago." in md
        assert "[inferred]" not in md

        # And llm_grounding="unsupported_premise" emits calibrated alert
        pkt_unv = builder.build(
            [m1],
            "When did Caroline climb Mount Everest?",
            llm_grounding={"grounding": "unsupported_premise"},
        )
        assert any("Evidence-Grounding Alert" in w for w in pkt_unv.warnings)

    def test_entity_bridge_boost_ignores_speaker_hubs_and_boosts_rare_bridges(self):
        # Caroline appears in every candidate (hub), whereas "Sweden" bridges Angle 0 and Angle 1
        cands = [
            RetrievedMemory(
                record=_rec(f"Caroline: generic turn {i}", speaker="Caroline"),
                relevance_score=0.6,
                retrieval_source="vector",
            )
            for i in range(10)
        ]
        bridge_mem = RetrievedMemory(
            record=_rec(
                "Caroline: My grandma in Sweden gave me this necklace.", speaker="Caroline"
            ),
            relevance_score=0.62,
            retrieval_source="vector",
        )
        cands.append(bridge_mem)
        angle0 = [cands[0], bridge_mem]
        angle1 = [
            cands[1],
            RetrievedMemory(
                record=_rec("Sweden is in Northern Europe."),
                relevance_score=0.60,
                retrieval_source="vector",
            ),
        ]

        RetrievalAugmentersMixin._apply_entity_bridge_boost(
            cands, [angle0, angle1], "Where did Caroline move from?"
        )
        assert bridge_mem.relevance_score > 0.65
        assert "sweden" in (bridge_mem.record.metadata or {}).get("_bridge_entities", [])
        # Generic Caroline hub turns were NOT boosted
        assert cands[0].relevance_score == 0.6

    @pytest.mark.asyncio
    async def test_reranker_protects_contiguity_source(self):
        m_contig = RetrievedMemory(
            record=_rec("Melanie: Painting together!", speaker="Melanie"),
            relevance_score=0.74,
            retrieval_source="contiguity",
        )
        reranker = MemoryReranker()
        ranked = await reranker.rerank([m_contig], "What creative project besides pottery?")
        assert len(ranked) == 1
