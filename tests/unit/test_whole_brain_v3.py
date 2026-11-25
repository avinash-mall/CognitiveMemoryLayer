"""Unit tests for CML v3.0 Whole-Brain Architecture subsystems."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.consolidation.windowed_consolidator import WindowedMicroSleepConsolidator
from src.extraction.temporal_resolver import extract_event_info, resolve_temporal_references
from src.memory.dmn import PersistentDMNStore
from src.memory.neocortical.hypergraph import CorticalHypergraphEngine, HyperedgeEngram
from src.memory.neuromodulation import NeuromodulatoryController
from src.memory.procedural import ProceduralMemoryEngine
from src.memory.sensory.perirhinal import PerirhinalSensoryStream
from src.retrieval.cross_encoder import CrossEncoderScorer
from src.retrieval.temporal_engine import HippocampalTimeCellEngine


def test_hippocampal_time_cells_granularity_exact() -> None:
    anchor = datetime(2023, 5, 25, 14, 0, tzinfo=UTC)

    info_month = extract_event_info("I visited the gallery last month.", anchor)
    assert info_month is not None
    assert info_month["granularity"] == "month"
    assert info_month["display"] == "April 2023"

    info_year = extract_event_info("We relocated from Boston last year.", anchor)
    assert info_year is not None
    assert info_year["granularity"] == "year"
    assert info_year["display"] == "2022"

    info_day = extract_event_info("I attended the support group yesterday.", anchor)
    assert info_day is not None
    assert info_day["granularity"] == "day"
    assert info_day["display"] == "2023-05-24"

    # Vague phrases should not stamp fake exact event_date displays
    info_vague = extract_event_info("I read a good novel recently.", anchor)
    assert info_vague is None
    raw_refs = resolve_temporal_references("I read a good novel recently.", anchor)
    assert len(raw_refs) == 1
    assert raw_refs[0]["granularity"] == "vague"


def test_temporal_engine_builds_chronological_deltas() -> None:
    class _StubRec:
        def __init__(self, text: str, ts: datetime, meta: dict) -> None:
            self.text = text
            self.timestamp = ts
            self.metadata = meta

    eps = [
        _StubRec(
            "Melanie exhibited her pottery.",
            datetime(2023, 8, 10),
            {"speaker": "Melanie", "event_date": "2023-08-09", "event_date_display": "2023-08-09"},
        ),
        _StubRec(
            "Melanie started taking pottery classes.",
            datetime(2023, 2, 15),
            {
                "speaker": "Melanie",
                "event_date": "2023-01-01",
                "event_date_display": "January 2023",
            },
        ),
    ]
    block = HippocampalTimeCellEngine.build_timeline_block(eps)
    assert "## Computed Event Timeline & Calendar Deltas" in block
    assert "January 2023" in block
    assert "2023-08-09" in block
    assert "Overall span" in block


@pytest.mark.asyncio
async def test_cortical_hypergraph_spreading_activation_and_plasticity() -> None:
    engine = CorticalHypergraphEngine()
    tid = "tenant-hyper"
    e1 = HyperedgeEngram(
        engram_key="h1",
        tenant_id=tid,
        event_summary="Caroline adopted a rescue greyhound named Bolt in Seattle",
        actors=["Caroline"],
        entities=["Bolt", "greyhound"],
        location="Seattle",
        prospective_cues=["weekend travel accommodation pet boarding"],
        episode_ids=["11111111-1111-1111-1111-111111111111"],
    )
    e2 = HyperedgeEngram(
        engram_key="h2",
        tenant_id=tid,
        event_summary="Bolt won agility ribbon at Green Lake Park",
        actors=["Caroline"],
        entities=["Bolt", "Green Lake Park"],
        episode_ids=["22222222-2222-2222-2222-222222222222"],
    )
    await engine.upsert_engrams(tid, [e1, e2])

    ep_scores, engrams = await engine.spreading_activation(
        tid, "Where did Caroline's rescue dog compete?", seed_entities=["Caroline"]
    )
    assert len(ep_scores) == 2
    assert len(engrams) == 2
    ep_ids = {eid for eid, _ in ep_scores}
    assert "22222222-2222-2222-2222-222222222222" in ep_ids

    mod_count = await engine.modulate_synaptic_weights(
        tid, ["22222222-2222-2222-2222-222222222222"], delta=0.35
    )
    assert mod_count == 1


def test_neuromodulation_ach_and_ne_gain() -> None:
    w_state = NeuromodulatoryController.for_write(
        "I fractured my ankle rock climbing in Moab!",
        recent_texts=["Let's review quarterly spreadsheets."],
    )
    assert w_state.ach_mode == "encoding"
    assert w_state.ach_level >= 0.6

    r_cog = NeuromodulatoryController.for_read(
        "Want to try that steep scramble trail this weekend?",
        routing_intent="cognitive",
        is_cognitive_cue=True,
    )
    r_single = NeuromodulatoryController.for_read(
        "What city does Alice live in?",
        routing_intent="single_hop",
        is_cognitive_cue=False,
    )
    assert r_cog.ach_mode == "retrieval"
    assert r_cog.recommended_candidate_pool > r_single.recommended_candidate_pool


def test_cross_encoder_mock_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "mock")
    scorer = CrossEncoderScorer()
    scores = scorer.score_pairs(
        "What instrument does Melanie play?",
        [
            "Melanie plays the clarinet in a community orchestra.",
            "The weather in Tokyo was rainy.",
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]


@pytest.mark.asyncio
async def test_dmn_persistent_profiles_and_perirhinal_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "mock")
    dmn = PersistentDMNStore()
    updated = await dmn.apply_deltas(
        "t-dmn",
        [
            {
                "speaker": "Melanie",
                "traits": ["empathetic", "creative"],
                "boundaries_fears": ["severe claustrophobia in caves"],
                "career_identity": ["art therapist"],
            }
        ],
    )
    assert updated == 1
    lines = await dmn.get_formatted_profiles("t-dmn", "Would Melanie enjoy caving?")
    assert len(lines) == 1
    assert "claustrophobia" in lines[0]

    descs = await PerirhinalSensoryStream.extract_descriptors(
        [{"type": "image", "caption": "Handmade blue ceramic vase on pottery wheel"}]
    )
    assert len(descs) == 1
    enriched = PerirhinalSensoryStream.enrich_content_with_sensory("Look what I made!", descs)
    assert "Handmade blue ceramic vase" in enriched


@pytest.mark.asyncio
async def test_procedural_memory_upsert_and_retrieval() -> None:
    proc = ProceduralMemoryEngine()
    await proc.upsert_skill(
        "t-proc",
        "deploy database migration to production",
        steps=["take pg_dump snapshot", "run alembic upgrade head"],
        pitfalls=["never drop columns without dual-read window"],
        weight_delta=0.25,
    )
    skills = await proc.retrieve_skills("t-proc", "how to run database migration safely")
    assert len(skills) == 1
    line = skills[0].format_playbook_line()
    assert "alembic upgrade head" in line
    assert "Avoid:" in line


@pytest.mark.asyncio
async def test_windowed_consolidator_llm_json_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "mock")
    monkeypatch.setenv("CML_WINDOW_CACHE_DISABLED", "1")
    from cml import EmbeddedCognitiveMemoryLayer, EmbeddedConfig

    class _StubLLM:
        model = "stub-window-llm"

        async def complete(self, prompt: str, **kwargs) -> str:
            return """{
              "facts": [{"subject": "Alice", "predicate": "allergy", "value": "peanuts", "category": "state", "turn_indices": [0], "event_date_display": ""}],
              "hyperedges": [{"actors": ["Alice"], "entities": ["peanuts", "Thai restaurant"], "event_summary": "Alice had an allergic reaction to peanuts", "temporal_anchor": "2023-05-01", "location": "Bangkok Kitchen", "causal_state": "severe peanut allergy", "prospective_cues": ["choosing Asian dinner spot or pad thai"], "turn_indices": [0]}],
              "proustian_pointers": [{"speaker": "Alice", "turn_index": 0, "implicit_lesson_or_boundary": "Severe peanut allergy", "future_cue_scenarios": ["Let's order pad thai for dinner tonight"]}],
              "chapter_gist": {"title": "Alice's Food Allergy", "summary": "Alice shared her severe peanut allergy experience.", "participants": ["Alice"]},
              "persona_deltas": [{"speaker": "Alice", "traits": ["cautious diner"], "boundaries_fears": ["peanut anaphylaxis"]}]
            }"""

    cfg = EmbeddedConfig(tenant_id="win-test")
    cfg.database.database_url = "sqlite+aiosqlite:///:memory:"
    cfg.embedding.provider = "mock"
    async with EmbeddedCognitiveMemoryLayer(config=cfg) as mem:
        await mem.write(
            "I ended up in urgent care after eating crushed peanuts in a sauce.",
            metadata={"speaker": "Alice"},
            timestamp=datetime(2023, 5, 1, tzinfo=UTC),
        )
        orch = mem._orchestrator
        consolidator = WindowedMicroSleepConsolidator(
            episodic_store=orch.hippocampal.store,
            neocortical_store=orch.neocortical,
            embedding_client=orch.hippocampal.embeddings,
            llm_client=_StubLLM(),
            dmn_store=orch.dmn_store,
        )
        stats = await consolidator.consolidate_tenant("win-test")
        assert stats["windows_processed"] == 1
        assert stats["facts_stored"] >= 1
        assert stats["hyperedges_stored"] == 1
        assert stats["proustian_pointers_stored"] == 1
        assert stats["chapters_stored"] == 1
        assert stats["dmn_profiles_updated"] == 1
