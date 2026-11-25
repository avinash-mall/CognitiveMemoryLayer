"""SDK & Embedded tests for CML v3.0 Whole-Brain features (FTS5, Hypergraph, Feedback, Simulate)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cml import EmbeddedCognitiveMemoryLayer, EmbeddedConfig


@pytest.mark.asyncio
async def test_embedded_whole_brain_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "mock")
    monkeypatch.setenv("FEATURES__USE_LLM_ENABLED", "false")

    cfg = EmbeddedConfig(tenant_id="sdk-v3-tenant")
    cfg.database.database_url = "sqlite+aiosqlite:///:memory:"
    cfg.embedding.provider = "mock"

    async with EmbeddedCognitiveMemoryLayer(config=cfg) as memory:
        w1 = await memory.write(
            "Caroline fractured her ankle rock climbing last month and cannot hike steep trails.",
            metadata={"speaker": "Caroline"},
            timestamp=datetime(2023, 6, 15, tzinfo=UTC),
            media=[{"type": "image", "caption": "Walking boot cast on right ankle"}],
        )
        assert w1.success
        assert w1.memory_id is not None

        # Run Windowed Micro-Sleep Consolidation (deterministic System-1 path)
        cons = await memory.consolidate_windowed(use_llm=False)
        assert cons.windows_processed >= 1
        assert cons.episodes_consolidated >= 1

        # Read via Dentate Gyrus SQLite FTS5 + Dense + Hypergraph
        r_read = await memory.read("Walking boot cast ankle Caroline", max_results=10)
        assert r_read.total_count >= 1
        assert any("Walking boot cast" in m.text for m in r_read.memories)
        assert r_read.retrieval_meta is not None
        assert "neuromodulation" in r_read.retrieval_meta

        # Dopaminergic RPE negative feedback with explicit user correction
        fb = await memory.feedback(
            outcome_reward=-0.8,
            query="Should we book a steep mountain hike for Caroline?",
            memory_ids=[str(w1.memory_id)],
            correction_text="Never suggest strenuous hiking while Caroline is in a walking boot.",
        )
        assert fb.rpe_delta < 0
        assert fb.memories_marked_labile >= 1
        assert fb.procedural_skill_key is not None
        assert fb.correction_memory_id is not None

        # Hippocampal Forward Preplay simulation
        sim = await memory.simulate("Should we book a steep mountain hike for Caroline?")
        assert sim.risk_score > 0.0
        assert len(sim.alerts) >= 1 or len(sim.violated_constraints) >= 1
