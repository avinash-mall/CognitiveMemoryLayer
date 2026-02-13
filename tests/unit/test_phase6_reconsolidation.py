"""Unit tests for Phase 6: reconsolidation (labile tracker, conflict detector, belief revision)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance
from src.reconsolidation.belief_revision import (
    BeliefRevisionEngine,
    RevisionStrategy,
)
from src.reconsolidation.conflict_detector import (
    ConflictDetector,
    ConflictResult,
    ConflictType,
)
from src.reconsolidation.labile_tracker import (
    LabileStateTracker,
)


def _make_memory(text: str, key: str | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id="t",
        context_tags=[],
        type=MemoryType.PREFERENCE,
        text=text,
        key=key,
        confidence=0.8,
        importance=0.5,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=datetime.now(UTC),
    )


class TestLabileStateTracker:
    @pytest.mark.asyncio
    async def test_mark_labile_and_get_session(self):
        tracker = LabileStateTracker(labile_duration_seconds=60, max_sessions_per_scope=5)
        mid = uuid4()
        session = await tracker.mark_labile(
            "tenant1",
            "user1",
            "turn1",
            [mid],
            "query",
            ["retrieved text"],
            [0.9],
            [0.8],
        )
        assert session.tenant_id == "tenant1"
        assert session.turn_id == "turn1"
        assert mid in session.memories
        assert session.memories[mid].relevance_score == 0.9

        got = await tracker.get_session("tenant1", "user1", "turn1")
        assert got is not None
        assert got.turn_id == "turn1"

    @pytest.mark.asyncio
    async def test_release_labile(self):
        tracker = LabileStateTracker(labile_duration_seconds=60)
        mid = uuid4()
        await tracker.mark_labile("t", "u", "turn1", [mid], "q", ["t1"], [0.5], [0.5])
        await tracker.release_labile("t", "u", "turn1")
        session = await tracker.get_session("t", "u", "turn1")
        assert session is None or len(session.memories) == 0

    @pytest.mark.asyncio
    async def test_get_labile_memories(self):
        tracker = LabileStateTracker(labile_duration_seconds=300)
        mid = uuid4()
        await tracker.mark_labile("t", "u", "turn1", [mid], "q", ["text"], [0.8], [0.7])
        labile = await tracker.get_labile_memories("t", "u", "turn1")
        assert len(labile) == 1
        assert labile[0].memory_id == mid


class TestConflictDetector:
    @pytest.mark.asyncio
    async def test_fast_correction_marker(self):
        detector = ConflictDetector(llm_client=None)
        old = _make_memory("I like coffee.")
        result = await detector.detect(old, "Actually, I prefer tea now.")
        assert result.conflict_type == ConflictType.CORRECTION
        assert result.is_superseding is True
        assert result.confidence > 0.8

    @pytest.mark.asyncio
    async def test_fast_no_conflict(self):
        detector = ConflictDetector(llm_client=None)
        old = _make_memory("I have a dog.")
        result = await detector.detect(old, "The weather is nice today.")
        assert result.conflict_type == ConflictType.NONE

    @pytest.mark.asyncio
    async def test_fast_preference_temporal(self):
        """Preference-to-preference with topic overlap is TEMPORAL_CHANGE (fast path)."""
        detector = ConflictDetector(llm_client=None)
        old = _make_memory("I prefer black coffee.")
        result = await detector.detect(old, "I prefer milk coffee.")
        assert result.conflict_type == ConflictType.TEMPORAL_CHANGE
        assert result.is_superseding is True


class TestBeliefRevisionEngine:
    def test_plan_reinforcement(self):
        engine = BeliefRevisionEngine()
        old = _make_memory("I like coffee.", key="pref:drink")
        conflict = ConflictResult(
            conflict_type=ConflictType.NONE,
            confidence=0.9,
            old_statement=old.text,
            new_statement="I still like coffee.",
            reasoning="Consistent",
        )
        plan = engine.plan_revision(conflict, old, MemoryType.PREFERENCE, "t", "ev1")
        assert plan.strategy == RevisionStrategy.REINFORCE
        assert len(plan.operations) == 1
        assert plan.operations[0].op_type.value == "reinforce"
        assert plan.operations[0].patch is not None
        assert plan.operations[0].patch["confidence"] > old.confidence

    def test_plan_correction(self):
        engine = BeliefRevisionEngine()
        old = _make_memory("I like coffee.", key="pref:drink")
        conflict = ConflictResult(
            conflict_type=ConflictType.CORRECTION,
            confidence=0.95,
            old_statement=old.text,
            new_statement="I prefer tea.",
            is_superseding=True,
            reasoning="User corrected",
        )
        plan = engine.plan_revision(conflict, old, MemoryType.PREFERENCE, "t", "ev1")
        assert plan.strategy == RevisionStrategy.TIME_SLICE
        assert len(plan.operations) == 2
        assert plan.operations[0].patch is not None
        assert plan.operations[0].patch.get("status") == "archived"
        assert plan.operations[1].new_record is not None
        assert plan.operations[1].new_record.text == "I prefer tea."

    def test_plan_temporal_time_slice(self):
        engine = BeliefRevisionEngine()
        old = _make_memory("I prefer coffee.", key="pref:drink")
        conflict = ConflictResult(
            conflict_type=ConflictType.TEMPORAL_CHANGE,
            confidence=0.85,
            old_statement=old.text,
            new_statement="I prefer tea.",
            is_superseding=True,
            reasoning="Preference changed",
        )
        plan = engine.plan_revision(conflict, old, MemoryType.PREFERENCE, "t", "ev1")
        assert plan.strategy == RevisionStrategy.TIME_SLICE
        assert len(plan.operations) == 2
        assert "valid_to" in (plan.operations[0].patch or {})
        assert plan.operations[1].new_record is not None
        assert plan.operations[1].new_record.text == "I prefer tea."
