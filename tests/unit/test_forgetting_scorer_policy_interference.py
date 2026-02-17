"""Unit tests for active forgetting (scorer, policy, interference)."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance
from src.forgetting.actions import ForgettingAction, ForgettingPolicyEngine
from src.forgetting.interference import InterferenceDetector
from src.forgetting.scorer import (
    RelevanceScore,
    RelevanceScorer,
    RelevanceWeights,
    ScorerConfig,
)


def _make_record(
    text: str = "test",
    importance: float = 0.5,
    confidence: float = 0.8,
    access_count: int = 0,
    memory_type: MemoryType = MemoryType.EPISODIC_EVENT,
    days_ago: float = 0,
) -> MemoryRecord:
    ts = datetime.now(UTC) - timedelta(days=days_ago)
    return MemoryRecord(
        id=uuid4(),
        tenant_id="t",
        context_tags=[],
        type=memory_type,
        text=text,
        confidence=confidence,
        importance=importance,
        access_count=access_count,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=ts,
        embedding=[0.1] * 8,
        entities=[],
    )


class TestRelevanceWeights:
    def test_validate_ok(self):
        w = RelevanceWeights()
        w.validate()

    def test_validate_fails_when_not_sum_to_one(self):
        w = RelevanceWeights(importance=0.5, recency=0.5)
        with pytest.raises(AssertionError, match="sum to 1.0"):
            w.validate()


class TestRelevanceScorer:
    def test_score_high_importance_suggests_keep(self):
        cfg = ScorerConfig(keep_threshold=0.5)
        scorer = RelevanceScorer(config=cfg)
        rec = _make_record(importance=0.9, confidence=0.9, access_count=10)
        score = scorer.score(rec)
        assert score.suggested_action == "keep"

    def test_score_constraint_always_keep(self):
        scorer = RelevanceScorer()
        rec = _make_record(
            importance=0.0,
            confidence=0.0,
            memory_type=MemoryType.CONSTRAINT,
        )
        score = scorer.score(rec)
        assert score.suggested_action == "keep"

    def test_score_low_suggests_decay_or_below(self):
        cfg = ScorerConfig(keep_threshold=0.9, decay_threshold=0.3)
        scorer = RelevanceScorer(config=cfg)
        rec = _make_record(
            importance=0.1,
            confidence=0.1,
            access_count=0,
            days_ago=365,
        )
        score = scorer.score(rec)
        assert score.total_score < 0.9
        assert score.suggested_action in ("decay", "silence", "compress", "delete")

    def test_score_batch(self):
        scorer = RelevanceScorer()
        recs = [_make_record(), _make_record(importance=0.9)]
        scores = scorer.score_batch(recs)
        assert len(scores) == 2
        assert all(hasattr(s, "total_score") for s in scores)

    def test_score_batch_with_dependency_counts(self):
        scorer = RelevanceScorer()
        recs = [_make_record()]
        dep = {str(recs[0].id): 5}
        scores = scorer.score_batch(recs, dependency_counts=dep)
        assert len(scores) == 1
        assert scores[0].dependency_score > 0


class TestForgettingPolicyEngine:
    def test_plan_operations_skips_keep(self):
        engine = ForgettingPolicyEngine()
        scores = [
            RelevanceScore(
                "id1",
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                suggested_action="keep",
            ),
        ]
        ops = engine.plan_operations(scores)
        assert len(ops) == 0

    def test_plan_operations_decay_has_new_confidence(self):
        engine = ForgettingPolicyEngine(decay_rate=0.15)
        mem_id = str(uuid4())
        scores = [
            RelevanceScore(
                mem_id,
                0.5,
                0.5,
                0.5,
                0.5,
                0.6,
                0.5,
                0.5,
                suggested_action="decay",
            ),
        ]
        ops = engine.plan_operations(scores)
        assert len(ops) == 1
        assert ops[0].action == ForgettingAction.DECAY
        assert ops[0].new_confidence is not None
        assert ops[0].new_confidence == pytest.approx(0.45)  # 0.6 - 0.15

    def test_plan_operations_respects_max_operations(self):
        engine = ForgettingPolicyEngine()
        scores = [
            RelevanceScore(
                str(uuid4()),
                0.3,
                0.3,
                0.3,
                0.3,
                0.3,
                0.3,
                0.3,
                suggested_action="decay",
            ),
            RelevanceScore(
                str(uuid4()),
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                suggested_action="silence",
            ),
        ]
        ops = engine.plan_operations(scores, max_operations=1)
        assert len(ops) == 1

    def test_create_compression_truncates(self):
        engine = ForgettingPolicyEngine(compression_max_chars=20)
        out = engine.create_compression("this is a long piece of text here")
        assert len(out) <= 20
        assert out.endswith("...")


class TestInterferenceDetector:
    def test_detect_overlapping_same_text(self):
        det = InterferenceDetector()
        r1 = _make_record(text="hello world foo bar")
        r2 = _make_record(text="hello world baz")
        overlaps = det.detect_overlapping([r1, r2], text_overlap_threshold=0.4)
        assert len(overlaps) >= 1
        assert overlaps[0].interference_type == "overlapping"

    def test_detect_duplicates_identical_embedding(self):
        det = InterferenceDetector(similarity_threshold=0.99)
        emb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        r1 = _make_record(text="a")
        r1.embedding = emb
        r2 = _make_record(text="b")
        r2.embedding = emb
        dups = det.detect_duplicates([r1, r2])
        assert len(dups) == 1
        assert dups[0].interference_type == "duplicate"
        assert dups[0].similarity == pytest.approx(1.0)

    def test_detect_duplicates_no_embedding_skipped(self):
        det = InterferenceDetector()
        r1 = _make_record(text="a")
        r1.embedding = None
        r2 = _make_record(text="b")
        r2.embedding = None
        dups = det.detect_duplicates([r1, r2])
        assert len(dups) == 0


class TestCompression:
    @pytest.mark.asyncio
    async def test_summarize_for_compression_short_text_unchanged(self):
        from src.forgetting.compression import summarize_for_compression

        out = await summarize_for_compression("Short.", max_chars=100)
        assert out == "Short."

    @pytest.mark.asyncio
    async def test_summarize_for_compression_truncate_without_llm(self):
        from src.forgetting.compression import summarize_for_compression

        long_text = "This is a very long piece of text that should be truncated."
        out = await summarize_for_compression(long_text, max_chars=20, llm_client=None)
        assert len(out) <= 20
        assert out.endswith("...")

    @pytest.mark.asyncio
    async def test_summarize_for_compression_uses_llm_when_provided(self):
        from src.forgetting.compression import summarize_for_compression
        from src.utils.llm import MockLLMClient

        mock = MockLLMClient(fixed_response="User likes pizza.")
        long_text = "The user said they really enjoy eating pizza on weekends."
        out = await summarize_for_compression(long_text, max_chars=100, llm_client=mock)
        assert "pizza" in out.lower()
        assert len(out) <= 100


# PostgresMemoryStore.count_references_to and ForgettingExecutor dependency checks
# are integration tests (require real Postgres); see tests/integration/test_forgetting_flow.py
