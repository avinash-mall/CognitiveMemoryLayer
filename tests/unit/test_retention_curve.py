"""One decay curve, read by both subsystems, honouring the per-record rate.

Before this, the forgetting scorer and the retrieval reranker used two different
aging functions and neither read ``MemoryRecord.decay_rate`` — the write path
assigned every memory a rate that nothing ever consulted.
"""

from datetime import UTC, datetime, timedelta

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.forgetting.scorer import RelevanceScorer
from src.retrieval.reranker import MemoryReranker
from src.utils.retention import frequency_score, retention


def _record(
    *,
    decay_rate: float = 0.01,
    days_ago: float = 0.0,
    access_count: int = 0,
    text: str = "a memory",
) -> MemoryRecord:
    return MemoryRecord(
        tenant_id="t",
        context_tags=[],
        type=MemoryType.EPISODIC_EVENT,
        text=text,
        decay_rate=decay_rate,
        access_count=access_count,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=datetime.now(UTC) - timedelta(days=days_ago),
    )


class TestRetentionCurve:
    def test_fresh_memory_is_fully_retained(self):
        assert retention(0.0, 0.1) == 1.0
        assert retention(-5.0, 0.1) == 1.0  # clock skew must not exceed 1.0

    def test_decay_is_monotonic_in_age_and_in_rate(self):
        assert retention(10, 0.1) > retention(20, 0.1)
        assert retention(10, 0.01) > retention(10, 0.5)

    def test_absent_rate_falls_back_to_the_column_default(self):
        assert retention(30, None) == pytest.approx(retention(30, 0.01))

    def test_absurd_rate_is_clamped_rather_than_trusted(self):
        """decay_rate arrives from an LLM and nothing validates it upstream."""
        assert retention(1, 50.0) == pytest.approx(retention(1, 1.0))
        assert retention(1, -3.0) == 1.0

    def test_frequency_is_capped_so_use_cannot_dominate(self):
        assert frequency_score(0) == 0.0
        assert frequency_score(9) == pytest.approx(1.0)
        assert frequency_score(10_000) == 1.0


class TestScorerReadsDecayRate:
    def test_ephemeral_memory_scores_below_stable_one_of_the_same_age(self):
        scorer = RelevanceScorer()
        stable = scorer.score(_record(decay_rate=0.01, days_ago=7))
        ephemeral = scorer.score(_record(decay_rate=0.5, days_ago=7))

        assert ephemeral.recency_score < stable.recency_score
        assert ephemeral.total_score < stable.total_score

    def test_rate_is_irrelevant_for_a_brand_new_memory(self):
        scorer = RelevanceScorer()
        assert scorer.score(_record(decay_rate=0.5)).recency_score == pytest.approx(
            scorer.score(_record(decay_rate=0.01)).recency_score
        )


class TestRerankerReadsUsage:
    def _retrieved(self, record: MemoryRecord, relevance: float = 0.5) -> RetrievedMemory:
        return RetrievedMemory(record=record, relevance_score=relevance, retrieval_source="vector")

    @pytest.mark.asyncio
    async def test_often_retrieved_memory_outranks_an_identical_unused_one(self):
        """The testing effect: use makes a memory easier to find, not just harder
        to delete. access_count was previously read only by the forgetting path."""
        reranker = MemoryReranker()
        used = self._retrieved(_record(access_count=50, text="used often"))
        unused = self._retrieved(_record(access_count=0, text="never used"))

        ranked = await reranker.rerank([unused, used], "q")
        assert ranked[0].record.text == "used often"

    @pytest.mark.asyncio
    async def test_frequency_cannot_outrank_a_real_relevance_gap(self):
        """Guards the rich-get-richer loop: only the vector prong increments
        access_count, so a heavy weight would bury fact- and graph-sourced hits."""
        reranker = MemoryReranker()
        popular = self._retrieved(_record(access_count=100_000, text="popular"), relevance=0.1)
        relevant = self._retrieved(_record(access_count=0, text="relevant"), relevance=0.9)

        ranked = await reranker.rerank([popular, relevant], "q")
        assert ranked[0].record.text == "relevant"

    @pytest.mark.asyncio
    async def test_reranker_ages_memories_by_their_own_rate(self):
        reranker = MemoryReranker()
        stable = self._retrieved(_record(decay_rate=0.01, days_ago=30, text="stable"))
        ephemeral = self._retrieved(_record(decay_rate=0.5, days_ago=30, text="ephemeral"))

        _, breakdown = await reranker.rerank_with_breakdown([ephemeral, stable], "q")
        by_text = {b["text"]: b["breakdown"]["recency"] for b in breakdown}
        assert by_text["ephemeral"] < by_text["stable"]

    @pytest.mark.asyncio
    async def test_scorer_and_reranker_agree_on_the_same_memory(self):
        """The two subsystems used different curves; a memory must not age at one
        speed when deciding what to delete and another when deciding what to show."""
        record = _record(decay_rate=0.1, days_ago=14)
        scorer_recency = RelevanceScorer().score(record).recency_score

        _, breakdown = await MemoryReranker().rerank_with_breakdown([self._retrieved(record)], "q")
        assert breakdown[0]["breakdown"]["recency"] == pytest.approx(scorer_recency, abs=0.02)
