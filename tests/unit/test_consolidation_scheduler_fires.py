"""Consolidation must actually fire on its own, and be checkable by count.

``ConsolidationWorker.start_background_worker`` had no caller anywhere in the repo, and
the ``ConsolidationScheduler`` registry it polled was populated only by tests — nothing
called ``register_user``, so ``check_triggers`` could not fire for any user on any
deployment. The documented 6-hour interval and 500-episode quota had therefore never run
in production; the two admin HTTP routes were the only live entry points. The database
agreed: 307 of 549,580 stored records carried a consolidated marker, 0.06%.

That is the fourth instance in this codebase of a feature that is flagged on, documented
as shipped, and has no caller (``encode_chunk``, eval-mode graph blindness, prospective
indexing were the others). Each cost a wrong conclusion. So these tests assert the
*firing*, not the configuration — reading the diff is what missed it three times.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.consolidation.triggers import ConsolidationScheduler, TriggerType
from src.consolidation.worker import ConsolidationWorker


class _Store:
    """Only the two methods the sweep touches."""

    def __init__(self, counts: list[tuple[str, int]] | None = None):
        self.counts = counts or []
        self.min_counts: list[int] = []

    async def unconsolidated_counts_by_tenant(self, min_count, types=None):
        self.min_counts.append(min_count)
        return [(t, n) for t, n in self.counts if n >= min_count]

    async def scan(self, *a, **k):
        return []


def _worker(store, **scheduler_kwargs):
    neocortical = MagicMock()
    neocortical.facts = MagicMock()
    return ConsolidationWorker(
        episodic_store=store,
        neocortical_store=neocortical,
        llm_client=None,
        scheduler=ConsolidationScheduler(**scheduler_kwargs),
    )


class TestTheSweepEnqueues:
    @pytest.mark.asyncio
    async def test_a_tenant_over_quota_is_enqueued(self):
        worker = _worker(_Store([("busy", 900)]), quota_threshold_episodes=500)

        assert await worker.sweep_once() == 1

        task = await worker.scheduler.get_next_task()
        assert task is not None
        assert task.tenant_id == "busy"
        assert task.trigger_type == TriggerType.QUOTA
        assert "900" in task.trigger_reason

    @pytest.mark.asyncio
    async def test_a_tenant_under_quota_is_not(self):
        worker = _worker(_Store([("quiet", 12)]), quota_threshold_episodes=500)

        assert await worker.sweep_once() == 0
        assert not worker.scheduler.has_pending_tasks()

    @pytest.mark.asyncio
    async def test_the_quota_reaching_the_query_is_the_schedulers(self):
        """The threshold is applied in SQL, so a wrong value here silently sweeps every
        tenant in the database on every interval."""
        store = _Store()
        await _worker(store, quota_threshold_episodes=250).sweep_once()

        assert store.min_counts == [250]


class TestTriggerCounts:
    """The observable the plan asks for: 'verify by trigger count, not by reading the
    diff'. These are what /admin/consolidation/status returns."""

    @pytest.mark.asyncio
    async def test_counts_start_at_zero_and_track_sweeps(self):
        worker = _worker(_Store([("busy", 900)]))
        assert worker.status["sweeps_run"] == 0
        assert worker.status["tasks_enqueued"] == 0

        await worker.sweep_once()
        await worker.sweep_once()

        assert worker.status["sweeps_run"] == 2
        assert worker.status["tasks_enqueued"] == 2
        assert worker.status["pending_tasks"] == 2

    @pytest.mark.asyncio
    async def test_a_sweep_that_finds_nothing_still_counts_as_a_sweep(self):
        """Otherwise "0 sweeps" is ambiguous between "not running" and "nothing due" —
        which is exactly the ambiguity that hid this bug."""
        worker = _worker(_Store([]))
        await worker.sweep_once()

        assert worker.status["sweeps_run"] == 1
        assert worker.status["tasks_enqueued"] == 0


class TestTheBackgroundLoops:
    @pytest.mark.asyncio
    async def test_starting_runs_a_sweep_without_being_asked(self):
        """The whole point: no HTTP call, no manual trigger."""
        worker = _worker(_Store([("busy", 900)]))
        worker.consolidate = AsyncMock(return_value=MagicMock(gists_extracted=0, clusters_formed=0))

        await worker.start_background_worker()
        try:
            for _ in range(50):
                if worker.consolidate.await_count:
                    break
                await asyncio.sleep(0.02)
        finally:
            await worker.stop_background_worker()

        assert worker.sweeps_run >= 1
        assert worker.consolidate.await_count >= 1
        assert worker.consolidate.await_args.args[0] == "busy"

    @pytest.mark.asyncio
    async def test_stopping_cancels_both_loops(self):
        """A surviving sweep task would keep consolidating after shutdown began."""
        worker = _worker(_Store([]))
        await worker.start_background_worker()
        await worker.stop_background_worker()

        assert worker._worker_task is None
        assert worker._sweep_task is None
        assert worker.status["running"] is False

    @pytest.mark.asyncio
    async def test_starting_twice_does_not_double_the_loops(self):
        worker = _worker(_Store([]))
        await worker.start_background_worker()
        first = worker._sweep_task
        await worker.start_background_worker()
        try:
            assert worker._sweep_task is first
        finally:
            await worker.stop_background_worker()


class TestItIsOffByDefault:
    def test_the_flag_defaults_off(self):
        """It spends LLM tokens on a write path already 95.8% LLM-bound, and no arm has
        measured it. Every other behaviour default here was set by a measurement."""
        from src.core.config import FeatureFlags

        assert FeatureFlags().consolidation_scheduler_enabled is False


class TestTheRecurrenceGate:
    """RecMem defers LLM consolidation until an interaction shows semantic recurrence,
    cutting memory-construction tokens 87%. Here that is a cluster-size threshold."""

    def test_it_gates_nothing_by_default(self):
        """At 2, a tenant whose episodes never cluster stops producing gists entirely.
        That trade is unmeasured, so the shipped default preserves current behaviour and
        the fresh-ingest arm moves the knob."""
        from src.core.config import FeatureFlags

        assert FeatureFlags().consolidation_recurrence_min == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("minimum", "expect_extracted", "expect_skipped"),
        [(1, 2, 0), (2, 1, 1), (3, 0, 2)],
    )
    async def test_clusters_below_the_threshold_never_reach_the_llm(
        self, monkeypatch, minimum, expect_extracted, expect_skipped
    ):
        from src.consolidation.clusterer import EpisodeCluster
        from src.core.config import get_settings
        from src.core.enums import MemoryType

        monkeypatch.setenv("FEATURES__CONSOLIDATION_RECURRENCE_MIN", str(minimum))
        get_settings.cache_clear()

        def _episode():
            return MagicMock(type=MemoryType.EPISODIC_EVENT, metadata={})

        singleton = EpisodeCluster(cluster_id=0, episodes=[_episode()], avg_confidence=1.0)
        pair = EpisodeCluster(cluster_id=1, episodes=[_episode(), _episode()], avg_confidence=1.0)
        clusters = [singleton, pair]

        worker = _worker(_Store([]))
        worker.sampler.sample = AsyncMock(return_value=[_episode()])
        worker.clusterer.cluster = MagicMock(return_value=clusters)
        worker.extractor.extract_from_clusters = AsyncMock(return_value=[])
        worker.aligner.align_batch = AsyncMock(return_value=[])
        worker.migrator.migrate = AsyncMock(
            return_value=MagicMock(errors=[], facts_created=0, episodes_marked=0)
        )

        report = await worker.consolidate("t", "u")

        sent = worker.extractor.extract_from_clusters.await_args.args[0]
        assert len(sent) == expect_extracted
        assert report.clusters_skipped_no_recurrence == expect_skipped
        # the report still counts every cluster formed, or the gate would hide its cost
        assert report.clusters_formed == 2

        get_settings.cache_clear()
