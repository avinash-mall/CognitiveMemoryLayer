"""Consolidation worker orchestrating the full flow."""

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime

from ..memory.neocortical.store import NeocorticalStore
from ..storage.postgres import PostgresMemoryStore
from ..utils.llm import LLMClient
from ..utils.logging_config import get_logger
from .clusterer import SemanticClusterer
from .migrator import ConsolidationMigrator, MigrationResult
from .sampler import EpisodeSampler
from .schema_aligner import SchemaAligner
from .summarizer import GistExtractor
from .triggers import ConsolidationScheduler, ConsolidationTask


@dataclass
class ConsolidationReport:
    """Report from a consolidation run."""

    tenant_id: str
    user_id: str
    started_at: datetime
    completed_at: datetime

    episodes_sampled: int
    clusters_formed: int
    gists_extracted: int
    migration: MigrationResult

    elapsed_seconds: float

    @property
    def success(self) -> bool:
        return len(self.migration.errors) == 0


class ConsolidationWorker:
    """Main consolidation worker that orchestrates the full process."""

    def __init__(
        self,
        episodic_store: PostgresMemoryStore,
        neocortical_store: NeocorticalStore,
        llm_client: LLMClient,
        scheduler: ConsolidationScheduler | None = None,
    ):
        self.sampler = EpisodeSampler(episodic_store)
        self.clusterer = SemanticClusterer()
        self.extractor = GistExtractor(llm_client)
        self.aligner = SchemaAligner(neocortical_store.facts)
        self.migrator = ConsolidationMigrator(neocortical_store, episodic_store)

        self.scheduler = scheduler or ConsolidationScheduler()

        self._running = False
        self._worker_task: asyncio.Task | None = None

    async def consolidate(
        self,
        tenant_id: str,
        user_id: str,
        task: ConsolidationTask | None = None,
    ) -> ConsolidationReport:
        """Run full consolidation for a user."""
        started = datetime.now(UTC)

        episode_limit = task.episode_limit if task else 200
        episodes = await self.sampler.sample(tenant_id, user_id, max_episodes=episode_limit)

        if not episodes:
            return ConsolidationReport(
                tenant_id=tenant_id,
                user_id=user_id,
                started_at=started,
                completed_at=datetime.now(UTC),
                episodes_sampled=0,
                clusters_formed=0,
                gists_extracted=0,
                migration=MigrationResult(0, 0, 0, 0, []),
                elapsed_seconds=0.0,
            )

        clusters = self.clusterer.cluster(episodes)
        gists = await self.extractor.extract_from_clusters(clusters)
        alignments = await self.aligner.align_batch(tenant_id, user_id, gists)
        migration = await self.migrator.migrate(
            tenant_id,
            user_id,
            alignments,
            mark_episodes_consolidated=True,
            compress_episodes=False,
        )

        completed = datetime.now(UTC)
        return ConsolidationReport(
            tenant_id=tenant_id,
            user_id=user_id,
            started_at=started,
            completed_at=completed,
            episodes_sampled=len(episodes),
            clusters_formed=len(clusters),
            gists_extracted=len(gists),
            migration=migration,
            elapsed_seconds=(completed - started).total_seconds(),
        )

    async def start_background_worker(self):
        """Start background consolidation worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_background_worker(self):
        """Stop background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

    async def _worker_loop(self):
        """Background worker loop."""
        log = get_logger(__name__)
        while self._running:
            task = await self.scheduler.get_next_task()
            if task:
                try:
                    report = await self.consolidate(
                        task.tenant_id,
                        task.user_id,
                        task,
                    )
                    if report.gists_extracted:
                        log.info(
                            "consolidation complete",
                            tenant_id=task.tenant_id,
                            user_id=task.user_id,
                            gists_extracted=report.gists_extracted,
                            clusters_formed=report.clusters_formed,
                        )
                except Exception:
                    log.exception(
                        "consolidation failed",
                        tenant_id=task.tenant_id,
                        user_id=task.user_id,
                    )
            else:
                await asyncio.sleep(1)
