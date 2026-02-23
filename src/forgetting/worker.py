"""Forgetting worker and scheduler."""

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from ..core.enums import MemoryStatus
from ..storage.base import MemoryStoreBase
from ..utils.llm import LLMClient
from ..utils.logging_config import get_logger
from .actions import (
    ForgettingAction,
    ForgettingOperation,
    ForgettingPolicyEngine,
    ForgettingResult,
)
from .executor import ForgettingExecutor
from .interference import InterferenceDetector, InterferenceResult
from .scorer import RelevanceScorer, ScorerConfig

logger = get_logger(__name__)


@dataclass
class ForgettingReport:
    """Report from a forgetting run."""

    tenant_id: str
    user_id: str
    started_at: datetime
    completed_at: datetime
    memories_scanned: int
    memories_scored: int
    result: ForgettingResult
    duplicates_found: int = 0
    duplicates_resolved: int = 0
    elapsed_seconds: float = 0.0


class ForgettingWorker:
    """Orchestrates the active forgetting process."""

    def __init__(
        self,
        store: MemoryStoreBase,
        scorer_config: ScorerConfig | None = None,
        archive_store: MemoryStoreBase | None = None,
        compression_llm_client: LLMClient | None = None,
        compression_max_chars: int = 100,
    ) -> None:
        self.store = store
        self.scorer = RelevanceScorer(scorer_config)
        self.policy = ForgettingPolicyEngine(
            compression_max_chars=compression_max_chars,
        )
        self.executor = ForgettingExecutor(
            store,
            archive_store,
            compression_llm_client=compression_llm_client,
            compression_max_chars=compression_max_chars,
        )
        self.interference = InterferenceDetector()

    async def run_forgetting(
        self,
        tenant_id: str,
        user_id: str,
        max_memories: int = 5000,
        dry_run: bool = False,
    ) -> ForgettingReport:
        """Run forgetting process for a user."""
        started = datetime.now(UTC)
        memories = await self.store.scan(
            tenant_id,
            filters={"status": MemoryStatus.ACTIVE.value},
            limit=max_memories,
        )

        if not memories:
            return ForgettingReport(
                tenant_id=tenant_id,
                user_id=user_id,
                started_at=started,
                completed_at=datetime.now(UTC),
                memories_scanned=0,
                memories_scored=0,
                result=ForgettingResult(0, 0),
            )

        dep_counts = await self._get_dependency_counts(tenant_id, user_id, memories)
        scores = self.scorer.score_batch(memories, dep_counts)
        operations = self.policy.plan_operations(scores)

        duplicates = self.interference.detect_duplicates(memories)
        dup_operations = self._plan_duplicate_resolution(duplicates)
        operations.extend(dup_operations)

        result = await self.executor.execute(operations, dry_run=dry_run)
        completed = datetime.now(UTC)

        return ForgettingReport(
            tenant_id=tenant_id,
            user_id=user_id,
            started_at=started,
            completed_at=completed,
            memories_scanned=len(memories),
            memories_scored=len(scores),
            result=result,
            duplicates_found=len(duplicates),
            duplicates_resolved=len(dup_operations),
            elapsed_seconds=(completed - started).total_seconds(),
        )

    async def _get_dependency_counts(
        self,
        tenant_id: str,
        user_id: str,
        memories: list,
    ) -> dict[str, int]:
        """Count how many other memories reference each memory.

        Prefers a single DB aggregation query (O(1) round-trips) when
        available, falling back to the O(n²) Python loop for stores that
        don't support ``bulk_dependency_counts``.
        """
        memory_ids = [str(m.id) for m in memories]

        # Phase 4.1: DB-side aggregation
        if hasattr(self.store, "bulk_dependency_counts"):
            try:
                return await self.store.bulk_dependency_counts(tenant_id, memory_ids)
            except Exception:
                logger.warning("bulk_dependency_counts_fallback", exc_info=True)

        # Fallback: O(n²) Python loop (original implementation)
        counts: dict[str, int] = {}
        for mem in memories:
            mem_id = str(mem.id)
            counts[mem_id] = 0
            for other in memories:
                if other.id == mem.id:
                    continue
                if other.supersedes_id and str(other.supersedes_id) == mem_id:
                    counts[mem_id] += 1
                refs = (other.metadata or {}).get("evidence_refs", [])
                if mem_id in refs:
                    counts[mem_id] += 1
        return counts

    def _plan_duplicate_resolution(
        self,
        duplicates: list[InterferenceResult],
    ) -> list[ForgettingOperation]:
        """Plan operations to resolve duplicates (keep one, delete other)."""
        operations: list[ForgettingOperation] = []
        resolved_ids: set[str] = set()

        for dup in duplicates:
            if dup.memory_id in resolved_ids or dup.interfering_memory_id in resolved_ids:
                continue
            if dup.keep_id:
                to_delete = (
                    dup.interfering_memory_id if dup.keep_id == dup.memory_id else dup.memory_id
                )
            else:
                to_delete = dup.interfering_memory_id
            operations.append(
                ForgettingOperation(
                    action=ForgettingAction.DELETE,
                    memory_id=UUID(to_delete),
                    reason=f"Duplicate of {dup.memory_id if to_delete == dup.interfering_memory_id else dup.interfering_memory_id}",
                )
            )
            resolved_ids.add(to_delete)
        return operations


class ForgettingScheduler:
    """Schedules and manages forgetting runs."""

    def __init__(
        self,
        worker: ForgettingWorker,
        interval_hours: float = 24.0,
    ) -> None:
        self.worker = worker
        self.interval = timedelta(hours=interval_hours)
        self._running = False
        self._task: asyncio.Task | None = None
        self._user_last_run: dict[str, datetime] = {}

    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def schedule_user(
        self,
        tenant_id: str,
        user_id: str,
        force: bool = False,
    ) -> ForgettingReport | None:
        """Schedule forgetting for a user; returns report if run."""
        key = f"{tenant_id}:{user_id}"
        now = datetime.now(UTC)
        last_run = self._user_last_run.get(key)
        if force or not last_run or (now - last_run) >= self.interval:
            report = await self.worker.run_forgetting(tenant_id, user_id)
            self._user_last_run[key] = now
            return report
        return None

    async def _scheduler_loop(self) -> None:
        """Background scheduler loop. DES-09: check first then sleep so first run is immediate."""
        while self._running:
            now = datetime.now(UTC)
            for key, last_run in list(self._user_last_run.items()):
                if (now - last_run) >= self.interval:
                    try:
                        parts = key.split(":", 1)
                        if len(parts) == 2:
                            tenant_id, user_id = parts
                            await self.worker.run_forgetting(tenant_id, user_id)
                            self._user_last_run[key] = now
                    except Exception:
                        logger.exception("forgetting_run_failed", extra={"key": key})
            await asyncio.sleep(self.interval.total_seconds())
