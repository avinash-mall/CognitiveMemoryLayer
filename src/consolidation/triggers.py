"""Consolidation triggers and scheduler."""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class TriggerType(StrEnum):
    SCHEDULED = "scheduled"
    QUOTA = "quota"
    EVENT = "event"
    MANUAL = "manual"


@dataclass
class ConsolidationTask:
    """A scheduled consolidation task."""

    tenant_id: str
    user_id: str
    trigger_type: TriggerType
    trigger_reason: str
    priority: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Scope
    episode_limit: int = 200


class ConsolidationScheduler:
    """Queue of pending consolidation work.

    This used to carry a per-user registry of ``TriggerCondition`` objects consulted by
    ``check_triggers``. Nothing ever called ``register_user``, so the registry was
    permanently empty and ``check_triggers`` could not fire for any user on any
    deployment — the documented 6-hour interval and 500-episode quota never once ran.
    The registry was removed rather than wired up: pacing now comes from
    ``ConsolidationWorker``'s sweep interval and eligibility from a SQL quota check, so
    per-user in-memory condition state would have been a second, restart-losing copy of
    both. Keeping it beside the live path is the dead-lookalike-code trap that has
    produced four wrong conclusions in this codebase already.
    """

    def __init__(
        self,
        default_interval_hours: float = 6.0,
        quota_threshold_episodes: int = 500,
    ):
        self.default_interval = timedelta(hours=default_interval_hours)
        self.quota_episodes = quota_threshold_episodes

        self._task_queue: asyncio.Queue[ConsolidationTask] = asyncio.Queue()

    async def enqueue(
        self,
        tenant_id: str,
        user_id: str,
        trigger_type: TriggerType,
        reason: str,
        priority: int = 0,
        episode_limit: int = 200,
    ) -> None:
        """Queue a consolidation task."""
        await self._task_queue.put(
            ConsolidationTask(
                tenant_id=tenant_id,
                user_id=user_id,
                trigger_type=trigger_type,
                trigger_reason=reason,
                priority=priority,
                episode_limit=episode_limit,
            )
        )

    async def trigger_manual(
        self,
        tenant_id: str,
        user_id: str,
        reason: str = "Manual trigger",
        priority: int = 10,
    ) -> None:
        """Manually trigger consolidation."""
        await self.enqueue(
            tenant_id,
            user_id,
            TriggerType.MANUAL,
            reason,
            priority=priority,
        )

    async def get_next_task(self) -> ConsolidationTask | None:
        """Get next consolidation task from queue."""
        try:
            return await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
        except TimeoutError:
            return None

    def has_pending_tasks(self) -> bool:
        return not self._task_queue.empty()

    def pending_count(self) -> int:
        return self._task_queue.qsize()
