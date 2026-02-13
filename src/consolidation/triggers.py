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
class TriggerCondition:
    """Condition that can trigger consolidation."""

    trigger_type: TriggerType

    # For SCHEDULED
    interval_hours: float | None = None

    # For QUOTA
    min_episodes: int | None = None
    max_memory_mb: float | None = None

    # For EVENT
    event_types: list[str] = field(default_factory=list)

    # Metadata
    last_triggered: datetime | None = None
    trigger_count: int = 0


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
    time_window_days: int = 7


class ConsolidationScheduler:
    """Manages consolidation scheduling and triggers."""

    def __init__(
        self,
        default_interval_hours: float = 6.0,
        quota_threshold_episodes: int = 500,
        quota_threshold_mb: float = 100.0,
    ):
        self.default_interval = timedelta(hours=default_interval_hours)
        self.quota_episodes = quota_threshold_episodes
        self.quota_mb = quota_threshold_mb

        self._conditions: dict[str, list[TriggerCondition]] = {}
        self._task_queue: asyncio.Queue[ConsolidationTask] = asyncio.Queue()
        self._running = False
        self._worker_task: asyncio.Task | None = None

    def _user_key(self, tenant_id: str, user_id: str) -> str:
        return f"{tenant_id}:{user_id}"

    def register_user(
        self,
        tenant_id: str,
        user_id: str,
        conditions: list[TriggerCondition] | None = None,
    ):
        """Register a user with their trigger conditions."""
        key = self._user_key(tenant_id, user_id)

        if conditions:
            self._conditions[key] = conditions
        else:
            self._conditions[key] = [
                TriggerCondition(
                    trigger_type=TriggerType.SCHEDULED,
                    interval_hours=self.default_interval.total_seconds() / 3600,
                ),
                TriggerCondition(
                    trigger_type=TriggerType.QUOTA,
                    min_episodes=self.quota_episodes,
                ),
            ]

    async def check_triggers(
        self,
        tenant_id: str,
        user_id: str,
        episode_count: int,
        memory_size_mb: float,
        event: str | None = None,
    ) -> bool:
        """Check if any trigger conditions are met. Returns True if consolidation should run."""
        key = self._user_key(tenant_id, user_id)
        conditions = self._conditions.get(key, [])

        now = datetime.now(UTC)
        triggered = False
        trigger_reason = ""

        for condition in conditions:
            should_trigger = False

            if condition.trigger_type == TriggerType.SCHEDULED:
                if condition.interval_hours and condition.last_triggered:
                    elapsed = (now - condition.last_triggered).total_seconds() / 3600
                    should_trigger = elapsed >= condition.interval_hours
                    trigger_reason = f"Scheduled: {elapsed:.1f}h since last run"
                elif condition.interval_hours:
                    should_trigger = True
                    trigger_reason = "Scheduled: first run"

            elif condition.trigger_type == TriggerType.QUOTA:
                if condition.min_episodes and episode_count >= condition.min_episodes:
                    should_trigger = True
                    trigger_reason = f"Quota: {episode_count} episodes"
                elif condition.max_memory_mb and memory_size_mb >= condition.max_memory_mb:
                    should_trigger = True
                    trigger_reason = f"Quota: {memory_size_mb:.1f}MB"

            elif condition.trigger_type == TriggerType.EVENT:
                if event and event in condition.event_types:
                    should_trigger = True
                    trigger_reason = f"Event: {event}"

            if should_trigger:
                condition.last_triggered = now
                condition.trigger_count += 1
                triggered = True

                await self._task_queue.put(
                    ConsolidationTask(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        trigger_type=condition.trigger_type,
                        trigger_reason=trigger_reason,
                    )
                )
                break

        return triggered

    async def trigger_manual(
        self,
        tenant_id: str,
        user_id: str,
        reason: str = "Manual trigger",
        priority: int = 10,
    ):
        """Manually trigger consolidation."""
        await self._task_queue.put(
            ConsolidationTask(
                tenant_id=tenant_id,
                user_id=user_id,
                trigger_type=TriggerType.MANUAL,
                trigger_reason=reason,
                priority=priority,
            )
        )

    async def get_next_task(self) -> ConsolidationTask | None:
        """Get next consolidation task from queue."""
        try:
            return await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
        except TimeoutError:
            return None

    def has_pending_tasks(self) -> bool:
        return not self._task_queue.empty()
