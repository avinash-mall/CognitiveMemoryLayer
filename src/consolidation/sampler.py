"""Episode sampling for consolidation."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from ..core.enums import MemoryStatus, MemoryType
from ..core.schemas import MemoryRecord
from ..storage.base import MemoryStoreBase
from ..storage.utils import naive_utc


@dataclass
class SamplingConfig:
    max_episodes: int = 200
    min_importance: float = 0.3
    min_confidence: float = 0.3

    importance_weight: float = 0.4
    access_count_weight: float = 0.3
    recency_weight: float = 0.3


class EpisodeSampler:
    """Samples episodes for consolidation. Prioritizes by importance, access frequency, recency."""

    def __init__(
        self,
        store: MemoryStoreBase,
        config: SamplingConfig | None = None,
    ):
        self.store = store
        self.config = config or SamplingConfig()

    #: Types sampled for gist extraction, alongside CONSTRAINT which is sampled
    #: separately so a quiet tenant's constraints cannot be crowded out by chatter.
    EPISODE_TYPES = (
        MemoryType.EPISODIC_EVENT.value,
        MemoryType.PREFERENCE.value,
        MemoryType.HYPOTHESIS.value,
    )

    async def sample(
        self,
        tenant_id: str,
        max_episodes: int | None = None,
        exclude_consolidated: bool = True,
    ) -> list[MemoryRecord]:
        """Sample episodes for consolidation.

        Eligibility is "not yet consolidated", never age. This used to scan a
        ``timestamp >= now - 7d`` window (90d for constraints), which was *absorbing*
        rather than sliding: an episode not sampled within its window was never a
        candidate again. Combined with a trigger that only fired when someone clicked,
        most episodes were never consolidation candidates at all. Recency still ranks
        candidates via ``_score``; it just no longer excludes them.
        """
        max_eps = max_episodes or self.config.max_episodes

        base: dict = {"status": MemoryStatus.ACTIVE.value}
        if exclude_consolidated:
            # Pushed into SQL rather than filtered afterwards: post-filtering a
            # limit-capped scan silently returns nothing once the newest rows are all
            # consolidated, which is the backlog case this method exists to drain.
            base["unconsolidated"] = True

        # Newest-first so the candidate set is deterministic under the limit; the
        # priority score below then reorders within it.
        candidates = await self.store.scan(
            tenant_id,
            filters={**base, "type": list(self.EPISODE_TYPES)},
            order_by="-timestamp",
            limit=max_eps * 3,
        )
        candidates.extend(
            await self.store.scan(
                tenant_id,
                filters={**base, "type": [MemoryType.CONSTRAINT.value]},
                order_by="-timestamp",
                limit=max_eps,
            )
        )

        if exclude_consolidated:
            # Belt and braces for stores whose scan ignores the filter key.
            candidates = [
                c for c in candidates if not (c.metadata and c.metadata.get("consolidated"))
            ]

        candidates = [
            c
            for c in candidates
            if c.importance >= self.config.min_importance
            and c.confidence >= self.config.min_confidence
        ]

        scored = [(self._score(c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [c for _, c in scored[:max_eps]]

    def _score(self, record: MemoryRecord) -> float:
        """Calculate priority score for a record."""
        importance_score = record.importance
        access_score = math.log1p(record.access_count) / 5.0
        access_score = min(access_score, 1.0)
        ts = record.timestamp or datetime.now(UTC)
        now_naive = naive_utc(datetime.now(UTC))
        ts_naive = naive_utc(ts)
        age_days = (now_naive - ts_naive).days if (now_naive and ts_naive) else 0
        recency_score = 1.0 / (1.0 + age_days * 0.1)
        return (
            self.config.importance_weight * importance_score
            + self.config.access_count_weight * access_score
            + self.config.recency_weight * recency_score
        )
