"""Episode sampling for consolidation."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ..core.enums import MemoryStatus, MemoryType
from ..core.schemas import MemoryRecord
from ..storage.postgres import PostgresMemoryStore
from ..storage.utils import naive_utc


@dataclass
class SamplingConfig:
    max_episodes: int = 200
    time_window_days: int = 7
    min_importance: float = 0.3
    min_confidence: float = 0.3

    importance_weight: float = 0.4
    access_count_weight: float = 0.3
    recency_weight: float = 0.3


class EpisodeSampler:
    """Samples episodes for consolidation. Prioritizes by importance, access frequency, recency."""

    def __init__(
        self,
        store: PostgresMemoryStore,
        config: SamplingConfig | None = None,
    ):
        self.store = store
        self.config = config or SamplingConfig()

    async def sample(
        self,
        tenant_id: str,
        user_id: str,
        max_episodes: int | None = None,
        exclude_consolidated: bool = True,
    ) -> list[MemoryRecord]:
        """Sample episodes for consolidation."""
        max_eps = max_episodes or self.config.max_episodes

        filters: dict = {
            "status": MemoryStatus.ACTIVE.value,
            "type": [
                MemoryType.EPISODIC_EVENT.value,
                MemoryType.PREFERENCE.value,
                MemoryType.HYPOTHESIS.value,
            ],
            "since": datetime.now(UTC) - timedelta(days=self.config.time_window_days),
        }

        candidates = await self.store.scan(
            tenant_id,
            filters=filters,
            limit=max_eps * 3,
        )

        if exclude_consolidated:
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
