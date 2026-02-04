"""Relevance scoring for active forgetting."""
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.enums import MemoryType
from ..core.schemas import MemoryRecord


@dataclass
class RelevanceWeights:
    """Weights for relevance score components."""

    importance: float = 0.25
    recency: float = 0.20
    frequency: float = 0.20
    confidence: float = 0.15
    type_bonus: float = 0.10
    dependency: float = 0.10

    def validate(self) -> None:
        total = (
            self.importance
            + self.recency
            + self.frequency
            + self.confidence
            + self.type_bonus
            + self.dependency
        )
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"


@dataclass
class RelevanceScore:
    """Detailed relevance score breakdown."""

    memory_id: str
    total_score: float

    importance_score: float
    recency_score: float
    frequency_score: float
    confidence_score: float
    type_bonus_score: float
    dependency_score: float

    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    suggested_action: str = "keep"  # keep, decay, silence, compress, delete


@dataclass
class ScorerConfig:
    """Configuration for RelevanceScorer."""

    weights: RelevanceWeights = field(default_factory=RelevanceWeights)
    recency_half_life_days: float = 30.0
    frequency_log_base: float = 10.0
    type_bonuses: Dict[str, float] = field(
        default_factory=lambda: {
            MemoryType.CONSTRAINT.value: 1.0,
            MemoryType.PREFERENCE.value: 0.8,
            MemoryType.SEMANTIC_FACT.value: 0.7,
            MemoryType.PROCEDURE.value: 0.6,
            MemoryType.EPISODIC_EVENT.value: 0.3,
            MemoryType.HYPOTHESIS.value: 0.2,
            MemoryType.TASK_STATE.value: 0.1,
        }
    )
    keep_threshold: float = 0.7
    decay_threshold: float = 0.4
    silence_threshold: float = 0.2
    compress_threshold: float = 0.1


class RelevanceScorer:
    """
    Calculates relevance scores for memories.

    Mimics biological forgetting where important, frequently accessed,
    recent, and high-confidence things are remembered.
    """

    def __init__(self, config: Optional[ScorerConfig] = None) -> None:
        self.config = config or ScorerConfig()
        self.config.weights.validate()

    def score(
        self,
        record: MemoryRecord,
        dependency_count: int = 0,
    ) -> RelevanceScore:
        """Calculate relevance score for a memory."""
        importance = record.importance

        ts = record.timestamp
        now = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (now - ts).total_seconds() / 86400
        recency = pow(0.5, age_days / self.config.recency_half_life_days)

        frequency = math.log(
            1 + record.access_count, self.config.frequency_log_base
        )
        frequency = min(frequency, 1.0)

        confidence = record.confidence

        record_type = (
            record.type if isinstance(record.type, str) else record.type.value
        )
        type_bonus = self.config.type_bonuses.get(record_type, 0.5)

        dependency = min(dependency_count / 10.0, 1.0)

        w = self.config.weights
        total = (
            w.importance * importance
            + w.recency * recency
            + w.frequency * frequency
            + w.confidence * confidence
            + w.type_bonus * type_bonus
            + w.dependency * dependency
        )

        suggested = self._suggest_action(total, record_type)

        return RelevanceScore(
            memory_id=str(record.id),
            total_score=total,
            importance_score=importance,
            recency_score=recency,
            frequency_score=frequency,
            confidence_score=confidence,
            type_bonus_score=type_bonus,
            dependency_score=dependency,
            suggested_action=suggested,
        )

    def score_batch(
        self,
        records: List[MemoryRecord],
        dependency_counts: Optional[Dict[str, int]] = None,
    ) -> List[RelevanceScore]:
        """Score multiple records."""
        dep_counts = dependency_counts or {}
        return [self.score(r, dep_counts.get(str(r.id), 0)) for r in records]

    def _suggest_action(self, score: float, memory_type: str) -> str:
        """Suggest forgetting action based on score."""
        if memory_type == MemoryType.CONSTRAINT.value:
            return "keep"
        if score >= self.config.keep_threshold:
            return "keep"
        if score >= self.config.decay_threshold:
            return "decay"
        if score >= self.config.silence_threshold:
            return "silence"
        if score >= self.config.compress_threshold:
            return "compress"
        return "delete"
