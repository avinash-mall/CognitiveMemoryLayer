"""Forgetting actions and policy engine."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from .scorer import RelevanceScore


class ForgettingAction(StrEnum):
    """Type of forgetting action."""

    KEEP = "keep"
    DECAY = "decay"
    SILENCE = "silence"
    COMPRESS = "compress"
    ARCHIVE = "archive"
    DELETE = "delete"


@dataclass
class ForgettingOperation:
    """A forgetting operation to apply."""

    action: ForgettingAction
    memory_id: UUID
    new_confidence: float | None = None
    compressed_text: str | None = None
    reason: str = ""
    relevance_score: float = 0.0


@dataclass
class ForgettingResult:
    """Result of applying forgetting operations."""

    operations_planned: int
    operations_applied: int
    kept: int = 0
    decayed: int = 0
    silenced: int = 0
    compressed: int = 0
    archived: int = 0
    deleted: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


class ForgettingPolicyEngine:
    """Applies forgetting policies to memories based on relevance scores."""

    def __init__(
        self,
        decay_rate: float = 0.1,
        min_confidence: float = 0.05,
        compression_max_chars: int = 100,
    ) -> None:
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence
        self.compression_max_chars = compression_max_chars

    def plan_operations(
        self,
        scores: list[RelevanceScore],
        max_operations: int | None = None,
    ) -> list[ForgettingOperation]:
        """Plan forgetting operations based on scores."""
        operations: list[ForgettingOperation] = []

        for score in scores:
            try:
                action = ForgettingAction(score.suggested_action)
            except ValueError:
                continue

            if action == ForgettingAction.KEEP:
                continue

            op = ForgettingOperation(
                action=action,
                memory_id=UUID(score.memory_id),
                relevance_score=score.total_score,
                reason=f"Score {score.total_score:.2f} below threshold",
            )

            if action == ForgettingAction.DECAY:
                op.new_confidence = max(
                    self.min_confidence,
                    score.confidence_score - self.decay_rate,
                )

            operations.append(op)
            if max_operations and len(operations) >= max_operations:
                break

        return operations

    def create_compression(self, text: str) -> str:
        """Create compressed version of text."""
        if len(text) <= self.compression_max_chars:
            return text
        return text[: self.compression_max_chars - 3] + "..."
