"""Working memory data structures: chunks and state."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ChunkType(StrEnum):
    """Type of semantic chunk."""

    STATEMENT = "statement"
    PREFERENCE = "preference"
    QUESTION = "question"
    INSTRUCTION = "instruction"
    FACT = "fact"
    EVENT = "event"
    OPINION = "opinion"
    CONSTRAINT = "constraint"


@dataclass
class SemanticChunk:
    """A semantically coherent unit of information."""

    id: str
    text: str
    chunk_type: ChunkType

    source_turn_id: str | None = None
    source_role: str | None = None
    start_idx: int | None = None
    end_idx: int | None = None

    entities: list[str] = field(default_factory=list)
    key_phrases: list[str] = field(default_factory=list)

    salience: float = 0.5
    novelty: float = 0.5
    confidence: float = 1.0

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemoryState:
    """Current state of working memory for a user."""

    tenant_id: str
    user_id: str

    chunks: list[SemanticChunk] = field(default_factory=list)
    max_chunks: int = 10

    current_topic: str | None = None
    current_intent: str | None = None

    turn_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_chunk(self, chunk: SemanticChunk) -> None:
        """Add chunk. Eviction: keep most recent N by recency, evict from older by salience."""
        self.chunks.append(chunk)
        self.last_updated = datetime.now(UTC)

        if len(self.chunks) <= self.max_chunks:
            return

        # Keep most recent N chunks regardless of salience
        recent_keep = min(3, self.max_chunks // 3)
        by_time = sorted(
            self.chunks,
            key=lambda c: c.timestamp,
            reverse=True,
        )
        recent_chunks = by_time[:recent_keep]

        # Evict from older chunks based on salience
        older_chunks = [c for c in self.chunks if c not in recent_chunks]
        older_chunks.sort(key=lambda c: c.salience, reverse=True)
        keep_older = self.max_chunks - recent_keep
        self.chunks = older_chunks[:keep_older] + recent_chunks
        self.chunks.sort(key=lambda c: c.timestamp)

    def get_high_salience_chunks(self, min_salience: float = 0.5) -> list[SemanticChunk]:
        """Get chunks above salience threshold."""
        return [c for c in self.chunks if c.salience >= min_salience]
