"""Working memory data structures: chunks and state."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ChunkType(str, Enum):
    """Type of semantic chunk."""

    STATEMENT = "statement"
    PREFERENCE = "preference"
    QUESTION = "question"
    INSTRUCTION = "instruction"
    FACT = "fact"
    EVENT = "event"
    OPINION = "opinion"


@dataclass
class SemanticChunk:
    """A semantically coherent unit of information."""

    id: str
    text: str
    chunk_type: ChunkType

    source_turn_id: Optional[str] = None
    source_role: Optional[str] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

    entities: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)

    salience: float = 0.5
    novelty: float = 0.5
    confidence: float = 1.0

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemoryState:
    """Current state of working memory for a user."""

    tenant_id: str
    user_id: str

    chunks: List[SemanticChunk] = field(default_factory=list)
    max_chunks: int = 10

    current_topic: Optional[str] = None
    current_intent: Optional[str] = None

    turn_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_chunk(self, chunk: SemanticChunk) -> None:
        """Add chunk. Eviction: keep most recent N by recency, evict from older by salience."""
        self.chunks.append(chunk)
        self.last_updated = datetime.now(timezone.utc)

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

    def get_high_salience_chunks(
        self, min_salience: float = 0.5
    ) -> List[SemanticChunk]:
        """Get chunks above salience threshold."""
        return [c for c in self.chunks if c.salience >= min_salience]
