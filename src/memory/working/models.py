"""Working memory data structures: chunks and state."""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

_PROPER_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}\b")
_DEICTIC_PRONOUN_RE = re.compile(
    r"\b(?:he|she|they|him|her|them|his|hers|their|there|that place|that city|that trip|that company)\b",
    re.IGNORECASE,
)
_IGNORE_CAP_WORDS = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Why",
        "How",
        "Yes",
        "Yeah",
        "No",
        "Not",
        "And",
        "But",
        "For",
        "With",
        "From",
        "Have",
        "Had",
        "Are",
        "Was",
        "Were",
        "Will",
        "Would",
        "Could",
        "Should",
        "Can",
        "Also",
        "Just",
        "Really",
        "Very",
        "Today",
        "Yesterday",
        "Tomorrow",
    }
)


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
    active_entities: list[str] = field(default_factory=list)

    turn_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def bind_episodic_buffer_context(self, chunk: SemanticChunk) -> None:
        """Baddeley Episodic Buffer: track active entities and bind coreference focus."""
        if not chunk.text:
            return
        extracted = [
            m.group(0)
            for m in _PROPER_ENTITY_RE.finditer(chunk.text)
            if m.group(0) not in _IGNORE_CAP_WORDS
        ]
        # If chunk has unresolved pronouns/deictics, attach preceding active entities
        if self.active_entities and _DEICTIC_PRONOUN_RE.search(chunk.text):
            inherited = [e for e in self.active_entities if e not in extracted][:3]
            if inherited:
                chunk.metadata["working_memory_focus"] = inherited
                for ent in inherited:
                    if ent not in chunk.entities:
                        chunk.entities.append(ent)

        # Update active entity focus (most recent first, max 7±2 items)
        for ent in reversed(extracted):
            if ent in self.active_entities:
                self.active_entities.remove(ent)
            self.active_entities.insert(0, ent)
            if ent not in chunk.entities:
                chunk.entities.append(ent)
        self.active_entities = self.active_entities[:7]
        if self.active_entities:
            self.current_topic = self.active_entities[0]

    def add_chunk(self, chunk: SemanticChunk) -> None:
        """Add chunk. Eviction: keep most recent N by recency, evict from older by salience."""
        self.bind_episodic_buffer_context(chunk)
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
