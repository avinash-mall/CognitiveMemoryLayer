"""Core Pydantic schemas for memory records, events, and retrieval."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enums import MemorySource, MemoryStatus, MemoryType


class Provenance(BaseModel):
    """Tracks origin and evidence for a memory."""

    source: MemorySource
    evidence_refs: list[str] = Field(default_factory=list)  # Event IDs, turn IDs
    tool_refs: list[str] = Field(default_factory=list)  # Tool call IDs
    model_version: str | None = None  # Extraction model
    extraction_prompt_hash: str | None = None  # For reproducibility


# Source monitoring: what the model is told about where a memory came from.
# Testimony (the user said it) is the unmarked default so the common case stays
# terse; everything the system produced itself is marked, because rendering an
# inference identically to a user statement invites the model to quote its own
# speculation back as fact.
_SOURCE_LABELS: dict[MemorySource, str] = {
    MemorySource.AGENT_INFERRED: "inferred",
    MemorySource.CONSOLIDATION: "consolidated",
    MemorySource.RECONSOLIDATION: "revised",
    MemorySource.TOOL_RESULT: "tool output",
}


def source_label(record: Any) -> str:
    """Return a bracketed origin marker, or '' for direct user testimony.

    Every renderer that puts memory text in front of a model routes through this,
    so agent-authored content cannot be presented as something the user said.
    """
    source = getattr(getattr(record, "provenance", None), "source", None)
    label = _SOURCE_LABELS.get(source) if isinstance(source, MemorySource) else None
    return f" [{label}]" if label else ""


class EntityMention(BaseModel):
    """An entity mentioned in the memory."""

    text: str
    normalized: str  # Canonical form
    entity_type: str  # PERSON, LOCATION, ORG, DATE, etc.
    start_char: int | None = None
    end_char: int | None = None


class Relation(BaseModel):
    """A relation triple extracted from memory."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


class MemoryRecord(BaseModel):
    """Core memory record stored in the system."""

    model_config = ConfigDict(from_attributes=True)

    # Identity
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    context_tags: list[str] = Field(default_factory=list)  # Flexible categorization
    source_session_id: str | None = None  # Origin tracking (not retrieval filter)
    agent_id: str | None = None
    namespace: str | None = None

    # Type and content
    type: MemoryType
    text: str  # Human-readable content
    key: str | None = None  # Unique key for facts (e.g., "user:location")
    embedding: list[float] | None = None  # Dense vector

    # Structured extractions
    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Temporal validity
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))  # When event occurred
    written_at: datetime = Field(default_factory=lambda: datetime.now(UTC))  # When stored
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    # Scoring
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # Usage tracking
    access_count: int = Field(default=0)
    last_accessed_at: datetime | None = None
    decay_rate: float = Field(default=0.01)  # Per day

    # Status
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    labile: bool = Field(default=False)  # Currently being reconsolidated

    # Provenance
    provenance: Provenance

    # Versioning
    version: int = Field(default=1)
    supersedes_id: UUID | None = None  # Previous version

    # Deduplication
    content_hash: str | None = None


class MemoryRecordCreate(BaseModel):
    """Schema for creating a new memory."""

    tenant_id: str
    context_tags: list[str] = Field(default_factory=list)
    source_session_id: str | None = None
    agent_id: str | None = None
    namespace: str | None = None
    type: MemoryType
    text: str
    key: str | None = None
    embedding: list[float] | None = None
    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime | None = None
    # When the described state of affairs held, as opposed to when the turn was
    # written. The columns and the reader already existed — vector_search's
    # exclude_expired filters on valid_to — but nothing could set them on a write.
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 0.5
    importance: float = 0.5
    decay_rate: float | None = None
    provenance: Provenance


class RetrievedMemory(BaseModel):
    """A single retrieved memory with score."""

    record: MemoryRecord
    relevance_score: float
    retrieval_source: str  # "vector", "graph", "lexical", "cache"


class MemoryPacket(BaseModel):
    """Structured bundle returned from retrieval."""

    query: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Categorized memories
    facts: list[RetrievedMemory] = Field(default_factory=list)
    recent_episodes: list[RetrievedMemory] = Field(default_factory=list)
    preferences: list[RetrievedMemory] = Field(default_factory=list)
    procedures: list[RetrievedMemory] = Field(default_factory=list)
    constraints: list[RetrievedMemory] = Field(default_factory=list)

    # Global reranked order (preserves reranker relevance ranking)
    ranked_memories: list[RetrievedMemory] = Field(default_factory=list)

    # Meta
    open_questions: list[str] = Field(default_factory=list)  # Needs confirmation
    warnings: list[str] = Field(default_factory=list)  # Conflicts detected
    retrieval_meta: dict[str, Any] | None = None
    sufficiency: dict[str, Any] | None = None  # Is there evidence here at all?

    @property
    def all_memories(self) -> list[RetrievedMemory]:
        if self.ranked_memories:
            return list(self.ranked_memories)
        return (
            self.facts
            + self.recent_episodes
            + self.preferences
            + self.procedures
            + self.constraints
        )

    def to_context_string(self, max_chars: int = 4000) -> str:
        """Format for LLM context injection."""
        lines = []
        for category, memories in [
            ("Facts", self.facts),
            ("Preferences", self.preferences),
            ("Recent Events", self.recent_episodes),
            ("Procedures", self.procedures),
            ("Constraints", self.constraints),
        ]:
            if memories:
                lines.append(f"## {category}")
                for m in memories[:5]:  # Limit per category
                    lines.append(
                        f"- {m.record.text}{source_label(m.record)} "
                        f"(confidence: {m.record.confidence:.2f})"
                    )

        result = "\n".join(lines)
        if len(result) <= max_chars:
            return result
        # Truncate at the last newline boundary to avoid mid-line/mid-character cuts (LOW-04)
        truncated = result[:max_chars]
        last_nl = truncated.rfind("\n")
        if last_nl > 0:
            return truncated[:last_nl] + "\n..."
        return truncated + "\n..."
