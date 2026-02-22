"""Core Pydantic schemas for memory records, events, and retrieval."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enums import MemorySource, MemoryStatus, MemoryType, OperationType


class Provenance(BaseModel):
    """Tracks origin and evidence for a memory."""

    source: MemorySource
    evidence_refs: list[str] = Field(default_factory=list)  # Event IDs, turn IDs
    tool_refs: list[str] = Field(default_factory=list)  # Tool call IDs
    model_version: str | None = None  # Extraction model
    extraction_prompt_hash: str | None = None  # For reproducibility


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
    confidence: float = 0.5
    importance: float = 0.5
    decay_rate: float | None = None
    provenance: Provenance


class EventLog(BaseModel):
    """Immutable event log entry."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    scope_id: str
    agent_id: str | None = None

    event_type: str  # "turn", "memory_op", "consolidation", etc.
    operation: OperationType | None = None

    # Content
    payload: dict[str, Any]  # Full turn data or operation details

    # References
    memory_ids: list[UUID] = Field(default_factory=list)  # Affected memories
    parent_event_id: UUID | None = None  # For chaining

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Audit
    ip_address: str | None = None
    user_agent: str | None = None


class MemoryOperation(BaseModel):
    """Planned operation on a memory."""

    op: OperationType
    record_id: UUID | None = None  # For UPDATE/DELETE
    record: MemoryRecordCreate | None = None  # For ADD
    patch: dict[str, Any] | None = None  # For UPDATE
    reason: str = ""
    confidence: float = 1.0


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

    # Meta
    open_questions: list[str] = Field(default_factory=list)  # Needs confirmation
    warnings: list[str] = Field(default_factory=list)  # Conflicts detected

    @property
    def all_memories(self) -> list[RetrievedMemory]:
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
                    lines.append(f"- {m.record.text} (confidence: {m.record.confidence:.2f})")

        result = "\n".join(lines)
        if len(result) <= max_chars:
            return result
        # Truncate at the last newline boundary to avoid mid-line/mid-character cuts (LOW-04)
        truncated = result[:max_chars]
        last_nl = truncated.rfind("\n")
        if last_nl > 0:
            return truncated[:last_nl] + "\n..."
        return truncated + "\n..."
