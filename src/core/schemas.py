"""Core Pydantic schemas for memory records, events, and retrieval."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enums import MemorySource, MemoryStatus, MemoryType, OperationType


class Provenance(BaseModel):
    """Tracks origin and evidence for a memory."""

    source: MemorySource
    evidence_refs: List[str] = Field(default_factory=list)  # Event IDs, turn IDs
    tool_refs: List[str] = Field(default_factory=list)  # Tool call IDs
    model_version: Optional[str] = None  # Extraction model
    extraction_prompt_hash: Optional[str] = None  # For reproducibility


class EntityMention(BaseModel):
    """An entity mentioned in the memory."""

    text: str
    normalized: str  # Canonical form
    entity_type: str  # PERSON, LOCATION, ORG, DATE, etc.
    start_char: Optional[int] = None
    end_char: Optional[int] = None


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
    context_tags: List[str] = Field(default_factory=list)  # Flexible categorization
    source_session_id: Optional[str] = None  # Origin tracking (not retrieval filter)
    agent_id: Optional[str] = None
    namespace: Optional[str] = None

    # Type and content
    type: MemoryType
    text: str  # Human-readable content
    key: Optional[str] = None  # Unique key for facts (e.g., "user:location")
    embedding: Optional[List[float]] = None  # Dense vector

    # Structured extractions
    entities: List[EntityMention] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Temporal validity
    timestamp: datetime = Field(default_factory=datetime.utcnow)  # When event occurred
    written_at: datetime = Field(default_factory=datetime.utcnow)  # When stored
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    # Scoring
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # Usage tracking
    access_count: int = Field(default=0)
    last_accessed_at: Optional[datetime] = None
    decay_rate: float = Field(default=0.01)  # Per day

    # Status
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    labile: bool = Field(default=False)  # Currently being reconsolidated

    # Provenance
    provenance: Provenance

    # Versioning
    version: int = Field(default=1)
    supersedes_id: Optional[UUID] = None  # Previous version

    # Deduplication
    content_hash: Optional[str] = None


class MemoryRecordCreate(BaseModel):
    """Schema for creating a new memory."""

    tenant_id: str
    context_tags: List[str] = Field(default_factory=list)
    source_session_id: Optional[str] = None
    agent_id: Optional[str] = None
    namespace: Optional[str] = None
    type: MemoryType
    text: str
    key: Optional[str] = None
    embedding: Optional[List[float]] = None
    entities: List[EntityMention] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None
    confidence: float = 0.5
    importance: float = 0.5
    provenance: Provenance


class EventLog(BaseModel):
    """Immutable event log entry."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    scope_id: str
    agent_id: Optional[str] = None

    event_type: str  # "turn", "memory_op", "consolidation", etc.
    operation: Optional[OperationType] = None

    # Content
    payload: Dict[str, Any]  # Full turn data or operation details

    # References
    memory_ids: List[UUID] = Field(default_factory=list)  # Affected memories
    parent_event_id: Optional[UUID] = None  # For chaining

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Audit
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class MemoryOperation(BaseModel):
    """Planned operation on a memory."""

    op: OperationType
    record_id: Optional[UUID] = None  # For UPDATE/DELETE
    record: Optional[MemoryRecordCreate] = None  # For ADD
    patch: Optional[Dict[str, Any]] = None  # For UPDATE
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
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)

    # Categorized memories
    facts: List[RetrievedMemory] = Field(default_factory=list)
    recent_episodes: List[RetrievedMemory] = Field(default_factory=list)
    preferences: List[RetrievedMemory] = Field(default_factory=list)
    procedures: List[RetrievedMemory] = Field(default_factory=list)
    constraints: List[RetrievedMemory] = Field(default_factory=list)

    # Meta
    open_questions: List[str] = Field(default_factory=list)  # Needs confirmation
    warnings: List[str] = Field(default_factory=list)  # Conflicts detected

    @property
    def all_memories(self) -> List[RetrievedMemory]:
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
        return result[:max_chars]
