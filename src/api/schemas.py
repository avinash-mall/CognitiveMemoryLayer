"""API request/response schemas."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ..core.enums import MemoryScope, MemoryType


class WriteMemoryRequest(BaseModel):
    """Request to store a memory."""

    scope: MemoryScope
    scope_id: str
    content: str
    memory_type: Optional[MemoryType] = None
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    turn_id: Optional[str] = None
    agent_id: Optional[str] = None


class CreateSessionRequest(BaseModel):
    """Request to create a new memory session."""

    name: Optional[str] = None
    ttl_hours: Optional[int] = 24
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    """Response from session creation."""

    session_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None


class WriteMemoryResponse(BaseModel):
    """Response from write operation."""

    success: bool
    memory_id: Optional[UUID] = None
    chunks_created: int = 0
    message: str = ""


class ReadMemoryRequest(BaseModel):
    """Request to retrieve memories."""

    scope: MemoryScope
    scope_id: str
    query: str
    max_results: int = Field(default=10, le=50)
    memory_types: Optional[List[MemoryType]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    format: str = "packet"  # "packet", "list", "llm_context"


class MemoryItem(BaseModel):
    """A single memory item."""

    id: UUID
    text: str
    type: str
    confidence: float
    relevance: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionContextResponse(BaseModel):
    """Full session context for LLM injection."""

    session_id: str
    messages: List[MemoryItem] = Field(default_factory=list)
    tool_results: List[MemoryItem] = Field(default_factory=list)
    scratch_pad: List[MemoryItem] = Field(default_factory=list)
    context_string: str = ""


class ReadMemoryResponse(BaseModel):
    """Response from read operation."""

    query: str
    memories: List[MemoryItem]
    facts: List[MemoryItem] = Field(default_factory=list)
    preferences: List[MemoryItem] = Field(default_factory=list)
    episodes: List[MemoryItem] = Field(default_factory=list)
    llm_context: Optional[str] = None
    total_count: int
    elapsed_ms: float


class UpdateMemoryRequest(BaseModel):
    """Request to update a memory."""

    memory_id: UUID
    scope: MemoryScope
    scope_id: str
    text: Optional[str] = None
    confidence: Optional[float] = None
    importance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None  # "correct", "incorrect", "outdated"


class UpdateMemoryResponse(BaseModel):
    """Response from update operation."""

    success: bool
    memory_id: UUID
    version: int
    message: str = ""


class ForgetRequest(BaseModel):
    """Request to forget memories."""

    scope: MemoryScope
    scope_id: str
    memory_ids: Optional[List[UUID]] = None
    query: Optional[str] = None
    before: Optional[datetime] = None
    action: str = "delete"  # "delete", "archive", "silence"


class ForgetResponse(BaseModel):
    """Response from forget operation."""

    success: bool
    affected_count: int
    message: str = ""


class MemoryStats(BaseModel):
    """Memory statistics for a scope."""

    scope: MemoryScope
    scope_id: str
    total_memories: int
    active_memories: int
    silent_memories: int
    archived_memories: int
    by_type: Dict[str, int]
    avg_confidence: float
    avg_importance: float
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None
    estimated_size_mb: float
