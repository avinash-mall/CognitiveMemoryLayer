"""Internal request models for constructing API payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .enums import MemoryType


class WriteRequest(BaseModel):
    """Write memory request payload."""

    content: str
    context_tags: list[str] | None = None
    session_id: str | None = None
    memory_type: MemoryType | None = None
    namespace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    turn_id: str | None = None
    agent_id: str | None = None
    timestamp: datetime | None = None


class ReadRequest(BaseModel):
    """Read memory request payload."""

    model_config = ConfigDict(populate_by_name=True)

    query: str
    max_results: int = Field(default=10, ge=1, le=50)
    context_filter: list[str] | None = None
    memory_types: list[MemoryType] | None = None
    since: datetime | None = None
    until: datetime | None = None
    response_format: Literal["packet", "list", "llm_context"] = Field(
        default="packet", alias="format"
    )
    user_timezone: str | None = (
        None  # IANA timezone (e.g. "America/New_York") for "today"/"yesterday"
    )


class TurnRequest(BaseModel):
    """Seamless turn request payload."""

    user_message: str
    assistant_response: str | None = None
    session_id: str | None = None
    max_context_tokens: int = 1500
    timestamp: datetime | None = None
    user_timezone: str | None = None  # IANA timezone for retrieval "today"/"yesterday"


class UpdateRequest(BaseModel):
    """Update memory request payload."""

    memory_id: UUID
    text: str | None = None
    confidence: float | None = None
    importance: float | None = None
    metadata: dict[str, Any] | None = None
    feedback: str | None = None


class ForgetRequest(BaseModel):
    """Forget memories request payload."""

    memory_ids: list[UUID] | None = None
    query: str | None = None
    before: datetime | None = None
    action: Literal["delete", "archive", "silence"] = "delete"


class CreateSessionRequest(BaseModel):
    """Request to create a new memory session."""

    name: str | None = None
    ttl_hours: int = 24
    metadata: dict[str, Any] = Field(default_factory=dict)


class DashboardConsolidateRequest(BaseModel):
    """Request to trigger consolidation."""

    tenant_id: str
    user_id: str | None = None


class DashboardForgetRequest(BaseModel):
    """Request to trigger forgetting."""

    tenant_id: str
    user_id: str | None = None
    dry_run: bool = True
    max_memories: int = 5000


class DashboardReconsolidateRequest(BaseModel):
    """Request to trigger reconsolidation."""

    tenant_id: str
    user_id: str | None = None


class ConfigUpdateRequest(BaseModel):
    """Request to update config settings."""

    updates: dict[str, Any] = Field(default_factory=dict)


class DashboardRetrievalRequest(BaseModel):
    """Request to test memory retrieval."""

    tenant_id: str
    query: str
    max_results: int = Field(default=10, le=50)
    context_filter: list[str] | None = None
    memory_types: list[str] | None = None
    format: Literal["packet", "list", "llm_context"] = "list"


class BulkActionRequest(BaseModel):
    """Request for bulk memory actions."""

    memory_ids: list[UUID]
    action: Literal["archive", "silence", "delete"]
