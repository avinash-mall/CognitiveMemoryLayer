"""Response models matching CML API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict
from uuid import UUID

from pydantic import BaseModel, Field

from cml_contracts.models import (
    ComponentStatus,
    ConfigItem,
    ConfigSection,
    DashboardComponentsResponse,
    DashboardConfigResponse,
    DashboardEventItem,
    DashboardEventListResponse,
    DashboardJobItem,
    DashboardJobsResponse,
    DashboardLabileResponse,
    DashboardMemoryDetail,
    DashboardMemoryListItem,
    DashboardMemoryListResponse,
    DashboardOverview,
    DashboardRateLimitsResponse,
    DashboardRetrievalResponse,
    DashboardSessionsResponse,
    DashboardTenantsResponse,
    DashboardTimelineResponse,
    ForgetResponse,
    GraphEdgeInfo,
    GraphExploreResponse,
    GraphNeo4jConfigResponse,
    GraphNodeInfo,
    GraphSearchResponse,
    GraphSearchResult,
    GraphStatsResponse,
    HourlyRequestCount,
    MemoryItem,
    RateLimitEntry,
    RequestStatsResponse,
    RetrievalResultItem,
    SessionContextResponse,
    SessionInfo,
    TenantInfo,
    TenantLabileInfo,
    TimelinePoint,
)

__all__ = [
    "ComponentStatus",
    "ConfigItem",
    "ConfigSection",
    "ConsolidationResult",
    "DashboardComponentsResponse",
    "DashboardConfigResponse",
    "DashboardEventItem",
    "DashboardEventListResponse",
    "DashboardFactItem",
    "DashboardFactListResponse",
    "DashboardJobItem",
    "DashboardJobsResponse",
    "DashboardLabileResponse",
    "DashboardMemoryDetail",
    "DashboardMemoryListItem",
    "DashboardMemoryListResponse",
    "DashboardOverview",
    "DashboardRateLimitsResponse",
    "DashboardRetrievalResponse",
    "DashboardSessionsResponse",
    "DashboardTenantsResponse",
    "DashboardTimelineResponse",
    "ForgetResponse",
    "ForgettingResult",
    "GraphEdgeInfo",
    "GraphExploreResponse",
    "GraphNeo4jConfigResponse",
    "GraphNodeInfo",
    "GraphSearchResponse",
    "GraphSearchResult",
    "GraphStatsResponse",
    "HealthResponse",
    "HourlyRequestCount",
    "MemoryItem",
    "RateLimitEntry",
    "ReadResponse",
    "ReconsolidationResult",
    "RequestStatsResponse",
    "RetrievalResultItem",
    "SessionContextResponse",
    "SessionInfo",
    "SessionResponse",
    "StatsResponse",
    "TenantInfo",
    "TenantLabileInfo",
    "TimelinePoint",
    "TurnResponse",
    "UpdateResponse",
    "WriteResponse",
]


class ConsolidationResult(TypedDict, total=False):
    """Typed dict for consolidate() result. Server may return additional fields."""

    episodes_sampled: int
    clusters_formed: int
    facts_extracted: int
    gists_extracted: int
    migrations_completed: int


class ForgettingResult(TypedDict, total=False):
    """Typed dict for run_forgetting() result. Server may return additional fields."""

    total_scored: int
    memories_scanned: int
    actions: dict[str, int]
    operations_applied: int
    dry_run: bool


class ReconsolidationResult(TypedDict, total=False):
    """Typed dict for reconsolidate() result. Server may return additional fields."""

    status: str
    tenant_id: str
    sessions_released: int


class WriteResponse(BaseModel):
    """Response from write operation. When server receives X-Eval-Mode: true, includes eval_outcome and eval_reason."""

    success: bool
    memory_id: UUID | None = None
    chunks_created: int = 0
    message: str = ""
    eval_outcome: str | None = None  # "stored" | "skipped" when X-Eval-Mode header was set
    eval_reason: str | None = None  # Write-gate reason when X-Eval-Mode header was set

    def __str__(self) -> str:
        return f"WriteResponse(success={self.success}, chunks={self.chunks_created})"


class ReadResponse(BaseModel):
    """Response from read operation."""

    query: str
    memories: list[MemoryItem]
    facts: list[MemoryItem] = Field(default_factory=list)
    preferences: list[MemoryItem] = Field(default_factory=list)
    episodes: list[MemoryItem] = Field(default_factory=list)
    constraints: list[MemoryItem] = Field(default_factory=list)
    llm_context: str | None = None
    total_count: int
    elapsed_ms: float
    retrieval_meta: dict | None = (
        None  # Server: sources_completed, sources_timed_out, total_elapsed_ms
    )

    @property
    def context(self) -> str:
        """Shortcut to get formatted LLM context string."""
        return self.llm_context or ""

    def __str__(self) -> str:
        lines = [f"Query: {self.query} ({self.total_count} results, {self.elapsed_ms:.0f}ms)"]
        for mem in self.memories[:10]:
            text_snippet = mem.text[:80] + "..." if len(mem.text) > 80 else mem.text
            lines.append(f"  [{mem.type}] {text_snippet} (rel={mem.relevance:.2f})")
        return "\n".join(lines)


class TurnResponse(BaseModel):
    """Response from seamless turn processing."""

    memory_context: str
    memories_retrieved: int
    memories_stored: int
    reconsolidation_applied: bool = False


class UpdateResponse(BaseModel):
    """Response from update operation."""

    success: bool
    memory_id: UUID
    version: int
    message: str = ""


class StatsResponse(BaseModel):
    """Memory statistics response."""

    total_memories: int
    active_memories: int
    silent_memories: int
    archived_memories: int
    by_type: dict[str, int]
    avg_confidence: float
    avg_importance: float
    oldest_memory: datetime | None = None
    newest_memory: datetime | None = None
    estimated_size_mb: float

    def __str__(self) -> str:
        return (
            f"Memory Stats: {self.total_memories} total "
            f"({self.active_memories} active, {self.silent_memories} silent, "
            f"{self.archived_memories} archived)"
        )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str | None = None
    components: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Session creation response."""

    session_id: str
    created_at: datetime
    expires_at: datetime | None = None


class DashboardFactItem(BaseModel):
    """Semantic fact row for dashboard facts view."""

    id: str
    tenant_id: str
    category: str
    key: str
    value: str
    confidence: float
    evidence_count: int
    is_current: bool
    version: int
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DashboardFactListResponse(BaseModel):
    """List response for dashboard semantic facts."""

    items: list[DashboardFactItem] = Field(default_factory=list)
    total: int = 0
