"""Response models matching CML API responses."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field

from cml_contracts.models import (
    ComponentStatus,
    ConfigItem,
    ConfigSection,
    CreateSessionResponse,
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
    FactItem,
    FactListResponse,
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
    MemoryStats,
    ProcessTurnResponse,
    RateLimitEntry,
    ReadMemoryResponse,
    RequestStatsResponse,
    RetrievalResultItem,
    SessionContextResponse,
    SessionInfo,
    TenantInfo,
    TenantLabileInfo,
    TimelinePoint,
    UpdateMemoryResponse,
    WriteMemoryResponse,
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


# Renamed SDK twins of the canonical server contracts, unified in
# ``cml_contracts.models`` (field-identical / value-compatible). Plain twins are
# kept as importable aliases; twins that carried a custom ``__str__`` are thin
# subclasses so that representation is preserved (fields stay single-source).
TurnResponse = ProcessTurnResponse
UpdateResponse = UpdateMemoryResponse
SessionResponse = CreateSessionResponse


class WriteResponse(WriteMemoryResponse):
    """SDK twin of canonical ``WriteMemoryResponse``; preserves the SDK ``__str__``."""

    def __str__(self) -> str:
        return f"WriteResponse(success={self.success}, chunks={self.chunks_created})"


class StatsResponse(MemoryStats):
    """SDK twin of canonical ``MemoryStats``; preserves the SDK ``__str__``."""

    def __str__(self) -> str:
        return (
            f"Memory Stats: {self.total_memories} total "
            f"({self.active_memories} active, {self.silent_memories} silent, "
            f"{self.archived_memories} archived)"
        )


class ReadResponse(ReadMemoryResponse):
    """Response from read operation.

    Field schema is the canonical ``ReadMemoryResponse`` from ``cml_contracts``;
    this SDK subclass adds the convenience ``context`` property and ``__str__``
    that SDK callers rely on.
    """

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


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str | None = None
    components: dict[str, Any] = Field(default_factory=dict)


# SDK twins of the canonical fact models (unified in cml_contracts.models).
DashboardFactItem = FactItem
DashboardFactListResponse = FactListResponse
