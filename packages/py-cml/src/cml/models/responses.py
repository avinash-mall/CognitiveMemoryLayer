"""Response models matching CML API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict
from uuid import UUID

from pydantic import BaseModel, Field


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


class MemoryItem(BaseModel):
    """A single memory item from retrieval."""

    id: UUID
    text: str
    type: str
    confidence: float
    relevance: float
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class ForgetResponse(BaseModel):
    """Response from forget operation."""

    success: bool
    affected_count: int
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


class SessionContextResponse(BaseModel):
    """Full session context for LLM injection."""

    session_id: str
    messages: list[MemoryItem] = Field(default_factory=list)
    tool_results: list[MemoryItem] = Field(default_factory=list)
    scratch_pad: list[MemoryItem] = Field(default_factory=list)
    context_string: str = ""


# ---- Dashboard & Admin Schemas ----


class DashboardOverview(BaseModel):
    """Comprehensive dashboard overview stats."""

    total_memories: int = 0
    active_memories: int = 0
    silent_memories: int = 0
    compressed_memories: int = 0
    archived_memories: int = 0
    deleted_memories: int = 0
    labile_memories: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_importance: float = 0.0
    avg_access_count: float = 0.0
    avg_decay_rate: float = 0.0
    oldest_memory: datetime | None = None
    newest_memory: datetime | None = None
    estimated_size_mb: float = 0.0
    total_semantic_facts: int = 0
    current_semantic_facts: int = 0
    facts_by_category: dict[str, int] = Field(default_factory=dict)
    avg_fact_confidence: float = 0.0
    avg_evidence_count: float = 0.0
    total_events: int = 0
    events_by_type: dict[str, int] = Field(default_factory=dict)
    events_by_operation: dict[str, int] = Field(default_factory=dict)


class DashboardMemoryListItem(BaseModel):
    """Memory item for dashboard list view."""

    id: UUID
    tenant_id: str
    agent_id: str | None = None
    type: str
    status: str
    text: str
    key: str | None = None
    namespace: str | None = None
    context_tags: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    importance: float = 0.5
    access_count: int = 0
    decay_rate: float = 0.01
    labile: bool = False
    version: int = 1
    timestamp: datetime | None = None
    written_at: datetime | None = None


class DashboardMemoryListResponse(BaseModel):
    """Paginated memory list response."""

    items: list[DashboardMemoryListItem]
    total: int
    page: int
    per_page: int
    total_pages: int


class DashboardMemoryDetail(BaseModel):
    """Full memory detail for dashboard."""

    id: UUID
    tenant_id: str
    agent_id: str | None = None
    type: str
    status: str
    text: str
    key: str | None = None
    namespace: str | None = None
    context_tags: list[str] = Field(default_factory=list)
    source_session_id: str | None = None
    entities: Any = None
    relations: Any = None
    metadata: Any = None
    confidence: float = 0.5
    importance: float = 0.5
    access_count: int = 0
    last_accessed_at: datetime | None = None
    decay_rate: float = 0.01
    labile: bool = False
    provenance: Any = None
    version: int = 1
    supersedes_id: UUID | None = None
    content_hash: str | None = None
    timestamp: datetime | None = None
    written_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    related_events: list[dict[str, Any]] = Field(default_factory=list)


class DashboardEventItem(BaseModel):
    """Event log item for dashboard."""

    id: UUID
    tenant_id: str
    scope_id: str
    agent_id: str | None = None
    event_type: str
    operation: str | None = None
    payload: Any = None
    memory_ids: list[UUID] = Field(default_factory=list)
    parent_event_id: UUID | None = None
    created_at: datetime | None = None


class DashboardEventListResponse(BaseModel):
    """Paginated event list response."""

    items: list[DashboardEventItem]
    total: int
    page: int
    per_page: int
    total_pages: int


class TimelinePoint(BaseModel):
    """Single data point in a timeline."""

    date: str
    count: int


class DashboardTimelineResponse(BaseModel):
    """Timeline data for charts."""

    points: list[TimelinePoint]
    total: int


class ComponentStatus(BaseModel):
    """Health status for a single component."""

    name: str
    status: str  # "ok", "error", "degraded", "unknown"
    latency_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class DashboardComponentsResponse(BaseModel):
    """Component health report."""

    components: list[ComponentStatus]


class TenantInfo(BaseModel):
    """Tenant summary info."""

    tenant_id: str
    memory_count: int = 0
    active_memory_count: int = 0
    fact_count: int = 0
    event_count: int = 0
    last_memory_at: datetime | None = None
    last_event_at: datetime | None = None


class DashboardTenantsResponse(BaseModel):
    """List of tenants."""

    tenants: list[TenantInfo]


class SessionInfo(BaseModel):
    """Session info from Redis + DB."""

    session_id: str
    tenant_id: str | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    ttl_seconds: int = -1
    memory_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DashboardSessionsResponse(BaseModel):
    """Sessions list."""

    sessions: list[SessionInfo]
    total_active: int = 0
    total_memories_with_session: int = 0


class RateLimitEntry(BaseModel):
    """A single rate limit bucket."""

    key_type: str  # "apikey" or "ip"
    identifier: str  # masked
    current_count: int = 0
    limit: int = 60
    ttl_seconds: int = -1
    utilization_pct: float = 0.0


class DashboardRateLimitsResponse(BaseModel):
    """Rate limit entries."""

    entries: list[RateLimitEntry]
    configured_rpm: int = 60


class HourlyRequestCount(BaseModel):
    """Hourly request count."""

    hour: str  # ISO format
    count: int


class RequestStatsResponse(BaseModel):
    """Request stats over time."""

    points: list[HourlyRequestCount]
    total_last_24h: int = 0


class GraphNodeInfo(BaseModel):
    """Graph node for rendering."""

    id: str
    entity: str
    entity_type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeInfo(BaseModel):
    """Graph edge for rendering."""

    source: str
    target: str
    predicate: str
    confidence: float = 0.0
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphStatsResponse(BaseModel):
    """Graph statistics."""

    total_nodes: int = 0
    total_edges: int = 0
    entity_types: dict[str, int] = Field(default_factory=dict)
    tenants_with_graph: list[str] = Field(default_factory=list)


class GraphExploreResponse(BaseModel):
    """Graph exploration result."""

    nodes: list[GraphNodeInfo] = Field(default_factory=list)
    edges: list[GraphEdgeInfo] = Field(default_factory=list)
    center_entity: str | None = None


class GraphSearchResult(BaseModel):
    """Entity search result."""

    entity: str
    entity_type: str
    tenant_id: str
    scope_id: str


class GraphSearchResponse(BaseModel):
    """Graph search results."""

    results: list[GraphSearchResult] = Field(default_factory=list)


class GraphNeo4jConfigResponse(BaseModel):
    """Neo4j connection config for browser (neovis.js). Admin-only."""

    server_url: str
    server_user: str
    server_password: str


class ConfigItem(BaseModel):
    """A single config setting."""

    key: str
    value: Any = None
    default_value: Any = None
    is_secret: bool = False
    is_editable: bool = False
    source: str = "default"  # "default", "env", "override"
    description: str = ""
    requires_restart: bool = True
    is_required: bool = False
    env_var: str = ""
    options: list[str] | None = None  # If set, UI shows dropdown with these values


class ConfigSection(BaseModel):
    """A group of config settings."""

    name: str
    items: list[ConfigItem] = Field(default_factory=list)


class DashboardConfigResponse(BaseModel):
    """Full config snapshot."""

    sections: list[ConfigSection] = Field(default_factory=list)


class TenantLabileInfo(BaseModel):
    """Labile state info per tenant."""

    tenant_id: str
    db_labile_count: int = 0
    redis_scope_count: int = 0
    redis_session_count: int = 0
    redis_memory_count: int = 0


class DashboardLabileResponse(BaseModel):
    """Labile state overview."""

    tenants: list[TenantLabileInfo] = Field(default_factory=list)
    total_db_labile: int = 0
    total_redis_scopes: int = 0
    total_redis_sessions: int = 0
    total_redis_memories: int = 0


class RetrievalResultItem(BaseModel):
    """A single retrieval result with score."""

    id: UUID
    text: str
    type: str
    confidence: float
    relevance_score: float
    retrieval_source: str = ""
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DashboardRetrievalResponse(BaseModel):
    """Retrieval test results."""

    query: str
    results: list[RetrievalResultItem] = Field(default_factory=list)
    llm_context: str | None = None
    total_count: int = 0
    elapsed_ms: float = 0.0


class DashboardJobItem(BaseModel):
    """A single job history entry."""

    id: UUID
    job_type: str
    tenant_id: str
    user_id: str | None = None
    dry_run: bool = False
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None


class DashboardJobsResponse(BaseModel):
    """Paginated job list."""

    items: list[DashboardJobItem]
    total: int
