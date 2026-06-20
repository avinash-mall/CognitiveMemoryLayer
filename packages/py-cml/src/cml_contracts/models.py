"""Shared API-contract Pydantic models for the CognitiveMemoryLayer API.

Single source of truth imported by both the server (``src``) and the SDK
(``cml``). Class bodies are relocated verbatim from ``src/api/schemas.py``
(the canonical definitions) so the server and SDK can never drift.

Depends only on ``pydantic`` and the shared ``cml_contracts.enums``.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """A single memory item."""

    id: UUID
    text: str
    type: str
    confidence: float
    relevance: float
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionContextResponse(BaseModel):
    """Full session context for LLM injection."""

    session_id: str
    messages: list[MemoryItem] = Field(default_factory=list)
    tool_results: list[MemoryItem] = Field(default_factory=list)
    scratch_pad: list[MemoryItem] = Field(default_factory=list)
    context_string: str = ""


class ForgetResponse(BaseModel):
    """Response from forget operation."""

    success: bool
    affected_count: int
    message: str = ""


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
    supersedes_id: UUID | None = None
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
    """Request to trigger reconsolidation (release all labile state for tenant)."""

    tenant_id: str
    user_id: str | None = None


class ConfigUpdateRequest(BaseModel):
    """Request to update config settings."""

    updates: dict[str, Any] = Field(default_factory=dict)


class BulkActionRequest(BaseModel):
    """Request for bulk memory actions."""

    memory_ids: list[UUID]
    action: Literal["archive", "silence", "delete"]
