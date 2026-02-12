"""API request/response schemas. Holistic: no scopes, tenant-only."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ..core.enums import MemoryType


class WriteMemoryRequest(BaseModel):
    """Request to store a memory. Holistic: tenant-only."""

    content: str
    context_tags: Optional[List[str]] = None
    session_id: Optional[str] = None
    memory_type: Optional[MemoryType] = None
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    turn_id: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: Optional[datetime] = None  # Optional event timestamp (defaults to now)


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
    # Eval mode (when X-Eval-Mode: true): outcome and reason from write gate
    eval_outcome: Optional[Literal["stored", "skipped"]] = None
    eval_reason: Optional[str] = None


class ReadMemoryRequest(BaseModel):
    """Request to retrieve memories. Holistic: tenant-only."""

    query: str
    max_results: int = Field(default=10, le=50)
    context_filter: Optional[List[str]] = None
    memory_types: Optional[List[MemoryType]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    format: Literal["packet", "list", "llm_context"] = "packet"


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


class ProcessTurnRequest(BaseModel):
    """Request to process a conversation turn with seamless memory (auto-retrieve + optional auto-store)."""

    user_message: str
    assistant_response: Optional[str] = None
    session_id: Optional[str] = None
    max_context_tokens: int = 1500
    timestamp: Optional[datetime] = None  # Optional event timestamp for the turn (defaults to now)


class ProcessTurnResponse(BaseModel):
    """Response from process_turn: memory context ready to inject into prompt."""

    memory_context: str
    memories_retrieved: int
    memories_stored: int
    reconsolidation_applied: bool = False


class UpdateMemoryRequest(BaseModel):
    """Request to update a memory. Holistic: tenant-only."""

    memory_id: UUID
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
    """Request to forget memories. Holistic: tenant-only."""

    memory_ids: Optional[List[UUID]] = None
    query: Optional[str] = None
    before: Optional[datetime] = None
    action: Literal["delete", "archive", "silence"] = "delete"


class ForgetResponse(BaseModel):
    """Response from forget operation."""

    success: bool
    affected_count: int
    message: str = ""


class MemoryStats(BaseModel):
    """Memory statistics for tenant. Holistic: tenant-only."""

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


# ---- Dashboard Schemas ----


class DashboardOverview(BaseModel):
    """Comprehensive dashboard overview stats."""

    total_memories: int = 0
    active_memories: int = 0
    silent_memories: int = 0
    compressed_memories: int = 0
    archived_memories: int = 0
    deleted_memories: int = 0
    labile_memories: int = 0
    by_type: Dict[str, int] = Field(default_factory=dict)
    by_status: Dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_importance: float = 0.0
    avg_access_count: float = 0.0
    avg_decay_rate: float = 0.0
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None
    estimated_size_mb: float = 0.0
    total_semantic_facts: int = 0
    current_semantic_facts: int = 0
    facts_by_category: Dict[str, int] = Field(default_factory=dict)
    avg_fact_confidence: float = 0.0
    avg_evidence_count: float = 0.0
    total_events: int = 0
    events_by_type: Dict[str, int] = Field(default_factory=dict)
    events_by_operation: Dict[str, int] = Field(default_factory=dict)


class DashboardMemoryListItem(BaseModel):
    """Memory item for dashboard list view."""

    id: UUID
    tenant_id: str
    agent_id: Optional[str] = None
    type: str
    status: str
    text: str
    key: Optional[str] = None
    namespace: Optional[str] = None
    context_tags: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    importance: float = 0.5
    access_count: int = 0
    decay_rate: float = 0.01
    labile: bool = False
    version: int = 1
    timestamp: Optional[datetime] = None
    written_at: Optional[datetime] = None


class DashboardMemoryListResponse(BaseModel):
    """Paginated memory list response."""

    items: List[DashboardMemoryListItem]
    total: int
    page: int
    per_page: int
    total_pages: int


class DashboardMemoryDetail(BaseModel):
    """Full memory detail for dashboard."""

    id: UUID
    tenant_id: str
    agent_id: Optional[str] = None
    type: str
    status: str
    text: str
    key: Optional[str] = None
    namespace: Optional[str] = None
    context_tags: List[str] = Field(default_factory=list)
    source_session_id: Optional[str] = None
    entities: Any = None
    relations: Any = None
    metadata: Any = None
    confidence: float = 0.5
    importance: float = 0.5
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    decay_rate: float = 0.01
    labile: bool = False
    provenance: Any = None
    version: int = 1
    supersedes_id: Optional[UUID] = None
    content_hash: Optional[str] = None
    timestamp: Optional[datetime] = None
    written_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    related_events: List[Dict[str, Any]] = Field(default_factory=list)


class DashboardEventItem(BaseModel):
    """Event log item for dashboard."""

    id: UUID
    tenant_id: str
    scope_id: str
    agent_id: Optional[str] = None
    event_type: str
    operation: Optional[str] = None
    payload: Any = None
    memory_ids: List[UUID] = Field(default_factory=list)
    parent_event_id: Optional[UUID] = None
    created_at: Optional[datetime] = None


class DashboardEventListResponse(BaseModel):
    """Paginated event list response."""

    items: List[DashboardEventItem]
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

    points: List[TimelinePoint]
    total: int


class ComponentStatus(BaseModel):
    """Health status for a single component."""

    name: str
    status: str  # "ok", "error", "degraded", "unknown"
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class DashboardComponentsResponse(BaseModel):
    """Component health report."""

    components: List[ComponentStatus]


class TenantInfo(BaseModel):
    """Tenant summary info."""

    tenant_id: str
    memory_count: int = 0
    active_memory_count: int = 0
    fact_count: int = 0
    event_count: int = 0
    last_memory_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None


class DashboardTenantsResponse(BaseModel):
    """List of tenants."""

    tenants: List[TenantInfo]


class DashboardConsolidateRequest(BaseModel):
    """Request to trigger consolidation."""

    tenant_id: str
    user_id: Optional[str] = None


class DashboardForgetRequest(BaseModel):
    """Request to trigger forgetting."""

    tenant_id: str
    user_id: Optional[str] = None
    dry_run: bool = True
    max_memories: int = 5000


# ---- Sessions Schemas ----


class SessionInfo(BaseModel):
    """Session info from Redis + DB."""

    session_id: str
    tenant_id: Optional[str] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    ttl_seconds: int = -1
    memory_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DashboardSessionsResponse(BaseModel):
    """Sessions list."""

    sessions: List[SessionInfo]
    total_active: int = 0
    total_memories_with_session: int = 0


# ---- Rate Limits Schemas ----


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

    entries: List[RateLimitEntry]
    configured_rpm: int = 60


class HourlyRequestCount(BaseModel):
    """Hourly request count."""

    hour: str  # ISO format
    count: int


class RequestStatsResponse(BaseModel):
    """Request stats over time."""

    points: List[HourlyRequestCount]
    total_last_24h: int = 0


# ---- Graph Schemas ----


class GraphNodeInfo(BaseModel):
    """Graph node for rendering."""

    id: str
    entity: str
    entity_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdgeInfo(BaseModel):
    """Graph edge for rendering."""

    source: str
    target: str
    predicate: str
    confidence: float = 0.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphStatsResponse(BaseModel):
    """Graph statistics."""

    total_nodes: int = 0
    total_edges: int = 0
    entity_types: Dict[str, int] = Field(default_factory=dict)
    tenants_with_graph: List[str] = Field(default_factory=list)


class GraphExploreResponse(BaseModel):
    """Graph exploration result."""

    nodes: List[GraphNodeInfo] = Field(default_factory=list)
    edges: List[GraphEdgeInfo] = Field(default_factory=list)
    center_entity: Optional[str] = None


class GraphSearchResult(BaseModel):
    """Entity search result."""

    entity: str
    entity_type: str
    tenant_id: str
    scope_id: str


class GraphSearchResponse(BaseModel):
    """Graph search results."""

    results: List[GraphSearchResult] = Field(default_factory=list)


# ---- Config Schemas ----


class ConfigItem(BaseModel):
    """A single config setting."""

    key: str
    value: Any = None
    default_value: Any = None
    is_secret: bool = False
    is_editable: bool = False
    source: str = "default"  # "default", "env", "override"
    description: str = ""


class ConfigSection(BaseModel):
    """A group of config settings."""

    name: str
    items: List[ConfigItem] = Field(default_factory=list)


class DashboardConfigResponse(BaseModel):
    """Full config snapshot."""

    sections: List[ConfigSection] = Field(default_factory=list)


class ConfigUpdateRequest(BaseModel):
    """Request to update config settings."""

    updates: Dict[str, Any] = Field(default_factory=dict)


# ---- Labile / Reconsolidation Schemas ----


class TenantLabileInfo(BaseModel):
    """Labile state info per tenant."""

    tenant_id: str
    db_labile_count: int = 0
    redis_scope_count: int = 0
    redis_session_count: int = 0
    redis_memory_count: int = 0


class DashboardLabileResponse(BaseModel):
    """Labile state overview."""

    tenants: List[TenantLabileInfo] = Field(default_factory=list)
    total_db_labile: int = 0
    total_redis_scopes: int = 0
    total_redis_sessions: int = 0
    total_redis_memories: int = 0


# ---- Retrieval / Read Test Schemas ----


class DashboardRetrievalRequest(BaseModel):
    """Request to test memory retrieval."""

    tenant_id: str
    query: str
    max_results: int = Field(default=10, le=50)
    context_filter: Optional[List[str]] = None
    memory_types: Optional[List[str]] = None
    format: Literal["packet", "list", "llm_context"] = "list"


class RetrievalResultItem(BaseModel):
    """A single retrieval result with score."""

    id: UUID
    text: str
    type: str
    confidence: float
    relevance_score: float
    retrieval_source: str = ""
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DashboardRetrievalResponse(BaseModel):
    """Retrieval test results."""

    query: str
    results: List[RetrievalResultItem] = Field(default_factory=list)
    llm_context: Optional[str] = None
    total_count: int = 0
    elapsed_ms: float = 0.0


# ---- Job History Schemas ----


class DashboardJobItem(BaseModel):
    """A single job history entry."""

    id: UUID
    job_type: str
    tenant_id: str
    user_id: Optional[str] = None
    dry_run: bool = False
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class DashboardJobsResponse(BaseModel):
    """Paginated job list."""

    items: List[DashboardJobItem]
    total: int


# ---- Bulk Actions Schemas ----


class BulkActionRequest(BaseModel):
    """Request for bulk memory actions."""

    memory_ids: List[UUID]
    action: Literal["archive", "silence", "delete"]
