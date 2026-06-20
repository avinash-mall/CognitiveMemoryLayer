"""API request/response schemas. Holistic: no scopes, tenant-only."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

# Shared API-contract models, relocated to the single-source-of-truth
# ``cml_contracts`` package so the server and SDK can never drift. Re-exported
# here (see ``__all__``) so existing ``from ..api.schemas import X`` paths still
# resolve.
from cml_contracts.models import (
    BulkActionRequest,
    ComponentStatus,
    ConfigItem,
    ConfigSection,
    ConfigUpdateRequest,
    DashboardComponentsResponse,
    DashboardConfigResponse,
    DashboardConsolidateRequest,
    DashboardEventItem,
    DashboardEventListResponse,
    DashboardForgetRequest,
    DashboardJobItem,
    DashboardJobsResponse,
    DashboardLabileResponse,
    DashboardMemoryDetail,
    DashboardMemoryListItem,
    DashboardMemoryListResponse,
    DashboardOverview,
    DashboardRateLimitsResponse,
    DashboardReconsolidateRequest,
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

from ..core.enums import MemoryType


class WriteMemoryRequest(BaseModel):
    """Request to store a memory. Holistic: tenant-only."""

    content: str = Field(..., min_length=1, max_length=100_000)
    context_tags: list[str] | None = None
    session_id: str | None = None
    memory_type: MemoryType | None = None
    namespace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    turn_id: str | None = None
    agent_id: str | None = None
    timestamp: datetime | None = None  # Optional event timestamp (defaults to now)


class CreateSessionRequest(BaseModel):
    """Request to create a new memory session."""

    name: str | None = None
    ttl_hours: int | None = 24
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    """Response from session creation."""

    session_id: str
    created_at: datetime
    expires_at: datetime | None = None


class WriteMemoryResponse(BaseModel):
    """Response from write operation."""

    success: bool
    memory_id: UUID | None = None
    chunks_created: int = 0
    message: str = ""
    # Eval mode (when X-Eval-Mode: true): outcome and reason from write gate
    eval_outcome: Literal["stored", "skipped"] | None = None
    eval_reason: str | None = None


class WriteBatchTurn(BaseModel):
    """A single turn within a batch write request."""

    content: str = Field(..., min_length=1, max_length=100_000)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    turn_id: str | None = None
    timestamp: datetime | None = None


class WriteBatchRequest(BaseModel):
    """Request to write multiple turns in a single call."""

    turns: list[WriteBatchTurn] = Field(..., min_length=1, max_length=500)


class WriteBatchResponse(BaseModel):
    """Response from batch write operation."""

    success: bool
    turns_processed: int = 0
    chunks_created: int = 0
    message: str = ""


class ReadMemoryRequest(BaseModel):
    """Request to retrieve memories. Holistic: tenant-only."""

    query: str
    max_results: int = Field(default=10, le=50)
    context_filter: list[str] | None = None
    memory_types: list[MemoryType] | None = None
    since: datetime | None = None
    until: datetime | None = None
    format: Literal["packet", "list", "llm_context"] = "packet"
    user_timezone: str | None = (
        None  # IANA timezone (e.g. "America/New_York") for "today"/"yesterday" filters
    )


class ReadMemoryResponse(BaseModel):
    """Response from read operation."""

    query: str
    memories: list[MemoryItem]
    facts: list[MemoryItem] = Field(default_factory=list)
    preferences: list[MemoryItem] = Field(default_factory=list)
    episodes: list[MemoryItem] = Field(default_factory=list)
    constraints: list[MemoryItem] = Field(default_factory=list)
    llm_context: str | None = None
    retrieval_meta: dict | None = None
    total_count: int
    elapsed_ms: float


class ProcessTurnRequest(BaseModel):
    """Request to process a conversation turn with seamless memory (auto-retrieve + optional auto-store)."""

    user_message: str
    assistant_response: str | None = None
    session_id: str | None = None
    max_context_tokens: int = 1500
    timestamp: datetime | None = None  # Optional event timestamp for the turn (defaults to now)
    user_timezone: str | None = (
        None  # IANA timezone for retrieval "today"/"yesterday" (e.g. "America/New_York")
    )


class ProcessTurnResponse(BaseModel):
    """Response from process_turn: memory context ready to inject into prompt."""

    memory_context: str
    memories_retrieved: int
    memories_stored: int
    reconsolidation_applied: bool = False


class UpdateMemoryRequest(BaseModel):
    """Request to update a memory. Holistic: tenant-only."""

    memory_id: UUID
    text: str | None = None
    confidence: float | None = None
    importance: float | None = None
    metadata: dict[str, Any] | None = None
    feedback: str | None = None  # "correct", "incorrect", "outdated"


class UpdateMemoryResponse(BaseModel):
    """Response from update operation."""

    success: bool
    memory_id: UUID
    version: int
    message: str = ""


class ForgetRequest(BaseModel):
    """Request to forget memories. Holistic: tenant-only."""

    memory_ids: list[UUID] | None = None
    query: str | None = None
    before: datetime | None = None
    action: Literal["delete", "archive", "silence"] = "delete"


class MemoryStats(BaseModel):
    """Memory statistics for tenant. Holistic: tenant-only."""

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


# ---- Retrieval / Read Test Schemas ----


class DashboardRetrievalRequest(BaseModel):
    """Request to test memory retrieval."""

    tenant_id: str
    query: str
    max_results: int = Field(default=10, le=50)
    context_filter: list[str] | None = None
    memory_types: list[str] | None = None
    format: Literal["packet", "list", "llm_context"] = "list"


# ---- Dashboard Explainability / Workbench Schemas ----


class RetrievalExplainAnalysis(BaseModel):
    """Query analysis used by the retrieval explain endpoint."""

    intent: str
    confidence: float
    entities: list[str] = Field(default_factory=list)
    key_phrases: list[str] = Field(default_factory=list)
    time_reference: str | None = None
    time_start: datetime | None = None
    time_end: datetime | None = None
    suggested_sources: list[str] = Field(default_factory=list)
    suggested_top_k: int = 10
    query_domain: str | None = None
    constraint_dimensions: list[str] = Field(default_factory=list)
    is_decision_query: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalExplainPlanStep(BaseModel):
    """Single retrieval plan step for explain mode."""

    source: str
    priority: int = 0
    key: str | None = None
    query: str | None = None
    seeds: list[str] = Field(default_factory=list)
    memory_types: list[str] = Field(default_factory=list)
    time_filter: dict[str, Any] | None = None
    min_confidence: float = 0.0
    top_k: int = 10
    timeout_ms: int = 0
    skip_if_found: bool = False
    associative_expansion: bool = False
    query_domain: str | None = None
    constraint_categories: list[str] = Field(default_factory=list)


class RetrievalExplainExecutionStep(BaseModel):
    """Observed execution details for a retrieval step."""

    source: str
    success: bool
    elapsed_ms: float = 0.0
    result_count: int = 0
    timed_out: bool = False
    error: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    query_preview: str | None = None
    result_preview: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalExplainRerankItem(BaseModel):
    """Reranker score breakdown for a retrieved memory."""

    id: UUID
    text: str
    source_type: str
    retrieval_source: str = ""
    rank: int
    selected: bool = True
    final_score: float
    breakdown: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class DashboardRetrievalExplainResponse(BaseModel):
    """Detailed retrieval explain response."""

    query: str
    analysis: RetrievalExplainAnalysis
    plan_steps: list[RetrievalExplainPlanStep] = Field(default_factory=list)
    parallel_groups: list[list[int]] = Field(default_factory=list)
    execution_steps: list[RetrievalExplainExecutionStep] = Field(default_factory=list)
    retrieval_meta: dict[str, Any] = Field(default_factory=dict)
    packet_warnings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    llm_context: str | None = None
    results: list[RetrievalResultItem] = Field(default_factory=list)
    rerank: list[RetrievalExplainRerankItem] = Field(default_factory=list)
    total_count: int = 0
    elapsed_ms: float = 0.0


class DashboardWriteSimulationRequest(BaseModel):
    """Dry-run write simulation request."""

    tenant_id: str
    content: str = Field(..., min_length=1, max_length=100_000)
    context_tags: list[str] | None = None
    session_id: str | None = None
    memory_type: str | None = None
    namespace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime | None = None
    compare_extractors: bool = False


class DashboardWriteSimulationChunk(BaseModel):
    """Dry-run write simulation result for a single chunk."""

    chunk_id: str
    text: str
    chunk_type: str
    salience: float = 0.0
    novelty: float = 0.0
    confidence: float = 0.0
    timestamp: datetime | None = None
    write_decision: str
    would_store: bool
    reason: str = ""
    chosen_memory_type: str | None = None
    key: str | None = None
    context_tags: list[str] = Field(default_factory=list)
    importance: float = 0.0
    decay_rate: float | None = None
    risk_flags: list[str] = Field(default_factory=list)
    redaction_required: bool = False
    redacted_text: str = ""
    redactions: list[dict[str, Any]] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    extracted_constraints: list[dict[str, Any]] = Field(default_factory=list)
    extracted_facts: list[dict[str, Any]] = Field(default_factory=list)
    extractor_outputs: dict[str, Any] = Field(default_factory=dict)


class DashboardWriteSimulationResponse(BaseModel):
    """Dry-run write simulation response."""

    tenant_id: str
    chunks: list[DashboardWriteSimulationChunk] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class FactItem(BaseModel):
    """Serialised semantic fact for dashboard display."""

    id: str
    tenant_id: str
    category: str
    key: str
    value: str
    confidence: float
    evidence_count: int
    is_current: bool
    version: int
    created_at: str | None = None
    updated_at: str | None = None


class FactListResponse(BaseModel):
    """List response for semantic facts."""

    items: list[FactItem] = Field(default_factory=list)
    total: int = 0


class DashboardFactDetail(BaseModel):
    """Full semantic fact detail."""

    id: str
    tenant_id: str
    category: str
    key: str
    subject: str
    predicate: str
    value: Any = None
    value_type: str
    context_tags: list[str] = Field(default_factory=list)
    confidence: float
    evidence_count: int
    evidence_ids: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    is_current: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None
    version: int
    supersedes_id: str | None = None
    lineage: list[dict[str, Any]] = Field(default_factory=list)
    superseded_by: list[dict[str, Any]] = Field(default_factory=list)


class DashboardFactEvidenceItem(BaseModel):
    """Evidence memory attached to a semantic fact."""

    id: UUID
    text: str
    type: str
    status: str
    confidence: float
    importance: float = 0.0
    source_session_id: str | None = None
    timestamp: datetime | None = None
    written_at: datetime | None = None
    supersedes_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DashboardFactEvidenceResponse(BaseModel):
    """Evidence breakdown for a fact."""

    fact_id: str
    evidence: list[DashboardFactEvidenceItem] = Field(default_factory=list)
    missing_evidence_ids: list[str] = Field(default_factory=list)


class DashboardLineageMemory(BaseModel):
    """Compact memory summary used in lineage views."""

    id: UUID
    text: str
    type: str
    status: str
    version: int = 1
    key: str | None = None
    confidence: float = 0.0
    importance: float = 0.0
    timestamp: datetime | None = None
    written_at: datetime | None = None
    supersedes_id: UUID | None = None


class DashboardRelatedJob(BaseModel):
    """Job reference linked from lineage views."""

    id: UUID
    job_type: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None


class DashboardMemoryLineageResponse(BaseModel):
    """Lifecycle and lineage view for a memory."""

    memory: DashboardMemoryDetail
    ancestors: list[DashboardLineageMemory] = Field(default_factory=list)
    descendants: list[DashboardLineageMemory] = Field(default_factory=list)
    same_key_versions: list[DashboardLineageMemory] = Field(default_factory=list)
    evidence_facts: list[FactItem] = Field(default_factory=list)
    related_entities: list[GraphSearchResult] = Field(default_factory=list)
    related_jobs: list[DashboardRelatedJob] = Field(default_factory=list)
    lifecycle_flags: list[str] = Field(default_factory=list)


class DashboardReconsolidationSessionItem(BaseModel):
    """Active labile session for reconsolidation workbench."""

    session_id: str
    tenant_id: str
    scope_id: str
    turn_id: str
    created_at: datetime | None = None
    expires_at: datetime | None = None
    query: str = ""
    retrieved_texts: list[str] = Field(default_factory=list)
    memories: list[dict[str, Any]] = Field(default_factory=list)


class DashboardReconsolidationSessionsResponse(BaseModel):
    """Reconsolidation workbench response."""

    items: list[DashboardReconsolidationSessionItem] = Field(default_factory=list)
    total: int = 0


class DashboardForgettingPreviewRequest(BaseModel):
    """Request to preview forgetting without mutating storage."""

    tenant_id: str
    user_id: str | None = None
    max_memories: int = Field(default=200, ge=1, le=5000)


class DashboardForgettingPreviewItem(BaseModel):
    """Preview item for forgetting decisions."""

    memory_id: UUID
    text: str
    type: str
    status: str
    confidence: float
    importance: float
    access_count: int
    dependency_count: int = 0
    total_score: float
    importance_score: float
    recency_score: float
    frequency_score: float
    confidence_score: float
    type_bonus_score: float
    dependency_score: float
    suggested_action: str
    protected: bool = False
    keep_reason: str | None = None
    duplicate_matches: list[dict[str, Any]] = Field(default_factory=list)


class DashboardForgettingPreviewResponse(BaseModel):
    """Dry-run forgetting preview."""

    tenant_id: str
    user_id: str
    scanned_count: int = 0
    duplicates_found: int = 0
    operations_planned: int = 0
    items: list[DashboardForgettingPreviewItem] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class DashboardQualityIssue(BaseModel):
    """Single data-quality or graph-hygiene issue."""

    label: str
    count: int
    description: str = ""
    sample_ids: list[str] = Field(default_factory=list)


class DashboardQualityResponse(BaseModel):
    """Aggregated data quality and graph hygiene summary."""

    tenant_id: str | None = None
    generated_at: datetime
    issues: list[DashboardQualityIssue] = Field(default_factory=list)
    invalidation_hotspots: list[dict[str, Any]] = Field(default_factory=list)
    graph_hygiene: dict[str, Any] = Field(default_factory=dict)
    labile_sessions: int = 0


class DashboardOpsMetric(BaseModel):
    """Single operational metric sample."""

    name: str
    labels: dict[str, str] = Field(default_factory=dict)
    value: float | int | str | None = None


class DashboardOpsResponse(BaseModel):
    """Operational metrics view for the dashboard."""

    generated_at: datetime
    highlights: dict[str, Any] = Field(default_factory=dict)
    metrics: list[DashboardOpsMetric] = Field(default_factory=list)


class DashboardEvaluationArtifact(BaseModel):
    """Evaluation file discovered on disk."""

    name: str
    path: str
    kind: str
    size_bytes: int = 0
    updated_at: datetime | None = None


class DashboardEvaluationSummary(BaseModel):
    """Evaluation readiness and discovered results."""

    generated_at: datetime
    readiness: dict[str, Any] = Field(default_factory=dict)
    latest_summary: dict[str, Any] | None = None
    latest_report: str | None = None
    artifacts: list[DashboardEvaluationArtifact] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "BulkActionRequest",
    "ComponentStatus",
    "ConfigItem",
    "ConfigSection",
    "ConfigUpdateRequest",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "DashboardComponentsResponse",
    "DashboardConfigResponse",
    "DashboardConsolidateRequest",
    "DashboardEvaluationArtifact",
    "DashboardEvaluationSummary",
    "DashboardEventItem",
    "DashboardEventListResponse",
    "DashboardFactDetail",
    "DashboardFactEvidenceItem",
    "DashboardFactEvidenceResponse",
    "DashboardForgetRequest",
    "DashboardForgettingPreviewItem",
    "DashboardForgettingPreviewRequest",
    "DashboardForgettingPreviewResponse",
    "DashboardJobItem",
    "DashboardJobsResponse",
    "DashboardLabileResponse",
    "DashboardLineageMemory",
    "DashboardMemoryDetail",
    "DashboardMemoryLineageResponse",
    "DashboardMemoryListItem",
    "DashboardMemoryListResponse",
    "DashboardOpsMetric",
    "DashboardOpsResponse",
    "DashboardOverview",
    "DashboardQualityIssue",
    "DashboardQualityResponse",
    "DashboardRateLimitsResponse",
    "DashboardReconsolidateRequest",
    "DashboardReconsolidationSessionItem",
    "DashboardReconsolidationSessionsResponse",
    "DashboardRelatedJob",
    "DashboardRetrievalExplainResponse",
    "DashboardRetrievalRequest",
    "DashboardRetrievalResponse",
    "DashboardSessionsResponse",
    "DashboardTenantsResponse",
    "DashboardTimelineResponse",
    "DashboardWriteSimulationChunk",
    "DashboardWriteSimulationRequest",
    "DashboardWriteSimulationResponse",
    "FactItem",
    "FactListResponse",
    "ForgetRequest",
    "ForgetResponse",
    "GraphEdgeInfo",
    "GraphExploreResponse",
    "GraphNeo4jConfigResponse",
    "GraphNodeInfo",
    "GraphSearchResponse",
    "GraphSearchResult",
    "GraphStatsResponse",
    "HourlyRequestCount",
    "MemoryItem",
    "MemoryStats",
    "ProcessTurnRequest",
    "ProcessTurnResponse",
    "RateLimitEntry",
    "ReadMemoryRequest",
    "ReadMemoryResponse",
    "RequestStatsResponse",
    "RetrievalExplainAnalysis",
    "RetrievalExplainExecutionStep",
    "RetrievalExplainPlanStep",
    "RetrievalExplainRerankItem",
    "RetrievalResultItem",
    "SessionContextResponse",
    "SessionInfo",
    "TenantInfo",
    "TenantLabileInfo",
    "TimelinePoint",
    "UpdateMemoryRequest",
    "UpdateMemoryResponse",
    "WriteBatchRequest",
    "WriteBatchResponse",
    "WriteBatchTurn",
    "WriteMemoryRequest",
    "WriteMemoryResponse",
]
