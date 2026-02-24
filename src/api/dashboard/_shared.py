"""Shared constants, helpers, and imports for dashboard routes."""

import os
from typing import Any

import structlog
from fastapi import Request

from ...storage.connection import DatabaseManager
from ..schemas import (
    BulkActionRequest,
    ComponentStatus,
    DashboardComponentsResponse,
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
    DashboardRetrievalRequest,
    DashboardRetrievalResponse,
    DashboardSessionsResponse,
    DashboardTenantsResponse,
    DashboardTimelineResponse,
    GraphEdgeInfo,
    GraphExploreResponse,
    GraphNeo4jConfigResponse,
    GraphNodeInfo,
    GraphSearchResponse,
    GraphSearchResult,
    GraphStatsResponse,
    HourlyRequestCount,
    RateLimitEntry,
    RequestStatsResponse,
    RetrievalResultItem,
    SessionInfo,
    TenantInfo,
    TenantLabileInfo,
    TimelinePoint,
)

__all__ = [
    "_CONFIG_KEY_TO_ENV",
    "_EDITABLE_SETTINGS",
    "_REQUEST_COUNT_PREFIX",
    "BulkActionRequest",
    "ComponentStatus",
    "DashboardComponentsResponse",
    "DashboardConsolidateRequest",
    "DashboardEventItem",
    "DashboardEventListResponse",
    "DashboardForgetRequest",
    "DashboardJobItem",
    "DashboardJobsResponse",
    "DashboardLabileResponse",
    "DashboardMemoryDetail",
    "DashboardMemoryListItem",
    "DashboardMemoryListResponse",
    "DashboardOverview",
    "DashboardRateLimitsResponse",
    "DashboardReconsolidateRequest",
    "DashboardRetrievalRequest",
    "DashboardRetrievalResponse",
    "DashboardSessionsResponse",
    "DashboardTenantsResponse",
    "DashboardTimelineResponse",
    "GraphEdgeInfo",
    "GraphExploreResponse",
    "GraphNeo4jConfigResponse",
    "GraphNodeInfo",
    "GraphSearchResponse",
    "GraphSearchResult",
    "GraphStatsResponse",
    "HourlyRequestCount",
    "RateLimitEntry",
    "RequestStatsResponse",
    "RetrievalResultItem",
    "SessionInfo",
    "TenantInfo",
    "TenantLabileInfo",
    "TimelinePoint",
    "_config_source",
    "_get_db",
    "logger",
]

logger = structlog.get_logger()

_REQUEST_COUNT_PREFIX = "dashboard:reqcount:"

# Settings that admins may edit (non-secret). Persisted to .env.
_EDITABLE_SETTINGS = frozenset(
    {
        "app_name",
        "debug",
        "cors_origins",
        "embedding_internal.provider",
        "embedding_internal.model",
        "embedding_internal.dimensions",
        "embedding_internal.local_model",
        "embedding_internal.base_url",
        "llm_internal.provider",
        "llm_internal.model",
        "llm_internal.base_url",
        "llm_eval.provider",
        "llm_eval.model",
        "llm_eval.base_url",
        "auth.default_tenant_id",
        "auth.rate_limit_requests_per_minute",
        "chunker.tokenizer",
        "chunker.chunk_size",
        "chunker.overlap_percent",
        "retrieval.episode_relevance_threshold",
        "retrieval.max_episodes_when_constraints",
        "retrieval.max_episodes_default",
        "retrieval.max_constraint_tokens",
        "retrieval.default_step_timeout_ms",
        "retrieval.total_timeout_ms",
        "retrieval.graph_timeout_ms",
        "retrieval.fact_timeout_ms",
        "retrieval.hnsw_ef_search",
        "retrieval.reranker.recency_weight",
        "retrieval.reranker.relevance_weight",
        "retrieval.reranker.confidence_weight",
        "retrieval.reranker.active_constraint_bonus",
        "features.stable_keys_enabled",
        "features.write_time_facts_enabled",
        "features.batch_embeddings_enabled",
        "features.store_async",
        "features.cached_embeddings_enabled",
        "features.retrieval_timeouts_enabled",
        "features.skip_if_found_cross_group",
        "features.db_dependency_counts",
        "features.bounded_state_enabled",
        "features.hnsw_ef_search_tuning",
        "features.constraint_extraction_enabled",
        "features.use_llm_constraint_extractor",
        "features.use_llm_write_time_facts",
        "features.use_llm_query_classifier_only",
        "features.use_llm_salience_refinement",
        "features.use_llm_pii_redaction",
        "features.use_llm_write_gate_importance",
        "features.use_llm_memory_type",
        "features.use_llm_confidence",
        "features.use_llm_context_tags",
        "features.use_llm_decay_rate",
        "features.use_llm_conflict_detection_only",
        "features.use_llm_constraint_reranker",
    }
)

# Config key -> env var for .env persistence
_CONFIG_KEY_TO_ENV: dict[str, str] = {
    "app_name": "APP_NAME",
    "debug": "DEBUG",
    "cors_origins": "CORS_ORIGINS",
    "database.postgres_url": "DATABASE__POSTGRES_URL",
    "database.neo4j_url": "DATABASE__NEO4J_URL",
    "database.neo4j_browser_url": "DATABASE__NEO4J_BROWSER_URL",
    "database.neo4j_user": "DATABASE__NEO4J_USER",
    "database.neo4j_password": "DATABASE__NEO4J_PASSWORD",
    "database.redis_url": "DATABASE__REDIS_URL",
    "embedding_internal.provider": "EMBEDDING_INTERNAL__PROVIDER",
    "embedding_internal.model": "EMBEDDING_INTERNAL__MODEL",
    "embedding_internal.dimensions": "EMBEDDING_INTERNAL__DIMENSIONS",
    "embedding_internal.local_model": "EMBEDDING_INTERNAL__LOCAL_MODEL",
    "embedding_internal.api_key": "EMBEDDING_INTERNAL__API_KEY",
    "embedding_internal.base_url": "EMBEDDING_INTERNAL__BASE_URL",
    "llm_internal.provider": "LLM_INTERNAL__PROVIDER",
    "llm_internal.model": "LLM_INTERNAL__MODEL",
    "llm_internal.base_url": "LLM_INTERNAL__BASE_URL",
    "llm_internal.api_key": "LLM_INTERNAL__API_KEY",
    "llm_eval.provider": "LLM_EVAL__PROVIDER",
    "llm_eval.model": "LLM_EVAL__MODEL",
    "llm_eval.base_url": "LLM_EVAL__BASE_URL",
    "llm_eval.api_key": "LLM_EVAL__API_KEY",
    "auth.api_key": "AUTH__API_KEY",
    "auth.admin_api_key": "AUTH__ADMIN_API_KEY",
    "auth.default_tenant_id": "AUTH__DEFAULT_TENANT_ID",
    "auth.rate_limit_requests_per_minute": "AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE",
    "chunker.tokenizer": "CHUNKER__TOKENIZER",
    "chunker.chunk_size": "CHUNKER__CHUNK_SIZE",
    "chunker.overlap_percent": "CHUNKER__OVERLAP_PERCENT",
    "retrieval.episode_relevance_threshold": "RETRIEVAL__EPISODE_RELEVANCE_THRESHOLD",
    "retrieval.max_episodes_when_constraints": "RETRIEVAL__MAX_EPISODES_WHEN_CONSTRAINTS",
    "retrieval.max_episodes_default": "RETRIEVAL__MAX_EPISODES_DEFAULT",
    "retrieval.max_constraint_tokens": "RETRIEVAL__MAX_CONSTRAINT_TOKENS",
    "retrieval.default_step_timeout_ms": "RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS",
    "retrieval.total_timeout_ms": "RETRIEVAL__TOTAL_TIMEOUT_MS",
    "retrieval.graph_timeout_ms": "RETRIEVAL__GRAPH_TIMEOUT_MS",
    "retrieval.fact_timeout_ms": "RETRIEVAL__FACT_TIMEOUT_MS",
    "retrieval.hnsw_ef_search": "RETRIEVAL__HNSW_EF_SEARCH",
    "retrieval.reranker.recency_weight": "RETRIEVAL__RERANKER__RECENCY_WEIGHT",
    "retrieval.reranker.relevance_weight": "RETRIEVAL__RERANKER__RELEVANCE_WEIGHT",
    "retrieval.reranker.confidence_weight": "RETRIEVAL__RERANKER__CONFIDENCE_WEIGHT",
    "retrieval.reranker.active_constraint_bonus": "RETRIEVAL__RERANKER__ACTIVE_CONSTRAINT_BONUS",
}
for _fk in (
    "stable_keys_enabled",
    "write_time_facts_enabled",
    "batch_embeddings_enabled",
    "store_async",
    "cached_embeddings_enabled",
    "retrieval_timeouts_enabled",
    "skip_if_found_cross_group",
    "db_dependency_counts",
    "bounded_state_enabled",
    "hnsw_ef_search_tuning",
    "constraint_extraction_enabled",
    "use_llm_constraint_extractor",
    "use_llm_write_time_facts",
    "use_llm_query_classifier_only",
    "use_llm_salience_refinement",
    "use_llm_pii_redaction",
    "use_llm_write_gate_importance",
    "use_llm_memory_type",
    "use_llm_confidence",
    "use_llm_context_tags",
    "use_llm_decay_rate",
    "use_llm_conflict_detection_only",
    "use_llm_constraint_reranker",
):
    _CONFIG_KEY_TO_ENV[f"features.{_fk}"] = f"FEATURES__{_fk.upper()}"

# Fields whose values must be masked in the config output.
_SECRET_FIELD_TOKENS = {"key", "password", "secret", "token"}


def _get_db(request: Request) -> DatabaseManager:
    """Get database manager from app state."""
    return request.app.state.db


def _config_source(env_var: str) -> str:
    """Return 'env' if env var is set, else 'default'."""
    return "env" if os.environ.get(env_var) is not None else "default"


def _mask_value(value: Any) -> str:
    return "****" if value else ""


def _is_secret_key(key: str) -> bool:
    lower = key.lower()
    return any(tok in lower for tok in _SECRET_FIELD_TOKENS)
