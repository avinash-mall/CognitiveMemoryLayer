"""Dashboard configuration routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...core.config import EmbeddingInternalSettings, get_settings
from ...core.env_file import get_env_path, update_env
from ...storage.connection import DatabaseManager
from ..auth import AuthContext, require_admin_permission
from ..schemas import (
    ConfigItem,
    ConfigSection,
    ConfigUpdateRequest,
    DashboardConfigResponse,
)
from ._shared import (
    _CONFIG_KEY_TO_ENV,
    _EDITABLE_SETTINGS,
    _config_source,
    _get_db,
    logger,
)

router = APIRouter()

_WRITE_PROTECTED_FIELDS = frozenset(
    {
        "database.postgres_url",
        "database.neo4j_url",
        "database.neo4j_user",
        "database.neo4j_password",
        "database.neo4j_browser_url",
        "database.redis_url",
        "embedding_internal.api_key",
        "llm_internal.api_key",
        "llm_eval.api_key",
        "auth.api_key",
        "auth.admin_api_key",
    }
)


def _validate_config_updates(updates: dict[str, Any]) -> list[str]:
    """Validate config updates. Returns list of error messages."""
    errors: list[str] = []
    for key, val in updates.items():
        if key == "embedding_internal.dimensions":
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append("embedding_internal.dimensions must be a positive integer")
        elif key == "chunker.chunk_size":
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append("chunker.chunk_size must be a positive integer")
        elif key == "chunker.overlap_percent":
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append("chunker.overlap_percent must be between 0 and 1")
        elif key == "auth.rate_limit_requests_per_minute":
            if not isinstance(val, (int, float)) or val < 0:
                errors.append("auth.rate_limit_requests_per_minute must be non-negative")
        elif key.startswith("retrieval.") and "timeout" in key:
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append(f"{key} must be a positive number")
        elif key.startswith("retrieval.reranker."):
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"{key} must be between 0 and 1")
    return errors


@router.get("/config", response_model=DashboardConfigResponse)
async def dashboard_config(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Configuration snapshot. Secrets are masked. Changes persist to .env; restart required for most settings."""
    settings = get_settings()
    sections: list[ConfigSection] = []

    # Application
    app_items = [
        ConfigItem(
            key="app_name",
            value=settings.app_name,
            default_value="CognitiveMemoryLayer",
            is_editable=True,
            source=_config_source("APP_NAME"),
            description="Application name. Used in responses and UI.",
            requires_restart=True,
            is_required=False,
            env_var="APP_NAME",
        ),
        ConfigItem(
            key="debug",
            value=settings.debug,
            default_value=False,
            is_editable=True,
            source=_config_source("DEBUG"),
            description="Enable debug mode. Exposes verbose 500 error details. Use false in production.",
            requires_restart=True,
            is_required=False,
            env_var="DEBUG",
            options=["true", "false"],
        ),
        ConfigItem(
            key="cors_origins",
            value=settings.cors_origins,
            default_value=None,
            is_editable=True,
            source=_config_source("CORS_ORIGINS"),
            description="Comma-separated CORS allowed origins. Unset uses default list; DEBUG=true allows '*'.",
            requires_restart=True,
            is_required=False,
            env_var="CORS_ORIGINS",
        ),
    ]
    sections.append(ConfigSection(name="Application", items=app_items))

    # Database
    db_s = settings.database
    db_items = [
        ConfigItem(
            key="database.postgres_url",
            value=db_s.postgres_url,
            default_value="postgresql+asyncpg://memory:memory@localhost/memory",
            is_editable=False,
            source=_config_source("DATABASE__POSTGRES_URL"),
            description="PostgreSQL connection URL. Required. Format: postgresql+asyncpg://user:pass@host:port/db",
            requires_restart=True,
            is_required=True,
            env_var="DATABASE__POSTGRES_URL",
        ),
        ConfigItem(
            key="database.neo4j_url",
            value=db_s.neo4j_url,
            default_value="bolt://localhost:7687",
            is_editable=False,
            source=_config_source("DATABASE__NEO4J_URL"),
            description="Neo4j Bolt URL for knowledge graph. Optional.",
            requires_restart=True,
            is_required=False,
            env_var="DATABASE__NEO4J_URL",
        ),
        ConfigItem(
            key="database.neo4j_user",
            value=db_s.neo4j_user,
            default_value="neo4j",
            is_editable=False,
            source=_config_source("DATABASE__NEO4J_USER"),
            description="Neo4j username. Optional.",
            requires_restart=True,
            is_required=False,
            env_var="DATABASE__NEO4J_USER",
        ),
        ConfigItem(
            key="database.neo4j_password",
            value="****" if db_s.neo4j_password else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("DATABASE__NEO4J_PASSWORD"),
            description="Neo4j password. Set when Neo4j is secured.",
            requires_restart=True,
            is_required=False,
            env_var="DATABASE__NEO4J_PASSWORD",
        ),
        ConfigItem(
            key="database.neo4j_browser_url",
            value=db_s.neo4j_browser_url or "",
            default_value="",
            is_editable=False,
            source=_config_source("DATABASE__NEO4J_BROWSER_URL"),
            description="Neo4j Bolt URL for dashboard graph (neovis.js). When API in Docker, set to bolt://localhost:7687.",
            requires_restart=True,
            is_required=False,
            env_var="DATABASE__NEO4J_BROWSER_URL",
        ),
        ConfigItem(
            key="database.redis_url",
            value=db_s.redis_url,
            default_value="redis://localhost:6379",
            is_editable=False,
            source=_config_source("DATABASE__REDIS_URL"),
            description="Redis URL for cache, Celery broker, rate limits.",
            requires_restart=True,
            is_required=False,
            env_var="DATABASE__REDIS_URL",
        ),
    ]
    sections.append(ConfigSection(name="Database", items=db_items))

    # Embedding (internal memory tasks)
    emb = getattr(settings, "embedding_internal", None) or EmbeddingInternalSettings()
    emb_items = [
        ConfigItem(
            key="embedding_internal.provider",
            value=emb.provider or "(default)",
            default_value=None,
            is_editable=True,
            source=_config_source("EMBEDDING_INTERNAL__PROVIDER"),
            description="Provider: openai | local | openai_compatible | ollama. Unset uses nomic-embed-text-v2-moe.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING_INTERNAL__PROVIDER",
            options=["openai", "local", "openai_compatible", "ollama"],
        ),
        ConfigItem(
            key="embedding_internal.model",
            value=emb.model or "(default)",
            default_value=None,
            is_editable=True,
            source=_config_source("EMBEDDING_INTERNAL__MODEL"),
            description="Model name. Must match provider. Changing impacts all new embeddings.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING_INTERNAL__MODEL",
        ),
        ConfigItem(
            key="embedding_internal.dimensions",
            value=emb.dimensions or 768,
            default_value=768,
            is_editable=True,
            source=_config_source("EMBEDDING_INTERNAL__DIMENSIONS"),
            description="Vector dimension. Must match model and DB schema. Default 768 for nomic.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING_INTERNAL__DIMENSIONS",
        ),
        ConfigItem(
            key="embedding_internal.local_model",
            value=emb.local_model or "(default)",
            default_value=None,
            is_editable=True,
            source=_config_source("EMBEDDING_INTERNAL__LOCAL_MODEL"),
            description="Model for provider=local. Default: nomic-embed-text-v2-moe.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING_INTERNAL__LOCAL_MODEL",
        ),
        ConfigItem(
            key="embedding_internal.api_key",
            value="****" if emb.api_key else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("EMBEDDING_INTERNAL__API_KEY"),
            description="API key for embedding provider. Fallback: OPENAI_API_KEY.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING_INTERNAL__API_KEY",
        ),
        ConfigItem(
            key="embedding_internal.base_url",
            value=emb.base_url or "",
            default_value="",
            is_editable=True,
            source=_config_source("EMBEDDING_INTERNAL__BASE_URL"),
            description="OpenAI-compatible endpoint URL. For ollama/openai_compatible.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING_INTERNAL__BASE_URL",
        ),
    ]
    sections.append(ConfigSection(name="Embedding (Internal)", items=emb_items))

    # LLM Internal (extraction, consolidation, retrieval)
    llmi = settings.llm_internal
    llmi_items = [
        ConfigItem(
            key="llm_internal.provider",
            value=llmi.provider,
            default_value="openai",
            is_editable=True,
            source=_config_source("LLM_INTERNAL__PROVIDER"),
            description="Provider: openai | ollama | anthropic | gemini | vllm | sglang | openai_compatible.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_INTERNAL__PROVIDER",
            options=[
                "openai",
                "openai_compatible",
                "ollama",
                "gemini",
                "claude",
                "anthropic",
                "vllm",
                "sglang",
            ],
        ),
        ConfigItem(
            key="llm_internal.model",
            value=llmi.model,
            default_value="gpt-4o-mini",
            is_editable=True,
            source=_config_source("LLM_INTERNAL__MODEL"),
            description="Model for internal tasks (extraction, consolidation, classifier).",
            requires_restart=True,
            is_required=False,
            env_var="LLM_INTERNAL__MODEL",
        ),
        ConfigItem(
            key="llm_internal.base_url",
            value=llmi.base_url or "",
            default_value="",
            is_editable=True,
            source=_config_source("LLM_INTERNAL__BASE_URL"),
            description="Base URL for internal LLM. Required for ollama/openai_compatible.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_INTERNAL__BASE_URL",
        ),
        ConfigItem(
            key="llm_internal.api_key",
            value="****" if llmi.api_key else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("LLM_INTERNAL__API_KEY"),
            description="API key for internal LLM. Fallback: OPENAI_API_KEY.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_INTERNAL__API_KEY",
        ),
    ]
    sections.append(ConfigSection(name="LLM Internal", items=llmi_items))

    # LLM Eval (QA, judge; fallback to LLM Internal when unset)
    llme = settings.llm_eval
    llme_items = [
        ConfigItem(
            key="llm_eval.provider",
            value=llme.provider or "(uses llm_internal)",
            default_value=None,
            is_editable=True,
            source=_config_source("LLM_EVAL__PROVIDER"),
            description="Provider for evaluation (QA, judge). Unset uses LLM_INTERNAL__*.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_EVAL__PROVIDER",
        ),
        ConfigItem(
            key="llm_eval.model",
            value=llme.model or "(uses llm_internal)",
            default_value=None,
            is_editable=True,
            source=_config_source("LLM_EVAL__MODEL"),
            description="Model for evaluation. Unset uses LLM_INTERNAL__MODEL.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_EVAL__MODEL",
        ),
        ConfigItem(
            key="llm_eval.base_url",
            value=llme.base_url or "",
            default_value="",
            is_editable=True,
            source=_config_source("LLM_EVAL__BASE_URL"),
            description="Base URL for eval LLM. Unset uses LLM_INTERNAL__BASE_URL.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_EVAL__BASE_URL",
        ),
    ]
    sections.append(ConfigSection(name="LLM Eval", items=llme_items))

    # Auth
    auth_s = settings.auth
    auth_items = [
        ConfigItem(
            key="auth.api_key",
            value="****" if auth_s.api_key else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("AUTH__API_KEY"),
            description="API key for memory operations (X-API-Key). Required.",
            requires_restart=True,
            is_required=True,
            env_var="AUTH__API_KEY",
        ),
        ConfigItem(
            key="auth.admin_api_key",
            value="****" if auth_s.admin_api_key else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("AUTH__ADMIN_API_KEY"),
            description="Admin API key for dashboard, consolidate, run_forgetting.",
            requires_restart=True,
            is_required=False,
            env_var="AUTH__ADMIN_API_KEY",
        ),
        ConfigItem(
            key="auth.default_tenant_id",
            value=auth_s.default_tenant_id,
            default_value="default",
            is_editable=True,
            source=_config_source("AUTH__DEFAULT_TENANT_ID"),
            description="Default tenant when X-Tenant-ID not sent.",
            requires_restart=True,
            is_required=False,
            env_var="AUTH__DEFAULT_TENANT_ID",
        ),
        ConfigItem(
            key="auth.rate_limit_requests_per_minute",
            value=auth_s.rate_limit_requests_per_minute,
            default_value=60,
            is_editable=True,
            source=_config_source("AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE"),
            description="Per-tenant rate limit. 0=disabled. Increase for bulk eval.",
            requires_restart=False,
            is_required=False,
            env_var="AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE",
        ),
    ]
    sections.append(ConfigSection(name="Auth", items=auth_items))

    # Chunker
    chk = settings.chunker
    chk_items = [
        ConfigItem(
            key="chunker.tokenizer",
            value=chk.tokenizer,
            default_value="google/flan-t5-base",
            is_editable=True,
            source=_config_source("CHUNKER__TOKENIZER"),
            description="Hugging Face tokenizer ID for semchunk.",
            requires_restart=True,
            is_required=False,
            env_var="CHUNKER__TOKENIZER",
        ),
        ConfigItem(
            key="chunker.chunk_size",
            value=chk.chunk_size,
            default_value=500,
            is_editable=True,
            source=_config_source("CHUNKER__CHUNK_SIZE"),
            description="Max tokens per chunk. Align with embedding model max input.",
            requires_restart=True,
            is_required=False,
            env_var="CHUNKER__CHUNK_SIZE",
        ),
        ConfigItem(
            key="chunker.overlap_percent",
            value=chk.overlap_percent,
            default_value=0.15,
            is_editable=True,
            source=_config_source("CHUNKER__OVERLAP_PERCENT"),
            description="Chunk overlap ratio 0-1 (e.g. 0.15 = 15%).",
            requires_restart=True,
            is_required=False,
            env_var="CHUNKER__OVERLAP_PERCENT",
        ),
    ]
    sections.append(ConfigSection(name="Chunker", items=chk_items))

    # Retrieval
    ret = settings.retrieval
    rer = ret.reranker
    ret_items = [
        ConfigItem(
            key="retrieval.episode_relevance_threshold",
            value=ret.episode_relevance_threshold,
            default_value=0.5,
            is_editable=True,
            source=_config_source("RETRIEVAL__EPISODE_RELEVANCE_THRESHOLD"),
            description="Min relevance for episodes in context.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__EPISODE_RELEVANCE_THRESHOLD",
        ),
        ConfigItem(
            key="retrieval.max_episodes_when_constraints",
            value=ret.max_episodes_when_constraints,
            default_value=3,
            is_editable=True,
            source=_config_source("RETRIEVAL__MAX_EPISODES_WHEN_CONSTRAINTS"),
            description="Max episodes when constraints exist.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__MAX_EPISODES_WHEN_CONSTRAINTS",
        ),
        ConfigItem(
            key="retrieval.max_episodes_default",
            value=ret.max_episodes_default,
            default_value=5,
            is_editable=True,
            source=_config_source("RETRIEVAL__MAX_EPISODES_DEFAULT"),
            description="Max episodes when no constraints.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__MAX_EPISODES_DEFAULT",
        ),
        ConfigItem(
            key="retrieval.max_constraint_tokens",
            value=ret.max_constraint_tokens,
            default_value=400,
            is_editable=True,
            source=_config_source("RETRIEVAL__MAX_CONSTRAINT_TOKENS"),
            description="Token budget for active constraints.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__MAX_CONSTRAINT_TOKENS",
        ),
        ConfigItem(
            key="retrieval.default_step_timeout_ms",
            value=ret.default_step_timeout_ms,
            default_value=500,
            is_editable=True,
            source=_config_source("RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS"),
            description="Per-step timeout (ms).",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS",
        ),
        ConfigItem(
            key="retrieval.total_timeout_ms",
            value=ret.total_timeout_ms,
            default_value=2000,
            is_editable=True,
            source=_config_source("RETRIEVAL__TOTAL_TIMEOUT_MS"),
            description="Total retrieval budget (ms).",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__TOTAL_TIMEOUT_MS",
        ),
        ConfigItem(
            key="retrieval.graph_timeout_ms",
            value=ret.graph_timeout_ms,
            default_value=1000,
            is_editable=True,
            source=_config_source("RETRIEVAL__GRAPH_TIMEOUT_MS"),
            description="Graph step timeout (ms).",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__GRAPH_TIMEOUT_MS",
        ),
        ConfigItem(
            key="retrieval.fact_timeout_ms",
            value=ret.fact_timeout_ms,
            default_value=200,
            is_editable=True,
            source=_config_source("RETRIEVAL__FACT_TIMEOUT_MS"),
            description="Fact lookup timeout (ms).",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__FACT_TIMEOUT_MS",
        ),
        ConfigItem(
            key="retrieval.hnsw_ef_search",
            value=ret.hnsw_ef_search,
            default_value=64,
            is_editable=True,
            source=_config_source("RETRIEVAL__HNSW_EF_SEARCH"),
            description="pgvector HNSW ef_search. Higher = recall, slower.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__HNSW_EF_SEARCH",
        ),
        ConfigItem(
            key="retrieval.reranker.recency_weight",
            value=rer.recency_weight,
            default_value=0.1,
            is_editable=True,
            source=_config_source("RETRIEVAL__RERANKER__RECENCY_WEIGHT"),
            description="Recency weight in reranking.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__RERANKER__RECENCY_WEIGHT",
        ),
        ConfigItem(
            key="retrieval.reranker.relevance_weight",
            value=rer.relevance_weight,
            default_value=0.5,
            is_editable=True,
            source=_config_source("RETRIEVAL__RERANKER__RELEVANCE_WEIGHT"),
            description="Relevance weight in reranking.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__RERANKER__RELEVANCE_WEIGHT",
        ),
        ConfigItem(
            key="retrieval.reranker.confidence_weight",
            value=rer.confidence_weight,
            default_value=0.2,
            is_editable=True,
            source=_config_source("RETRIEVAL__RERANKER__CONFIDENCE_WEIGHT"),
            description="Confidence weight in reranking.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__RERANKER__CONFIDENCE_WEIGHT",
        ),
        ConfigItem(
            key="retrieval.reranker.active_constraint_bonus",
            value=rer.active_constraint_bonus,
            default_value=0.2,
            is_editable=True,
            source=_config_source("RETRIEVAL__RERANKER__ACTIVE_CONSTRAINT_BONUS"),
            description="Bonus for active constraints.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__RERANKER__ACTIVE_CONSTRAINT_BONUS",
        ),
    ]
    sections.append(ConfigSection(name="Retrieval", items=ret_items))

    # Features
    feat = settings.features
    feat_items = [
        ConfigItem(
            key="features.stable_keys_enabled",
            value=feat.stable_keys_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__STABLE_KEYS_ENABLED"),
            description="SHA256-based stable keys for consolidation.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__STABLE_KEYS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.write_time_facts_enabled",
            value=feat.write_time_facts_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__WRITE_TIME_FACTS_ENABLED"),
            description="Populate semantic store at write time.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__WRITE_TIME_FACTS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.batch_embeddings_enabled",
            value=feat.batch_embeddings_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__BATCH_EMBEDDINGS_ENABLED"),
            description="Single embed_batch() per turn.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__BATCH_EMBEDDINGS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.store_async",
            value=feat.store_async,
            default_value=False,
            is_editable=True,
            source=_config_source("FEATURES__STORE_ASYNC"),
            description="Enqueue turn writes to Redis. Reduces latency; requires Redis.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__STORE_ASYNC",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.cached_embeddings_enabled",
            value=feat.cached_embeddings_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__CACHED_EMBEDDINGS_ENABLED"),
            description="Cache embeddings in Redis.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__CACHED_EMBEDDINGS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.retrieval_timeouts_enabled",
            value=feat.retrieval_timeouts_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__RETRIEVAL_TIMEOUTS_ENABLED"),
            description="Per-step asyncio.wait_for timeouts.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__RETRIEVAL_TIMEOUTS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.skip_if_found_cross_group",
            value=feat.skip_if_found_cross_group,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__SKIP_IF_FOUND_CROSS_GROUP"),
            description="Skip remaining steps on fact hit.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__SKIP_IF_FOUND_CROSS_GROUP",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.db_dependency_counts",
            value=feat.db_dependency_counts,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__DB_DEPENDENCY_COUNTS"),
            description="DB-side aggregation for forgetting.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__DB_DEPENDENCY_COUNTS",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.bounded_state_enabled",
            value=feat.bounded_state_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__BOUNDED_STATE_ENABLED"),
            description="LRU+TTL state maps in working memory.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__BOUNDED_STATE_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.hnsw_ef_search_tuning",
            value=feat.hnsw_ef_search_tuning,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__HNSW_EF_SEARCH_TUNING"),
            description="Query-time HNSW ef_search tuning.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__HNSW_EF_SEARCH_TUNING",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.constraint_extraction_enabled",
            value=feat.constraint_extraction_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__CONSTRAINT_EXTRACTION_ENABLED"),
            description="Extract goals, values, policies at write time.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__CONSTRAINT_EXTRACTION_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_constraint_extractor",
            value=feat.use_llm_constraint_extractor,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR"),
            description="LLM for constraint extraction (vs rule-based).",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_write_time_facts",
            value=feat.use_llm_write_time_facts,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_WRITE_TIME_FACTS"),
            description="LLM for write-time fact extraction.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_WRITE_TIME_FACTS",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_query_classifier_only",
            value=feat.use_llm_query_classifier_only,
            default_value=False,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY"),
            description="Always use LLM for query classification; skip fast pattern.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_salience_refinement",
            value=feat.use_llm_salience_refinement,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_SALIENCE_REFINEMENT"),
            description="LLM salience in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_SALIENCE_REFINEMENT",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_pii_redaction",
            value=feat.use_llm_pii_redaction,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_PII_REDACTION"),
            description="LLM PII spans in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_PII_REDACTION",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_write_gate_importance",
            value=feat.use_llm_write_gate_importance,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_WRITE_GATE_IMPORTANCE"),
            description="LLM importance in write gate.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_WRITE_GATE_IMPORTANCE",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_memory_type",
            value=feat.use_llm_memory_type,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_MEMORY_TYPE"),
            description="LLM memory_type in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_MEMORY_TYPE",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_confidence",
            value=feat.use_llm_confidence,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONFIDENCE"),
            description="LLM confidence in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONFIDENCE",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_context_tags",
            value=feat.use_llm_context_tags,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONTEXT_TAGS"),
            description="LLM context_tags when caller does not provide tags.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONTEXT_TAGS",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_decay_rate",
            value=feat.use_llm_decay_rate,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_DECAY_RATE"),
            description="LLM decay_rate in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_DECAY_RATE",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_conflict_detection_only",
            value=feat.use_llm_conflict_detection_only,
            default_value=False,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONFLICT_DETECTION_ONLY"),
            description="Always use LLM for conflict detection.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONFLICT_DETECTION_ONLY",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_constraint_reranker",
            value=feat.use_llm_constraint_reranker,
            default_value=False,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONSTRAINT_RERANKER"),
            description="LLM for constraint reranking (1 call per read).",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONSTRAINT_RERANKER",
            options=["true", "false"],
        ),
    ]
    sections.append(ConfigSection(name="Features", items=feat_items))

    return DashboardConfigResponse(sections=sections)


@router.put("/config")
async def dashboard_config_update(
    body: ConfigUpdateRequest,
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Update editable config settings. Persisted to .env. Restart required for most changes."""
    for key in body.updates:
        if key in _WRITE_PROTECTED_FIELDS:
            raise HTTPException(
                status_code=403,
                detail=f"Cannot modify secret field '{key}' via dashboard",
            )
        if key not in _EDITABLE_SETTINGS:
            raise HTTPException(status_code=400, detail=f"Setting '{key}' is not editable")

    errors = _validate_config_updates(body.updates)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    env_updates: dict[str, Any] = {}
    for key, val in body.updates.items():
        env_var = _CONFIG_KEY_TO_ENV.get(key)
        if env_var:
            if key == "embedding_internal.dimensions":
                env_updates[env_var] = int(val)
            elif key == "cors_origins":
                if isinstance(val, list):
                    env_updates[env_var] = ",".join(str(v) for v in val)
                elif val is None or val == "":
                    env_updates[env_var] = ""
                else:
                    env_updates[env_var] = str(val)
            else:
                env_updates[env_var] = val

    try:
        env_path = get_env_path()
        if not env_path.parent.exists():
            raise HTTPException(
                status_code=503,
                detail="Project root not found; cannot persist to .env",
            )
        update_env(env_updates)
    except OSError as e:
        logger.warning("config_persist_env_failed", path=str(get_env_path()), error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"Cannot write to .env: {e}",
        ) from e

    return {
        "success": True,
        "message": "Settings saved to .env. Restart the server for changes to take effect.",
    }
