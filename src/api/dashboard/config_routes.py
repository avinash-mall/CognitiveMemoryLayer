"""Dashboard configuration routes."""

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...core.config import (
    AuthSettings,
    ChunkerSettings,
    EmbeddingInternalSettings,
    FeatureFlags,
    PerformanceSettings,
    RerankerSettings,
    RetrievalSettings,
    get_settings,
)
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


def _field_default(model_cls: Any, field_name: str) -> Any:
    """Read the canonical default value directly from a Pydantic config model."""
    return model_cls.model_fields[field_name].default


def _feature_item(
    feat: FeatureFlags,
    field_name: str,
    *,
    description: str | None = None,
) -> ConfigItem:
    """Build a ConfigItem for a FeatureFlags field directly from Pydantic metadata."""
    field_info = FeatureFlags.model_fields[field_name]
    default_val = field_info.default
    env_var = f"FEATURES__{field_name.upper()}"
    is_bool = isinstance(default_val, bool)
    return ConfigItem(
        key=f"features.{field_name}",
        value=getattr(feat, field_name),
        default_value=default_val,
        is_editable=True,
        source=_config_source(env_var),
        description=description or field_info.description or "",
        requires_restart=True,
        is_required=False,
        env_var=env_var,
        options=["true", "false"] if is_bool else None,
    )


def _validate_config_updates(updates: dict[str, Any]) -> list[str]:
    """Validate config updates. Returns list of error messages."""
    errors: list[str] = []
    for key, val in updates.items():
        if key == "embedding_internal.dimensions":
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append("embedding_internal.dimensions must be a positive integer")
        elif key == "embedding_internal.device":
            if str(val) not in {"auto", "cpu", "cuda"}:
                errors.append("embedding_internal.device must be one of: auto, cpu, cuda")
        elif key == "embedding_internal.local_batch_size":
            if not isinstance(val, (int, float)) or val < 0:
                errors.append("embedding_internal.local_batch_size must be >= 0 (0 = auto)")
        elif key == "embedding_internal.batch_wait_ms":
            if not isinstance(val, (int, float)) or val < 0:
                errors.append("embedding_internal.batch_wait_ms must be >= 0")
        elif key in {"llm_internal.extra_body", "llm_eval.extra_body"}:
            if isinstance(val, str) and val.strip():
                try:
                    parsed = json.loads(val)
                    if not isinstance(parsed, dict):
                        errors.append(f"{key} must be a JSON object")
                except json.JSONDecodeError as exc:
                    errors.append(f"{key} must be valid JSON: {exc.msg}")
            elif val is not None and not isinstance(val, (dict, str)):
                errors.append(f"{key} must be a JSON object")
        elif key == "chunker.chunk_size":
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append("chunker.chunk_size must be a positive integer")
        elif key == "chunker.overlap_percent":
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append("chunker.overlap_percent must be between 0 and 1")
        elif key == "auth.rate_limit_requests_per_minute":
            if not isinstance(val, (int, float)) or val < 0:
                errors.append("auth.rate_limit_requests_per_minute must be non-negative")
        elif key == "performance.gate_executor_workers":
            if not isinstance(val, (int, float)) or val < 0:
                errors.append("performance.gate_executor_workers must be >= 0 (0 = auto)")
        elif key.startswith("retrieval.") and "timeout" in key:
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append(f"{key} must be a positive number")
        elif key == "retrieval.cross_encoder_weight" or key.startswith("retrieval.reranker."):
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"{key} must be between 0 and 1")
        elif key == "features.thalamic_surprise_threshold":
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append("features.thalamic_surprise_threshold must be between 0 and 1")
        elif key == "features.temporal_contiguity_neighbours":
            if not isinstance(val, (int, float)) or val < 1 or val > 10:
                errors.append("features.temporal_contiguity_neighbours must be between 1 and 10")
        elif key == "features.windowed_consolidation_turn_size":
            if not isinstance(val, (int, float)) or val < 4 or val > 80:
                errors.append("features.windowed_consolidation_turn_size must be between 4 and 80")
        elif key in {"features.prospective_index_count", "features.consolidation_recurrence_min"}:
            if not isinstance(val, (int, float)) or val < 1:
                errors.append(f"{key} must be >= 1")
    return errors


@router.get("/config", response_model=DashboardConfigResponse)
async def dashboard_config(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Configuration snapshot. Secrets are masked. Changes persist to .env; restart required for most settings."""
    settings = get_settings()
    sections: list[ConfigSection] = []

    # ── Application ────────────────────────────────────────────────────────────
    sections.append(
        ConfigSection(
            name="Application",
            items=[
                ConfigItem(
                    key="app_name",
                    value=settings.app_name,
                    default_value="CognitiveMemoryLayer",
                    is_editable=True,
                    source=_config_source("APP_NAME"),
                    description="Application name used in responses and UI.",
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
                    description="Enable debug mode. Exposes verbose 500 error details. Set false in production.",
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
                    description='Comma-separated CORS allowed origins. Unset uses default list; ["*"] disables credentials.',
                    requires_restart=True,
                    is_required=False,
                    env_var="CORS_ORIGINS",
                ),
            ],
        )
    )

    # ── Auth ───────────────────────────────────────────────────────────────────
    auth_s = settings.auth
    sections.append(
        ConfigSection(
            name="Auth",
            items=[
                ConfigItem(
                    key="auth.api_key",
                    value="****" if auth_s.api_key else "",
                    default_value="",
                    is_secret=True,
                    is_editable=False,
                    source=_config_source("AUTH__API_KEY"),
                    description="API key for all memory operations (X-API-Key header). Required.",
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
                    description="Admin API key for dashboard, consolidation, and forgetting endpoints.",
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
                    description="Default tenant ID when X-Tenant-ID header is not provided.",
                    requires_restart=True,
                    is_required=False,
                    env_var="AUTH__DEFAULT_TENANT_ID",
                ),
                ConfigItem(
                    key="auth.rate_limit_requests_per_minute",
                    value=auth_s.rate_limit_requests_per_minute,
                    default_value=_field_default(AuthSettings, "rate_limit_requests_per_minute"),
                    is_editable=True,
                    source=_config_source("AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE"),
                    description="Per-tenant rate limit (requests/min). 0 disables rate limiting (default).",
                    requires_restart=False,
                    is_required=False,
                    env_var="AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE",
                ),
            ],
        )
    )

    # ── Database ───────────────────────────────────────────────────────────────
    db_s = settings.database
    sections.append(
        ConfigSection(
            name="Database",
            items=[
                ConfigItem(
                    key="database.postgres_url",
                    value=db_s.postgres_url,
                    default_value="postgresql+asyncpg://memory:memory@localhost/memory",
                    is_editable=False,
                    source=_config_source("DATABASE__POSTGRES_URL"),
                    description="PostgreSQL connection URL. Format: postgresql+asyncpg://user:pass@host:port/db. Required.",
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
                    description="Neo4j Bolt URL for the knowledge graph. Optional.",
                    requires_restart=True,
                    is_required=False,
                    env_var="DATABASE__NEO4J_URL",
                ),
                ConfigItem(
                    key="database.neo4j_browser_url",
                    value=db_s.neo4j_browser_url or "",
                    default_value="",
                    is_editable=False,
                    source=_config_source("DATABASE__NEO4J_BROWSER_URL"),
                    description="Neo4j Bolt URL for the dashboard graph viewer (neovis.js). When the API runs in Docker, set to bolt://localhost:7687.",
                    requires_restart=True,
                    is_required=False,
                    env_var="DATABASE__NEO4J_BROWSER_URL",
                ),
                ConfigItem(
                    key="database.neo4j_user",
                    value=db_s.neo4j_user,
                    default_value="neo4j",
                    is_editable=False,
                    source=_config_source("DATABASE__NEO4J_USER"),
                    description="Neo4j username.",
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
                    description="Neo4j password. Set when Neo4j auth is enabled.",
                    requires_restart=True,
                    is_required=False,
                    env_var="DATABASE__NEO4J_PASSWORD",
                ),
                ConfigItem(
                    key="database.redis_url",
                    value=db_s.redis_url,
                    default_value="redis://localhost:6379",
                    is_editable=False,
                    source=_config_source("DATABASE__REDIS_URL"),
                    description="Redis URL for embedding cache, rate limiting, labile state, and Celery broker.",
                    requires_restart=True,
                    is_required=False,
                    env_var="DATABASE__REDIS_URL",
                ),
            ],
        )
    )

    # ── Embedding (Internal) ───────────────────────────────────────────────────
    emb = settings.embedding_internal
    sections.append(
        ConfigSection(
            name="Embedding (Internal)",
            items=[
                ConfigItem(
                    key="embedding_internal.provider",
                    value=emb.provider or "local",
                    default_value="local",
                    is_editable=True,
                    source=_config_source("EMBEDDING_INTERNAL__PROVIDER"),
                    description="Embedding provider: local | openai | openai_compatible | ollama | vllm | mock. 'local' runs sentence-transformers on GPU (auto-detects CUDA) or CPU.",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__PROVIDER",
                    options=["local", "openai", "openai_compatible", "ollama", "vllm", "mock"],
                ),
                ConfigItem(
                    key="embedding_internal.model",
                    value=emb.model or "nomic-ai/nomic-embed-text-v2-moe",
                    default_value="nomic-ai/nomic-embed-text-v2-moe",
                    is_editable=True,
                    source=_config_source("EMBEDDING_INTERNAL__MODEL"),
                    description="Embedding model name. Must match provider. Changing requires re-embedding all stored memories.",
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
                    description="Vector dimension. Must match model output and DB schema (Vector column). Default 768 for nomic-embed-text-v2-moe.",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__DIMENSIONS",
                ),
                ConfigItem(
                    key="embedding_internal.local_model",
                    value=emb.local_model or "nomic-ai/nomic-embed-text-v2-moe",
                    default_value=_field_default(EmbeddingInternalSettings, "local_model"),
                    is_editable=True,
                    source=_config_source("EMBEDDING_INTERNAL__LOCAL_MODEL"),
                    description="Model ID for provider=local (sentence-transformers). Use embedding_internal.device to choose auto, CPU, or CUDA.",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__LOCAL_MODEL",
                ),
                ConfigItem(
                    key="embedding_internal.revision",
                    value=emb.revision or "",
                    default_value=_field_default(EmbeddingInternalSettings, "revision"),
                    is_editable=True,
                    source=_config_source("EMBEDDING_INTERNAL__REVISION"),
                    description="Pinned HuggingFace commit revision for local embedding model reproducibility.",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__REVISION",
                ),
                ConfigItem(
                    key="embedding_internal.device",
                    value=emb.device,
                    default_value=_field_default(EmbeddingInternalSettings, "device"),
                    is_editable=True,
                    source=_config_source("EMBEDDING_INTERNAL__DEVICE"),
                    description="Device for provider=local embeddings. auto prefers CUDA when available; cpu keeps the embedder off GPU; cuda requests GPU.",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__DEVICE",
                    options=["auto", "cpu", "cuda"],
                ),
                ConfigItem(
                    key="embedding_internal.base_url",
                    value=emb.base_url or "",
                    default_value="",
                    is_editable=True,
                    source=_config_source("EMBEDDING_INTERNAL__BASE_URL"),
                    description="OpenAI-compatible base URL. Required for provider=openai_compatible, ollama, or vllm.",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__BASE_URL",
                ),
                ConfigItem(
                    key="embedding_internal.local_batch_size",
                    value=emb.local_batch_size,
                    default_value=_field_default(EmbeddingInternalSettings, "local_batch_size"),
                    is_editable=True,
                    source=_config_source("EMBEDDING_INTERNAL__LOCAL_BATCH_SIZE"),
                    description="Batch size for local embedding inference. 0 = auto (64 on CUDA, 8 on CPU). Reduce if GPU OOM.",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__LOCAL_BATCH_SIZE",
                ),
                ConfigItem(
                    key="embedding_internal.batch_wait_ms",
                    value=emb.batch_wait_ms,
                    default_value=_field_default(EmbeddingInternalSettings, "batch_wait_ms"),
                    is_editable=True,
                    source=_config_source("EMBEDDING_INTERNAL__BATCH_WAIT_MS"),
                    description="Cross-request embedding coalescing window in milliseconds (0 disables batching).",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__BATCH_WAIT_MS",
                ),
                ConfigItem(
                    key="embedding_internal.api_key",
                    value="****" if emb.api_key else "",
                    default_value="",
                    is_secret=True,
                    is_editable=False,
                    source=_config_source("EMBEDDING_INTERNAL__API_KEY"),
                    description="API key for embedding provider. Falls back to OPENAI_API_KEY.",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__API_KEY",
                ),
            ],
        )
    )

    # ── LLM Internal ──────────────────────────────────────────────────────────
    llmi = settings.llm_internal
    sections.append(
        ConfigSection(
            name="LLM Internal",
            items=[
                ConfigItem(
                    key="llm_internal.provider",
                    value=llmi.provider,
                    default_value="openai",
                    is_editable=True,
                    source=_config_source("LLM_INTERNAL__PROVIDER"),
                    description="LLM provider for extraction, consolidation, and query classification: openai | openai_compatible | ollama | anthropic | gemini | vllm | sglang.",
                    requires_restart=True,
                    is_required=False,
                    env_var="LLM_INTERNAL__PROVIDER",
                    options=[
                        "openai",
                        "openai_compatible",
                        "ollama",
                        "anthropic",
                        "gemini",
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
                    description="Model name for internal tasks. ~1 LLM call per write chunk; ~1-2 per read; ~5-10 per turn.",
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
                    description="Base URL for the internal LLM. Required for ollama, vllm, sglang, or openai_compatible.",
                    requires_restart=True,
                    is_required=False,
                    env_var="LLM_INTERNAL__BASE_URL",
                ),
                ConfigItem(
                    key="llm_internal.extra_body",
                    value=json.dumps(llmi.extra_body) if llmi.extra_body else "",
                    default_value="",
                    is_editable=True,
                    source=_config_source("LLM_INTERNAL__EXTRA_BODY"),
                    description='Optional JSON object merged into chat completion requests (e.g. {"chat_template_kwargs": {"enable_thinking": false}}).',
                    requires_restart=True,
                    is_required=False,
                    env_var="LLM_INTERNAL__EXTRA_BODY",
                ),
                ConfigItem(
                    key="llm_internal.api_key",
                    value="****" if llmi.api_key else "",
                    default_value="",
                    is_secret=True,
                    is_editable=False,
                    source=_config_source("LLM_INTERNAL__API_KEY"),
                    description="API key for the internal LLM. Falls back to OPENAI_API_KEY.",
                    requires_restart=True,
                    is_required=False,
                    env_var="LLM_INTERNAL__API_KEY",
                ),
            ],
        )
    )

    # ── LLM Eval ──────────────────────────────────────────────────────────────
    llme = settings.llm_eval
    sections.append(
        ConfigSection(
            name="LLM Eval",
            items=[
                ConfigItem(
                    key="llm_eval.provider",
                    value=llme.provider or "(uses llm_internal)",
                    default_value=None,
                    is_editable=True,
                    source=_config_source("LLM_EVAL__PROVIDER"),
                    description="LLM provider for evaluation QA and judge. Unset falls back to LLM_INTERNAL__PROVIDER.",
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
                    description="Model for evaluation QA and judge. Unset falls back to LLM_INTERNAL__MODEL.",
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
                    description="Base URL for eval LLM. Unset falls back to LLM_INTERNAL__BASE_URL.",
                    requires_restart=True,
                    is_required=False,
                    env_var="LLM_EVAL__BASE_URL",
                ),
                ConfigItem(
                    key="llm_eval.extra_body",
                    value=json.dumps(llme.extra_body) if llme.extra_body else "",
                    default_value="",
                    is_editable=True,
                    source=_config_source("LLM_EVAL__EXTRA_BODY"),
                    description="Optional JSON object merged into evaluation LLM chat completion requests.",
                    requires_restart=True,
                    is_required=False,
                    env_var="LLM_EVAL__EXTRA_BODY",
                ),
                ConfigItem(
                    key="llm_eval.api_key",
                    value="****" if llme.api_key else "",
                    default_value="",
                    is_secret=True,
                    is_editable=False,
                    source=_config_source("LLM_EVAL__API_KEY"),
                    description="API key for eval LLM. Falls back to OPENAI_API_KEY.",
                    requires_restart=True,
                    is_required=False,
                    env_var="LLM_EVAL__API_KEY",
                ),
            ],
        )
    )

    # ── Chunker ────────────────────────────────────────────────────────────────
    chk = settings.chunker
    sections.append(
        ConfigSection(
            name="Chunker",
            items=[
                ConfigItem(
                    key="chunker.tokenizer",
                    value=chk.tokenizer,
                    default_value=_field_default(ChunkerSettings, "tokenizer"),
                    is_editable=True,
                    source=_config_source("CHUNKER__TOKENIZER"),
                    description="Hugging Face tokenizer ID for semchunk semantic chunking.",
                    requires_restart=True,
                    is_required=False,
                    env_var="CHUNKER__TOKENIZER",
                ),
                ConfigItem(
                    key="chunker.chunk_size",
                    value=chk.chunk_size,
                    default_value=_field_default(ChunkerSettings, "chunk_size"),
                    is_editable=True,
                    source=_config_source("CHUNKER__CHUNK_SIZE"),
                    description="Maximum tokens per chunk. Align with the embedding model's max input length.",
                    requires_restart=True,
                    is_required=False,
                    env_var="CHUNKER__CHUNK_SIZE",
                ),
                ConfigItem(
                    key="chunker.overlap_percent",
                    value=chk.overlap_percent,
                    default_value=_field_default(ChunkerSettings, "overlap_percent"),
                    is_editable=True,
                    source=_config_source("CHUNKER__OVERLAP_PERCENT"),
                    description="Chunk overlap ratio 0-1 (e.g. 0.15 = 15% overlap between adjacent chunks).",
                    requires_restart=True,
                    is_required=False,
                    env_var="CHUNKER__OVERLAP_PERCENT",
                ),
            ],
        )
    )

    # ── Retrieval ──────────────────────────────────────────────────────────────
    ret = settings.retrieval
    sections.append(
        ConfigSection(
            name="Retrieval",
            items=[
                ConfigItem(
                    key="retrieval.episode_relevance_threshold",
                    value=ret.episode_relevance_threshold,
                    default_value=_field_default(RetrievalSettings, "episode_relevance_threshold"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__EPISODE_RELEVANCE_THRESHOLD"),
                    description="Minimum relevance score for episodes to appear in context. Prevents low-signal episodes diluting constraints.",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__EPISODE_RELEVANCE_THRESHOLD",
                ),
                ConfigItem(
                    key="retrieval.max_episodes_when_constraints",
                    value=ret.max_episodes_when_constraints,
                    default_value=_field_default(
                        RetrievalSettings, "max_episodes_when_constraints"
                    ),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__MAX_EPISODES_WHEN_CONSTRAINTS"),
                    description="Maximum episodes returned when active constraints are present (reduces dilution).",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__MAX_EPISODES_WHEN_CONSTRAINTS",
                ),
                ConfigItem(
                    key="retrieval.max_episodes_default",
                    value=ret.max_episodes_default,
                    default_value=_field_default(RetrievalSettings, "max_episodes_default"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__MAX_EPISODES_DEFAULT"),
                    description="Maximum episodes returned when no active constraints are present.",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__MAX_EPISODES_DEFAULT",
                ),
                ConfigItem(
                    key="retrieval.max_constraint_tokens",
                    value=ret.max_constraint_tokens,
                    default_value=_field_default(RetrievalSettings, "max_constraint_tokens"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__MAX_CONSTRAINT_TOKENS"),
                    description="Token budget reserved for active constraints in the assembled memory packet.",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__MAX_CONSTRAINT_TOKENS",
                ),
                ConfigItem(
                    key="retrieval.hnsw_ef_search",
                    value=ret.hnsw_ef_search,
                    default_value=_field_default(RetrievalSettings, "hnsw_ef_search"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__HNSW_EF_SEARCH"),
                    description="pgvector HNSW ef_search override. Higher values improve recall at the cost of speed.",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__HNSW_EF_SEARCH",
                ),
                ConfigItem(
                    key="retrieval.default_step_timeout_ms",
                    value=ret.default_step_timeout_ms,
                    default_value=_field_default(RetrievalSettings, "default_step_timeout_ms"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS"),
                    description="Per-step asyncio timeout (ms). Applied to each retrieval step (vector, graph, facts, constraints).",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS",
                ),
                ConfigItem(
                    key="retrieval.total_timeout_ms",
                    value=ret.total_timeout_ms,
                    default_value=_field_default(RetrievalSettings, "total_timeout_ms"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__TOTAL_TIMEOUT_MS"),
                    description="Total retrieval budget (ms). Steps completing after this are discarded.",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__TOTAL_TIMEOUT_MS",
                ),
                ConfigItem(
                    key="retrieval.graph_timeout_ms",
                    value=ret.graph_timeout_ms,
                    default_value=_field_default(RetrievalSettings, "graph_timeout_ms"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__GRAPH_TIMEOUT_MS"),
                    description="Timeout for the graph (Neo4j) retrieval step (ms).",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__GRAPH_TIMEOUT_MS",
                ),
                ConfigItem(
                    key="retrieval.fact_timeout_ms",
                    value=ret.fact_timeout_ms,
                    default_value=_field_default(RetrievalSettings, "fact_timeout_ms"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__FACT_TIMEOUT_MS"),
                    description="Timeout for semantic fact lookup (ms).",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__FACT_TIMEOUT_MS",
                ),
                ConfigItem(
                    key="retrieval.cross_encoder_model",
                    value=ret.cross_encoder_model,
                    default_value=_field_default(RetrievalSettings, "cross_encoder_model"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__CROSS_ENCODER_MODEL"),
                    description="Cross-Encoder model ID for fine-grained candidate reranking (v3.0).",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__CROSS_ENCODER_MODEL",
                ),
                ConfigItem(
                    key="retrieval.cross_encoder_weight",
                    value=ret.cross_encoder_weight,
                    default_value=_field_default(RetrievalSettings, "cross_encoder_weight"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__CROSS_ENCODER_WEIGHT"),
                    description="Blend weight (0-1) of Cross-Encoder probability in final reranker score.",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__CROSS_ENCODER_WEIGHT",
                ),
            ],
        )
    )

    # ── Retrieval — Reranker ───────────────────────────────────────────────────
    rer = ret.reranker
    sections.append(
        ConfigSection(
            name="Retrieval — Reranker",
            items=[
                ConfigItem(
                    key="retrieval.reranker.relevance_weight",
                    value=rer.relevance_weight,
                    default_value=_field_default(RerankerSettings, "relevance_weight"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__RERANKER__RELEVANCE_WEIGHT"),
                    description="Weight applied to semantic relevance score (0-1). Dominant factor in final rank.",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__RERANKER__RELEVANCE_WEIGHT",
                ),
                ConfigItem(
                    key="retrieval.reranker.confidence_weight",
                    value=rer.confidence_weight,
                    default_value=_field_default(RerankerSettings, "confidence_weight"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__RERANKER__CONFIDENCE_WEIGHT"),
                    description="Weight applied to memory confidence score (0-1).",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__RERANKER__CONFIDENCE_WEIGHT",
                ),
                ConfigItem(
                    key="retrieval.reranker.recency_weight",
                    value=rer.recency_weight,
                    default_value=_field_default(RerankerSettings, "recency_weight"),
                    is_editable=True,
                    source=_config_source("RETRIEVAL__RERANKER__RECENCY_WEIGHT"),
                    description="Weight applied to recency (0-1). Keep low to avoid recency bias over relevance.",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__RERANKER__RECENCY_WEIGHT",
                ),
            ],
        )
    )

    # ── Performance ────────────────────────────────────────────────────────────
    perf = settings.performance
    sections.append(
        ConfigSection(
            name="Performance",
            items=[
                ConfigItem(
                    key="performance.gate_executor_workers",
                    value=perf.gate_executor_workers,
                    default_value=_field_default(PerformanceSettings, "gate_executor_workers"),
                    is_editable=True,
                    source=_config_source("PERFORMANCE__GATE_EXECUTOR_WORKERS"),
                    description="Thread-pool size for write-gate novelty checks. 0 = auto (min(cpu_count, 8)).",
                    requires_restart=True,
                    is_required=False,
                    env_var="PERFORMANCE__GATE_EXECUTOR_WORKERS",
                ),
            ],
        )
    )

    # ── Features — Core & LLM ─────────────────────────────────────────────────
    feat = settings.features
    sections.append(
        ConfigSection(
            name="Features — Core & LLM",
            items=[
                _feature_item(feat, "use_llm_enabled"),
                _feature_item(feat, "batch_write_system1_fastpath"),
                _feature_item(feat, "write_time_facts_enabled"),
                _feature_item(feat, "constraint_extraction_enabled"),
                _feature_item(feat, "cached_embeddings_enabled"),
                _feature_item(feat, "retrieval_timeouts_enabled"),
                _feature_item(feat, "skip_if_found_cross_group"),
                _feature_item(feat, "hnsw_ef_search_tuning"),
            ],
        )
    )

    # ── Features — Retrieval & Consolidation ──────────────────────────────────
    sections.append(
        ConfigSection(
            name="Features — Retrieval & Consolidation",
            items=[
                _feature_item(feat, "graph_results_in_packet"),
                _feature_item(feat, "graph_pagerank_enabled"),
                _feature_item(feat, "temporal_contiguity_enabled"),
                _feature_item(feat, "temporal_contiguity_neighbours"),
                _feature_item(feat, "hyde_retrieval_enabled"),
                _feature_item(feat, "llm_multi_angle_expansion_enabled"),
                _feature_item(feat, "llm_judged_delivery_enabled"),
                _feature_item(feat, "temporal_resolution_enabled"),
                _feature_item(feat, "adaptive_episode_budget_enabled"),
                _feature_item(feat, "pii_redaction_enabled"),
                _feature_item(feat, "prospective_indexing_enabled"),
                _feature_item(feat, "prospective_index_count"),
                _feature_item(feat, "consolidation_scheduler_enabled"),
                _feature_item(feat, "consolidation_recurrence_min"),
                _feature_item(feat, "consolidation_detail_recovery_enabled"),
            ],
        )
    )

    # ── Features — Neuro-Cognitive & ActiveCML ────────────────────────────────
    sections.append(
        ConfigSection(
            name="Features — Neuro-Cognitive & ActiveCML",
            items=[
                _feature_item(feat, "somatic_markers_enabled"),
                _feature_item(feat, "working_memory_context_binding_enabled"),
                _feature_item(feat, "thalamic_triage_enabled"),
                _feature_item(feat, "thalamic_surprise_threshold"),
                _feature_item(feat, "ecphory_reactivation_enabled"),
                _feature_item(feat, "reality_monitoring_enabled"),
                _feature_item(feat, "dmn_theory_of_mind_enabled"),
                _feature_item(feat, "multi_hop_recollection_enabled"),
                _feature_item(feat, "intrinsic_memory_enabled"),
                _feature_item(feat, "laya_system1_enabled"),
                _feature_item(feat, "laya_model_id"),
            ],
        )
    )

    # ── Features — CML v3.0 Whole-Brain ───────────────────────────────────────
    sections.append(
        ConfigSection(
            name="Features — CML v3.0 Whole-Brain",
            items=[
                _feature_item(feat, "sparse_bm25_enabled"),
                _feature_item(feat, "hypergraph_ppr_enabled"),
                _feature_item(feat, "conway_hierarchy_enabled"),
                _feature_item(feat, "cross_encoder_enabled"),
                _feature_item(feat, "acc_iterative_loop_enabled"),
                _feature_item(feat, "neuromodulation_enabled"),
                _feature_item(feat, "preplay_simulation_enabled"),
                _feature_item(feat, "procedural_rpe_enabled"),
                _feature_item(feat, "perirhinal_multimodal_enabled"),
                _feature_item(feat, "windowed_consolidation_enabled"),
                _feature_item(feat, "windowed_consolidation_turn_size"),
            ],
        )
    )

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
            elif key in {"llm_internal.extra_body", "llm_eval.extra_body"}:
                if isinstance(val, dict):
                    env_updates[env_var] = json.dumps(val)
                elif val is None or str(val).strip() == "":
                    env_updates[env_var] = ""
                else:
                    env_updates[env_var] = str(val).strip()
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
