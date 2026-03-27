"""Dashboard configuration routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...core.config import EmbeddingInternalSettings, SummarizerInternalSettings, get_settings
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
        elif key == "embedding_internal.device":
            if str(val) not in {"auto", "cpu", "cuda"}:
                errors.append("embedding_internal.device must be one of: auto, cpu, cuda")
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
        elif key in (
            "summarizer_internal.max_input_chars",
            "summarizer_internal.max_output_chars",
            "summarizer_internal.max_length",
        ):
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append(f"{key} must be a positive integer")
        elif key == "summarizer_internal.min_length":
            if not isinstance(val, (int, float)) or val < 0:
                errors.append("summarizer_internal.min_length must be non-negative")
        elif key == "summarizer_internal.device":
            if not isinstance(val, (int, float)) or val < -1:
                errors.append("summarizer_internal.device must be -1 (CPU) or >= 0 (CUDA index)")
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
                    default_value=60,
                    is_editable=True,
                    source=_config_source("AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE"),
                    description="Per-tenant rate limit (requests/min). 0 disables. Raise to 600+ for bulk eval.",
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
    emb = getattr(settings, "embedding_internal", None) or EmbeddingInternalSettings()
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
                    default_value="nomic-ai/nomic-embed-text-v2-moe",
                    is_editable=True,
                    source=_config_source("EMBEDDING_INTERNAL__LOCAL_MODEL"),
                    description="Model ID for provider=local (sentence-transformers). Use embedding_internal.device to choose auto, CPU, or CUDA.",
                    requires_restart=True,
                    is_required=False,
                    env_var="EMBEDDING_INTERNAL__LOCAL_MODEL",
                ),
                ConfigItem(
                    key="embedding_internal.device",
                    value=emb.device,
                    default_value="auto",
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

    # ── Summarizer (Internal) ──────────────────────────────────────────────────
    summ = getattr(settings, "summarizer_internal", None) or SummarizerInternalSettings()
    sections.append(
        ConfigSection(
            name="Summarizer (Internal)",
            items=[
                ConfigItem(
                    key="summarizer_internal.provider",
                    value=summ.provider,
                    default_value="huggingface",
                    is_editable=True,
                    source=_config_source("SUMMARIZER_INTERNAL__PROVIDER"),
                    description="Summarizer backend. Only 'huggingface' is currently supported.",
                    requires_restart=True,
                    is_required=False,
                    env_var="SUMMARIZER_INTERNAL__PROVIDER",
                    options=["huggingface"],
                ),
                ConfigItem(
                    key="summarizer_internal.model",
                    value=summ.model,
                    default_value="Falconsai/text_summarization",
                    is_editable=True,
                    source=_config_source("SUMMARIZER_INTERNAL__MODEL"),
                    description="Hugging Face model ID for summarization. Used by consolidation gist extraction when LLM is disabled.",
                    requires_restart=True,
                    is_required=False,
                    env_var="SUMMARIZER_INTERNAL__MODEL",
                ),
                ConfigItem(
                    key="summarizer_internal.task",
                    value=summ.task,
                    default_value="summarization",
                    is_editable=True,
                    source=_config_source("SUMMARIZER_INTERNAL__TASK"),
                    description="transformers pipeline task name.",
                    requires_restart=True,
                    is_required=False,
                    env_var="SUMMARIZER_INTERNAL__TASK",
                ),
                ConfigItem(
                    key="summarizer_internal.max_input_chars",
                    value=summ.max_input_chars,
                    default_value=2400,
                    is_editable=True,
                    source=_config_source("SUMMARIZER_INTERNAL__MAX_INPUT_CHARS"),
                    description="Input text is truncated to this many characters before summarization.",
                    requires_restart=True,
                    is_required=False,
                    env_var="SUMMARIZER_INTERNAL__MAX_INPUT_CHARS",
                ),
                ConfigItem(
                    key="summarizer_internal.max_output_chars",
                    value=summ.max_output_chars,
                    default_value=320,
                    is_editable=True,
                    source=_config_source("SUMMARIZER_INTERNAL__MAX_OUTPUT_CHARS"),
                    description="Output is truncated to this many characters after summarization.",
                    requires_restart=True,
                    is_required=False,
                    env_var="SUMMARIZER_INTERNAL__MAX_OUTPUT_CHARS",
                ),
                ConfigItem(
                    key="summarizer_internal.min_length",
                    value=summ.min_length,
                    default_value=24,
                    is_editable=True,
                    source=_config_source("SUMMARIZER_INTERNAL__MIN_LENGTH"),
                    description="Minimum token length for generated summaries (passed to transformers pipeline).",
                    requires_restart=True,
                    is_required=False,
                    env_var="SUMMARIZER_INTERNAL__MIN_LENGTH",
                ),
                ConfigItem(
                    key="summarizer_internal.max_length",
                    value=summ.max_length,
                    default_value=96,
                    is_editable=True,
                    source=_config_source("SUMMARIZER_INTERNAL__MAX_LENGTH"),
                    description="Maximum token length for generated summaries (passed to transformers pipeline).",
                    requires_restart=True,
                    is_required=False,
                    env_var="SUMMARIZER_INTERNAL__MAX_LENGTH",
                ),
                ConfigItem(
                    key="summarizer_internal.device",
                    value=summ.device,
                    default_value=-1,
                    is_editable=True,
                    source=_config_source("SUMMARIZER_INTERNAL__DEVICE"),
                    description="Device for transformers pipeline. -1 = auto-detect (uses GPU 0 if CUDA available, otherwise CPU); 0, 1, … = specific CUDA device index.",
                    requires_restart=True,
                    is_required=False,
                    env_var="SUMMARIZER_INTERNAL__DEVICE",
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
                    default_value="google/flan-t5-base",
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
                    default_value=500,
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
                    default_value=0.15,
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
                    default_value=0.5,
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
                    default_value=3,
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
                    default_value=5,
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
                    default_value=400,
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
                    default_value=64,
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
                    default_value=500,
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
                    default_value=2000,
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
                    default_value=1000,
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
                    default_value=500,
                    is_editable=True,
                    source=_config_source("RETRIEVAL__FACT_TIMEOUT_MS"),
                    description="Timeout for semantic fact lookup (ms).",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__FACT_TIMEOUT_MS",
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
                    default_value=0.5,
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
                    default_value=0.2,
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
                    default_value=0.1,
                    is_editable=True,
                    source=_config_source("RETRIEVAL__RERANKER__RECENCY_WEIGHT"),
                    description="Weight applied to recency (0-1). Keep low to avoid recency bias over relevance.",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__RERANKER__RECENCY_WEIGHT",
                ),
                ConfigItem(
                    key="retrieval.reranker.active_constraint_bonus",
                    value=rer.active_constraint_bonus,
                    default_value=0.2,
                    is_editable=True,
                    source=_config_source("RETRIEVAL__RERANKER__ACTIVE_CONSTRAINT_BONUS"),
                    description="Score bonus added to active constraint memories during reranking (0-1).",
                    requires_restart=True,
                    is_required=False,
                    env_var="RETRIEVAL__RERANKER__ACTIVE_CONSTRAINT_BONUS",
                ),
            ],
        )
    )

    # ── Features — Core ────────────────────────────────────────────────────────
    feat = settings.features
    sections.append(
        ConfigSection(
            name="Features — Core",
            items=[
                ConfigItem(
                    key="features.stable_keys_enabled",
                    value=feat.stable_keys_enabled,
                    default_value=True,
                    is_editable=True,
                    source=_config_source("FEATURES__STABLE_KEYS_ENABLED"),
                    description="Use SHA256-based stable keys for deduplication and consolidation linking.",
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
                    description="Extract and store semantic facts at write time. Populates ReadResponse.facts.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__WRITE_TIME_FACTS_ENABLED",
                    options=["true", "false"],
                ),
                ConfigItem(
                    key="features.constraint_extraction_enabled",
                    value=feat.constraint_extraction_enabled,
                    default_value=True,
                    is_editable=True,
                    source=_config_source("FEATURES__CONSTRAINT_EXTRACTION_ENABLED"),
                    description="Extract cognitive constraints (goals, values, policies, states, causal rules) at write time. Populates ReadResponse.constraints on decision queries.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__CONSTRAINT_EXTRACTION_ENABLED",
                    options=["true", "false"],
                ),
                ConfigItem(
                    key="features.batch_embeddings_enabled",
                    value=feat.batch_embeddings_enabled,
                    default_value=True,
                    is_editable=True,
                    source=_config_source("FEATURES__BATCH_EMBEDDINGS_ENABLED"),
                    description="Batch all embed() calls within a single turn into one embed_batch() call.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__BATCH_EMBEDDINGS_ENABLED",
                    options=["true", "false"],
                ),
                ConfigItem(
                    key="features.cached_embeddings_enabled",
                    value=feat.cached_embeddings_enabled,
                    default_value=True,
                    is_editable=True,
                    source=_config_source("FEATURES__CACHED_EMBEDDINGS_ENABLED"),
                    description="Cache embedding vectors in Redis to avoid re-computing identical text.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__CACHED_EMBEDDINGS_ENABLED",
                    options=["true", "false"],
                ),
                ConfigItem(
                    key="features.store_async",
                    value=feat.store_async,
                    default_value=False,
                    is_editable=True,
                    source=_config_source("FEATURES__STORE_ASYNC"),
                    description="Enqueue writes to Redis for async storage. Reduces write latency. Requires Redis and Celery worker.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__STORE_ASYNC",
                    options=["true", "false"],
                ),
                ConfigItem(
                    key="features.retrieval_timeouts_enabled",
                    value=feat.retrieval_timeouts_enabled,
                    default_value=True,
                    is_editable=True,
                    source=_config_source("FEATURES__RETRIEVAL_TIMEOUTS_ENABLED"),
                    description="Enforce per-step and total asyncio.wait_for timeouts during retrieval.",
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
                    description="Skip remaining retrieval steps when a fact step returns results (avoids redundant lookups).",
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
                    description="Use DB-side aggregation for memory dependency counting in the forgetting scorer.",
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
                    description="Use LRU+TTL bounded state maps in working memory to cap in-process memory growth.",
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
                    description="Dynamically tune pgvector HNSW ef_search at query time based on max_results.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__HNSW_EF_SEARCH_TUNING",
                    options=["true", "false"],
                ),
            ],
        )
    )

    # ── Features — LLM ────────────────────────────────────────────────────────
    sections.append(
        ConfigSection(
            name="Features — LLM",
            items=[
                ConfigItem(
                    key="features.use_llm_enabled",
                    value=feat.use_llm_enabled,
                    default_value=False,
                    is_editable=True,
                    source=_config_source("FEATURES__USE_LLM_ENABLED"),
                    description="Master LLM switch. When false, all internal LLM calls are disabled; the runtime uses modelpack and NER instead. All fine-grained USE_LLM_* flags below are only effective when this is true.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__USE_LLM_ENABLED",
                    options=["true", "false"],
                ),
                ConfigItem(
                    key="features.use_llm_constraint_extractor",
                    value=feat.use_llm_constraint_extractor,
                    default_value=True,
                    is_editable=True,
                    source=_config_source("FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR"),
                    description="Use LLM (unified extractor) for constraint extraction instead of the rule-based path.",
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
                    description="Use LLM for write-time fact extraction instead of the spaCy dependency-parse path.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__USE_LLM_WRITE_TIME_FACTS",
                    options=["true", "false"],
                ),
                ConfigItem(
                    key="features.use_llm_write_gate_importance",
                    value=feat.use_llm_write_gate_importance,
                    default_value=True,
                    is_editable=True,
                    source=_config_source("FEATURES__USE_LLM_WRITE_GATE_IMPORTANCE"),
                    description="Use LLM importance score from the unified extractor instead of the modelpack/rule-based importance.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__USE_LLM_WRITE_GATE_IMPORTANCE",
                    options=["true", "false"],
                ),
                ConfigItem(
                    key="features.use_llm_salience_refinement",
                    value=feat.use_llm_salience_refinement,
                    default_value=True,
                    is_editable=True,
                    source=_config_source("FEATURES__USE_LLM_SALIENCE_REFINEMENT"),
                    description="Use LLM salience score from the unified extractor instead of rule-based salience boosts.",
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
                    description="Merge LLM-detected PII spans from the unified extractor with the regex PII baseline.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__USE_LLM_PII_REDACTION",
                    options=["true", "false"],
                ),
                ConfigItem(
                    key="features.use_llm_memory_type",
                    value=feat.use_llm_memory_type,
                    default_value=True,
                    is_editable=True,
                    source=_config_source("FEATURES__USE_LLM_MEMORY_TYPE"),
                    description="Use LLM memory_type classification from the unified extractor. API-level override still takes precedence.",
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
                    description="Use LLM confidence score from the unified extractor instead of the modelpack 3-bin estimate.",
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
                    description="Use LLM context_tags from the unified extractor when the caller does not provide tags.",
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
                    description="Use LLM decay_rate from the unified extractor instead of the modelpack 5-profile estimate.",
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
                    description="Bypass the modelpack conflict detector entirely and always use LLM for conflict detection.",
                    requires_restart=True,
                    is_required=False,
                    env_var="FEATURES__USE_LLM_CONFLICT_DETECTION_ONLY",
                    options=["true", "false"],
                ),
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
