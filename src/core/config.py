"""Configuration management with pydantic-settings."""

import re
from functools import lru_cache

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic_settings import BaseSettings


def ensure_asyncpg_url(url: str) -> str:
    """Normalise a PostgreSQL URL to always use the asyncpg driver.

    Handles ``postgresql://``, ``postgresql+psycopg2://``, ``postgresql+psycopg://``,
    and any other ``postgresql+<driver>://`` variant, converting them all to
    ``postgresql+asyncpg://``.  URLs that already contain ``+asyncpg`` are returned
    unchanged.
    """
    if "+asyncpg" in url:
        return url
    return re.sub(r"^postgresql(\+\w+)?://", "postgresql+asyncpg://", url)


class DatabaseSettings(PydanticBaseModel):
    """Database connection settings (nested; read via Settings env prefix)."""

    postgres_url: str = Field(default="postgresql+asyncpg://memory:memory@localhost/memory")
    neo4j_url: str = Field(default="bolt://localhost:7687")
    neo4j_browser_url: str | None = Field(
        default=None,
        description="Neo4j bolt URL for browser (neovis.js). When set, used by dashboard; else neo4j_url.",
    )
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    redis_url: str = Field(default="redis://localhost:6379")


# Embedding config: EMBEDDING_INTERNAL__* (internal memory tasks). When unset, defaults to
# nomic-ai/nomic-embed-text-v2-moe (768 dims, 512 max seq) via LocalEmbeddings.
class EmbeddingInternalSettings(PydanticBaseModel):
    """Embedding provider for internal memory tasks. When unset, uses local nomic-embed-text-v2-moe."""

    provider: str | None = Field(default=None)  # openai | local | openai_compatible | ollama
    model: str | None = Field(default=None)
    dimensions: int | None = Field(default=None)
    local_model: str | None = Field(
        default=None
    )  # for provider=local; default nomic-embed-text-v2-moe
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)


# LLM config: LLM_INTERNAL__* (internal tasks) and LLM_EVAL__* (evaluation QA/judge).
# Supported providers: openai, ollama, anthropic, gemini, vllm, sglang, openai_compatible.
class LLMInternalSettings(PydanticBaseModel):
    """LLM for internal tasks (extraction, consolidation, retrieval). Env: LLM_INTERNAL__*."""

    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    base_url: str | None = Field(default=None)
    api_key: str | None = Field(default=None)


class LLMEvalSettings(PydanticBaseModel):
    """LLM for evaluation (QA, judge). Env: LLM_EVAL__*. If unset, falls back to LLM_INTERNAL__*."""

    provider: str | None = Field(default=None)
    model: str | None = Field(default=None)
    base_url: str | None = Field(default=None)
    api_key: str | None = Field(default=None)


class ChunkerSettings(PydanticBaseModel):
    """Chunker configuration (semchunk; Hugging Face tokenizer)."""

    tokenizer: str = Field(
        default="google/flan-t5-base",
        description="Hugging Face tokenizer model ID",
    )
    chunk_size: int = Field(
        default=500,
        description="Max tokens per chunk (align with embedding model max input)",
    )
    overlap_percent: float = Field(
        default=0.15,
        description="Overlap ratio 0-1 (e.g. 0.15 = 15%)",
    )


class AuthSettings(PydanticBaseModel):
    """
    API authentication (keys from environment).
    Env vars (with env_nested_delimiter='__'): AUTH__API_KEY, AUTH__ADMIN_API_KEY, AUTH__DEFAULT_TENANT_ID, AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE.
    """

    api_key: str | None = Field(default=None)
    admin_api_key: str | None = Field(default=None)
    default_tenant_id: str = Field(default="default")
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Rate limit per tenant (0 = disable). Use higher value for bulk eval (e.g. 600).",
    )


class FeatureFlags(PydanticBaseModel):
    """Feature flags for gradual rollout of improvements.

    All flags default to *True* for new deployments.  Set to *False* to
    revert individual features without a full rollback.
    """

    stable_keys_enabled: bool = Field(
        default=True, description="Phase 1.1-1.2: SHA256-based stable keys"
    )
    write_time_facts_enabled: bool = Field(
        default=True, description="Phase 1.3: populate semantic store at write time"
    )
    batch_embeddings_enabled: bool = Field(
        default=True, description="Phase 2.1: batch embed_batch() calls"
    )
    store_async: bool = Field(
        default=False, description="Phase 2.2: async storage pipeline (opt-in)"
    )
    cached_embeddings_enabled: bool = Field(
        default=True, description="Phase 2.3: Redis embedding cache"
    )
    retrieval_timeouts_enabled: bool = Field(
        default=True, description="Phase 3.1: per-step asyncio.wait_for"
    )
    skip_if_found_cross_group: bool = Field(
        default=True, description="Phase 3.2: cross-group skip on fact hit"
    )
    db_dependency_counts: bool = Field(
        default=True, description="Phase 4.1: DB-side aggregation for forgetting"
    )
    bounded_state_enabled: bool = Field(default=True, description="Phase 5.2: LRU+TTL state maps")
    hnsw_ef_search_tuning: bool = Field(
        default=True, description="Phase 6.1: query-time HNSW tuning"
    )
    constraint_extraction_enabled: bool = Field(
        default=True, description="Cognitive: extract and store latent constraints at write time"
    )
    # Master LLM switch (default false): when false, all internal LLM is disabled; heuristics used.
    # Quality degrades when false; enabling improves quality but lowers performance (LLM latency).
    use_llm_enabled: bool = Field(
        default=False,
        description="Master switch: when false, all internal LLM disabled (heuristics used). "
        "Quality degrades when false; enabling improves quality but lowers performance.",
    )
    # Fine-grained LLM flags (only effective when use_llm_enabled=true)
    use_llm_constraint_extractor: bool = Field(
        default=True,
        description="Use LLM (via unified extractor) for constraint extraction instead of rule-based",
    )
    use_llm_write_time_facts: bool = Field(
        default=True,
        description="Use LLM (via unified extractor) for write-time fact extraction instead of rule-based",
    )
    use_llm_query_classifier_only: bool = Field(
        default=False,
        description="Skip fast pattern path, always use LLM for query classification",
    )
    use_llm_salience_refinement: bool = Field(
        default=True,
        description="Use LLM salience from unified extractor instead of rule-based boosts",
    )
    use_llm_pii_redaction: bool = Field(
        default=True,
        description="Use LLM PII spans from unified extractor, merged with regex redaction",
    )
    use_llm_write_gate_importance: bool = Field(
        default=True,
        description="Use LLM importance from unified extractor instead of rule-based _compute_importance",
    )
    use_llm_memory_type: bool = Field(
        default=True,
        description="Use LLM memory_type from unified extractor when present and valid",
    )
    use_llm_confidence: bool = Field(
        default=True,
        description="Use LLM confidence from unified extractor when present",
    )
    use_llm_context_tags: bool = Field(
        default=True,
        description="Use LLM context_tags from unified extractor when caller does not provide tags",
    )
    use_llm_decay_rate: bool = Field(
        default=True,
        description="Use LLM decay_rate from unified extractor when valid",
    )
    use_llm_conflict_detection_only: bool = Field(
        default=False,
        description="Skip fast pattern path, always use LLM for conflict detection",
    )
    use_llm_constraint_reranker: bool = Field(
        default=False,
        description="Use LLM to score constraint relevance during reranking (1 call per read); "
        "when False, falls back to fast text-similarity scoring",
    )


class RerankerSettings(PydanticBaseModel):
    """Reranker weights (tune to reduce recency bias)."""

    recency_weight: float = Field(default=0.1, description="Weight for recency in reranking")
    relevance_weight: float = Field(default=0.5, description="Weight for relevance score")
    confidence_weight: float = Field(default=0.2, description="Weight for confidence")
    active_constraint_bonus: float = Field(
        default=0.2,
        description="Optional bonus for active constraints (for future use in reranker)",
    )


class RetrievalSettings(PydanticBaseModel):
    """Retrieval tuning knobs."""

    episode_relevance_threshold: float = Field(
        default=0.5,
        description="Min relevance for episodes in context (avoid diluting constraints)",
    )
    max_episodes_when_constraints: int = Field(
        default=3,
        description="Max episodes to show when constraints exist (reduces dilution)",
    )
    max_episodes_default: int = Field(default=5, description="Max episodes when no constraints")
    max_constraint_tokens: int = Field(
        default=400,
        description="Token budget reserved for Active Constraints (ensures constraints are not truncated)",
    )
    default_step_timeout_ms: int = Field(default=500, description="Per-step timeout (ms)")
    total_timeout_ms: int = Field(default=2000, description="Total retrieval budget (ms)")
    graph_timeout_ms: int = Field(default=1000, description="Graph step timeout (ms)")
    fact_timeout_ms: int = Field(default=500, description="Fact lookup timeout (ms)")
    hnsw_ef_search: int = Field(default=64, description="pgvector HNSW ef_search override")
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)


class Settings(BaseSettings):
    """Application settings with nested configuration."""

    app_name: str = Field(default="CognitiveMemoryLayer")
    debug: bool = Field(default=False)
    cors_origins: list[str] | None = Field(
        default=None
    )  # None = use default; ["*"] disables credentials

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding_internal: EmbeddingInternalSettings = Field(default_factory=EmbeddingInternalSettings)
    llm_internal: LLMInternalSettings = Field(default_factory=LLMInternalSettings)
    llm_eval: LLMEvalSettings = Field(default_factory=LLMEvalSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    chunker: ChunkerSettings = Field(default_factory=ChunkerSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "extra": "ignore",  # allow .env to contain CML_* and other vars for tests/examples
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings.

    The result is cached via ``@lru_cache`` for the process lifetime.
    **Testing note (LOW-01):** call ``get_settings.cache_clear()`` after
    overriding environment variables via ``monkeypatch`` to pick up new
    values. An autouse fixture in ``tests/conftest.py`` does this
    automatically after each test.
    """
    return Settings()


def validate_embedding_dimensions(settings: Settings | None = None) -> None:
    """Validate that the configured embedding dimension matches the DB schema.

    Call this at application startup (e.g. in the lifespan handler) to
    catch mismatches between ``EMBEDDING_INTERNAL__DIMENSIONS`` and the ``Vector(N)``
    column defined in ``MemoryRecordModel`` (MED-04).

    Raises ``ValueError`` if the dimensions disagree.
    """
    settings = settings or get_settings()
    ei = settings.embedding_internal
    configured = (
        ei.dimensions if ei.dimensions is not None else 768  # default nomic-embed-text-v2-moe
    )
    try:
        from ..storage.models import MemoryRecordModel

        col = MemoryRecordModel.__table__.columns["embedding"]
        db_dim = getattr(col.type, "dim", None)
        if db_dim is not None and configured != db_dim:
            raise ValueError(
                f"Configured embedding dimensions ({configured}) do not match the "
                f"database Vector column dimension ({db_dim}). Update "
                f"EMBEDDING_INTERNAL__DIMENSIONS or create a new migration."
            )
    except ImportError:
        pass  # storage.models not available (e.g. during setup)
