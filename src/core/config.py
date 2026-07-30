"""Configuration management with pydantic-settings."""

import os
import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from pydantic_settings import BaseSettings


def _auto_cpu_count() -> int:
    """Return usable CPU count, capped at physical cores."""
    return os.cpu_count() or 4


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


# Embedding config: EMBEDDING_INTERNAL__* (internal memory tasks).
# Defaults (when unset in .env): provider=local, model=nomic-embed-text-v2-moe, dimensions=768.
class EmbeddingInternalSettings(PydanticBaseModel):
    """Embedding provider for internal memory tasks. Defaults: local, nomic-embed-text-v2-moe, 768d."""

    provider: str | None = Field(
        default="local"
    )  # openai | local | openai_compatible | ollama | vllm | mock
    model: str | None = Field(default="nomic-ai/nomic-embed-text-v2-moe")
    dimensions: int | None = Field(default=768)
    local_model: str | None = Field(default="nomic-ai/nomic-embed-text-v2-moe")
    revision: str | None = Field(default="1066b6599d099fbb93dfcb64f9c37a7c9e503e85")
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)
    local_batch_size: int = Field(
        default=0,
        description="Batch size for local sentence-transformers encode(). "
        "0 = auto (64 on CUDA, 8 on CPU). Increase for large GPUs; lower to avoid OOM.",
    )
    device: Literal["auto", "cpu", "cuda"] = Field(default="auto")
    batch_wait_ms: float = Field(
        default=10.0
    )  # Cross-request coalescing window in ms; 0 = disabled


# LLM config: LLM_INTERNAL__* (internal tasks) and LLM_EVAL__* (evaluation QA/judge).
# Supported providers: openai, ollama, anthropic, gemini, vllm, sglang, openai_compatible.
class LLMInternalSettings(PydanticBaseModel):
    """LLM for internal tasks (extraction, consolidation, retrieval). Env: LLM_INTERNAL__*."""

    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    base_url: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    extra_body: dict | None = Field(
        default=None,
        description='Extra JSON merged into chat completion requests, e.g. {"chat_template_kwargs": {"enable_thinking": false}} for vLLM reasoning models without a reasoning parser.',
    )


class LLMEvalSettings(PydanticBaseModel):
    """LLM for evaluation (QA, judge). Env: LLM_EVAL__*. If unset, falls back to LLM_INTERNAL__*."""

    provider: str | None = Field(default=None)
    model: str | None = Field(default=None)
    base_url: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    extra_body: dict | None = Field(default=None)


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
        default=0,
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
    # Master LLM switch (default false): when false, internal LLM calls are disabled
    # and the runtime degrades to heuristics (regex PII, Jaccard novelty, no extraction).
    use_llm_enabled: bool = Field(
        default=False,
        description="Master switch: when true, the internal LLM drives extraction, "
        "classification, and enrichment; when false, heuristic-only mode.",
    )
    # --- Improvement Report features (LoCoMo-Plus / Kumiho-inspired) ---
    prospective_indexing_enabled: bool = Field(
        default=True,
        description="Generate forward-looking implications at write time (Kumiho-inspired). "
        "Bridges cue-trigger semantic disconnect for cognitive memory queries.",
    )
    prospective_index_count: int = Field(
        default=4,
        description="Number of prospective implications to generate per memory.",
    )
    bm25_retrieval_enabled: bool = Field(
        default=True,
        description="Add BM25 sparse retrieval as a second pathway with RRF fusion.",
    )
    hyde_retrieval_enabled: bool = Field(
        default=True,
        description="Use Hypothetical Document Embedding (HyDE) for cognitive memory queries.",
    )
    temporal_resolution_enabled: bool = Field(
        default=True,
        description="Resolve relative time references to absolute dates at extraction time.",
    )
    adversarial_verification_enabled: bool = Field(
        default=True,
        description="Add verification step to detect unanswerable questions.",
    )
    answer_compression_enabled: bool = Field(
        default=True,
        description="Post-process LLM answers to extract short factual core for F1 scoring.",
    )
    cross_encoder_reranking_enabled: bool = Field(
        default=True,
        description="Use cross-encoder model for reranking. "
        "Auto-downloads BAAI/bge-reranker-v2-m3 (~568MB) on first use.",
    )
    cross_encoder_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for reranking.",
    )
    pii_redaction_enabled: bool = Field(
        default=True,
        description="Master switch for PII redaction (regex + model). "
        "Set to false for benchmark evaluation where entity names are critical for scoring.",
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
        default=5,
        description="Max episodes to show when constraints exist (reduces dilution)",
    )
    max_episodes_default: int = Field(default=8, description="Max episodes when no constraints")
    max_constraint_tokens: int = Field(
        default=400,
        description="Token budget reserved for Active Constraints (ensures constraints are not truncated)",
    )
    default_step_timeout_ms: int = Field(default=800, description="Per-step timeout (ms)")
    total_timeout_ms: int = Field(default=5000, description="Total retrieval budget (ms)")
    graph_timeout_ms: int = Field(default=2000, description="Graph step timeout (ms)")
    fact_timeout_ms: int = Field(default=1000, description="Fact lookup timeout (ms)")
    hnsw_ef_search: int = Field(default=64, description="pgvector HNSW ef_search override")
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)


class PerformanceSettings(PydanticBaseModel):
    """Write-path thread pool and batching tuning knobs. Env: PERFORMANCE__*.

    All values default to 0 (auto-detect), which selects safe defaults based
    on the host's CPU count.  Override in .env for specific hardware.
    """

    gate_executor_workers: int = Field(
        default=0,
        description="Thread-pool size for write-gate novelty checks. "
        "0 = auto (min(cpu_count, 8)). Lower on small machines to reduce GIL contention.",
    )
    def resolved_gate_workers(self) -> int:
        if self.gate_executor_workers > 0:
            return self.gate_executor_workers
        return min(_auto_cpu_count(), 8)


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
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)

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


def get_embedding_dimensions() -> int:
    """Return configured embedding dimension (from EMBEDDING_INTERNAL__DIMENSIONS), or 768.
    Safe when embedding_internal is missing (e.g. older settings or env quirks)."""
    s = get_settings()
    ei = getattr(s, "embedding_internal", None)
    return (getattr(ei, "dimensions", None) if ei is not None else None) or 768


def validate_embedding_dimensions(settings: Settings | None = None) -> None:
    """Validate that the configured embedding dimension matches the DB schema.

    Call this at application startup (e.g. in the lifespan handler) to
    catch mismatches between ``EMBEDDING_INTERNAL__DIMENSIONS`` and the ``Vector(N)``
    column defined in ``MemoryRecordModel`` (MED-04).

    Raises ``ValueError`` if the dimensions disagree.
    """
    configured = get_embedding_dimensions()
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
