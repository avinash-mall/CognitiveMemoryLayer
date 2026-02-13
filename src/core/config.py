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
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    redis_url: str = Field(default="redis://localhost:6379")


class EmbeddingSettings(PydanticBaseModel):
    """Embedding provider settings."""

    provider: str = Field(default="openai")  # openai | local | openai_compatible | ollama
    model: str = Field(default="text-embedding-3-small")
    dimensions: int = Field(default=1536)
    local_model: str = Field(default="all-MiniLM-L6-v2")
    api_key: str | None = Field(default=None)  # OpenAI; can use OPENAI_API_KEY env
    base_url: str | None = Field(default=None)  # Optional; OpenAI-compatible embedding endpoint


class LLMSettings(PydanticBaseModel):
    """LLM provider settings."""

    provider: str = Field(default="openai")  # openai | openai_compatible | ollama | gemini | claude
    model: str = Field(default="gpt-4o-mini")
    api_key: str | None = Field(default=None)  # API key; can use OPENAI_API_KEY env
    base_url: str | None = Field(
        default=None
    )  # OpenAI-compatible endpoint; for openai_compatible/ollama or proxy


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


class Settings(BaseSettings):
    """Application settings with nested configuration."""

    app_name: str = Field(default="CognitiveMemoryLayer")
    debug: bool = Field(default=False)
    cors_origins: list[str] | None = Field(
        default=None
    )  # None = use default; ["*"] disables credentials

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)

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
    catch mismatches between ``EMBEDDING__DIMENSIONS`` and the ``Vector(N)``
    column defined in ``MemoryRecordModel`` (MED-04).

    Raises ``ValueError`` if the dimensions disagree.
    """
    settings = settings or get_settings()
    configured = settings.embedding.dimensions
    try:
        from ..storage.models import MemoryRecordModel

        col = MemoryRecordModel.__table__.columns["embedding"]
        db_dim = getattr(col.type, "dim", None)
        if db_dim is not None and configured != db_dim:
            raise ValueError(
                f"Configured embedding dimensions ({configured}) do not match the "
                f"database Vector column dimension ({db_dim}). Update "
                f"EMBEDDING__DIMENSIONS or create a new migration."
            )
    except ImportError:
        pass  # storage.models not available (e.g. during setup)
