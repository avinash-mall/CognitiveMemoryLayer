"""Configuration management with pydantic-settings."""

import re
from functools import lru_cache

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


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    postgres_url: str = Field(default="postgresql+asyncpg://localhost/memory")
    neo4j_url: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")
    redis_url: str = Field(default="redis://localhost:6379")


class EmbeddingSettings(BaseSettings):
    """Embedding provider settings."""

    provider: str = Field(default="openai")  # openai | local
    model: str = Field(default="text-embedding-3-small")
    dimensions: int = Field(default=1536)
    local_model: str = Field(default="all-MiniLM-L6-v2")
    api_key: str | None = Field(default=None)  # OpenAI; can use OPENAI_API_KEY env


class LLMSettings(BaseSettings):
    """LLM provider settings."""

    provider: str = Field(default="openai")  # openai | vllm
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0)
    api_key: str | None = Field(default=None)  # OpenAI API key; can use OPENAI_API_KEY env
    # vLLM (OpenAI-compatible) for local/compression
    vllm_base_url: str | None = Field(default=None)  # e.g. http://vllm:8000/v1
    vllm_model: str = Field(default="meta-llama/Llama-3.2-1B-Instruct")


class MemorySettings(BaseSettings):
    """Memory system tuning parameters."""

    sensory_buffer_max_tokens: int = Field(default=500)
    sensory_buffer_decay_seconds: float = Field(default=30.0)
    working_memory_max_chunks: int = Field(default=10)
    write_gate_threshold: float = Field(default=0.3)
    consolidation_interval_hours: int = Field(default=6)
    forgetting_interval_hours: int = Field(default=24)


class AuthSettings(BaseSettings):
    """
    API authentication (keys from environment).
    Env vars (with env_nested_delimiter='__'): AUTH__API_KEY, AUTH__ADMIN_API_KEY, AUTH__DEFAULT_TENANT_ID.
    """

    api_key: str | None = Field(default=None)
    admin_api_key: str | None = Field(default=None)
    default_tenant_id: str = Field(default="default")


class Settings(BaseSettings):
    """Application settings with nested configuration."""

    app_name: str = Field(default="CognitiveMemoryLayer")
    debug: bool = Field(default=False)

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)

    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
