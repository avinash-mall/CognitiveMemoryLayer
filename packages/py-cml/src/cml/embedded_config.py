"""Configuration for embedded CML mode."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EmbeddedDatabaseConfig(BaseModel):
    """Database configuration for embedded mode."""

    database_url: str = Field(
        default="sqlite+aiosqlite:///cml_memory.db",
        description="Database URL. Use sqlite+aiosqlite:// for lite mode.",
        alias="postgres_url",  # backward compat with old name
    )
    neo4j_url: str | None = Field(default=None, description="Neo4j bolt URL")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    redis_url: str | None = Field(default=None, description="Redis URL")


class EmbeddedEmbeddingConfig(BaseModel):
    """Embedding configuration for embedded mode."""

    provider: Literal["openai", "local", "vllm"] = Field(default="local")
    model: str = Field(default="all-MiniLM-L6-v2")
    dimensions: int = Field(default=384)
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)


class EmbeddedLLMConfig(BaseModel):
    """LLM configuration for embedded mode."""

    provider: Literal["openai", "vllm", "ollama", "gemini", "claude"] = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)


class EmbeddedConfig(BaseModel):
    """Full configuration for embedded CognitiveMemoryLayer."""

    storage_mode: Literal["lite", "standard", "full"] = Field(
        default="lite",
        description="Storage backend complexity level",
    )
    tenant_id: str = Field(default="default")
    database: EmbeddedDatabaseConfig = Field(default_factory=EmbeddedDatabaseConfig)
    embedding: EmbeddedEmbeddingConfig = Field(default_factory=EmbeddedEmbeddingConfig)
    llm: EmbeddedLLMConfig = Field(default_factory=EmbeddedLLMConfig)
    auto_consolidate: bool = Field(
        default=False,
        description="Automatically run consolidation periodically",
    )
    auto_forget: bool = Field(
        default=False,
        description="Automatically run active forgetting periodically",
    )
