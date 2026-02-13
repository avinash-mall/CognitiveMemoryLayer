"""Configuration for embedded CML mode. All URLs and model names come from env when not set."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, model_validator

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(key) or "").strip()


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


def _env_int(key: str, default: int = 0) -> int:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class EmbeddedEmbeddingConfig(BaseModel):
    """Embedding configuration for embedded mode. Set EMBEDDING__MODEL, EMBEDDING__BASE_URL, EMBEDDING__DIMENSIONS in .env."""

    provider: Literal["openai", "local", "openai_compatible", "vllm"] = Field(default="local")
    model: str = Field(default="", description="Set EMBEDDING__MODEL in .env")
    dimensions: int = Field(default=384, description="Set EMBEDDING__DIMENSIONS in .env")
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def from_env(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if (not data.get("model") or data.get("model") == "") and _env("EMBEDDING__MODEL"):
            data = {**data, "model": _env("EMBEDDING__MODEL")}
        if data.get("base_url") is None and _env("EMBEDDING__BASE_URL"):
            data = {**data, "base_url": _env("EMBEDDING__BASE_URL")}
        dims = _env_int("EMBEDDING__DIMENSIONS")
        if dims > 0:
            data = {**data, "dimensions": dims}
        return data


class EmbeddedLLMConfig(BaseModel):
    """LLM configuration for embedded mode. Set LLM__MODEL, LLM__BASE_URL in .env."""

    provider: Literal["openai", "openai_compatible", "vllm", "ollama", "gemini", "claude"] = Field(
        default="openai"
    )
    model: str = Field(default="", description="Set LLM__MODEL in .env")
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def from_env(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if (not data.get("model") or data.get("model") == "") and _env("LLM__MODEL"):
            data = {**data, "model": _env("LLM__MODEL")}
        if data.get("base_url") is None and _env("LLM__BASE_URL"):
            data = {**data, "base_url": _env("LLM__BASE_URL")}
        return data


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
