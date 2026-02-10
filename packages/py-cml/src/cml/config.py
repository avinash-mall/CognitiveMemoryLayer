"""Configuration management for py-cml."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Load .env so CML_* vars are available before Pydantic runs
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class CMLConfig(BaseModel):
    """Configuration for CognitiveMemoryLayer client.

    Parameters can be set directly, via environment variables, or
    via a .env file. Environment variables use the CML_ prefix.

    Env vars:
        CML_API_KEY: API key for authentication
        CML_BASE_URL: Base URL of the CML server
        CML_TENANT_ID: Tenant identifier
        CML_TIMEOUT: Request timeout in seconds
        CML_MAX_RETRIES: Maximum retry attempts
        CML_RETRY_DELAY: Delay between retries in seconds
        CML_ADMIN_API_KEY: Admin API key (for admin operations)
        CML_VERIFY_SSL: Verify SSL certificates (true/false)
    """

    api_key: str | None = Field(default=None, description="API key for authentication")
    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the CML server",
    )
    tenant_id: str = Field(default="default", description="Tenant identifier")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0,
        description="Base delay between retries (seconds)",
    )
    max_retry_delay: float = Field(
        default=60.0,
        description="Maximum delay for backoff (seconds); caps exponential backoff",
    )
    admin_api_key: str | None = Field(default=None, description="Admin API key")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, values: Any) -> Any:
        """Load unset values from environment variables."""
        if not isinstance(values, dict):
            return values
        env_map = {
            "api_key": "CML_API_KEY",
            "base_url": "CML_BASE_URL",
            "tenant_id": "CML_TENANT_ID",
            "timeout": "CML_TIMEOUT",
            "max_retries": "CML_MAX_RETRIES",
            "retry_delay": "CML_RETRY_DELAY",
            "max_retry_delay": "CML_MAX_RETRY_DELAY",
            "admin_api_key": "CML_ADMIN_API_KEY",
            "verify_ssl": "CML_VERIFY_SSL",
        }
        for field, env_var in env_map.items():
            if field not in values or values[field] is None:
                env_val = os.environ.get(env_var)
                if env_val is not None:
                    if field == "verify_ssl":
                        values[field] = env_val.strip().lower() in ("1", "true", "yes")
                    else:
                        values[field] = env_val
        return values

    @model_validator(mode="after")
    def validate_and_normalize(self) -> CMLConfig:
        """Normalize base_url and validate numeric fields."""
        self.base_url = self.base_url.rstrip("/")
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")
        return self
