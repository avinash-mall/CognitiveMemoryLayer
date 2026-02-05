"""Redis client for cache and session storage."""
from typing import Any

from ..core.config import get_settings


def get_redis_client() -> Any:
    """Return an async Redis client from application settings."""
    import redis.asyncio as redis

    settings = get_settings()
    return redis.from_url(settings.database.redis_url)
