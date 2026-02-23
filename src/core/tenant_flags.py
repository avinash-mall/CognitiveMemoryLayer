"""Per-tenant feature flag overrides (A-02).

Allows per-tenant feature flag customisation via Redis, enabling A/B testing
and gradual rollout without process restarts.

Flags are stored in Redis as a JSON hash at key ``tenant_flags:{tenant_id}``.
Missing keys fall back to the process-wide ``FeatureSettings`` from config.

Usage:
    from src.core.tenant_flags import get_tenant_features

    features = await get_tenant_features(tenant_id, redis_client)
    if features.use_llm_enabled:
        ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_REDIS_KEY_PREFIX = "tenant_flags:"


@dataclass
class TenantFeatureOverrides:
    """Sparse overrides — only non-None fields override the global default."""

    use_llm_enabled: bool | None = None
    constraint_extraction_enabled: bool | None = None
    write_time_facts_enabled: bool | None = None
    use_llm_constraint_extractor: bool | None = None
    use_llm_write_time_facts: bool | None = None
    use_llm_pii_redaction: bool | None = None
    use_llm_salience_refinement: bool | None = None
    use_llm_memory_type: bool | None = None
    use_llm_write_gate_importance: bool | None = None
    use_llm_confidence: bool | None = None
    use_llm_context_tags: bool | None = None
    use_llm_decay_rate: bool | None = None


async def get_tenant_overrides(
    tenant_id: str,
    redis_client: Any | None,
) -> TenantFeatureOverrides | None:
    """Load per-tenant overrides from Redis, or None if unavailable."""
    if redis_client is None:
        return None
    try:
        raw = await redis_client.get(f"{_REDIS_KEY_PREFIX}{tenant_id}")
        if not raw:
            return None
        data = json.loads(raw)
        valid_fields = {f.name for f in fields(TenantFeatureOverrides)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return TenantFeatureOverrides(**filtered)
    except Exception as e:
        logger.debug("tenant_flags_load_failed", tenant_id=tenant_id, error=str(e))
        return None


async def set_tenant_overrides(
    tenant_id: str,
    redis_client: Any,
    overrides: dict[str, bool],
    ttl_seconds: int = 86400,
) -> None:
    """Persist per-tenant feature flag overrides to Redis."""
    valid_fields = {f.name for f in fields(TenantFeatureOverrides)}
    filtered = {k: v for k, v in overrides.items() if k in valid_fields}
    await redis_client.setex(
        f"{_REDIS_KEY_PREFIX}{tenant_id}",
        ttl_seconds,
        json.dumps(filtered),
    )
    logger.info("tenant_flags_set", tenant_id=tenant_id, flags=list(filtered.keys()))


async def delete_tenant_overrides(tenant_id: str, redis_client: Any) -> None:
    """Remove all per-tenant overrides (fall back to global config)."""
    await redis_client.delete(f"{_REDIS_KEY_PREFIX}{tenant_id}")
    logger.info("tenant_flags_deleted", tenant_id=tenant_id)


def apply_overrides(global_features: Any, overrides: TenantFeatureOverrides | None) -> Any:
    """Return feature settings with per-tenant overrides applied.

    Non-mutating: returns the original object when no overrides exist.
    """
    if overrides is None:
        return global_features
    # Shallow copy to avoid mutating the cached global settings
    import copy

    merged = copy.copy(global_features)
    for field in fields(overrides):
        val = getattr(overrides, field.name)
        if val is not None and hasattr(merged, field.name):
            object.__setattr__(merged, field.name, val)
    return merged
