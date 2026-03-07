from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.core.tenant_flags import (
    TenantFeatureOverrides,
    apply_overrides,
    delete_tenant_overrides,
    get_tenant_overrides,
    set_tenant_overrides,
)


class _FakeRedis:
    def __init__(self, payload: dict[str, str] | None = None) -> None:
        self.payload = payload or {}
        self.setex_calls: list[tuple[str, int, str]] = []
        self.deleted: list[str] = []

    async def get(self, key: str):
        return self.payload.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.setex_calls.append((key, ttl, value))
        self.payload[key] = value

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.payload.pop(key, None)


@dataclass
class _FeatureSettings:
    use_llm_enabled: bool = False
    use_llm_constraint_extractor: bool = False
    write_time_facts_enabled: bool = True
    untouched: str = "keep"


@pytest.mark.asyncio
async def test_get_tenant_overrides_returns_none_without_redis() -> None:
    assert await get_tenant_overrides("tenant-a", None) is None


@pytest.mark.asyncio
async def test_get_tenant_overrides_returns_none_for_missing_key() -> None:
    redis = _FakeRedis()
    assert await get_tenant_overrides("tenant-a", redis) is None


@pytest.mark.asyncio
async def test_get_tenant_overrides_filters_unknown_fields() -> None:
    redis = _FakeRedis(
        {
            "tenant_flags:tenant-a": json.dumps(
                {
                    "use_llm_enabled": True,
                    "use_llm_constraint_extractor": True,
                    "unknown_flag": "ignored",
                }
            )
        }
    )

    overrides = await get_tenant_overrides("tenant-a", redis)

    assert overrides == TenantFeatureOverrides(
        use_llm_enabled=True,
        use_llm_constraint_extractor=True,
    )


@pytest.mark.asyncio
async def test_get_tenant_overrides_returns_none_on_invalid_json() -> None:
    redis = _FakeRedis({"tenant_flags:tenant-a": "{not-json"})
    assert await get_tenant_overrides("tenant-a", redis) is None


@pytest.mark.asyncio
async def test_set_tenant_overrides_filters_unknown_fields() -> None:
    redis = _FakeRedis()

    await set_tenant_overrides(
        "tenant-a",
        redis,
        {"use_llm_enabled": True, "write_time_facts_enabled": False, "bad": True},
        ttl_seconds=123,
    )

    key, ttl, raw = redis.setex_calls[0]
    assert key == "tenant_flags:tenant-a"
    assert ttl == 123
    assert json.loads(raw) == {
        "use_llm_enabled": True,
        "write_time_facts_enabled": False,
    }


@pytest.mark.asyncio
async def test_delete_tenant_overrides_removes_key() -> None:
    redis = _FakeRedis({"tenant_flags:tenant-a": "{}"})

    await delete_tenant_overrides("tenant-a", redis)

    assert redis.deleted == ["tenant_flags:tenant-a"]
    assert "tenant_flags:tenant-a" not in redis.payload


def test_apply_overrides_returns_same_object_when_none() -> None:
    features = _FeatureSettings()
    assert apply_overrides(features, None) is features


def test_apply_overrides_returns_shallow_copy_without_mutating_original() -> None:
    features = _FeatureSettings()
    overrides = TenantFeatureOverrides(
        use_llm_enabled=True,
        use_llm_constraint_extractor=True,
    )

    merged = apply_overrides(features, overrides)

    assert merged is not features
    assert merged.use_llm_enabled is True
    assert merged.use_llm_constraint_extractor is True
    assert merged.write_time_facts_enabled is True
    assert merged.untouched == "keep"
    assert features.use_llm_enabled is False
    assert features.use_llm_constraint_extractor is False
