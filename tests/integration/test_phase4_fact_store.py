"""Integration tests for Phase 4: SemanticFactStore with PostgreSQL."""
from uuid import uuid4

import pytest

from src.memory.neocortical.fact_store import SemanticFactStore
from src.memory.neocortical.schemas import FactCategory


@pytest.mark.asyncio
async def test_upsert_and_get_fact(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    fact = await store.upsert_fact(
        tenant_id, user_id, "user:identity:name", "Alice", confidence=0.9
    )
    assert fact.id
    assert fact.value == "Alice"
    assert fact.key == "user:identity:name"
    assert fact.category == FactCategory.IDENTITY

    got = await store.get_fact(tenant_id, user_id, "user:identity:name")
    assert got is not None
    assert got.value == "Alice"


@pytest.mark.asyncio
async def test_get_facts_by_category(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    await store.upsert_fact(tenant_id, user_id, "user:preference:cuisine", "Italian")
    await store.upsert_fact(tenant_id, user_id, "user:preference:theme", "dark")

    facts = await store.get_facts_by_category(tenant_id, user_id, FactCategory.PREFERENCE)
    assert len(facts) >= 2
    predicates = {f.predicate for f in facts}
    assert "cuisine" in predicates or "theme" in predicates


@pytest.mark.asyncio
async def test_get_user_profile(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    await store.upsert_fact(tenant_id, user_id, "user:identity:name", "Bob")
    await store.upsert_fact(tenant_id, user_id, "user:location:current_city", "Paris")

    profile = await store.get_user_profile(tenant_id, user_id)
    assert "identity" in profile or "location" in profile


@pytest.mark.asyncio
async def test_search_facts(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    await store.upsert_fact(tenant_id, user_id, "user:identity:name", "Charlie")

    results = await store.search_facts(tenant_id, user_id, "Charlie", limit=5)
    assert len(results) >= 1
    assert any(f.value == "Charlie" for f in results)
