"""Integration tests for SemanticFactStore with PostgreSQL."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import and_, update

from src.memory.neocortical.fact_store import SemanticFactStore
from src.memory.neocortical.schemas import FactCategory
from src.storage.models import SemanticFactModel


@pytest.mark.asyncio
async def test_upsert_and_get_fact(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    fact = await store.upsert_fact(tenant_id, "user:identity:name", "Alice", confidence=0.9)
    assert fact.id
    assert fact.value == "Alice"
    assert fact.key == "user:identity:name"
    assert fact.category == FactCategory.IDENTITY

    got = await store.get_fact(tenant_id, "user:identity:name")
    assert got is not None
    assert got.value == "Alice"


@pytest.mark.asyncio
async def test_get_facts_by_category(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    await store.upsert_fact(tenant_id, "user:preference:cuisine", "Italian")
    await store.upsert_fact(tenant_id, "user:preference:theme", "dark")

    facts = await store.get_facts_by_category(tenant_id, FactCategory.PREFERENCE)
    assert len(facts) >= 2
    predicates = {f.predicate for f in facts}
    assert "cuisine" in predicates or "theme" in predicates


@pytest.mark.asyncio
async def test_get_user_profile(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    await store.upsert_fact(tenant_id, "user:identity:name", "Bob")
    await store.upsert_fact(tenant_id, "user:location:current_city", "Paris")

    profile = await store.get_tenant_profile(tenant_id)
    assert "identity" in profile or "location" in profile


@pytest.mark.asyncio
async def test_search_facts(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    await store.upsert_fact(tenant_id, "user:identity:name", "Charlie")

    results = await store.search_facts(tenant_id, "Charlie", limit=5)
    assert len(results) >= 1
    assert any(f.value == "Charlie" for f in results)


@pytest.mark.asyncio
async def test_get_facts_by_category_respects_limit(pg_session_factory):
    """get_facts_by_category returns at most limit facts."""
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    for i in range(60):
        await store.upsert_fact(
            tenant_id, f"user:preference:item_{i}", f"value_{i}", confidence=0.8
        )

    facts = await store.get_facts_by_category(tenant_id, FactCategory.PREFERENCE, limit=10)
    assert len(facts) == 10


@pytest.mark.asyncio
async def test_get_facts_by_category_excludes_expired(pg_session_factory):
    """get_facts_by_category excludes facts with valid_to in the past."""
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    await store.upsert_fact(
        tenant_id, "user:policy:expired_constraint", "I used to avoid X", confidence=0.9
    )

    # Directly set valid_to to the past while keeping is_current=True (simulates
    # temporal expiry without supersession)
    now = datetime.now(UTC).replace(tzinfo=None)
    past = now - timedelta(days=1)
    async with pg_session_factory() as session:
        await session.execute(
            update(SemanticFactModel)
            .where(
                and_(
                    SemanticFactModel.tenant_id == tenant_id,
                    SemanticFactModel.key == "user:policy:expired_constraint",
                )
            )
            .values(valid_to=past)
        )
        await session.commit()

    facts = await store.get_facts_by_category(tenant_id, FactCategory.POLICY, current_only=True)
    predicate_values = {f.predicate: f.value for f in facts}
    assert "expired_constraint" not in predicate_values
