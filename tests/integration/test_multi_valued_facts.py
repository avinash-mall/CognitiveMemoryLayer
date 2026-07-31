"""A multi-valued fact schema must accumulate, not supersede.

``_update_fact`` accepted a ``schema`` argument and never referenced it, so the one
schema declared ``multi_valued=True`` — ``user:preference:cuisine`` — behaved like every
other key: saying "I like Thai" after "I like Italian" superseded Italian rather than
adding to it, and the earlier preference was silently lost.
"""

from uuid import uuid4

import pytest

from src.memory.neocortical.fact_store import SemanticFactStore


@pytest.mark.asyncio
async def test_multi_valued_key_accumulates_values(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    key = "user:preference:cuisine"

    await store.upsert_fact(tenant_id, key, "Italian", confidence=0.8)
    await store.upsert_fact(tenant_id, key, "Thai", confidence=0.8)

    fact = await store.get_fact(tenant_id, key)
    assert fact is not None
    assert isinstance(fact.value, list)
    assert set(fact.value) == {"Italian", "Thai"}, "the earlier preference must survive"


@pytest.mark.asyncio
async def test_repeating_a_known_value_reinforces_instead_of_versioning(pg_session_factory):
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    key = "user:preference:cuisine"

    await store.upsert_fact(tenant_id, key, "Italian", confidence=0.8)
    first = await store.get_fact(tenant_id, key)
    await store.upsert_fact(tenant_id, key, "Italian", confidence=0.8)
    second = await store.get_fact(tenant_id, key)

    assert first is not None and second is not None
    assert second.version == first.version, "no new version for an already-known value"
    assert second.evidence_count > first.evidence_count, "repetition should reinforce"


@pytest.mark.asyncio
async def test_single_valued_key_still_supersedes(pg_session_factory):
    """The accumulate behaviour must be confined to schemas that ask for it."""
    store = SemanticFactStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    key = "user:location:current_city"

    await store.upsert_fact(tenant_id, key, "Chennai", confidence=0.8)
    await store.upsert_fact(tenant_id, key, "Berlin", confidence=0.9)

    fact = await store.get_fact(tenant_id, key)
    assert fact is not None
    assert fact.value == "Berlin"
