"""Unit tests for MemoryOrchestrator.forget() timestamp comparison."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.memory.orchestrator import MemoryOrchestrator


@pytest.mark.asyncio
async def test_forget_before_handles_aware_cutoff_against_naive_db_timestamps() -> None:
    """forget(before=aware_dt) must not raise TypeError against naive UTC stored timestamps."""
    old_id = uuid4()
    new_id = uuid4()
    old_rec = SimpleNamespace(id=old_id, tenant_id="t1", timestamp=datetime(2024, 1, 1, 0, 0, 0))
    new_rec = SimpleNamespace(id=new_id, tenant_id="t1", timestamp=datetime(2026, 6, 1, 0, 0, 0))

    store = MagicMock()
    store.scan = AsyncMock(side_effect=[[old_rec, new_rec], []])
    store.delete = AsyncMock()

    orch = MemoryOrchestrator.__new__(MemoryOrchestrator)
    orch.hippocampal = SimpleNamespace(store=store)
    orch.retriever = MagicMock()

    # Pass timezone-aware UTC datetime (and non-UTC offset)
    res = await orch.forget(
        tenant_id="t1",
        before=datetime(2025, 1, 1, 4, 0, 0, tzinfo=timezone(timedelta(hours=4))),
        action="delete",
    )

    assert res["affected_count"] == 1
    store.delete.assert_awaited_once_with(old_id, hard=True)


@pytest.mark.asyncio
async def test_forget_before_handles_naive_cutoff_against_aware_timestamps() -> None:
    """forget(before=naive_dt) must also work when stored records have tz-aware timestamps."""
    old_id = uuid4()
    old_rec = SimpleNamespace(
        id=old_id, tenant_id="t1", timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    )

    store = MagicMock()
    store.scan = AsyncMock(side_effect=[[old_rec], []])
    store.delete = AsyncMock()
    store.update = AsyncMock()

    orch = MemoryOrchestrator.__new__(MemoryOrchestrator)
    orch.hippocampal = SimpleNamespace(store=store)
    orch.retriever = MagicMock()

    res = await orch.forget(
        tenant_id="t1",
        before=datetime(2025, 1, 1, 0, 0, 0),
        action="archive",
    )

    assert res["affected_count"] == 1
    store.update.assert_awaited_once_with(old_id, {"status": "archived"}, increment_version=False)
    store.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_forget_silence_sets_silent_status() -> None:
    """forget(action='silence') marks memories silent rather than deleted."""
    mid = uuid4()
    rec = SimpleNamespace(id=mid, tenant_id="t1", timestamp=datetime(2025, 1, 1))
    store = MagicMock()
    store.get_by_ids_batch = AsyncMock(return_value=[rec])
    store.update = AsyncMock()
    store.delete = AsyncMock()

    orch = MemoryOrchestrator.__new__(MemoryOrchestrator)
    orch.hippocampal = SimpleNamespace(store=store)
    orch.retriever = MagicMock()

    res = await orch.forget(tenant_id="t1", memory_ids=[mid], action="silence")
    assert res["affected_count"] == 1
    store.update.assert_awaited_once_with(mid, {"status": "silent"}, increment_version=False)
    store.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_forget_query_uses_vector_search_min_similarity() -> None:
    """forget(query=...) uses direct vector_search with min_similarity instead of wiping unrelated memories."""
    match_id = uuid4()
    matched_rec = SimpleNamespace(id=match_id, tenant_id="t1", timestamp=datetime(2025, 1, 1))
    other_tenant_rec = SimpleNamespace(
        id=uuid4(), tenant_id="other", timestamp=datetime(2025, 1, 1)
    )

    store = MagicMock()
    store.vector_search = AsyncMock(return_value=[matched_rec, other_tenant_rec])
    store.delete = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=SimpleNamespace(embedding=[0.1] * 8))

    orch = MemoryOrchestrator.__new__(MemoryOrchestrator)
    orch.hippocampal = SimpleNamespace(store=store, embeddings=embeddings)
    orch.retriever = MagicMock()
    orch.retriever.retrieve = AsyncMock()

    res = await orch.forget(tenant_id="t1", query="obsolete fax machine", action="delete")

    assert res["affected_count"] == 1
    store.vector_search.assert_awaited_once()
    assert store.vector_search.await_args.kwargs["min_similarity"] == 0.55
    store.delete.assert_awaited_once_with(match_id, hard=True)
    orch.retriever.retrieve.assert_not_awaited()
