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

    orch = MemoryOrchestrator.__new__(MemoryOrchestrator)
    orch.hippocampal = SimpleNamespace(store=store)
    orch.retriever = MagicMock()

    res = await orch.forget(
        tenant_id="t1",
        before=datetime(2025, 1, 1, 0, 0, 0),
        action="archive",
    )

    assert res["affected_count"] == 1
    store.delete.assert_awaited_once_with(old_id, hard=False)
