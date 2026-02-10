"""Unit tests for forgetting executor."""

from datetime import datetime, timezone
from uuid import uuid4

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance
from src.forgetting.actions import ForgettingAction, ForgettingOperation
from src.forgetting.executor import ForgettingExecutor


def _make_record(
    text: str = "test",
    confidence: float = 0.8,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id="t1",
        context_tags=[],
        type=MemoryType.EPISODIC_EVENT,
        text=text,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
    )


class TestForgettingExecutor:
    """ForgettingExecutor with mocked store."""

    @pytest.mark.asyncio
    async def test_execute_keep_increments_kept_only(self):
        mock_store = MagicMock()
        executor = ForgettingExecutor(store=mock_store)
        rec = _make_record()
        op_keep = ForgettingOperation(action=ForgettingAction.KEEP, memory_id=rec.id)
        result = await executor.execute(operations=[op_keep])
        assert result.operations_planned == 1
        assert result.operations_applied == 0
        assert result.kept == 1
        mock_store.get_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_dry_run_counts_no_writes(self):
        mock_store = MagicMock()
        mock_store.get_by_id = AsyncMock(return_value=None)
        executor = ForgettingExecutor(store=mock_store)
        rec = _make_record()
        op_decay = ForgettingOperation(
            action=ForgettingAction.DECAY,
            memory_id=rec.id,
            new_confidence=0.4,
        )
        result = await executor.execute(operations=[op_decay], dry_run=True)
        assert result.operations_planned == 1
        assert result.operations_applied == 1
        mock_store.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_decay_updates_confidence(self):
        rec = _make_record(confidence=0.7)
        mock_store = MagicMock()
        mock_store.get_by_id = AsyncMock(return_value=rec)
        mock_store.update = AsyncMock(return_value=rec)
        executor = ForgettingExecutor(store=mock_store)
        op = ForgettingOperation(
            action=ForgettingAction.DECAY,
            memory_id=rec.id,
            new_confidence=0.4,
        )
        result = await executor.execute(operations=[op])
        assert result.decayed == 1
        assert result.operations_applied == 1
        mock_store.update.assert_called_once()
        # update(memory_id, patch, ...) -> patch is second positional arg
        patch_arg = mock_store.update.call_args[0][1]
        assert patch_arg["confidence"] == 0.4

    @pytest.mark.asyncio
    async def test_execute_silence_updates_status(self):
        rec = _make_record()
        mock_store = MagicMock()
        mock_store.get_by_id = AsyncMock(return_value=rec)
        mock_store.update = AsyncMock(return_value=rec)
        executor = ForgettingExecutor(store=mock_store)
        op = ForgettingOperation(action=ForgettingAction.SILENCE, memory_id=rec.id)
        result = await executor.execute(operations=[op])
        assert result.silenced == 1
        mock_store.update.assert_called_once()
        patch_arg = mock_store.update.call_args[0][1]
        assert patch_arg["status"] == "silent"

    @pytest.mark.asyncio
    async def test_execute_delete_skips_when_references_exist(self):
        rec = _make_record()
        mock_store = MagicMock()
        mock_store.count_references_to = AsyncMock(return_value=2)
        mock_store.delete = AsyncMock(return_value=True)
        executor = ForgettingExecutor(store=mock_store)
        op = ForgettingOperation(action=ForgettingAction.DELETE, memory_id=rec.id)
        result = await executor.execute(operations=[op])
        assert result.deleted == 0
        assert len(result.errors) == 1
        assert "dependency" in result.errors[0].lower()
        mock_store.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_delete_calls_delete_when_no_references(self):
        rec = _make_record()
        mock_store = MagicMock()
        mock_store.count_references_to = AsyncMock(return_value=0)
        mock_store.delete = AsyncMock(return_value=True)
        executor = ForgettingExecutor(store=mock_store)
        op = ForgettingOperation(action=ForgettingAction.DELETE, memory_id=rec.id)
        result = await executor.execute(operations=[op])
        assert result.deleted == 1
        assert result.errors == []
        mock_store.delete.assert_called_once_with(rec.id, hard=False)
