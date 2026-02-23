"""Execution of forgetting operations on the memory store."""

from datetime import UTC, datetime

from ..core.enums import MemorySource, MemoryStatus
from ..core.schemas import MemoryRecord, MemoryRecordCreate, Provenance
from ..storage.base import MemoryStoreBase
from ..utils.llm import LLMClient
from .actions import ForgettingAction, ForgettingOperation, ForgettingResult
from .compression import summarize_for_compression


class ForgettingExecutor:
    """Executes forgetting operations on the memory store."""

    def __init__(
        self,
        store: MemoryStoreBase,
        archive_store: MemoryStoreBase | None = None,
        compression_llm_client: LLMClient | None = None,
        compression_max_chars: int = 100,
    ) -> None:
        self.store = store
        self.archive_store = archive_store
        self.compression_llm_client = compression_llm_client
        self.compression_max_chars = compression_max_chars

    async def execute(
        self,
        operations: list[ForgettingOperation],
        dry_run: bool = False,
    ) -> ForgettingResult:
        """Execute forgetting operations. If dry_run, only count, no writes."""
        result = ForgettingResult(
            operations_planned=len(operations),
            operations_applied=0,
        )

        for op in operations:
            try:
                if op.action == ForgettingAction.KEEP:
                    result.kept += 1
                    continue

                if dry_run:
                    self._count_action(result, op.action)
                    result.operations_applied += 1
                    continue

                success, skip_reason = await self._execute_operation(op)
                if success:
                    self._count_action(result, op.action)
                    result.operations_applied += 1
                else:
                    msg = skip_reason or f"Failed to execute {op.action} on {op.memory_id}"
                    if result.errors is not None:
                        result.errors.append(msg)
            except Exception as e:
                if result.errors is not None:
                    result.errors.append(f"Error executing {op.action} on {op.memory_id}: {e}")

        return result

    async def _execute_operation(self, op: ForgettingOperation) -> tuple[bool, str | None]:
        """Execute a single operation. Returns (success, skip_reason)."""
        if op.action == ForgettingAction.DECAY:
            ok = await self._execute_decay(op)
            return (ok, None)
        if op.action == ForgettingAction.SILENCE:
            ok = await self._execute_silence(op)
            return (ok, None)
        if op.action == ForgettingAction.COMPRESS:
            ok = await self._execute_compress(op)
            return (ok, None)
        if op.action == ForgettingAction.ARCHIVE:
            ok = await self._execute_archive(op)
            return (ok, None)
        if op.action == ForgettingAction.DELETE:
            return await self._execute_delete(op)
        return (False, None)

    async def _execute_decay(self, op: ForgettingOperation) -> bool:
        """Reduce confidence of a memory. BUG-05: merge metadata."""
        if op.new_confidence is None:
            return False
        record = await self.store.get_by_id(op.memory_id)
        if not record:
            return False
        merged_meta = {
            **(record.metadata or {}),
            "last_decay": datetime.now(UTC).isoformat(),
        }
        patch: dict[str, object] = {"confidence": op.new_confidence, "metadata": merged_meta}
        result = await self.store.update(op.memory_id, patch, increment_version=False)
        return result is not None

    async def _execute_silence(self, op: ForgettingOperation) -> bool:
        """Mark memory as silent (hard to retrieve). BUG-05: merge metadata."""
        record = await self.store.get_by_id(op.memory_id)
        if not record:
            return False
        merged_meta = {
            **(record.metadata or {}),
            "silenced_at": datetime.now(UTC).isoformat(),
        }
        patch: dict[str, object] = {"status": MemoryStatus.SILENT.value, "metadata": merged_meta}
        result = await self.store.update(op.memory_id, patch)
        return result is not None

    async def _execute_compress(self, op: ForgettingOperation) -> bool:
        """Compress memory to gist only (LLM summarization when client provided)."""
        record = await self.store.get_by_id(op.memory_id)
        if not record:
            return False
        if op.compressed_text:
            compressed = op.compressed_text
        elif self.compression_llm_client and len(record.text) > self.compression_max_chars:
            compressed = await summarize_for_compression(
                record.text,
                max_chars=self.compression_max_chars,
                llm_client=self.compression_llm_client,
            )
        else:
            compressed = record.text[: self.compression_max_chars]
            if len(record.text) > self.compression_max_chars:
                compressed = record.text[: self.compression_max_chars - 3] + "..."
        meta = dict(record.metadata)
        meta["compressed_at"] = datetime.now(UTC).isoformat()
        meta["original_length"] = len(record.text)
        patch: dict[str, object] = {
            "text": compressed,
            "status": MemoryStatus.COMPRESSED.value,
            "embedding": None,
            "entities": [],
            "relations": [],
            "metadata": meta,
        }
        result = await self.store.update(op.memory_id, patch)
        return result is not None

    def _record_to_create_schema(self, record: MemoryRecord) -> MemoryRecordCreate:
        """Convert MemoryRecord to MemoryRecordCreate for archive upsert.

        All fields required by MemoryRecordCreate are copied, including provenance.
        Used as the single code path when archive_store is set (no data loss).
        """
        provenance = record.provenance
        if provenance is None:
            provenance = Provenance(source=MemorySource.AGENT_INFERRED)
        return MemoryRecordCreate(
            tenant_id=record.tenant_id,
            context_tags=record.context_tags or [],
            source_session_id=record.source_session_id,
            agent_id=record.agent_id,
            namespace=record.namespace,
            type=record.type,
            text=record.text,
            key=record.key,
            embedding=record.embedding,
            entities=record.entities,
            relations=record.relations,
            metadata=record.metadata or {},
            timestamp=record.timestamp,
            confidence=record.confidence,
            importance=record.importance,
            provenance=provenance,
        )

    async def _execute_archive(self, op: ForgettingOperation) -> bool:
        """Move to archive store or mark archived. BUG-04: two-phase for atomicity."""
        record = await self.store.get_by_id(op.memory_id)
        if not record:
            return False
        now_iso = datetime.now(UTC).isoformat()
        merged_meta = {**(record.metadata or {}), "archived_at": now_iso}

        if not self.archive_store:
            patch = {
                "status": MemoryStatus.ARCHIVED.value,
                "metadata": merged_meta,
            }
            result = await self.store.update(op.memory_id, patch)
            return result is not None

        # Phase 1: mark ARCHIVED in primary so failure after this doesn't lose data
        await self.store.update(
            op.memory_id,
            {"status": MemoryStatus.ARCHIVED.value, "metadata": merged_meta},
        )
        # Phase 2: copy to archive
        await self.archive_store.upsert(self._record_to_create_schema(record))
        # Phase 3: hard delete from primary
        await self.store.delete(op.memory_id, hard=True)
        return True

    async def _execute_delete(self, op: ForgettingOperation) -> tuple[bool, str | None]:
        """Soft-delete memory from store; skip if other memories reference it."""
        ref_count = await self.store.count_references_to(op.memory_id)
        if ref_count > 0:
            return (
                False,
                f"Skipped delete {op.memory_id}: {ref_count} dependency(ies)",
            )
        ok = await self.store.delete(op.memory_id, hard=False)
        return (ok, None)

    def _count_action(self, result: ForgettingResult, action: ForgettingAction) -> None:
        """Increment counter for action type."""
        if action == ForgettingAction.KEEP:
            result.kept += 1
        elif action == ForgettingAction.DECAY:
            result.decayed += 1
        elif action == ForgettingAction.SILENCE:
            result.silenced += 1
        elif action == ForgettingAction.COMPRESS:
            result.compressed += 1
        elif action == ForgettingAction.ARCHIVE:
            result.archived += 1
        elif action == ForgettingAction.DELETE:
            result.deleted += 1
