"""Ephemeral working memory (scratch pad) for multi-step reasoning."""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.enums import MemoryScope, MemorySource, MemoryType
from ..core.schemas import MemoryRecordCreate, Provenance
from ..storage.base import MemoryStoreBase


class ScratchPad:
    """
    Ephemeral working memory for multi-step reasoning.
    Stores key-value data per session; values are JSON-serialized.
    Uses the same store as other memories with type=SCRATCH and key prefix.
    """

    SCRATCH_KEY_PREFIX = "scratch:"

    def __init__(self, store: MemoryStoreBase) -> None:
        self.store = store

    def _scratch_key(self, key: str) -> str:
        return f"{self.SCRATCH_KEY_PREFIX}{key}"

    async def set(
        self,
        tenant_id: str,
        session_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Store a scratch value (overwrites if key exists)."""
        text = json.dumps(value) if not isinstance(value, str) else value
        record = MemoryRecordCreate(
            tenant_id=tenant_id,
            scope=MemoryScope.SESSION,
            scope_id=session_id,
            user_id=None,
            session_id=session_id,
            type=MemoryType.SCRATCH,
            text=text,
            key=self._scratch_key(key),
            embedding=None,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
        )
        await self.store.upsert(record)

    async def get(
        self,
        tenant_id: str,
        session_id: str,
        key: str,
    ) -> Optional[Any]:
        """Retrieve a scratch value by key."""
        record = await self.store.get_by_key(
            tenant_id=tenant_id,
            user_id=session_id,
            key=self._scratch_key(key),
        )
        if not record or not record.text:
            return None
        try:
            return json.loads(record.text)
        except json.JSONDecodeError:
            return record.text

    async def append(
        self,
        tenant_id: str,
        session_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Append to a list in scratch (creates list if key missing)."""
        existing = await self.get(tenant_id, session_id, key)
        if existing is None:
            new_list: List[Any] = [value]
        elif isinstance(existing, list):
            new_list = existing + [value]
        else:
            new_list = [existing, value]
        await self.set(tenant_id, session_id, key, new_list)

    async def clear(
        self,
        tenant_id: str,
        session_id: str,
    ) -> None:
        """Clear all scratch data for the session."""
        records = await self.store.scan(
            tenant_id=tenant_id,
            user_id=session_id,
            filters={"status": "active", "type": MemoryType.SCRATCH.value},
            limit=1000,
        )
        for r in records:
            if r.key and r.key.startswith(self.SCRATCH_KEY_PREFIX):
                await self.store.delete(r.id, hard=True)
