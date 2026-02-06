"""Tool result storage for agentic workflows."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..core.enums import MemorySource, MemoryType
from ..core.schemas import MemoryRecordCreate, Provenance
from ..storage.base import MemoryStoreBase


class ToolMemory:
    """
    Store outputs from tool executions per session.
    Enables agents to remember and reuse tool results within a session.
    Holistic: tenant-only, session_id for origin tracking.
    """

    def __init__(self, store: MemoryStoreBase) -> None:
        self.store = store

    async def store_result(
        self,
        tenant_id: str,
        session_id: str,
        tool_name: str,
        input_params: Dict[str, Any],
        output: Any,
    ) -> UUID:
        """Store a tool execution result. Returns the memory record id."""
        text = json.dumps({"tool": tool_name, "input": input_params, "output": output})
        meta: Dict[str, Any] = {
            "tool_name": tool_name,
            "input_params": input_params,
        }
        record = MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=["conversation", "tool_result"],
            source_session_id=session_id,
            type=MemoryType.TOOL_RESULT,
            text=text,
            key=f"tool_result:{session_id}:{tool_name}:{datetime.now(timezone.utc).isoformat()}",
            embedding=None,
            metadata=meta,
            provenance=Provenance(source=MemorySource.TOOL_RESULT, tool_refs=[tool_name]),
        )
        created = await self.store.upsert(record)
        return created.id

    async def get_results(
        self,
        tenant_id: str,
        session_id: str,
        tool_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve tool results for the session, optionally filtered by tool_name."""
        filters: Dict[str, Any] = {
            "status": "active",
            "type": MemoryType.TOOL_RESULT.value,
            "source_session_id": session_id,
        }
        records = await self.store.scan(
            tenant_id=tenant_id,
            filters=filters,
            order_by="-timestamp",
            limit=limit,
        )
        out: List[Dict[str, Any]] = []
        for r in records:
            meta = r.metadata or {}
            if tool_name and meta.get("tool_name") != tool_name:
                continue
            try:
                data = json.loads(r.text)
                out.append(
                    {
                        "id": r.id,
                        "tool_name": data.get("tool", meta.get("tool_name")),
                        "input": data.get("input", meta.get("input_params")),
                        "output": data.get("output", r.text),
                        "timestamp": r.timestamp,
                    }
                )
            except json.JSONDecodeError:
                out.append(
                    {
                        "id": r.id,
                        "tool_name": meta.get("tool_name"),
                        "input": meta.get("input_params"),
                        "output": r.text,
                        "timestamp": r.timestamp,
                    }
                )
        return out
