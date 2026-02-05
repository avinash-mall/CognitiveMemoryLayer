"""Conversation memory: store and retrieve conversation history."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..core.enums import MemoryScope, MemorySource, MemoryType
from ..core.schemas import MemoryRecordCreate, Provenance
from ..storage.base import MemoryStoreBase


class ConversationMemory:
    """
    Store and retrieve conversation history per session.
    Messages are stored as memory records with type MESSAGE or CONVERSATION.
    """

    def __init__(self, store: MemoryStoreBase) -> None:
        self.store = store

    async def add_message(
        self,
        tenant_id: str,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> UUID:
        """Add a conversation message. Returns the memory record id."""
        meta: Dict[str, Any] = {"role": role}
        if tool_calls:
            meta["tool_calls"] = tool_calls
        text = content
        record = MemoryRecordCreate(
            tenant_id=tenant_id,
            scope=MemoryScope.SESSION,
            scope_id=session_id,
            user_id=None,
            session_id=session_id,
            type=MemoryType.MESSAGE,
            text=text,
            key=None,
            embedding=None,
            metadata=meta,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT if role == "user" else MemorySource.AGENT_INFERRED),
        )
        created = await self.store.upsert(record)
        return created.id

    async def get_history(
        self,
        tenant_id: str,
        session_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get recent conversation history (messages in time order)."""
        records = await self.store.scan(
            tenant_id=tenant_id,
            user_id=session_id,
            filters={"status": "active", "type": [MemoryType.MESSAGE.value, MemoryType.CONVERSATION.value]},
            order_by="-timestamp",
            limit=limit,
        )
        out: List[Dict[str, Any]] = []
        for r in reversed(records):
            meta = r.metadata or {}
            out.append({
                "id": r.id,
                "role": meta.get("role", "user"),
                "content": r.text,
                "timestamp": r.timestamp,
                "tool_calls": meta.get("tool_calls"),
            })
        return out

    async def summarize_and_compress(
        self,
        tenant_id: str,
        session_id: str,
        keep_recent: int = 10,
        summary_text: Optional[str] = None,
    ) -> str:
        """
        Summarize old messages and keep recent ones.
        Returns a combined context string (summary + recent).
        Caller can use an LLM to produce summary_text from get_history(limit=large).
        """
        history = await self.get_history(tenant_id, session_id, limit=keep_recent + 100)
        if len(history) <= keep_recent:
            lines = [f"{m['role']}: {m['content']}" for m in history]
            return "\n".join(lines)
        recent = history[-keep_recent:]
        if summary_text:
            return summary_text.strip() + "\n\n--- Recent ---\n" + "\n".join(
                f"{m['role']}: {m['content']}" for m in recent
            )
        lines = [f"{m['role']}: {m['content']}" for m in recent]
        return "\n".join(lines)
