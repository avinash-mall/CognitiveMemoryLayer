"""Labile state tracking for memories after retrieval."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from uuid import UUID
import asyncio


@dataclass
class LabileMemory:
    """A memory in labile state."""

    memory_id: UUID
    retrieved_at: datetime
    context: str  # Query that triggered retrieval
    relevance_score: float
    original_confidence: float
    expires_at: datetime  # When labile state expires


@dataclass
class LabileSession:
    """Tracks labile memories for a scope session."""

    tenant_id: str
    scope_id: str
    turn_id: str

    memories: Dict[UUID, LabileMemory] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Context from retrieval
    query: str = ""
    retrieved_texts: List[str] = field(default_factory=list)


class LabileStateTracker:
    """
    Tracks memories in labile (unstable) state.

    After retrieval, memories enter a labile state where they
    can be modified based on new information. This mimics
    biological reconsolidation.
    """

    def __init__(
        self,
        labile_duration_seconds: float = 300,  # 5 minutes
        max_sessions_per_scope: int = 10,
    ):
        self.labile_duration = timedelta(seconds=labile_duration_seconds)
        self.max_sessions = max_sessions_per_scope

        # Sessions indexed by (tenant_id, scope_id, turn_id)
        self._sessions: Dict[str, LabileSession] = {}
        self._scope_sessions: Dict[str, List[str]] = {}  # scope -> [session_keys]
        self._lock = asyncio.Lock()

    def _session_key(self, tenant_id: str, scope_id: str, turn_id: str) -> str:
        return f"{tenant_id}:{scope_id}:{turn_id}"

    def _scope_key(self, tenant_id: str, scope_id: str) -> str:
        return f"{tenant_id}:{scope_id}"

    async def mark_labile(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: str,
        memory_ids: List[UUID],
        query: str,
        retrieved_texts: List[str],
        relevance_scores: List[float],
        confidences: List[float],
    ) -> LabileSession:
        """Mark memories as labile after retrieval."""
        async with self._lock:
            session_key = self._session_key(tenant_id, scope_id, turn_id)
            scope_key = self._scope_key(tenant_id, scope_id)
            now = datetime.now(timezone.utc)
            expires = now + self.labile_duration

            session = LabileSession(
                tenant_id=tenant_id,
                scope_id=scope_id,
                turn_id=turn_id,
                query=query,
                retrieved_texts=retrieved_texts,
            )

            for mid, score, conf in zip(memory_ids, relevance_scores, confidences):
                session.memories[mid] = LabileMemory(
                    memory_id=mid,
                    retrieved_at=now,
                    context=query,
                    relevance_score=score,
                    original_confidence=conf,
                    expires_at=expires,
                )

            self._sessions[session_key] = session

            if scope_key not in self._scope_sessions:
                self._scope_sessions[scope_key] = []
            self._scope_sessions[scope_key].append(session_key)

            await self._cleanup_old_sessions(scope_key)

            return session

    async def get_labile_memories(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: Optional[str] = None,
    ) -> List[LabileMemory]:
        """Get all currently labile memories for a scope."""
        async with self._lock:
            scope_key = self._scope_key(tenant_id, scope_id)
            now = datetime.now(timezone.utc)
            labile = []
            session_keys = self._scope_sessions.get(scope_key, [])

            for sk in session_keys:
                if turn_id and sk != self._session_key(tenant_id, scope_id, turn_id):
                    continue
                session = self._sessions.get(sk)
                if not session:
                    continue
                for mem in session.memories.values():
                    if mem.expires_at > now:
                        labile.append(mem)
            return labile

    async def get_session(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: str,
    ) -> Optional[LabileSession]:
        """Get a specific session."""
        async with self._lock:
            session_key = self._session_key(tenant_id, scope_id, turn_id)
            return self._sessions.get(session_key)

    async def release_labile(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: str,
        memory_ids: Optional[List[UUID]] = None,
    ) -> None:
        """Release memories from labile state. Called after reconsolidation is complete."""
        async with self._lock:
            session_key = self._session_key(tenant_id, scope_id, turn_id)
            session = self._sessions.get(session_key)
            if not session:
                return
            if memory_ids:
                for mid in memory_ids:
                    session.memories.pop(mid, None)
            else:
                session.memories.clear()
            if not session.memories:
                del self._sessions[session_key]
                scope_key = self._scope_key(tenant_id, scope_id)
                if scope_key in self._scope_sessions:
                    self._scope_sessions[scope_key] = [
                        k for k in self._scope_sessions[scope_key] if k != session_key
                    ]

    async def _cleanup_old_sessions(self, scope_key: str) -> None:
        """Remove old sessions for a scope."""
        sessions = self._scope_sessions.get(scope_key, [])
        if len(sessions) <= self.max_sessions:
            return
        now = datetime.now(timezone.utc)
        to_remove = []
        for sk in list(sessions):
            session = self._sessions.get(sk)
            if not session:
                to_remove.append(sk)
                continue
            all_expired = all(m.expires_at <= now for m in session.memories.values())
            if all_expired:
                to_remove.append(sk)
        for sk in to_remove:
            self._sessions.pop(sk, None)
            if scope_key in self._scope_sessions:
                self._scope_sessions[scope_key] = [
                    k for k in self._scope_sessions[scope_key] if k != sk
                ]
        sessions = self._scope_sessions.get(scope_key, [])
        while len(sessions) > self.max_sessions:
            oldest_key = sessions.pop(0)
            self._sessions.pop(oldest_key, None)
