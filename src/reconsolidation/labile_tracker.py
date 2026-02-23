"""Labile state tracking for memories after retrieval."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

# Redis key prefixes and TTL multiplier (session TTL = labile_duration * this)
_LABILE_PREFIX = "labile:"
_SESSION_PREFIX = _LABILE_PREFIX + "session:"
_SCOPE_PREFIX = _LABILE_PREFIX + "scope:"
_SESSION_TTL_MULTIPLIER = 2  # Redis TTL so keys expire after labile window

# CON-03: Lua script for atomic SETEX + LPUSH + LTRIM (no inter-worker race)
_LUA_MARK_LABILE = """
redis.call('SETEX', KEYS[1], ARGV[1], ARGV[2])
redis.call('LPUSH', KEYS[2], ARGV[3])
redis.call('LTRIM', KEYS[2], 0, ARGV[4])
return 1
"""


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

    memories: dict[UUID, LabileMemory] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Context from retrieval
    query: str = ""
    retrieved_texts: list[str] = field(default_factory=list)


def _serialize_memory(m: LabileMemory) -> dict[str, Any]:
    return {
        "memory_id": str(m.memory_id),
        "retrieved_at": m.retrieved_at.isoformat(),
        "context": m.context,
        "relevance_score": m.relevance_score,
        "original_confidence": m.original_confidence,
        "expires_at": m.expires_at.isoformat(),
    }


def _deserialize_memory(d: dict[str, Any]) -> LabileMemory:
    return LabileMemory(
        memory_id=UUID(d["memory_id"]),
        retrieved_at=datetime.fromisoformat(d["retrieved_at"].replace("Z", "+00:00")),
        context=d["context"],
        relevance_score=float(d["relevance_score"]),
        original_confidence=float(d["original_confidence"]),
        expires_at=datetime.fromisoformat(d["expires_at"].replace("Z", "+00:00")),
    )


def _serialize_session(s: LabileSession) -> dict[str, Any]:
    return {
        "tenant_id": s.tenant_id,
        "scope_id": s.scope_id,
        "turn_id": s.turn_id,
        "memories": {str(k): _serialize_memory(v) for k, v in s.memories.items()},
        "created_at": s.created_at.isoformat(),
        "query": s.query,
        "retrieved_texts": s.retrieved_texts,
    }


def _deserialize_session(d: dict[str, Any]) -> LabileSession:
    memories = {UUID(k): _deserialize_memory(v) for k, v in (d.get("memories") or {}).items()}
    created = d.get("created_at")
    created_at = (
        datetime.fromisoformat(created.replace("Z", "+00:00")) if created else datetime.now(UTC)
    )
    return LabileSession(
        tenant_id=d["tenant_id"],
        scope_id=d["scope_id"],
        turn_id=d["turn_id"],
        memories=memories,
        created_at=created_at,
        query=d.get("query", ""),
        retrieved_texts=d.get("retrieved_texts") or [],
    )


class LabileStateTracker:
    """
    Tracks memories in labile (unstable) state.

    After retrieval, memories enter a labile state where they
    can be modified based on new information. This mimics
    biological reconsolidation.

    When redis_client is provided, state is stored in Redis so it is
    shared across workers (multi-worker safe). When redis_client is None,
    state is kept in process memory (single-worker only).
    """

    def __init__(
        self,
        labile_duration_seconds: float = 300,  # 5 minutes
        max_sessions_per_scope: int = 10,
        redis_client: Any | None = None,
    ):
        self.labile_duration = timedelta(seconds=labile_duration_seconds)
        self.max_sessions = max_sessions_per_scope
        self._redis = redis_client

        # In-memory state (used when redis_client is None)
        self._sessions: dict[str, LabileSession] = {}
        self._scope_sessions: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    def _session_key(self, tenant_id: str, scope_id: str, turn_id: str) -> str:
        return f"{tenant_id}:{scope_id}:{turn_id}"

    def _scope_key(self, tenant_id: str, scope_id: str) -> str:
        return f"{tenant_id}:{scope_id}"

    def _redis_session_key(self, session_key: str) -> str:
        return _SESSION_PREFIX + session_key

    def _redis_scope_key(self, scope_key: str) -> str:
        return _SCOPE_PREFIX + scope_key

    async def mark_labile(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: str,
        memory_ids: list[UUID],
        query: str,
        retrieved_texts: list[str],
        relevance_scores: list[float],
        confidences: list[float],
    ) -> LabileSession:
        """Mark memories as labile after retrieval."""
        session_key = self._session_key(tenant_id, scope_id, turn_id)
        scope_key = self._scope_key(tenant_id, scope_id)
        now = datetime.now(UTC)
        expires = now + self.labile_duration

        session = LabileSession(
            tenant_id=tenant_id,
            scope_id=scope_id,
            turn_id=turn_id,
            query=query,
            retrieved_texts=retrieved_texts,
        )
        for mid, score, conf in zip(memory_ids, relevance_scores, confidences, strict=False):
            session.memories[mid] = LabileMemory(
                memory_id=mid,
                retrieved_at=now,
                context=query,
                relevance_score=score,
                original_confidence=conf,
                expires_at=expires,
            )

        if self._redis is not None:
            async with self._lock:
                await self._mark_labile_redis(session_key, scope_key, session)
        else:
            async with self._lock:
                self._sessions[session_key] = session
                if scope_key not in self._scope_sessions:
                    self._scope_sessions[scope_key] = []
                self._scope_sessions[scope_key].append(session_key)
                await self._cleanup_old_sessions(scope_key)

        return session

    async def _mark_labile_redis(
        self, session_key: str, scope_key: str, session: LabileSession
    ) -> None:
        assert self._redis is not None  # caller ensures Redis backend
        ttl = int(self.labile_duration.total_seconds() * _SESSION_TTL_MULTIPLIER)
        rk = self._redis_session_key(session_key)
        sk = self._redis_scope_key(scope_key)
        payload = json.dumps(_serialize_session(session))
        # CON-03: atomic SETEX + LPUSH + LTRIM so other workers cannot interleave
        try:
            await self._redis.eval(
                _LUA_MARK_LABILE,
                2,
                rk,
                sk,
                ttl,
                payload,
                session_key,
                self.max_sessions - 1,
            )
        except Exception as e:
            logger.warning("labile_redis_lua_failed falling back to non-atomic: %s", e)
            await self._redis.setex(rk, ttl, payload)
            await self._redis.lpush(sk, session_key)
            await self._redis.ltrim(sk, 0, self.max_sessions - 1)
        await self._cleanup_old_sessions_redis(scope_key)

    async def get_labile_memories(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: str | None = None,
    ) -> list[LabileMemory]:
        """Get all currently labile memories for a scope."""
        if self._redis is not None:
            async with self._lock:
                return await self._get_labile_memories_redis(tenant_id, scope_id, turn_id)
        async with self._lock:
            scope_key = self._scope_key(tenant_id, scope_id)
            now = datetime.now(UTC)
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

    async def _get_labile_memories_redis(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: str | None,
    ) -> list[LabileMemory]:
        assert self._redis is not None  # caller ensures Redis backend
        scope_key = self._scope_key(tenant_id, scope_id)
        sk = self._redis_scope_key(scope_key)
        raw_list = await self._redis.lrange(sk, 0, -1)
        session_keys = [k.decode() if isinstance(k, bytes) else k for k in raw_list]
        now = datetime.now(UTC)
        labile: list[LabileMemory] = []
        target_session = self._session_key(tenant_id, scope_id, turn_id) if turn_id else None
        for sess_key in session_keys:
            if target_session and sess_key != target_session:
                continue
            rk = self._redis_session_key(sess_key)
            data = await self._redis.get(rk)
            if not data:
                continue
            try:
                doc = json.loads(data.decode() if isinstance(data, bytes) else data)
                session = _deserialize_session(doc)
                for mem in session.memories.values():
                    if mem.expires_at > now:
                        labile.append(mem)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("labile_deserialize_failed key=%s error=%s", rk, e)
        return labile

    async def get_session(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: str,
    ) -> LabileSession | None:
        """Get a specific session."""
        session_key = self._session_key(tenant_id, scope_id, turn_id)
        if self._redis is not None:
            async with self._lock:
                rk = self._redis_session_key(session_key)
                data = await self._redis.get(rk)
                if not data:
                    return None
                try:
                    doc = json.loads(data.decode() if isinstance(data, bytes) else data)
                    return _deserialize_session(doc)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning("labile_deserialize_failed key=%s error=%s", rk, e)
                    return None
        async with self._lock:
            return self._sessions.get(session_key)

    async def release_labile(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: str,
        memory_ids: list[UUID] | None = None,
    ) -> None:
        """Release memories from labile state. Called after reconsolidation is complete."""
        session_key = self._session_key(tenant_id, scope_id, turn_id)
        scope_key = self._scope_key(tenant_id, scope_id)
        if self._redis is not None:
            async with self._lock:
                await self._release_labile_redis(session_key, scope_key, memory_ids)
            return
        async with self._lock:
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
                if scope_key in self._scope_sessions:
                    self._scope_sessions[scope_key] = [
                        k for k in self._scope_sessions[scope_key] if k != session_key
                    ]

    async def _release_labile_redis(
        self,
        session_key: str,
        scope_key: str,
        memory_ids: list[UUID] | None,
    ) -> None:
        assert self._redis is not None  # caller ensures Redis backend
        rk = self._redis_session_key(session_key)
        sk = self._redis_scope_key(scope_key)
        data = await self._redis.get(rk)
        if not data:
            return
        try:
            doc = json.loads(data.decode() if isinstance(data, bytes) else data)
            session = _deserialize_session(doc)
        except (json.JSONDecodeError, KeyError, TypeError):
            await self._redis.delete(rk)
            await self._redis.lrem(sk, 0, session_key)
            return
        if memory_ids:
            for mid in memory_ids:
                session.memories.pop(mid, None)
        else:
            session.memories.clear()
        if not session.memories:
            await self._redis.delete(rk)
            await self._redis.lrem(sk, 0, session_key)
        else:
            ttl = int(self.labile_duration.total_seconds() * _SESSION_TTL_MULTIPLIER)
            await self._redis.setex(rk, ttl, json.dumps(_serialize_session(session)))

    async def _cleanup_old_sessions(self, scope_key: str) -> None:
        """Remove old sessions for a scope (in-memory backend)."""
        sessions = self._scope_sessions.get(scope_key, [])
        if len(sessions) <= self.max_sessions:
            return
        now = datetime.now(UTC)
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

    async def _cleanup_old_sessions_redis(self, scope_key: str) -> None:
        """Remove expired sessions and trim scope list (Redis backend)."""
        assert self._redis is not None  # caller ensures Redis backend
        sk = self._redis_scope_key(scope_key)
        raw_list = await self._redis.lrange(sk, 0, -1)
        session_keys = [k.decode() if isinstance(k, bytes) else k for k in raw_list]
        now = datetime.now(UTC)
        to_remove = []
        for sess_key in session_keys:
            rk = self._redis_session_key(sess_key)
            data = await self._redis.get(rk)
            if not data:
                to_remove.append(sess_key)
                continue
            try:
                doc = json.loads(data.decode() if isinstance(data, bytes) else data)
                session = _deserialize_session(doc)
                all_expired = all(m.expires_at <= now for m in session.memories.values())
                if all_expired:
                    to_remove.append(sess_key)
            except (json.JSONDecodeError, KeyError, TypeError):
                to_remove.append(sess_key)
        for sess_key in to_remove:
            await self._redis.delete(self._redis_session_key(sess_key))
            await self._redis.lrem(sk, 0, sess_key)
        # Trim to max_sessions (list is newest-first from LPUSH)
        await self._redis.ltrim(sk, 0, self.max_sessions - 1)

    async def release_all_for_tenant(self, tenant_id: str) -> int:
        """
        Release all labile sessions for the given tenant (clear labile state).
        Returns the number of sessions released. No belief revision is applied.
        """
        if self._redis is not None:
            async with self._lock:
                return await self._release_all_for_tenant_redis(tenant_id)
        async with self._lock:
            return await self._release_all_for_tenant_memory(tenant_id)

    async def _release_all_for_tenant_memory(self, tenant_id: str) -> int:
        """Release all labile sessions for tenant (in-memory backend)."""
        prefix = tenant_id + ":"
        count = 0
        for scope_key in list(self._scope_sessions.keys()):
            if not scope_key.startswith(prefix):
                continue
            for session_key in self._scope_sessions.get(scope_key, []):
                if session_key in self._sessions:
                    del self._sessions[session_key]
                    count += 1
            del self._scope_sessions[scope_key]
        return count

    async def _release_all_for_tenant_redis(self, tenant_id: str) -> int:
        """Release all labile sessions for tenant (Redis backend)."""
        assert self._redis is not None  # caller ensures Redis backend
        # Redis scope keys: labile:scope:tenant_id:scope_id
        match_pattern = f"labile:scope:{tenant_id}:*"
        cursor = 0
        total_released = 0
        while True:
            cursor, keys = await self._redis.scan(cursor, match=match_pattern, count=200)
            for key in keys:
                sk = key.decode() if isinstance(key, bytes) else key
                # sk is full Redis key e.g. "labile:scope:lp-0:session_1"
                raw_list = await self._redis.lrange(sk, 0, -1)
                session_keys = [k.decode() if isinstance(k, bytes) else k for k in raw_list]
                for sess_key in session_keys:
                    rk = self._redis_session_key(sess_key)
                    await self._redis.delete(rk)
                    total_released += 1
                await self._redis.delete(sk)
            if cursor == 0:
                break
        return total_released
