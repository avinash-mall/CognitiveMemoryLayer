"""SQLite-based memory store for embedded lite mode."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

try:
    from src.core.enums import (  # type: ignore[import-untyped]
        MemorySource,
        MemoryStatus,
        MemoryType,
    )
    from src.core.schemas import (  # type: ignore[import-untyped]
        EntityMention,
        MemoryRecord,
        MemoryRecordCreate,
        Provenance,
        Relation,
    )
    from src.storage.base import MemoryStoreBase  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError(
        "Embedded lite mode requires the CML engine. "
        "From the repo root: pip install -e . then pip install -e packages/py-cml[embedded]."
    ) from e

import aiosqlite  # type: ignore[import-not-found]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SQLiteMemoryStore(MemoryStoreBase):  # type: ignore[misc]
    """Memory store backed by SQLite with in-memory cosine similarity for vector search."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create tables and indexes."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                text TEXT NOT NULL,
                key TEXT,
                namespace TEXT,
                embedding TEXT,
                entities TEXT,
                relations TEXT,
                metadata TEXT,
                context_tags TEXT,
                confidence REAL DEFAULT 0.5,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                decay_rate REAL DEFAULT 0.01,
                timestamp TEXT,
                written_at TEXT,
                valid_from TEXT,
                valid_to TEXT,
                content_hash TEXT,
                version INTEGER DEFAULT 1,
                supersedes_id TEXT,
                provenance TEXT,
                source_session_id TEXT,
                agent_id TEXT,
                last_accessed_at TEXT,
                labile INTEGER DEFAULT 0
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_tenant ON memories(tenant_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_tenant_status ON memories(tenant_id, status)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _hash_content(self, text: str, tenant_id: str) -> str:
        content = f"{tenant_id}:{text.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _dt_iso(self, dt: datetime | None) -> str | None:
        if dt is None:
            return None
        return dt.isoformat()

    def _parse_dt(self, s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    async def upsert(self, record: MemoryRecordCreate) -> MemoryRecord:
        content_hash = self._hash_content(record.text, record.tenant_id)
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        # Simple insert (no dedup by content_hash for lite; could add later)
        record_id = uuid4()
        now = datetime.now(UTC)
        ts = record.timestamp or now
        await self._db.execute(
            """
            INSERT INTO memories (
                id, tenant_id, type, status, text, key, namespace, embedding,
                entities, relations, metadata, context_tags, confidence, importance,
                access_count, decay_rate, timestamp, written_at, content_hash,
                version, provenance, source_session_id, agent_id, labile
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, 1, ?, ?, ?, 0)
            """,
            (
                str(record_id),
                record.tenant_id,
                record.type.value,
                MemoryStatus.ACTIVE.value,
                record.text,
                record.key,
                record.namespace,
                json.dumps(record.embedding) if record.embedding else None,
                json.dumps([e.model_dump() for e in record.entities]),
                json.dumps([r.model_dump() for r in record.relations]),
                json.dumps(record.metadata or {}),
                json.dumps(record.context_tags or []),
                record.confidence,
                record.importance,
                0.01,
                ts.isoformat(),
                now.isoformat(),
                content_hash,
                record.provenance.model_dump_json(),
                record.source_session_id,
                record.agent_id,
            ),
        )
        await self._db.commit()
        out = await self.get_by_id(record_id)
        if out is None:
            raise RuntimeError("upsert failed to return record")
        return out

    async def get_by_id(self, record_id: UUID) -> MemoryRecord | None:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM memories WHERE id = ?", (str(record_id),)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(dict(row))

    async def get_by_key(
        self,
        tenant_id: str,
        key: str,
        context_filter: list[str] | None = None,
    ) -> MemoryRecord | None:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM memories WHERE tenant_id = ? AND key = ? AND status = ?",
            (tenant_id, key, MemoryStatus.ACTIVE.value),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(dict(row))

    async def delete(self, record_id: UUID, hard: bool = False) -> bool:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        if hard:
            cursor = await self._db.execute("DELETE FROM memories WHERE id = ?", (str(record_id),))
        else:
            cursor = await self._db.execute(
                "UPDATE memories SET status = ? WHERE id = ?",
                (MemoryStatus.DELETED.value, str(record_id)),
            )
        await self._db.commit()
        return bool(cursor.rowcount and cursor.rowcount > 0)

    async def update(
        self,
        record_id: UUID,
        patch: dict[str, Any],
        increment_version: bool = True,
    ) -> MemoryRecord | None:
        rec = await self.get_by_id(record_id)
        if rec is None:
            return None
        if self._db is None:
            return None
        allow = {
            "access_count",
            "last_accessed_at",
            "confidence",
            "importance",
            "status",
            "metadata",
            "text",
            "embedding",
            "valid_to",
            "entities",
            "relations",
            "context_tags",
            "source_session_id",
        }
        updates: list[str] = []
        params: list[Any] = []
        for k, v in patch.items():
            key = "metadata" if k == "meta" else k
            if key not in allow:
                continue
            if key == "embedding":
                updates.append("embedding = ?")
                params.append(json.dumps(v) if v else None)
            elif key in ("entities", "relations", "context_tags", "metadata"):
                updates.append(f"{key} = ?")
                params.append(json.dumps(v) if v else "[]" if key != "metadata" else "{}")
            elif key in ("last_accessed_at", "valid_to") and isinstance(v, datetime):
                updates.append(f"{key} = ?")
                params.append(v.isoformat())
            else:
                updates.append(f"{key} = ?")
                params.append(v)
        if increment_version:
            updates.append("version = version + 1")
        params.append(str(record_id))
        await self._db.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)
        await self._db.commit()
        return await self.get_by_id(record_id)

    async def vector_search(
        self,
        tenant_id: str,
        embedding: list[float],
        top_k: int = 10,
        context_filter: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[MemoryRecord]:
        rows = await self.scan(
            tenant_id=tenant_id,
            filters=filters or {},
            limit=5000,
            offset=0,
        )
        scored: list[tuple[float, MemoryRecord]] = []
        for rec in rows:
            if not rec.embedding:
                continue
            sim = _cosine_similarity(embedding, rec.embedding)
            if sim >= min_similarity:
                scored.append((sim, rec))
        scored.sort(key=lambda x: -x[0])
        return [rec for _, rec in scored[:top_k]]

    async def scan(
        self,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        q = "SELECT * FROM memories WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if filters:
            if filters.get("status"):
                q += " AND status = ?"
                params.append(filters["status"])
            if filters.get("type"):
                q += " AND type = ?"
                params.append(filters["type"])
        q += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        async with self._db.execute(q, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    async def count(
        self,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
    ) -> int:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        q = "SELECT COUNT(*) FROM memories WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if filters:
            if filters.get("status"):
                q += " AND status = ?"
                params.append(filters["status"])
            if filters.get("type"):
                q += " AND type = ?"
                params.append(filters["type"])
        async with self._db.execute(q, params) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    def _row_to_record(self, row: dict[str, Any]) -> MemoryRecord:
        try:
            mem_type = MemoryType(row["type"])
        except (ValueError, KeyError):
            mem_type = MemoryType.EPISODIC_EVENT
        try:
            status = MemoryStatus(row.get("status", "active"))
        except (ValueError, KeyError):
            status = MemoryStatus.ACTIVE
        prov = row.get("provenance")
        if isinstance(prov, str):
            prov = json.loads(prov) if prov else {}
        if not isinstance(prov, dict):
            prov = {}
        if "source" not in prov:
            prov["source"] = MemorySource.AGENT_INFERRED.value
        provenance = Provenance(**prov)
        entities = row.get("entities")
        if isinstance(entities, str):
            entities = json.loads(entities) if entities else []
        entities = [EntityMention(**e) for e in (entities or [])]
        relations = row.get("relations")
        if isinstance(relations, str):
            relations = json.loads(relations) if relations else []
        relations = [Relation(**r) for r in (relations or [])]
        context_tags = row.get("context_tags")
        if isinstance(context_tags, str):
            context_tags = json.loads(context_tags) if context_tags else []
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        emb = row.get("embedding")
        if isinstance(emb, str):
            emb = json.loads(emb) if emb else None
        return MemoryRecord(
            id=UUID(row["id"]),
            tenant_id=row["tenant_id"],
            context_tags=context_tags or [],
            source_session_id=row.get("source_session_id"),
            agent_id=row.get("agent_id"),
            namespace=row.get("namespace"),
            type=mem_type,
            text=row["text"],
            key=row.get("key"),
            embedding=emb,
            entities=entities,
            relations=relations,
            metadata=metadata or {},
            timestamp=self._parse_dt(row.get("timestamp")) or datetime.now(UTC),
            written_at=self._parse_dt(row.get("written_at")) or datetime.now(UTC),
            valid_from=self._parse_dt(row.get("valid_from")),
            valid_to=self._parse_dt(row.get("valid_to")),
            confidence=float(row.get("confidence", 0.5)),
            importance=float(row.get("importance", 0.5)),
            access_count=int(row.get("access_count", 0)),
            last_accessed_at=self._parse_dt(row.get("last_accessed_at")),
            decay_rate=float(row.get("decay_rate", 0.01)),
            status=status,
            labile=bool(row.get("labile", 0)),
            provenance=provenance,
            version=int(row.get("version", 1)),
            supersedes_id=(UUID(row["supersedes_id"]) if row.get("supersedes_id") else None),
            content_hash=row.get("content_hash"),
        )
