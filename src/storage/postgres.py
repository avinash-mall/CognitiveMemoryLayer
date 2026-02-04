"""PostgreSQL memory store with pgvector."""
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID


def _naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert to naive UTC for PostgreSQL TIMESTAMP WITHOUT TIME ZONE."""
    if dt is None:
        return None
    if dt.tzinfo:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.enums import MemoryStatus, MemoryType
from ..core.schemas import EntityMention, MemoryRecord, MemoryRecordCreate, Provenance, Relation
from .base import MemoryStoreBase
from .models import MemoryRecordModel


class PostgresMemoryStore(MemoryStoreBase):
    """PostgreSQL-based memory store with pgvector."""

    def __init__(self, session_factory: Any) -> None:
        self.session_factory = session_factory


    async def upsert(self, record: MemoryRecordCreate) -> MemoryRecord:
        content_hash = self._hash_content(
            record.text, record.tenant_id, record.user_id
        )
        async with self.session_factory() as session:
            existing = await session.execute(
                select(MemoryRecordModel).where(
                    and_(
                        MemoryRecordModel.content_hash == content_hash,
                        MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                    )
                )
            )
            existing_record = existing.scalar_one_or_none()
            if existing_record:
                existing_record.access_count += 1
                existing_record.last_accessed_at = _naive_utc(datetime.now(timezone.utc))
                existing_record.confidence = max(
                    existing_record.confidence, record.confidence
                )
                await session.commit()
                await session.refresh(existing_record)
                return self._to_schema(existing_record)

            ts = _naive_utc(record.timestamp or datetime.now(timezone.utc))
            now_naive = _naive_utc(datetime.now(timezone.utc))
            model = MemoryRecordModel(
                tenant_id=record.tenant_id,
                user_id=record.user_id,
                agent_id=record.agent_id,
                type=record.type.value,
                text=record.text,
                key=record.key,
                embedding=record.embedding if record.embedding else None,
                entities=[e.model_dump() for e in record.entities],
                relations=[r.model_dump() for r in record.relations],
                meta=record.metadata,
                timestamp=ts,
                written_at=now_naive,
                confidence=record.confidence,
                importance=record.importance,
                provenance=record.provenance.model_dump(),
                content_hash=content_hash,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_schema(model)

    async def get_by_id(self, record_id: UUID) -> Optional[MemoryRecord]:
        async with self.session_factory() as session:
            r = await session.execute(
                select(MemoryRecordModel).where(MemoryRecordModel.id == record_id)
            )
            model = r.scalar_one_or_none()
            return self._to_schema(model) if model else None

    async def get_by_key(
        self, tenant_id: str, user_id: str, key: str
    ) -> Optional[MemoryRecord]:
        async with self.session_factory() as session:
            r = await session.execute(
                select(MemoryRecordModel).where(
                    and_(
                        MemoryRecordModel.tenant_id == tenant_id,
                        MemoryRecordModel.user_id == user_id,
                        MemoryRecordModel.key == key,
                        MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                    )
                )
            )
            model = r.scalar_one_or_none()
            return self._to_schema(model) if model else None

    async def delete(self, record_id: UUID, hard: bool = False) -> bool:
        async with self.session_factory() as session:
            if hard:
                r = await session.execute(
                    delete(MemoryRecordModel).where(
                        MemoryRecordModel.id == record_id
                    )
                )
            else:
                r = await session.execute(
                    update(MemoryRecordModel)
                    .where(MemoryRecordModel.id == record_id)
                    .values(status=MemoryStatus.DELETED.value)
                )
            await session.commit()
            return r.rowcount > 0

    async def update(
        self,
        record_id: UUID,
        patch: Dict[str, Any],
        increment_version: bool = True,
    ) -> Optional[MemoryRecord]:
        async with self.session_factory() as session:
            r = await session.execute(
                select(MemoryRecordModel).where(
                    MemoryRecordModel.id == record_id
                )
            )
            model = r.scalar_one_or_none()
            if not model:
                return None
            key_allow = {
                "access_count", "last_accessed_at", "confidence",
                "importance", "status", "meta", "text", "embedding",
                "valid_to", "metadata", "entities", "relations",
            }
            for key, value in patch.items():
                if key == "metadata":
                    model.meta = value
                elif key in key_allow and hasattr(model, key):
                    setattr(model, key, value)
            if increment_version:
                model.version += 1
            await session.commit()
            await session.refresh(model)
            return self._to_schema(model)

    async def vector_search(
        self,
        tenant_id: str,
        user_id: str,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0,
    ) -> List[MemoryRecord]:
        async with self.session_factory() as session:
            q = select(
                MemoryRecordModel,
                (
                    1
                    - MemoryRecordModel.embedding.cosine_distance(embedding)
                ).label("similarity"),
            ).where(
                and_(
                    MemoryRecordModel.tenant_id == tenant_id,
                    MemoryRecordModel.user_id == user_id,
                    MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                    MemoryRecordModel.embedding.isnot(None),
                )
            )
            if filters:
                if "type" in filters:
                    t = filters["type"]
                    if isinstance(t, list):
                        q = q.where(MemoryRecordModel.type.in_(t))
                    else:
                        q = q.where(MemoryRecordModel.type == t)
                if "since" in filters:
                    q = q.where(
                        MemoryRecordModel.timestamp >= filters["since"]
                    )
                if "until" in filters:
                    q = q.where(
                        MemoryRecordModel.timestamp <= filters["until"]
                    )
            q = q.order_by(
                MemoryRecordModel.embedding.cosine_distance(embedding)
            ).limit(top_k)
            result = await session.execute(q)
            records = []
            for row in result:
                model, similarity = row[0], row[1]
                if similarity >= min_similarity:
                    rec = self._to_schema(model)
                    rec.metadata["_similarity"] = similarity
                    records.append(rec)
            return records

    async def scan(
        self,
        tenant_id: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        async with self.session_factory() as session:
            q = select(MemoryRecordModel).where(
                and_(
                    MemoryRecordModel.tenant_id == tenant_id,
                    MemoryRecordModel.user_id == user_id,
                )
            )
            if filters:
                if "status" in filters:
                    q = q.where(
                        MemoryRecordModel.status == filters["status"]
                    )
                if "type" in filters:
                    t = filters["type"]
                    if isinstance(t, list):
                        q = q.where(MemoryRecordModel.type.in_(t))
                    else:
                        q = q.where(MemoryRecordModel.type == t)
                if "since" in filters:
                    since = _naive_utc(filters["since"])
                    if since is not None:
                        q = q.where(MemoryRecordModel.timestamp >= since)
            if order_by:
                col_name = order_by.lstrip("-")
                col = getattr(MemoryRecordModel, col_name, None)
                if col is not None:
                    q = q.order_by(
                        col.desc() if order_by.startswith("-") else col
                    )
            q = q.offset(offset).limit(limit)
            r = await session.execute(q)
            return [self._to_schema(m) for m in r.scalars().all()]

    async def count(
        self,
        tenant_id: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        async with self.session_factory() as session:
            q = select(func.count(MemoryRecordModel.id)).where(
                and_(
                    MemoryRecordModel.tenant_id == tenant_id,
                    MemoryRecordModel.user_id == user_id,
                )
            )
            if filters and "status" in filters:
                q = q.where(
                    MemoryRecordModel.status == filters["status"]
                )
            r = await session.execute(q)
            return r.scalar() or 0

    async def count_references_to(self, record_id: UUID) -> int:
        """
        Count how many other memory records reference this one (supersedes_id or evidence_refs).
        Used to block delete when dependencies exist.
        """
        record = await self.get_by_id(record_id)
        if not record:
            return 0
        refs = 0
        # Scan same tenant/user; limit to avoid huge scans
        others = await self.scan(
            record.tenant_id,
            record.user_id,
            limit=5000,
        )
        mid_str = str(record_id)
        for r in others:
            if r.id == record_id:
                continue
            if r.supersedes_id == record_id:
                refs += 1
                continue
            if mid_str in (r.metadata or {}).get("evidence_refs", []):
                refs += 1
        return refs

    def _hash_content(self, text: str, tenant_id: str, user_id: str) -> str:
        content = f"{tenant_id}:{user_id}:{text.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _to_schema(self, model: Optional[MemoryRecordModel]) -> Optional[MemoryRecord]:
        if model is None:
            return None
        try:
            mem_type = MemoryType(model.type)
        except ValueError:
            mem_type = MemoryType.EPISODIC_EVENT
        return MemoryRecord(
            id=model.id,
            tenant_id=model.tenant_id,
            user_id=model.user_id,
            agent_id=model.agent_id,
            type=mem_type,
            text=model.text,
            key=model.key,
            embedding=list(model.embedding) if model.embedding is not None else None,
            entities=[
                EntityMention(**e) for e in (model.entities or [])
            ],
            relations=[Relation(**r) for r in (model.relations or [])],
            metadata=model.meta or {},
            timestamp=model.timestamp,
            written_at=model.written_at,
            valid_from=model.valid_from,
            valid_to=model.valid_to,
            confidence=model.confidence,
            importance=model.importance,
            access_count=model.access_count,
            last_accessed_at=model.last_accessed_at,
            decay_rate=model.decay_rate,
            status=MemoryStatus(model.status),
            labile=model.labile,
            provenance=Provenance(**model.provenance),
            version=model.version,
            supersedes_id=model.supersedes_id,
            content_hash=model.content_hash,
        )
