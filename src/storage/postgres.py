"""PostgreSQL memory store with pgvector."""

import hashlib
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select, text, update

from ..core.enums import MemorySource, MemoryStatus, MemoryType
from ..core.schemas import EntityMention, MemoryRecord, MemoryRecordCreate, Provenance, Relation
from ..utils.logging_config import get_logger
from .base import MemoryStoreBase
from .models import MemoryRecordModel
from .utils import naive_utc as _naive_utc

_logger = get_logger(__name__)

_DATETIME_KEYS = frozenset(
    {"last_accessed_at", "timestamp", "written_at", "valid_from", "valid_to"}
)


class PostgresMemoryStore(MemoryStoreBase):
    """PostgreSQL-based memory store with pgvector."""

    def __init__(self, session_factory: Any) -> None:
        self.session_factory = session_factory

    async def upsert(self, record: MemoryRecordCreate) -> MemoryRecord:
        content_hash = self._hash_content(record.text, record.tenant_id)
        async with self.session_factory() as session:
            # BUG-06: Check by key first if provided (stable identity), then by content_hash (deduplication)
            existing_record = None

            if record.key:
                existing_key = await session.execute(
                    select(MemoryRecordModel).where(
                        and_(
                            MemoryRecordModel.tenant_id == record.tenant_id,
                            MemoryRecordModel.key == record.key,
                            MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                        )
                    )
                )
                existing_record = existing_key.scalar_one_or_none()

            if not existing_record:
                existing_hash = await session.execute(
                    select(MemoryRecordModel).where(
                        and_(
                            MemoryRecordModel.tenant_id == record.tenant_id,
                            MemoryRecordModel.content_hash == content_hash,
                            MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                        )
                    )
                )
                existing_record = existing_hash.scalar_one_or_none()

            if existing_record:
                # Update existing record
                existing_record.access_count += 1
                existing_record.last_accessed_at = _naive_utc(datetime.now(UTC))
                existing_record.confidence = max(existing_record.confidence, record.confidence)

                # If we found it by key but content changed, update content & hash
                if (
                    record.key
                    and existing_record.key == record.key
                    and existing_record.content_hash != content_hash
                ):
                    existing_record.text = record.text
                    existing_record.content_hash = content_hash
                    existing_record.embedding = record.embedding
                    # Update other fields that might have changed
                    existing_record.meta = record.metadata

                await session.commit()
                await session.refresh(existing_record)
                return cast("MemoryRecord", self._to_schema(existing_record))

            ts = _naive_utc(record.timestamp or datetime.now(UTC))
            now_naive = _naive_utc(datetime.now(UTC))
            model = MemoryRecordModel(
                tenant_id=record.tenant_id,
                agent_id=record.agent_id,
                context_tags=record.context_tags or [],
                source_session_id=record.source_session_id,
                namespace=record.namespace,
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
                decay_rate=record.decay_rate if record.decay_rate is not None else 0.01,
                provenance=record.provenance.model_dump(),
                content_hash=content_hash,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return cast("MemoryRecord", self._to_schema(model))

    async def get_by_id(self, record_id: UUID) -> MemoryRecord | None:
        async with self.session_factory() as session:
            r = await session.execute(
                select(MemoryRecordModel).where(MemoryRecordModel.id == record_id)
            )
            model = r.scalar_one_or_none()
            return self._to_schema(model) if model else None

    async def get_by_ids_batch(self, record_ids: list[UUID]) -> list[MemoryRecord]:
        """Get multiple records by IDs in a single query."""
        if not record_ids:
            return []
        async with self.session_factory() as session:
            r = await session.execute(
                select(MemoryRecordModel).where(MemoryRecordModel.id.in_(record_ids))
            )
            models = r.scalars().all()
            return [rec for m in models if (rec := self._to_schema(m)) is not None]

    async def get_by_key(
        self,
        tenant_id: str,
        key: str,
        context_filter: list[str] | None = None,
    ) -> MemoryRecord | None:
        async with self.session_factory() as session:
            q = select(MemoryRecordModel).where(
                and_(
                    MemoryRecordModel.tenant_id == tenant_id,
                    MemoryRecordModel.key == key,
                    MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                )
            )
            if context_filter:
                q = q.where(MemoryRecordModel.context_tags.overlap(context_filter))
            r = await session.execute(q)
            model = r.scalar_one_or_none()
            return self._to_schema(model) if model else None

    async def delete(self, record_id: UUID, hard: bool = False) -> bool:
        async with self.session_factory() as session:
            if hard:
                r = await session.execute(
                    delete(MemoryRecordModel).where(MemoryRecordModel.id == record_id)
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
        patch: dict[str, Any],
        increment_version: bool = True,
    ) -> MemoryRecord | None:
        async with self.session_factory() as session:
            r = await session.execute(
                select(MemoryRecordModel).where(MemoryRecordModel.id == record_id)
            )
            model = r.scalar_one_or_none()
            if not model:
                return None
            key_allow = {
                "access_count",
                "last_accessed_at",
                "confidence",
                "importance",
                "status",
                "meta",
                "text",
                "embedding",
                "valid_to",
                "entities",
                "relations",
                "context_tags",
                "source_session_id",
            }
            for key, value in patch.items():
                if key == "metadata":
                    model.meta = value
                elif key in key_allow and hasattr(model, key):
                    if key in _DATETIME_KEYS and isinstance(value, datetime):
                        value = _naive_utc(value)
                    setattr(model, key, value)
            if increment_version:
                model.version += 1
            await session.commit()
            await session.refresh(model)
            return self._to_schema(model)

    async def vector_search(
        self,
        tenant_id: str,
        embedding: list[float],
        top_k: int = 10,
        context_filter: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[MemoryRecord]:
        async with self.session_factory() as session:
            # Phase 6.1: Set HNSW ef_search for this query transaction
            try:
                from ..core.config import get_settings

                settings = get_settings()
                if settings.features.hnsw_ef_search_tuning:
                    ef_search = max(settings.retrieval.hnsw_ef_search, top_k)
                    await session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
            except Exception as exc:
                _logger.debug("hnsw_ef_search_set_failed", error=str(exc))

            base = and_(
                MemoryRecordModel.tenant_id == tenant_id,
                MemoryRecordModel.embedding.isnot(None),
            )
            if context_filter:
                base = and_(base, MemoryRecordModel.context_tags.overlap(context_filter))
            q = select(
                MemoryRecordModel,
                (1 - MemoryRecordModel.embedding.cosine_distance(embedding)).label("similarity"),
            ).where(
                and_(
                    base,
                    MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                )
            )
            if filters:
                if "type" in filters:
                    t = filters["type"]
                    if isinstance(t, list):
                        q = q.where(MemoryRecordModel.type.in_(t))
                    else:
                        q = q.where(MemoryRecordModel.type == t)
                if "source_session_id" in filters:
                    source_session_id = filters["source_session_id"]
                    if isinstance(source_session_id, list):
                        q = q.where(MemoryRecordModel.source_session_id.in_(source_session_id))
                    else:
                        q = q.where(MemoryRecordModel.source_session_id == source_session_id)
                if "since" in filters:
                    since = _naive_utc(filters["since"])
                    if since is not None:
                        q = q.where(MemoryRecordModel.timestamp >= since)
                if "until" in filters:
                    until = _naive_utc(filters["until"])
                    if until is not None:
                        q = q.where(MemoryRecordModel.timestamp <= until)
                if "min_confidence" in filters:
                    q = q.where(MemoryRecordModel.confidence >= filters["min_confidence"])
                if filters.get("exclude_expired"):
                    now = datetime.now(UTC).replace(tzinfo=None)
                    q = q.where(
                        or_(
                            MemoryRecordModel.valid_to.is_(None),
                            MemoryRecordModel.valid_to >= now,
                        )
                    )
            q = q.order_by(MemoryRecordModel.embedding.cosine_distance(embedding)).limit(top_k)
            result = await session.execute(q)
            records = []
            for row in result:
                model, similarity = row[0], row[1]
                if similarity >= min_similarity:
                    rec = self._to_schema(model)
                    if rec is not None:
                        rec.metadata["_similarity"] = similarity
                        records.append(rec)
            return records

    async def scan(
        self,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        async with self.session_factory() as session:
            q = select(MemoryRecordModel).where(MemoryRecordModel.tenant_id == tenant_id)
            if filters and "context_tags" in filters:
                q = q.where(MemoryRecordModel.context_tags.overlap(filters["context_tags"]))
            if filters and "source_session_id" in filters:
                source_session_id = filters["source_session_id"]
                if isinstance(source_session_id, list):
                    q = q.where(MemoryRecordModel.source_session_id.in_(source_session_id))
                else:
                    q = q.where(MemoryRecordModel.source_session_id == source_session_id)
            if filters:
                if "status" in filters:
                    q = q.where(MemoryRecordModel.status == filters["status"])
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
                    q = q.order_by(col.desc() if order_by.startswith("-") else col)
            q = q.offset(offset).limit(limit)
            r = await session.execute(q)
            return [rec for m in r.scalars().all() if (rec := self._to_schema(m)) is not None]

    async def count(
        self,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
    ) -> int:
        async with self.session_factory() as session:
            q = select(func.count(MemoryRecordModel.id)).where(
                MemoryRecordModel.tenant_id == tenant_id
            )
            if filters and "context_tags" in filters:
                q = q.where(MemoryRecordModel.context_tags.overlap(filters["context_tags"]))
            if filters and "source_session_id" in filters:
                source_session_id = filters["source_session_id"]
                if isinstance(source_session_id, list):
                    q = q.where(MemoryRecordModel.source_session_id.in_(source_session_id))
                else:
                    q = q.where(MemoryRecordModel.source_session_id == source_session_id)
            if filters and "status" in filters:
                q = q.where(MemoryRecordModel.status == filters["status"])
            if filters and "type" in filters:
                t = filters["type"]
                if isinstance(t, list):
                    q = q.where(MemoryRecordModel.type.in_(t))
                else:
                    q = q.where(MemoryRecordModel.type == t)
            if filters and "since" in filters:
                since = _naive_utc(filters["since"])
                if since is not None:
                    q = q.where(MemoryRecordModel.timestamp >= since)
            if filters and "until" in filters:
                until = _naive_utc(filters["until"])
                if until is not None:
                    q = q.where(MemoryRecordModel.timestamp <= until)
            r = await session.execute(q)
            return r.scalar() or 0

    async def delete_by_filter(
        self,
        tenant_id: str,
        filters: dict[str, Any],
    ) -> int:
        """Delete records matching filters. Used for efficient ScratchPad clearing (BUG-07)."""
        async with self.session_factory() as session:
            q = delete(MemoryRecordModel).where(MemoryRecordModel.tenant_id == tenant_id)
            if "context_tags" in filters:
                q = q.where(MemoryRecordModel.context_tags.overlap(filters["context_tags"]))
            if "source_session_id" in filters:
                source_session_id = filters["source_session_id"]
                if isinstance(source_session_id, list):
                    q = q.where(MemoryRecordModel.source_session_id.in_(source_session_id))
                else:
                    q = q.where(MemoryRecordModel.source_session_id == source_session_id)
            if "status" in filters:
                q = q.where(MemoryRecordModel.status == filters["status"])
            if "type" in filters:
                t = filters["type"]
                if isinstance(t, list):
                    q = q.where(MemoryRecordModel.type.in_(t))
                else:
                    q = q.where(MemoryRecordModel.type == t)

            has_specific_filter = any(
                k in filters for k in ("context_tags", "source_session_id", "status", "type")
            )
            if not has_specific_filter:
                _logger.warning(
                    "delete_by_filter_blocked_no_specific_filter",
                    extra={"tenant_id": tenant_id, "filters": list(filters.keys())},
                )
                return 0

            r = await session.execute(q)
            await session.commit()
            return r.rowcount

    async def count_references_to(self, record_id: UUID) -> int:
        """
        Count how many other memory records reference this one (supersedes_id or evidence_refs).
        Used to block delete when dependencies exist.
        """
        async with self.session_factory() as session:
            # supersedes_id or metadata.evidence_refs array contains record_id
            refs_contains = text(
                "(metadata::jsonb)->'evidence_refs' IS NOT NULL AND EXISTS ("
                "SELECT 1 FROM jsonb_array_elements_text((metadata::jsonb)->'evidence_refs') AS elem "
                "WHERE elem = :rid)"
            ).bindparams(rid=str(record_id))
            q = select(func.count(MemoryRecordModel.id)).where(
                or_(
                    MemoryRecordModel.supersedes_id == record_id,
                    refs_contains,
                )
            )
            r = await session.execute(q)
            return r.scalar() or 0

    async def increment_access_counts(
        self, record_ids: list[UUID], last_accessed_at: datetime | None = None
    ) -> None:
        """Atomic increment of access_count for given records (BUG-02: avoid lost update)."""
        if not record_ids:
            return
        now = _naive_utc(last_accessed_at or datetime.now(UTC))
        async with self.session_factory() as session:
            await session.execute(
                update(MemoryRecordModel)
                .where(MemoryRecordModel.id.in_(record_ids))
                .values(
                    access_count=MemoryRecordModel.access_count + 1,
                    last_accessed_at=now,
                )
            )
            await session.commit()

    async def deactivate_constraints_by_key(
        self,
        tenant_id: str,
        constraint_key: str,
        superseded_by_key: str | None = None,
    ) -> int:
        """Deactivate episodic CONSTRAINT records with the given fact key.

        When ``superseded_by_key`` is provided, append lightweight lineage
        metadata so retrieval and audits can trace which newer constraint
        superseded the old one.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(MemoryRecordModel).where(
                    and_(
                        MemoryRecordModel.tenant_id == tenant_id,
                        MemoryRecordModel.type == MemoryType.CONSTRAINT.value,
                        MemoryRecordModel.key == constraint_key,
                        MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
                    )
                )
            )
            rows = result.scalars().all()
            if not rows:
                return 0

            now = _naive_utc(datetime.now(UTC))
            assert now is not None
            for row in rows:
                row.status = MemoryStatus.SILENT.value
                row.valid_to = now
                if superseded_by_key:
                    meta = dict(getattr(row, "meta", {}) or {})
                    lineage = meta.get("supersession_lineage", [])
                    if not isinstance(lineage, list):
                        lineage = []
                    lineage.append(
                        {
                            "superseded_by_key": superseded_by_key,
                            "superseded_at": now.isoformat(),
                        }
                    )
                    meta["supersession_lineage"] = lineage[-5:]
                    row.meta = meta
            await session.commit()
            return len(rows)

    def _hash_content(self, text: str, tenant_id: str) -> str:
        content = f"{tenant_id}:{text.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _to_schema(self, model: MemoryRecordModel | None) -> MemoryRecord | None:
        if model is None:
            return None
        try:
            mem_type = MemoryType(cast("str", model.type))
        except ValueError:
            _logger.warning(
                "unknown_memory_type_in_db record_id=%s value=%s",
                model.id,
                model.type,
            )
            mem_type = MemoryType.EPISODIC_EVENT
        try:
            status = MemoryStatus(cast("str", model.status))
        except ValueError:
            _logger.warning(
                "unknown_memory_status_in_db record_id=%s value=%s",
                model.id,
                model.status,
            )
            status = MemoryStatus.ACTIVE
        try:
            provenance = Provenance(**(cast("dict", model.provenance) or {}))
        except (TypeError, ValueError):
            provenance = Provenance(source=MemorySource.AGENT_INFERRED)
        context_tags = getattr(model, "context_tags", None) or []
        source_session_id = getattr(model, "source_session_id", None)
        return MemoryRecord(
            id=cast("UUID", model.id),
            tenant_id=cast("str", model.tenant_id),
            context_tags=list(context_tags),
            source_session_id=source_session_id,
            agent_id=cast("str | None", model.agent_id),
            namespace=getattr(model, "namespace", None),
            type=mem_type,
            text=cast("str", model.text),
            key=cast("str | None", model.key),
            embedding=list(model.embedding) if model.embedding is not None else None,
            entities=[EntityMention(**e) for e in (cast("list", model.entities) or [])],
            relations=[Relation(**r) for r in (cast("list", model.relations) or [])],
            metadata=cast("dict", model.meta) or {},
            timestamp=cast("datetime", model.timestamp),
            written_at=cast("datetime", model.written_at),
            valid_from=cast("datetime | None", model.valid_from),
            valid_to=cast("datetime | None", model.valid_to),
            confidence=cast("float", model.confidence),
            importance=cast("float", model.importance),
            access_count=cast("int", model.access_count),
            last_accessed_at=cast("datetime | None", model.last_accessed_at),
            decay_rate=cast("float", model.decay_rate),
            status=status,
            labile=cast("bool", model.labile),
            provenance=provenance,
            version=cast("int", model.version),
            supersedes_id=cast("UUID | None", model.supersedes_id),
            content_hash=cast("str | None", model.content_hash),
        )

    # ── Phase 4.1: Bulk dependency counts ──────────────────────────

    async def bulk_dependency_counts(
        self,
        tenant_id: str,
        memory_ids: list[str],
    ) -> dict[str, int]:
        """Count references to each memory ID in a single DB query.

        Counts two reference types:
        1. ``supersedes_id`` — direct version chains.
        2. ``metadata.evidence_refs`` — JSON array back-references.

        Returns ``{memory_id: reference_count}``.
        """
        if not memory_ids:
            return {}

        query_str = """
        WITH target_ids AS (
            SELECT unnest(:ids ::text[]) AS target_id
        ),
        supersede_counts AS (
            SELECT
                CAST(supersedes_id AS text) AS target_id,
                COUNT(*) AS cnt
            FROM memory_records
            WHERE tenant_id = :tenant_id
              AND supersedes_id IS NOT NULL
              AND CAST(supersedes_id AS text) = ANY(:ids)
            GROUP BY supersedes_id
        ),
        evidence_counts AS (
            SELECT
                ref.value #>> '{}' AS target_id,
                COUNT(*) AS cnt
            FROM memory_records,
                 jsonb_array_elements(
                     COALESCE(meta -> 'evidence_refs', '[]'::jsonb)
                 ) AS ref(value)
            WHERE tenant_id = :tenant_id
              AND ref.value #>> '{}' = ANY(:ids)
            GROUP BY ref.value #>> '{}'
        )
        SELECT
            t.target_id,
            COALESCE(s.cnt, 0) + COALESCE(e.cnt, 0) AS total_refs
        FROM target_ids t
        LEFT JOIN supersede_counts s ON t.target_id = s.target_id
        LEFT JOIN evidence_counts e ON t.target_id = e.target_id
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(query_str),
                {"tenant_id": tenant_id, "ids": memory_ids},
            )
            counts: dict[str, int] = {}
            for row in result:
                counts[row.target_id] = int(row.total_refs or 0)
            # Ensure every requested ID has an entry
            for mid in memory_ids:
                counts.setdefault(mid, 0)
            return counts
