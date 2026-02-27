"""Semantic fact store with versioning and schema alignment."""

import contextlib
import json
from datetime import UTC, datetime
from typing import Any
from typing import cast as typing_cast
from uuid import UUID, uuid4

from sqlalchemy import String, and_, cast, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ...storage.models import SemanticFactModel
from ...storage.utils import naive_utc
from .schemas import DEFAULT_FACT_SCHEMAS, FactCategory, FactSchema, SemanticFact


class SemanticFactStore:
    """
    Stores and manages semantic facts.
    Handles versioning, temporal validity, and schema alignment.
    """

    def __init__(self, session_factory: Any, schemas: dict[str, FactSchema] | None = None):
        self.session_factory = session_factory
        self.schemas = schemas or DEFAULT_FACT_SCHEMAS

    async def upsert_fact(
        self,
        tenant_id: str,
        key: str,
        value: Any,
        confidence: float = 0.8,
        evidence_ids: list[str] | None = None,
        valid_from: datetime | None = None,
        context_tags: list[str] | None = None,
    ) -> SemanticFact:
        """Insert or update a semantic fact. Holistic: tenant-only."""
        async with self.session_factory() as session:
            category, predicate = self._parse_key(key)
            schema = self._get_schema(key)
            existing = await self._get_existing_fact(session, tenant_id, key)

            if existing:
                return await self._update_fact(
                    session, existing, value, confidence, evidence_ids, schema, valid_from
                )
            return await self._create_fact(
                session,
                tenant_id,
                key,
                category,
                predicate,
                value,
                confidence,
                evidence_ids,
                valid_from,
                context_tags or [],
            )

    async def get_fact(
        self,
        tenant_id: str,
        key: str,
        include_historical: bool = False,
    ) -> SemanticFact | None:
        """Get a fact by key. Holistic: tenant-only."""
        async with self.session_factory() as session:
            q = select(SemanticFactModel).where(
                and_(
                    SemanticFactModel.tenant_id == tenant_id,
                    SemanticFactModel.key == key,
                )
            )
            if not include_historical:
                q = q.where(SemanticFactModel.is_current.is_(True))
            q = q.order_by(SemanticFactModel.version.desc()).limit(1)
            result = await session.execute(q)
            model = result.scalar_one_or_none()
            return self._model_to_fact(model) if model else None

    async def get_facts_by_category(
        self,
        tenant_id: str,
        category: FactCategory,
        current_only: bool = True,
        limit: int = 50,
    ) -> list[SemanticFact]:
        """Get facts in a category. Holistic: tenant-only."""
        async with self.session_factory() as session:
            q = select(SemanticFactModel).where(
                and_(
                    SemanticFactModel.tenant_id == tenant_id,
                    SemanticFactModel.category == category.value,
                )
            )
            if current_only:
                q = q.where(SemanticFactModel.is_current.is_(True))
                # Exclude expired facts: valid_to is null or in the future
                now = datetime.now(UTC).replace(tzinfo=None)
                q = q.where(
                    or_(
                        SemanticFactModel.valid_to.is_(None),
                        SemanticFactModel.valid_to >= now,
                    )
                )
            q = q.limit(limit)
            result = await session.execute(q)
            rows = result.scalars().all()
            return [self._model_to_fact(r) for r in rows]

    async def get_facts_by_categories(
        self,
        tenant_id: str,
        categories: list[FactCategory],
        current_only: bool = True,
        limit: int = 200,
    ) -> list[SemanticFact]:
        """Get facts across multiple categories in a single query.

        Uses ``WHERE category IN (...)`` to avoid one query per category on
        the hot retrieval path.
        """
        if not categories:
            return []

        category_values = list(dict.fromkeys(c.value for c in categories))
        async with self.session_factory() as session:
            q = select(SemanticFactModel).where(
                and_(
                    SemanticFactModel.tenant_id == tenant_id,
                    SemanticFactModel.category.in_(category_values),
                )
            )
            if current_only:
                q = q.where(SemanticFactModel.is_current.is_(True))
                now = datetime.now(UTC).replace(tzinfo=None)
                q = q.where(
                    or_(
                        SemanticFactModel.valid_to.is_(None),
                        SemanticFactModel.valid_to >= now,
                    )
                )
            q = q.limit(limit)
            result = await session.execute(q)
            rows = result.scalars().all()
            return [self._model_to_fact(r) for r in rows]

    async def get_tenant_profile(self, tenant_id: str) -> dict[str, Any]:
        """Get complete profile as structured dict by category. Holistic: tenant-only."""
        profile: dict[str, Any] = {}
        for category in FactCategory:
            facts = await self.get_facts_by_category(tenant_id, category)
            if facts:
                profile[category.value] = {f.predicate: f.value for f in facts}
        return profile

    async def search_facts(
        self,
        tenant_id: str,
        query: str,
        limit: int = 20,
    ) -> list[SemanticFact]:
        """Search facts by text (key, subject, value). Holistic: tenant-only."""
        async with self.session_factory() as session:
            q = (
                select(SemanticFactModel)
                .where(
                    and_(
                        SemanticFactModel.tenant_id == tenant_id,
                        SemanticFactModel.is_current.is_(True),
                        (
                            SemanticFactModel.key.ilike(f"%{query}%")
                            | SemanticFactModel.subject.ilike(f"%{query}%")
                            | cast(SemanticFactModel.value, String).ilike(f"%{query}%")
                        ),
                    )
                )
                .order_by(SemanticFactModel.confidence.desc(), SemanticFactModel.updated_at.desc())
                .limit(limit)
            )
            result = await session.execute(q)
            rows = result.scalars().all()
            return [self._model_to_fact(r) for r in rows]

    async def search_facts_batch(
        self,
        tenant_id: str,
        entity_names: list[str],
        limit_per_entity: int = 5,
    ) -> dict[str, list[SemanticFact]]:
        """Fetch facts for multiple entities in one query (exact subject match).

        Uses ``subject IN (entity_names)`` instead of the ILIKE text search
        that :meth:`search_facts` performs.  This is more precise and faster
        for the multi-hop use case where we want facts *about* specific
        PPR-ranked entities.

        Returns a mapping ``entity_name -> list[SemanticFact]``, at most
        *limit_per_entity* per entity, ordered by confidence descending.
        Entities with no matching facts are absent from the dict.
        """
        if not entity_names:
            return {}

        async with self.session_factory() as session:
            q = (
                select(SemanticFactModel)
                .where(
                    and_(
                        SemanticFactModel.tenant_id == tenant_id,
                        SemanticFactModel.is_current.is_(True),
                        SemanticFactModel.subject.in_(entity_names),
                    )
                )
                .order_by(
                    SemanticFactModel.confidence.desc(),
                    SemanticFactModel.updated_at.desc(),
                )
                .limit(len(entity_names) * limit_per_entity)
            )
            result = await session.execute(q)
            rows = result.scalars().all()

        # Group by subject and cap per entity
        grouped: dict[str, list[SemanticFact]] = {}
        for row in rows:
            fact = self._model_to_fact(row)
            bucket = grouped.setdefault(fact.subject, [])
            if len(bucket) < limit_per_entity:
                bucket.append(fact)
        return grouped

    async def invalidate_fact(
        self,
        tenant_id: str,
        key: str,
        reason: str = "superseded",
    ) -> bool:
        """Mark a fact as no longer current. Holistic: tenant-only."""
        async with self.session_factory() as session:
            now = datetime.now(UTC)
            stmt = (
                update(SemanticFactModel)
                .where(
                    and_(
                        SemanticFactModel.tenant_id == tenant_id,
                        SemanticFactModel.key == key,
                        SemanticFactModel.is_current.is_(True),
                    )
                )
                .values(is_current=False, valid_to=now)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def _get_existing_fact(
        self,
        session: AsyncSession,
        tenant_id: str,
        key: str,
    ) -> SemanticFactModel | None:
        """Get existing current fact as the ORM model (avoids double-fetch, LOW-10)."""
        q = select(SemanticFactModel).where(
            and_(
                SemanticFactModel.tenant_id == tenant_id,
                SemanticFactModel.key == key,
                SemanticFactModel.is_current.is_(True),
            )
        )
        result = await session.execute(q)
        return result.scalar_one_or_none()

    async def _update_fact(
        self,
        session: AsyncSession,
        existing_model: SemanticFactModel,
        new_value: Any,
        confidence: float,
        evidence_ids: list[str] | None,
        schema: FactSchema | None,
        valid_from: datetime | None,
    ) -> SemanticFact:
        """Update existing fact (reinforce or new version).

        Receives the ORM model directly to avoid a redundant DB round-trip (LOW-10).
        """
        model = existing_model
        existing = self._model_to_fact(model)

        if existing.value == new_value:
            setattr(
                model,
                "confidence",
                min(1.0, typing_cast("float", model.confidence) + 0.1),
            )
            setattr(
                model,
                "evidence_count",
                typing_cast("int", model.evidence_count) + 1,
            )
            setattr(
                model,
                "evidence_ids",
                list(typing_cast("list", model.evidence_ids) or []) + (evidence_ids or []),
            )
            setattr(model, "updated_at", naive_utc(datetime.now(UTC)))
            await session.commit()
            await session.refresh(model)
            return self._model_to_fact(model)
        else:
            # Always supersede old fact when value changes (at most one current per key)
            setattr(model, "is_current", False)
            setattr(model, "valid_to", naive_utc(valid_from or datetime.now(UTC)))
            await session.flush()
            value_type = "str" if new_value is None else type(new_value).__name__.lower()
            new_fact = SemanticFact(
                id=str(uuid4()),
                tenant_id=existing.tenant_id,
                context_tags=existing.context_tags,
                category=existing.category,
                key=existing.key,
                subject=existing.subject,
                predicate=existing.predicate,
                value=new_value,
                value_type=value_type,
                confidence=confidence,
                evidence_count=1,
                evidence_ids=evidence_ids or [],
                valid_from=naive_utc(valid_from or datetime.now(UTC)),
                is_current=True,
                version=existing.version + 1,
                supersedes_id=existing.id,
            )
            await self._insert_fact(session, new_fact)
            return new_fact

    async def _create_fact(
        self,
        session: AsyncSession,
        tenant_id: str,
        key: str,
        category: FactCategory,
        predicate: str,
        value: Any,
        confidence: float,
        evidence_ids: list[str] | None,
        valid_from: datetime | None,
        context_tags: list[str],
    ) -> SemanticFact:
        """Create new fact."""
        value_type = "str" if value is None else type(value).__name__.lower()
        fact = SemanticFact(
            id=str(uuid4()),
            tenant_id=tenant_id,
            context_tags=context_tags,
            category=category,
            key=key,
            subject="user",
            predicate=predicate,
            value=value,
            value_type=value_type,
            confidence=confidence,
            evidence_count=1,
            evidence_ids=evidence_ids or [],
            valid_from=naive_utc(valid_from or datetime.now(UTC)),
            is_current=True,
            version=1,
        )
        await self._insert_fact(session, fact)
        return fact

    async def _insert_fact(self, session: AsyncSession, fact: SemanticFact) -> None:
        """Insert fact row."""
        payload = fact.value
        if not isinstance(payload, (str, int, float, bool, type(None))):
            payload = json.loads(json.dumps(payload, default=str))
        model = SemanticFactModel(
            id=UUID(fact.id),
            tenant_id=fact.tenant_id,
            context_tags=fact.context_tags or [],
            category=fact.category.value,
            key=fact.key,
            subject=fact.subject,
            predicate=fact.predicate,
            value=payload,
            value_type=fact.value_type,
            confidence=fact.confidence,
            evidence_count=fact.evidence_count,
            evidence_ids=fact.evidence_ids,
            valid_from=naive_utc(fact.valid_from),
            valid_to=naive_utc(fact.valid_to),
            is_current=fact.is_current,
            created_at=naive_utc(fact.created_at),
            updated_at=naive_utc(fact.updated_at),
            version=fact.version,
            supersedes_id=UUID(fact.supersedes_id) if fact.supersedes_id else None,
        )
        session.add(model)
        await session.commit()

    def _parse_key(self, key: str) -> tuple:
        """Parse key into category and predicate."""
        parts = key.split(":")
        if len(parts) >= 3:
            try:
                category = FactCategory(parts[1])
            except ValueError:
                category = FactCategory.CUSTOM
            predicate = ":".join(parts[2:])
        else:
            category = FactCategory.CUSTOM
            predicate = key
        return category, predicate

    def _get_schema(self, key: str) -> FactSchema | None:
        """Get schema for a key."""
        if key in self.schemas:
            return self.schemas[key]
        for pattern, schema in self.schemas.items():
            if "*" in pattern:
                prefix = pattern.split("*")[0]
                if key.startswith(prefix):
                    return schema
        return None

    def _model_to_fact(self, model: SemanticFactModel) -> SemanticFact:
        """Convert ORM model to SemanticFact."""
        val = model.value
        if isinstance(val, str) and val.strip().startswith(("{", "[")):
            with contextlib.suppress(json.JSONDecodeError):
                val = json.loads(val)
        context_tags = getattr(model, "context_tags", None) or []
        return SemanticFact(
            id=str(model.id),
            tenant_id=typing_cast("str", model.tenant_id),
            context_tags=list(context_tags),
            category=FactCategory(typing_cast("str", model.category)),
            key=typing_cast("str", model.key),
            subject=typing_cast("str", model.subject),
            predicate=typing_cast("str", model.predicate),
            value=val,
            value_type=typing_cast("str", model.value_type),
            confidence=typing_cast("float", model.confidence),
            evidence_count=typing_cast("int", model.evidence_count),
            evidence_ids=list(typing_cast("list", model.evidence_ids) or []),
            valid_from=typing_cast("datetime | None", model.valid_from),
            valid_to=typing_cast("datetime | None", model.valid_to),
            is_current=typing_cast("bool", model.is_current),
            created_at=typing_cast("datetime", model.created_at),
            updated_at=typing_cast("datetime", model.updated_at),
            version=typing_cast("int", model.version),
            supersedes_id=str(model.supersedes_id) if model.supersedes_id else None,
        )
