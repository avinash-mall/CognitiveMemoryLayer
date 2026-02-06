"""Event log repository - append-only event store."""

from datetime import datetime
from typing import AsyncIterator, List, Optional
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.enums import OperationType
from ..core.schemas import EventLog
from .models import EventLogModel
from .utils import naive_utc


class EventLogRepository:
    """Append-only event log with replay support."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def append(self, event: EventLog, auto_commit: bool = True) -> EventLog:
        """Append an event to the log. Events are immutable.

        Args:
            event: The event to append.
            auto_commit: If True (default), commit immediately. Set to False
                         to let the caller manage the transaction boundary.
        """
        model = EventLogModel(
            id=event.id,
            tenant_id=event.tenant_id,
            scope_id=event.scope_id,
            agent_id=event.agent_id,
            event_type=event.event_type,
            operation=event.operation.value if event.operation else None,
            payload=event.payload,
            memory_ids=event.memory_ids,
            parent_event_id=event.parent_event_id,
            created_at=naive_utc(event.created_at),
            ip_address=event.ip_address,
            user_agent=event.user_agent,
        )
        self.session.add(model)
        if auto_commit:
            await self.session.commit()
        return event

    async def get_by_id(self, event_id: UUID) -> Optional[EventLog]:
        """Fetch a single event by ID."""
        result = await self.session.execute(
            select(EventLogModel).where(EventLogModel.id == event_id)
        )
        model = result.scalar_one_or_none()
        return self._to_schema(model) if model else None

    async def get_user_events(
        self,
        tenant_id: str,
        user_id: str,
        since: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[EventLog]:
        """List events for a user with optional filters."""
        query = select(EventLogModel).where(
            and_(
                EventLogModel.tenant_id == tenant_id,
                EventLogModel.scope_id == user_id,
            )
        )

        if since:
            query = query.where(EventLogModel.created_at >= naive_utc(since))
        if event_types:
            query = query.where(EventLogModel.event_type.in_(event_types))

        query = query.order_by(EventLogModel.created_at.desc()).limit(limit)

        result = await self.session.execute(query)
        return [self._to_schema(m) for m in result.scalars().all()]

    async def replay_events(
        self,
        tenant_id: str,
        user_id: str,
        from_event_id: Optional[UUID] = None,
    ) -> AsyncIterator[EventLog]:
        """Generator for replaying events (for rebuilding state)."""
        query = (
            select(EventLogModel)
            .where(
                and_(
                    EventLogModel.tenant_id == tenant_id,
                    EventLogModel.scope_id == user_id,
                )
            )
            .order_by(EventLogModel.created_at.asc())
        )

        if from_event_id:
            from_event = await self.get_by_id(from_event_id)
            if from_event:
                since_naive = naive_utc(from_event.created_at)
                if since_naive is not None:
                    query = query.where(EventLogModel.created_at > since_naive)

        result = await self.session.stream(query)
        async for model in result.scalars():
            yield self._to_schema(model)

    @staticmethod
    def _to_schema(model: EventLogModel) -> EventLog:
        """Convert ORM model to EventLog schema."""
        return EventLog(
            id=model.id,
            tenant_id=model.tenant_id,
            scope_id=model.scope_id,
            agent_id=model.agent_id,
            event_type=model.event_type,
            operation=OperationType(model.operation) if model.operation else None,
            payload=model.payload,
            memory_ids=model.memory_ids or [],
            parent_event_id=model.parent_event_id,
            created_at=model.created_at,
            ip_address=model.ip_address,
            user_agent=model.user_agent,
        )
