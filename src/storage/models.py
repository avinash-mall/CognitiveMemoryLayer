"""SQLAlchemy models for PostgreSQL (event log and memory records)."""

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import DeclarativeBase

from ..core.config import get_settings

# Embedding vector dimension driven by EMBEDDING__DIMENSIONS in .env / config.
_EMBEDDING_DIM = get_settings().embedding.dimensions


class Base(DeclarativeBase):
    """Declarative base for all storage models."""

    pass


class EventLogModel(Base):
    """Event log table - append-only audit trail."""

    __tablename__ = "event_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    scope_id = Column(String(100), nullable=False, index=True)
    agent_id = Column(String(100), nullable=True)

    event_type = Column(String(50), nullable=False, index=True)
    operation = Column(String(20), nullable=True)

    payload = Column(JSON, nullable=False)
    memory_ids = Column(ARRAY(UUID(as_uuid=True)), default=list)
    parent_event_id = Column(UUID(as_uuid=True), nullable=True)

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
        index=True,
    )

    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_event_log_tenant_scope_time", "tenant_id", "scope_id", "created_at"),
    )


class MemoryRecordModel(Base):
    """Memory records table with vector embedding support."""

    __tablename__ = "memory_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    agent_id = Column(String(100), nullable=True)

    context_tags = Column(ARRAY(String), default=list, nullable=False)
    source_session_id = Column(String(100), nullable=True)
    namespace = Column(String(100), nullable=True, index=True)

    type = Column(String(30), nullable=False, index=True)
    text = Column(Text, nullable=False)
    key = Column(String(200), nullable=True, index=True)
    embedding = Column(Vector(_EMBEDDING_DIM), nullable=True)

    entities = Column(JSON, default=list)
    relations = Column(JSON, default=list)
    meta = Column("metadata", JSON, default=dict)  # DB column "metadata"

    timestamp = Column(DateTime, nullable=False, index=True)
    written_at = Column(DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None))
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)

    confidence = Column(Float, default=0.5)
    importance = Column(Float, default=0.5)

    access_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime, nullable=True)
    decay_rate = Column(Float, default=0.01)

    status = Column(String(20), default="active", index=True)
    labile = Column(Boolean, default=False)

    provenance = Column(JSON, nullable=False)

    version = Column(Integer, default=1)
    supersedes_id = Column(UUID(as_uuid=True), nullable=True)
    content_hash = Column(String(64), nullable=True, index=True)

    __table_args__ = (
        Index("ix_memory_tenant_namespace", "tenant_id", "namespace", "status"),
        Index("ix_memory_tenant_status", "tenant_id", "status"),
        Index("ix_memory_tenant_type", "tenant_id", "type"),
        Index("ix_memory_tenant_key", "tenant_id", "key"),
    )


class SemanticFactModel(Base):
    """Semantic facts table for neocortical store."""

    __tablename__ = "semantic_facts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    context_tags = Column(ARRAY(String), default=list, nullable=False)
    category = Column(String(30), nullable=False, index=True)
    key = Column(String(200), nullable=False, index=True)
    subject = Column(String(200), nullable=False)
    predicate = Column(String(200), nullable=False)
    value = Column(JSON, nullable=False)
    value_type = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.8)
    evidence_count = Column(Integer, default=1)
    evidence_ids = Column(ARRAY(String), default=list)
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    is_current = Column(Boolean, default=True, index=True)
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
    )
    version = Column(Integer, default=1)
    supersedes_id = Column(UUID(as_uuid=True), nullable=True)

    __table_args__ = (
        Index("ix_semantic_facts_tenant_key", "tenant_id", "key", "is_current"),
        Index("ix_semantic_facts_tenant_category", "tenant_id", "category", "is_current"),
    )


class DashboardJobModel(Base):
    """Dashboard job history for consolidation/forgetting runs."""

    __tablename__ = "dashboard_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(30), nullable=False, index=True)  # consolidate / forget
    tenant_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=True)
    dry_run = Column(Boolean, default=False)
    status = Column(
        String(20), nullable=False, default="running", index=True
    )  # running / completed / failed
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    started_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
    )
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (Index("ix_dashboard_jobs_tenant_type", "tenant_id", "job_type"),)
