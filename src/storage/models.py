"""SQLAlchemy models for PostgreSQL (event log and memory records)."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import DeclarativeBase

from pgvector.sqlalchemy import Vector


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
    memory_ids = Column(ARRAY(UUID(as_uuid=True)), default=[])
    parent_event_id = Column(UUID(as_uuid=True), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

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

    scope = Column(String(20), nullable=False, index=True)
    scope_id = Column(String(100), nullable=False, index=True)
    namespace = Column(String(100), nullable=True, index=True)

    type = Column(String(30), nullable=False, index=True)
    text = Column(Text, nullable=False)
    key = Column(String(200), nullable=True, index=True)
    embedding = Column(Vector(1536), nullable=True)

    entities = Column(JSON, default=[])
    relations = Column(JSON, default=[])
    meta = Column("metadata", JSON, default={})  # DB column "metadata"

    timestamp = Column(DateTime, nullable=False, index=True)
    written_at = Column(DateTime, default=datetime.utcnow)
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
        Index("ix_memory_tenant_scope", "tenant_id", "scope", "scope_id", "status"),
        Index("ix_memory_tenant_namespace", "tenant_id", "namespace", "status"),
        Index("ix_memory_tenant_scope_type", "tenant_id", "scope_id", "type"),
        Index("ix_memory_tenant_scope_key", "tenant_id", "scope_id", "key"),
    )


class SemanticFactModel(Base):
    """Semantic facts table for neocortical store."""

    __tablename__ = "semantic_facts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    scope_id = Column(String(100), nullable=False, index=True)
    category = Column(String(30), nullable=False, index=True)
    key = Column(String(200), nullable=False, index=True)
    subject = Column(String(200), nullable=False)
    predicate = Column(String(200), nullable=False)
    value = Column(JSON, nullable=False)
    value_type = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.8)
    evidence_count = Column(Integer, default=1)
    evidence_ids = Column(ARRAY(String), default=[])
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    is_current = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    version = Column(Integer, default=1)
    supersedes_id = Column(UUID(as_uuid=True), nullable=True)

    __table_args__ = (
        Index("ix_semantic_facts_tenant_scope_key", "tenant_id", "scope_id", "key", "is_current"),
        Index("ix_semantic_facts_tenant_scope_category", "tenant_id", "scope_id", "category", "is_current"),
    )
