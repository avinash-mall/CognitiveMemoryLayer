"""SQLAlchemy models for PostgreSQL (event log and memory records)."""

import uuid
from datetime import UTC, datetime
from typing import Any, cast

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import DeclarativeBase

from ..core.config import get_embedding_dimensions

# Embedding vector dimension driven by EMBEDDING_INTERNAL__DIMENSIONS in .env / config.
_EMBEDDING_DIM = get_embedding_dimensions()


class Base(DeclarativeBase):
    """Declarative base for all storage models."""

    pass


class MemoryRecordModel(Base):
    """Memory records table with vector embedding support."""

    __tablename__ = "memory_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    agent_id = Column(String(100), nullable=True)

    context_tags = cast("Any", Column(ARRAY(String), default=list, nullable=False))
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
    # When the *described event* happened, as opposed to when the turn was written.
    # Mirrors metadata['event_date'], which is still what the packet renderer reads.
    event_date = Column(DateTime, nullable=True)
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
        Index("ix_memory_tenant_event_date", "tenant_id", "event_date"),
    )


class SemanticFactModel(Base):
    """Semantic facts table for neocortical store."""

    __tablename__ = "semantic_facts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    context_tags = cast("Any", Column(ARRAY(String), default=list, nullable=False))
    category = Column(String(30), nullable=False, index=True)
    key = Column(String(200), nullable=False, index=True)
    subject = Column(String(200), nullable=False)
    predicate = Column(String(200), nullable=False)
    value = Column(JSON, nullable=False)
    value_type = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.8)
    evidence_count = Column(Integer, default=1)
    evidence_ids = cast("Any", Column(ARRAY(String), default=list))
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
    job_type = Column(
        String(30), nullable=False, index=True
    )  # consolidate / forget / reconsolidate
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


class HyperedgeEngramModel(Base):
    """N-ary Cortical Hypergraph Engrams binding actors, event, time, place, and cues."""

    __tablename__ = "hyperedge_engrams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    engram_key = Column(String(200), nullable=False, index=True)
    actors = Column(JSON, default=list, nullable=False)
    entities = Column(JSON, default=list, nullable=False)
    event_summary = Column(Text, nullable=False)
    temporal_anchor = Column(String(120), nullable=True)
    location = Column(String(200), nullable=True)
    causal_state = Column(Text, nullable=True)
    prospective_cues = Column(JSON, default=list, nullable=False)
    sensory_descriptors = Column(JSON, default=list, nullable=False)
    episode_ids = Column(JSON, default=list, nullable=False)
    confidence = Column(Float, default=0.82)
    synaptic_weight = Column(Float, default=1.0)
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

    __table_args__ = (Index("ix_hyperedge_tenant_key", "tenant_id", "engram_key"),)


class DMNSelfProfileModel(Base):
    """Conway Tier-3 persistent DMN Conceptual-Self profile per speaker/subject."""

    __tablename__ = "dmn_self_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    subject = Column(String(150), nullable=False, index=True)
    profile_json = Column(JSON, default=dict, nullable=False)
    evidence_count = Column(Integer, default=1)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
    )

    __table_args__ = (Index("ix_dmn_self_tenant_subject", "tenant_id", "subject", unique=True),)


class ProceduralSkillModel(Base):
    """Basal Ganglia / Cerebellar Procedural Skill Playbooks reinforced via Dopamine RPE."""

    __tablename__ = "procedural_skills"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    skill_key = Column(String(200), nullable=False, index=True)
    trigger_pattern = Column(Text, nullable=False)
    steps = Column(JSON, default=list, nullable=False)
    pitfalls = Column(JSON, default=list, nullable=False)
    success_count = Column(Integer, default=1)
    failure_count = Column(Integer, default=0)
    synaptic_weight = Column(Float, default=1.0)
    embedding_json = Column(JSON, nullable=True)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
    )

    __table_args__ = (
        Index("ix_procedural_skills_tenant_key", "tenant_id", "skill_key", unique=True),
    )
