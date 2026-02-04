# Phase 1: Foundation & Core Data Models

## Overview
**Duration**: Week 1-2  
**Goal**: Establish project structure, core data models, event logging, and storage abstractions.

## Architecture Decisions

- **Event Sourcing**: All state changes recorded in append-only log
- **CQRS Pattern**: Separate write models from read views
- **Repository Pattern**: Abstract storage backends
- **Dependency Injection**: Configurable components

---

## Task 1.1: Project Setup & Structure

### Description
Initialize the Python project with proper structure, dependencies, and configuration management.

### Subtask 1.1.1: Initialize Python Project

```bash
# Create project with poetry
poetry init --name cognitive-memory-layer --python "^3.11"

# Core dependencies
poetry add fastapi uvicorn[standard] pydantic sqlalchemy[asyncio] asyncpg
poetry add pgvector neo4j redis celery
poetry add openai sentence-transformers tiktoken
poetry add python-dotenv pydantic-settings structlog

# Dev dependencies
poetry add --group dev pytest pytest-asyncio pytest-cov black ruff mypy
poetry add --group dev httpx factory-boy faker
```

### Subtask 1.1.2: Directory Structure Creation

```python
# scripts/init_structure.py
import os
from pathlib import Path

STRUCTURE = {
    "src": {
        "api": ["__init__.py", "routes.py", "dependencies.py", "middleware.py"],
        "core": ["__init__.py", "models.py", "schemas.py", "enums.py", "exceptions.py"],
        "memory": {
            "sensory": ["__init__.py", "buffer.py"],
            "working": ["__init__.py", "manager.py", "chunker.py"],
            "hippocampal": ["__init__.py", "store.py", "encoder.py"],
            "neocortical": ["__init__.py", "store.py", "schema_manager.py"],
            "__init__.py": None,
            "orchestrator.py": None,
        },
        "retrieval": ["__init__.py", "planner.py", "retriever.py", "reranker.py"],
        "consolidation": ["__init__.py", "worker.py", "clusterer.py", "summarizer.py"],
        "forgetting": ["__init__.py", "scorer.py", "worker.py"],
        "extraction": ["__init__.py", "entity_extractor.py", "fact_extractor.py"],
        "storage": ["__init__.py", "postgres.py", "neo4j.py", "redis.py", "base.py"],
        "utils": ["__init__.py", "embeddings.py", "llm.py", "timing.py"],
        "__init__.py": None,
    },
    "tests": {
        "unit": ["__init__.py"],
        "integration": ["__init__.py"],
        "conftest.py": None,
    },
    "config": ["settings.yaml", "logging.yaml"],
    "migrations": ["__init__.py"],
    "docker": ["Dockerfile", "docker-compose.yml"],
    "docs": [],
}

def create_structure(base_path: Path, structure: dict):
    for name, content in structure.items():
        path = base_path / name
        if isinstance(content, dict):
            path.mkdir(exist_ok=True)
            create_structure(path, content)
        elif isinstance(content, list):
            path.mkdir(exist_ok=True)
            for file in content:
                (path / file).touch()
        elif content is None:
            path.touch()

if __name__ == "__main__":
    create_structure(Path("."), STRUCTURE)
```

### Subtask 1.1.3: Configuration Management

```python
# src/core/config.py
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache

class DatabaseSettings(BaseSettings):
    postgres_url: str = Field(default="postgresql+asyncpg://localhost/memory")
    neo4j_url: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")
    redis_url: str = Field(default="redis://localhost:6379")

class EmbeddingSettings(BaseSettings):
    provider: str = Field(default="openai")  # openai | local
    model: str = Field(default="text-embedding-3-small")
    dimensions: int = Field(default=1536)
    local_model: str = Field(default="all-MiniLM-L6-v2")

class LLMSettings(BaseSettings):
    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0)

class MemorySettings(BaseSettings):
    sensory_buffer_max_tokens: int = Field(default=500)
    sensory_buffer_decay_seconds: float = Field(default=30.0)
    working_memory_max_chunks: int = Field(default=10)
    write_gate_threshold: float = Field(default=0.3)
    consolidation_interval_hours: int = Field(default=6)
    forgetting_interval_hours: int = Field(default=24)

class Settings(BaseSettings):
    app_name: str = Field(default="CognitiveMemoryLayer")
    debug: bool = Field(default=False)
    
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## Task 1.2: Core Data Models

### Description
Define the fundamental data structures for memory records, events, and operations.

### Subtask 1.2.1: Memory Type Enums

```python
# src/core/enums.py
from enum import Enum

class MemoryType(str, Enum):
    EPISODIC_EVENT = "episodic_event"
    SEMANTIC_FACT = "semantic_fact"
    PREFERENCE = "preference"
    TASK_STATE = "task_state"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    HYPOTHESIS = "hypothesis"

class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SILENT = "silent"       # Hard to retrieve, needs strong cue
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    DELETED = "deleted"

class MemorySource(str, Enum):
    USER_EXPLICIT = "user_explicit"      # User directly stated
    USER_CONFIRMED = "user_confirmed"    # User confirmed inference
    AGENT_INFERRED = "agent_inferred"    # Agent extracted/inferred
    TOOL_RESULT = "tool_result"          # From tool execution
    CONSOLIDATION = "consolidation"      # From consolidation process
    RECONSOLIDATION = "reconsolidation"  # Updated after retrieval

class OperationType(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"
    REINFORCE = "reinforce"
    DECAY = "decay"
    SILENCE = "silence"
    COMPRESS = "compress"
```

### Subtask 1.2.2: Core Memory Record Schema

```python
# src/core/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from .enums import MemoryType, MemoryStatus, MemorySource

class Provenance(BaseModel):
    """Tracks origin and evidence for a memory."""
    source: MemorySource
    evidence_refs: List[str] = Field(default_factory=list)  # Event IDs, turn IDs
    tool_refs: List[str] = Field(default_factory=list)      # Tool call IDs
    model_version: Optional[str] = None                      # Extraction model
    extraction_prompt_hash: Optional[str] = None             # For reproducibility

class EntityMention(BaseModel):
    """An entity mentioned in the memory."""
    text: str
    normalized: str          # Canonical form
    entity_type: str         # PERSON, LOCATION, ORG, DATE, etc.
    start_char: Optional[int] = None
    end_char: Optional[int] = None

class Relation(BaseModel):
    """A relation triple extracted from memory."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0

class MemoryRecord(BaseModel):
    """Core memory record stored in the system."""
    model_config = ConfigDict(from_attributes=True)
    
    # Identity
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    user_id: str
    agent_id: Optional[str] = None
    
    # Type and content
    type: MemoryType
    text: str                                    # Human-readable content
    key: Optional[str] = None                    # Unique key for facts (e.g., "user:location")
    embedding: Optional[List[float]] = None      # Dense vector
    
    # Structured extractions
    entities: List[EntityMention] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Temporal validity
    timestamp: datetime = Field(default_factory=datetime.utcnow)  # When event occurred
    written_at: datetime = Field(default_factory=datetime.utcnow) # When stored
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    
    # Scoring
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Usage tracking
    access_count: int = Field(default=0)
    last_accessed_at: Optional[datetime] = None
    decay_rate: float = Field(default=0.01)  # Per day
    
    # Status
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    labile: bool = Field(default=False)  # Currently being reconsolidated
    
    # Provenance
    provenance: Provenance
    
    # Versioning
    version: int = Field(default=1)
    supersedes_id: Optional[UUID] = None  # Previous version
    
    # Deduplication
    content_hash: Optional[str] = None
    
class MemoryRecordCreate(BaseModel):
    """Schema for creating a new memory."""
    tenant_id: str
    user_id: str
    agent_id: Optional[str] = None
    type: MemoryType
    text: str
    key: Optional[str] = None
    entities: List[EntityMention] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None
    confidence: float = 0.5
    importance: float = 0.5
    provenance: Provenance
```

### Subtask 1.2.3: Event Log Schema

```python
# src/core/schemas.py (continued)

class EventLog(BaseModel):
    """Immutable event log entry."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    user_id: str
    agent_id: Optional[str] = None
    
    event_type: str          # "turn", "memory_op", "consolidation", etc.
    operation: Optional[OperationType] = None
    
    # Content
    payload: Dict[str, Any]  # Full turn data or operation details
    
    # References
    memory_ids: List[UUID] = Field(default_factory=list)  # Affected memories
    parent_event_id: Optional[UUID] = None                 # For chaining
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Audit
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class MemoryOperation(BaseModel):
    """Planned operation on a memory."""
    op: OperationType
    record_id: Optional[UUID] = None      # For UPDATE/DELETE
    record: Optional[MemoryRecordCreate] = None  # For ADD
    patch: Optional[Dict[str, Any]] = None       # For UPDATE
    reason: str = ""
    confidence: float = 1.0
```

### Subtask 1.2.4: Memory Packet (Retrieval Response)

```python
# src/core/schemas.py (continued)

class RetrievedMemory(BaseModel):
    """A single retrieved memory with score."""
    record: MemoryRecord
    relevance_score: float
    retrieval_source: str  # "vector", "graph", "lexical", "cache"

class MemoryPacket(BaseModel):
    """Structured bundle returned from retrieval."""
    query: str
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Categorized memories
    facts: List[RetrievedMemory] = Field(default_factory=list)
    recent_episodes: List[RetrievedMemory] = Field(default_factory=list)
    preferences: List[RetrievedMemory] = Field(default_factory=list)
    procedures: List[RetrievedMemory] = Field(default_factory=list)
    constraints: List[RetrievedMemory] = Field(default_factory=list)
    
    # Meta
    open_questions: List[str] = Field(default_factory=list)  # Needs confirmation
    warnings: List[str] = Field(default_factory=list)        # Conflicts detected
    
    @property
    def all_memories(self) -> List[RetrievedMemory]:
        return (self.facts + self.recent_episodes + self.preferences + 
                self.procedures + self.constraints)
    
    def to_context_string(self, max_chars: int = 4000) -> str:
        """Format for LLM context injection."""
        lines = []
        for category, memories in [
            ("Facts", self.facts),
            ("Preferences", self.preferences),
            ("Recent Events", self.recent_episodes),
            ("Procedures", self.procedures),
            ("Constraints", self.constraints),
        ]:
            if memories:
                lines.append(f"## {category}")
                for m in memories[:5]:  # Limit per category
                    lines.append(f"- {m.record.text} (confidence: {m.record.confidence:.2f})")
        
        result = "\n".join(lines)
        return result[:max_chars]
```

---

## Task 1.3: Event Log Implementation

### Description
Implement the append-only event log for audit trail and replay capability.

### Subtask 1.3.1: SQLAlchemy Models

```python
# src/storage/models.py
from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Boolean, Text
from sqlalchemy import ForeignKey, Index, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass

class EventLogModel(Base):
    __tablename__ = "event_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
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
        Index('ix_event_log_tenant_user_time', 'tenant_id', 'user_id', 'created_at'),
    )

class MemoryRecordModel(Base):
    __tablename__ = "memory_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    agent_id = Column(String(100), nullable=True)
    
    type = Column(String(30), nullable=False, index=True)
    text = Column(Text, nullable=False)
    key = Column(String(200), nullable=True, index=True)  # For keyed lookups
    embedding = Column(Vector(1536), nullable=True)       # pgvector
    
    entities = Column(JSON, default=[])
    relations = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    
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
        Index('ix_memory_tenant_user_status', 'tenant_id', 'user_id', 'status'),
        Index('ix_memory_tenant_user_type', 'tenant_id', 'user_id', 'type'),
        Index('ix_memory_tenant_user_key', 'tenant_id', 'user_id', 'key'),
    )
```

### Subtask 1.3.2: Event Log Repository

```python
# src/storage/event_log.py
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from .models import EventLogModel
from ..core.schemas import EventLog

class EventLogRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def append(self, event: EventLog) -> EventLog:
        """Append an event to the log. Events are immutable."""
        model = EventLogModel(
            id=event.id,
            tenant_id=event.tenant_id,
            user_id=event.user_id,
            agent_id=event.agent_id,
            event_type=event.event_type,
            operation=event.operation.value if event.operation else None,
            payload=event.payload,
            memory_ids=event.memory_ids,
            parent_event_id=event.parent_event_id,
            created_at=event.created_at,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
        )
        self.session.add(model)
        await self.session.commit()
        return event
    
    async def get_by_id(self, event_id: UUID) -> Optional[EventLog]:
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
        limit: int = 100
    ) -> List[EventLog]:
        query = select(EventLogModel).where(
            and_(
                EventLogModel.tenant_id == tenant_id,
                EventLogModel.user_id == user_id
            )
        )
        
        if since:
            query = query.where(EventLogModel.created_at >= since)
        if event_types:
            query = query.where(EventLogModel.event_type.in_(event_types))
        
        query = query.order_by(EventLogModel.created_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return [self._to_schema(m) for m in result.scalars().all()]
    
    async def replay_events(
        self,
        tenant_id: str,
        user_id: str,
        from_event_id: Optional[UUID] = None
    ):
        """Generator for replaying events (for rebuilding state)."""
        query = select(EventLogModel).where(
            and_(
                EventLogModel.tenant_id == tenant_id,
                EventLogModel.user_id == user_id
            )
        ).order_by(EventLogModel.created_at.asc())
        
        if from_event_id:
            # Get timestamp of the from_event
            from_event = await self.get_by_id(from_event_id)
            if from_event:
                query = query.where(EventLogModel.created_at > from_event.created_at)
        
        result = await self.session.stream(query)
        async for model in result.scalars():
            yield self._to_schema(model)
    
    def _to_schema(self, model: EventLogModel) -> EventLog:
        from ..core.enums import OperationType
        return EventLog(
            id=model.id,
            tenant_id=model.tenant_id,
            user_id=model.user_id,
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
```

---

## Task 1.4: Storage Abstraction Layer

### Description
Create abstract interfaces for storage backends to enable swapping implementations.

### Subtask 1.4.1: Base Repository Interface

```python
# src/storage/base.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, AsyncIterator
from uuid import UUID
from datetime import datetime
from ..core.schemas import MemoryRecord, MemoryRecordCreate

class MemoryStoreBase(ABC):
    """Abstract base for memory storage backends."""
    
    @abstractmethod
    async def upsert(self, record: MemoryRecordCreate) -> MemoryRecord:
        """Insert or update a memory record."""
        pass
    
    @abstractmethod
    async def get_by_id(self, record_id: UUID) -> Optional[MemoryRecord]:
        """Get a single record by ID."""
        pass
    
    @abstractmethod
    async def get_by_key(
        self, 
        tenant_id: str, 
        user_id: str, 
        key: str
    ) -> Optional[MemoryRecord]:
        """Get a record by its unique key (for facts/preferences)."""
        pass
    
    @abstractmethod
    async def delete(self, record_id: UUID, hard: bool = False) -> bool:
        """Delete (soft or hard) a record."""
        pass
    
    @abstractmethod
    async def update(
        self, 
        record_id: UUID, 
        patch: Dict[str, Any],
        increment_version: bool = True
    ) -> Optional[MemoryRecord]:
        """Partial update with optimistic locking."""
        pass
    
    @abstractmethod
    async def vector_search(
        self,
        tenant_id: str,
        user_id: str,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[MemoryRecord]:
        """Search by vector similarity."""
        pass
    
    @abstractmethod
    async def scan(
        self,
        tenant_id: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryRecord]:
        """Scan records with filters."""
        pass
    
    @abstractmethod
    async def count(
        self,
        tenant_id: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count records matching filters."""
        pass


class GraphStoreBase(ABC):
    """Abstract base for knowledge graph storage."""
    
    @abstractmethod
    async def merge_node(
        self,
        tenant_id: str,
        user_id: str,
        entity: str,
        entity_type: str,
        properties: Dict[str, Any] = None
    ) -> str:
        """Create or update a node, return node ID."""
        pass
    
    @abstractmethod
    async def merge_edge(
        self,
        tenant_id: str,
        user_id: str,
        subject: str,
        predicate: str,
        object: str,
        properties: Dict[str, Any] = None
    ) -> str:
        """Create or update an edge."""
        pass
    
    @abstractmethod
    async def get_neighbors(
        self,
        tenant_id: str,
        user_id: str,
        entity: str,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes up to max_depth."""
        pass
    
    @abstractmethod
    async def personalized_pagerank(
        self,
        tenant_id: str,
        user_id: str,
        seed_entities: List[str],
        top_k: int = 20,
        damping: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Run PPR from seed entities."""
        pass
```

### Subtask 1.4.2: Database Connection Manager

```python
# src/storage/connection.py
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from neo4j import AsyncGraphDatabase
import redis.asyncio as redis
from ..core.config import get_settings

class DatabaseManager:
    _instance = None
    
    def __init__(self):
        settings = get_settings()
        
        # PostgreSQL
        self.pg_engine = create_async_engine(
            settings.database.postgres_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self.pg_session_factory = async_sessionmaker(
            self.pg_engine, 
            expire_on_commit=False
        )
        
        # Neo4j
        self.neo4j_driver = AsyncGraphDatabase.driver(
            settings.database.neo4j_url,
            auth=(settings.database.neo4j_user, settings.database.neo4j_password)
        )
        
        # Redis
        self.redis = redis.from_url(settings.database.redis_url)
    
    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @asynccontextmanager
    async def pg_session(self) -> AsyncGenerator[AsyncSession, None]:
        session = self.pg_session_factory()
        try:
            yield session
        finally:
            await session.close()
    
    @asynccontextmanager
    async def neo4j_session(self):
        session = self.neo4j_driver.session()
        try:
            yield session
        finally:
            await session.close()
    
    async def close(self):
        await self.pg_engine.dispose()
        await self.neo4j_driver.close()
        await self.redis.close()
```

---

## Task 1.5: Database Migrations

### Description
Set up Alembic for database migrations.

### Subtask 1.5.1: Alembic Configuration

```python
# migrations/env.py
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from src.storage.models import Base
from src.core.config import get_settings

config = context.config
settings = get_settings()

config.set_main_option("sqlalchemy.url", settings.database.postgres_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()

def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Subtask 1.5.2: Initial Migration Script

```python
# migrations/versions/001_initial.py
"""Initial schema

Revision ID: 001
Create Date: 2026-02-03
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSON
from pgvector.sqlalchemy import Vector

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Event log table
    op.create_table(
        'event_log',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('user_id', sa.String(100), nullable=False),
        sa.Column('agent_id', sa.String(100), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('operation', sa.String(20), nullable=True),
        sa.Column('payload', JSON, nullable=False),
        sa.Column('memory_ids', ARRAY(UUID(as_uuid=True)), default=[]),
        sa.Column('parent_event_id', UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('ip_address', sa.String(50), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
    )
    op.create_index('ix_event_log_tenant_user_time', 'event_log', 
                    ['tenant_id', 'user_id', 'created_at'])
    
    # Memory records table
    op.create_table(
        'memory_records',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('user_id', sa.String(100), nullable=False),
        sa.Column('agent_id', sa.String(100), nullable=True),
        sa.Column('type', sa.String(30), nullable=False),
        sa.Column('text', sa.Text, nullable=False),
        sa.Column('key', sa.String(200), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('entities', JSON, default=[]),
        sa.Column('relations', JSON, default=[]),
        sa.Column('metadata', JSON, default={}),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('written_at', sa.DateTime, nullable=False),
        sa.Column('valid_from', sa.DateTime, nullable=True),
        sa.Column('valid_to', sa.DateTime, nullable=True),
        sa.Column('confidence', sa.Float, default=0.5),
        sa.Column('importance', sa.Float, default=0.5),
        sa.Column('access_count', sa.Integer, default=0),
        sa.Column('last_accessed_at', sa.DateTime, nullable=True),
        sa.Column('decay_rate', sa.Float, default=0.01),
        sa.Column('status', sa.String(20), default='active'),
        sa.Column('labile', sa.Boolean, default=False),
        sa.Column('provenance', JSON, nullable=False),
        sa.Column('version', sa.Integer, default=1),
        sa.Column('supersedes_id', UUID(as_uuid=True), nullable=True),
        sa.Column('content_hash', sa.String(64), nullable=True),
    )
    
    # Indexes for memory_records
    op.create_index('ix_memory_tenant_user_status', 'memory_records',
                    ['tenant_id', 'user_id', 'status'])
    op.create_index('ix_memory_tenant_user_type', 'memory_records',
                    ['tenant_id', 'user_id', 'type'])
    op.create_index('ix_memory_tenant_user_key', 'memory_records',
                    ['tenant_id', 'user_id', 'key'])
    op.create_index('ix_memory_content_hash', 'memory_records', ['content_hash'])
    
    # Vector index for similarity search
    op.execute('''
        CREATE INDEX ix_memory_embedding_hnsw 
        ON memory_records 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    ''')

def downgrade() -> None:
    op.drop_table('memory_records')
    op.drop_table('event_log')
```

---

## Deliverables Checklist

- [x] Poetry project initialized with all dependencies
- [x] Directory structure created
- [x] Configuration management with pydantic-settings
- [x] Core enums defined (MemoryType, MemoryStatus, etc.)
- [x] MemoryRecord schema with all fields
- [x] EventLog schema for audit trail
- [x] MemoryPacket schema for retrieval responses
- [x] SQLAlchemy models for PostgreSQL
- [x] EventLogRepository implementation
- [x] Abstract base classes for storage backends
- [x] Database connection manager
- [x] Alembic migrations configured
- [x] Initial migration with all tables and indexes
