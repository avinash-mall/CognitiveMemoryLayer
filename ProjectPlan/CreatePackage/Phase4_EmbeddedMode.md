# Phase 4: Embedded Mode

## Objective

Provide an in-process CognitiveMemoryLayer engine that runs entirely within the user's Python process, eliminating the need for a separate CML server, Docker containers, or external infrastructure. This mode is ideal for development, testing, single-user applications, and environments where running a server is impractical.

---

## Task 4.1: Architecture Design

### Sub-Task 4.1.1: Embedded vs Client Architecture

**Architecture**: The embedded mode reuses the CML core engine (`src/`) from the parent CognitiveMemoryLayer project directly within the Python process. It exposes the same API surface as the HTTP client but routes operations through the `MemoryOrchestrator` in-process instead of over HTTP.

```
┌─────────────────────────────────────────────────┐
│  User's Python Application                       │
│                                                   │
│  from cml import EmbeddedCognitiveMemoryLayer     │
│  memory = EmbeddedCognitiveMemoryLayer(...)       │
│       │                                           │
│       ▼                                           │
│  ┌─────────────────────────────────┐              │
│  │  EmbeddedCognitiveMemoryLayer    │             │
│  │    └── MemoryOrchestrator        │             │
│  │         ├── ShortTermMemory      │             │
│  │         ├── HippocampalStore     │             │
│  │         ├── NeocorticalStore     │             │
│  │         ├── MemoryRetriever      │             │
│  │         ├── ReconsolidationSvc   │             │
│  │         ├── ConsolidationWorker  │             │
│  │         └── ForgettingWorker     │             │
│  └─────────────────────────────────┘              │
│       │              │              │              │
│       ▼              ▼              ▼              │
│  PostgreSQL      Neo4j          Redis              │
│  (or SQLite)   (optional)     (optional)           │
└─────────────────────────────────────────────────┘
```

**Design decisions**:
```
1. FULL ENGINE: Re-use the complete CML engine, not a simplified version
2. SAME API: Same method signatures as HTTP client (write, read, turn, etc.)
3. OPTIONAL DEPS: Only installed with `pip install py-cml[embedded]`
4. LIGHTWEIGHT OPTION: Support SQLite+local embeddings for zero-infra dev
5. FULL OPTION: Support PostgreSQL+Neo4j+Redis for production embedded use
```

### Sub-Task 4.1.2: Storage Backend Options

**Architecture**: Support multiple storage configurations for different use cases.

| Mode | PostgreSQL | Neo4j | Redis | Embeddings | Use Case |
|:-----|:-----------|:------|:------|:-----------|:---------|
| **Lite** | SQLite (in-memory or file) | Disabled | Disabled | Local (sentence-transformers) | Dev, testing, prototyping |
| **Standard** | PostgreSQL + pgvector | Disabled | Disabled | OpenAI or local | Single-user apps |
| **Full** | PostgreSQL + pgvector | Neo4j | Redis | OpenAI or local | Production embedded |

**Pseudo-code for mode selection**:
```
FUNCTION select_storage_mode(config):
    IF config.storage_mode == "lite":
        Use SQLite (aiosqlite)
        Disable Neo4j graph store
        Disable Redis cache
        Use local embeddings (sentence-transformers)
    ELIF config.storage_mode == "standard":
        Use PostgreSQL + pgvector
        Disable Neo4j (use fact-only neocortical store)
        Disable Redis
        Use configured embedding provider
    ELIF config.storage_mode == "full":
        Use PostgreSQL + pgvector
        Use Neo4j
        Use Redis
        Use configured embedding provider
```

---

## Task 4.2: Embedded Configuration

### Sub-Task 4.2.1: EmbeddedConfig

**Implementation** (`src/cml/embedded_config.py`):
```python
"""Configuration for embedded CML mode."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class EmbeddedDatabaseConfig(BaseModel):
    """Database configuration for embedded mode."""
    postgres_url: str = Field(
        default="sqlite+aiosqlite:///cml_memory.db",
        description="Database URL. Use sqlite+aiosqlite:// for lite mode."
    )
    neo4j_url: Optional[str] = Field(default=None, description="Neo4j bolt URL")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    redis_url: Optional[str] = Field(default=None, description="Redis URL")


class EmbeddedEmbeddingConfig(BaseModel):
    """Embedding configuration for embedded mode."""
    provider: Literal["openai", "local", "vllm"] = Field(default="local")
    model: str = Field(default="all-MiniLM-L6-v2")
    dimensions: int = Field(default=384)
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)


class EmbeddedLLMConfig(BaseModel):
    """LLM configuration for embedded mode."""
    provider: Literal["openai", "vllm", "ollama", "gemini", "claude"] = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)


class EmbeddedConfig(BaseModel):
    """Full configuration for embedded CognitiveMemoryLayer."""
    storage_mode: Literal["lite", "standard", "full"] = Field(
        default="lite",
        description="Storage backend complexity level"
    )
    tenant_id: str = Field(default="default")
    database: EmbeddedDatabaseConfig = Field(default_factory=EmbeddedDatabaseConfig)
    embedding: EmbeddedEmbeddingConfig = Field(default_factory=EmbeddedEmbeddingConfig)
    llm: EmbeddedLLMConfig = Field(default_factory=EmbeddedLLMConfig)
    auto_consolidate: bool = Field(
        default=False,
        description="Automatically run consolidation periodically"
    )
    auto_forget: bool = Field(
        default=False,
        description="Automatically run active forgetting periodically"
    )
```

**Pseudo-code**:
```
CLASS EmbeddedConfig:
    storage_mode: "lite" | "standard" | "full"
    tenant_id: str

    database:
        postgres_url: str  (SQLite URL for lite, PostgreSQL for standard/full)
        neo4j_url: str?    (None for lite/standard)
        redis_url: str?    (None for lite/standard)

    embedding:
        provider: "local" | "openai" | "vllm"
        model: str
        dimensions: int
        api_key: str?

    llm:
        provider: "openai" | "vllm" | "ollama" | ...
        model: str
        api_key: str?

    auto_consolidate: bool  (run consolidation in background thread)
    auto_forget: bool       (run forgetting in background thread)
```

---

## Task 4.3: SQLite Storage Adapter

### Sub-Task 4.3.1: SQLite Memory Store

**Architecture**: Create a lightweight SQLite-based memory store that implements the same interface as `PostgresMemoryStore` but uses SQLite with in-memory vector similarity (no pgvector extension needed).

**Implementation** (`src/cml/storage/sqlite_store.py`):
```python
"""SQLite-based memory store for lite embedded mode."""

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiosqlite

from ..models.memory import MemoryRecord


class SQLiteMemoryStore:
    """Memory store backed by SQLite.

    Uses in-memory cosine similarity for vector search
    (suitable for up to ~10,000 records).
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create tables and indexes."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                text TEXT NOT NULL,
                key TEXT,
                namespace TEXT,
                embedding TEXT,  -- JSON array of floats
                entities TEXT,   -- JSON array
                relations TEXT,  -- JSON array
                metadata TEXT,   -- JSON object
                context_tags TEXT, -- JSON array
                confidence REAL DEFAULT 0.5,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                decay_rate REAL DEFAULT 0.01,
                timestamp TEXT,
                written_at TEXT,
                content_hash TEXT,
                version INTEGER DEFAULT 1,
                provenance TEXT  -- JSON object
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tenant ON memories(tenant_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_status ON memories(tenant_id, status)"
        )
        await self._db.commit()

    async def store(self, record: MemoryRecord) -> MemoryRecord:
        """Store a memory record."""
        # ... serialize and INSERT

    async def get_by_id(self, memory_id: UUID) -> Optional[MemoryRecord]:
        """Retrieve a single record by ID."""
        # ... SELECT by id, deserialize

    async def vector_search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[MemoryRecord]:
        """Search by vector similarity using in-memory cosine similarity."""
        # ... Load all embeddings for tenant, compute cosine similarity, sort, limit

    async def scan(
        self,
        tenant_id: str,
        filters: Optional[Dict] = None,
        limit: int = 100,
        order_by: str = "-timestamp",
    ) -> List[MemoryRecord]:
        """Scan records with filters."""
        # ... SELECT with WHERE clauses

    async def update(self, memory_id: UUID, patch: Dict) -> Optional[MemoryRecord]:
        """Update a record."""
        # ... UPDATE with patch fields

    async def delete(self, memory_id: UUID, hard: bool = False) -> None:
        """Delete or soft-delete a record."""
        # ... DELETE or UPDATE status

    async def count(self, tenant_id: str, filters: Optional[Dict] = None) -> int:
        """Count records matching filters."""
        # ... SELECT COUNT(*)
```

**Pseudo-code for in-memory vector search**:
```
METHOD vector_search(tenant_id, query_embedding, limit):
    1. SELECT all rows WHERE tenant_id = ? AND status = 'active'
    2. FOR EACH row:
       a. Parse embedding from JSON
       b. Compute cosine_similarity(query_embedding, row.embedding)
       c. Store (record, similarity) pair
    3. Sort by similarity DESC
    4. Return top `limit` records

FUNCTION cosine_similarity(a, b):
    dot_product = SUM(a[i] * b[i] for i in range(len(a)))
    norm_a = SQRT(SUM(x^2 for x in a))
    norm_b = SQRT(SUM(x^2 for x in b))
    IF norm_a == 0 OR norm_b == 0: RETURN 0.0
    RETURN dot_product / (norm_a * norm_b)
```

---

## Task 4.4: Embedded Client Implementation

### Sub-Task 4.4.1: EmbeddedCognitiveMemoryLayer Class

**Architecture**: Wraps the `MemoryOrchestrator` from the CML engine, providing the same API surface as the HTTP clients.

**Implementation** (`src/cml/embedded.py`):
```python
"""Embedded CognitiveMemoryLayer — runs in-process without a server."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from .embedded_config import EmbeddedConfig
from .models.responses import (
    ForgetResponse,
    ReadResponse,
    StatsResponse,
    TurnResponse,
    UpdateResponse,
    WriteResponse,
    MemoryItem,
)
from .models.enums import MemoryType


class EmbeddedCognitiveMemoryLayer:
    """In-process CognitiveMemoryLayer engine.

    Runs the full CML engine within your Python process.
    No server, no Docker, no HTTP overhead.

    Install with: pip install py-cml[embedded]

    Usage:
        async with EmbeddedCognitiveMemoryLayer() as memory:
            await memory.write("User prefers vegetarian food.")
            result = await memory.read("dietary preferences")

    Storage modes:
        - "lite": SQLite + local embeddings (zero infrastructure)
        - "standard": PostgreSQL + pgvector (requires PostgreSQL)
        - "full": PostgreSQL + Neo4j + Redis (full feature set)
    """

    def __init__(
        self,
        *,
        config: Optional[EmbeddedConfig] = None,
        storage_mode: str = "lite",
        tenant_id: str = "default",
        db_path: Optional[str] = None,
        embedding_provider: str = "local",
        llm_api_key: Optional[str] = None,
    ):
        if config:
            self._config = config
        else:
            self._config = EmbeddedConfig(
                storage_mode=storage_mode,
                tenant_id=tenant_id,
                # ... map other params to config
            )
        self._orchestrator = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize storage backends and the memory orchestrator."""
        self._check_embedded_deps()
        # 1. Initialize storage based on storage_mode
        # 2. Initialize embedding client
        # 3. Initialize LLM client
        # 4. Create MemoryOrchestrator with all dependencies
        # 5. Run schema migrations if needed
        self._initialized = True

    def _check_embedded_deps(self) -> None:
        """Verify embedded dependencies are installed."""
        try:
            import sqlalchemy
            import asyncpg  # or aiosqlite for lite mode
        except ImportError:
            raise ImportError(
                "Embedded mode requires additional dependencies. "
                "Install with: pip install py-cml[embedded]"
            )

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Shutdown storage connections and background workers."""
        # Close database connections
        # Stop background consolidation/forgetting threads
        self._initialized = False

    # --- Memory Operations (same signature as HTTP client) ---

    async def write(self, content: str, **kwargs) -> WriteResponse:
        """Store new information in memory."""
        self._ensure_initialized()
        result = await self._orchestrator.write(
            tenant_id=self._config.tenant_id,
            content=content,
            **kwargs,
        )
        return WriteResponse(
            success=True,
            memory_id=result.get("memory_id"),
            chunks_created=result.get("chunks_created", 0),
            message=result.get("message", ""),
        )

    async def read(self, query: str, **kwargs) -> ReadResponse:
        """Retrieve relevant memories."""
        self._ensure_initialized()
        packet = await self._orchestrator.read(
            tenant_id=self._config.tenant_id,
            query=query,
            **kwargs,
        )
        # Convert MemoryPacket → ReadResponse
        return self._packet_to_read_response(query, packet)

    async def turn(self, user_message: str, **kwargs) -> TurnResponse:
        """Process a conversation turn with seamless memory."""
        self._ensure_initialized()
        # Use SeamlessMemoryProvider internally
        # ... implementation details

    async def update(self, memory_id: UUID, **kwargs) -> UpdateResponse:
        """Update an existing memory."""
        self._ensure_initialized()
        result = await self._orchestrator.update(
            tenant_id=self._config.tenant_id,
            memory_id=memory_id,
            **kwargs,
        )
        return UpdateResponse(
            success=True,
            memory_id=memory_id,
            version=result.get("version", 1),
        )

    async def forget(self, **kwargs) -> ForgetResponse:
        """Forget memories."""
        self._ensure_initialized()
        result = await self._orchestrator.forget(
            tenant_id=self._config.tenant_id,
            **kwargs,
        )
        return ForgetResponse(
            success=True,
            affected_count=result.get("affected_count", 0),
        )

    async def stats(self) -> StatsResponse:
        """Get memory statistics."""
        self._ensure_initialized()
        result = await self._orchestrator.get_stats(
            tenant_id=self._config.tenant_id,
        )
        return StatsResponse(**result)

    async def consolidate(self) -> Dict[str, Any]:
        """Manually trigger memory consolidation."""
        self._ensure_initialized()
        return await self._orchestrator.consolidation.run(
            tenant_id=self._config.tenant_id,
        )

    async def run_forgetting(self, *, dry_run: bool = True) -> Dict[str, Any]:
        """Manually trigger active forgetting."""
        self._ensure_initialized()
        return await self._orchestrator.forgetting.run(
            tenant_id=self._config.tenant_id,
            dry_run=dry_run,
        )

    # --- Internal Helpers ---

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "EmbeddedCognitiveMemoryLayer not initialized. "
                "Use `async with` or call `await memory.initialize()` first."
            )

    def _packet_to_read_response(self, query, packet) -> ReadResponse:
        """Convert internal MemoryPacket to client ReadResponse."""
        # Map packet.facts, packet.preferences, etc. to MemoryItem lists
        # Generate llm_context string
        # Return ReadResponse
```

**Pseudo-code**:
```
CLASS EmbeddedCognitiveMemoryLayer:

    INIT(config or individual params):
        Build EmbeddedConfig
        Set orchestrator = None
        Set initialized = False

    ASYNC initialize():
        1. Check embedded dependencies are installed
        2. SWITCH storage_mode:
           "lite":
              Create SQLiteMemoryStore
              Create local EmbeddingClient (sentence-transformers)
              Create LLM client (for extraction, chunking)
           "standard":
              Create PostgresMemoryStore (asyncpg)
              Run alembic migrations
              Create configured EmbeddingClient
              Create LLM client
           "full":
              Create PostgresMemoryStore
              Create Neo4jGraphStore
              Create RedisClient
              Run migrations
              Create EmbeddingClient
              Create LLM client
        3. Assemble MemoryOrchestrator with all components
        4. IF auto_consolidate → start background consolidation
        5. IF auto_forget → start background forgetting
        6. Set initialized = True

    ASYNC write(content, **kwargs) -> WriteResponse:
        Ensure initialized
        result = orchestrator.write(tenant_id, content, **kwargs)
        RETURN WriteResponse from result dict

    ASYNC read(query, **kwargs) -> ReadResponse:
        Ensure initialized
        packet = orchestrator.read(tenant_id, query, **kwargs)
        Convert MemoryPacket → ReadResponse
        RETURN ReadResponse

    ASYNC turn(user_message, **kwargs) -> TurnResponse:
        Ensure initialized
        Use SeamlessMemoryProvider.process_turn()
        RETURN TurnResponse

    ASYNC close():
        Close all database connections
        Stop background workers
        Set initialized = False
```

---

## Task 4.5: Background Workers (Optional)

### Sub-Task 4.5.1: In-Process Consolidation Worker

**Architecture**: Run consolidation periodically in a background asyncio task instead of using Celery.

**Pseudo-code**:
```
CLASS BackgroundConsolidation:
    INIT(orchestrator, interval_hours=24):
        self.orchestrator = orchestrator
        self.interval = interval_hours * 3600
        self.task = None

    ASYNC start():
        self.task = asyncio.create_task(self._run_loop())

    ASYNC _run_loop():
        WHILE True:
            TRY:
                await asyncio.sleep(self.interval)
                await self.orchestrator.consolidation.run(tenant_id)
                LOG "Consolidation completed"
            CATCH Exception as e:
                LOG "Consolidation failed: {e}"

    ASYNC stop():
        IF self.task:
            self.task.cancel()
            TRY:
                await self.task
            CATCH asyncio.CancelledError:
                pass
```

### Sub-Task 4.5.2: In-Process Forgetting Worker

**Pseudo-code**:
```
CLASS BackgroundForgetting:
    INIT(orchestrator, interval_hours=24):
        Same pattern as BackgroundConsolidation
        Uses orchestrator.forgetting.run()
```

---

## Task 4.6: Lite Mode Shortcuts

### Sub-Task 4.6.1: Zero-Config Quick Start

**Architecture**: Allow creating an embedded instance with zero configuration for instant prototyping.

**Implementation**:
```python
# Absolute minimum setup:
from cml import EmbeddedCognitiveMemoryLayer

async def main():
    async with EmbeddedCognitiveMemoryLayer() as memory:
        await memory.write("User prefers vegetarian food.")
        result = await memory.read("dietary preferences")
        print(result.context)

# This uses:
# - SQLite in-memory database (lost on exit)
# - Local sentence-transformers embeddings
# - No Neo4j, no Redis
# - No API keys needed (for embeddings)
```

### Sub-Task 4.6.2: Persistent Lite Mode

**Implementation**:
```python
# Persistent storage to file:
from cml import EmbeddedCognitiveMemoryLayer

memory = EmbeddedCognitiveMemoryLayer(
    db_path="./my_memories.db",   # SQLite file
    tenant_id="my-app",
)
await memory.initialize()

# Data persists between restarts
await memory.write("User's birthday is March 15th")
```

---

## Task 4.7: Embedded-to-Server Migration

### Sub-Task 4.7.1: Export/Import Utilities

**Architecture**: Provide utilities to export memories from embedded mode and import them into a CML server (or vice versa).

**Pseudo-code**:
```python
async def export_memories(
    source: EmbeddedCognitiveMemoryLayer,
    output_path: str,
    format: Literal["json", "jsonl"] = "jsonl",
) -> int:
    """Export all memories to a file."""
    records = await source._orchestrator.hippocampal.store.scan(
        tenant_id=source._config.tenant_id,
        limit=100000,
    )
    with open(output_path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    return len(records)


async def import_memories(
    target: CognitiveMemoryLayer | EmbeddedCognitiveMemoryLayer,
    input_path: str,
) -> int:
    """Import memories from a file."""
    count = 0
    with open(input_path) as f:
        for line in f:
            record = MemoryRecord.model_validate_json(line)
            await target.write(record.text, metadata=record.metadata)
            count += 1
    return count
```

---

## Acceptance Criteria

- [ ] `pip install py-cml[embedded]` installs all required dependencies
- [ ] `EmbeddedCognitiveMemoryLayer()` works with zero configuration (lite mode)
- [ ] Lite mode uses SQLite + local embeddings (no external services)
- [ ] Standard mode connects to PostgreSQL with pgvector
- [ ] Full mode uses PostgreSQL + Neo4j + Redis
- [ ] Same API surface as HTTP client (write, read, turn, update, forget, stats)
- [ ] Context manager protocol (`async with`) handles init/teardown
- [ ] Background consolidation/forgetting works via asyncio tasks
- [ ] Export/import utilities support data migration
- [ ] Missing `[embedded]` dependencies produce clear error message
- [ ] Embedded mode passes the same functional tests as HTTP client
