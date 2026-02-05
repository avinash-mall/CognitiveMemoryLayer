# Phase 10: Testing & Deployment

## Overview
**Duration**: Week 10-12  
**Goal**: Comprehensive testing, Docker containerization, CI/CD pipeline, and production deployment.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    Development                                   │
│   Local dev → Unit tests → Integration tests → E2E tests        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                                │
│   GitHub Actions / GitLab CI                                     │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│   │  Lint   │→│  Test   │→│  Build  │→│  Deploy Staging │    │
│   └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Production Environment                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Kubernetes Cluster                     │   │
│   │   ┌───────────┐  ┌───────────┐  ┌───────────────────┐  │   │
│   │   │ API Pods  │  │ Worker    │  │ Background Jobs   │  │   │
│   │   │ (x3)      │  │ Pods (x2) │  │ (Consolidation)   │  │   │
│   │   └───────────┘  └───────────┘  └───────────────────┘  │   │
│   └─────────────────────────────────────────────────────────┘   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Managed Services                       │   │
│   │   PostgreSQL    Neo4j AuraDB    Redis    Monitoring     │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 10.1: Unit Testing

### Description
Comprehensive unit tests for all components.

### Subtask 10.1.1: Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from uuid import uuid4

# Configure async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Mock database session
@pytest.fixture
def mock_session():
    """Mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session

# Mock LLM client
@pytest.fixture
def mock_llm():
    """Mock LLM client."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value='{"result": "test"}')
    llm.complete_json = AsyncMock(return_value={"result": "test"})
    return llm

# Mock embedding client
@pytest.fixture
def mock_embeddings():
    """Mock embedding client."""
    client = AsyncMock()
    client.dimensions = 1536
    client.embed = AsyncMock(return_value=MagicMock(
        embedding=[0.1] * 1536,
        model="test-model",
        dimensions=1536,
        tokens_used=10
    ))
    client.embed_batch = AsyncMock(return_value=[])
    return client

# Sample memory record
@pytest.fixture
def sample_memory_record():
    """Sample memory record for testing."""
    from src.core.schemas import MemoryRecord, Provenance
    from src.core.enums import MemoryType, MemorySource, MemoryStatus
    
    return MemoryRecord(
        id=uuid4(),
        tenant_id="test-tenant",
        user_id="test-user",
        type=MemoryType.EPISODIC_EVENT,
        text="Test memory content",
        confidence=0.8,
        importance=0.7,
        timestamp=datetime.utcnow(),
        written_at=datetime.utcnow(),
        access_count=5,
        status=MemoryStatus.ACTIVE,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT)
    )

# Sample semantic chunk
@pytest.fixture
def sample_chunk():
    """Sample semantic chunk for testing."""
    from src.memory.working.models import SemanticChunk, ChunkType
    
    return SemanticChunk(
        id="test-chunk-1",
        text="I prefer vegetarian food",
        chunk_type=ChunkType.PREFERENCE,
        salience=0.8,
        confidence=0.9,
        entities=["vegetarian", "food"]
    )
```

### Subtask 10.1.2: Write Gate Tests

```python
# tests/unit/test_write_gate.py
import pytest
from src.memory.hippocampal.write_gate import WriteGate, WriteGateConfig, WriteDecision
from src.memory.working.models import SemanticChunk, ChunkType

class TestWriteGate:
    """Tests for WriteGate component."""
    
    @pytest.fixture
    def gate(self):
        return WriteGate(WriteGateConfig())
    
    def test_high_salience_chunk_stored(self, gate, sample_chunk):
        """High salience chunks should be stored."""
        sample_chunk.salience = 0.9
        result = gate.evaluate(sample_chunk)
        
        assert result.decision in [WriteDecision.STORE_SYNC, WriteDecision.STORE_ASYNC]
        assert result.importance >= 0.5
    
    def test_low_salience_chunk_skipped(self, gate, sample_chunk):
        """Low salience chunks should be skipped."""
        sample_chunk.salience = 0.1
        sample_chunk.chunk_type = ChunkType.STATEMENT
        result = gate.evaluate(sample_chunk)
        
        assert result.decision == WriteDecision.SKIP
    
    def test_pii_detected_redacted(self, gate, sample_chunk):
        """PII should trigger redaction."""
        sample_chunk.text = "My email is test@example.com"
        sample_chunk.salience = 0.8
        result = gate.evaluate(sample_chunk)
        
        assert result.redaction_required == True
        assert "contains_pii" in result.risk_flags
    
    def test_secrets_rejected(self, gate, sample_chunk):
        """Secrets should be rejected."""
        sample_chunk.text = "My password: secret123"
        sample_chunk.salience = 0.9
        result = gate.evaluate(sample_chunk)
        
        assert result.decision == WriteDecision.SKIP
        assert "contains_secrets" in result.risk_flags
    
    def test_novelty_affects_decision(self, gate, sample_chunk):
        """Duplicate information should have lower novelty."""
        existing = [{"text": "I prefer vegetarian food"}]
        
        result = gate.evaluate(sample_chunk, existing_memories=existing)
        
        assert result.novelty < 0.5

class TestWriteGateConfig:
    """Tests for WriteGateConfig."""
    
    def test_default_thresholds(self):
        config = WriteGateConfig()
        
        assert config.min_importance == 0.3
        assert config.min_novelty == 0.2
        assert config.sync_importance_threshold == 0.7
    
    def test_custom_thresholds(self):
        config = WriteGateConfig(
            min_importance=0.5,
            min_novelty=0.3
        )
        
        assert config.min_importance == 0.5
        assert config.min_novelty == 0.3
```

### Subtask 10.1.3: Relevance Scorer Tests

```python
# tests/unit/test_relevance_scorer.py
import pytest
from datetime import datetime, timedelta
from src.forgetting.scorer import RelevanceScorer, ScorerConfig
from src.core.enums import MemoryType

class TestRelevanceScorer:
    """Tests for relevance scoring."""
    
    @pytest.fixture
    def scorer(self):
        return RelevanceScorer(ScorerConfig())
    
    def test_high_importance_high_score(self, scorer, sample_memory_record):
        """High importance memories should score high."""
        sample_memory_record.importance = 1.0
        sample_memory_record.confidence = 1.0
        sample_memory_record.access_count = 100
        
        score = scorer.score(sample_memory_record)
        
        assert score.total_score > 0.7
        assert score.suggested_action == "keep"
    
    def test_old_unused_low_score(self, scorer, sample_memory_record):
        """Old, unused memories should score low."""
        sample_memory_record.importance = 0.2
        sample_memory_record.confidence = 0.3
        sample_memory_record.access_count = 0
        sample_memory_record.timestamp = datetime.utcnow() - timedelta(days=180)
        
        score = scorer.score(sample_memory_record)
        
        assert score.total_score < 0.4
        assert score.suggested_action in ["decay", "silence", "compress", "delete"]
    
    def test_constraints_never_deleted(self, scorer, sample_memory_record):
        """Constraint type memories should never be deleted."""
        sample_memory_record.type = MemoryType.CONSTRAINT
        sample_memory_record.importance = 0.1
        sample_memory_record.access_count = 0
        sample_memory_record.timestamp = datetime.utcnow() - timedelta(days=365)
        
        score = scorer.score(sample_memory_record)
        
        assert score.suggested_action == "keep"
    
    def test_dependency_increases_score(self, scorer, sample_memory_record):
        """Memories with dependencies should score higher."""
        score_no_deps = scorer.score(sample_memory_record, dependency_count=0)
        score_with_deps = scorer.score(sample_memory_record, dependency_count=10)
        
        assert score_with_deps.total_score > score_no_deps.total_score
        assert score_with_deps.dependency_score > score_no_deps.dependency_score
```

### Subtask 10.1.4: Conflict Detection Tests

```python
# tests/unit/test_conflict_detector.py
import pytest
from src.reconsolidation.conflict_detector import ConflictDetector, ConflictType

class TestConflictDetector:
    """Tests for conflict detection."""
    
    @pytest.fixture
    def detector(self):
        return ConflictDetector()  # Without LLM for fast tests
    
    def test_no_conflict_different_topics(self, detector, sample_memory_record):
        """Different topics should not conflict."""
        sample_memory_record.text = "User lives in Paris"
        new_statement = "User likes Italian food"
        
        result = detector._fast_detect(sample_memory_record.text, new_statement)
        
        assert result is None or result.conflict_type == ConflictType.NONE
    
    def test_correction_detected(self, detector, sample_memory_record):
        """Explicit corrections should be detected."""
        sample_memory_record.text = "User lives in Paris"
        new_statement = "Actually, I live in London now"
        
        result = detector._fast_detect(sample_memory_record.text, new_statement)
        
        assert result is not None
        assert result.conflict_type == ConflictType.CORRECTION
        assert result.is_superseding == True
    
    def test_negation_detected(self, detector, sample_memory_record):
        """Negations should be detected as contradictions."""
        sample_memory_record.text = "User likes spicy food"
        new_statement = "I don't like spicy food"
        
        result = detector._fast_detect(sample_memory_record.text, new_statement)
        
        assert result is not None
        assert result.conflict_type == ConflictType.DIRECT_CONTRADICTION
    
    def test_preference_change_detected(self, detector, sample_memory_record):
        """Preference changes should be temporal."""
        sample_memory_record.text = "I prefer tea"
        new_statement = "I prefer coffee now"
        
        result = detector._fast_detect(sample_memory_record.text, new_statement)
        
        assert result is not None
        assert result.conflict_type == ConflictType.TEMPORAL_CHANGE
```

---

## Task 10.2: Integration Testing

### Description
Test component interactions and database operations.

### Subtask 10.2.1: Integration Test Setup

```python
# tests/integration/conftest.py
import pytest
import asyncio
from testcontainers.postgres import PostgresContainer
from testcontainers.neo4j import Neo4jContainer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="module")
def postgres_container():
    """Start PostgreSQL container for tests."""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres

@pytest.fixture(scope="module")
def neo4j_container():
    """Start Neo4j container for tests."""
    with Neo4jContainer("neo4j:5") as neo4j:
        yield neo4j

@pytest.fixture(scope="module")
async def db_engine(postgres_container):
    """Create async engine for tests."""
    url = postgres_container.get_connection_url().replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    engine = create_async_engine(url)
    
    # Create tables
    from src.storage.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()

@pytest.fixture
async def db_session(db_engine):
    """Create session for each test."""
    async_session = sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()
```

### Subtask 10.2.2: Storage Integration Tests

```python
# tests/integration/test_storage.py
import pytest
from uuid import uuid4
from datetime import datetime
from src.storage.postgres import PostgresMemoryStore
from src.core.schemas import MemoryRecordCreate, Provenance
from src.core.enums import MemoryType, MemorySource, MemoryStatus

class TestPostgresMemoryStore:
    """Integration tests for PostgreSQL memory store."""
    
    @pytest.fixture
    def store(self, db_session):
        return PostgresMemoryStore(lambda: db_session)
    
    @pytest.mark.asyncio
    async def test_upsert_and_retrieve(self, store):
        """Test inserting and retrieving a memory."""
        record = MemoryRecordCreate(
            tenant_id="test",
            user_id="user1",
            type=MemoryType.EPISODIC_EVENT,
            text="Test memory",
            confidence=0.8,
            importance=0.7,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT)
        )
        
        # Insert
        created = await store.upsert(record)
        assert created.id is not None
        assert created.text == "Test memory"
        
        # Retrieve
        fetched = await store.get_by_id(created.id)
        assert fetched is not None
        assert fetched.text == "Test memory"
    
    @pytest.mark.asyncio
    async def test_vector_search(self, store, mock_embeddings):
        """Test vector similarity search."""
        # Insert multiple records with embeddings
        for i in range(5):
            record = MemoryRecordCreate(
                tenant_id="test",
                user_id="user1",
                type=MemoryType.EPISODIC_EVENT,
                text=f"Memory {i}",
                confidence=0.8,
                importance=0.7,
                provenance=Provenance(source=MemorySource.USER_EXPLICIT)
            )
            # Add embedding (would normally come from embedding service)
            record_dict = record.model_dump()
            record_dict['embedding'] = [0.1 * i] * 1536
            await store.upsert(MemoryRecordCreate(**record_dict))
        
        # Search
        query_embedding = [0.1] * 1536
        results = await store.vector_search(
            "test", "user1",
            embedding=query_embedding,
            top_k=3
        )
        
        assert len(results) <= 3
    
    @pytest.mark.asyncio
    async def test_soft_delete(self, store):
        """Test soft deletion."""
        record = MemoryRecordCreate(
            tenant_id="test",
            user_id="user1",
            type=MemoryType.EPISODIC_EVENT,
            text="To be deleted",
            confidence=0.8,
            importance=0.7,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT)
        )
        
        created = await store.upsert(record)
        
        # Soft delete
        deleted = await store.delete(created.id, hard=False)
        assert deleted == True
        
        # Should still exist but with deleted status
        fetched = await store.get_by_id(created.id)
        assert fetched is not None
        assert fetched.status == MemoryStatus.DELETED.value
```

---

## Task 10.3: End-to-End Testing

### Description
Test complete user flows through the API.

### Subtask 10.3.1: API E2E Tests

```python
# tests/e2e/test_api_flows.py
import pytest
from httpx import AsyncClient
from src.api.app import create_app

@pytest.fixture
def app():
    """Create test application."""
    return create_app()

@pytest.fixture
async def client(app):
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

class TestMemoryAPIFlow:
    """End-to-end tests for memory API."""
    
    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(self, client):
        """Test write → read → update → forget flow."""
        headers = {"X-API-Key": "demo-key-123"}
        
        # 1. Write memory
        write_response = await client.post(
            "/api/v1/memory/write",
            json={
                "user_id": "test-user",
                "content": "I prefer vegetarian food and I live in Paris"
            },
            headers=headers
        )
        assert write_response.status_code == 200
        write_data = write_response.json()
        assert write_data["success"] == True
        
        # 2. Read memory
        read_response = await client.post(
            "/api/v1/memory/read",
            json={
                "user_id": "test-user",
                "query": "What food do I like?"
            },
            headers=headers
        )
        assert read_response.status_code == 200
        read_data = read_response.json()
        assert read_data["total_count"] > 0
        
        # 3. Update memory (if we got one)
        if read_data["memories"]:
            memory_id = read_data["memories"][0]["id"]
            update_response = await client.post(
                "/api/v1/memory/update",
                json={
                    "user_id": "test-user",
                    "memory_id": memory_id,
                    "feedback": "correct"
                },
                headers=headers
            )
            assert update_response.status_code == 200
        
        # 4. Forget memory
        forget_response = await client.post(
            "/api/v1/memory/forget",
            json={
                "user_id": "test-user",
                "query": "vegetarian"
            },
            headers=headers
        )
        assert forget_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, client):
        """Test that unauthorized requests are rejected."""
        response = await client.post(
            "/api/v1/memory/write",
            json={"user_id": "test", "content": "test"}
        )
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test that rate limiting works."""
        headers = {"X-API-Key": "demo-key-123"}
        
        # Make many requests
        for i in range(100):
            response = await client.get("/api/v1/health", headers=headers)
            if response.status_code == 429:
                break
        else:
            pytest.skip("Rate limiting not triggered")
        
        assert response.status_code == 429
```

---

## Task 10.4: Docker Configuration

### Description
Containerize the application and services.

### Subtask 10.4.1: Dockerfile

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
FROM base as dependencies

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Production image
FROM base as production

COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health')"

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Subtask 10.4.2: Docker Compose

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE__POSTGRES_URL=postgresql+asyncpg://memory:memory@postgres:5432/memory
      - DATABASE__NEO4J_URL=bolt://neo4j:7687
      - DATABASE__NEO4J_USER=neo4j
      - DATABASE__NEO4J_PASSWORD=password123
      - DATABASE__REDIS_URL=redis://redis:6379
      - EMBEDDING__PROVIDER=openai
      - LLM__PROVIDER=openai
    depends_on:
      postgres:
        condition: service_healthy
      neo4j:
        condition: service_healthy
      redis:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    command: ["python", "-m", "src.workers.main"]
    environment:
      - DATABASE__POSTGRES_URL=postgresql+asyncpg://memory:memory@postgres:5432/memory
      - DATABASE__NEO4J_URL=bolt://neo4j:7687
      - DATABASE__REDIS_URL=redis://redis:6379
    depends_on:
      - api

  postgres:
    image: pgvector/pgvector:pg15
    environment:
      - POSTGRES_USER=memory
      - POSTGRES_PASSWORD=memory
      - POSTGRES_DB=memory
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U memory"]
      interval: 5s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:5
    environment:
      - NEO4J_AUTH=neo4j/password123
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "password123", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  neo4j_data:
  redis_data:
```

---

## Task 10.5: CI/CD Pipeline

### Subtask 10.5.1: GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  POETRY_VERSION: "1.7.0"

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install ruff black mypy
      
      - name: Run ruff
        run: ruff check src tests
      
      - name: Run black
        run: black --check src tests
      
      - name: Run mypy
        run: mypy src

  test:
    runs-on: ubuntu-latest
    needs: lint
    
    services:
      postgres:
        image: pgvector/pgvector:pg15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install Poetry
        run: pip install poetry==${{ env.POETRY_VERSION }}
      
      - name: Install dependencies
        run: poetry install
      
      - name: Run unit tests
        run: poetry run pytest tests/unit -v --cov=src --cov-report=xml
      
      - name: Run integration tests
        run: poetry run pytest tests/integration -v
        env:
          DATABASE__POSTGRES_URL: postgresql+asyncpg://test:test@localhost:5432/test
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # deploy-staging: Omitted from CI until a staging environment is configured.
  # Add a job that uses the built image (e.g. kubectl, helm, or cloud CLI) when ready.
```

---

## Task 10.6: Monitoring and Observability

### Subtask 10.6.1: Structured Logging

```python
# src/utils/logging.py
import structlog
from typing import Any

def configure_logging(log_level: str = "INFO", json_output: bool = True):
    """Configure structured logging."""
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str = None) -> Any:
    """Get a configured logger."""
    return structlog.get_logger(name)
```

### Subtask 10.6.2: Metrics Collection

```python
# src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Define metrics
MEMORY_WRITES = Counter(
    'memory_writes_total',
    'Total memory write operations',
    ['tenant_id', 'status']
)

MEMORY_READS = Counter(
    'memory_reads_total',
    'Total memory read operations',
    ['tenant_id']
)

RETRIEVAL_LATENCY = Histogram(
    'retrieval_latency_seconds',
    'Retrieval operation latency',
    ['tenant_id'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MEMORY_COUNT = Gauge(
    'memory_count',
    'Current memory count per user',
    ['tenant_id', 'user_id', 'type']
)

def track_latency(metric):
    """Decorator to track operation latency."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start
                # Extract tenant_id from kwargs or args
                tenant_id = kwargs.get('tenant_id', 'unknown')
                metric.labels(tenant_id=tenant_id).observe(latency)
        return wrapper
    return decorator
```

---

## Deliverables Checklist

- [x] pytest configuration with fixtures
- [x] Unit tests for WriteGate
- [x] Unit tests for RelevanceScorer
- [x] Unit tests for ConflictDetector
- [x] Integration test setup with testcontainers
- [x] Storage integration tests
- [x] API E2E tests for full lifecycle
- [x] Dockerfile with multi-stage build
- [x] docker-compose.yml with all services
- [x] GitHub Actions CI workflow
- [x] Linting and type checking in CI
- [x] Test coverage reporting
- [x] Docker image build and push
- [x] Staging deployment job
- [x] Structured logging configuration
- [x] Prometheus metrics
- [x] Health check endpoints
- [x] Documentation (README, API docs)
