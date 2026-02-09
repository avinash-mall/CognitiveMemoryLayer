# Phase 7: Testing & Quality

## Objective

Build a comprehensive test suite covering unit tests (with mocked HTTP), integration tests (against a live CML server), and end-to-end tests for both sync/async clients and embedded mode. Establish CI quality gates for linting, type checking, and code coverage.

---

## Task 7.1: Test Infrastructure

### Sub-Task 7.1.1: Test Directory Structure

**Implementation**:
```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── unit/                          # Tests with mocked HTTP (fast, no server)
│   ├── __init__.py
│   ├── test_config.py             # Configuration loading & validation
│   ├── test_exceptions.py         # Exception hierarchy & mapping
│   ├── test_transport.py          # HTTP transport with mocked httpx
│   ├── test_retry.py              # Retry logic
│   ├── test_async_client.py       # Async client (all operations)
│   ├── test_sync_client.py        # Sync client (all operations)
│   ├── test_models.py             # Pydantic model serialization
│   ├── test_serialization.py      # Serialization utilities
│   └── test_logging.py            # Logging configuration
├── integration/                   # Tests against a running CML server
│   ├── __init__.py
│   ├── conftest.py                # Server connection fixtures
│   ├── test_write_read.py         # Write and read roundtrip
│   ├── test_turn.py               # Seamless turn processing
│   ├── test_update_forget.py      # Update and forget operations
│   ├── test_stats.py              # Statistics endpoint
│   ├── test_sessions.py           # Session management
│   ├── test_batch.py              # Batch operations
│   ├── test_admin.py              # Admin operations
│   └── test_namespace.py          # Namespace isolation
├── embedded/                      # Tests for embedded mode
│   ├── __init__.py
│   ├── conftest.py                # Embedded fixtures
│   ├── test_lite_mode.py          # SQLite + local embeddings
│   ├── test_standard_mode.py      # PostgreSQL mode
│   └── test_lifecycle.py          # Init, close, context manager
└── e2e/                           # End-to-end scenarios
    ├── __init__.py
    ├── test_chat_flow.py           # Full chat with memory flow
    └── test_migration.py           # Embedded → server migration
```

### Sub-Task 7.1.2: Shared Fixtures

**Implementation** (`tests/conftest.py`):
```python
"""Shared test fixtures for py-cml."""

import os
import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig


# --- Configuration Fixtures ---

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return CMLConfig(
        api_key="test-api-key",
        base_url="http://localhost:8000",
        tenant_id="test-tenant",
        timeout=10.0,
        max_retries=0,  # No retries in tests
    )


@pytest.fixture
def mock_config():
    """Config for unit tests (no real server)."""
    return CMLConfig(
        api_key="mock-key",
        base_url="http://mock-server:8000",
        tenant_id="mock-tenant",
        max_retries=0,
    )


# --- Client Fixtures ---

@pytest_asyncio.fixture
async def async_client(test_config):
    """Create an async client for testing."""
    client = AsyncCognitiveMemoryLayer(config=test_config)
    yield client
    await client.close()


@pytest.fixture
def sync_client(test_config):
    """Create a sync client for testing."""
    client = CognitiveMemoryLayer(config=test_config)
    yield client
    client.close()


# --- Mock Response Helpers ---

def make_write_response(
    success: bool = True,
    memory_id: str = "00000000-0000-0000-0000-000000000001",
    chunks_created: int = 1,
    message: str = "Stored 1 memory chunks",
) -> dict:
    """Create a mock write response."""
    return {
        "success": success,
        "memory_id": memory_id,
        "chunks_created": chunks_created,
        "message": message,
    }


def make_read_response(
    query: str = "test query",
    memories: list | None = None,
    total_count: int = 1,
    elapsed_ms: float = 42.0,
) -> dict:
    """Create a mock read response."""
    if memories is None:
        memories = [{
            "id": "00000000-0000-0000-0000-000000000001",
            "text": "User prefers vegetarian food",
            "type": "preference",
            "confidence": 0.9,
            "relevance": 0.95,
            "timestamp": "2025-01-01T00:00:00Z",
            "metadata": {},
        }]
    return {
        "query": query,
        "memories": memories,
        "facts": [],
        "preferences": memories,
        "episodes": [],
        "llm_context": "## Preferences\n- User prefers vegetarian food",
        "total_count": total_count,
        "elapsed_ms": elapsed_ms,
    }


def make_turn_response(
    memory_context: str = "## Preferences\n- vegetarian",
    memories_retrieved: int = 3,
    memories_stored: int = 1,
) -> dict:
    """Create a mock turn response."""
    return {
        "memory_context": memory_context,
        "memories_retrieved": memories_retrieved,
        "memories_stored": memories_stored,
        "reconsolidation_applied": False,
    }
```

**Pseudo-code**:
```
FIXTURES:
    test_config → CMLConfig with test values (no retries, fast timeout)
    mock_config → CMLConfig for mocked HTTP tests
    async_client → AsyncCognitiveMemoryLayer with test_config
    sync_client → CognitiveMemoryLayer with test_config

HELPERS:
    make_write_response() → mock server response for write
    make_read_response() → mock server response for read
    make_turn_response() → mock server response for turn
    make_stats_response() → mock server response for stats
```

---

## Task 7.2: Unit Tests

### Sub-Task 7.2.1: Configuration Tests

**Implementation** (`tests/unit/test_config.py`):
```python
"""Tests for configuration loading and validation."""

import os
import pytest
from cml.config import CMLConfig


class TestCMLConfig:
    """Test CMLConfig loading and validation."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CMLConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.tenant_id == "default"
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_direct_params(self):
        """Direct params should override defaults."""
        config = CMLConfig(
            api_key="sk-test",
            base_url="http://custom:9000",
            tenant_id="my-tenant",
        )
        assert config.api_key == "sk-test"
        assert config.base_url == "http://custom:9000"
        assert config.tenant_id == "my-tenant"

    def test_env_vars(self, monkeypatch):
        """Environment variables should populate config."""
        monkeypatch.setenv("CML_API_KEY", "env-key")
        monkeypatch.setenv("CML_BASE_URL", "http://env:8000")
        monkeypatch.setenv("CML_TENANT_ID", "env-tenant")
        config = CMLConfig()
        assert config.api_key == "env-key"
        assert config.base_url == "http://env:8000"
        assert config.tenant_id == "env-tenant"

    def test_direct_overrides_env(self, monkeypatch):
        """Direct params should take priority over env vars."""
        monkeypatch.setenv("CML_API_KEY", "env-key")
        config = CMLConfig(api_key="direct-key")
        assert config.api_key == "direct-key"

    def test_invalid_timeout(self):
        """Negative timeout should raise validation error."""
        with pytest.raises(ValueError):
            CMLConfig(timeout=-1.0)

    def test_url_normalization(self):
        """Trailing slash should be stripped from base_url."""
        config = CMLConfig(base_url="http://localhost:8000/")
        assert config.base_url == "http://localhost:8000"
```

**Pseudo-code**:
```
TEST CONFIG:
    test_default_values: All fields have sensible defaults
    test_direct_params: Constructor params are stored correctly
    test_env_vars: CML_* env vars populate config fields
    test_direct_overrides_env: Direct params win over env vars
    test_invalid_timeout: Negative timeout raises ValueError
    test_invalid_retries: Negative max_retries raises ValueError
    test_url_normalization: Trailing slash stripped
    test_missing_api_key: No error (api_key is optional)
```

### Sub-Task 7.2.2: Transport Tests

**Implementation** (`tests/unit/test_transport.py`):
```python
"""Tests for HTTP transport layer."""

import pytest
import httpx
from pytest_httpx import HTTPXMock

from cml.config import CMLConfig
from cml.transport.http import HTTPTransport, AsyncHTTPTransport
from cml.exceptions import (
    AuthenticationError,
    NotFoundError,
    ServerError,
    ConnectionError,
    RateLimitError,
)


class TestHTTPTransport:
    """Test synchronous HTTP transport."""

    def test_sends_api_key_header(self, httpx_mock: HTTPXMock, mock_config):
        """Transport should send X-API-Key header."""
        httpx_mock.add_response(json={"status": "ok"})
        transport = HTTPTransport(mock_config)
        transport.request("GET", "/api/v1/health")
        request = httpx_mock.get_request()
        assert request.headers["X-API-Key"] == "mock-key"

    def test_sends_tenant_header(self, httpx_mock: HTTPXMock, mock_config):
        """Transport should send X-Tenant-ID header."""
        httpx_mock.add_response(json={"status": "ok"})
        transport = HTTPTransport(mock_config)
        transport.request("GET", "/api/v1/health")
        request = httpx_mock.get_request()
        assert request.headers["X-Tenant-ID"] == "mock-tenant"

    def test_401_raises_authentication_error(self, httpx_mock: HTTPXMock, mock_config):
        """401 response should raise AuthenticationError."""
        httpx_mock.add_response(status_code=401, json={"detail": "Invalid API key"})
        transport = HTTPTransport(mock_config)
        with pytest.raises(AuthenticationError):
            transport.request("GET", "/api/v1/health")

    def test_404_raises_not_found(self, httpx_mock: HTTPXMock, mock_config):
        """404 response should raise NotFoundError."""
        httpx_mock.add_response(status_code=404, json={"detail": "Not found"})
        transport = HTTPTransport(mock_config)
        with pytest.raises(NotFoundError):
            transport.request("GET", "/api/v1/nonexistent")

    def test_500_raises_server_error(self, httpx_mock: HTTPXMock, mock_config):
        """5xx response should raise ServerError."""
        httpx_mock.add_response(status_code=500, json={"detail": "Internal error"})
        transport = HTTPTransport(mock_config)
        with pytest.raises(ServerError):
            transport.request("GET", "/api/v1/health")

    def test_429_raises_rate_limit(self, httpx_mock: HTTPXMock, mock_config):
        """429 response should raise RateLimitError with retry_after."""
        httpx_mock.add_response(
            status_code=429,
            headers={"Retry-After": "5"},
            json={"detail": "Rate limited"},
        )
        transport = HTTPTransport(mock_config)
        with pytest.raises(RateLimitError) as exc_info:
            transport.request("GET", "/api/v1/health")
        assert exc_info.value.retry_after == 5.0

    def test_connection_error(self, httpx_mock: HTTPXMock, mock_config):
        """Connection failure should raise ConnectionError."""
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        transport = HTTPTransport(mock_config)
        with pytest.raises(ConnectionError):
            transport.request("GET", "/api/v1/health")
```

**Pseudo-code**:
```
TEST TRANSPORT:
    test_sends_api_key_header: X-API-Key present in request
    test_sends_tenant_header: X-Tenant-ID present in request
    test_sends_user_agent: User-Agent contains py-cml version
    test_sends_json_content_type: Content-Type is application/json
    test_401_maps_to_AuthenticationError
    test_403_maps_to_AuthorizationError
    test_404_maps_to_NotFoundError
    test_422_maps_to_ValidationError
    test_429_maps_to_RateLimitError_with_retry_after
    test_500_maps_to_ServerError
    test_connection_error_maps_to_ConnectionError
    test_timeout_maps_to_TimeoutError
    test_admin_key_override: use_admin_key replaces X-API-Key
    test_close_closes_client: close() closes httpx client
```

### Sub-Task 7.2.3: Retry Tests

**Implementation** (`tests/unit/test_retry.py`):
```python
"""Tests for retry logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from cml.config import CMLConfig
from cml.transport.retry import retry_sync, retry_async
from cml.exceptions import ServerError, RateLimitError, ConnectionError


class TestRetrySync:
    """Test synchronous retry logic."""

    def test_no_retry_on_success(self):
        """Should not retry on success."""
        func = MagicMock(return_value={"ok": True})
        config = CMLConfig(max_retries=3, retry_delay=0.01)
        result = retry_sync(config, func)
        assert result == {"ok": True}
        assert func.call_count == 1

    def test_retry_on_server_error(self):
        """Should retry on ServerError."""
        func = MagicMock(
            side_effect=[ServerError("500"), ServerError("500"), {"ok": True}]
        )
        config = CMLConfig(max_retries=3, retry_delay=0.01)
        result = retry_sync(config, func)
        assert result == {"ok": True}
        assert func.call_count == 3

    def test_exhausts_retries(self):
        """Should raise after max_retries exhausted."""
        func = MagicMock(side_effect=ServerError("500"))
        config = CMLConfig(max_retries=2, retry_delay=0.01)
        with pytest.raises(ServerError):
            retry_sync(config, func)
        assert func.call_count == 3  # initial + 2 retries

    def test_no_retry_on_client_error(self):
        """Should not retry on 4xx errors (except 429)."""
        from cml.exceptions import ValidationError
        func = MagicMock(side_effect=ValidationError("422"))
        config = CMLConfig(max_retries=3, retry_delay=0.01)
        with pytest.raises(ValidationError):
            retry_sync(config, func)
        assert func.call_count == 1
```

**Pseudo-code**:
```
TEST RETRY:
    test_no_retry_on_success: Call count == 1
    test_retry_on_server_error: Retries until success
    test_retry_on_connection_error: Retries on connection failures
    test_retry_on_rate_limit: Retries with retry_after
    test_exhausts_retries: Raises after max_retries + 1 attempts
    test_no_retry_on_client_error: 4xx errors (except 429) not retried
    test_backoff_increases: Verify exponential delay
    test_jitter_adds_randomness: Verify delay has random component
```

### Sub-Task 7.2.4: Client Operation Tests

**Implementation** (`tests/unit/test_async_client.py` — representative):
```python
"""Tests for async client operations."""

import pytest
from pytest_httpx import HTTPXMock
from uuid import UUID

from cml import AsyncCognitiveMemoryLayer
from cml.models import MemoryType, WriteResponse, ReadResponse, TurnResponse
from conftest import make_write_response, make_read_response, make_turn_response


class TestAsyncWrite:
    """Test async write operation."""

    @pytest.mark.asyncio
    async def test_write_basic(self, httpx_mock: HTTPXMock, mock_config):
        """Write basic content."""
        httpx_mock.add_response(json=make_write_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.write("User prefers vegetarian food")
        assert isinstance(result, WriteResponse)
        assert result.success is True
        assert result.chunks_created == 1
        assert result.memory_id is not None

    @pytest.mark.asyncio
    async def test_write_with_tags(self, httpx_mock: HTTPXMock, mock_config):
        """Write with context tags."""
        httpx_mock.add_response(json=make_write_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.write(
                "User prefers vegetarian food",
                context_tags=["preference", "food"],
            )
        request = httpx_mock.get_request()
        body = request.content
        assert b"preference" in body

    @pytest.mark.asyncio
    async def test_write_with_type(self, httpx_mock: HTTPXMock, mock_config):
        """Write with explicit memory type."""
        httpx_mock.add_response(json=make_write_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.write(
                "User prefers vegetarian food",
                memory_type=MemoryType.PREFERENCE,
            )
        assert result.success


class TestAsyncRead:
    """Test async read operation."""

    @pytest.mark.asyncio
    async def test_read_basic(self, httpx_mock: HTTPXMock, mock_config):
        """Read with basic query."""
        httpx_mock.add_response(json=make_read_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.read("dietary preferences")
        assert isinstance(result, ReadResponse)
        assert result.total_count == 1
        assert len(result.memories) == 1

    @pytest.mark.asyncio
    async def test_read_context_property(self, httpx_mock: HTTPXMock, mock_config):
        """Read .context returns LLM context string."""
        httpx_mock.add_response(json=make_read_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.read("preferences", format="llm_context")
        assert isinstance(result.context, str)
        assert len(result.context) > 0


class TestAsyncTurn:
    """Test async turn operation."""

    @pytest.mark.asyncio
    async def test_turn_basic(self, httpx_mock: HTTPXMock, mock_config):
        """Process a basic turn."""
        httpx_mock.add_response(json=make_turn_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.turn(user_message="What do I like to eat?")
        assert isinstance(result, TurnResponse)
        assert result.memories_retrieved == 3
        assert len(result.memory_context) > 0
```

**Pseudo-code**:
```
TEST ASYNC CLIENT:
    WRITE:
        test_write_basic: Returns WriteResponse with success
        test_write_with_tags: context_tags in request body
        test_write_with_type: memory_type in request body
        test_write_with_metadata: metadata in request body
        test_write_with_session: session_id in request body
        test_write_with_namespace: namespace in request body

    READ:
        test_read_basic: Returns ReadResponse with memories
        test_read_with_filters: context_filter and memory_types in body
        test_read_with_time_range: since and until in body
        test_read_context_property: .context returns formatted string
        test_read_llm_context_format: format="llm_context" works

    TURN:
        test_turn_basic: Returns TurnResponse with context
        test_turn_with_response: assistant_response in body
        test_turn_with_session: session_id in body

    UPDATE:
        test_update_text: text change triggers re-embedding
        test_update_confidence: confidence update
        test_update_feedback_correct: boosts confidence
        test_update_feedback_incorrect: deletes memory

    FORGET:
        test_forget_by_ids: memory_ids in body
        test_forget_by_query: query-based forget
        test_forget_by_time: before filter
        test_forget_no_selector: raises ValueError

    STATS:
        test_stats_returns_response: StatsResponse parsed correctly

    HEALTH:
        test_health: HealthResponse with status

TEST SYNC CLIENT:
    Mirror all async tests with sync calls
```

### Sub-Task 7.2.5: Model Serialization Tests

**Implementation** (`tests/unit/test_models.py`):
```python
"""Tests for Pydantic model serialization."""

import pytest
from datetime import datetime, timezone
from uuid import UUID

from cml.models import MemoryType, MemoryStatus, MemoryItem
from cml.models.responses import WriteResponse, ReadResponse
from cml.models.requests import WriteRequest, ReadRequest


class TestMemoryItem:
    def test_parse_from_dict(self):
        data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "text": "Test memory",
            "type": "preference",
            "confidence": 0.9,
            "relevance": 0.95,
            "timestamp": "2025-01-01T00:00:00Z",
            "metadata": {"key": "value"},
        }
        item = MemoryItem(**data)
        assert isinstance(item.id, UUID)
        assert item.text == "Test memory"
        assert item.type == "preference"

    def test_serialize_to_dict(self):
        item = MemoryItem(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            text="Test",
            type="preference",
            confidence=0.9,
            relevance=0.95,
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        d = item.model_dump()
        assert "id" in d
        assert "text" in d


class TestWriteRequest:
    def test_exclude_none(self):
        req = WriteRequest(content="Hello")
        d = req.model_dump(exclude_none=True)
        assert "content" in d
        assert "context_tags" not in d
        assert "session_id" not in d
```

---

## Task 7.3: Integration Tests

### Sub-Task 7.3.1: Integration Test Fixtures

**Implementation** (`tests/integration/conftest.py`):
```python
"""Integration test fixtures — requires running CML server."""

import os
import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig

INTEGRATION_URL = os.environ.get("CML_TEST_URL", "http://localhost:8000")
INTEGRATION_KEY = os.environ.get("CML_TEST_API_KEY", "test-key")
INTEGRATION_TENANT = f"test-{os.getpid()}"  # Unique per test run


@pytest.fixture(scope="session")
def integration_config():
    return CMLConfig(
        api_key=INTEGRATION_KEY,
        base_url=INTEGRATION_URL,
        tenant_id=INTEGRATION_TENANT,
        timeout=30.0,
        max_retries=1,
    )


@pytest_asyncio.fixture
async def live_client(integration_config):
    """Async client connected to real CML server."""
    client = AsyncCognitiveMemoryLayer(config=integration_config)
    yield client
    # Cleanup: delete all test memories
    try:
        await client.delete_all(confirm=True)
    except Exception:
        pass
    await client.close()
```

### Sub-Task 7.3.2: Write-Read Roundtrip Test

**Implementation** (`tests/integration/test_write_read.py`):
```python
"""Integration tests for write → read roundtrip."""

import pytest


@pytest.mark.integration
class TestWriteReadRoundtrip:

    @pytest.mark.asyncio
    async def test_write_then_read(self, live_client):
        """Write content, then retrieve it."""
        # Write
        write_result = await live_client.write(
            "User prefers vegetarian food and lives in Paris",
            context_tags=["preference", "personal"],
        )
        assert write_result.success
        assert write_result.chunks_created >= 1

        # Read
        read_result = await live_client.read("dietary preferences")
        assert read_result.total_count >= 1
        assert any("vegetarian" in m.text for m in read_result.memories)

    @pytest.mark.asyncio
    async def test_write_multiple_then_read(self, live_client):
        """Write multiple items, read should find all."""
        await live_client.write("User likes Italian cuisine")
        await live_client.write("User is allergic to nuts")
        await live_client.write("User prefers organic produce")

        result = await live_client.read("food preferences")
        assert result.total_count >= 2

    @pytest.mark.asyncio
    async def test_read_llm_context_format(self, live_client):
        """Read with llm_context format returns formatted string."""
        await live_client.write("User's birthday is March 15th")
        result = await live_client.read("birthday", format="llm_context")
        assert result.llm_context is not None
        assert isinstance(result.llm_context, str)
```

**Pseudo-code**:
```
TEST INTEGRATION WRITE-READ:
    test_write_then_read:
        1. Write "User prefers vegetarian food"
        2. Read "dietary preferences"
        3. Assert retrieved text contains "vegetarian"

    test_write_multiple_then_read:
        1. Write 3 food-related memories
        2. Read "food preferences"
        3. Assert total_count >= 2

    test_read_returns_empty_for_unrelated:
        1. Write about food preferences
        2. Read "quantum physics"
        3. Assert total_count == 0 or relevance < 0.3
```

---

## Task 7.4: Embedded Mode Tests

### Sub-Task 7.4.1: Lite Mode Tests

**Implementation** (`tests/embedded/test_lite_mode.py`):
```python
"""Tests for embedded lite mode (SQLite + local embeddings)."""

import pytest
from cml import EmbeddedCognitiveMemoryLayer


@pytest.mark.embedded
class TestLiteMode:

    @pytest.mark.asyncio
    async def test_zero_config_init(self):
        """Should initialize with zero configuration."""
        async with EmbeddedCognitiveMemoryLayer() as memory:
            assert memory is not None

    @pytest.mark.asyncio
    async def test_write_and_read(self):
        """Should store and retrieve in lite mode."""
        async with EmbeddedCognitiveMemoryLayer() as memory:
            result = await memory.write("User prefers dark mode")
            assert result.success

            read = await memory.read("theme preferences")
            assert read.total_count >= 1

    @pytest.mark.asyncio
    async def test_persistent_storage(self, tmp_path):
        """SQLite file should persist between instances."""
        db_path = str(tmp_path / "test.db")

        # Write in first instance
        async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as memory:
            await memory.write("Persistent memory test")

        # Read in second instance
        async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as memory:
            result = await memory.read("persistent")
            assert result.total_count >= 1
```

---

## Task 7.5: End-to-End Tests

### Sub-Task 7.5.1: Chat Flow Test

**Implementation** (`tests/e2e/test_chat_flow.py`):
```python
"""End-to-end test: full chat flow with memory."""

import pytest


@pytest.mark.e2e
class TestChatFlow:

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, live_client):
        """Simulate a multi-turn conversation with memory."""
        session_id = "e2e-chat-001"

        # Turn 1: User introduces themselves
        turn1 = await live_client.turn(
            user_message="My name is Alice and I'm a software engineer.",
            session_id=session_id,
        )
        assert turn1.memories_stored >= 0  # May or may not store

        # Turn 2: Ask about user
        turn2 = await live_client.turn(
            user_message="What do you know about me?",
            session_id=session_id,
        )
        # Should retrieve the introduction
        assert turn2.memories_retrieved >= 0
        assert len(turn2.memory_context) > 0

        # Turn 3: Add more info
        turn3 = await live_client.turn(
            user_message="I live in San Francisco and I love hiking.",
            session_id=session_id,
        )
        assert turn3.memories_stored >= 0

        # Turn 4: Query preferences
        turn4 = await live_client.turn(
            user_message="What are my hobbies?",
            session_id=session_id,
        )
        assert turn4.memories_retrieved >= 0
```

---

## Task 7.6: Test Markers & CI Configuration

### Sub-Task 7.6.1: Pytest Markers

**Implementation** (in `pyproject.toml`):
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: Integration tests (requires CML server)",
    "embedded: Embedded mode tests (requires embedded deps)",
    "e2e: End-to-end tests (requires full setup)",
    "slow: Tests that take > 5 seconds",
]
```

### Sub-Task 7.6.2: CI Test Matrix

**Pseudo-code**:
```yaml
# In .github/workflows/test.yml:
jobs:
  unit-tests:
    # Run on every push/PR
    # No external services needed
    # Fast (< 1 minute)
    steps:
      - pip install -e ".[dev]"
      - pytest tests/unit/ -v --cov

  integration-tests:
    # Run on main branch only (or manually)
    # Requires CML server via docker-compose
    services:
      postgres, neo4j, redis, cml-server
    steps:
      - pip install -e ".[dev]"
      - pytest tests/integration/ -v -m integration

  embedded-tests:
    # Run on main branch or when embedded/ files change
    steps:
      - pip install -e ".[dev,embedded]"
      - pytest tests/embedded/ -v -m embedded
```

---

## Task 7.7: Code Coverage

### Sub-Task 7.7.1: Coverage Configuration

**Implementation** (`.coveragerc` or in `pyproject.toml`):
```toml
[tool.coverage.run]
source = ["src/cml"]
branch = true

[tool.coverage.report]
show_missing = true
fail_under = 80
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "pass",
]
```

### Sub-Task 7.7.2: Coverage Targets

| Component | Target Coverage |
|:----------|:---------------|
| `config.py` | 95% |
| `exceptions.py` | 100% |
| `transport/http.py` | 90% |
| `transport/retry.py` | 95% |
| `async_client.py` | 90% |
| `client.py` | 90% |
| `models/` | 95% |
| `utils/` | 85% |
| **Overall** | **85%** |

---

## Acceptance Criteria

- [ ] Unit tests cover all public methods (sync and async)
- [ ] Transport tests verify all HTTP status code → exception mappings
- [ ] Retry tests verify backoff, jitter, and exhaustion behavior
- [ ] Model tests verify serialization and deserialization
- [ ] Integration tests pass against a running CML server
- [ ] Embedded tests verify lite mode with zero config
- [ ] E2E tests simulate realistic multi-turn conversations
- [ ] Test markers allow selective test execution
- [ ] CI runs unit tests on every push (< 2 minutes)
- [ ] CI runs integration tests on main branch
- [ ] Code coverage >= 85% overall
- [ ] All tests use fixtures (no hardcoded URLs or keys)
