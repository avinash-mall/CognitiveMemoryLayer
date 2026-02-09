# Phase 2: Core Client SDK

## Objective

Build the foundational HTTP client that connects to a running CognitiveMemoryLayer server, handling authentication, connection management, request/response serialization, retry logic, and error handling.

---

## Task 2.1: Configuration System

### Sub-Task 2.1.1: CMLConfig Dataclass

**Architecture**: Single configuration object that accepts parameters directly, from environment variables, or from a `.env` file. Uses Pydantic for validation.

**Implementation** (`src/cml/config.py`):
```python
"""Configuration management for py-cml."""

import os
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CMLConfig(BaseModel):
    """Configuration for CognitiveMemoryLayer client.

    Parameters can be set directly, via environment variables, or
    via a .env file. Environment variables use the CML_ prefix.

    Env vars:
        CML_API_KEY: API key for authentication
        CML_BASE_URL: Base URL of the CML server
        CML_TENANT_ID: Tenant identifier
        CML_TIMEOUT: Request timeout in seconds
        CML_MAX_RETRIES: Maximum retry attempts
        CML_RETRY_DELAY: Delay between retries in seconds
        CML_ADMIN_API_KEY: Admin API key (for admin operations)
    """

    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the CML server"
    )
    tenant_id: str = Field(default="default", description="Tenant identifier")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Base delay between retries (seconds)")
    admin_api_key: Optional[str] = Field(default=None, description="Admin API key")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, values: dict) -> dict:
        """Load unset values from environment variables."""
        env_map = {
            "api_key": "CML_API_KEY",
            "base_url": "CML_BASE_URL",
            "tenant_id": "CML_TENANT_ID",
            "timeout": "CML_TIMEOUT",
            "max_retries": "CML_MAX_RETRIES",
            "retry_delay": "CML_RETRY_DELAY",
            "admin_api_key": "CML_ADMIN_API_KEY",
        }
        for field, env_var in env_map.items():
            if field not in values or values[field] is None:
                env_val = os.environ.get(env_var)
                if env_val is not None:
                    values[field] = env_val
        return values
```

**Pseudo-code for resolution order**:
```
1. Direct parameter passed to constructor (highest priority)
2. Environment variable with CML_ prefix
3. .env file loaded via python-dotenv
4. Default value from Field definition (lowest priority)

For each config field:
    IF explicitly passed → use it
    ELSE IF CML_{FIELD_UPPER} in os.environ → use env var
    ELSE IF .env file has CML_{FIELD_UPPER} → use .env value
    ELSE → use default
```

### Sub-Task 2.1.2: Config Validation

**Pseudo-code**:
```
VALIDATE config:
    IF base_url does not start with http:// or https:// → raise ValueError
    IF timeout <= 0 → raise ValueError
    IF max_retries < 0 → raise ValueError
    IF retry_delay < 0 → raise ValueError
    NORMALIZE base_url: strip trailing slash
```

---

## Task 2.2: Exception Hierarchy

### Sub-Task 2.2.1: Define Exception Classes

**Architecture**: Create a rich exception hierarchy that maps HTTP status codes and transport errors to meaningful Python exceptions.

**Implementation** (`src/cml/exceptions.py`):
```python
"""Exception hierarchy for py-cml."""

from typing import Any, Dict, Optional


class CMLError(Exception):
    """Base exception for all CML errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(CMLError):
    """Raised when authentication fails (401)."""
    pass


class AuthorizationError(CMLError):
    """Raised when authorization fails (403)."""
    pass


class NotFoundError(CMLError):
    """Raised when a resource is not found (404)."""
    pass


class ValidationError(CMLError):
    """Raised when request validation fails (422)."""
    pass


class RateLimitError(CMLError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ServerError(CMLError):
    """Raised when the server returns 5xx error."""
    pass


class ConnectionError(CMLError):
    """Raised when unable to connect to the CML server."""
    pass


class TimeoutError(CMLError):
    """Raised when a request times out."""
    pass
```

**Pseudo-code for status code mapping**:
```
FUNCTION map_status_code(status_code, response_body):
    MATCH status_code:
        401 → raise AuthenticationError
        403 → raise AuthorizationError
        404 → raise NotFoundError
        422 → raise ValidationError
        429 → raise RateLimitError (extract retry_after header)
        >= 500 → raise ServerError
        ELSE → raise CMLError
```

---

## Task 2.3: HTTP Transport Layer

### Sub-Task 2.3.1: Base HTTP Transport

**Architecture**: Use `httpx` for both sync and async HTTP. Wrap all requests with consistent headers, authentication, error mapping, and serialization.

**Implementation** (`src/cml/transport/http.py`):
```python
"""HTTP transport layer using httpx."""

from typing import Any, Dict, Optional
import httpx

from ..config import CMLConfig
from ..exceptions import (
    AuthenticationError,
    AuthorizationError,
    CMLError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)


class HTTPTransport:
    """Synchronous HTTP transport for CML API."""

    def __init__(self, config: CMLConfig):
        self._config = config
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                verify=self._config.verify_ssl,
                headers=self._build_headers(),
            )
        return self._client

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"py-cml/{__version__}",
        }
        if self._config.api_key:
            headers["X-API-Key"] = self._config.api_key
        if self._config.tenant_id:
            headers["X-Tenant-ID"] = self._config.tenant_id
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        use_admin_key: bool = False,
    ) -> Dict[str, Any]:
        """Execute an HTTP request with error handling."""
        headers = {}
        if use_admin_key and self._config.admin_api_key:
            headers["X-API-Key"] = self._config.admin_api_key

        try:
            response = self.client.request(
                method=method,
                url=path,
                json=json,
                params=params,
                headers=headers,
            )
            self._raise_for_status(response)
            return response.json()
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to {self._config.base_url}: {e}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out after {self._config.timeout}s: {e}")

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status codes to CML exceptions."""
        if response.is_success:
            return

        body = None
        try:
            body = response.json()
        except Exception:
            pass

        msg = f"HTTP {response.status_code}"
        if body and "detail" in body:
            msg = body["detail"]

        match response.status_code:
            case 401:
                raise AuthenticationError(msg, status_code=401, response_body=body)
            case 403:
                raise AuthorizationError(msg, status_code=403, response_body=body)
            case 404:
                raise NotFoundError(msg, status_code=404, response_body=body)
            case 422:
                raise ValidationError(msg, status_code=422, response_body=body)
            case 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    msg,
                    status_code=429,
                    response_body=body,
                    retry_after=float(retry_after) if retry_after else None,
                )
            case code if code >= 500:
                raise ServerError(msg, status_code=code, response_body=body)
            case _:
                raise CMLError(msg, status_code=response.status_code, response_body=body)

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()
```

**Pseudo-code**:
```
CLASS HTTPTransport:
    INIT(config):
        Store config
        Create lazy httpx.Client with:
            base_url from config
            timeout from config
            default headers (Content-Type, Accept, User-Agent, X-API-Key, X-Tenant-ID)

    METHOD request(method, path, json=None, params=None):
        TRY:
            response = client.request(method, path, json, params)
            IF response.is_error:
                Map status code → CMLError subclass
                Raise with message, status_code, response_body
            RETURN response.json()
        CATCH httpx.ConnectError:
            Raise ConnectionError
        CATCH httpx.TimeoutException:
            Raise TimeoutError

    METHOD close():
        Close httpx.Client
```

### Sub-Task 2.3.2: Async HTTP Transport

**Architecture**: Mirror the sync transport but use `httpx.AsyncClient`.

**Implementation** (`src/cml/transport/http.py` — async portion):
```python
class AsyncHTTPTransport:
    """Asynchronous HTTP transport for CML API."""

    def __init__(self, config: CMLConfig):
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                verify=self._config.verify_ssl,
                headers=self._build_headers(),
            )
        return self._client

    # _build_headers() — same as sync
    # _raise_for_status() — same as sync

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        use_admin_key: bool = False,
    ) -> Dict[str, Any]:
        headers = {}
        if use_admin_key and self._config.admin_api_key:
            headers["X-API-Key"] = self._config.admin_api_key

        try:
            response = await self.client.request(
                method=method,
                url=path,
                json=json,
                params=params,
                headers=headers,
            )
            self._raise_for_status(response)
            return response.json()
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect: {e}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
```

### Sub-Task 2.3.3: Retry Logic

**Architecture**: Implement exponential backoff with jitter for retryable errors (5xx, 429, connection errors).

**Implementation** (`src/cml/transport/retry.py`):
```python
"""Retry logic with exponential backoff and jitter."""

import asyncio
import random
import time
from typing import Callable, TypeVar

from ..config import CMLConfig
from ..exceptions import CMLError, RateLimitError, ServerError, ConnectionError, TimeoutError

T = TypeVar("T")

RETRYABLE_EXCEPTIONS = (ServerError, ConnectionError, TimeoutError, RateLimitError)


def retry_sync(config: CMLConfig, func: Callable[..., T], *args, **kwargs) -> T:
    """Execute func with sync retry logic."""
    last_exception = None
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            last_exception = e
            if e.retry_after:
                time.sleep(e.retry_after)
            else:
                _sleep_with_backoff(attempt, config.retry_delay)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < config.max_retries:
                _sleep_with_backoff(attempt, config.retry_delay)
    raise last_exception


async def retry_async(config: CMLConfig, func, *args, **kwargs):
    """Execute func with async retry logic."""
    last_exception = None
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except RateLimitError as e:
            last_exception = e
            if e.retry_after:
                await asyncio.sleep(e.retry_after)
            else:
                await _async_sleep_with_backoff(attempt, config.retry_delay)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < config.max_retries:
                await _async_sleep_with_backoff(attempt, config.retry_delay)
    raise last_exception


def _sleep_with_backoff(attempt: int, base_delay: float) -> None:
    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
    time.sleep(delay)


async def _async_sleep_with_backoff(attempt: int, base_delay: float) -> None:
    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
    await asyncio.sleep(delay)
```

**Pseudo-code**:
```
FUNCTION retry(config, callable, *args, **kwargs):
    FOR attempt IN range(max_retries + 1):
        TRY:
            RETURN callable(*args, **kwargs)
        CATCH RateLimitError:
            IF retry_after header → sleep(retry_after)
            ELSE → sleep(backoff_with_jitter(attempt))
        CATCH (ServerError, ConnectionError, TimeoutError):
            IF attempt < max_retries:
                sleep(backoff_with_jitter(attempt))

    RAISE last_exception

FUNCTION backoff_with_jitter(attempt, base_delay):
    delay = base_delay * (2 ^ attempt) + random(0, base_delay)
    RETURN delay
```

---

## Task 2.4: Pydantic Models

### Sub-Task 2.4.1: Enum Definitions

**Architecture**: Mirror the CML server enums so the SDK provides typed constants.

**Implementation** (`src/cml/models/enums.py`):
```python
"""Enums for memory types, status, and operations."""

from enum import Enum


class MemoryType(str, Enum):
    """Type of memory record."""
    EPISODIC_EVENT = "episodic_event"
    SEMANTIC_FACT = "semantic_fact"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    HYPOTHESIS = "hypothesis"
    PREFERENCE = "preference"
    TASK_STATE = "task_state"
    CONVERSATION = "conversation"
    MESSAGE = "message"
    TOOL_RESULT = "tool_result"
    REASONING_STEP = "reasoning_step"
    SCRATCH = "scratch"
    KNOWLEDGE = "knowledge"
    OBSERVATION = "observation"
    PLAN = "plan"


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""
    ACTIVE = "active"
    SILENT = "silent"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemorySource(str, Enum):
    """Provenance source of a memory."""
    USER_EXPLICIT = "user_explicit"
    USER_CONFIRMED = "user_confirmed"
    AGENT_INFERRED = "agent_inferred"
    TOOL_RESULT = "tool_result"
    CONSOLIDATION = "consolidation"
    RECONSOLIDATION = "reconsolidation"


class OperationType(str, Enum):
    """Type of operation in event log."""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"
    REINFORCE = "reinforce"
    DECAY = "decay"
    SILENCE = "silence"
    COMPRESS = "compress"
```

### Sub-Task 2.4.2: Response Models

**Implementation** (`src/cml/models/responses.py`):
```python
"""Response models matching CML API responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .enums import MemoryType


class MemoryItem(BaseModel):
    """A single memory item from retrieval."""
    id: UUID
    text: str
    type: str
    confidence: float
    relevance: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WriteResponse(BaseModel):
    """Response from write operation."""
    success: bool
    memory_id: Optional[UUID] = None
    chunks_created: int = 0
    message: str = ""


class ReadResponse(BaseModel):
    """Response from read operation."""
    query: str
    memories: List[MemoryItem]
    facts: List[MemoryItem] = Field(default_factory=list)
    preferences: List[MemoryItem] = Field(default_factory=list)
    episodes: List[MemoryItem] = Field(default_factory=list)
    llm_context: Optional[str] = None
    total_count: int
    elapsed_ms: float

    @property
    def context(self) -> str:
        """Shortcut to get formatted LLM context string."""
        return self.llm_context or ""


class TurnResponse(BaseModel):
    """Response from seamless turn processing."""
    memory_context: str
    memories_retrieved: int
    memories_stored: int
    reconsolidation_applied: bool = False


class UpdateResponse(BaseModel):
    """Response from update operation."""
    success: bool
    memory_id: UUID
    version: int
    message: str = ""


class ForgetResponse(BaseModel):
    """Response from forget operation."""
    success: bool
    affected_count: int
    message: str = ""


class StatsResponse(BaseModel):
    """Memory statistics response."""
    total_memories: int
    active_memories: int
    silent_memories: int
    archived_memories: int
    by_type: Dict[str, int]
    avg_confidence: float
    avg_importance: float
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None
    estimated_size_mb: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: Optional[str] = None
    components: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Session creation response."""
    session_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
```

### Sub-Task 2.4.3: Request Models

**Implementation** (`src/cml/models/requests.py`):
```python
"""Internal request models for constructing API payloads."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .enums import MemoryType


class WriteRequest(BaseModel):
    """Write memory request payload."""
    content: str
    context_tags: Optional[List[str]] = None
    session_id: Optional[str] = None
    memory_type: Optional[MemoryType] = None
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    turn_id: Optional[str] = None
    agent_id: Optional[str] = None


class ReadRequest(BaseModel):
    """Read memory request payload."""
    query: str
    max_results: int = Field(default=10, le=50)
    context_filter: Optional[List[str]] = None
    memory_types: Optional[List[MemoryType]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    format: Literal["packet", "list", "llm_context"] = "packet"


class TurnRequest(BaseModel):
    """Seamless turn request payload."""
    user_message: str
    assistant_response: Optional[str] = None
    session_id: Optional[str] = None
    max_context_tokens: int = 1500


class UpdateRequest(BaseModel):
    """Update memory request payload."""
    memory_id: UUID
    text: Optional[str] = None
    confidence: Optional[float] = None
    importance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None


class ForgetRequest(BaseModel):
    """Forget memories request payload."""
    memory_ids: Optional[List[UUID]] = None
    query: Optional[str] = None
    before: Optional[datetime] = None
    action: Literal["delete", "archive", "silence"] = "delete"
```

### Sub-Task 2.4.4: Model Exports

**Implementation** (`src/cml/models/__init__.py`):
```python
"""Public model exports."""

from .enums import MemorySource, MemoryStatus, MemoryType, OperationType
from .responses import (
    ForgetResponse,
    HealthResponse,
    MemoryItem,
    ReadResponse,
    SessionResponse,
    StatsResponse,
    TurnResponse,
    UpdateResponse,
    WriteResponse,
)

__all__ = [
    "MemoryType",
    "MemoryStatus",
    "MemorySource",
    "OperationType",
    "MemoryItem",
    "WriteResponse",
    "ReadResponse",
    "TurnResponse",
    "UpdateResponse",
    "ForgetResponse",
    "StatsResponse",
    "HealthResponse",
    "SessionResponse",
]
```

---

## Task 2.5: Async Client Foundation

### Sub-Task 2.5.1: AsyncCognitiveMemoryLayer Class

**Architecture**: The async client is the primary implementation. All operations are async methods. Uses `AsyncHTTPTransport` for requests.

**Implementation** (`src/cml/async_client.py`):
```python
"""Async client for CognitiveMemoryLayer."""

from typing import Optional

from .config import CMLConfig
from .transport.http import AsyncHTTPTransport


class AsyncCognitiveMemoryLayer:
    """Async Python client for the CognitiveMemoryLayer API.

    Usage:
        async with AsyncCognitiveMemoryLayer(api_key="sk-...") as memory:
            await memory.write("User prefers vegetarian food.")
            result = await memory.read("dietary preferences")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        tenant_id: str = "default",
        *,
        config: Optional[CMLConfig] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        if config:
            self._config = config
        else:
            self._config = CMLConfig(
                api_key=api_key,
                base_url=base_url,
                tenant_id=tenant_id,
                timeout=timeout,
                max_retries=max_retries,
            )
        self._transport = AsyncHTTPTransport(self._config)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP connection."""
        await self._transport.close()

    # Memory operations defined in Phase 3
```

**Pseudo-code**:
```
CLASS AsyncCognitiveMemoryLayer:
    INIT(api_key, base_url, tenant_id, config, timeout, max_retries):
        IF config provided → use it
        ELSE → create CMLConfig from individual params
        Create AsyncHTTPTransport(config)

    ASYNC CONTEXT MANAGER:
        __aenter__ → return self
        __aexit__ → await close()

    ASYNC close():
        Close transport connection

    # All memory operations (write, read, etc.) return parsed Pydantic models
    # Each operation:
    #   1. Build request model
    #   2. Serialize to dict (exclude_none=True)
    #   3. Call transport.request(method, path, json=payload)
    #   4. Parse response into response model
    #   5. Return response model
```

### Sub-Task 2.5.2: Sync Client Wrapper

**Architecture**: The sync client wraps the async client using `asyncio.run()` or an existing event loop.

**Implementation** (`src/cml/client.py`):
```python
"""Synchronous client wrapping the async client."""

import asyncio
from typing import Optional

from .async_client import AsyncCognitiveMemoryLayer
from .config import CMLConfig


class CognitiveMemoryLayer:
    """Synchronous Python client for the CognitiveMemoryLayer API.

    Usage:
        with CognitiveMemoryLayer(api_key="sk-...") as memory:
            memory.write("User prefers vegetarian food.")
            result = memory.read("dietary preferences")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        tenant_id: str = "default",
        *,
        config: Optional[CMLConfig] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        if config:
            self._config = config
        else:
            self._config = CMLConfig(
                api_key=api_key,
                base_url=base_url,
                tenant_id=tenant_id,
                timeout=timeout,
                max_retries=max_retries,
            )
        # Use sync transport directly (not async wrapper)
        from .transport.http import HTTPTransport
        self._transport = HTTPTransport(self._config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        self._transport.close()

    # Memory operations defined in Phase 3
    # Each method mirrors the async version but calls
    # self._transport.request() synchronously
```

**Pseudo-code**:
```
CLASS CognitiveMemoryLayer:
    INIT(same params as async):
        Create CMLConfig
        Create HTTPTransport (sync)

    CONTEXT MANAGER:
        __enter__ → return self
        __exit__ → close()

    close():
        Close transport

    # For each operation (write, read, etc.):
    #   Same logic as async but using sync transport
    #   No asyncio.run() needed — direct sync httpx calls
```

---

## Task 2.6: Connection Verification

### Sub-Task 2.6.1: Health Check

**Pseudo-code**:
```
METHOD health() -> HealthResponse:
    response = transport.request("GET", "/api/v1/health")
    RETURN HealthResponse(**response)
```

### Sub-Task 2.6.2: Connection Test on Init (Optional)

**Pseudo-code**:
```
METHOD verify_connection() -> bool:
    TRY:
        health_response = self.health()
        RETURN health_response.status == "ok"
    CATCH CMLError:
        RETURN False
```

---

## Acceptance Criteria

- [ ] `CMLConfig` loads from direct params, env vars, and `.env` files
- [ ] Config validation rejects invalid values (bad URLs, negative timeouts)
- [ ] Exception hierarchy covers all HTTP status codes
- [ ] `HTTPTransport` sends requests with correct headers (API key, tenant ID)
- [ ] `AsyncHTTPTransport` works with `async/await`
- [ ] Retry logic retries on 5xx, 429, connection errors
- [ ] Retry uses exponential backoff with jitter
- [ ] `RateLimitError` respects `Retry-After` header
- [ ] Both sync and async clients support context manager protocol
- [ ] All response models parse server JSON correctly
- [ ] `health()` method returns `HealthResponse`
- [ ] Type annotations pass `mypy --strict`
