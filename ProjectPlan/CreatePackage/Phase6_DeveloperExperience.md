# Phase 6: Developer Experience

## Objective

Polish the SDK's developer experience with robust error handling, comprehensive type annotations, logging, context manager patterns, serialization utilities, and IDE-friendly features that make py-cml a joy to use.

---

## Task 6.1: Error Handling & Recovery

### Sub-Task 6.1.1: Structured Error Messages

**Architecture**: Every exception should carry actionable information — what went wrong, what the developer can do about it, and the raw server response for debugging.

**Implementation**:
```python
class CMLError(Exception):
    """Base exception for all CML errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code (if from HTTP response).
        response_body: Raw server response dict (if available).
        request_id: Server request ID for support debugging.
        suggestion: Actionable suggestion for the developer.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        full_message = message
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"
        if request_id:
            full_message += f"\n  Request ID: {request_id}"
        super().__init__(full_message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id
        self.suggestion = suggestion

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code})"
        )
```

**Pseudo-code for enhanced error mapping**:
```
FUNCTION _raise_for_status(response):
    IF response.is_success: RETURN

    body = try_parse_json(response)
    request_id = response.headers.get("X-Request-ID")

    MATCH response.status_code:
        401 → raise AuthenticationError(
            "Invalid or missing API key",
            suggestion="Set CML_API_KEY env var or pass api_key= to constructor"
        )
        403 → raise AuthorizationError(
            "Insufficient permissions",
            suggestion="This operation requires admin API key. Set CML_ADMIN_API_KEY."
        )
        404 → raise NotFoundError(
            f"Resource not found: {response.url.path}",
            suggestion="Check that the CML server version supports this endpoint"
        )
        422 → raise ValidationError(
            f"Request validation failed: {body.get('detail', '')}",
            suggestion="Check request parameters match the API schema"
        )
        429 → raise RateLimitError(
            "Rate limit exceeded",
            retry_after=response.headers.get("Retry-After"),
            suggestion="Reduce request frequency or contact admin"
        )
        503 → raise ServerError(
            "CML server is unavailable",
            suggestion="Check that all backend services (PostgreSQL, Neo4j, Redis) are running"
        )
```

### Sub-Task 6.1.2: Graceful Degradation

**Architecture**: For non-critical failures, provide fallback behavior instead of crashing.

**Pseudo-code**:
```
# Example: read() with graceful degradation
ASYNC METHOD read_safe(query, **kwargs) -> ReadResponse:
    """Read with graceful degradation — returns empty result on failure."""
    TRY:
        RETURN await self.read(query, **kwargs)
    CATCH ConnectionError:
        LOG.warning("CML server unreachable, returning empty context")
        RETURN ReadResponse(
            query=query,
            memories=[],
            total_count=0,
            elapsed_ms=0,
        )
    CATCH TimeoutError:
        LOG.warning("CML request timed out, returning empty context")
        RETURN ReadResponse(query=query, memories=[], total_count=0, elapsed_ms=0)
```

---

## Task 6.2: Logging

### Sub-Task 6.2.1: Structured Logging

**Architecture**: Use Python's standard `logging` module with structured context. Never log secrets (API keys, tokens).

**Implementation** (`src/cml/utils/logging.py`):
```python
"""Logging configuration for py-cml."""

import logging
from typing import Any


logger = logging.getLogger("cml")


def configure_logging(
    level: str = "WARNING",
    handler: logging.Handler | None = None,
) -> None:
    """Configure py-cml logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        handler: Custom handler. Defaults to StreamHandler.

    Example:
        import cml
        cml.configure_logging("DEBUG")  # Enable debug logging
    """
    logger.setLevel(getattr(logging, level.upper()))
    if handler:
        logger.addHandler(handler)
    elif not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(h)


def _redact(value: str, visible_chars: int = 4) -> str:
    """Redact a secret, showing only the last N characters."""
    if len(value) <= visible_chars:
        return "***"
    return f"***{value[-visible_chars:]}"
```

**Pseudo-code for log points**:
```
# Transport layer:
DEBUG: "POST /api/v1/memory/write → 200 (123ms)"
DEBUG: "Retry attempt 2/3 after ServerError, sleeping 2.1s"
WARNING: "Rate limited, retrying after 5.0s"

# Client layer:
INFO: "Connected to CML server at http://localhost:8000"
INFO: "Stored memory: {id} ({chunks_created} chunks)"
DEBUG: "Read query: {query} → {total_count} results in {elapsed_ms}ms"

# NEVER log:
# - API keys (redact to ***abcd)
# - Full memory content (truncate to first 50 chars)
# - Embedding vectors
```

### Sub-Task 6.2.2: Expose configure_logging in Package Init

**Implementation** (`src/cml/__init__.py`):
```python
from cml.utils.logging import configure_logging

__all__ = [
    # ... existing exports
    "configure_logging",
]
```

---

## Task 6.3: Type Annotations & IDE Support

### Sub-Task 6.3.1: Complete Type Stubs

**Architecture**: Every public method should have complete type annotations that provide rich IDE autocomplete and hover documentation.

**Pseudo-code for type coverage**:
```
ENSURE all public methods have:
    1. Parameter types (positional and keyword)
    2. Return type
    3. Docstring with Args, Returns, Raises, Example sections
    4. Optional overload decorators for polymorphic methods

EXAMPLE:
    @overload
    async def read(
        self,
        query: str,
        *,
        format: Literal["llm_context"],
        **kwargs,
    ) -> ReadResponse: ...  # .llm_context is always populated

    @overload
    async def read(
        self,
        query: str,
        *,
        format: Literal["packet", "list"] = "packet",
        **kwargs,
    ) -> ReadResponse: ...

    async def read(self, query, *, format="packet", **kwargs) -> ReadResponse:
        # Implementation
```

### Sub-Task 6.3.2: TypedDict for Unstructured Returns

**Implementation**:
```python
from typing import TypedDict

class ConsolidationResult(TypedDict):
    episodes_sampled: int
    clusters_formed: int
    facts_extracted: int
    migrations_completed: int

class ForgettingResult(TypedDict):
    total_scored: int
    actions: Dict[str, int]
    dry_run: bool
```

### Sub-Task 6.3.3: py.typed and Inline Types

**Implementation**:
```
# Ensure py.typed marker exists at src/cml/py.typed (empty file)
# This tells type checkers (mypy, pyright) that this package ships types

# All modules use from __future__ import annotations for forward refs
# All public classes use __slots__ where appropriate for memory efficiency
```

---

## Task 6.4: Context Manager Patterns

### Sub-Task 6.4.1: Client Lifecycle

**Architecture**: Ensure both sync and async clients properly manage connection lifecycle through context managers.

**Implementation**:
```python
# Async context manager (already defined)
async with AsyncCognitiveMemoryLayer(api_key="...") as memory:
    await memory.write("...")
    # Transport is automatically closed on exit

# Sync context manager (already defined)
with CognitiveMemoryLayer(api_key="...") as memory:
    memory.write("...")
    # Transport is automatically closed on exit

# Without context manager (manual close required)
memory = CognitiveMemoryLayer(api_key="...")
try:
    memory.write("...")
finally:
    memory.close()
```

### Sub-Task 6.4.2: Session Context Manager

**Architecture**: Provide a session context manager that auto-creates and manages a session.

**Implementation**:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def session(
    self,
    *,
    name: Optional[str] = None,
    ttl_hours: int = 24,
):
    """Create a session-scoped memory context.

    All operations within the context use the same session_id.

    Example:
        async with memory.session(name="onboarding") as sess:
            await sess.write("User prefers dark mode")
            await sess.write("User is a Python developer")
            result = await sess.read("user preferences")
            # sess.session_id is auto-generated
    """
    session_response = await self.create_session(name=name, ttl_hours=ttl_hours)
    session_id = session_response.session_id

    class SessionScope:
        def __init__(scope_self):
            scope_self.session_id = session_id

        async def write(scope_self, content, **kwargs):
            return await self.write(content, session_id=session_id, **kwargs)

        async def read(scope_self, query, **kwargs):
            return await self.read(query, **kwargs)

        async def turn(scope_self, user_message, **kwargs):
            return await self.turn(
                user_message, session_id=session_id, **kwargs
            )

    yield SessionScope()
```

**Pseudo-code**:
```
ASYNC CONTEXT MANAGER session(name?, ttl_hours?) -> SessionScope:
    1. Create session via API → get session_id
    2. Create SessionScope wrapper that injects session_id into all writes/turns
    3. YIELD SessionScope
    4. On exit: session expires naturally via TTL (no explicit cleanup needed)
```

---

## Task 6.5: Serialization Utilities

### Sub-Task 6.5.1: JSON Serialization Helpers

**Implementation** (`src/cml/utils/serialization.py`):
```python
"""Serialization utilities for py-cml."""

import json
from datetime import datetime
from uuid import UUID
from typing import Any


class CMLJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for CML types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def serialize_for_api(data: dict) -> dict:
    """Prepare a dict for API transmission.

    - Converts UUID to string
    - Converts datetime to ISO format
    - Removes None values
    """
    result = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, UUID):
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_for_api(value)
        elif isinstance(value, list):
            result[key] = [
                serialize_for_api(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            result[key] = value
    return result
```

### Sub-Task 6.5.2: Response Pretty-Printing

**Implementation**:
```python
# On ReadResponse:
def __str__(self) -> str:
    """Pretty-print retrieval results."""
    lines = [f"Query: {self.query} ({self.total_count} results, {self.elapsed_ms:.0f}ms)"]
    for mem in self.memories[:10]:
        lines.append(f"  [{mem.type}] {mem.text[:80]}... (rel={mem.relevance:.2f})")
    return "\n".join(lines)

# On WriteResponse:
def __str__(self) -> str:
    return f"WriteResponse(success={self.success}, chunks={self.chunks_created})"

# On StatsResponse:
def __str__(self) -> str:
    return (
        f"Memory Stats: {self.total_memories} total "
        f"({self.active_memories} active, {self.silent_memories} silent, "
        f"{self.archived_memories} archived)"
    )
```

---

## Task 6.6: Connection Pooling & Reuse

### Sub-Task 6.6.1: HTTP/2 Support

**Architecture**: Enable HTTP/2 by default in httpx for connection multiplexing.

**Pseudo-code**:
```
# In HTTPTransport.__init__:
self._client = httpx.Client(
    http2=True,  # Enable HTTP/2 for multiplexing
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    ),
    ...
)

# In AsyncHTTPTransport.__init__:
self._client = httpx.AsyncClient(
    http2=True,
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    ),
    ...
)
```

### Sub-Task 6.6.2: Connection Health Monitoring

**Pseudo-code**:
```
METHOD _ensure_healthy_connection():
    IF connection has been idle > 60 seconds:
        Send lightweight health check
        IF health check fails:
            Reconnect (create new httpx client)
    RETURN client
```

---

## Task 6.7: Deprecation & Versioning

### Sub-Task 6.7.1: API Compatibility Layer

**Architecture**: Provide mechanisms for graceful API evolution.

**Implementation**:
```python
import warnings
from functools import wraps


def deprecated(alternative: str, removal_version: str):
    """Mark a method as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in "
                f"v{removal_version}. Use {alternative} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Usage:
@deprecated("write()", "2.0.0")
def store(self, content: str, **kwargs):
    """Deprecated alias for write()."""
    return self.write(content, **kwargs)
```

### Sub-Task 6.7.2: Server Version Compatibility

**Pseudo-code**:
```
ASYNC METHOD _check_server_version():
    health = await self.health()
    server_version = health.version
    IF server_version < MIN_SUPPORTED_VERSION:
        LOG.warning(
            f"CML server {server_version} may not be fully compatible "
            f"with py-cml {__version__}. Upgrade to >= {MIN_SUPPORTED_VERSION}."
        )
```

---

## Task 6.8: Thread Safety

### Sub-Task 6.8.1: Sync Client Thread Safety

**Architecture**: The sync `CognitiveMemoryLayer` should be thread-safe for use in multi-threaded applications.

**Pseudo-code**:
```
CLASS CognitiveMemoryLayer:
    INIT:
        self._lock = threading.Lock()

    METHOD write(content, **kwargs):
        # httpx.Client is thread-safe by default
        # Pydantic model construction is thread-safe
        # No additional locking needed for stateless operations

    METHOD set_tenant(tenant_id):
        WITH self._lock:
            Update config.tenant_id
            Update transport headers

    NOTE: httpx.Client is already thread-safe.
    Only mutable shared state (tenant_id changes) needs locking.
```

### Sub-Task 6.8.2: Async Client Event Loop Safety

**Pseudo-code**:
```
CLASS AsyncCognitiveMemoryLayer:
    # asyncio operations are inherently single-threaded per event loop.
    # No additional synchronization needed.
    # Users should not share an async client across multiple event loops.

    METHOD _ensure_same_loop():
        IF current event loop != creation event loop:
            raise RuntimeError(
                "AsyncCognitiveMemoryLayer must be used "
                "in the same event loop it was created in."
            )
```

---

## Acceptance Criteria

- [ ] All exceptions include actionable suggestions
- [ ] API keys are never logged (redacted to `***xxxx`)
- [ ] `configure_logging()` available at package level
- [ ] Debug logging shows request/response timing
- [ ] All public methods have complete type annotations
- [ ] `py.typed` marker enables IDE type checking
- [ ] Context managers properly close connections
- [ ] Session context manager auto-creates and scopes sessions
- [ ] Serialization handles UUID, datetime, and nested dicts
- [ ] Response objects have human-readable `__str__` methods
- [ ] HTTP/2 enabled for connection multiplexing
- [ ] Deprecation decorator produces proper warnings
- [ ] Sync client is thread-safe
- [ ] Async client validates event loop consistency
