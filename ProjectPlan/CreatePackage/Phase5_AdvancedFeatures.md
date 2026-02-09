# Phase 5: Advanced Features

## Objective

Implement advanced memory management features including admin operations (consolidation, forgetting), batch operations, streaming, tenant management, and framework integrations that make py-cml production-ready.

---

## Task 5.1: Admin Operations

### Sub-Task 5.1.1: Trigger Consolidation

**Architecture**: Expose the server's consolidation endpoint (episodic-to-semantic migration) via the client. Requires admin API key.

**Implementation**:
```python
async def consolidate(
    self,
    *,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Trigger memory consolidation (episodic → semantic migration).

    Runs the "sleep cycle" — samples recent episodes, clusters them,
    extracts semantic gists, and migrates to the neocortical store.

    Requires admin API key.

    Args:
        tenant_id: Target tenant (defaults to configured tenant).
        user_id: Optional specific user within tenant.

    Returns:
        Dict with consolidation results:
            - episodes_sampled: int
            - clusters_formed: int
            - facts_extracted: int
            - migrations_completed: int
    """
    payload = {
        "tenant_id": tenant_id or self._config.tenant_id,
    }
    if user_id:
        payload["user_id"] = user_id

    return await self._transport.request(
        "POST",
        "/api/v1/admin/consolidate",
        json=payload,
        use_admin_key=True,
    )
```

**Pseudo-code**:
```
ASYNC METHOD consolidate(tenant_id?, user_id?) -> Dict:
    1. Build payload with tenant_id (default to config)
    2. POST /api/v1/admin/consolidate (admin API key)
    3. Server runs consolidation pipeline:
       a. EpisodeSampler: Select recent episodes
       b. SemanticClusterer: Group similar memories
       c. Summarizer: Extract semantic gists
       d. SchemaAligner: Align with existing schemas
       e. Migrator: Move hippo → neocortex
    4. RETURN results dict
```

### Sub-Task 5.1.2: Trigger Active Forgetting

**Implementation**:
```python
async def run_forgetting(
    self,
    *,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    dry_run: bool = True,
    max_memories: int = 5000,
) -> Dict[str, Any]:
    """Trigger active forgetting cycle.

    Scores all memories by relevance and applies forgetting actions:
    KEEP (>0.7), DECAY (>0.5), SILENCE (>0.3), COMPRESS (>0.1), DELETE (<=0.1).

    Requires admin API key.

    Args:
        tenant_id: Target tenant (defaults to configured tenant).
        user_id: Optional specific user.
        dry_run: If True, only report what would happen (default: True).
        max_memories: Maximum memories to process.

    Returns:
        Dict with forgetting results:
            - total_scored: int
            - actions: Dict[str, int] (e.g., {"KEEP": 100, "DECAY": 20, ...})
            - dry_run: bool
    """
    payload = {
        "tenant_id": tenant_id or self._config.tenant_id,
        "dry_run": dry_run,
        "max_memories": max_memories,
    }
    if user_id:
        payload["user_id"] = user_id

    return await self._transport.request(
        "POST",
        "/api/v1/admin/forget",
        json=payload,
        use_admin_key=True,
    )
```

**Pseudo-code**:
```
ASYNC METHOD run_forgetting(dry_run=True, max_memories=5000) -> Dict:
    1. Build payload
    2. POST /api/v1/admin/forget (admin API key)
    3. Server runs forgetting pipeline:
       a. RelevanceScorer: Score each memory
       b. ForgettingExecutor: Determine action per score threshold
       c. IF NOT dry_run:
          - DECAY: Reduce confidence
          - SILENCE: Set status = SILENT
          - COMPRESS: LLM-summarize content
          - DELETE: Hard or soft delete
       d. IF dry_run: Only return planned actions
    4. RETURN results with action counts
```

---

## Task 5.2: Batch Operations

### Sub-Task 5.2.1: Batch Write

**Architecture**: Write multiple memories in a single call to reduce HTTP overhead. Server processes them sequentially but in one request.

**Implementation**:
```python
async def batch_write(
    self,
    items: List[Dict[str, Any]],
    *,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> List[WriteResponse]:
    """Write multiple memories in a single request.

    Args:
        items: List of dicts, each containing at minimum 'content'.
            Optional keys: context_tags, memory_type, metadata, agent_id.
        session_id: Shared session for all items.
        namespace: Shared namespace for all items.

    Returns:
        List of WriteResponse, one per item.

    Example:
        results = await memory.batch_write([
            {"content": "User likes Italian food", "context_tags": ["preference"]},
            {"content": "User works at Acme Corp", "context_tags": ["personal"]},
            {"content": "User lives in Paris", "context_tags": ["personal"]},
        ])
    """
    responses = []
    for item in items:
        resp = await self.write(
            content=item["content"],
            context_tags=item.get("context_tags"),
            session_id=session_id or item.get("session_id"),
            memory_type=item.get("memory_type"),
            namespace=namespace or item.get("namespace"),
            metadata=item.get("metadata"),
            agent_id=item.get("agent_id"),
        )
        responses.append(resp)
    return responses
```

**Pseudo-code**:
```
ASYNC METHOD batch_write(items, session_id?, namespace?) -> List[WriteResponse]:
    responses = []
    FOR EACH item IN items:
        resp = await self.write(
            content=item["content"],
            **merge(defaults, item_options)
        )
        responses.append(resp)
    RETURN responses

    FUTURE OPTIMIZATION:
    - Send all items in single POST /api/v1/memory/batch_write
    - Server processes in parallel
    - Return all results at once
```

### Sub-Task 5.2.2: Batch Read

**Implementation**:
```python
async def batch_read(
    self,
    queries: List[str],
    *,
    max_results: int = 10,
    format: Literal["packet", "list", "llm_context"] = "packet",
) -> List[ReadResponse]:
    """Execute multiple read queries.

    Args:
        queries: List of search queries.
        max_results: Max results per query.
        format: Response format for all queries.

    Returns:
        List of ReadResponse, one per query.
    """
    import asyncio
    tasks = [
        self.read(query, max_results=max_results, format=format)
        for query in queries
    ]
    return await asyncio.gather(*tasks)
```

**Pseudo-code**:
```
ASYNC METHOD batch_read(queries, **options) -> List[ReadResponse]:
    1. Create async tasks for each query
    2. Execute all concurrently with asyncio.gather()
    3. RETURN list of results in query order
```

### Sub-Task 5.2.3: Sync Batch Operations

**Implementation**:
```python
# Sync batch_write — sequential execution
def batch_write(self, items, **kwargs) -> List[WriteResponse]:
    return [self.write(item["content"], **kwargs) for item in items]

# Sync batch_read — sequential execution
def batch_read(self, queries, **kwargs) -> List[ReadResponse]:
    return [self.read(query, **kwargs) for query in queries]
```

---

## Task 5.3: Tenant Management

### Sub-Task 5.3.1: Multi-Tenant Support

**Architecture**: Support switching tenants on the same client instance, and listing tenant information (admin only).

**Implementation**:
```python
def set_tenant(self, tenant_id: str) -> None:
    """Switch the active tenant for subsequent operations.

    Args:
        tenant_id: New tenant identifier.
    """
    self._config.tenant_id = tenant_id
    # Update transport headers
    self._transport.update_header("X-Tenant-ID", tenant_id)


@property
def tenant_id(self) -> str:
    """Get the current active tenant ID."""
    return self._config.tenant_id


async def list_tenants(self) -> List[Dict[str, Any]]:
    """List all tenants and their memory counts (admin only).

    Returns:
        List of tenant info dicts with tenant_id, memory_count, etc.
    """
    data = await self._transport.request(
        "GET",
        "/api/v1/admin/tenants",
        use_admin_key=True,
    )
    return data.get("tenants", [])
```

**Pseudo-code**:
```
METHOD set_tenant(tenant_id):
    Update config.tenant_id
    Update X-Tenant-ID header on transport

PROPERTY tenant_id -> str:
    Return config.tenant_id

ASYNC METHOD list_tenants() -> List[Dict]:
    GET /api/v1/admin/tenants (admin key)
    RETURN list of {tenant_id, memory_count, fact_count, event_count}
```

---

## Task 5.4: Event Log Access

### Sub-Task 5.4.1: Query Events

**Implementation**:
```python
async def get_events(
    self,
    *,
    limit: int = 50,
    page: int = 1,
    event_type: Optional[str] = None,
    since: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Query the event log (admin only).

    Args:
        limit: Results per page.
        page: Page number.
        event_type: Filter by event type (e.g., "memory_op", "consolidation").
        since: Only events after this time.

    Returns:
        Paginated event list with items, total, page, per_page, total_pages.
    """
    params = {"per_page": limit, "page": page}
    if event_type:
        params["event_type"] = event_type
    if since:
        params["since"] = since.isoformat()

    return await self._transport.request(
        "GET",
        "/api/v1/admin/events",
        params=params,
        use_admin_key=True,
    )
```

---

## Task 5.5: Component Health Monitoring

### Sub-Task 5.5.1: Detailed Health Check

**Implementation**:
```python
async def component_health(self) -> Dict[str, Any]:
    """Get detailed health status of all CML components.

    Returns:
        Dict with component statuses:
            - postgres: {status, latency_ms, details}
            - neo4j: {status, latency_ms, details}
            - redis: {status, latency_ms, details}
    """
    return await self._transport.request(
        "GET",
        "/api/v1/admin/components",
        use_admin_key=True,
    )
```

---

## Task 5.6: Framework Integrations

### Sub-Task 5.6.1: OpenAI Integration Helper

**Architecture**: Provide a helper class that wraps OpenAI chat completions with CML memory.

**Implementation** (`src/cml/integrations/openai_helper.py`):
```python
"""OpenAI integration helper for py-cml."""

from typing import Any, Dict, List, Optional


class CMLOpenAIHelper:
    """Helper for integrating CML memory with OpenAI chat completions.

    Example:
        from openai import OpenAI
        from cml import CognitiveMemoryLayer
        from cml.integrations import CMLOpenAIHelper

        memory = CognitiveMemoryLayer(api_key="...")
        openai_client = OpenAI()
        helper = CMLOpenAIHelper(memory, openai_client)

        response = helper.chat("What should I eat tonight?", session_id="s1")
    """

    def __init__(self, memory_client, openai_client, *, model: str = "gpt-4o"):
        self.memory = memory_client
        self.openai = openai_client
        self.model = model

    def chat(
        self,
        user_message: str,
        *,
        session_id: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send a message with automatic memory context.

        1. Retrieves relevant memories
        2. Injects them into the system prompt
        3. Calls OpenAI chat completion
        4. Stores the exchange for future recall
        """
        # 1. Get memory context
        turn_result = self.memory.turn(
            user_message=user_message,
            session_id=session_id,
        )

        # 2. Build messages
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n"
                           f"## Relevant Memories\n{turn_result.memory_context}",
            }
        ]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_message})

        # 3. Call OpenAI
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        assistant_message = response.choices[0].message.content

        # 4. Store the exchange
        self.memory.turn(
            user_message=user_message,
            assistant_response=assistant_message,
            session_id=session_id,
        )

        return assistant_message
```

### Sub-Task 5.6.2: Generic LLM Integration Protocol

**Architecture**: Define a protocol (interface) that any LLM provider integration can implement.

**Pseudo-code**:
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for memory-enhanced LLM providers."""

    def get_context(self, query: str) -> str:
        """Get memory context for a query."""
        ...

    def store_exchange(
        self,
        user_message: str,
        assistant_response: str,
        session_id: str,
    ) -> None:
        """Store a conversation exchange."""
        ...

    def clear_session(self, session_id: str) -> None:
        """Clear a session's memories."""
        ...
```

---

## Task 5.7: Namespace Isolation

### Sub-Task 5.7.1: Namespace-Scoped Operations

**Architecture**: Support namespace prefixing for all operations, allowing logical isolation within a tenant.

**Implementation**:
```python
def with_namespace(self, namespace: str) -> "NamespacedClient":
    """Create a namespace-scoped view of this client.

    All operations through the returned client will be scoped
    to the given namespace.

    Args:
        namespace: Namespace identifier.

    Returns:
        NamespacedClient that adds namespace to all operations.

    Example:
        user_memory = memory.with_namespace("user-123")
        await user_memory.write("Prefers dark mode")
        # Stored with namespace="user-123"
    """
    return NamespacedClient(self, namespace)


class NamespacedClient:
    """Namespace-scoped wrapper around a CML client."""

    def __init__(self, parent, namespace: str):
        self._parent = parent
        self._namespace = namespace

    async def write(self, content: str, **kwargs) -> WriteResponse:
        return await self._parent.write(
            content, namespace=self._namespace, **kwargs
        )

    async def read(self, query: str, **kwargs) -> ReadResponse:
        return await self._parent.read(query, **kwargs)

    # ... delegate all other methods with namespace injected
```

**Pseudo-code**:
```
METHOD with_namespace(namespace) -> NamespacedClient:
    Create wrapper that auto-injects namespace into all write operations
    Read operations can optionally filter by namespace
    RETURN NamespacedClient

CLASS NamespacedClient:
    Wraps parent client
    Adds namespace= param to all write/update calls
    Delegates all other calls to parent
```

---

## Task 5.8: Memory Iteration & Pagination

### Sub-Task 5.8.1: Iterate All Memories

**Architecture**: Provide an async iterator for scanning all memories with pagination.

**Implementation**:
```python
async def iter_memories(
    self,
    *,
    memory_types: Optional[List[MemoryType]] = None,
    status: Optional[str] = "active",
    batch_size: int = 100,
) -> AsyncIterator[MemoryItem]:
    """Iterate over all memories with automatic pagination.

    Args:
        memory_types: Filter by types.
        status: Filter by status.
        batch_size: Items per page.

    Yields:
        MemoryItem for each memory.

    Example:
        async for memory in client.iter_memories():
            print(f"{memory.type}: {memory.text[:50]}")
    """
    page = 1
    while True:
        data = await self._transport.request(
            "GET",
            "/api/v1/admin/memories",
            params={
                "page": page,
                "per_page": batch_size,
                "status": status,
            },
            use_admin_key=True,
        )
        items = data.get("items", [])
        if not items:
            break
        for item in items:
            yield MemoryItem(**item)
        if page >= data.get("total_pages", 1):
            break
        page += 1
```

**Pseudo-code**:
```
ASYNC GENERATOR iter_memories(**filters) -> AsyncIterator[MemoryItem]:
    page = 1
    LOOP:
        GET /api/v1/admin/memories?page={page}&per_page={batch_size}
        IF no items → BREAK
        FOR EACH item → YIELD MemoryItem
        IF page >= total_pages → BREAK
        page += 1
```

---

## Acceptance Criteria

- [ ] `consolidate()` triggers consolidation via admin endpoint
- [ ] `run_forgetting()` triggers forgetting with dry_run support
- [ ] `batch_write()` writes multiple memories efficiently
- [ ] `batch_read()` executes queries concurrently (async) or sequentially (sync)
- [ ] `set_tenant()` switches active tenant
- [ ] `list_tenants()` returns tenant list (admin only)
- [ ] `get_events()` queries event log with pagination
- [ ] `component_health()` returns per-component status
- [ ] `with_namespace()` returns namespace-scoped client
- [ ] `iter_memories()` provides async iterator with pagination
- [ ] OpenAI helper integrates memory into chat completions
- [ ] All admin operations require admin API key
- [ ] All new methods have full docstrings and type annotations
