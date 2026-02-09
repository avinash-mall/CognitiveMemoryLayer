# API Reference

## CognitiveMemoryLayer (Sync Client)

### Constructor

```python
CognitiveMemoryLayer(
    api_key: str | None = None,
    base_url: str = "http://localhost:8000",
    tenant_id: str = "default",
    *,
    config: CMLConfig | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    verify_ssl: bool = True,
)
```

Use `with CognitiveMemoryLayer(...) as memory:` or call `memory.close()` when done.

### Methods

- **write(content, \*, context_tags, session_id, memory_type, namespace, metadata, turn_id, agent_id)** → `WriteResponse` — Store new memory.
- **read(query, \*, max_results=10, context_filter, memory_types, since, until, format)** → `ReadResponse` — Retrieve memories. `format`: "packet", "list", "llm_context".
- **read_safe(query, \*\*kwargs)** → `ReadResponse` — Like read; returns empty result on connection/timeout.
- **turn(user_message, \*, assistant_response, session_id, max_context_tokens=1500)** → `TurnResponse` — Process a turn; retrieve context and optionally store exchange.
- **update(memory_id, \*, text, confidence, importance, metadata, feedback)** → `UpdateResponse` — Update an existing memory.
- **forget(\*, memory_ids, query, before, action="delete")** → `ForgetResponse` — Forget memories. At least one of memory_ids, query, before required.
- **stats()** → `StatsResponse` — Memory statistics.
- **health()** → `HealthResponse` — Server health check.
- **get_context(query, \*, max_results=10, ...)** → `str` — Formatted LLM context string.
- **create_session(\*, name, ttl_hours=24, metadata)** → `SessionResponse`
- **get_session_context(session_id)** → `SessionContextResponse`
- **delete_all(\*, confirm=False)** → `int` — Delete all memories; requires confirm=True.
- **remember(content, \*\*kwargs)** — Alias for write.
- **search(query, \*\*kwargs)** — Alias for read.

Admin/batch: consolidate, run_forgetting, batch_write, batch_read, list_tenants, get_events, component_health, with_namespace(namespace), iter_memories(...).

## AsyncCognitiveMemoryLayer

Same methods as sync client, all async. Use `async with AsyncCognitiveMemoryLayer(...) as memory:` then `await memory.write(...)` etc.

## EmbeddedCognitiveMemoryLayer

Same API as async client. Use `async with EmbeddedCognitiveMemoryLayer()` or pass `db_path` for persistence. Only lite mode (SQLite + local embeddings) is implemented.

## Models

- **MemoryType** — EPISODIC_EVENT, SEMANTIC_FACT, PREFERENCE, CONVERSATION, MESSAGE, etc.
- **MemoryStatus** — ACTIVE, SILENT, COMPRESSED, ARCHIVED, DELETED
- **MemoryItem** — id, text, type, confidence, relevance, timestamp, metadata
- **WriteResponse, ReadResponse, TurnResponse, UpdateResponse, ForgetResponse, StatsResponse, HealthResponse, SessionResponse, SessionContextResponse** — See docstrings in cml.models.

## Exceptions

All inherit from **CMLError**. **AuthenticationError** (401), **AuthorizationError** (403), **NotFoundError** (404), **ValidationError** (422), **RateLimitError** (429, has retry_after), **ServerError** (5xx), **ConnectionError**, **TimeoutError**. Use str(e) for full message with suggestion.

## Configuration

**CMLConfig** — api_key, base_url, tenant_id, timeout, max_retries, retry_delay, admin_api_key, verify_ssl. See [Configuration](configuration.md) for env vars.
