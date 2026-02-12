# API Reference

## CognitiveMemoryLayer (Sync Client)

### Constructor

```python
CognitiveMemoryLayer(
    api_key: str | None = None,
    base_url: str = "",   # from CML_BASE_URL in .env when unset; no hardcoded default
    tenant_id: str = "default",
    *,
    config: CMLConfig | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    verify_ssl: bool = True,
)
```

Set `CML_BASE_URL` and `CML_API_KEY` in `.env` or pass them. Use `with CognitiveMemoryLayer(...) as memory:` or call `memory.close()` when done.

### Methods

- **write(content, \*, context_tags, session_id, memory_type, namespace, metadata, turn_id, agent_id, timestamp)** → `WriteResponse` — Store new memory. Request `metadata` is merged into the stored record; optional `memory_type` overrides automatic classification. Optional `timestamp` (datetime) for event time; defaults to now.
- **read(query, \*, max_results=10, context_filter, memory_types, since, until, response_format)** → `ReadResponse` — Retrieve memories. The server applies `memory_types`, `since`, and `until`. `response_format`: "packet" (categorized), "list" (flat), "llm_context" (markdown string).
- **read_safe(query, \*\*kwargs)** → `ReadResponse` — Like read; returns empty result on connection/timeout.
- **turn(user_message, \*, assistant_response, session_id, max_context_tokens=1500, timestamp)** → `TurnResponse` — Process a turn; retrieve context and optionally store exchange. Optional `timestamp` (datetime) for event time; defaults to now.
- **update(memory_id, \*, text, confidence, importance, metadata, feedback)** → `UpdateResponse` — Update an existing memory.
- **forget(\*, memory_ids, query, before, action="delete")** → `ForgetResponse` — Forget memories. At least one of memory_ids, query, before required.
- **stats()** → `StatsResponse` — Memory statistics.
- **health()** → `HealthResponse` — Server health check.
- **get_context(query, \*, max_results=10, ...)** → `str` — Formatted LLM context string.
- **create_session(\*, name, ttl_hours=24, metadata)** → `SessionResponse`
- **get_session_context(session_id)** → `SessionContextResponse` — Session context is scoped to memories with that `session_id` when provided.
- **delete_all(\*, confirm=False)** → `int` — Delete all memories; requires confirm=True. Requires admin API key. Server implements DELETE /api/v1/memory/all.
- **remember(content, \*\*kwargs)** — Alias for write. Also accepts `timestamp` parameter.
- **search(query, \*\*kwargs)** — Alias for read.

Admin/batch: consolidate, run_forgetting, batch_write, batch_read, list_tenants, get_events, component_health, with_namespace(namespace), iter_memories(...).

### Temporal Fidelity

The optional `timestamp` parameter in `write()`, `turn()`, and `remember()` enables **temporal fidelity** for historical data replay:

```python
from datetime import datetime, timezone

# Store a memory with a specific event timestamp
historical_time = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
memory.write(
    "User mentioned preferring dark mode",
    timestamp=historical_time
)

# Process a turn with a specific timestamp (e.g., for benchmark evaluation)
memory.turn(
    user_message="Hello",
    assistant_response="Hi there!",
    timestamp=historical_time
)
```

When `timestamp` is not provided, it defaults to the current time. This feature is particularly useful for:
- Benchmark evaluations (e.g., Locomo) that replay historical conversations
- Importing historical data with correct event times
- Testing temporal reasoning capabilities

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

All inherit from **CMLError**. **AuthenticationError** (401), **AuthorizationError** (403), **NotFoundError** (404), **ValidationError** (422), **RateLimitError** (429, has retry_after), **ServerError** (5xx), **CMLConnectionError**, **CMLTimeoutError** (also exported as **ConnectionError** and **TimeoutError** for backward compatibility; prefer the CML-prefixed names to avoid shadowing Python builtins). Use str(e) for full message with suggestion.

## Configuration

**CMLConfig** — api_key, base_url, tenant_id, timeout, max_retries, retry_delay, admin_api_key, verify_ssl. See [Configuration](configuration.md) for env vars.
