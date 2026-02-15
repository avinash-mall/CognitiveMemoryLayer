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

- **write(content, \*, context_tags, session_id, memory_type, namespace, metadata, turn_id, agent_id, timestamp, eval_mode=False)** → `WriteResponse` — Store new memory. Request `metadata` is merged into the stored record; optional `memory_type` overrides automatic classification. Optional `timestamp` (datetime) for event time; defaults to now. When **eval_mode=True**, the client sends `X-Eval-Mode: true` and the response includes `eval_outcome` ("stored"|"skipped") and `eval_reason` (write-gate reason)—useful for benchmark scripts to aggregate gating statistics.
- **read(query, \*, max_results=10, context_filter, memory_types, since, until, response_format, user_timezone=None)** → `ReadResponse` — Retrieve memories. The server applies `memory_types`, `since`, and `until`. `response_format`: "packet" (categorized), "list" (flat), "llm_context" (markdown string). Optional `user_timezone` (IANA string, e.g. `"America/New_York"`) for timezone-aware "today"/"yesterday" filters in retrieval.
- **read_safe(query, \*\*kwargs)** → `ReadResponse` — Like read; returns empty result on connection/timeout. Accepts same kwargs as read (including `user_timezone`).
- **turn(user_message, \*, assistant_response, session_id, max_context_tokens=1500, timestamp, user_timezone=None)** → `TurnResponse` — Process a turn; retrieve context and optionally store exchange. Optional `timestamp` (datetime) for event time; defaults to now. Optional `user_timezone` for retrieval "today"/"yesterday" (e.g. `"America/New_York"`).
- **update(memory_id, \*, text, confidence, importance, metadata, feedback)** → `UpdateResponse` — Update an existing memory.
- **forget(\*, memory_ids, query, before, action="delete")** → `ForgetResponse` — Forget memories. At least one of memory_ids, query, before required.
- **stats()** → `StatsResponse` — Memory statistics.
- **health()** → `HealthResponse` — Server health check.
- **get_context(query, \*, max_results=10, ...)** → `str` — Formatted LLM context string.
- **create_session(\*, name, ttl_hours=24, metadata)** → `SessionResponse`
- **get_session_context(session_id)** → `SessionContextResponse` — Session context is scoped to memories with that `session_id` when provided.
- **delete_all(\*, confirm=False)** → `int` — Delete all memories; requires confirm=True. Requires admin API key. Server implements DELETE /api/v1/memory/all.
- **remember(content, \*\*kwargs)** — Alias for write. Also accepts `timestamp` and `eval_mode` parameters.
- **search(query, \*\*kwargs)** — Alias for read.

### Admin & Batch Methods

**Existing admin methods:**

- **consolidate(\*, tenant_id, user_id)** → `dict` — Trigger memory consolidation (episodic → semantic). Requires admin API key.
- **run_forgetting(\*, tenant_id, user_id, dry_run=True, max_memories=5000)** → `dict` — Trigger active forgetting cycle.
- **reconsolidate(\*, tenant_id, user_id)** → `dict` — Release all labile state for a tenant (no belief revision). Requires admin API key.
- **batch_write(items, \*, session_id, namespace)** → `list[WriteResponse]` — Write multiple memories sequentially.
- **batch_read(queries, \*, max_results, response_format)** → `list[ReadResponse]` — Execute multiple read queries.
- **list_tenants()** → `list[dict]` — List all tenants with memory/fact/event counts and last activity.
- **get_events(\*, limit, page, event_type, since)** → `dict` — Query the event log with pagination.
- **component_health()** → `dict` — Detailed health status of all CML components.
- **with_namespace(namespace)** → `NamespacedClient` — Create a namespace-scoped view.
- **iter_memories(\*, memory_types, status, batch_size)** → `Iterator[MemoryItem]` — Paginated iteration over memories.

**New dashboard admin methods (v1.1.0):**

- **get_sessions(\*, tenant_id)** → `dict` — List active sessions from Redis with TTL and memory counts per session.
- **get_rate_limits()** → `dict` — Current rate-limit bucket usage per API key with remaining capacity and TTL.
- **get_request_stats(\*, hours=24)** → `dict` — Hourly request volume over the last N hours (1–48).
- **get_graph_stats()** → `dict` — Knowledge graph statistics from Neo4j (total nodes, edges, entity types).
- **explore_graph(\*, tenant_id, entity, scope_id="default", depth=2)** → `dict` — Explore the neighborhood of an entity in the knowledge graph (1–5 hops).
- **search_graph(query, \*, tenant_id, limit=25)** → `dict` — Search entities by name pattern.
- **get_config()** → `dict` — Read-only application configuration snapshot with secrets masked.
- **update_config(updates)** → `dict` — Set runtime configuration overrides stored in Redis.
- **get_labile_status(\*, tenant_id)** → `dict` — Reconsolidation / labile memory status per tenant.
- **test_retrieval(query, \*, tenant_id, max_results=10, context_filter, memory_types, response_format="list")** → `dict` — Test retrieval via the dashboard API; returns scored memories and optional LLM context.
- **get_jobs(\*, tenant_id, job_type, limit=50)** → `dict` — List recent consolidation/forgetting/reconsolidation job history with status and results.
- **bulk_memory_action(memory_ids, action)** → `dict` — Apply "archive", "silence", or "delete" to multiple memories in bulk.

All admin methods require `CML_ADMIN_API_KEY` to be configured. They are available on both `CognitiveMemoryLayer` / `AsyncCognitiveMemoryLayer` and their `NamespacedClient` / `AsyncNamespacedClient` wrappers.

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

### Eval mode (write gate)

For benchmark or evaluation scripts, use **eval_mode=True** so the server returns whether each write was stored or skipped and why:

```python
resp = memory.write("User said okay.", eval_mode=True)
print(resp.eval_outcome)  # "stored" or "skipped"
print(resp.eval_reason)   # e.g. "1 chunk(s) stored" or "Below novelty threshold: ..."
```

Use this to aggregate gating statistics (e.g. stored vs skipped counts and skip reasons) when running evaluations. Works with sync, async, and embedded clients.

## AsyncCognitiveMemoryLayer

Same methods as sync client, all async. Use `async with AsyncCognitiveMemoryLayer(...) as memory:` then `await memory.write(...)` etc.

## EmbeddedCognitiveMemoryLayer

Same API as async client. Use `async with EmbeddedCognitiveMemoryLayer()` or pass `db_path` for persistence. Only lite mode (SQLite + local embeddings) is implemented.

## Models

- **MemoryType** — EPISODIC_EVENT, SEMANTIC_FACT, PREFERENCE, CONSTRAINT, CONVERSATION, MESSAGE, etc.
- **MemoryStatus** — ACTIVE, SILENT, COMPRESSED, ARCHIVED, DELETED
- **MemoryItem** — id, text, type, confidence, relevance, timestamp, metadata
- **WriteResponse** — success, memory_id, chunks_created, message; when eval_mode was used, optional eval_outcome ("stored"|"skipped") and eval_reason.
- **ReadResponse** — context, memories; when format is "packet": categorized into `facts`, `preferences`, `episodes`, and `constraints` (list of `MemoryItem`, default empty). The `constraints` field contains cognitive constraints (goals, values, policies, states, causal rules) extracted and stored by the server when `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` is true.
- **TurnResponse, UpdateResponse, ForgetResponse, StatsResponse, HealthResponse, SessionResponse, SessionContextResponse** — See docstrings in cml.models.

## Exceptions

All inherit from **CMLError**. **AuthenticationError** (401), **AuthorizationError** (403), **NotFoundError** (404), **ValidationError** (422), **RateLimitError** (429, has retry_after), **ServerError** (5xx), **CMLConnectionError**, **CMLTimeoutError** (also exported as **ConnectionError** and **TimeoutError** for backward compatibility; prefer the CML-prefixed names to avoid shadowing Python builtins). Use str(e) for full message with suggestion.

## Configuration

**CMLConfig** — api_key, base_url, tenant_id, timeout, max_retries, retry_delay, admin_api_key, verify_ssl. See [Configuration](configuration.md) for env vars.
