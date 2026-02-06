# Current Issues - CognitiveMemoryLayer

> **Audit Date:** February 6, 2026
> **Total Issues Found:** 83
> **Breakdown:** 7 Critical | 14 High | 42 Medium | 20 Low

---

## Table of Contents

- [Summary](#summary)
- [Critical Issues](#critical-issues)
- [High Issues](#high-issues)
- [Medium Issues](#medium-issues)
- [Low Issues](#low-issues)
- [Prioritized Action Plan](#prioritized-action-plan)

---

## Summary

| Severity | Count | Description |
|----------|-------|-------------|
| **Critical** | 7 | Data loss, broken functionality, non-functional code |
| **High** | 14 | Security vulnerabilities, data corruption, race conditions |
| **Medium** | 42 | Logic errors, missing validation, deprecated APIs, config issues |
| **Low** | 20 | Code quality, dead code, style inconsistencies |

---

## Critical Issues

### CRIT-01: Celery Beat Schedule Missing Required Task Arguments

- **File:** `src/celery_app.py` (lines 31-37)
- **Description:** The `beat_schedule` for `"forgetting-daily"` does not include `args` or `kwargs`, but the task `run_forgetting_task` requires `tenant_id: str` and `user_id: str`. When Celery Beat fires this task, it fails immediately with `TypeError: run_forgetting_task() missing 2 required positional arguments`.
- **Resolution:** Create a fan-out wrapper task that queries all registered tenants/users and dispatches individual `run_forgetting_task` calls:

```python
@app.task(name="src.celery_app.fan_out_forgetting")
def fan_out_forgetting():
    """Discover all tenants/users and dispatch individual forgetting tasks."""
    for tenant_id, user_id in get_all_tenant_user_pairs():
        run_forgetting_task.delay(tenant_id, user_id)

app.conf.beat_schedule = {
    "forgetting-daily": {
        "task": "src.celery_app.fan_out_forgetting",
        "schedule": 86400.0,
    },
}
```

---

### CRIT-02: Duplicate `is_current=True` Facts on Non-Temporal Update

- **File:** `src/memory/neocortical/fact_store.py` (lines 201-225)
- **Description:** When a fact's value changes and the schema is NOT temporal (or no schema exists), the code creates a new fact version with `is_current=True` without marking the old one as `is_current=False`. This results in two "current" facts with the same key, corrupting all queries that assume at most one current fact per key.
- **Resolution:** Always mark the old fact as non-current when creating a new version:

```python
# Always supersede old fact when value changes
model.is_current = False
model.valid_to = valid_from or datetime.now(timezone.utc)
await session.flush()
```

---

### CRIT-03: `value_type` Produces `"strtype"` for None Values

- **File:** `src/memory/neocortical/fact_store.py` (lines 215, 250)
- **Description:** The expression `type(new_value).__name__.lower().replace("none", "str")` produces `"strtype"` for `None` values because `type(None).__name__` is `"NoneType"`, lowered to `"nonetype"`, and `.replace("none", "str")` yields `"strtype"`.
- **Resolution:** Use an explicit check:

```python
value_type = "str" if new_value is None else type(new_value).__name__.lower()
```

---

### CRIT-04: Archive Operation Causes Permanent Data Loss

- **File:** `src/forgetting/executor.py` (lines 142-146)
- **Description:** `_execute_archive` fetches a record from the primary store and then hard-deletes it, but **never writes it to the archive store**. The record is permanently lost.
- **Resolution:** Save to the archive store before deleting:

```python
record = await self.store.get_by_id(op.memory_id)
if not record:
    return False
if self.archive_store:
    await self.archive_store.upsert(self._to_create_schema(record))
await self.store.delete(op.memory_id, hard=True)
return True
```

---

### CRIT-05: Migration/Model Column Name Mismatch (`user_id` vs `scope_id`)

- **File:** `migrations/versions/001_initial_schema.py` (line 31) vs `src/storage/models.py` (line 26)
- **Description:** Migration 001 creates the `event_log` table with a column named `user_id`. The SQLAlchemy model `EventLogModel` defines `scope_id = Column(String(100), ...)`. No subsequent migration renames `user_id` to `scope_id`. All EventLog queries fail with a column-not-found error.
- **Resolution:** Create a new migration (005) that renames the column:

```python
def upgrade():
    op.alter_column('event_log', 'user_id', new_column_name='scope_id')
    op.drop_index('ix_event_log_tenant_user_time', table_name='event_log')
    op.create_index('ix_event_log_tenant_scope_time', 'event_log',
                     ['tenant_id', 'scope_id', 'created_at'])
```

---

### CRIT-06: `create_session` Sets `expires_at=now` (Session Expires Immediately)

- **File:** `src/api/routes.py` (lines 241-246)
- **Description:** The `create_session` endpoint sets `expires_at=now`, completely ignoring the `body.ttl_hours` field. Every created session is expired the instant it's returned.
- **Resolution:** Calculate the correct expiry:

```python
from datetime import timedelta

expires_at = now + timedelta(hours=body.ttl_hours or 24)
return CreateSessionResponse(
    session_id=session_id,
    created_at=now,
    expires_at=expires_at,
)
```

---

### CRIT-07: `examples/async_usage.py` Is Completely Non-Functional

- **File:** `examples/async_usage.py` (entire file)
- **Description:** Every call to `client.write()` and `client.read()` passes `scope=` and `scope_id=` keyword arguments that do not exist in the `AsyncCognitiveMemoryClient` API. Every function in this file raises `TypeError: unexpected keyword argument` on invocation.
- **Resolution:** Rewrite all calls to use the correct client API parameters (`content`, `session_id`, `memory_type`, `query`, `format`, etc.) matching the holistic tenant-based API.

---

## High Issues

### HIGH-01: Mutable Default Arguments on SQLAlchemy Columns

- **File:** `src/storage/models.py` (lines 33, 55, 64-66, 104, 113)
- **Description:** Using mutable objects (`[]`, `{}`) as `default=` values on SQLAlchemy Column definitions is the classic mutable default bug. All ORM instances sharing the default will share the same Python object, causing cross-instance data corruption.
- **Affected columns:** `memory_ids`, `context_tags`, `entities`, `relations`, `meta`, `evidence_ids`
- **Resolution:** Use callables instead:

```python
memory_ids = Column(ARRAY(UUID(as_uuid=True)), default=list)
context_tags = Column(ARRAY(String), default=list)
entities = Column(JSON, default=list)
relations = Column(JSON, default=list)
meta = Column("metadata", JSON, default=dict)
evidence_ids = Column(ARRAY(String), default=list)
```

---

### HIGH-02: O(n) Full-Table Scan for Reference Counting

- **File:** `src/storage/postgres.py` (lines 262-284)
- **Description:** `count_references_to` scans up to 5,000 records in Python to count references. For a production system this is extremely slow and degrades as data grows.
- **Resolution:** Use a SQL query:

```python
async def count_references_to(self, record_id: UUID) -> int:
    async with self.session_factory() as session:
        q = select(func.count(MemoryRecordModel.id)).where(
            MemoryRecordModel.supersedes_id == record_id
        )
        r = await session.execute(q)
        return r.scalar() or 0
```

---

### HIGH-03: CORS Wildcard with Credentials

- **File:** `src/api/app.py` (lines 43-49)
- **Description:** `allow_origins=["*"]` combined with `allow_credentials=True` allows any origin to make credentialed requests. Starlette reflects the request Origin header, making any site able to make authenticated requests.
- **Resolution:** Specify explicit allowed origins or make them configurable:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### HIGH-04: `HTTPException` Raised Inside Starlette Middleware

- **File:** `src/api/middleware.py`
- **Description:** `RateLimitMiddleware` raises `HTTPException` inside `BaseHTTPMiddleware.dispatch()`. In Starlette, exceptions raised in middleware are not guaranteed to be caught by FastAPI's exception handlers -- they may bubble up as bare 500 errors instead of the intended 429.
- **Resolution:** Return a `JSONResponse` directly:

```python
from starlette.responses import JSONResponse

return JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded. Try again later."}
)
```

---

### HIGH-05: Timing Attack on API Key Comparison

- **File:** `src/api/auth.py` (line 60)
- **Description:** `api_keys.get(api_key)` uses standard dictionary lookup. An attacker can exploit timing differences to brute-force API keys.
- **Resolution:** Use `hmac.compare_digest()`:

```python
import hmac

context = None
for known_key, ctx in api_keys.items():
    if hmac.compare_digest(api_key, known_key):
        context = ctx
        break
```

---

### HIGH-06: Rate Limiter Memory Leak

- **File:** `src/api/middleware.py` (line 54)
- **Description:** The `_buckets` dictionary accumulates entries for every unique tenant ID with no eviction of old entries. Over time in a multi-tenant deployment, this consumes increasing memory.
- **Resolution:** Add periodic cleanup that removes entries older than the rate-limit window:

```python
if len(self._buckets) > 10000:
    cutoff = now - timedelta(minutes=2)
    self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
```

---

### HIGH-07: `create_session` Never Persists the Session

- **File:** `src/api/routes.py` (lines 234-246)
- **Description:** The endpoint generates a UUID and returns it but never stores it anywhere. Session-based endpoints accept any arbitrary `session_id` with no validation. Sessions are essentially decorative.
- **Resolution:** Persist the session in the database or Redis, and validate session IDs in session-scoped endpoints.

---

### HIGH-08: `delete_all` Limited to 10,000 Records (GDPR Gap)

- **File:** `src/memory/orchestrator.py` (line 324)
- **Description:** If a tenant has more than 10,000 records, the remaining records are silently not deleted. This is a data retention issue for GDPR compliance.
- **Resolution:** Loop with pagination until all records are deleted:

```python
while True:
    records = await self.hippocampal.store.scan(tenant_id, limit=1000)
    if not records:
        break
    for r in records:
        await self.hippocampal.store.delete(r.id, hard=True)
        affected += 1
```

---

### HIGH-09: Episode Duplication During Small-Cluster Merge

- **File:** `src/consolidation/clusterer.py` (lines 124-137)
- **Description:** When merging small clusters into large ones, if an episode fails to find a nearest large cluster, the entire small cluster is appended to `large` and the inner loop breaks. Episodes already reassigned to a large cluster are duplicated -- they exist in both their new large cluster and in the re-added small cluster.
- **Resolution:** Track which episodes were already reassigned and remove them from the small cluster before appending it back, or create a new cluster containing only the un-reassigned episodes.

---

### HIGH-10: `_recommend_resolution` Always Returns `"keep_newer"` / Wrong Record Deleted

- **File:** `src/forgetting/interference.py` (line 114) and `src/forgetting/worker.py` (lines 146-150)
- **Description:** The condition `if r1.timestamp > r2.timestamp or r2.timestamp > r1.timestamp` is equivalent to `r1.timestamp != r2.timestamp`, which is almost always `True`, making `"merge"` unreachable. Additionally, `_plan_duplicate_resolution` in `worker.py` always deletes `dup.interfering_memory_id` without checking which record is actually newer or has higher confidence.
- **Resolution:** Fix the resolution logic to determine and return which specific record to keep/delete based on actual timestamp/confidence comparison.

---

### HIGH-11: Race Condition in `get_session`

- **File:** `src/reconsolidation/labile_tracker.py` (lines 136-144)
- **Description:** All other methods (`mark_labile`, `get_labile_memories`, `release_labile`) acquire `self._lock` before accessing `self._sessions`. `get_session` does not, creating a race condition where it could read a partially-written session.
- **Resolution:** Add `async with self._lock:` around the dict access.

---

### HIGH-12: `_apply_operation` Silently Swallows All Exceptions

- **File:** `src/reconsolidation/service.py` (lines 205-206)
- **Description:** `except Exception: return False` catches every error (database failures, constraint violations, serialization errors) silently. The caller only sees `success: False` with no indication of what went wrong.
- **Resolution:** Log the exception:

```python
except Exception as e:
    logger.error("revision_operation_failed", op_type=op.op_type.value, error=str(e))
    return False
```

---

### HIGH-13: LangChain Integration Example Is Broken

- **File:** `examples/langchain_integration.py`
- **Description:** Multiple critical issues: (1) Uses deprecated LangChain 0.1.x imports that fail with 0.2+, (2) Uses Pydantic v1 `Config` class in a v2 context, (3) `chat_memory` property returns `None` causing `AttributeError` when LangChain internals call methods on it.
- **Resolution:** Update all imports to `langchain_core`/`langchain_community`, use `model_config = ConfigDict(...)` for Pydantic v2, provide a real `ChatMessageHistory` implementation or extend `BaseMemory` instead of `BaseChatMemory`.

---

### HIGH-14: `AsyncCognitiveMemoryClient` Is Incomplete

- **File:** `examples/memory_client.py` (lines 335-448)
- **Description:** The async client is missing `stats()`, `update()`, `forget()`, and `process_turn()` methods that exist in the sync `CognitiveMemoryClient`. Users get `AttributeError` when trying to use these.
- **Resolution:** Add the missing methods mirroring the sync client implementations with `async`/`await`.

---

## Medium Issues

### MED-01: Nested Settings Classes Inherit from `BaseSettings`

- **File:** `src/core/config.py` (lines 23-66)
- **Description:** `DatabaseSettings`, `EmbeddingSettings`, `LLMSettings`, `MemorySettings`, and `AuthSettings` all inherit from `BaseSettings`. Since they are nested inside `Settings` (which uses `env_nested_delimiter="__"`), each nested class independently reads from environment variables without a prefix, causing confusing dual-lookup precedence.
- **Resolution:** Change nested classes from `BaseSettings` to `BaseModel`.

---

### MED-02: Hardcoded Default Neo4j Password

- **File:** `src/core/config.py` (line 29)
- **Description:** `neo4j_password: str = Field(default="password")` ships a default credential. If the env var is not set, the application silently uses a weak password.
- **Resolution:** Set the default to `""` and validate at startup that a password was provided.

---

### MED-03: Widespread `datetime.utcnow()` Deprecation

- **Files:** `src/core/schemas.py` (lines 66-67, 135, 165), `src/storage/models.py` (lines 36, 69, 117, 118), `src/storage/neo4j.py` (lines 82, 130), `src/api/middleware.py` (line 71), `src/memory/neocortical/schemas.py` (lines 60-61), `src/memory/tool_memory.py` (line 43), `src/consolidation/sampler.py` (lines 53, 84-85), and others
- **Description:** `datetime.utcnow()` is deprecated since Python 3.12 and returns naive datetimes. Some files (e.g., `postgres.py`) already use the correct `datetime.now(timezone.utc)`, creating inconsistency.
- **Resolution:** Replace all occurrences with `datetime.now(timezone.utc)` project-wide.

---

### MED-04: Hardcoded Vector Dimension

- **File:** `src/storage/models.py` (line 62)
- **Description:** `embedding = Column(Vector(1536))` hardcodes the dimension. If `EmbeddingSettings.dimensions` is changed, the database schema and config will be out of sync.
- **Resolution:** Either generate the dimension from config at migration time, or add a startup check that validates config matches the DB schema.

---

### MED-05: Singleton Leaks Resources on Partial Initialization

- **File:** `src/storage/connection.py` (lines 18-44)
- **Description:** If the PostgreSQL engine creates successfully but the Neo4j driver fails, the PG engine is leaked (never disposed). The `_instance` is never set so next call retries, but the leaked engine holds open connections.
- **Resolution:** Wrap initialization in try/except and clean up all already-created resources on failure.

---

### MED-06: `append` Commits Immediately, Breaking Transactions

- **File:** `src/storage/event_log.py` (line 38)
- **Description:** `await self.session.commit()` inside `append` means the caller cannot compose multiple operations into a single transaction.
- **Resolution:** Remove the `commit()` and let the caller manage the transaction boundary, or accept an optional `auto_commit` parameter.

---

### MED-07: `merge_node`/`merge_edge` Signatures Don't Match Abstract Base

- **File:** `src/storage/neo4j.py` (lines 69, 114)
- **Description:** Both methods add a `namespace: Optional[str] = None` parameter not present in `GraphStoreBase`, breaking the Liskov Substitution Principle.
- **Resolution:** Either add `namespace` to the abstract base class, or move it into the `properties` dict.

---

### MED-08: Silent Exception Swallowing in Neo4j Store

- **File:** `src/storage/neo4j.py` (lines 221, 301)
- **Description:** Bare `except Exception` catches all errors (including programming bugs like `TypeError`, `AttributeError`) and silently falls back to a simpler query.
- **Resolution:** Catch only the specific `neo4j.exceptions.ClientError` for APOC/GDS unavailability.

---

### MED-09: f-String Cypher Query Construction

- **File:** `src/storage/neo4j.py` (lines 132, 183, 365)
- **Description:** `rel_type` and `max_depth` are interpolated directly into Cypher queries via f-strings. While `_sanitize_rel_type` strips dangerous characters, the pattern is fragile and could become a Cypher injection vector.
- **Resolution:** Add explicit validation after sanitization (`assert rel_type.isidentifier()`) and validate `max_depth` is a positive integer.

---

### MED-10: `vector_search` Missing Timezone Normalization

- **File:** `src/storage/postgres.py` (lines 194-196)
- **Description:** The `scan` method calls `_naive_utc(filters["since"])` to normalize timezone-aware datetimes, but `vector_search` passes `since`/`until` directly. Timezone-aware datetime comparison with naive DB timestamps will fail or return incorrect results.
- **Resolution:** Apply `_naive_utc()` to filter datetimes in `vector_search` as well.

---

### MED-11: No Fallback for Invalid `MemoryStatus`

- **File:** `src/storage/postgres.py` (line 322)
- **Description:** `MemoryType(model.type)` has a try/except with a fallback to `EPISODIC_EVENT`, but `MemoryStatus(model.status)` has no such protection. An invalid status string raises `ValueError` and crashes the entire query.
- **Resolution:** Add a try/except fallback: `try: status = MemoryStatus(model.status) except ValueError: status = MemoryStatus.ACTIVE`.

---

### MED-12: No Defensive Check on Provenance Deserialization

- **File:** `src/storage/postgres.py` (line 324)
- **Description:** `Provenance(**model.provenance)` crashes if `model.provenance` is `None`, empty, or has missing required fields.
- **Resolution:** Add try/except with a fallback default Provenance.

---

### MED-13: `asyncio.run()` Per Celery Task Invocation

- **File:** `src/celery_app.py` (line 86)
- **Description:** `asyncio.run(_run())` creates and destroys an event loop on every task execution. This is expensive and precludes connection pool reuse across tasks. If the worker uses `gevent`/`eventlet`, this conflicts with the existing event loop.
- **Resolution:** Use a persistent event loop per worker or restructure the database layer to support synchronous access for Celery tasks.

---

### MED-14: Async Connections Created in Sync Celery Context

- **File:** `src/celery_app.py` (lines 59-60)
- **Description:** `DatabaseManager.get_instance()` creates async drivers at instantiation time from within a synchronous Celery task. The async connections may not be safe to share across different `asyncio.run()` event loops.
- **Resolution:** Create the `DatabaseManager` inside the async `_run()` function, or use a per-event-loop singleton.

---

### MED-15: Search Returns Stale Access Tracking Data (N+1 Queries)

- **File:** `src/memory/hippocampal/store.py` (lines 161-170)
- **Description:** After a vector search, each result's `access_count` and `last_accessed_at` are updated individually (N+1 problem), but the returned `results` list still contains the pre-update values.
- **Resolution:** Use a batch update (single `UPDATE ... WHERE id IN (...)` statement) and refresh the returned records.

---

### MED-16: `_known_facts` Cache Is Dead Code

- **File:** `src/memory/hippocampal/write_gate.py` (lines 62-65, 193-194)
- **Description:** The `_known_facts` set is initialized and can be populated via `add_known_fact()`, but is never read in `evaluate()`, `_compute_novelty()`, or any other method. It has zero effect on write gate decisions.
- **Resolution:** Either integrate `_known_facts` into the novelty/importance computation, or remove it.

---

### MED-17: `min_novelty` Config Field Unused

- **File:** `src/memory/hippocampal/write_gate.py` (line 34)
- **Description:** `WriteGateConfig.min_novelty = 0.2` is defined but never referenced in any logic.
- **Resolution:** Either add a novelty check (`if novelty < self.config.min_novelty: skip`) or remove the field.

---

### MED-18: `get_stats` Incomplete for Large Tenants

- **File:** `src/memory/orchestrator.py` (line 351)
- **Description:** The `total` count may be much higher than 1,000, but the per-type breakdown only considers the first 1,000 records.
- **Resolution:** Use a `GROUP BY` count query on the storage layer, or paginate the scan.

---

### MED-19: `forget` Double-Counts for Overlapping Criteria

- **File:** `src/memory/orchestrator.py` (lines 248-271)
- **Description:** When multiple criteria (`memory_ids`, `query`, `before`) are provided, they're applied independently without deduplication. A memory matching both gets counted and deleted twice.
- **Resolution:** Collect all target IDs into a set first, then delete once.

---

### MED-20: `threading.Lock` in Async Context

- **File:** `src/memory/sensory/buffer.py` (lines 9, 152)
- **Description:** `SensoryBuffer` uses `threading.Lock` but operates in an async context. This can block the event loop.
- **Resolution:** Use `asyncio.Lock` for async paths, or separate sync and async locking.

---

### MED-21: `inactive_seconds` Parameter Ignored

- **File:** `src/memory/sensory/manager.py` (lines 61-66)
- **Description:** The `inactive_seconds` parameter is accepted but never used. The method only checks `buffer.is_empty`.
- **Resolution:** Track last activity time per buffer and compare against `inactive_seconds`.

---

### MED-22: Non-Deterministic `hash()` for Chunk IDs

- **File:** `src/memory/working/chunker.py` (line 234)
- **Description:** `RuleBasedChunker` uses `hash(sentence) % 10000` for chunk IDs, but Python's `hash()` is randomized across sessions via `PYTHONHASHSEED`. `SemanticChunker` correctly uses `hashlib.sha256`.
- **Resolution:** Use `hashlib.sha256(sentence.encode()).hexdigest()[:8]` for deterministic IDs.

---

### MED-23: Tautological Entity Extraction Condition

- **File:** `src/retrieval/classifier.py` (lines 170-173)
- **Description:** The guard `(prev_ends or i > 0)` is always `True`, so every capitalized word of length > 1 is treated as an entity, including sentence-initial words ("The", "What", "How").
- **Resolution:** Change to `if w and w[0].isupper() and len(w) > 1 and not prev_ends:` to exclude sentence-start words.

---

### MED-24: `parallel_groups` References Out-of-Bounds Index

- **File:** `src/retrieval/planner.py` (line 101)
- **Description:** When `MULTI_HOP` has no entities and only 2 steps are created, `parallel_groups = [[0, 1], [2]]` references index 2 which doesn't exist. The retriever guards against this, but the plan is semantically incorrect.
- **Resolution:** Change to `parallel_groups = [[0, 1, 2]] if len(steps) == 3 else [[0, 1]]`.

---

### MED-25: `_retrieve_vector` Ignores `min_confidence`

- **File:** `src/retrieval/retriever.py` (lines 148-159)
- **Description:** The code checks `step.min_confidence > 0` to conditionally build a `filters` dict, but never actually adds the `min_confidence` value to it.
- **Resolution:** Add `filters["min_confidence"] = step.min_confidence` when the condition is met.

---

### MED-26: `_llm_classify` Doesn't Catch LLM Network Errors

- **File:** `src/retrieval/classifier.py` (line 141)
- **Description:** The try/except only catches `json.JSONDecodeError, ValueError, TypeError`. LLM API failures (network timeout, rate limit) propagate uncaught.
- **Resolution:** Add a broader except clause or catch the LLM client's specific exceptions.

---

### MED-27: False Positive Preference Detection in Conflict Detector

- **File:** `src/reconsolidation/conflict_detector.py` (lines 140-158)
- **Description:** If both old and new statements contain any preference keyword (like, prefer, enjoy, etc.), the result is `TEMPORAL_CHANGE` even when the statements are about completely different topics.
- **Resolution:** Add a topic overlap check before classifying as temporal change.

---

### MED-28: In-Memory-Only Labile State

- **File:** `src/reconsolidation/labile_tracker.py`
- **Description:** `LabileStateTracker` stores all state in instance-level dicts. In a multi-worker deployment, each worker has its own tracker, so labile state is fragmented.
- **Resolution:** Use Redis or a shared database backend for labile state in production.

---

### MED-29: `/memory/turn` Missing Write Permission Check

- **File:** `src/api/routes.py` (lines 72-76)
- **Description:** The `/memory/turn` endpoint uses `get_auth_context` (read-only) but internally calls `orchestrator.write()`. A read-only API key should not be able to write memories.
- **Resolution:** Change `Depends(get_auth_context)` to `Depends(require_write_permission)`.

---

### MED-30: `x_tenant_id` Header Accepted but Ignored

- **File:** `src/api/auth.py` (line 52)
- **Description:** The `x_tenant_id` parameter is declared (appears in OpenAPI docs) but never used. The tenant always comes from the API key map.
- **Resolution:** Either use it for tenant override or remove the parameter.

---

### MED-31: Middleware Execution Order Hides Rate-Limited Requests

- **File:** `src/api/app.py` (lines 51-52)
- **Description:** FastAPI processes middleware in reverse addition order. Since logging is added first and rate limiting second, the rate limiter runs before the logger. Rate-limited 429 requests are never logged.
- **Resolution:** Swap the middleware addition order.

---

### MED-32: `ForgettingScheduler._scheduler_loop` Is a No-Op

- **File:** `src/forgetting/worker.py` (lines 207-210)
- **Description:** The scheduler loop only sleeps for the configured interval. It never enumerates users or triggers forgetting runs.
- **Resolution:** Add logic to iterate over registered users/tenants and call `self.worker.run_forgetting()`.

---

### MED-33: Mixed Naive/Aware Datetimes in Consolidation

- **File:** `src/consolidation/sampler.py` (lines 53, 84-85)
- **Description:** Uses `datetime.utcnow()` (naive) while other parts use `datetime.now(timezone.utc)` (aware). Subtracting mixed types raises `TypeError`.
- **Resolution:** Use `datetime.now(timezone.utc)` consistently.

---

### MED-34: `gist` May Be Unbound in Exception Handler

- **File:** `src/consolidation/migrator.py` (line 74)
- **Description:** If `alignment.gist` raises an exception, the except block references `gist.text[:50]` which raises `UnboundLocalError`.
- **Resolution:** Use a fallback in the error message: `gist_text = gist.text[:50] if 'gist' in dir() else 'unknown'`.

---

### MED-35: Port 8000 Conflict Between Services

- **File:** `docker/docker-compose.yml` (lines 46, 106)
- **Description:** Both `vllm` and `api` services map host port 8000. Running both causes a bind conflict.
- **Resolution:** Change the vllm host port to `8080:8000`.

---

### MED-36: `app` Service Missing Dependencies

- **File:** `docker/docker-compose.yml` (lines 82-84)
- **Description:** The `app` service (test runner) only depends on `postgres`. Tests requiring Neo4j or Redis will fail.
- **Resolution:** Add `neo4j` and `redis` to `depends_on` with health conditions.

---

### MED-37: Neo4j CI Service Has No Health Check

- **File:** `.github/workflows/ci.yml` (lines 52-57)
- **Description:** Neo4j can take 10-30 seconds to start. Tests may begin before Neo4j is ready, causing intermittent failures.
- **Resolution:** Add health check options or a wait-for step before tests.

---

### MED-38: Deprecated `event_loop` Fixture

- **File:** `tests/conftest.py` (lines 30-34) and `tests/integration/conftest.py` (lines 54-59)
- **Description:** Custom `event_loop` fixtures are deprecated in `pytest-asyncio >= 0.22` when `asyncio_mode = "auto"` is set.
- **Resolution:** Remove the custom fixtures. Use `loop_scope = "session"` in `[tool.pytest.ini_options]` if needed.

---

### MED-39: `admin_client` Fixture Skips Lifespan Events

- **File:** `tests/integration/test_phase9_api_flow.py` (lines 30-43)
- **Description:** The `admin_client` fixture uses `return` instead of `yield` inside a context manager, meaning FastAPI startup/shutdown lifespan events are not triggered. Also missing `get_settings.cache_clear()` cleanup.
- **Resolution:** Restructure to use `with TestClient(app) as c: yield c` and add try/finally cleanup.

---

### MED-40: `scripts/init_structure.py` Is Stale

- **File:** `scripts/init_structure.py` (lines 4-37)
- **Description:** The STRUCTURE dict doesn't reflect the actual project. Missing many files and directories added since the script was written.
- **Resolution:** Update the STRUCTURE dict to match the actual codebase, and add a guard against overwriting existing files.

---

### MED-41: `examples/standalone_demo.py` Uses Non-Existent API Fields

- **File:** `examples/standalone_demo.py` (entire file)
- **Description:** All HTTP request payloads include `"scope"` and `"scope_id"` fields that don't exist in API schemas. Fields are silently ignored by Pydantic v2, creating a false impression that scope-based filtering works.
- **Resolution:** Remove `scope`/`scope_id` from all payloads. Use `session_id` and `context_tags`.

---

### MED-42: LangChain Dependency Version Too Broad

- **File:** `examples/requirements.txt` (line 14)
- **Description:** `langchain>=0.1.0` allows any version including 0.2+ which has breaking import changes.
- **Resolution:** Pin to `langchain>=0.1.0,<0.2.0` or update the integration code for the latest version.

---

### MED-43: `ForgetRequest.action` and `ReadMemoryRequest.format` Lack Validation

- **File:** `src/api/schemas.py` (lines 59, 141)
- **Description:** Both fields accept any string but only specific values are meaningful. Invalid values cause silent no-ops or downstream errors.
- **Resolution:** Use `Literal["delete", "archive", "silence"]` and `Literal["packet", "list", "llm_context"]` respectively.

---

### MED-44: No Error Handling on Admin Endpoints

- **File:** `src/api/admin_routes.py` (lines 16-55)
- **Description:** Neither `trigger_consolidation` nor `trigger_forgetting` have try/except blocks. Unhandled exceptions return raw 500 errors.
- **Resolution:** Wrap each in try/except with appropriate `HTTPException` responses.

---

### MED-45: `_llm_detect` Uses `complete()` Instead of `complete_json()`

- **File:** `src/reconsolidation/conflict_detector.py` (line 175)
- **Description:** Manually parses JSON and strips markdown fences when `LLMClient` already provides `complete_json()`.
- **Resolution:** Replace with `data = await self.llm.complete_json(prompt, temperature=0.0)`.

---

### MED-46: Deprecated `asyncio.get_event_loop()`

- **File:** `src/utils/embeddings.py` (lines 115, 127)
- **Description:** `asyncio.get_event_loop()` is deprecated since Python 3.10.
- **Resolution:** Replace with `asyncio.get_running_loop()`.

---

### MED-47: Settings Cache Pollution Between Test Classes

- **Files:** `tests/unit/test_phase9_api.py`, `tests/unit/test_phase1_core.py`
- **Description:** Tests that modify settings via `monkeypatch` and call `get_settings.cache_clear()` don't always clear the cache in teardown, leaving stale cached results for subsequent tests.
- **Resolution:** Add an autouse fixture that clears the settings cache after each test.

---

### MED-48: Async Client Write Signature Differs from Sync Version

- **File:** `examples/memory_client.py` (lines 380-388)
- **Description:** The async `write()` is missing parameters that the sync version has: `turn_id`, `agent_id`, `namespace`.
- **Resolution:** Add the missing parameters to match the sync client's signature.

---

## Low Issues

### LOW-01: `@lru_cache` Prevents Settings Override in Tests

- **File:** `src/core/config.py` (line 95)
- **Description:** No clear/reset mechanism is documented. Tests must know to call `get_settings.cache_clear()`.
- **Resolution:** Document the escape hatch or use a dependency-injection pattern.

---

### LOW-02: Empty `exceptions.py` -- No Custom Exception Hierarchy

- **File:** `src/core/exceptions.py`
- **Description:** The codebase has no custom exceptions. All error handling uses built-in exceptions.
- **Resolution:** Define `MemoryNotFoundError`, `DuplicateMemoryError`, `StorageConnectionError`, etc.

---

### LOW-03: Empty `models.py` Module

- **File:** `src/core/models.py`
- **Description:** Empty, unused module adding confusion.
- **Resolution:** Populate it with domain models or remove it.

---

### LOW-04: `to_context_string` Truncates Mid-Character

- **File:** `src/core/schemas.py` (line 204)
- **Description:** `result[:max_chars]` can cut a string mid-line or mid-UTF8 character.
- **Resolution:** Truncate at the last newline: `result[:max_chars].rsplit("\n", 1)[0] + "\n..."`.

---

### LOW-05: No Explicit Rollback on Session Error

- **File:** `src/storage/connection.py` (lines 52-58)
- **Description:** The `pg_session` context manager only calls `session.close()` on error. While SQLAlchemy auto-rollbacks on close, explicit rollback is best practice.
- **Resolution:** Add `await session.rollback()` in the except block.

---

### LOW-06: Time-Based Filter Misses Concurrent Events

- **File:** `src/storage/event_log.py` (lines 93-96)
- **Description:** `replay_events` filters by `created_at > from_event.created_at`, skipping events at the exact same timestamp.
- **Resolution:** Use ID-based ordering or `>=` with explicit exclusion of the reference event.

---

### LOW-07: `object` Parameter Shadows Built-in

- **Files:** `src/storage/neo4j.py` (lines 105, 120), `src/memory/knowledge_base.py` (lines 19, 47)
- **Description:** Multiple methods use `object` as a parameter name, shadowing Python's built-in.
- **Resolution:** Rename to `object_entity` or `target`.

---

### LOW-08: Dead `"metadata"` Entry in `key_allow` Set

- **File:** `src/storage/postgres.py` (lines 144, 150-153)
- **Description:** `"metadata"` is in `key_allow` but is handled by a dedicated branch before the `elif key in key_allow` check, making it dead code.
- **Resolution:** Remove `"metadata"` from `key_allow`.

---

### LOW-09: Hardcoded `reload=False`

- **File:** `src/main.py` (line 10)
- **Description:** In development, `reload=True` is typically desired.
- **Resolution:** Read from settings: `uvicorn.run(..., reload=settings.debug)`.

---

### LOW-10: Unnecessary Double-Fetch of ORM Model

- **File:** `src/memory/neocortical/fact_store.py` (line 189)
- **Description:** `_get_existing_fact` queries the DB and converts to a dataclass. Then `_update_fact` re-fetches the ORM model by ID. This wastes a DB round-trip.
- **Resolution:** Have `_get_existing_fact` return the ORM model directly.

---

### LOW-11: Overlapping PII Matches Produce Garbled Text

- **File:** `src/memory/hippocampal/redactor.py` (lines 37-48)
- **Description:** If two PII patterns overlap, both are independently replaced in reverse order, potentially producing corrupted output.
- **Resolution:** Merge overlapping ranges before replacing.

---

### LOW-12: No Markdown Fence Stripping in Entity/Relation Extractors

- **Files:** `src/extraction/entity_extractor.py`, `src/extraction/relation_extractor.py`
- **Description:** `LLMFactExtractor` strips markdown fences from LLM output, but entity and relation extractors do not. LLM JSON wrapped in fences will cause `json.loads()` to fail.
- **Resolution:** Extract the fence-stripping logic into a shared utility and apply to all extractors.

---

### LOW-13: Hardcoded `diversity = 1.0` in Reranker

- **File:** `src/retrieval/reranker.py` (line 62)
- **Description:** The diversity score is a constant 1.0, adding a fixed `0.1` to every score. Actual diversity is only handled by MMR post-hoc.
- **Resolution:** Either compute actual per-item diversity or remove the misleading weight.

---

### LOW-14: Unused `analysis` Parameter

- **File:** `src/retrieval/retriever.py` (lines 223-226)
- **Description:** The `analysis: QueryAnalysis` parameter is accepted but never used.
- **Resolution:** Remove the parameter or use it for scoring adjustments.

---

### LOW-15: Mock Embeddings Are Mostly Zeros

- **File:** `src/utils/embeddings.py` (lines 151-154)
- **Description:** SHA256 produces 32 bytes, but dimensions default to 1536. After 32 values from the hash, the remaining 1504 are filled with `0.0`, making cosine similarity unreliable in tests.
- **Resolution:** Use the hash to seed a deterministic PRNG generating all dimensions.

---

### LOW-16: Empty Files Serving No Purpose

- **Files:** `src/utils/timing.py`, `config/logging.yaml`, `config/settings.yaml`
- **Description:** These files exist but are empty. The codebase doesn't load them.
- **Resolution:** Populate them with useful content or remove them.

---

### LOW-17: Example Imports Only Work from `examples/` Directory

- **Files:** All `examples/*.py` files
- **Description:** `from memory_client import CognitiveMemoryClient` only works if CWD is `examples/`.
- **Resolution:** Add a path fixup at the top of each example file:

```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
```

---

### LOW-18: Redundant `@pytest.mark.asyncio` Markers

- **Files:** Various test files
- **Description:** With `asyncio_mode = "auto"` in pyproject.toml, the decorators are unnecessary.
- **Resolution:** Remove the decorators -- auto mode detects async tests automatically.

---

### LOW-19: Duplicate `to_item` Helper Function

- **File:** `src/api/routes.py` (lines 126-137 and 304-315)
- **Description:** The same `to_item(mem)` closure is defined identically in two endpoints, violating DRY.
- **Resolution:** Extract to a module-level helper function.

---

### LOW-20: Redundant Exception Catch in Migrator

- **File:** `src/consolidation/migrator.py` (line 127)
- **Description:** `except (ValueError, Exception)` -- `Exception` is a superclass of `ValueError`, making `ValueError` redundant.
- **Resolution:** Simplify to `except Exception`.

---

## Prioritized Action Plan

### Phase 1: Critical Fixes (Immediate)

These issues cause data loss, broken functionality, or completely non-functional code.

1. Fix `_execute_archive` data loss (CRIT-04)
2. Fix migration column mismatch `user_id` vs `scope_id` (CRIT-05)
3. Fix duplicate `is_current=True` facts (CRIT-02)
4. Fix `value_type` "strtype" bug (CRIT-03)
5. Fix `create_session` expiry (CRIT-06)
6. Fix Celery beat schedule arguments (CRIT-01)
7. Rewrite `examples/async_usage.py` (CRIT-07)

### Phase 2: High-Priority Fixes (This Sprint)

Security, data corruption, and race conditions.

1. Fix mutable SQLAlchemy column defaults (HIGH-01)
2. Fix CORS wildcard + credentials (HIGH-03)
3. Fix API key timing attack (HIGH-05)
4. Fix middleware HTTPException handling (HIGH-04)
5. Fix rate limiter memory leak (HIGH-06)
6. Fix `delete_all` GDPR gap (HIGH-08)
7. Fix labile tracker race condition (HIGH-11)
8. Fix silent exception swallowing in reconsolidation (HIGH-12)
9. Fix episode duplication in clusterer (HIGH-09)
10. Fix interference resolution logic (HIGH-10)
11. Add missing async client methods (HIGH-14)
12. Fix O(n) reference counting (HIGH-02)
13. Persist sessions (HIGH-07)
14. Fix LangChain integration (HIGH-13)

### Phase 3: Medium-Priority Fixes (Next Sprint)

Logic errors, missing validation, deprecated APIs.

1. Replace all `datetime.utcnow()` with `datetime.now(timezone.utc)` (MED-03)
2. Fix nested `BaseSettings` -> `BaseModel` (MED-01)
3. Fix write permission on `/memory/turn` (MED-29)
4. Add API schema validation (MED-43)
5. Fix `ForgettingScheduler` no-op loop (MED-32)
6. Fix entity extraction false positives (MED-23)
7. Fix `min_confidence` filter (MED-25)
8. Fix Docker port conflict and dependencies (MED-35, MED-36)
9. Add CI health checks (MED-37)
10. Fix test fixtures (MED-38, MED-39, MED-47)
11. Update stale examples (MED-41, MED-42)
12. Address remaining medium issues

### Phase 4: Low-Priority Improvements (Backlog)

Code quality, documentation, cleanup.

1. Define custom exception hierarchy (LOW-02)
2. Clean up empty/dead files (LOW-03, LOW-16)
3. Fix example import paths (LOW-17)
4. Remove redundant test markers (LOW-18)
5. Address remaining low issues
