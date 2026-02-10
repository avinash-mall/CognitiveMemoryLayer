# Base CML — Codebase Issues

> Comprehensive catalogue of issues identified during a full code review of `src/`.
> Each issue includes severity, affected file(s), description, and proposed solution.

---

## Summary

| Category | CRITICAL | HIGH | MEDIUM | LOW |
|----------|----------|------|--------|-----|
| Security | 2 | 2 | 2 | 1 |
| Correctness / Bugs | 0 | 5 | 14 | 8 |
| Concurrency / Thread-Safety | 1 | 2 | 3 | 0 |
| Design / Architecture | 0 | 1 | 8 | 13 |
| Performance | 0 | 0 | 9 | 5 |
| Observability / Error Handling | 0 | 0 | 5 | 3 |
| **Total** | **3** | **10** | **41** | **30** |

---

## 1 · Security

### SEC-01 · SQL Injection via f-string in Neo4j Cypher queries

| Attribute | Details |
|-----------|---------|
| **Severity** | CRITICAL |
| **File** | `src/storage/neo4j.py` — `merge_edge()` |
| **Description** | The `merge_edge` method constructs Cypher queries using Python f-strings with user-supplied `rel_type`: `f"MERGE (a)-[r:{rel_type}]->(b)"`. A malicious `rel_type` value such as `OWNS]->(b) DELETE a,b //` can inject arbitrary Cypher and destroy data. |
| **Proposed Solution** | (1) Sanitize `rel_type` with a strict allowlist regex (e.g. `^[A-Z_]+$`). (2) Reject any value that does not match. (3) Consider using parameterized relationship type insertion via APOC procedures (`apoc.merge.relationship`). |

---

### SEC-02 · Raw SQL with string interpolation in Postgres store

| Attribute | Details |
|-----------|---------|
| **Severity** | CRITICAL |
| **File** | `src/storage/postgres.py` — `count_references_to()` |
| **Description** | Uses `text(f"SELECT ... WHERE metadata::text LIKE '%{memory_id}%'")` with a UUID that is not parameterized. While UUIDs are inherently safe-ish, this pattern normalizes raw SQL construction. Any future caller passing unvalidated input into similar patterns inherits the risk. |
| **Proposed Solution** | Replace with a parameterized query: `text("SELECT ... WHERE metadata::text LIKE :pattern")` bound with `{"pattern": f"%{memory_id}%"}`. Audit all `text()` calls for similar patterns. |

---

### SEC-03 · Tenant ID override without authorization check

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `src/api/auth.py` — `get_auth_context()` |
| **Description** | The `X-Tenant-Id` header allows any authenticated caller to override the tenant ID derived from their API key. A user with a valid API key for tenant A can set `X-Tenant-Id: tenantB` and access/modify tenant B's data. |
| **Proposed Solution** | (1) Only allow tenant override for admin keys (`can_admin=True`). (2) For non-admin keys, ignore the `X-Tenant-Id` header or validate it against a list of allowed tenants for that key. |

---

### SEC-04 · Error responses leak internal details

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/api/routes.py` — all exception handlers |
| **Description** | Every `except Exception as e` block raises `HTTPException(status_code=500, detail=str(e))`. This can expose internal stack traces, database error messages, or file paths to the API consumer. |
| **Proposed Solution** | Return a generic error message to the client (e.g. `"Internal server error"`). Log the full exception with traceback server-side. Add a `debug` setting that enables detailed error responses only in development. |

---

### SEC-05 · Dashboard search `ilike` not sanitized for SQL wildcards

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/api/dashboard_routes.py` — search endpoint |
| **Description** | The `search` query parameter is passed directly into a `col.ilike(f"%{search}%")` filter without escaping SQL wildcard characters (`%`, `_`). A user can inject `%` or `_` to match unintended rows, or craft patterns causing expensive full-table scans. |
| **Proposed Solution** | Escape `%` and `_` in the user-supplied search string before embedding in the `ilike` pattern: `escaped = search.replace("%", "\\%").replace("_", "\\_")`. |

---

### SEC-06 · `subprocess` call in dashboard reset endpoint

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/api/dashboard_routes.py` — reset endpoint |
| **Description** | The dashboard reset endpoint uses `subprocess.run(...)` to execute system commands. Even though the command is not user-controlled, having `subprocess` in an API handler is risky practice — future modifications could inadvertently pass user input. The endpoint itself lacks admin-level permission checks. |
| **Proposed Solution** | Replace the subprocess call with a native Python/SQLAlchemy equivalent. Add `require_admin_permission` dependency to the route. |

---

## 2 · Correctness / Bugs

### BUG-01 · `pg_session` context manager lacks explicit `commit()`

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `src/storage/connection.py` — `DatabaseManager.pg_session()` |
| **Description** | The `pg_session` context manager creates a session and yields it. On successful exit it only calls `session.close()` — there is no explicit `session.commit()`. Unless the session is configured for `autocommit=True` (it is not), writes are silently rolled back when the context exits. |
| **Proposed Solution** | Add `await session.commit()` before `session.close()` in the `try` block of the context manager. Ensure `await session.rollback()` is called in the `except` block. |

---

### BUG-02 · `search()` writes stale access counts

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `src/memory/hippocampal/store.py` — `search()` |
| **Description** | After vector search retrieves records, the method increments `access_count` and updates `last_accessed_at` on the **in-memory** `MemoryRecord` objects and then fires concurrent `store.update()` calls. However, the update patch is built from the old in-memory value (`record.access_count + 1`). If two concurrent searches retrieve the same record, both will read the same `access_count` and write the same incremented value, losing one increment (lost-update race). |
| **Proposed Solution** | Use an atomic SQL increment: `SET access_count = access_count + 1` instead of setting to a pre-computed value. Alternatively, batch these updates with a single SQL `UPDATE ... SET access_count = access_count + 1 WHERE id IN (...)`. |

---

### BUG-03 · `_mark_episodes_consolidated` overwrites metadata dict

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `src/consolidation/worker.py` — `_mark_episodes_consolidated()` |
| **Description** | When marking episodes as consolidated, the method sets `metadata={"consolidated_at": ...}`. This **replaces** the entire metadata dict, discarding any existing metadata keys (e.g. `source`, `tags`, `provenance` entries). |
| **Proposed Solution** | Merge with existing metadata: `patch = {"metadata": {**episode.metadata, "consolidated_at": now.isoformat()}}`. |

---

### BUG-04 · `_execute_archive` is non-atomic (archive + delete can partially fail)

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `src/forgetting/executor.py` — `_execute_archive()` |
| **Description** | The archive action first writes the record to an archive store, then deletes it from the primary store. If the archive succeeds but the delete fails, the record exists in both locations (duplicate). If the delete succeeds but archive fails, data is lost. |
| **Proposed Solution** | Wrap both operations in a database transaction so they succeed or fail together. Alternatively, use a two-phase approach: (1) mark the record as `status=ARCHIVED` first (single update), (2) write to archive in a background task, (3) hard-delete only after archive is confirmed. |

---

### BUG-05 · `_execute_decay` and `_execute_silence` overwrite metadata

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `src/forgetting/executor.py` |
| **Description** | Both `_execute_decay` and `_execute_silence` set `metadata={"decayed_at": ...}` or `metadata={"silenced_at": ...}`, replacing the entire metadata dict (same pattern as BUG-03). |
| **Proposed Solution** | Merge: `{**memory.metadata, "decayed_at": now.isoformat()}`. |

---

### BUG-06 · `ScratchPad.set` may create duplicates

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/scratch_pad.py` — `set()` |
| **Description** | The `set` method calls `store.upsert()` with a `content_hash` derived from the value. If the same key is set with a different value, the content hash changes, and `upsert` may insert a new row rather than updating the existing one. This creates duplicates for the same key. |
| **Proposed Solution** | Base the content hash on the key (not the value), or query-and-update by key before inserting. The `upsert` logic should match on `key` + `tenant_id` + `type=SCRATCH`, not on `content_hash`. |

---

### BUG-07 · `ScratchPad.clear` uses unbatched deletes

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/scratch_pad.py` — `clear()` |
| **Description** | Clears all scratch entries by issuing individual `store.delete()` calls in a loop. For a large number of entries, this creates N individual delete transactions. |
| **Proposed Solution** | Add a batch delete method to the store API: `store.delete_batch(ids)` or `store.delete_by_filter(tenant_id=..., type=SCRATCH)`. |

---

### BUG-08 · `ConsolidationScheduler` creates `asyncio.Queue` at init-time

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/consolidation/worker.py` — `ConsolidationScheduler.__init__()` |
| **Description** | An `asyncio.Queue` is created during `__init__`. If the scheduler is instantiated before the event loop starts (e.g. at module import time or in a sync context), the queue may be bound to a different or no event loop, causing `RuntimeError` when used later in an async context (Python 3.10+ changed this behavior). |
| **Proposed Solution** | Lazily initialize the queue on first use, or create it inside `start()` / the running coroutine. Same pattern applies to `asyncio.Lock()` in `LabileStateTracker.__init__`. |

---

### BUG-09 · `_extract_new_facts` fallback misses assistant-side facts

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/reconsolidation/service.py` — `_extract_new_facts()` |
| **Description** | When no `fact_extractor` is configured, the heuristic fallback only scans `user_message` for keywords like "I am", "I live". It ignores `assistant_response` entirely, missing facts the assistant may have stated or confirmed. |
| **Proposed Solution** | Also scan `assistant_response` for fact patterns, or at minimum, combine both texts for keyword matching. |

---

### BUG-10 · `_worker_loop` continues on error

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/consolidation/worker.py` — `_worker_loop()` |
| **Description** | The worker loop catches all exceptions and continues. If a persistent error occurs (e.g. database is down), the loop will spin continuously, logging errors and consuming CPU without making progress. |
| **Proposed Solution** | Add exponential backoff on repeated errors. After N consecutive failures, pause the loop for an increasing duration. Add a circuit-breaker pattern. |

---

### BUG-11 · `EpisodeCluster.common_entities` uses mutable default

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/consolidation/clusterer.py` — `EpisodeCluster` dataclass |
| **Description** | `common_entities: List[str] = []` uses a mutable default argument. All instances created without explicitly passing `common_entities` will share the same list. Appending to one instance's list mutates all others. |
| **Proposed Solution** | Use `field(default_factory=list)`: `common_entities: List[str] = field(default_factory=list)`. |

---

### BUG-12 · `session/create` silently succeeds without Redis

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/api/routes.py` — `create_session()` |
| **Description** | If Redis is not available, the endpoint returns a `session_id` but never persists the session. Subsequent calls referencing that `session_id` for write/read will work (because they just pass it as `source_session_id` to the orchestrator), but the session metadata (creation time, expiry, tenant binding) is lost. |
| **Proposed Solution** | Either (1) return an error when Redis is unavailable and sessions are requested, or (2) fall back to in-memory session storage with a warning log, or (3) document that sessions require Redis. |

---

### BUG-13 · `_extract_new_facts` fallback splits only on `.` — misses `!` and `?`

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/reconsolidation/service.py` — `_extract_new_facts()` |
| **Description** | The heuristic fallback splits `user_message` using `split(".")`. Sentences ending with `!` or `?` are treated as a single long string, potentially concatenating two unrelated sentences and either missing facts or producing malformed extractions. |
| **Proposed Solution** | Use a regex split: `re.split(r'[.!?]+', user_message)` to handle all sentence terminators. |

---

### BUG-14 · `_apply_operation` returns `True` for unknown operation types

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/reconsolidation/service.py` — `_apply_operation()` |
| **Description** | The method checks `ADD`, `UPDATE/REINFORCE/DECAY`, and `DELETE`. Any other `OperationType` (present or future) falls through and returns `True` — silent success for an operation that was never executed. |
| **Proposed Solution** | Add an `else` branch that logs a warning and returns `False` for unknown operation types. |

---

### BUG-15 · `conversation.add_message` deduplication via content_hash loses distinct messages

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/conversation.py` — `add_message()` |
| **Description** | The `content_hash` for deduplication is based on the message text. If a user sends the exact same message twice in a conversation (e.g. "OK", "Yes"), the second message is silently dropped by the upsert's duplicate check. This loses valid conversation turns. |
| **Proposed Solution** | Include the `session_id`, `turn_number`, or timestamp in the content hash computation so identical text at different points in the conversation produces different hashes. |

---

### BUG-16 · `WorkingMemoryState.add_chunk` eviction uses identity comparison

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/memory/working/models.py` — `add_chunk()` |
| **Description** | Eviction partitions chunks into `recent_chunks` and `older_chunks` using `if c not in recent_chunks`. The `not in` operator uses object identity (or `__eq__` which falls back to identity for dataclasses without explicit definition). While `dataclass` auto-generates `__eq__`, the comparison iterates through all chunks for every check, making eviction O(k·n) per chunk addition. |
| **Proposed Solution** | Use a set of chunk IDs for the lookup: `recent_ids = {c.id for c in recent_chunks}; older = [c for c in self.chunks if c.id not in recent_ids]`. |

---

### BUG-17 · `knowledge_base.Fact.object` shadows Python builtin

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/memory/knowledge_base.py` — `Fact` dataclass |
| **Description** | The `Fact` dataclass defines a field named `object`, which shadows the Python builtin `object`. While this works syntactically, it makes code confusing and prevents using `object()` inside methods of `Fact`. |
| **Proposed Solution** | Rename the field to `object_value` or `fact_object` to avoid shadowing. |

---

### BUG-18 · `tool_memory` filter on `tool_name` is done in Python, not SQL

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/memory/tool_memory.py` — `get_results()` |
| **Description** | When retrieving tool results filtered by `tool_name`, all tool memories for the tenant are fetched from the store, then filtered in Python. This fetches far more data than needed. |
| **Proposed Solution** | Add the `tool_name` as a filter to the store query (e.g. via `key` prefix matching or a metadata filter) to push the filtering down to SQL. |

---

## 3 · Concurrency / Thread-Safety

### CON-01 · `DatabaseManager` singleton is not thread-safe

| Attribute | Details |
|-----------|---------|
| **Severity** | CRITICAL |
| **File** | `src/storage/connection.py` — `get_instance()` |
| **Description** | The singleton pattern uses `if cls._instance is None: cls._instance = cls()` without any locking. In multi-threaded environments (e.g. Celery workers, uvicorn with multiple threads), two threads can simultaneously see `_instance` as `None` and create two separate instances, leading to duplicate database connections and potential resource leaks. |
| **Proposed Solution** | Use a threading lock for the singleton check: `with cls._lock: if cls._instance is None: ...`. Alternatively, use a module-level instance with `threading.Lock()`. |

---

### CON-02 · Rate limiter state is per-process / non-distributed

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `src/api/middleware.py` — `RateLimitMiddleware` |
| **Description** | Rate limiting uses an in-process `dict` protected by `asyncio.Lock`. In multi-worker deployments (e.g. uvicorn with `--workers 4` or Kubernetes with replicas), each worker has its own rate limit state. A client can get `4× requests_per_minute` by spreading requests across workers. |
| **Proposed Solution** | Use Redis-based rate limiting (e.g. `redis.incr` + `redis.expire` with a sliding window). Fall back to per-process limiting when Redis is unavailable, with a log warning. |

---

### CON-03 · `LabileStateTracker._lock` is per-process only

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `src/reconsolidation/labile_tracker.py` |
| **Description** | The `asyncio.Lock()` in `LabileStateTracker` protects in-memory state and Redis operations. However, when using the Redis backend, multiple workers share the same Redis but each has their own `asyncio.Lock`. The lock only serializes operations within a single process. Concurrent workers can still race on Redis `GET → modify → SET` sequences. |
| **Proposed Solution** | Use Redis-native atomic operations (e.g. `WATCH`/`MULTI`/`EXEC` or Lua scripts) when the Redis backend is in use, rather than relying on in-process locks. |

---

### CON-04 · `_build_api_keys()` rebuilds on every request

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/api/auth.py` — `_build_api_keys()` |
| **Description** | Called on every request via `get_auth_context()`. It reads settings, creates `AuthContext` objects, and builds a dict. While not inherently wrong, it causes unnecessary allocations on every request. With `get_settings()` being `lru_cache`-decorated, the settings read is fast, but the dict/dataclass construction is repeated. |
| **Proposed Solution** | Cache the API key map at module level or use `@lru_cache`. Clear the cache when settings change (if hot-reload is ever implemented). |

---

### CON-05 · `WorkingMemoryManager` and `SensoryBufferManager` grow unbounded

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/working/manager.py`, `src/memory/sensory/manager.py` |
| **Description** | Both managers store per-scope state in in-memory dicts (`_states`, `_buffers`). Like `SensoryBufferManager` (DES-02), `WorkingMemoryManager` also lacks any eviction policy. In a long-running server with many tenants/scopes, memory grows without bound. While `SensoryBufferManager` has `cleanup_inactive()`, the `WorkingMemoryManager` has no equivalent. |
| **Proposed Solution** | Add `cleanup_inactive()` to `WorkingMemoryManager` that evicts states unused for a threshold period. Consider using an LRU cache with a maximum size for both managers. |

---

## 4 · Design / Architecture

### DES-01 · `get_settings()` is `lru_cache`-decorated — prevents runtime reload

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/core/config.py` — `get_settings()` |
| **Description** | Because `get_settings` is wrapped with `@lru_cache`, the settings object is created once and cached forever. Environment variable changes or config file updates at runtime are invisible. This makes it impossible to change settings without restarting the process. |
| **Proposed Solution** | (1) Accept this as intentional for performance and document it. (2) Or replace with a `_settings` module-level variable and a `reload_settings()` function for explicit refreshes. (3) Or use a TTL-based cache that re-reads periodically. |

---

### DES-02 · `SensoryBufferManager` grows unbounded in-memory

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/sensory/manager.py` |
| **Description** | Every unique `(tenant_id, scope_id)` pair creates a new `SensoryBuffer` instance stored in a dict. There is no eviction policy for inactive buffers. In a long-running server with many tenants, this dict grows without bound. |
| **Proposed Solution** | Add a max-size LRU eviction policy. Track `last_active` per buffer and periodically evict buffers that have been idle beyond a threshold (e.g. 30 minutes). |

---

### DES-03 · `RuleBasedChunker` is sync while `SemanticChunker` is async

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/working/chunker.py` |
| **Description** | `SemanticChunker.chunk()` is `async` (calls LLM), while `RuleBasedChunker.chunk()` is a regular sync method. The caller (`WorkingMemoryManager`) must know which type it has and call it differently, or `await` a sync function (which works but is misleading). |
| **Proposed Solution** | Make `RuleBasedChunker.chunk()` async as well (trivially, just add the `async` keyword). This ensures a uniform `Chunker` interface where all implementations are awaitable. |

---

### DES-04 · Duplicate `_strip_markdown_fences` utility

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/extraction/entity_extractor.py`, `src/extraction/relation_extractor.py` |
| **Description** | Both files define an identical `_strip_markdown_fences()` function. Code duplication that will diverge over time. |
| **Proposed Solution** | Extract into `src/utils/text.py` or `src/extraction/__init__.py` and import from both. |

---

### DES-05 · `_tokenize` reinstantiates `tiktoken` encoder on every call

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/memory/sensory/buffer.py` — `_tokenize()` |
| **Description** | Each call to `_tokenize` creates a new `tiktoken.get_encoding("cl100k_base")` instance. Encoder loading involves file I/O (reading a BPE merge table), which is wasteful when called repeatedly. |
| **Proposed Solution** | Cache the encoder at the class or module level: `_ENCODER = tiktoken.get_encoding("cl100k_base")`. |

---

### DES-06 · `WorkingMemoryState` naming inconsistency

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/memory/working/models.py` |
| **Description** | The dataclass `WorkingMemoryState` holds `chunks` + `max_chunks` + `current_turn` + `current_topic`. The name suggests a state object, but it also encapsulates capacity policy (`max_chunks`). Minor naming/responsibility confusion. |
| **Proposed Solution** | Either rename to `WorkingMemoryBuffer` (emphasizing it holds data with capacity), or extract configuration into a separate `WorkingMemoryConfig` object. Low priority. |

---

### DES-07 · Module-level `app = create_app()` in `api/app.py`

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/api/app.py` — line 101 |
| **Description** | The module-level `app = create_app()` call means the FastAPI app (and its middleware, routes) is created on import. This makes it difficult to test with different configurations or to delay initialization. |
| **Proposed Solution** | Remove the module-level call. Let the entry point (e.g. `main.py`, `uvicorn` CLI) call `create_app()` explicitly. Update `uvicorn.run` calls to use `"src.api.app:create_app"` factory mode. |

---

### DES-08 · `process_turn` does N×M conflict checks in reconsolidation

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/reconsolidation/service.py` — `process_turn()` |
| **Description** | The nested loop `for new_fact in new_facts: for memory in retrieved_memories: detect(...)` performs N×M LLM calls (or heuristic checks). With 5 facts and 10 memories, this is 50 conflict checks. Since `detect()` may invoke the LLM, this can be extremely slow and expensive. |
| **Proposed Solution** | (1) Pre-filter: only check memories whose embedding similarity to the new fact exceeds a threshold. (2) Batch conflict detection into a single LLM call with multiple pairs. (3) Use `detect_batch` instead of individual calls. (4) Set a hard cap (e.g. max 20 checks per turn). |

---

### DES-09 · `_scheduler_loop` sleeps full interval before first check

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/consolidation/worker.py` — `_scheduler_loop()` |
| **Description** | The scheduler loop structure is `while True: await asyncio.sleep(interval); check_triggers()`. This means the first trigger check only happens after one full interval (default: minutes). If the system starts with pending work, it sits idle for the entire first interval. |
| **Proposed Solution** | Move the sleep to the end of the loop, or run the first check immediately: `while True: check_triggers(); await asyncio.sleep(interval)`. |

---

### DES-10 · `seamless_provider.py` circular import with lazy `try/except ImportError`

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/seamless_provider.py` |
| **Description** | The module wraps `from ..memory.orchestrator import MemoryOrchestrator` in a `try/except ImportError` block, setting `MemoryOrchestrator = None` on failure. This is a fragile pattern for breaking circular imports — it silently degrades to `None` type hints and will cause confusing runtime `TypeError`s if the import order changes. |
| **Proposed Solution** | Use `TYPE_CHECKING` guard for type annotations and pass the orchestrator instance at runtime. Or restructure the import graph to eliminate the cycle. |

---

### DES-11 · `memory/__init__.py` missing exports for several public modules

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/memory/__init__.py` |
| **Description** | Only exports `MemoryOrchestrator`, `ShortTermMemory`, `HippocampalStore`, `NeocorticalStore`. The `ScratchPad`, `ConversationMemory`, `ToolMemory`, `KnowledgeBase`, and `SeamlessMemoryProvider` are not re-exported, forcing consumers to import from deep submodules. |
| **Proposed Solution** | Add the missing classes to `__all__` and the import list. |

---

### DES-12 · `to_llm_context` parameter `format` shadows builtin

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/retrieval/packet_builder.py` — `to_llm_context()` |
| **Description** | The `format` parameter in `to_llm_context()` and `retrieve_for_llm()` shadows the Python builtin `format()` function. While not causing runtime errors, it's poor style and may confuse linters. |
| **Proposed Solution** | Rename the parameter to `output_format` or `fmt`. |

---

### DES-13 · `reconsolidation/service.py` lazy imports for `FactExtractor` and `LLMClient`

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/reconsolidation/service.py` |
| **Description** | Module-level `try: from ..extraction.fact_extractor import FactExtractor / except ImportError: FactExtractor = None` and similar for `LLMClient`. These modules definitely exist in the same package, suggesting this is defensive coding against import order issues rather than optional dependencies. The pattern masks actual import errors during development. |
| **Proposed Solution** | Remove the `try/except ImportError` guards in favor of direct imports. If the intention is optional dependencies, use `TYPE_CHECKING` for type annotations and document the optionality. |

---

### DES-14 · `ForgettingScheduler._scheduler_loop` silently swallows all exceptions

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/forgetting/worker.py` — `_scheduler_loop()` |
| **Description** | The `except Exception: pass` clause discards all errors during scheduled forgetting runs. If `run_forgetting` fails consistently (e.g. database unreachable), there is no logging or alerting — failures are completely invisible. |
| **Proposed Solution** | Add `logger.exception("forgetting_run_failed", ...)` in the except block. Consider adding backoff for consecutive failures analogous to BUG-10. |

---

### DES-15 · `ForgettingScheduler._user_last_run` grows unbounded with no eviction

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/forgetting/worker.py` — `ForgettingScheduler` |
| **Description** | The `_user_last_run` dict tracks last-run times for every `tenant:user` ever seen. In a multi-tenant system with many users, this grows indefinitely without eviction. |
| **Proposed Solution** | Use an LRU dict with max size, or periodically evict entries older than a threshold (e.g. 7 days). |

---

## 5 · Performance

### PERF-01 · `_get_dependency_counts` has O(n²) complexity

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/forgetting/scorer.py` — `_get_dependency_counts()` |
| **Description** | For each memory, this method iterates over all other memories to count references. With 10,000 memories, this is ~100 million comparisons, which will be extremely slow. |
| **Proposed Solution** | (1) Pre-compute a dependency index (dict mapping `memory_id → set of referencing memory IDs`). (2) Or use an SQL query: `SELECT referenced_id, COUNT(*) FROM ... GROUP BY referenced_id`. (3) Cache dependency counts between forgetting runs. |

---

### PERF-02 · `_calculate_score` has O(n²) embedding comparison

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/forgetting/interference.py` — `detect_duplicates()` |
| **Description** | Pairwise cosine similarity comparison between all memory embeddings. For N memories, this is O(n²) comparisons. At 10,000 memories, ~50 million similarity calculations. |
| **Proposed Solution** | Use approximate nearest neighbor search (e.g. FAISS or pgvector's built-in index) to find candidate duplicates before computing exact similarity. Only compare within candidate sets. |

---

### PERF-03 · `store_relations_batch` is sequential

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/neocortical/store.py` |
| **Description** | Stores each relation one-at-a-time via individual `merge_edge()` calls. For a chunk with 15 relations, this is 15 sequential Neo4j transactions. |
| **Proposed Solution** | Batch the Cypher queries into a single transaction using `UNWIND` or batch the `merge_edge` calls with `asyncio.gather`. |

---

### PERF-04 · `CachedEmbeddings.embed_batch` makes sequential Redis calls for cache lookups

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/utils/embeddings.py` — `CachedEmbeddings.embed_batch()` |
| **Description** | For each text in the batch, individual `await self.redis.get(cache_key)` calls are made. With 20 texts, this is 20 serial round trips to Redis. |
| **Proposed Solution** | Use `redis.mget(*keys)` to fetch all cache entries in a single round trip. Similarly, use `redis.mset()` or a pipeline for cache writes. |

---

### PERF-05 · Retriever silently ignores exceptions from parallel steps

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/retrieval/retriever.py` |
| **Description** | The retriever runs multiple retrieval strategies in parallel via `asyncio.gather(return_exceptions=True)`. Failed tasks return `Exception` objects that are silently discarded. This means partial failures produce incomplete results without any indication. |
| **Proposed Solution** | Log exceptions from failed retrieval steps at WARNING level. Consider emitting a metric for retrieval failures to aid monitoring. |

---

### PERF-06 · `_fast_detect` correction markers may false-positive

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/reconsolidation/conflict_detector.py` — `_fast_detect()` |
| **Description** | Simple substring checks (e.g. `"changed"` in new_lower) can match innocuous sentences like "The weather changed yesterday" or "I actually like pizza" against any memory, triggering a correction-type conflict with confidence 0.85. |
| **Proposed Solution** | (1) Require the marker to appear at the start of the sentence. (2) Check that old and new statements share some topic overlap before flagging. (3) Lower the confidence for pure marker-based detection (e.g. 0.6 instead of 0.85). |

---

### PERF-07 · Reranker `_calculate_score` computes O(n²) pairwise similarity

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/retrieval/reranker.py` — `_calculate_score()` |
| **Description** | For each memory, the method computes word-overlap similarity against all other memories to calculate a diversity score. With N results, this is O(n²). Combined with the MMR `_apply_diversity` (also O(n·k²)), total reranking cost is quadratic. |
| **Proposed Solution** | (1) Pre-compute the similarity matrix once instead of per-memory. (2) Use embedding-based similarity (cosine of stored embeddings) instead of word overlap for better quality and potential vectorization. (3) Limit the comparison set to top-k candidates. |

---

### PERF-08 · `_get_dependency_counts` in forgetting worker is also O(n²)

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/forgetting/worker.py` — `_get_dependency_counts()` |
| **Description** | Same O(n²) pattern as PERF-01 (scorer.py variant): for each memory, iterates all other memories to count `supersedes_id` and `evidence_refs` references. With 5000 memories (the default limit), ~25 million comparisons. |
| **Proposed Solution** | Build a reverse-index in a single pass: iterate once, and for each memory's `supersedes_id` and `evidence_refs`, increment the counts in a dict. |

---

### PERF-09 · `labile_tracker._cleanup_old_sessions_redis` N+1 Redis queries

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/reconsolidation/labile_tracker.py` — `_cleanup_old_sessions_redis()` |
| **Description** | For each session key in the scope list, issues an individual `await self._redis.get(rk)` call. With 10 sessions, this is 10 sequential Redis round trips during cleanup. |
| **Proposed Solution** | Use `redis.mget(*keys)` to fetch all session data in a single round trip. Process results in Python and then batch-delete expired keys with a pipeline. |

---

### PERF-10 · `orchestrator.delete_all` issues N sequential deletes

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/memory/orchestrator.py` — `delete_all()` |
| **Description** | Deletes all memories by fetching results and then calling `store.delete(id)` in a loop. For a tenant with 1000 memories, this is 1000 individual `DELETE` statements. |
| **Proposed Solution** | Add a `delete_by_filter(tenant_id=...)` or `delete_all(tenant_id=...)` method to the store that issues a single `DELETE FROM memories WHERE tenant_id = :t`. |

---

## 6 · Observability / Error Handling

### OBS-01 · `PostgresMemoryStore._to_schema` silently defaults unknown enums

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/storage/postgres.py` — `_to_schema()` |
| **Description** | When converting DB rows to `MemoryRecord`, unknown `MemoryType`/`MemoryStatus` string values are silently converted to defaults (e.g. `MemoryType.EPISODIC_EVENT`). This masks data integrity issues — if a record has `type="invalid_value"`, it surfaces as a valid `EPISODIC_EVENT` without any log. |
| **Proposed Solution** | Log a warning when an unknown enum value is encountered, including the record ID and the bad value. This preserves the fallback-to-default behavior while making data issues visible. |

---

### OBS-02 · Extraction silently returns empty lists on all errors

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/extraction/entity_extractor.py`, `src/extraction/relation_extractor.py`, `src/extraction/fact_extractor.py` |
| **Description** | All three extractors catch `(json.JSONDecodeError, KeyError, TypeError)` and return `[]`. There is no logging. If the LLM consistently returns malformed output (e.g. after a model change), extraction silently stops working. |
| **Proposed Solution** | Add `logger.warning("extraction_parse_failed", ...)` in each except block. Include the raw LLM response (truncated) for debugging. |

---

### OBS-03 · `track_retrieval_latency` decorator always uses `"default"` tenant

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/utils/metrics.py` — `track_retrieval_latency()` |
| **Description** | The decorator takes `tenant_id` as a parameter with default `"default"`. Since decorators are applied at definition time, the `tenant_id` is fixed. All retrieval latency metrics are recorded under the same tenant_id label unless the decorator is applied with a specific value (which is impractical for multi-tenant use). |
| **Proposed Solution** | Make the decorator extract `tenant_id` from the function arguments at call time (by inspecting `kwargs` or the first argument's `.tenant_id` attribute). |

---

### OBS-04 · Datetime timezone stripping in models

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/storage/models.py` |
| **Description** | All datetime columns use `DateTime` without `timezone=True`. The `naive_utc()` utility strips timezone info before storing. While consistent, this means any non-UTC timezone information from the application layer is silently lost and cannot be recovered. |
| **Proposed Solution** | Use `DateTime(timezone=True)` in the ORM model and store timezone-aware UTC timestamps. This preserves timezone metadata and aligns with PostgreSQL best practice (`TIMESTAMPTZ`). |

---

### OBS-05 · `SemanticChunker.chunk()` silently falls back on LLM failure without logging

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/working/chunker.py` — `SemanticChunker.chunk()` |
| **Description** | When the LLM call fails or returns unparseable JSON, the method falls back to creating a single `STATEMENT` chunk from the entire text. There is no logging of the failure. If the LLM is consistently failing, all input is chunked as single undifferentiated statements with low confidence, silently degrading memory quality. |
| **Proposed Solution** | Log a warning on fallback including the exception message. Consider incrementing a metric to track LLM chunking failure rates. |

---

### OBS-06 · `classifier._llm_classify` uses `logging.getLogger` instead of structured logger

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/retrieval/classifier.py` — `_llm_classify()` |
| **Description** | The LLM classify fallback imports `logging` inline and uses `logging.getLogger(__name__).warning(...)`. The rest of the codebase uses `structlog` via `get_logger()` from `utils/logging_config.py`. This inconsistency means classification failures may not appear in structured log pipelines. |
| **Proposed Solution** | Replace with `get_logger(__name__).warning(...)` for consistency with the project's logging infrastructure. |

---

## Recommended Prioritization

### Immediate (before next release)
1. **SEC-01** — Neo4j Cypher injection
2. **SEC-02** — Postgres raw SQL
3. **SEC-03** — Tenant ID override authorization
4. **CON-01** — Thread-safe singleton
5. **BUG-01** — Missing `commit()` in pg_session

### High Priority (next sprint)
6. **BUG-02** — Stale access count writes (lost update)
7. **BUG-03** — Metadata overwrite in consolidation
8. **BUG-04** — Non-atomic archive
9. **BUG-05** — Metadata overwrite in forgetting
10. **CON-02** — Distributed rate limiting
11. **CON-03** — Redis race in labile tracker
12. **SEC-04** — Error response information leakage

### Medium Priority (planned maintenance)
13. **SEC-05**, **SEC-06** — Dashboard search injection, subprocess call
14. **BUG-06** through **BUG-15** — Various correctness issues
15. **CON-05** — Unbounded in-memory manager growth
16. **DES-08**, **DES-10** — N×M conflict checks, circular imports
17. **PERF-01** through **PERF-04**, **PERF-07** through **PERF-09** — Performance issues
18. **OBS-01**, **OBS-02**, **OBS-05** — Silent error swallowing
19. **DES-01** through **DES-03** — Design improvements

### Low Priority (tech debt)
20. **DES-04** through **DES-07**, **DES-11** through **DES-15** — Code quality / DRY
21. **BUG-16** through **BUG-18** — Minor correctness issues
22. **PERF-05**, **PERF-06**, **PERF-10** — Minor performance / correctness
23. **OBS-03**, **OBS-04**, **OBS-06** — Observability refinements
24. **SEC-07** — Metrics cardinality DoS
25. **PERF-11** — Sequential batch encoding
26. **BUG-19** — Auth key iteration
27. **BUG-20** — Forget limit silent truncation
28. **BUG-21** — Provenance hardcoding

---

## Additional Findings (New)

### SEC-07 · Metrics cardinality DoS via Tenant ID
| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `src/api/routes.py` — `write_memory`, `read_memory`, etc. |
| **Description** | Prometheus metrics `MEMORY_WRITES` and `MEMORY_READS` use `tenant_id` as a label. If `SEC-03` (tenant override) allows arbitrary tenant IDs, or if a privileged user generates many random tenant IDs, the cardinality of the metrics will explode, causing memory exhaustion in the metrics client/server. |
| **Proposed Solution** | (1) Remove `tenant_id` from high-cardinality metrics or use a bounded set of "known" tenants. (2) Strictly validate `tenant_id` against a provisioning database before recording metrics. |

---

### PERF-11 · `HippocampalStore.encode_batch` is sequential
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/hippocampal/store.py` — `encode_batch()` |
| **Description** | The method iterates through chunks and calls `encode_chunk` sequentially. Each `encode_chunk` performs redaction, embedding (API call), and extraction (LLM call). For a batch of 10 chunks, this results in 10 sequential chains of latent operations, significantly increasing latency. |
| **Proposed Solution** | Use `asyncio.gather` to parallelize `encode_chunk` calls. Note that `WriteGate` logic might need adjustment if it strictly depends on the "previous" chunk in the batch being already committed, but generally intra-batch parallelism is acceptable. |

---

### BUG-19 · `get_auth_context` iterates all keys
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/api/auth.py` — `get_auth_context()` |
| **Description** | storage of API keys and iteration: `for known_key, ctx in api_keys.items(): ...`. While secure (constant time comparison per key), it iterates all defined keys for every request. If the number of keys grows large, this becomes a CPU bottleneck. |
| **Proposed Solution** | Use a dictionary lookup `api_keys.get(api_key)` if keys are stored in a hashable way. For constant-time comparison safety, hash the incoming key (e.g. SHA256) and look up the hash. |

---

### BUG-20 · `forget()` limit causes silent partial deletion
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/orchestrator.py` — `forget()` |
| **Description** | When `before` or `query` arguments are used, the method uses `limit=500` (or 100) to find records. If more than 500 records match, only the first 500 are deleted. The return value `{"affected_count": 500}` gives no indication that more records remain. |
| **Proposed Solution** | Change the logic to loop until all matching records are processed (pagination), or strictly document the limit and return a `has_more` flag. |

---

### BUG-21 · Provenance hardcoded to `AGENT_INFERRED`
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/memory/hippocampal/store.py` — `encode_chunk()` |
| **Description** | `provenance=Provenance(source=MemorySource.AGENT_INFERRED...)` is hardcoded. However, chunks often come directly from `ingest_turn` which processes user messages. These should likely be `MemorySource.USER_INPUT`. |
| **Proposed Solution** | Pass the `source` or `role` from `SemanticChunk` down to `encode_chunk` and set `MemorySource` accordingly. |

---

### PERF-12 · Sequential health checks in dashboard
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/api/dashboard_routes.py` — `dashboard_health()` |
| **Description** | The health check endpoint pings PostgreSQL, Neo4j, and Redis sequentially. If one service is slow, the entire health check response is delayed by the sum of latencies. |
| **Proposed Solution** | Use `asyncio.gather()` to ping all three services in parallel. |

---

### BUG-22 · Inefficient and non-atomic `MemoryOrchestrator.delete_all`
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/memory/orchestrator.py` — `delete_all()` |
| **Description** | `delete_all` fetches all record IDs first and then iterates through them, calling `store.delete(id)` for each. This is O(N) in database roundtrips and is not atomic. If the process crashes mid-way, only some records are deleted. |
| **Proposed Solution** | Implement a `delete_by_tenant(tenant_id)` method in the store layer that executes a single `DELETE` statement. |

---

### PERF-13 · Multiple individual count queries in `dashboard_overview`
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/api/dashboard_routes.py` — `dashboard_overview()` |
| **Description** | The overview endpoint issues individual `COUNT` queries for different memory types and statuses. This results in multiple roundtrips to PostgreSQL. |
| **Proposed Solution** | Use a single SQL query with `GROUP BY type, status` to fetch all counts in one roundtrip. |

---

### DES-16 · Thread-local event loop risks in `celery_app.py`
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/celery_app.py` |
| **Description** | The Celery worker setup uses `asyncio.new_event_loop()` per thread and runs forever. This can lead to race conditions or resource management issues if the parent task doesn't properly handle the lifecycle of these loops, especially during worker shutdown. |
| **Proposed Solution** | Use a more robust async Celery worker pattern (e.g., `celery[asyncio]`) or ensure strict cleanup of loops in `worker_process_shutdown` signals. |

---

### DES-17 · High complexity and coupling in `MemoryOrchestrator.create` factory
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `src/memory/orchestrator.py` — `create()` |
| **Description** | The `create` method is a massive factory that instantiates ~10 different components and stores, tightly coupling the orchestrator to every leaf implementation. This makes unit testing and swapping components difficult. |
| **Proposed Solution** | Use an abstract factory pattern or a dependency injection container to decouple instantiation from the orchestrator logic. |

---

### OBS-07 · Silent LLM failure in `LLMFactExtractor` without raw response logging
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `src/extraction/fact_extractor.py` — `LLMFactExtractor.extract()` |
| **Description** | If the LLM returns invalid JSON or fails, the extractor catches the error and returns `[]` silently. No log record is made of the raw response that failed parsing, making it impossible to debug prompt engineering issues in production. |
| **Proposed Solution** | Log the raw LLM response at `DEBUG` or `WARNING` level when parsing fails. |

---

## 8 · DeviOps / Infrastructure / Examples

### DOCKER-01 · Race condition in `docker-compose` migration command
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `docker/docker-compose.yml` — `app` and `api` command |
| **Description** | Both `app` and `api` services run `alembic upgrade head` on startup. In a scaled environment (or even just these two containers starting simultaneously), they will race to apply migrations, potentially causing database locking errors or partial application. |
| **Proposed Solution** | Move migrations to a separate init container (e.g. `service: migration`) that runs once and exits, or run them manually/CI before deployment. Remove `alembic upgrade head` from the runtime command. |

---

### CI-01 · `pip install` in CI may miss optional dependencies
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `.github/workflows/ci.yml` — `job: test` |
| **Description** | The test job installs `requirements-docker.txt` then runs `pip install -e . --no-deps`. The `pyproject.toml` defines `sentence-transformers` as a dependency (which is needed for `LocalEmbeddings` if tests use it), but `requirements-docker.txt` explicitly excludes it. If tests import `src.utils.embeddings`, they might fail on missing dependency. |
| **Proposed Solution** | Install with `pip install -e .[test]` (defining a test extra) or ensure `requirements-docker.txt` includes everything needed for the test suite to import the package successfully. |

---

### CONFIG-01 · Hardcoded placeholder credentials in `alembic.ini`
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `alembic.ini` |
| **Description** | The file contains `sqlalchemy.url = driver://user:pass@localhost/dbname`. While `env.py` overrides this with settings, having a placeholder with "user:pass" is confusing and potentially risky if fallback logic is triggered. |
| **Proposed Solution** | Comment it out or use environment variable interpolation format `%(DATABASE_URL)s` if supported, or remove it entirely relying on `env.py`. |

---

### EXAMPLE-01 · Examples duplicate `memory_client` logic
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `examples/memory_client.py` |
| **Description** | The `examples/` directory contains its own `memory_client.py` which duplicates the logic of `packages/py-cml`. It lacks features like correct connection pooling (recreates client on init) and retry logic found in the package. |
| **Proposed Solution** | Update examples to import from `py-cml` (e.g. `from cml import CognitiveMemoryClient`) and delete the local `memory_client.py`. This ensures examples demonstrate best practices using the actual SDK. |

---

### EXAMPLE-02 · `memory_client` shadows builtin `format`
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `examples/memory_client.py` — `read()` |
| **Description** | The `read` method uses `format` as an argument name, which shadows the Python builtin `format()`. This is bad style and can lead to subtle bugs if `format()` is needed within the function. |
| **Proposed Solution** | Rename the parameter to `response_format` to match the package client and avoid shadowing. |

---

## Python Client (py-cml) — packages/py-cml/src

> Full code review of the CognitiveMemoryLayer Python SDK (`packages/py-cml/src/cml`).
> Issues below are specific to the client package; solutions target the py-cml codebase.

### Summary (py-cml)

| Category | CRITICAL | HIGH | MEDIUM | LOW |
|----------|----------|------|--------|-----|
| Correctness / Bugs | 0 | 2 | 4 | 3 |
| Design / Architecture | 0 | 1 | 3 | 4 |
| Integration / Dependencies | 0 | 1 | 2 | 0 |
| Performance | 0 | 0 | 1 | 1 |
| API / Consistency | 0 | 0 | 2 | 2 |
| **Total** | **0** | **4** | **12** | **10** |

---

### PY-BUG-01 · `batch_write` / `batch_read` do not validate item shape

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `packages/py-cml/src/cml/client.py`, `async_client.py` — `batch_write()` |
| **Description** | `batch_write(items, ...)` expects each item to be a dict with at least `"content"`. The code does `item["content"]`, so if any item lacks `"content"` a `KeyError` is raised. No validation or helpful error message. |
| **Proposed Solution** | Validate each item (e.g. require `"content"` key) and raise `ValidationError` or `ValueError` with a clear message (e.g. "Each item must have a 'content' key"). Optionally accept Pydantic models and convert. |

---

### PY-BUG-02 · `serialize_for_api` does not serialize UUID/datetime inside lists

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `packages/py-cml/src/cml/utils/serialization.py` — `serialize_for_api()` |
| **Description** | For list values, the code only recurses when the element is a dict: `[serialize_for_api(v) if isinstance(v, dict) else v for v in value]`. UUID and datetime inside lists are left unchanged, so `json` serialization (e.g. in httpx) can fail or behave inconsistently when request payloads contain lists of UUIDs or datetimes. |
| **Proposed Solution** | When processing list elements, also convert UUID to str and datetime to ISO string (e.g. helper that handles UUID, datetime, dict, list, and passthrough for other types). Apply recursively so nested structures are fully serializable. |

---

### PY-BUG-03 · Retry logic ignores `config.max_retry_delay`

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/transport/retry.py` |
| **Description** | Backoff is capped with a module-level `MAX_RETRY_DELAY = 60.0`. `CMLConfig` defines `max_retry_delay` (and loads it from `CML_MAX_RETRY_DELAY`), but the retry module never uses it, so user/config max delay is ignored. |
| **Proposed Solution** | In `_sleep_with_backoff` and `_async_sleep_with_backoff`, accept `config: CMLConfig` (or `max_delay: float`) and use `min(delay, config.max_retry_delay)` instead of `MAX_RETRY_DELAY`. Thread config through from `retry_sync` / `retry_async`. |

---

### PY-BUG-04 · `dashboard_item_to_memory_item` can raise `KeyError` for missing `id`

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/utils/converters.py` — `dashboard_item_to_memory_item()` |
| **Description** | The function uses `raw["id"]` without checking presence. If the dashboard API returns an item without `id` (malformed or schema change), callers get `KeyError` with no context. |
| **Proposed Solution** | Use `raw.get("id")` and raise a descriptive `ValueError` or `ValidationError` if `id` is missing (e.g. "Dashboard item must include 'id'"). Optionally coerce `id` to UUID and handle invalid UUID strings with a clear error. |

---

### PY-BUG-05 · `EmbeddedCognitiveMemoryLayer.read()` parameter `format` is unused

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/embedded.py` — `read()` |
| **Description** | The method accepts `format: str = "packet"` but never passes it to the orchestrator or uses it. Response shape is always the same. The parameter also shadows the builtin `format`. Callers may expect "list" or "llm_context" to change behavior. |
| **Proposed Solution** | Either (1) remove the parameter and document that embedded read returns a fixed structure, or (2) wire `format` to the orchestrator/response builder (e.g. only include `llm_context` when `format == "llm_context"`) and rename to `response_format` to match the HTTP client API and avoid shadowing. |

---

### PY-BUG-06 · Redundant `memory_types` length check in `iter_memories`

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/client.py`, `async_client.py` — `iter_memories()` |
| **Description** | The method raises if `len(memory_types) > 1` at entry, then inside the loop the same condition is checked again with a second `raise ValueError(...)`. The inner check is unreachable. |
| **Proposed Solution** | Remove the duplicate check inside the loop. Keep the single validation at the start of the method. |

---

### PY-BUG-07 · `import_memories_async` blocks event loop when target is sync client

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/embedded_utils.py` — `import_memories_async()` |
| **Description** | When `target` is `CognitiveMemoryLayer` (sync), the code calls `target.write(text, metadata=meta)` inside the async function. Sync `write()` blocks the event loop for every line, which can cause noticeable freezes for large imports. |
| **Proposed Solution** | For sync clients, run each write in a thread: e.g. `await asyncio.to_thread(target.write, text, metadata=meta)` (or `run_in_executor`), so the event loop is not blocked. |

---

### PY-BUG-08 · Exception aliases shadow builtins

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/exceptions.py` |
| **Description** | `ConnectionError` and `TimeoutError` are exported as aliases for CML-specific exceptions. They shadow Python's builtin `ConnectionError` and `TimeoutError`. Code that does `except TimeoutError` may catch the wrong one depending on import order; docstrings in client refer to "ConnectionError and TimeoutError" which is ambiguous. |
| **Proposed Solution** | Document clearly that CML exports are `CMLConnectionError` and `CMLTimeoutError` and that the aliases exist for backward compatibility. In new code and examples, prefer the `CML*` names. Consider deprecating the aliases in a future major version. |

---

### PY-BUG-09 · `AsyncCognitiveMemoryLayer.set_tenant` has no lock

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/async_client.py` — `set_tenant()` |
| **Description** | The sync client uses `with self._lock` in `set_tenant` to avoid races. The async client updates `self._config.tenant_id` and closes the transport without any lock. Concurrent `set_tenant` and request calls could see inconsistent state. |
| **Proposed Solution** | Add an `asyncio.Lock` and use `async with self._lock` in `set_tenant` (and optionally around transport access if needed) so tenant switch is atomic with respect to in-flight requests. |

---

### PY-DES-01 · Embedded mode depends on server codebase imports

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `packages/py-cml/src/cml/embedded.py`, `storage/sqlite_store.py` |
| **Description** | Embedded mode imports from `src.memory.orchestrator`, `src.utils.embeddings`, `src.utils.llm`, `src.retrieval.packet_builder`, `src.core.enums`, `src.core.schemas`, `src.storage.base`. These paths assume the CML server/engine is installed (e.g. from repo root). When py-cml is installed as a standalone package without the engine, embedded mode fails with opaque import errors. |
| **Proposed Solution** | Document that embedded mode requires the full CML engine (monorepo or a separate engine package). Optionally provide a lazy import wrapper that raises a clear error with install instructions. Long-term: ship a minimal engine or a formal `cml-engine` dependency so embedded works from PyPI without the repo. |

---

### PY-DES-02 · Duplicate code between sync and async clients

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/client.py`, `async_client.py` |
| **Description** | Sync and async clients duplicate almost all method signatures and docstrings. Changes (e.g. new parameters, new endpoints) must be applied in two places, increasing risk of drift and inconsistency. |
| **Proposed Solution** | Consider a shared layer: e.g. a generic implementation parameterized by transport (sync vs async) and small sync/async facades that delegate, or code generation from a single spec. At minimum, add a test or checklist that asserts parity of public method signatures. |

---

### PY-DES-03 · `CMLOpenAIHelper` only supports sync client

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/integrations/openai_helper.py` |
| **Description** | `CMLOpenAIHelper` takes `CognitiveMemoryLayer` (sync). There is no equivalent for `AsyncCognitiveMemoryLayer`. Async applications must either use the sync client (blocking) or implement their own helper. |
| **Proposed Solution** | Add `AsyncCMLOpenAIHelper` that accepts `AsyncCognitiveMemoryLayer` and exposes async `chat()` (and optionally async `get_context` / store methods). Consider a protocol or base that both sync and async helpers implement. |

---

### PY-DES-04 · `MemoryProvider` protocol is sync-only

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/integrations/openai_helper.py` — `MemoryProvider` |
| **Description** | The protocol defines `get_context`, `store_exchange`, `clear_session` as synchronous. Async backends (e.g. async CML client) cannot implement the protocol without blocking. |
| **Proposed Solution** | Introduce an async variant of the protocol (e.g. `AsyncMemoryProvider` with async methods) or allow protocol methods to return awaitables so both sync and async implementations can be used behind a common interface. |

---

### PY-DES-05 · `HTTPTransport` rebuilds headers on every request when using admin key

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/transport/http.py` |
| **Description** | For `use_admin_key=True`, `_do_request` calls `_build_headers(use_admin_key=True)` and passes them per request. The default client is created with normal headers; admin requests overlay headers. No caching of admin headers, so small extra work per admin call. |
| **Proposed Solution** | Low priority. Optionally cache the two header dicts (normal and admin) on the transport and reuse. Only worth it if profiling shows header build in hot path. |

---

### PY-DES-06 · Storage module uses `__getattr__` for lazy load

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/storage/__init__.py` |
| **Description** | `SQLiteMemoryStore` is loaded via `__getattr__` to avoid importing `sqlite_store` (and thus aiosqlite and engine deps) until needed. This can surprise static analysis and IDE "go to definition", and the error message on missing attribute is generic. |
| **Proposed Solution** | Document the lazy load in the module docstring. Alternatively, use a direct import and make aiosqlite/engine imports conditional inside `sqlite_store.py` so `from cml.storage import SQLiteMemoryStore` works without pulling heavy deps until the class is used. |

---

### PY-DES-07 · No explicit upper bound on `max_results` in read/get_context

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/client.py`, `async_client.py` |
| **Description** | `ReadRequest` enforces `ge=1, le=50` for `max_results`, but the public `read()` and `get_context()` accept `max_results: int = 10` without validation before building the request. A caller passing `max_results=1000` gets a validation error from Pydantic rather than a clear SDK-level message. |
| **Proposed Solution** | Either validate in the client and raise with a message like "max_results must be between 1 and 50", or document that validation is delegated to the request model and link to the same bounds. |

---

### PY-DES-08 · `ReadResponse` uses `llm_context` and `context` property

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/models/responses.py` |
| **Description** | The model has both `llm_context: str | None` and a `context` property that returns `llm_context or ""`. Two ways to access the same value can confuse users and documentation. |
| **Proposed Solution** | Document that `context` is the preferred shorthand for LLM injection and that `llm_context` is the raw field. Optionally deprecate `llm_context` in favor of `context` in a future version, or keep both and state the relationship in the docstring. |

---

### PY-INT-01 · SQLite store requires engine schemas and base class

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `packages/py-cml/src/cml/storage/sqlite_store.py` |
| **Description** | The module imports `MemoryStoreBase`, `MemoryRecord`, `MemoryRecordCreate`, `Provenance`, `EntityMention`, `Relation`, and enums from `src.core.*` and `src.storage.base`. These live in the CML server/engine repo. Standalone py-cml install (e.g. from PyPI) cannot use embedded lite mode without the engine. |
| **Proposed Solution** | Same as PY-DES-01: document the requirement clearly and/or provide a dedicated engine package. If py-cml is to be distributable without the repo, consider a minimal abstract interface in py-cml and an optional dependency that supplies the engine implementation. |

---

### PY-INT-02 · Embedded `_packet_to_read_response` imports server code

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/embedded.py` — `_packet_to_read_response()` |
| **Description** | The function tries to import `MemoryPacketBuilder` from `src.retrieval.packet_builder` and on failure falls back to a simple string join. The import path is server-specific; failure is silent (warning log) and changes behavior. |
| **Proposed Solution** | Document that full LLM context building requires the engine. Consider making the fallback explicit (e.g. a parameter or config flag "use_simple_context") so behavior is predictable when the engine is not available. |

---

### PY-INT-03 · Embedded utils export accesses private `_ensure_initialized`

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/embedded_utils.py` — `export_memories_async()` |
| **Description** | The function calls `source._ensure_initialized()` and uses `source._orchestrator.hippocampal.store`. These are private attributes; refactors of `EmbeddedCognitiveMemoryLayer` could break the export API. |
| **Proposed Solution** | Add a public method on `EmbeddedCognitiveMemoryLayer` such as `get_store()` or `export_memories_to_path()` that encapsulates the dependency, and implement `export_memories_async` by calling it. This keeps the public API stable and hides internals. |

---

### PY-PERF-01 · Sync `batch_write` and `batch_read` are strictly sequential

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/client.py` |
| **Description** | `batch_write` loops over items and calls `self.write()` for each; `batch_read` builds a list of `self.read()` calls. Sync client cannot overlap I/O. For large batches, latency is the sum of all request latencies. |
| **Proposed Solution** | Document that sync batch methods are sequential. For parallelism, suggest using the async client (e.g. `batch_read` with `asyncio.gather`). Optionally add an optional `concurrency` parameter that uses a thread pool to run multiple requests in parallel (with a note on GIL and connection limits). |

---

### PY-PERF-02 · Sync client connection reuse after `set_tenant`

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/client.py` — `set_tenant()` |
| **Description** | `set_tenant` closes the transport so the next request creates a new httpx client. Connection reuse is lost on tenant switch. This is correct behavior but may cause a short latency spike on the first request after switch. |
| **Proposed Solution** | Document that switching tenant closes the connection and the next request will open a new one. No code change required unless connection pooling across tenants is desired. |

---

### PY-API-01 · Inconsistent parameter name: `response_format` vs `format`

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `packages/py-cml/src/cml/client.py` (`response_format`), `embedded.py` (`format`) |
| **Description** | The HTTP client uses `response_format` in `read()` and `ReadRequest` (with alias `format` for JSON). The embedded client uses `format` in `read()`. Naming differs and embedded `format` is unused (see PY-BUG-05). |
| **Proposed Solution** | Use `response_format` everywhere (sync, async, embedded) and align with `ReadRequest`. In embedded, wire it to behavior and remove the builtin shadow. |

---

### PY-API-02 · `get_context` does not accept `response_format`

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/client.py`, `async_client.py` — `get_context()` |
| **Description** | `get_context()` is a convenience that calls `read(..., response_format="llm_context")` and returns `result.context`. It does not expose `response_format`; callers who want "list" or "packet" and then format themselves cannot use `get_context`. |
| **Proposed Solution** | Either keep as-is and document that `get_context` is for LLM context only, or add an optional `response_format` (default `"llm_context"`) and return the appropriate attribute (e.g. `result.context` for llm_context, or a serialized form for others). |

---

### PY-API-03 · `ForgetRequest` / `forget()` use string `action`; literal in type hint

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `packages/py-cml/src/cml/models/requests.py`, `client.py` — `ForgetRequest`, `forget()` |
| **Description** | `action` is typed as `Literal["delete", "archive", "silence"]` in the request and in the client method. The API and server may accept additional actions in the future; the client would need a release to support them. |
| **Proposed Solution** | Document the supported actions. If the server adds new actions, consider accepting `str` and passing through, with a note in the docstring listing supported values, so new server features work without a client release. |

---

### Recommended prioritization (py-cml)

**Immediate**
1. **PY-BUG-01** — Validate `batch_write` item shape.
2. **PY-BUG-02** — Serialize UUID/datetime in lists in `serialize_for_api`.
3. **PY-DES-01** / **PY-INT-01** — Document or fix embedded/engine dependency.

**High**
4. **PY-BUG-03** — Use `config.max_retry_delay` in retry.
5. **PY-BUG-04** — Safe handling of missing `id` in dashboard converter.
6. **PY-BUG-05** — Use or remove `format` in embedded `read()`.
7. **PY-BUG-07** — Non-blocking sync client write in `import_memories_async`.
8. **PY-DES-03** — Async OpenAI helper.

**Medium**
9. **PY-BUG-06**, **PY-BUG-08**, **PY-BUG-09** — Redundant check, exception aliases, async lock.
10. **PY-DES-02**, **PY-DES-04**, **PY-INT-02**, **PY-INT-03** — Duplication, protocol, imports, export API.
11. **PY-PERF-01** — Document or improve sync batch behavior.
12. **PY-API-01** — Unify `response_format` / `format`.

**Low**
13. **PY-DES-05** through **PY-DES-08**, **PY-PERF-02**, **PY-API-02**, **PY-API-03** — Minor design and API polish.

---

## Root / Repo (excluding packages and src)

> Issues in config, docker, examples, migrations, scripts, tests, and repo root.
> Does not duplicate existing DOCKER-01, CI-01, CONFIG-01, EXAMPLE-01, EXAMPLE-02.

### Summary (root/repo)

| Category | CRITICAL | HIGH | MEDIUM | LOW |
|----------|----------|------|--------|-----|
| Config / Migrations | 0 | 1 | 2 | 1 |
| Tests | 0 | 0 | 0 | 1 |
| Scripts | 0 | 0 | 2 | 0 |
| Examples | 0 | 0 | 4 | 1 |
| CI / Docker | 0 | 0 | 1 | 2 |
| Documentation | 0 | 0 | 0 | 2 |
| **Total** | **0** | **1** | **9** | **7** |

---

### ROOT-CONFIG-01 · Empty config files

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `config/logging.yaml`, `config/settings.yaml` |
| **Description** | Both files are empty. If application or documentation expects default logging/settings content here, they provide no value. Scripts (e.g. `scripts/init_structure.py`) reference them, so they exist as placeholders only. |
| **Proposed Solution** | Either (1) add minimal default content (e.g. logging level, empty settings schema) and document usage, or (2) remove from init_structure and document that config is driven by env/.env and code defaults. |

---

### ROOT-MIG-01 · Alembic env fallback leaves invalid URL on config failure

| Attribute | Details |
|-----------|---------|
| **Severity** | HIGH |
| **File** | `migrations/env.py` |
| **Description** | When `get_settings()` fails (e.g. `ImportError`, missing `.env`, or config validation error), the code does `except Exception: pass` and does not set `sqlalchemy.url`. Alembic then uses the placeholder from `alembic.ini` (`driver://user:pass@localhost/dbname`), which is invalid for async PostgreSQL. Migrations may fail with an opaque driver or connection error. |
| **Proposed Solution** | On exception, set a clear placeholder and log a warning, or re-raise with a message like "Could not load database URL from settings. Set DATABASE__POSTGRES_URL or fix config." Avoid silently falling back to an invalid URL. |

---

### ROOT-MIG-02 · Migration imports get_settings at module load

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `migrations/versions/001_initial_schema.py` |
| **Description** | The migration does `from src.core.config import get_settings` and `_EMBEDDING_DIM = get_settings().embedding.dimensions` at import time. If the environment has no valid config (e.g. no .env in CI step before env vars are set), the migration module fails to load and Alembic cannot list or run migrations. The dimension is fixed once at load time. |
| **Proposed Solution** | Document that running migrations requires a valid app config (or set EMBEDDING__DIMENSIONS in env). Optionally read dimension inside `upgrade()` from env with a default (e.g. 1536) so the migration file can load without full config. |

---

### ROOT-MIG-03 · CONFIG-01 (alembic.ini placeholder) — see existing

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `alembic.ini` |
| **Description** | Already documented as CONFIG-01. Placeholder URL can confuse when env.py fallback is used. |
| **Proposed Solution** | See CONFIG-01. |

---

### ROOT-TEST-01 · Deprecated datetime.utcnow() in tests

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `tests/conftest.py` — `sample_memory_record` fixture |
| **Description** | The fixture uses `datetime.utcnow()` for `timestamp` and `written_at`. In Python 3.12+ `datetime.utcnow()` is deprecated in favor of `datetime.now(datetime.UTC)` or `datetime.now(timezone.utc)`. |
| **Proposed Solution** | Replace with `from datetime import datetime, timezone` and `datetime.now(timezone.utc)` (or `datetime.now(datetime.UTC)` on 3.11+). |

---

### ROOT-SCRIPT-01 · init_structure.py is outdated and risky

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `scripts/init_structure.py` |
| **Description** | The `STRUCTURE` dict does not match the current repo: e.g. `src.api` lists `admin_routes.py` but the codebase uses `dashboard_routes.py`; examples list omits `ollama_chat_test.py`, `ollama_chatbot_app.py`; tests list omits `test_dashboard_flow.py`, `test_dashboard_routes.py`. The script only `touch()`es paths, so running it creates empty files for missing entries and does not remove obsolete ones. New contributors or automation might run it and get a wrong or incomplete layout. |
| **Proposed Solution** | (1) Update STRUCTURE to match the current tree and add a comment that it must be kept in sync. (2) Or make the script read the desired layout from a manifest file that is updated when structure changes. (3) Add a warning or dry-run mode that prints what would be created. Consider deprecating the script if the repo is no longer "scaffolded" from scratch. |

---

### ROOT-EX-01 · examples/memory_client parse_item can raise KeyError

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `examples/memory_client.py` — `parse_item()` inside `read()` |
| **Description** | `parse_item` uses `item["id"]`, `item["timestamp"]`, `item["relevance"]` etc. If the API response omits any of these (e.g. optional field, schema change, or error payload), the code raises `KeyError` with no helpful message. |
| **Proposed Solution** | Use `.get()` with sensible defaults (e.g. `item.get("id")` and raise a clear error if missing; `item.get("timestamp")` with fallback to now or skip; `item.get("relevance", 0.0)`). Alternatively, document that the client expects the full API schema and catch KeyError to raise a more descriptive error. |

---

### ROOT-EX-02 · Async memory client error handling inconsistent with sync

| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `examples/memory_client.py` — `AsyncCognitiveMemoryClient._request()` |
| **Description** | The sync `_request` builds a detailed error message from `response.json()` or `response.text` and raises `httpx.HTTPStatusError` with that message. The async `_request` only calls `response.raise_for_status()` and returns `response.json()`; on 4xx/5xx it raises without including the response body in the exception message, making debugging harder. |
| **Proposed Solution** | Mirror the sync behavior: on non-success, read body/detail and raise an exception that includes status and body (e.g. same `HTTPStatusError` pattern or a custom exception with `response.text` / `response.json()`). |

---

### ROOT-EX-03 · EXAMPLE-01 / EXAMPLE-02 — see existing

| Attribute | Details |
|-----------|---------|
| **Severity** | — |
| **File** | `examples/` |
| **Description** | EXAMPLE-01 (examples duplicate py-cml client), EXAMPLE-02 (parameter name `format` shadows builtin) are already documented above. |
| **Proposed Solution** | See EXAMPLE-01, EXAMPLE-02. |

---

### ROOT-CI-01 · CI does not set AUTH vars for tests

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `.github/workflows/ci.yml` — `job: test` env |
| **Description** | The test job sets `DATABASE__*` and `NEO4J_*` but not `AUTH__API_KEY` or `AUTH__ADMIN_API_KEY`. Tests that need auth use `monkeypatch` in their fixtures, so they work. Any test or app code that reads auth from env without going through a patched fixture could see None and behave differently than in a local run with .env. |
| **Proposed Solution** | Add to the test job env: `AUTH__API_KEY: test-key` and `AUTH__ADMIN_API_KEY: test-key` so the default app config is valid for all tests. This makes CI behavior consistent and avoids subtle failures if a new test forgets to monkeypatch. |

---

### ROOT-DOCKER-02 · Dockerfile HEALTHCHECK inappropriate for app service

| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `docker/Dockerfile`, `docker/docker-compose.yml` |
| **Description** | The Dockerfile defines `HEALTHCHECK` that curls the API on port 8000. The `app` service in compose overrides `command` to `alembic upgrade head && pytest tests` and exits after tests. For that service, the healthcheck is irrelevant (container exits) and could briefly run before exit. Only the `api` service keeps the server running; for `api` the healthcheck is correct. |
| **Proposed Solution** | Document in the Dockerfile or compose that the default image is intended for the `api` service. Optionally, in compose for the `app` service set `healthcheck: disable` so the test runner is not expected to be healthy. |

### ROOT-DOC-01 · Broken link in `CONTRIBUTING.md`
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `CONTRIBUTING.md` |
| **Description** | The file references `ProjectPlan/CurrentIssues.md` which has been renamed or consolidated into `ProjectPlan/BaseCML/Issues.md`. |
| **Proposed Solution** | Update the link to point to the correct issues tracking file. |

---

### ROOT-DOC-02 · Static test count badge in `README.md`
| Attribute | Details |
|-----------|---------|
| **Severity** | LOW |
| **File** | `README.md` |
| **Description** | The README contains a static badge: "Tests 138 Passed". This is misleading if the test suite grows or some tests fail, as it doesn't reflect the actual CI state. |
| **Proposed Solution** | Use a dynamic badge from GitHub Actions or remove the "138 Passed" hardcoded text, leaving just "Tests". |

---

### ROOT-SCRIPT-02 · Missing error handling in `init_structure.py`
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `scripts/init_structure.py` |
| **Description** | The script performs `mkdir` and `touch` without any try/except blocks. If run in a directory with restricted permissions, it will crash with an unhandled exception. |
| **Proposed Solution** | Add basic error handling and print descriptive messages if a directory or file cannot be created. |

---

### ROOT-CI-02 · Redundant/slow wait loop for Neo4j in CI
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `.github/workflows/ci.yml` |
| **Description** | The `test` job has a manual `wget` loop to wait for Neo4j, while the service already has a `healthcheck`. GHA services wait for healthchecks if configured; the secondary wait with a 2-second sleep is redundant and slows down CI. |
| **Proposed Solution** | Reliable GHA service healthchecks should be enough. If a secondary wait is needed, use a tool like `timeout 60 sh -c 'until wget...; do sleep 1; done'` for better efficiency. |

---

### ROOT-EX-04 · Brittle JSON parsing in `chatbot_with_memory.py` extraction
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `examples/chatbot_with_memory.py` — `_extract_memorable_info()` |
| **Description** | While the extraction logic is fairly robust, it catches `Exception` and prints a warning without logging the actual content that failed to parse. This makes it hard for developers using the example to debug why extraction isn't working for their specific model/prompt. |
| **Proposed Solution** | Log the raw response content from the LLM when `json.loads` or coordinate access fails. |

---

### ROOT-EX-05 · Silent failure on missing `AUTH__API_KEY` in examples
| Attribute | Details |
|-----------|---------|
| **Severity** | MEDIUM |
| **File** | `examples/chatbot_with_memory.py`, `basic_usage.py` |
| **Description** | The client is initialized with `os.environ.get("AUTH__API_KEY", "")`. If the variable is missing, it defaults to an empty string. The `CognitiveMemoryClient` (from `memory_client.py`) does not validate the key on init, leading to 401 errors during the first API call, which can be confusing for new users. |
| **Proposed Solution** | Add a check for empty API key in the example script and print a helpful error message + `sys.exit(1)` before starting the conversation loop. |

---

### Recommended prioritization (root/repo)

**High**
1. **ROOT-MIG-01** — Fix Alembic env fallback so invalid URL is not used silently.

**Medium**
2. **ROOT-MIG-02** — Migration config load at import.
3. **ROOT-SCRIPT-01**, **ROOT-SCRIPT-02** — Update/deprecate/robustify init_structure.py.
4. **ROOT-EX-01** through **ROOT-EX-05** — Improve example robustness and error feedback.
5. **ROOT-CI-02** — Optimize Neo4j wait loop in CI.

**Low**
6. **ROOT-CONFIG-01**, **ROOT-TEST-01**, **ROOT-CI-01**, **ROOT-DOCKER-02**, **ROOT-DOC-01**, **ROOT-DOC-02** — General housekeeping.
