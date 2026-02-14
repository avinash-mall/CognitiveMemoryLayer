# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **user_timezone for read and turn** — Optional `user_timezone` parameter on `read()`, `read_safe()`, and `turn()` (sync, async, and embedded clients). When provided (e.g. `"America/New_York"`), the server uses it for timezone-aware "today"/"yesterday" filters in retrieval. Requires a CML server that accepts `user_timezone` on the read and turn APIs.
- **Optional embedding/LLM in tests** — Embedded lite-mode tests `test_write_and_read` and `test_persistent_storage` now skip (instead of fail) when the embedding model is unavailable. Skip is triggered on `ImportError`, `OSError`, or `RuntimeError`, or when any other exception message contains "model", "embed", or "rate" (e.g. sentence-transformers load failure, API or rate-limit errors). See repo `tests/README.md` § Optional LLM/embedding tests.

### Fixed

- **Embedded read() filter passthrough** — `EmbeddedCognitiveMemoryLayer.read()` now passes `memory_types`, `since`, and `until` to the orchestrator (previously accepted but ignored).

### Changed

- **Embedded config from .env** — `EmbeddedEmbeddingConfig` now reads `EMBEDDING__DIMENSIONS` from the environment when set; default remains 384. Enables alignment with server embedding dimension when using the same `.env`. See [configuration](docs/configuration.md).

### Documentation

- **configuration.md** — Embedded configuration table documents that `dimensions` can be set via `EMBEDDING__DIMENSIONS` in `.env` (default 384). Added "Server-side feature flags and retrieval" subsection with link to main project Configuration Reference for `FEATURES__*` and `RETRIEVAL__*`.
- **api-reference.md** — Documented `user_timezone` on `read()` and `turn()` (optional IANA timezone for "today"/"yesterday" retrieval).
- **examples.md** — Mentioned `read(..., user_timezone=...)` and `turn(..., user_timezone=...)` for timezone-aware queries.
- **README.md** — Added one line about optional `user_timezone` on read/turn in the Memory API features table.

## [1.1.0] - 2026-02-12

### Added

- **Dashboard admin methods** — 13 new admin methods on both sync and async clients:
  - `get_sessions()` — list active sessions from Redis with TTL and memory counts
  - `get_rate_limits()` — current rate-limit usage per API key
  - `get_request_stats()` — hourly request volume over the last N hours
  - `get_graph_stats()` — knowledge graph node/edge statistics from Neo4j
  - `explore_graph()` — explore entity neighborhood in the knowledge graph
  - `search_graph()` — search entities by name pattern
  - `get_config()` — read-only configuration snapshot (secrets masked)
  - `update_config()` — set runtime configuration overrides via Redis
  - `get_labile_status()` — reconsolidation / labile memory status per tenant
  - `test_retrieval()` — interactive retrieval test via dashboard API
  - `get_jobs()` — consolidation and forgetting job history
  - `bulk_memory_action()` — archive, silence, or delete memories in bulk
- All new methods are also available on `NamespacedClient` / `AsyncNamespacedClient`

## [1.0.11] - 2025-02-11

### Added

- **Eval mode for write** — `write()` and `remember()` now accept an optional `eval_mode: bool = False`. When `True`, the client sends the `X-Eval-Mode: true` header; the server (and embedded orchestrator) return `eval_outcome` (`"stored"` or `"skipped"`) and `eval_reason` in the response. Use this in benchmark or evaluation scripts to aggregate write-gate gating statistics. Available on sync, async, and embedded clients. `WriteResponse` includes optional `eval_outcome` and `eval_reason` fields.

## [1.0.10] - 2025-02-12

### Changed

- **delete_all** — The server now implements `DELETE /api/v1/memory/all`; `delete_all(confirm=True)` no longer returns 404 when using the admin API key.

## [1.0.9] - 2025-02-12

### Added

- **Optional timestamp parameter**
  - `write(timestamp=...)` and `turn(timestamp=...)` methods now accept optional `datetime` for event time
  - Enables temporal fidelity for historical data replay (e.g., benchmark evaluation)
  - Available in sync, async, and embedded clients
  - Defaults to server-side "now" if not provided

### Changed

- **Configuration and examples:** No hardcoded API URLs or model names. Set `CML_BASE_URL`, `CML_API_KEY`, and (for OpenAI helper and chat examples) `OPENAI_MODEL` or `LLM__MODEL` in `.env`. Embedded config reads `LLM__MODEL`, `EMBEDDING__MODEL`, `LLM__BASE_URL`, `EMBEDDING__BASE_URL` from env. See [Configuration](docs/configuration.md) and [.env.example](../../.env.example).

### Added (Previous Phases)

#### Phase 1: Project setup
- Initial project structure and build configuration (Hatchling, src layout)
- Package `cml` with sync and async client classes (`CognitiveMemoryLayer`, `AsyncCognitiveMemoryLayer`)
- Development tooling: ruff, mypy, pytest, pre-commit, editorconfig
- CI workflows: py-cml-test (Python 3.11–3.13), py-cml-lint, py-cml-publish (on `py-cml-v*` tags)
- README, CONTRIBUTING, and CHANGELOG

#### Phase 2: Core client SDK
- **Configuration:** Pydantic `CMLConfig` with validation; loading from direct params, env vars (`CML_*`), and `.env` (python-dotenv)
- **Exceptions:** Full hierarchy — `CMLError` (with `status_code`, `response_body`), `AuthenticationError`, `AuthorizationError`, `NotFoundError`, `ValidationError`, `RateLimitError` (with `retry_after`), `ServerError`, `ConnectionError`, `TimeoutError`
- **Transport:** Sync `HTTPTransport` and async `AsyncHTTPTransport` (httpx), path prefix `/api/v1`, standard headers (API key, tenant ID, User-Agent), status-code mapping to exceptions
- **Retry:** Exponential backoff with jitter for 5xx, 429, connection, and timeout; `RateLimitError` respects `Retry-After` header; retry integrated in transport `request()`
- **Models:** Enums `MemoryType`, `MemoryStatus`, `MemorySource`, `OperationType`; request models (`WriteRequest`, `ReadRequest`, `TurnRequest`, `UpdateRequest`, `ForgetRequest`); response models (`HealthResponse`, `WriteResponse`, `ReadResponse`, `TurnResponse`, etc., `MemoryItem`)
- **Clients:** Context manager support (`with` / `async with`), `close()`, `health()` returning `HealthResponse`
- Unit tests for config, exceptions, retry, transport (mocked), and client health (29 tests)

#### Phase 3: Memory operations
- **Memory API:** `write`, `read`, `turn`, `update`, `forget`, `stats` on both sync and async clients
- **Sessions:** `create_session`, `get_session_context`; request/response models `CreateSessionRequest`, `SessionContextResponse`
- **Convenience:** `get_context` (read with format=llm_context), `remember` (alias for write), `search` (alias for read)
- **Admin:** `delete_all(confirm=True)` (requires confirm; server route may be added later)
- **Validation:** `forget()` requires at least one of `memory_ids`, `query`, or `before`; docstrings with Args, Returns, Raises
- Unit tests for memory operations (mocked transport): write/read responses, forget/delete_all validation, get_context, remember/search delegation

#### Phase 4: Embedded mode
- **EmbeddedCognitiveMemoryLayer:** In-process CML engine with same API as HTTP client (write, read, turn, update, forget, stats, get_context, remember, search, create_session, get_session_context, delete_all)
- **EmbeddedConfig:** Storage mode (lite/standard/full), database, embedding, and LLM config; lite mode uses SQLite + local embeddings
- **SQLite memory store:** `cml.storage.sqlite_store.SQLiteMemoryStore` implementing engine MemoryStoreBase; in-memory cosine similarity for vector search (~10k records)
- **Lite mode:** Zero-config `EmbeddedCognitiveMemoryLayer()` uses in-memory SQLite and sentence-transformers; optional `db_path` for persistence
- **Background workers:** Optional asyncio tasks for consolidation and forgetting when `auto_consolidate` / `auto_forget` are True
- **Export/import:** `cml.embedded_utils.export_memories_async`, `import_memories_async` (and sync wrappers) for migration between embedded and server
- **Parent engine:** HippocampalStore accepts MemoryStoreBase; MemoryOrchestrator.create_lite(episodic_store, embedding_client, llm_client); NoOpGraphStore and NoOpFactStore for lite
- Unit tests for embedded config and mocked orchestrator

#### Phase 5: Advanced features
- **Admin operations:** `consolidate(tenant_id=..., user_id=...)` and `run_forgetting(tenant_id=..., user_id=..., dry_run=..., max_memories=...)` (dashboard routes; require admin API key)
- **Batch operations:** `batch_write(items, session_id=..., namespace=...)` (sequential) and `batch_read(queries, max_results=..., format=...)` (concurrent on async, sequential on sync)
- **Tenant management:** `set_tenant(tenant_id)`, `tenant_id` property, `list_tenants()` (admin only)
- **Event log:** `get_events(limit=..., page=..., event_type=..., since=...)` (admin only)
- **Component health:** `component_health()` (admin only)
- **Namespace isolation:** `with_namespace(namespace)` returning `NamespacedClient` / `AsyncNamespacedClient` that inject namespace into write/update/batch_write
- **Memory iteration:** `iter_memories(memory_types=..., status=..., batch_size=...)` yielding `MemoryItem` with pagination (admin only)
- **OpenAI integration:** `cml.integrations.CMLOpenAIHelper(memory_client, openai_client, model=...)` with `chat(user_message, session_id=..., system_prompt=..., extra_messages=...)`; optional `MemoryProvider` protocol
- Unit tests for Phase 5 (mocked transport)

#### Phase 6: Developer experience
- **Structured errors:** CMLError and subclasses include optional `suggestion`, `request_id`; `_raise_for_status` passes actionable suggestions per status code
- **Logging:** `cml.utils.logging` with `configure_logging(level, handler)`, `_redact()`; DEBUG log for request/response timing in transport; DEBUG/WARNING in retry
- **Graceful degradation:** `read_safe(query, **kwargs)` on sync and async clients returns empty ReadResponse on ConnectionError/TimeoutError
- **TypedDict:** `ConsolidationResult`, `ForgettingResult` in models for admin return types
- **Response __str__:** ReadResponse, WriteResponse, StatsResponse have human-readable `__str__`
- **Serialization:** `CMLJSONEncoder`, `serialize_for_api()` in `cml.utils.serialization`
- **Session context manager:** `memory.session(name=..., ttl_hours=...)` (sync and async) yielding SessionScope / AsyncSessionScope with write, read, turn, remember
- **HTTP/2 and limits:** httpx Client/AsyncClient created with `http2=True` and `Limits(max_connections=100, ...)`
- **Deprecation:** `cml.utils.deprecation.deprecated(alternative, removal_version)` decorator
- **Thread safety:** Sync client uses `threading.RLock` in `set_tenant()`
- **Async loop check:** Async client stores creation event loop and raises RuntimeError if used in a different loop
- Unit tests for Phase 6 (exceptions, logging, read_safe, __str__, serialization, session, deprecation, thread safety, event loop)

#### Phase 8: Documentation and publishing
- **README:** PyPI and Python version badges, one-line tagline, Documentation links to getting-started, api-reference, configuration, examples
- **Docs:** getting-started.md, api-reference.md, configuration.md, examples.md in packages/py-cml/docs/
- **Examples:** quickstart.py, chat_with_memory.py, async_example.py, embedded_mode.py, agent_integration.py in examples/
- **GitHub:** Issue templates (bug_report, feature_request), pull_request_template with Summary, Changes, Testing, Documentation checklists
- **SECURITY.md:** Supported versions, report via GitHub Security Advisories
- **CONTRIBUTING:** Releasing py-cml section (version bump, tag py-cml-v*, PyPI publish); fork/venv in dev setup; PR checklist alignment
- Publish workflow (py-cml-publish.yml on py-cml-v* tag) already in place; no code changes

## [1.0.8] - 2025-02-11

### Changed

- Version bump for PyPI release.
