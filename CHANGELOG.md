# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Optional LLM/embedding tests** — Integration tests that depend on a real LLM or embedding model now skip (instead of fail) when the service is unavailable. Server: `tests/integration/test_phase8_llm_compression.py` — `llm_client` fixture skips when no LLM is configured; `_ensure_llm_reachable()` skips on any exception (e.g. rate limit 429, connection error); both tests are marked `@pytest.mark.requires_llm`. py-cml: embedded lite-mode write/read tests in `packages/py-cml/tests/embedded/test_lite_mode.py` skip on `ImportError`/`OSError`/`RuntimeError` or when the exception message indicates model/embed/rate-limit issues. See `tests/README.md` § Optional LLM/embedding tests.

### Changed

- **Tests and config from .env** — All test and Docker configuration (including `EMBEDDING__DIMENSIONS`) is read from the project root `.env`. Removed the `EMBEDDING__DIMENSIONS` override from `docker/docker-compose.yml` for the `app` service so both `app` and `api` use `.env` and migrations/tests share the same vector dimension. Server test fixtures (`mock_embeddings` in `tests/conftest.py`, `test_phase5_retrieval_flow.py`) use `get_settings().embedding.dimensions` instead of hardcoded values.
- **Pytest** — Marker `requires_llm` added in `pyproject.toml` under `[tool.pytest.ini_options]` for integration tests that need a reachable LLM; run with `pytest -m requires_llm` to execute only those tests when the LLM is available.

### Documentation

- **tests/README.md** — Expanded Configuration with “Environment and .env” (Auth, Database, Embedding vars), embedding dimensions and Docker note, py-cml tests (host), and “Optional LLM/embedding tests (skip when unavailable)”.
- **.env.example** — Clarified `EMBEDDING__DIMENSIONS` (server, migrations, and tests read from .env; Docker does not override). Added `EMBEDDING__DIMENSIONS` to the embedded section. Updated Part F (development & testing) to state that server and py-cml tests read all variables from this file.
- **README.md** — Run Tests section now states that tests read config (including `EMBEDDING__DIMENSIONS`) from `.env` and points to `tests/README.md`.
- **ProjectPlan/UsageDocumentation.md** — `EMBEDDING__DIMENSIONS` description updated to note that server and tests read from `.env` and Docker app/api use `.env` without override.
- **evaluation/README.md** — Noted that server and tests read `EMBEDDING__DIMENSIONS` from `.env` and Docker `app`/`api` do not override it.
- **packages/py-cml/CONTRIBUTING.md** — Integration/E2E section: tests load config from repo root `.env`, copy `.env.example` to `.env`, and optional LLM/embedding tests skip when the service is unavailable.

### Added (dashboard and features)

- **Dashboard expansion** — Major enhancement of the admin dashboard with 6 new pages and multiple new features:
  - **Tenants page** — Lists all tenants with memory/fact/event counts, active memory counts, last activity timestamps, and quick-link buttons to filter Overview/Memories/Events by tenant.
  - **Sessions page** — Shows active sessions from Redis (with TTL badges and metadata) and memory counts per `source_session_id` from the database. Click a session to filter Memory Explorer.
  - **Knowledge Graph page** — Interactive vis-network visualization of entities and relations from Neo4j. Search entities by name, explore neighborhoods with configurable depth (1-5 hops), view node/edge details.
  - **API Usage page** — Current rate-limit buckets with utilization bars, hourly request volume chart (Chart.js). KPI cards for active keys, avg utilization, configured RPM, and 24h request count.
  - **Configuration page** — Read-only config snapshot showing all settings grouped by section (Application, Database, Embedding, LLM, Auth) with secrets masked. Editable settings can be changed inline at runtime (stored in Redis).
  - **Retrieval Test page** — Interactive query tool for debugging memory retrieval. Input tenant + query with optional filters; returns scored memories with relevance bars, type badges, and metadata.
  - **Enhanced Overview** — Reconsolidation queue status KPI and 24h request sparkline chart.
  - **Enhanced Management** — Reconsolidation/labile status per tenant; job history table persisted in PostgreSQL (`dashboard_jobs` table via Alembic migration `002`).
  - **Enhanced Memory Explorer** — Bulk actions (archive/silence/delete selected memories via checkboxes), select-all, JSON export button.
  - **New backend endpoints** — `/sessions`, `/ratelimits`, `/request-stats`, `/graph/stats`, `/graph/explore`, `/graph/search`, `/config` (GET + PUT), `/labile`, `/retrieval`, `/jobs`, `/memories/bulk-action`, `/export/memories`.
  - **Request counting middleware** — `RequestLoggingMiddleware` now increments hourly counters in Redis for the API Usage page.
  - **Alembic migration** — `002_dashboard_jobs.py` adds the `dashboard_jobs` table for consolidation/forgetting job history.

- **X-Eval-Mode header on write** — Optional request header `X-Eval-Mode: true` on `POST /memory/write` and `POST /session/{session_id}/write`. When set, the response includes `eval_outcome` (`"stored"` or `"skipped"`) and `eval_reason` (write-gate reason) so evaluation scripts can aggregate gating statistics. See Usage documentation and `ProjectPlan/LocomoEval/RunEvaluation.md`.
- **LoCoMo evaluation: gating and timing** — `evaluation/scripts/eval_locomo.py` supports `--eval-mode` (sends X-Eval-Mode on writes, writes `locomo10_gating_stats.json`) and `--log-timing` (records per-question CML read and Ollama latency and token usage, writes `locomo10_qa_cml_timing.json`). See evaluation README and RunEvaluation.md §4.2.
- **DELETE /memory/all** — New admin-only endpoint; SDK `delete_all(confirm=True)` is now supported by the server.
- **Startup validation** — `validate_embedding_dimensions()` is called in the app lifespan to catch embedding-dimension vs DB schema mismatch at startup.

- **Temporal fidelity**
  - Optional `timestamp` field in `WriteMemoryRequest` and `ProcessTurnRequest` allows specifying event time for memories
  - Enables historical replay for benchmarks (e.g., Locomo evaluation with session dates)
  - Defaults to current time if not provided (backward compatible)
  - Threads through: API → Orchestrator → Short-term → Working → Chunker → Hippocampal → Storage

- **Documentation**
  - `CONTRIBUTING.md` — Guidelines for contributors and development setup.
  - `SECURITY.md` — Security policy and vulnerability reporting.
  - `.env.example` — Example environment configuration for database, API, LLM, and embedding settings.

- **Public API**
  - Meaningful exports in package `__init__.py` files:
    - `src.core`: `get_settings`, `Settings`, `MemoryContext`, `MemoryType`, `MemoryStatus`, `MemorySource`, `MemoryRecord`, `MemoryRecordCreate`, `MemoryPacket`.
    - `src.api`: `create_app`, `AuthContext`, `get_auth_context`, `require_write_permission`, `require_admin_permission`.
    - `src.storage`: `DatabaseManager`, `Base`, `EventLogModel`, `MemoryRecordModel`, `SemanticFactModel`, `PostgresMemoryStore`.
    - `src.utils`: `EmbeddingClient`, `OpenAIEmbeddings`, `LLMClient`, `get_llm_client`.
    - `src.memory`: `MemoryOrchestrator`, `ShortTermMemory`, `HippocampalStore`, `NeocorticalStore`.
    - `src.retrieval`: `MemoryRetriever`.
    - `src.extraction`: `EntityExtractor`, `FactExtractor`, `LLMFactExtractor`.
    - `src.consolidation`: `ConsolidationReport`, `ConsolidationWorker`.
    - `src.forgetting`: `ForgettingReport`, `ForgettingWorker`, `ForgettingScheduler`.
    - `src.reconsolidation`: `LabileMemory`, `LabileSession`, `LabileStateTracker`, `ReconsolidationResult`, `ReconsolidationService`.

- **Sensory buffer**
  - `SensoryBuffer._tokenize()` now uses tiktoken (cl100k_base) when available for accurate token counting, with a whitespace-split fallback when tiktoken is not installed.

### Changed

- **CORS** — Default origins no longer use a placeholder. Tiered behavior: use `cors_origins` if set; in debug mode use `["*"]`; otherwise use `["http://localhost:3000", "http://localhost:8080"]`.
- **DatabaseManager.close()** — Safe shutdown: only disposes/closes connections that are not `None`, avoiding `AttributeError` on partial initialization.
- **Orchestrator read API** — Read now supports `memory_types`, `since`, and `until`; these are forwarded from the API through the orchestrator into the retriever and applied to retrieval steps.
- **Packaging** — Root package name renamed from `cognitive-memory-layer` to `cml-server` to avoid pip collision with the SDK package.
- **WriteDecision** — `STORE_SYNC` and `STORE_ASYNC` are collapsed into a single `STORE` (aliases kept for compatibility); async write path is not implemented.
- **RetrievalSource** — Removed unused `LEXICAL` enum value (no lexical retrieval implementation).
- **Read format "list"** — When `format` is `"list"`, the response now returns a flat `memories` list with empty `facts`, `preferences`, and `episodes` (differentiated from `"packet"`).

### Fixed

- **Retrieval fact typing** — Semantic-fact text search results from the neocortical store are now correctly typed as `semantic_fact` (and appear under `packet.facts`) instead of falling back to episodic.
- **API contract: write** — Request `metadata` is now merged into memory records (user keys override system defaults). Optional `memory_type` is respected as an override for the write gate classification.
- **API contract: read** — Request fields `memory_types`, `since`, and `until` are now applied: they are forwarded from the API through the orchestrator into the retriever and used to filter vector/search steps.
- **Rate limiting** — Rate-limit key is now derived from the `X-API-Key` header (hashed) instead of the spoofable `X-Tenant-Id`; fallback to client IP when no API key is present.
- **CORS** — When `origins` includes `"*"`, `allow_credentials` is set to `False` to comply with the CORS spec.
- **Session scoping** — `get_session_context(session_id)` now filters by `source_session_id`, returning only memories for that session when `session_id` is provided.
- **Stats by type** — `PostgresMemoryStore.count()` now supports `type` (and `since`/`until`) filters so orchestrator `get_stats()` returns correct `by_type` counts.
- **Dashboard routes** — Removed duplicate `/dashboard` route; the catch-all `/dashboard/{rest_of_path:path}` serves the root.
- Empty `src/api/dependencies.py` — Implemented with re-exports of auth dependencies.
- Redundant empty `src/memory/hippocampal/encoder.py` — Removed; encoding is handled by `HippocampalStore` in `store.py`.

---

*For earlier history and design docs, see the [ProjectPlan/](ProjectPlan/) directory (e.g. [ActiveCML/Issues.md](ProjectPlan/ActiveCML/Issues.md), [CreatePackage/CreatePackageStatus.md](ProjectPlan/CreatePackage/CreatePackageStatus.md)).*
