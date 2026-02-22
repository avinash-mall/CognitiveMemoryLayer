# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.3.3] - 2026-02-22

### Documentation

- **Documentation consolidation** — README restructured to focus on architecture, research, basic usage, and eval highlights; consolidated and deduplicated documentation across the repo.

### Added

- **Dashboard Knowledge Graph: neovis.js (offline)** — Replaced vis-network CDN with [neovis.js](https://github.com/neo4j-contrib/neovis.js) for the Knowledge Graph page. The graph connects directly from the browser to Neo4j (bolt/WebSocket) for visualization. Requires `npm run build` in `src/dashboard/` for local dev; Docker builds the bundle at image build time. New endpoint `GET /api/v1/dashboard/graph/neo4j-config` returns Neo4j connection config for the browser (admin-only). New env var `DATABASE__NEO4J_BROWSER_URL` for when Neo4j is not reachable at `DATABASE__NEO4J_URL` from the browser (e.g. Docker: set to `bolt://localhost:7687`). Cypher updated for Neo4j 5 (`COUNT {}` instead of deprecated `size()`). See `.env.example`, `ProjectPlan/UsageDocumentation.md`.

- **Neo4j graph: Unified Extractor prompt engineering and graph sync** — Fixes incorrect relation names, wrong entity text (e.g. system prompts), and entity_type issues in the knowledge graph. Unified Extractor prompts redesigned with schema-first, few-shot, and explicit exclusion rules (cross-model techniques: Reintech, PromptPort, PromptNER). Entities now use typed objects `{text, normalized, type}` with allowed types (PERSON, LOCATION, ORGANIZATION, etc.); relations use `{subject, predicate, object}` with snake_case predicates. Exclusion rule instructs the LLM to omit system prompts, role instructions, and non-conversational content. Relation schema bug fixed (`subject`/`predicate`/`object` instead of `source`/`target`/`type`). Hippocampal store uses unified entities and relations for graph sync when the unified path is enabled. Fallbacks removed (no regex JSON recovery, no batch-to-per-chunk fallback). New: `tests/unit/test_unified_write_extractor.py` (12 tests), `scripts/verify_neo4j_graph.py` (diagnostic script for Neo4j). See plan: `neo4j_graph_troubleshooting_b3afb682.plan.md`.

### Changed

- **LLM feature flags gating** — When `FEATURES__USE_LLM_*` flags are on (default), only the LLM path updates salience, importance, constraints, PII, and facts; rule-based logic for those fields is skipped. WriteGate accepts optional `unified_result` and uses `unified_result.importance`/`unified_result.salience`/`unified_result.pii_spans` when the respective flags are on. Orchestrator constraint supersession gates on `use_llm_constraint_extractor`. `encode_chunk` runs unified extraction before the gate when the LLM path is enabled. `encode_batch` Phase 1 passes `unified_result` to the gate; Phase 1 regex redaction skipped when PII comes from LLM. See UsageDocumentation § Write Path LLM Gating, BaseCMLStatus.

- **Chunker replaced with semchunk** — Replaced `SemanticChunker`, `RuleBasedChunker`, and `ChonkieChunkerAdapter` with a single `SemchunkChunker` using [semchunk](https://github.com/isaacus-dev/semchunk) and a Hugging Face tokenizer. New config: `CHUNKER__TOKENIZER` (default: google/flan-t5-base), `CHUNKER__CHUNK_SIZE` (default: 500), `CHUNKER__OVERLAP_PERCENT` (default: 0.15). Removed `chonkie[semantic]`; added `semchunk` and `transformers`. Removed feature flags: `use_fast_chunker`, `use_chonkie_for_large_text`, `chunker_large_text_threshold_chars`, `chunker_require_llm`. See `.env.example`, README, evaluation/README.

### Added

- **Rule-based extractor LLM replacement** — Implements [RuleBasedExtractorsAndLLMReplacement.md](ProjectPlan/BaseCML/RuleBasedExtractorsAndLLMReplacement.md): 8 LLM-based replacements for rule-based extractors with unified write-path extraction (one LLM call per chunk returns constraints, facts, salience, importance, PII spans). New feature flags (default true): `FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR`, `FEATURES__USE_LLM_WRITE_TIME_FACTS`, `FEATURES__CHUNKER_REQUIRE_LLM`, `FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY`, `FEATURES__USE_LLM_SALIENCE_REFINEMENT`, `FEATURES__USE_LLM_PII_REDACTION`, `FEATURES__USE_LLM_WRITE_GATE_IMPORTANCE`, `FEATURES__USE_LLM_CONFLICT_DETECTION_ONLY`. Uses `LLM_INTERNAL__*` when set, else `LLM__*`. New tests: `test_llm_constraint_extractor`, `test_llm_write_time_facts`, `test_chunker_require_llm`, `test_llm_query_classifier_only`, `test_llm_salience_refinement`, `test_llm_pii_redaction`, `test_llm_write_gate_importance`, `test_llm_conflict_detection_only`, and `test_unified_write_path` integration. See `.env.example`, UsageDocumentation § Configuration Reference, BaseCMLStatus.

- **BaseCML constraint-first write/read and supersession (ImplementationPlan)** — Implements constraint-first semantics and fixes semantic disconnect, constraint dilution, and stale constraints:
  - **Constraint write path**: Episodic CONSTRAINT records now use `ConstraintExtractor.constraint_fact_key(constraint)` as their `key` (in `encode_chunk` and `encode_batch`) so supersession can match by key. `PostgresMemoryStore.deactivate_constraints_by_key(tenant_id, constraint_key)` sets `status=SILENT` for existing episodic CONSTRAINT records with that key. Orchestrator deactivates by key *before* `encode_batch` for all extracted fact keys so the new constraint is written as ACTIVE and previous ones become SILENT.
  - **Constraint read path**: `RetrievalStep` has optional `constraint_categories: list[str] | None`; the CONSTRAINTS step is built with `constraint_categories=analysis.constraint_dimensions`. `HybridRetriever._retrieve_constraints()` uses `step.constraint_categories` to filter semantic-fact lookup to those categories when present; otherwise uses all cognitive categories (GOAL, VALUE, STATE, CAUSAL, POLICY).
  - **Context assembly**: In `MemoryPacketBuilder._format_markdown`, "Recent Context" includes only episodes with `relevance_score > EPISODE_RELEVANCE_THRESHOLD` (0.4) to avoid diluting constraints. Constraint lines are prefixed with `[!IMPORTANT]` in the markdown.
  - **Consolidation**: After gist extraction, the worker re-runs `ConstraintExtractor` on each gist text and persists extracted constraints as semantic facts via `migrator.semantic.store_fact()` with the same key format, so constraints are not lost in summarization.
  - **Config**: `RerankerSettings` (recency_weight, relevance_weight, confidence_weight, active_constraint_bonus) added under `RetrievalSettings`; `MemoryRetriever` builds `RerankerConfig` from `get_settings().retrieval.reranker`.
  - **Verification**: New unit tests — retrieval plan for "Can I afford dinner?" includes CONSTRAINTS step with `constraint_categories`; packet markdown filters low-relevance episodes and shows `[!IMPORTANT]` for constraints; `search()` calls `increment_access_counts` once with result IDs. Integration test `test_constraint_supersession_first_silent_second_active` asserts first constraint SILENT and second ACTIVE after deactivate + second write. Optional script `scripts/verify_constraint_flow.py` ingests constraints and prints packet markdown for manual check of "Active Constraints (Must Follow)".

- **Dashboard reconsolidate and eval Phase A–B pipeline** — New `POST /api/v1/dashboard/reconsolidate` endpoint to release all labile state for a tenant (no belief revision), with job tracking in `dashboard_jobs`. `LabileStateTracker.release_all_for_tenant(tenant_id)` added for Redis and in-memory backends. Dashboard Management page includes a Reconsolidation panel (tenant selector, "Release labile" button, result and job history). Evaluation script `eval_locomo_plus.py` runs consolidation then reconsolidation for each eval tenant between Phase A (ingestion) and Phase B (QA), unless `--skip-consolidation` is set; the API key must have dashboard/admin permission for this step. Documentation updated in evaluation/README.md, README.md, ProjectPlan/UsageDocumentation.md, packages/py-cml/README.md, and `dashboard_jobs.job_type` comment (values: consolidate, forget, reconsolidate).

- **Cognitive Constraint Layer (LoCoMo-Plus Level-2 optimisation)** — Full implementation of latent constraint extraction, storage, and retrieval based on the constraint-layer deep-research analysis. Addresses issues ISS-01 through ISS-09 across 5 phases:
  - **Phase 1 – Evaluation harness correctness (ISS-04, ISS-05)**: `eval_locomo_plus.py` now parses LoCoMo `DATE:` lines into UTC timestamps and passes them to the CML write API (`timestamp` field). Metadata enriched with `speaker`, `date_str`, `session_idx`. The `COGNITIVE_PROMPT` replaced with a neutral continuation prompt (no memory-aware task disclosure).
  - **Phase 2 – Constraint extraction & storage (ISS-01, ISS-06, ISS-03)**: New `ChunkType.CONSTRAINT` in the working-memory model. `RuleBasedChunker` and `SemanticChunker` detect constraint cue phrases (goals, values, policies, causal reasoning) with salience 0.85+. New `ConstraintExtractor` (`src/extraction/constraint_extractor.py`) with `ConstraintObject` schema and rule-based pattern groups (goal/value/state/causal/policy). Write gate maps `ChunkType.CONSTRAINT` → `MemoryType.CONSTRAINT` with importance boost. `HippocampalStore` runs constraint extraction on every chunk, attaches structured constraints to `metadata["constraints"]`, and overrides memory type when high-confidence. Orchestrator stores constraints as semantic facts gated by `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` (default: true). New cognitive `FactCategory` values (`GOAL`, `STATE`, `VALUE`, `CAUSAL`, `POLICY`) and matching `DEFAULT_FACT_SCHEMAS`.
  - **Phase 3 – Constraint-aware retrieval routing (ISS-02, ISS-07)**: `QueryAnalysis` extended with `constraint_dimensions` and `is_decision_query`. Classifier detects `CONSTRAINT_CHECK` intent via fast patterns ("should I", "can I", "is it ok", "recommend", etc.) and enriches constraint dimensions. New `RetrievalSource.CONSTRAINTS` with highest-priority retrieval step. `HybridRetriever._retrieve_constraints()` uses two-pronged approach: vector search filtered to `MemoryType.CONSTRAINT` + semantic fact lookup across cognitive categories. Packet builder shows up to 6 constraints with provenance in "Active Constraints (Must Follow)" section. Reranker uses type-dependent recency weights (stable=0.05, semi-stable=0.15, volatile=0.2) and caps pairwise diversity at min(n, 50).
  - **Phase 4 – Supersession & consolidation fixes (ISS-08, ISS-03)**: Consolidation sampler includes `MemoryType.CONSTRAINT` with 90-day window (vs 7-day for episodes). Summarizer prompt updated to output cognitive gist types. Schema aligner maps cognitive types to `FactCategory` and generates integration keys. `ConstraintExtractor.detect_supersession()` for same-type+scope constraint replacement.
  - **Phase 5 – Observability & silent-failure fixes (ISS-09)**: Silent `except: pass` blocks in orchestrator replaced with structured `logger.warning()` calls with context. Eval script `--verbose` flag emits per-sample retrieval type counts. `_cml_read` supports `return_full` for diagnostics.
  - **API**: `ReadMemoryResponse` includes `constraints` field (list of `MemoryItem`). Read endpoint populates constraints from `packet.constraints`.
  - **Package exports**: `src/extraction/__init__.py` exports `ConstraintExtractor` and `ConstraintObject`. `WriteTimeFactExtractor` now also processes `ChunkType.CONSTRAINT` chunks.
  - **Config**: New feature flag `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` (default true).

- **Deep-research implementation (BaseCML plan, all 6 phases)** — Performance, correctness, and observability improvements:
  - **Phase 1 – Correctness**: Stable fact keys (SHA256) in consolidation migrator; content-based hippocampal keys so different facts don’t overwrite; write-time fact extraction (`WriteTimeFactExtractor`) for preference/identity/location/occupation when `FEATURES__WRITE_TIME_FACTS_ENABLED` is true.
  - **Phase 2 – Hot path**: Hippocampal `encode_batch()` refactored to gate+redact → single `embed_batch()` → bounded concurrency upsert; optional `AsyncStoragePipeline` (Redis queue) for turn writes; `CachedEmbeddings` wrapper when Redis + `FEATURES__CACHED_EMBEDDINGS_ENABLED`; feature flags `batch_embeddings_enabled`, `store_async`, `cached_embeddings_enabled`.
  - **Phase 3 – Retrieval**: Per-step and total retrieval timeouts (`RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS`, `RETRIEVAL__TOTAL_TIMEOUT_MS`); `QueryAnalysis.user_timezone` and timezone-aware "today"/"yesterday" in planner; cross-group `skip_if_found`; step duration/result metrics.
  - **Phase 4 – Background**: `bulk_dependency_counts()` in Postgres for forgetting (one query vs O(n²)); neocortical `multi_hop_query()` uses `asyncio.gather` for entity lookups.
  - **Phase 5 – Stability**: Sensory buffer stores token IDs (tiktoken), batch decode in `get_text()`; `BoundedStateMap` (LRU+TTL) for working/sensory in-memory state; feature flags `bounded_state_enabled`, `db_dependency_counts`.
  - **Phase 6 – Observability**: Optional HNSW ef_search tuning per query; metrics `RETRIEVAL_STEP_DURATION`, `RETRIEVAL_STEP_RESULT_COUNT`, `RETRIEVAL_TIMEOUT_COUNT`, `FACT_HIT_RATE`.
  - **Config**: `FeatureFlags` and `RetrievalSettings` in `src/core/config.py`; all toggles and retrieval timeouts documented in UsageDocumentation and `.env.example`. New unit tests in `tests/unit/test_deep_research_improvements.py` (38 tests); total server tests 301.

- **user_timezone on read and turn API** — `ReadMemoryRequest` and `ProcessTurnRequest` now accept optional `user_timezone` (IANA string, e.g. `America/New_York`). The retrieval planner uses it for timezone-aware "today"/"yesterday" filters. The Python SDK (`packages/py-cml`) supports `user_timezone` on `read()`, `read_safe()`, and `turn()` (sync, async, and embedded). Embedded `read()` now passes `memory_types`, `since`, and `until` to the orchestrator (previously accepted but not passed). See ProjectPlan/UsageDocumentation.md and packages/py-cml CHANGELOG.

- **Optional LLM/embedding tests** — Integration tests that depend on a real LLM or embedding model now skip (instead of fail) when the service is unavailable.

- **Optional LLM_INTERNAL for internal tasks** — New optional `LLM_INTERNAL__*` env vars for a dedicated model for chunking, entity/relation extraction, consolidation, and forgetting. When set, these tasks use the internal model instead of the primary `LLM__*` model; otherwise they fall back to the primary LLM. Helps accelerate bulk ingestion (e.g. evaluation) by using a smaller/faster model for extraction. See `.env.example`, `ProjectPlan/UsageDocumentation.md`, and `packages/py-cml/docs/configuration.md`.

- **Batch entity and relation extraction** — `RelationExtractor.extract_batch()` processes multiple (chunk, entities) pairs in one LLM call. Hippocampal `encode_batch()` uses batch extraction for Phase 3/4 entity and relation extraction instead of per-chunk calls, reducing latency for bulk writes.

- **Evaluation concurrent ingestion** — `eval_locomo_plus.py` Phase A ingestion supports `--ingestion-workers N` (default 10). When N > 1, samples are ingested concurrently via `ThreadPoolExecutor`; when N == 1, behaviour matches the previous sequential flow with inter-sample delay. See `evaluation/README.md` and `ProjectPlan/LocomoEval/RunEvaluation.md`. Server: `tests/integration/test_forgetting_llm_compression.py` — `llm_client` fixture skips when no LLM is configured; `_ensure_llm_reachable()` skips on any exception (e.g. rate limit 429, connection error); both tests are marked `@pytest.mark.requires_llm`. py-cml: embedded lite-mode write/read tests in `packages/py-cml/tests/embedded/test_lite_mode.py` skip on `ImportError`/`OSError`/`RuntimeError` or when the exception message indicates model/embed/rate-limit issues. See `tests/README.md` § Optional LLM/embedding tests.

### Changed

- **Evaluation: Locomo-Plus only** — Removed legacy `evaluation/locomo/` (cloned LoCoMo repo), `eval_locomo.py`, and `compare_results.py`. All evaluation now uses `eval_locomo_plus.py` with unified LoCoMo + Locomo-Plus (LLM-as-judge). `locomo10.json` moved to `evaluation/locomo_plus/data/`. `run_full_eval.py` runs eval_locomo_plus and prints the performance table. See evaluation/README.md and ProjectPlan/LocomoEval/RunEvaluation.md.
- **Tests and config from .env** — All test and Docker configuration (including `EMBEDDING__DIMENSIONS`) is read from the project root `.env`. Removed the `EMBEDDING__DIMENSIONS` override from `docker/docker-compose.yml` for the `app` service so both `app` and `api` use `.env` and migrations/tests share the same vector dimension. Server test fixtures (`mock_embeddings` in `tests/conftest.py`, `test_retrieval_flow.py`) use `get_settings().embedding.dimensions` instead of hardcoded values.
- **Pytest** — Marker `requires_llm` added in `pyproject.toml` under `[tool.pytest.ini_options]` for integration tests that need a reachable LLM; run with `pytest -m requires_llm` to execute only those tests when the LLM is available.

### Documentation

- **Cognitive Constraint Layer docs** — README: new "Cognitive Constraint Layer (Level-2 Memory)" section documenting constraint extraction pipeline, retrieval routing, consolidation, supersession, and API fields. Updated neuroscience mapping tables (Hippocampal Store: constraint extraction row; Neocortical Store: cognitive fact categories; Retrieval: constraint retrieval). Feature flags table: added `FEATURES__CONSTRAINT_EXTRACTION_ENABLED`. Project structure: extraction folder description updated. UsageDocumentation: `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` in Configuration Reference; `constraints` field in read response formats; `constraint` memory type expanded with cognitive extraction note. `.env.example`: new `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` flag. evaluation/README: new "Level-2 Cognitive Memory" section; `--verbose` flag documented. py-cml CHANGELOG: `constraints` field on `ReadResponse`. py-cml README: constraint mention in server compatibility. py-cml api-reference: `ReadResponse` model updated with `constraints` field and `CONSTRAINT` memory type; models section expanded.
- **Deep-research implementation docs** — README: "Performance & reliability" section (feature flags table), updated neuroscience mapping and project structure, test count 529. UsageDocumentation: Configuration Reference extended with Feature Flags (`FEATURES__*`) and Retrieval settings (`RETRIEVAL__*`), plus retrieval Prometheus metrics. UsageDocumentation: POST /memory/read and POST /memory/turn request bodies now document optional `user_timezone`. README: optional `user_timezone` noted for retrieve and seamless turn with link to UsageDocumentation. `.env.example`: optional `FEATURES__*` and `RETRIEVAL__*` with comments. tests/README: total 301 tests, note on `test_deep_research_improvements.py`. ImplementationPlan: implementation status callout (all 6 phases complete) and links to config docs.
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
  - **Knowledge Graph page** — Interactive neovis.js visualization of entities and relations from Neo4j. Connects directly from the browser to Neo4j. Search entities by name, explore neighborhoods with configurable depth (1-5 hops), view node/edge details. Bundled for offline use.
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

- **Read API `session_read` constraints** — The `/session/{session_id}/read` endpoint now correctly maps and includes constraint memories in the `ReadMemoryResponse` object (previously they were silently dropped from the response).
- **Silent database errors in `encode_batch`** — The phase 4 `upsert` loop in `HippocampalStore.encode_batch` now catches and logs (`logger.error`) underlying database exceptions rather than silently swallowing them and losing data.
- **Unsafe async initialization** — `DatabaseManager.__init__` exception cleanup fallback now avoids calling `asyncio.run()` in a synchronous constructor when no event loop is running, logging the error safely instead of crashing.
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
