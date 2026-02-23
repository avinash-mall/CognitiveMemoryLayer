# Running Tests

This document describes how to run the test suite for the Cognitive Memory Layer project.

## Prerequisites

- **Python 3.11+**
- Project dependencies installed (e.g. `pip install -e ".[dev]"` from repo root, or `pip install -e ".[dev,embedded]"` for embedded tests)

## Quick start

From the project root:

```bash
# Run all server tests (unit + integration + e2e)
pytest tests/unit tests/integration tests/e2e -v

# Short tracebacks
pytest tests/unit tests/integration tests/e2e -v --tb=short

# Run server + SDK tests (704 total)
pytest tests/unit tests/integration tests/e2e packages/py-cml/tests -v
```

## Test layout

| Directory           | Purpose |
|---------------------|--------|
| `tests/unit/`       | Unit tests (mocked dependencies; no DB required for most). See below for file names by domain. |
| `tests/integration/`| Integration tests (real app, Postgres; optional Neo4j/Redis). |
| `tests/e2e/`        | End-to-end API flow tests. |

**Server total: 529 tests** (unit + integration + e2e). With `packages/py-cml/tests` (175 tests), **combined total: 704**. Run `python scripts/update_readme_badges.py` from the repo root to refresh version and test-count badges in the READMEs.

### Unit test files (by domain)

| File | Domain |
|------|--------|
| `test_core_enums_schemas_config.py` | Core enums, schemas, config |
| `test_sensory_buffer_working_memory.py` | Sensory buffer, working memory, chunker |
| `test_hippocampal_write_gate_redactor.py` | Write gate, PII redactor |
| `test_embeddings_mock_client.py` | Mock embedding client |
| `test_neocortical_schemas.py` | Neocortical fact schemas, SchemaManager |
| `test_retrieval_classifier_planner_reranker.py` | Retrieval classifier, planner, reranker, packet builder |
| `test_reconsolidation_labile_conflict_belief.py` | Labile tracker, conflict detector, belief revision |
| `test_consolidation_triggers_clusterer_sampler.py` | Consolidation scheduler, clusterer, sampler |
| `test_forgetting_scorer_policy_interference.py` | Forgetting scorer, policy, interference |
| `test_celery_forgetting_task.py` | Celery forgetting task and beat schedule |
| `test_api_auth_schemas.py` | API auth config and request/response schemas |
| `test_api_routes.py` | Health and memory API routes |
| `test_api_request_logging_rate_limit.py` | Request logging and rate-limit middleware |
| `test_dashboard_routes.py` | Dashboard API routes (mocked DB) |
| `test_extraction_entity_relation_fact.py` | Entity, relation, and fact extraction |
| `test_unified_write_extractor.py` | Unified extractor: typed entities, relations, exclusion rules, graph sync |
| `test_orchestrator_seamless_provider.py` | Memory orchestrator and seamless provider |
| `test_memory_modules_conversation_scratch_tool.py` | ConversationMemory, ScratchPad, ToolMemory, KnowledgeBase |
| `test_chunker_chonkie_adapter.py` | Chonkie chunker adapter |
| `test_constraint_layer.py` | Constraint layer (extraction, retrieval, consolidation) |
| `test_deep_research_improvements.py` | Stable keys, batch embeddings, retrieval timeouts, write-time facts, BoundedStateMap |
| `test_consolidation_migrator_sampler.py` | Consolidation migrator and episode sampler |
| `test_storage_*.py`, `test_routes_helpers.py`, `test_forgetting_executor.py`, `test_timestamp_feature.py`, `test_utils_timing_metrics.py`, `test_core_exceptions.py` | Storage, routes, forgetting executor, utils, core |

### Integration test files (by domain)

| File | Domain |
|------|--------|
| `test_storage_event_log_repository.py` | Event log repository (Postgres) |
| `test_short_term_memory_flow.py` | Short-term memory ingest flow |
| `test_hippocampal_encode_flow.py` | Hippocampal encode and retrieval |
| `test_neocortical_store_flow.py` | Neocortical store operations |
| `test_fact_store_integration.py` | Fact store upsert, get, search |
| `test_retrieval_flow.py` | Full retrieval flow |
| `test_reconsolidation_flow.py` | Reconsolidation and correction flow |
| `test_consolidation_flow.py` | Consolidation flow |
| `test_forgetting_flow.py` | Forgetting decay flow |
| `test_forgetting_llm_compression.py` | LLM-based compression (optional, `@pytest.mark.requires_llm`) |
| `test_api_flow.py` | API health and auth flow |
| `test_dashboard_flow.py` | Dashboard API with real DB |
| `test_memory_type_storage_retrieval.py` | Memory types round-trip |

Key test areas: constraint layer, deep research improvements (stable keys, batch embeddings, write-time facts), **batch extraction** (`RelationExtractor.extract_batch` in `test_extraction_entity_relation_fact.py`; `UnifiedWritePathExtractor` in `test_unified_write_extractor.py`; orchestrator uses `get_internal_llm_client` in `test_orchestrator_seamless_provider.py`), evaluation (timestamp parsing, verbose diagnostics). Use `python scripts/verify_neo4j_graph.py` to run Neo4j diagnostics (relationship types, entity types, long/suspicious entities).

## Running subsets

```bash
# Unit tests only (fast; no services required for dashboard/API route tests)
pytest tests/unit -v

# Integration tests only (requires Postgres; optional Neo4j/Redis/testcontainers)
pytest tests/integration -v

# E2E tests only
pytest tests/e2e -v

# Dashboard tests only (unit + integration)
pytest tests/unit/test_dashboard_routes.py tests/integration/test_dashboard_flow.py -v

# By keyword (e.g. "dashboard")
pytest -v -k dashboard
```

## Integration tests and services

Integration tests use the real FastAPI app and expect a database (and optionally Neo4j/Redis):

1. **Use env DB (recommended for local/CI)**  
   Set `USE_ENV_DB=1` and provide URLs, e.g.:

   ```bash
   export USE_ENV_DB=1
   export DATABASE__POSTGRES_URL=postgresql+asyncpg://user:pass@localhost:5432/memory
   # Optional:
   export DATABASE__NEO4J_URL=bolt://localhost:7687
   export DATABASE__REDIS_URL=redis://localhost:6379
   alembic upgrade head
   pytest tests/integration -v
   ```

2. **Docker Compose**  
   Start Postgres (and other services) with `docker compose -f docker/docker-compose.yml up -d`, then set the same env vars and run migrations before `pytest tests/integration`.

3. **Testcontainers**  
   If `testcontainers` is installed and Docker is available, integration tests can start Postgres/Neo4j containers automatically when `USE_ENV_DB` and `DATABASE__POSTGRES_URL` are not set.

## Coverage

```bash
# Coverage report in terminal
pytest tests --cov=src --cov-report=term-missing

# Coverage report as HTML (open htmlcov/index.html)
pytest tests --cov=src --cov-report=html

# XML (for CI/Codecov)
pytest tests --cov=src --cov-report=xml
```

## Dashboard tests

- **Unit** (`tests/unit/test_dashboard_routes.py`): Auth (401 without key, 403 without admin key) and response shape with a **mocked** DB. No Postgres/Redis/Neo4j needed.
- **Integration** (`tests/integration/test_dashboard_flow.py`): Full dashboard API with admin key and **real** app state (DB from app lifespan). Requires Postgres (and env or testcontainers as above).

```bash
# Dashboard unit only (no services)
pytest tests/unit/test_dashboard_routes.py -v

# Dashboard integration (with DB)
USE_ENV_DB=1 DATABASE__POSTGRES_URL=... pytest tests/integration/test_dashboard_flow.py -v
```

## Configuration

- **Pytest config:** `pyproject.toml` under `[tool.pytest.ini_options]` (e.g. `testpaths = ["tests"]`, `asyncio_mode = "auto"`).

### Environment and `.env`

Tests read **all** configuration from the project root **`.env`** (loaded via `conftest.py`). There are no hardcoded fallbacks for URLs, API keys, or embedding dimensions.

| Purpose | Variables (set in `.env`) |
|--------|---------------------------|
| **Auth** | `AUTH__API_KEY` (server), `AUTH__ADMIN_API_KEY` (dashboard). For py-cml/examples set `CML_API_KEY` (use same value as `AUTH__API_KEY` for local dev, e.g. `test-key`) |
| **Database** | `DATABASE__POSTGRES_URL`, `DATABASE__NEO4J_URL`, `DATABASE__REDIS_URL` (for integration tests) |
| **Embedding** | `EMBEDDING__DIMENSIONS`, `EMBEDDING__PROVIDER`, `EMBEDDING__MODEL`, etc. |

- **Embedding dimensions:** The DB vector column is created from `EMBEDDING__DIMENSIONS` at migration time. Server and tests must use the **same** value (from `.env`). If you see "expected N dimensions, not M", ensure `.env` has the correct `EMBEDDING__DIMENSIONS` and run `docker compose -f docker/docker-compose.yml down -v` before re-running migrations and tests.
- **Docker:** The `app` and `api` services use `env_file: ../.env` and do **not** override `EMBEDDING__DIMENSIONS`, so migrations and tests in Docker use the same dimensions as in `.env`.

### py-cml tests (host)

- **Integration/e2e:** Require the CML API to be running. Set `CML_API_KEY` and `CML_BASE_URL` in repo root `.env` (use same value as server `AUTH__API_KEY`, e.g. `test-key`). Tests use `CML_TEST_URL` / `CML_TEST_API_KEY` when set, else `CML_BASE_URL` / `CML_API_KEY`. To run the API with test keys without editing `.env`, use: `docker compose -f docker/docker-compose.yml -f docker/docker-compose.test-key.yml up -d api` (see `docker/docker-compose.test-key.yml`).
- **Embedded tests:** Require `pip install -e ".[embedded]"` in `packages/py-cml` so `aiosqlite` and embedded dependencies are available. Embedded config reads `EMBEDDING__DIMENSIONS` (and other `EMBEDDING__*` / `LLM__*` vars) from `.env` when not set in code.

### Optional LLM/embedding tests (skip when unavailable)

- **Server:** `tests/integration/test_forgetting_llm_compression.py` uses the configured LLM for compression. These tests **skip** (instead of fail) when the LLM is not configured, unreachable, or rate-limited (e.g. 429). Marker: `@pytest.mark.requires_llm`. Run only these when a reachable LLM is available: `pytest -m requires_llm tests/integration/test_forgetting_llm_compression.py -v`.
- **py-cml embedded:** `packages/py-cml/tests/embedded/test_lite_mode.py` write/read tests **skip** when the embedding model is unavailable (e.g. sentence-transformers load failure, API/rate-limit errors).

## Skipped tests

Some tests skip when dependencies or services are unavailable. Summary:

| Location | Cause of skip | Fix to run |
|----------|---------------|------------|
| `tests/unit/test_celery_forgetting_task.py` | Celery not installed | `pip install celery` |
| `packages/py-cml/tests/e2e/test_migration.py` | Embedded embedding unavailable | Working embedded embedding (model/API) |
| `packages/py-cml/tests/e2e/conftest.py`, `packages/py-cml/tests/integration/conftest.py` | CML server not reachable | Start CML API; set `CML_BASE_URL` (and auth) |
| `packages/py-cml/tests/integration/test_*` (write/read) | Server stored no chunks or read returned 0 | CML API running with embedding/write-gate |
| `packages/py-cml/tests/embedded/test_*` | Embedding/LLM unavailable in embedded | Install `.[embedded]` and have embedding model/API available |

See the Skipped tests table above for a summary.

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint and then tests with Postgres, Neo4j, and Redis services, migrations, and coverage. Local runs can mirror that by using the same env vars and `pytest tests -v --tb=short --cov=src`.
