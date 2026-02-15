# Running Tests

This document describes how to run the test suite for the Cognitive Memory Layer project.

## Prerequisites

- **Python 3.11+**
- Project dependencies installed (e.g. `pip install -e ".[dev]"` or `poetry install`)

## Quick start

From the project root:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with short tracebacks
pytest -v --tb=short
```

## Test layout

| Directory        | Purpose |
|------------------|--------|
| `tests/unit/`    | Unit tests (mocked dependencies; no DB required for most). Includes `test_deep_research_improvements.py` for stable keys, batch embeddings, retrieval timeouts, write-time facts, BoundedStateMap, feature flags. |
| `tests/integration/` | Integration tests (real app, DB, Redis, Neo4j when available) |
| `tests/e2e/`     | End-to-end API flow tests |

Total: **301** tests (unit + integration + e2e). Run `python scripts/update_readme_badges.py` from the repo root to refresh the README badge count.

Key test areas: constraint layer, deep research improvements (stable keys, batch embeddings, write-time facts), **batch extraction** (`RelationExtractor.extract_batch` in `test_extraction.py`; orchestrator uses `get_internal_llm_client` in `test_orchestrator_seamless.py`, `test_api_flows.py`), evaluation (timestamp parsing, verbose diagnostics).

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
| **Auth** | `AUTH__API_KEY`, `AUTH__ADMIN_API_KEY` (use `test-key` for local/dev; dashboard tests need admin key) |
| **Database** | `DATABASE__POSTGRES_URL`, `DATABASE__NEO4J_URL`, `DATABASE__REDIS_URL` (for integration tests) |
| **Embedding** | `EMBEDDING__DIMENSIONS`, `EMBEDDING__PROVIDER`, `EMBEDDING__MODEL`, etc. |

- **Embedding dimensions:** The DB vector column is created from `EMBEDDING__DIMENSIONS` at migration time. Server and tests must use the **same** value (from `.env`). If you see "expected N dimensions, not M", ensure `.env` has the correct `EMBEDDING__DIMENSIONS` and run `docker compose -f docker/docker-compose.yml down -v` before re-running migrations and tests.
- **Docker:** The `app` and `api` services use `env_file: ../.env` and do **not** override `EMBEDDING__DIMENSIONS`, so migrations and tests in Docker use the same dimensions as in `.env`.

### py-cml tests (host)

- **Integration/e2e:** Require the CML API to be running; use the same `AUTH__API_KEY` (e.g. `test-key`) in `.env` as the server. Tests load repo root `.env` (e.g. `CML_BASE_URL`, `CML_API_KEY` or `AUTH__API_KEY`).
- **Embedded tests:** Require `pip install -e ".[embedded]"` in `packages/py-cml` so `aiosqlite` and embedded dependencies are available. Embedded config reads `EMBEDDING__DIMENSIONS` (and other `EMBEDDING__*` / `LLM__*` vars) from `.env` when not set in code.

### Optional LLM/embedding tests (skip when unavailable)

- **Server:** `tests/integration/test_phase8_llm_compression.py` uses the configured LLM for compression. These tests **skip** (instead of fail) when the LLM is not configured, unreachable, or rate-limited (e.g. 429). Marker: `@pytest.mark.requires_llm`. Run only these when a reachable LLM is available: `pytest -m requires_llm tests/integration/test_phase8_llm_compression.py -v`.
- **py-cml embedded:** `packages/py-cml/tests/embedded/test_lite_mode.py` write/read tests **skip** when the embedding model is unavailable (e.g. sentence-transformers load failure, API/rate-limit errors).

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint and then tests with Postgres, Neo4j, and Redis services, migrations, and coverage. Local runs can mirror that by using the same env vars and `pytest tests -v --tb=short --cov=src`.
