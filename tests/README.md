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
| `tests/unit/`    | Unit tests (mocked dependencies; no DB required for most) |
| `tests/integration/` | Integration tests (real app, DB, Redis, Neo4j when available) |
| `tests/e2e/`     | End-to-end API flow tests |

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

- Pytest config: `pyproject.toml` under `[tool.pytest.ini_options]` (e.g. `testpaths = ["tests"]`, `asyncio_mode = "auto"`).
- Auth for tests: set `AUTH__API_KEY`, `AUTH__ADMIN_API_KEY`, and `AUTH__DEFAULT_TENANT_ID` in the environment (or via `monkeypatch` in fixtures); dashboard endpoints require the admin key.

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint and then tests with Postgres, Neo4j, and Redis services, migrations, and coverage. Local runs can mirror that by using the same env vars and `pytest tests -v --tb=short --cov=src`.
