# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Cognitive Memory Layer (CML) is a FastAPI-based neuro-inspired memory system. Key services: **FastAPI API** (port 8000), **PostgreSQL+pgvector** (port 5432), **Neo4j** (port 7687), **Redis** (port 6379).

### Starting database services

```bash
sudo dockerd &>/tmp/dockerd.log &
sleep 3
sudo docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
```

Wait for all three to be healthy (`sudo docker compose -f docker/docker-compose.yml ps`). Neo4j takes ~30-60s to initialize.

### Neo4j auth gotcha

The Neo4j container's default credentials are set by the docker-compose file's `NEO4J_AUTH` fallback. If you change `NEO4J_AUTH` in `.env`, you **must** delete old Neo4j volumes first (`sudo docker compose -f docker/docker-compose.yml down -v`) or the old credentials persist in the volume.

### Injected cloud secrets override `.env`

The cloud agent environment injects secrets as environment variables (all `DATABASE__*`, `AUTH__*`, etc.). Since pydantic-settings reads env vars with higher priority than `.env`, the server will use the injected values. When starting the API server locally against Docker containers, **explicitly override** all `DATABASE__*`, `AUTH__*`, and `EMBEDDING_INTERNAL__*` env vars on the command line to match your local Docker container settings. See `.env.example` for the full list and expected format.

### Running migrations

```bash
alembic upgrade head
```

Must set `DATABASE__POSTGRES_URL` pointing to the local Postgres first.

### Running tests

- **Unit tests** (no services needed for most): `pytest tests/unit -v --tb=short`
- **Integration tests** (require Postgres + migrations): `pytest tests/integration -v --tb=short`
- **E2E tests** (require running API): `pytest tests/e2e -v`
- Skip `tests/unit/test_api_ingestion.py` and `tests/unit/test_workflow.py` as they require a running API server and are mislabeled as unit tests.
- Override injected secrets with local DB values when running integration tests (same approach as for the API server).

### Memory and embedding model

The local embedding model (`nomic-ai/nomic-embed-text-v2-moe` via sentence-transformers + PyTorch) uses ~3GB RAM. When running tests, set `EMBEDDING_INTERNAL__PROVIDER=openai` to avoid loading the model (unit/integration tests mock embeddings). The API server needs the model loaded for real write/read operations.

### Linting

```bash
ruff check .
```

See `pyproject.toml` `[tool.ruff]` for configuration. Standard commands are documented in `CONTRIBUTING.md`.

### API usage

See `README.md` for curl examples for write, read, and turn endpoints. Use the value of `AUTH__API_KEY` as the `X-API-Key` header.
