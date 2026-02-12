# Configuration

## Environment Variables

Unset options are loaded from the environment (or a `.env` file). Use the `CML_` prefix. **No hardcoded defaults for URLs or model names** — set `CML_BASE_URL` (and for the OpenAI helper, `OPENAI_MODEL` or `LLM__MODEL`) in `.env`.

| Variable | Description | Default |
|----------|-------------|---------|
| `CML_API_KEY` | API key for authentication | — |
| `CML_BASE_URL` | Base URL of the CML server (required for client) | — (set in .env) |
| `CML_TENANT_ID` | Tenant identifier | `default` |
| `CML_TIMEOUT` | Request timeout in seconds | `30.0` |
| `CML_MAX_RETRIES` | Maximum retry attempts | `3` |
| `CML_RETRY_DELAY` | Base delay between retries (seconds) | `1.0` |
| `CML_MAX_RETRY_DELAY` | Maximum backoff delay (seconds); caps exponential backoff | `60.0` |
| `CML_ADMIN_API_KEY` | Admin API key (for admin operations, e.g. `delete_all`, `list_tenants`, `get_events`) | — |
| `CML_VERIFY_SSL` | Verify SSL certificates (`true`/`false`) | `true` |

For **CMLOpenAIHelper**, set `OPENAI_MODEL` or `LLM__MODEL` in `.env` (or pass `model=` to the helper); no default model in code.

## Direct Initialization

Pass parameters to the client constructor. These override environment variables.

```python
from cml import CognitiveMemoryLayer

memory = CognitiveMemoryLayer(
    api_key="sk-...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
    timeout=30.0,
    max_retries=3,
    retry_delay=1.0,
    verify_ssl=True,
)
```

## Config Object

Use `CMLConfig` for reusable or validated configuration:

```python
from cml import CognitiveMemoryLayer
from cml.config import CMLConfig

config = CMLConfig(
    api_key="sk-...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
    timeout=30.0,
    max_retries=3,
    retry_delay=1.0,
    max_retry_delay=60.0,
    admin_api_key=None,
    verify_ssl=True,
)
memory = CognitiveMemoryLayer(config=config)
```

`CMLConfig` validates `base_url` (must be `http://` or `https://`), `timeout` (positive), and `max_retries` / `retry_delay` (non-negative).

## Priority Order

1. **Direct parameters** — Values passed to the constructor or `CMLConfig(...)`
2. **Environment variables** — `CML_API_KEY`, `CML_BASE_URL`, etc.
3. **.env file** — Loaded by python-dotenv when available
4. **Defaults** — `tenant_id`, `timeout`, `max_retries`, `retry_delay`, `verify_ssl` as in the table. `base_url` has no default; set `CML_BASE_URL` in `.env` or pass it.

## Testing (integration and e2e)

Integration and e2e tests use **`CML_TEST_URL`** (default `http://localhost:8000`) and **`CML_TEST_API_KEY`**. If `CML_TEST_API_KEY` is not set, the test conftests load the repository root `.env` and use **`AUTH__API_KEY`** and **`AUTH__ADMIN_API_KEY`** so the same keys as the server can be used. The project `.env.example` sets both to `test-key` for local development and testing.

## Embedded Configuration

For **EmbeddedCognitiveMemoryLayer**, use `EmbeddedConfig` (or pass constructor args).

| Area | Options |
|------|--------|
| **storage_mode** | `lite` (default), `standard`, `full` — only `lite` is implemented (SQLite + local embeddings) |
| **database** | `EmbeddedDatabaseConfig`: `database_url` (default SQLite), optional Neo4j/Redis |
| **embedding** | `EmbeddedEmbeddingConfig`: `provider`, `model` (set `EMBEDDING__MODEL` in .env), `dimensions`, `api_key`, `base_url` |
| **llm** | `EmbeddedLLMConfig`: `provider`, `model` (set `LLM__MODEL` in .env), `api_key`, `base_url` |
| **auto_consolidate** / **auto_forget** | Optional background tasks (default `False`) |

**Lite mode** uses SQLite (in-memory or file via `db_path`) and local sentence-transformers embeddings. **Standard** and **full** modes require PostgreSQL, Neo4j, and Redis and are not yet implemented in the SDK.

## Versioning

The project follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- **MAJOR** — Breaking API changes (removed methods, changed required parameters or return types)
- **MINOR** — New features, backwards compatible (new methods, optional parameters)
- **PATCH** — Bug fixes, performance, documentation

See [CHANGELOG.md](../CHANGELOG.md) for version history.
