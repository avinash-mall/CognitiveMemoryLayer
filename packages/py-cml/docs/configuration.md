# Configuration

## Environment Variables

Unset options are loaded from the environment (or a `.env` file). Use the `CML_` prefix.

| Variable | Description | Default |
|----------|-------------|---------|
| `CML_API_KEY` | API key for authentication | — |
| `CML_BASE_URL` | Base URL of the CML server | `http://localhost:8000` |
| `CML_TENANT_ID` | Tenant identifier | `default` |
| `CML_TIMEOUT` | Request timeout in seconds | `30.0` |
| `CML_MAX_RETRIES` | Maximum retry attempts | `3` |
| `CML_RETRY_DELAY` | Base delay between retries (seconds) | `1.0` |
| `CML_ADMIN_API_KEY` | Admin API key (for admin operations) | — |
| `CML_VERIFY_SSL` | Verify SSL certificates (`true`/`false`) | `true` |

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
4. **Defaults** — `base_url`, `tenant_id`, `timeout`, `max_retries`, `retry_delay`, `verify_ssl` as in the table

## Embedded Configuration

For **EmbeddedCognitiveMemoryLayer**, use `EmbeddedConfig` (or pass constructor args).

| Area | Options |
|------|--------|
| **storage_mode** | `lite` (default), `standard`, `full` — only `lite` is implemented (SQLite + local embeddings) |
| **database** | `EmbeddedDatabaseConfig`: `postgres_url` (default SQLite), optional Neo4j/Redis |
| **embedding** | `EmbeddedEmbeddingConfig`: `provider` (`local`, `openai`, `vllm`), `model`, `dimensions`, `api_key`, `base_url` |
| **llm** | `EmbeddedLLMConfig`: `provider`, `model`, `api_key`, `base_url` |
| **auto_consolidate** / **auto_forget** | Optional background tasks (default `False`) |

**Lite mode** uses SQLite (in-memory or file via `db_path`) and local sentence-transformers embeddings. **Standard** and **full** modes require PostgreSQL, Neo4j, and Redis and are not yet implemented in the SDK.

## Versioning

The project follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- **MAJOR** — Breaking API changes (removed methods, changed required parameters or return types)
- **MINOR** — New features, backwards compatible (new methods, optional parameters)
- **PATCH** — Bug fixes, performance, documentation

See [CHANGELOG.md](../CHANGELOG.md) for version history.
