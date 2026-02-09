# py-cml

**Python SDK for [CognitiveMemoryLayer](https://github.com/<org>/CognitiveMemoryLayer)** — neuro-inspired memory for AI applications.

[![PyPI](https://img.shields.io/pypi/v/py-cml)](https://pypi.org/project/py-cml/)
[![Python](https://img.shields.io/pypi/pyversions/py-cml)](https://pypi.org/project/py-cml/)
[![License](https://img.shields.io/github/license/<org>/CognitiveMemoryLayer)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/<org>/CognitiveMemoryLayer/py-cml-test.yml?branch=main)](https://github.com/<org>/CognitiveMemoryLayer/actions)

Give your AI applications human-like memory — store, retrieve, consolidate, and forget information just like the brain does.

## Features

- **Client mode** — connect to a running CML server over HTTP (all routes under `/api/v1`)
- **Sync and async** — `CognitiveMemoryLayer` and `AsyncCognitiveMemoryLayer` with context manager support
- **Configuration** — API key, base URL, tenant ID via constructor, environment variables (`CML_*`), or `.env` (Pydantic-validated)
- **Health check** — `health()` returns server status; retry with exponential backoff on 5xx, 429, and connection errors
- **Typed** — full type hints, Pydantic request/response models, and `py.typed` marker
- **Memory operations** — write, read, update, forget, turn, stats (Phase 3)
- **Embedded mode** — run the full CML engine in-process with `EmbeddedCognitiveMemoryLayer` (Phase 4, optional)
- **Advanced features** (Phase 5) — admin (consolidate, run_forgetting), batch write/read, tenant management, event log, component health, namespace isolation (`with_namespace`), memory iteration (`iter_memories`), OpenAI helper (`CMLOpenAIHelper`)

## Installation

```bash
pip install py-cml
```

With optional embedded mode (run CML in-process without a server):

```bash
pip install py-cml[embedded]
```

## Quickstart

```python
from cml import CognitiveMemoryLayer

# Context manager ensures the HTTP client is closed
with CognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000") as memory:
    health = memory.health()
    print(health.status)  # e.g. "healthy"

    # Memory operations
    memory.write("User prefers vegetarian food.")
    result = memory.read("What does the user eat?")
    print(result.context)  # Formatted for LLM injection
    turn = memory.turn(user_message="What should I eat tonight?", session_id="session-001")
    print(turn.memory_context)
```

Async client:

```python
from cml import AsyncCognitiveMemoryLayer

async def main():
    async with AsyncCognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000") as memory:
        health = await memory.health()
        print(health.status)
        await memory.write("User prefers dark mode.")
        result = await memory.read("user preferences")
        print(result.context)

# asyncio.run(main())
```

## Embedded mode

Run the CML engine in-process (no server, no HTTP). Install with `pip install py-cml[embedded]` and, from the monorepo root, install the engine first: `pip install -e .` then `pip install -e packages/py-cml[embedded]`.

```python
from cml import EmbeddedCognitiveMemoryLayer

async def main():
    async with EmbeddedCognitiveMemoryLayer() as memory:
        await memory.write("User prefers vegetarian food.")
        result = await memory.read("dietary preferences")
        print(result.context)

# asyncio.run(main())
```

**Lite mode** (default) uses SQLite (in-memory or file) and local sentence-transformers embeddings — no API keys or external services. For persistent storage, pass `db_path="./my_memories.db"`.

**Export/import:** Use `cml.embedded_utils.export_memories_async()` and `import_memories_async()` to migrate data from embedded to a CML server (or vice versa).

## Configuration

**Option 1: Direct parameters**

```python
memory = CognitiveMemoryLayer(
    api_key="sk-...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
)
```

**Option 2: Environment variables or `.env`**

Unset fields are loaded from the environment (or a `.env` file). Supported variables: `CML_API_KEY`, `CML_BASE_URL`, `CML_TENANT_ID`, `CML_TIMEOUT`, `CML_MAX_RETRIES`, `CML_RETRY_DELAY`, `CML_ADMIN_API_KEY`, `CML_VERIFY_SSL`.

**Option 3: Config object**

```python
from cml import CognitiveMemoryLayer
from cml.config import CMLConfig

config = CMLConfig(
    api_key="sk-...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
    timeout=30.0,
    max_retries=3,
)
memory = CognitiveMemoryLayer(config=config)
```

**Base URL:** Use the server root (e.g. `http://localhost:8000`). The client sends requests to `/api/v1/*` automatically.

## Phase 5: Advanced features

**Admin operations** (require `admin_api_key` in config): `consolidate(tenant_id=..., user_id=...)` triggers episodic-to-semantic migration; `run_forgetting(tenant_id=..., dry_run=True, max_memories=5000)` runs the forgetting pipeline.

**Batch operations:** `batch_write(items, session_id=..., namespace=...)` writes multiple memories (sequential); `batch_read(queries, max_results=..., format=...)` runs multiple read queries (concurrent on async client, sequential on sync).

**Tenant management:** `set_tenant(tenant_id)` switches the active tenant for subsequent calls; `tenant_id` property returns the current tenant; `list_tenants()` returns all tenants and counts (admin only).

**Event log and health:** `get_events(limit=50, page=1, event_type=..., since=...)` returns paginated event log (admin only); `component_health()` returns per-component status (admin only).

**Namespace isolation:** `with_namespace(namespace)` returns a `NamespacedClient` (or `AsyncNamespacedClient`) that injects the namespace into all write/update/batch_write calls.

**Memory iteration:** `iter_memories(memory_types=..., status="active", batch_size=100)` yields `MemoryItem` with automatic pagination (admin only). Sync client returns a generator; async client an async generator.

**OpenAI integration:** Use `CMLOpenAIHelper(memory_client, openai_client, model="gpt-4o")` and `helper.chat(user_message, session_id=..., system_prompt=...)` to run chat completions with automatic memory context injection and exchange storage.

```python
from cml import CognitiveMemoryLayer, CMLOpenAIHelper
from openai import OpenAI

memory = CognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000")
openai_client = OpenAI()
helper = CMLOpenAIHelper(memory, openai_client)
reply = helper.chat("What should I eat tonight?", session_id="s1")
```

## Phase 6: Developer experience

**Logging:** `cml.configure_logging("DEBUG")` enables debug logs (request timing, retries). API keys are never logged in full.

**Graceful degradation:** `read_safe(query, **kwargs)` returns an empty `ReadResponse` on connection or timeout errors instead of raising, so callers can continue without memory context.

**Session context:** Use `with memory.session(name="onboarding") as sess:` (sync) or `async with memory.session(...) as sess:` (async) to get a scoped object that injects `session_id` into all `sess.write()`, `sess.turn()`, and `sess.remember()` calls.

**Exceptions:** All CML exceptions include optional `suggestion` and `request_id` (from `X-Request-ID`) for easier debugging. Use `str(e)` for a full message with suggestions.

**Serialization:** `cml.utils.serialization.serialize_for_api(dict)` and `CMLJSONEncoder` handle UUID, datetime, and nested structures for API payloads.

**Thread safety:** The sync client uses a lock around `set_tenant()`. The async client must be used in the same event loop it was created in (enforced with a clear error).

## Testing

The SDK uses **pytest** with shared fixtures and markers. CI runs **unit tests only** by default (no server required).

**Run unit tests** (fast, no server):

```bash
cd packages/py-cml
pip install -e ".[dev]"
pytest tests/unit/ -v
```

With coverage:

```bash
pytest tests/unit/ -v --cov=cml --cov-report=term-missing --cov-branch
```

**Run integration tests** (require a running CML server):

```bash
export CML_TEST_URL=http://localhost:8000   # optional, default localhost:8000
export CML_TEST_API_KEY=your-key            # optional
pytest tests/integration/ -v -m integration
```

If the server is unreachable, integration tests are skipped.

**Run embedded tests** (require embedded extras and CML engine):

```bash
pip install -e ".[dev,embedded]"
pytest tests/embedded/ -v -m embedded
```

**Run e2e tests** (multi-turn chat, migration; require live server):

```bash
pytest tests/e2e/ -v -m e2e
```

**Markers:** `integration`, `embedded`, `e2e`, `slow`. Exclude non-unit: `pytest -m "not integration and not embedded and not e2e"`.

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Configuration](docs/configuration.md)
- [Examples](docs/examples.md)

See also the [CognitiveMemoryLayer project](https://github.com/<org>/CognitiveMemoryLayer) for server and architecture.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
