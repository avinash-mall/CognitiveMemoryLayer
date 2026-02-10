# cognitive-memory-layer

<<<<<<< HEAD
**Python SDK for [CognitiveMemoryLayer](https://github.com/avinash-mall/CognitiveMemoryLayer)** — neuro-inspired memory for AI applications.

[![PyPI](https://img.shields.io/pypi/v/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![Python](https://img.shields.io/pypi/pyversions/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![License](https://img.shields.io/github/license/avinash-mall/CognitiveMemoryLayer)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/avinash-mall/CognitiveMemoryLayer/py-cml.yml?branch=main)](https://github.com/avinash-mall/CognitiveMemoryLayer/actions)
=======
**Python SDK for the Cognitive Memory Layer** — neuro-inspired memory for AI applications. Give your apps human-like memory: store, retrieve, consolidate, and forget information just like the brain does.

[![PyPI](https://img.shields.io/pypi/v/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![Python](https://img.shields.io/pypi/pyversions/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
>>>>>>> 42897739dbe59559f3754da63c76f08f1e7a6549

**Source code:** [GitHub — CognitiveMemoryLayer](https://github.com/avinash-mall/CognitiveMemoryLayer) (repository and full documentation)

---

## Installation

```bash
pip install cognitive-memory-layer
```

Optional **embedded mode** (run the CML engine in-process, no server):

```bash
pip install cognitive-memory-layer[embedded]
```

---

## Quick start

**Sync client** (connect to a CML server):

```python
from cml import CognitiveMemoryLayer

with CognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000") as memory:
    memory.write("User prefers vegetarian food.")
    result = memory.read("What does the user eat?")
    print(result.context)  # Formatted for LLM injection
    turn = memory.turn(user_message="What should I eat tonight?", session_id="session-001")
    print(turn.memory_context)
```

**Async client:**

```python
from cml import AsyncCognitiveMemoryLayer

async def main():
    async with AsyncCognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000") as memory:
        await memory.write("User prefers dark mode.")
        result = await memory.read("user preferences")
        print(result.context)
```

**Embedded mode** (no server; SQLite + local embeddings):

```python
from cml import EmbeddedCognitiveMemoryLayer

async def main():
    async with EmbeddedCognitiveMemoryLayer() as memory:
        await memory.write("User prefers vegetarian food.")
        result = await memory.read("dietary preferences")
        print(result.context)
# Persistent storage: EmbeddedCognitiveMemoryLayer(db_path="./my_memories.db")
```

---

## Features

- **Client mode** — HTTP client for a running CML server (sync and async, context managers)
- **Embedded mode** — run the full CML engine in-process (optional extra: `cognitive-memory-layer[embedded]`)
- **Configuration** — API key, base URL, tenant via constructor, env vars (`CML_*`), or `.env`
- **Memory API** — write, read, turn, update, forget, stats; sessions; `get_context(query)` for LLM injection
- **Typed** — Pydantic models, type hints, `py.typed` marker
- **Advanced** — batch write/read, tenant management, namespace isolation, OpenAI helper, admin operations (consolidate, forgetting)

---

## Configuration

**Environment variables:** `CML_API_KEY`, `CML_BASE_URL` (default `http://localhost:8000`), `CML_TENANT_ID`, `CML_TIMEOUT`, `CML_MAX_RETRIES`, `CML_ADMIN_API_KEY`, etc. Or pass directly:

```python
memory = CognitiveMemoryLayer(
    api_key="sk-...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
)
```

**Config object:** `from cml.config import CMLConfig` for validated, reusable config.

---

## Documentation and links

- **GitHub repository:** [CognitiveMemoryLayer](https://github.com/avinash-mall/CognitiveMemoryLayer) — source code, issue tracker, and full docs
- **Package docs (on GitHub):** [Getting started](https://github.com/avinash-mall/CognitiveMemoryLayer/tree/main/packages/py-cml/docs/getting-started.md), [API reference](https://github.com/avinash-mall/CognitiveMemoryLayer/tree/main/packages/py-cml/docs/api-reference.md), [Configuration](https://github.com/avinash-mall/CognitiveMemoryLayer/tree/main/packages/py-cml/docs/configuration.md), [Examples](https://github.com/avinash-mall/CognitiveMemoryLayer/tree/main/packages/py-cml/docs/examples.md)
- **Changelog:** [CHANGELOG.md](https://github.com/avinash-mall/CognitiveMemoryLayer/blob/main/packages/py-cml/CHANGELOG.md) on GitHub

<<<<<<< HEAD
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
pytest tests/unit/ -v --cov=src/cml --cov-report=term-missing --cov-branch
```

**Run integration and e2e tests** (require a running CML server):

1. From the **repository root**, start the API and dependencies:
   ```bash
   docker compose -f docker/docker-compose.yml up -d postgres neo4j redis api
   ```
2. Ensure the server accepts the test API key. The project **.env.example** uses `AUTH__API_KEY=test-key` and `AUTH__ADMIN_API_KEY=test-key` for local dev; copy to `.env` or set the same in your `.env` so the API and tests use the same key.
3. From `packages/py-cml` run:
   ```bash
   pytest tests/integration/ tests/e2e/ -v -m "integration or e2e"
   ```

If `CML_TEST_API_KEY` is not set, integration and e2e conftests load the repo root `.env` and use `AUTH__API_KEY` (and `AUTH__ADMIN_API_KEY` for admin) so one key works for both server and tests. You can override with `CML_TEST_URL` and `CML_TEST_API_KEY`. If the server is unreachable, tests are skipped.

**Run embedded tests** (require embedded extras and CML engine):

```bash
pip install -e ".[dev,embedded]"
pytest tests/embedded/ -v -m embedded
```

**Markers:** `integration`, `embedded`, `e2e`, `slow`. Exclude non-unit: `pytest -m "not integration and not embedded and not e2e"`.

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Configuration](docs/configuration.md)
- [Examples](docs/examples.md)

See also the [CognitiveMemoryLayer project](https://github.com/avinash-mall/CognitiveMemoryLayer) for server and architecture.
=======
---
>>>>>>> 42897739dbe59559f3754da63c76f08f1e7a6549

## License

GPL-3.0-or-later. See [LICENSE](https://github.com/avinash-mall/CognitiveMemoryLayer/blob/main/packages/py-cml/LICENSE) in the repository.
