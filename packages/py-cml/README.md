# cognitive-memory-layer

**Python SDK for the Cognitive Memory Layer** — neuro-inspired memory for AI applications. Store, retrieve, and reason over memories with sync/async clients or an in-process embedded engine.

[![PyPI](https://img.shields.io/pypi/v/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![Python](https://img.shields.io/pypi/pyversions/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## Installation

```bash
pip install cognitive-memory-layer
```

**Embedded mode** (run the CML engine in-process, no server):

```bash
pip install cognitive-memory-layer[embedded]
```

Embedded mode requires the CML engine from the monorepo. From the repo root:

```bash
pip install -e .
pip install -e packages/py-cml[embedded]
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
import asyncio
from cml import AsyncCognitiveMemoryLayer

async def main():
    async with AsyncCognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000") as memory:
        await memory.write("User prefers dark mode.")
        result = await memory.read("user preferences")
        print(result.context)

asyncio.run(main())
```

**Embedded mode** (no server; SQLite + local embeddings):

```python
import asyncio
from cml import EmbeddedCognitiveMemoryLayer

async def main():
    async with EmbeddedCognitiveMemoryLayer() as memory:
        await memory.write("User prefers vegetarian food.")
        result = await memory.read("dietary preferences")
        print(result.context)

asyncio.run(main())
# Persistent: EmbeddedCognitiveMemoryLayer(db_path="./my_memories.db")
```

---

## Configuration

**Environment variables** (use `.env` or set directly): `CML_API_KEY`, `CML_BASE_URL`, `CML_TENANT_ID`, `CML_TIMEOUT`, `CML_MAX_RETRIES`, `CML_ADMIN_API_KEY`, `CML_VERIFY_SSL`. See [Configuration](docs/configuration.md).

**Constructor:**

```python
memory = CognitiveMemoryLayer(
    api_key="sk-...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
)
```

Or use `CMLConfig` for validated, reusable config: `from cml import CMLConfig`.

---

## Features

| Mode | Description |
|------|-------------|
| **Client** | Sync and async HTTP clients for a running CML server; context managers |
| **Embedded** | In-process engine (lite mode: SQLite + local embeddings); no server |

**Memory API:** `write`, `read`, `turn`, `update`, `forget`, `stats`, `get_context`, `create_session`, `get_session_context`, `delete_all`

**Server compatibility:** The CML server supports `delete_all` (admin API key), read filters (`memory_types`, `since`, `until`) and response formats (`packet`, `list`, `llm_context`), and persists write `metadata` and optional `memory_type`. Session context is scoped by `session_id` when provided.

**Advanced:** `batch_write`, `batch_read`, `consolidate`, `run_forgetting`, `with_namespace`, `iter_memories`, `list_tenants`, `get_events`, `component_health`

**OpenAI integration:** `CMLOpenAIHelper(memory_client, openai_client)` — `helper.chat(user_message, session_id=...)` for memory-augmented chat. Set `OPENAI_MODEL` or `LLM__MODEL` in `.env`.

**Developer:** `read_safe` (returns empty on connection/timeout), `memory.session(name=...)` for scoped session injection, `configure_logging("DEBUG")`, typed models (`py.typed`)

**Temporal fidelity:** Optional `timestamp` in `write()`, `turn()`, `remember()` for historical data replay (benchmarks, migration, testing).

---

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Configuration](docs/configuration.md)
- [Examples](docs/examples.md)
- [Temporal Fidelity](docs/temporal-fidelity.md)

[GitHub repository](https://github.com/avinash-mall/CognitiveMemoryLayer) — source, issues, server setup

[CHANGELOG](CHANGELOG.md)

---

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
