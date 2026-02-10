# cognitive-memory-layer

**Python SDK for the Cognitive Memory Layer** — neuro-inspired memory for AI applications. Give your apps human-like memory: store, retrieve, consolidate, and forget information just like the brain does.

[![PyPI](https://img.shields.io/pypi/v/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![Python](https://img.shields.io/pypi/pyversions/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

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

---

## License

GPL-3.0-or-later. See [LICENSE](https://github.com/avinash-mall/CognitiveMemoryLayer/blob/main/packages/py-cml/LICENSE) in the repository.
