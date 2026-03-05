# cognitive-memory-layer

**Python SDK for the Cognitive Memory Layer** — neuro-inspired memory for AI applications. Store, retrieve, and reason over memories with sync/async clients or an in-process embedded engine.

The Cognitive Memory Layer (CML) gives LLMs a neuro-inspired memory system: episodic and semantic storage, consolidation, and active forgetting. It fits into agents, RAG pipelines, and personalized apps as a persistent, queryable memory backend. This SDK provides sync and async HTTP clients for a CML server, plus an optional in-process embedded engine (lite mode: SQLite and local embeddings, no server). You get write/read/turn, sessions, admin and batch operations, and a helper for OpenAI chat.

**Who it's for:** Developers building AI applications that need persistent, queryable memory — chatbots, agents, evaluation pipelines, and personalized assistants.

**What you can do:**

- Power agent loops with retrieved context and store observations in memory.
- Add memory to RAG pipelines so retrieval is informed by prior interactions.
- Personalize by user or session with namespaces and session-scoped context.
- Run benchmarks with eval mode and temporal fidelity (historical timestamps). For bulk evaluation, the server supports `LLM_INTERNAL__*` and the eval script supports `--ingestion-workers`; see [configuration](docs/configuration.md).
- Run embedded without a server for development, demos, or single-machine apps.

[![PyPI](https://img.shields.io/pypi/v/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![Python](https://img.shields.io/pypi/pyversions/cognitive-memory-layer)](https://pypi.org/project/cognitive-memory-layer/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Tests](https://img.shields.io/badge/Tests-175-brightgreen?logo=pytest)](https://github.com/avinash-mall/CognitiveMemoryLayer/tree/main/packages/py-cml/tests)
[![Version](https://img.shields.io/badge/version-1.3.6-blue)](https://github.com/avinash-mall/CognitiveMemoryLayer)

**What's new (1.3.x):** session-scoped `write` route support in `SessionScope`/`AsyncSessionScope`, new dashboard/admin helpers (`dashboard_facts`, `dashboard_invalidate_fact`, `dashboard_export_memories`, `graph_overview`, `admin_consolidate`, `admin_forget`), and wrapper parity updates for `user_timezone`/`timestamp`. See [CHANGELOG](CHANGELOG.md).

---

## Installation

```bash
pip install cognitive-memory-layer
```

**Embedded mode** (run the CML engine in-process, no server). In lite mode, only the **episodic** (vector) store is used; the neocortical (graph/semantic) store is disabled, so there is no knowledge graph or semantic consolidation. Best for development, demos, or single-machine apps.

```bash
pip install cognitive-memory-layer[embedded]
```

From the monorepo, the server and SDK are built from the **repository root** (single `pyproject.toml`). Install in editable mode with optional extras:

```bash
# From repo root: install server + SDK
pip install -e .

# With embedded mode (in-process engine)
pip install -e ".[embedded]"
```

---

## Quick start

**Sync client** — Connect to a CML server, write a memory, read by query, and run a turn with a session; use `result.context` for LLM injection and `result.memories` (or `result.constraints` when the server returns them) for structured access.

```python
from cml import CognitiveMemoryLayer

with CognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000") as memory:
    memory.write("User prefers vegetarian food.")
    result = memory.read("What does the user eat?")
    print(result.context)  # Formatted for LLM injection
    for m in result.memories:
        print(m.text, m.relevance)
    turn = memory.turn(user_message="What should I eat tonight?", session_id="session-001")
    print(turn.memory_context)
```

**Async client** — Same flow as sync; use `async with` and `await` for all operations.

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

**Embedded mode** — No server: SQLite plus local embeddings (lite mode). Use `db_path` for persistence.

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

**Get context for injection** — Use `get_context(query)` when you only need a formatted string for the LLM:

```python
with CognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000") as memory:
    context = memory.get_context("user preferences")
    # Inject context into your system prompt or RAG pipeline
```

**Session-scoped flow** — Use `memory.session(name="...")` to scope writes and reads to a session:

```python
with CognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000") as memory:
    with memory.session(name="session-001") as session:
        session.write("User asked about Italian food.")
        session.read("What did I ask earlier?")
        session.turn(user_message="Any good places nearby?", assistant_response="...")
```

`SessionScope.write()`/`AsyncSessionScope.write()` call `/session/{session_id}/write`, and `SessionScope.read()`/`AsyncSessionScope.read()` call `/session/{session_id}/read`, so session wrappers stay path-scoped on both write and read.

**More usage:** Timezone-aware retrieval with `read(..., user_timezone="America/New_York")` or `turn(..., user_timezone="America/New_York")`. Batch operations: `batch_write([{"content": "..."}, ...])` and `batch_read(["query1", "query2"])` for multiple writes or reads.

---

## Configuration

**Client:** Environment variables (use `.env` or set directly): `CML_API_KEY`, `CML_BASE_URL`, `CML_TENANT_ID`, `CML_TIMEOUT`, `CML_MAX_RETRIES`, `CML_ADMIN_API_KEY`, `CML_VERIFY_SSL`. Use `CMLConfig` for validated, reusable config. See [Configuration](docs/configuration.md).

**Constructor:**

```python
memory = CognitiveMemoryLayer(
    api_key="sk-...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
)
```

Or pass a config object: `from cml import CMLConfig` then `CognitiveMemoryLayer(config=config)`.

**Embedded:** Use `EmbeddedConfig` (or constructor args). Options: `storage_mode` (`lite` | `standard` | `full`; only `lite` is implemented), `tenant_id`, `database`, `embedding`, `llm`, `auto_consolidate`, `auto_forget`. Embedding and LLM are read from `.env` when not set: `EMBEDDING_INTERNAL__PROVIDER`, `EMBEDDING_INTERNAL__MODEL`, `EMBEDDING_INTERNAL__DIMENSIONS`, `EMBEDDING_INTERNAL__BASE_URL`, `LLM_INTERNAL__MODEL`, `LLM_INTERNAL__BASE_URL`. Lite mode uses SQLite and local embeddings; pass `db_path` for a persistent database. Full details in [Configuration](docs/configuration.md).

---

## Features

| Mode | Description |
|------|-------------|
| **Client** | Sync and async HTTP clients for a running CML server; context managers |
| **Embedded** | In-process engine (lite mode: SQLite + local embeddings); no server. Embedded `read()` passes `memory_types`, `since`, and `until` to the orchestrator. |

**Memory API:** `write`, `read`, `read_stream`, `read_safe`, `turn`, `update`, `forget`, `stats`, `get_context`, `create_session`, `get_session_context`, `delete_all`, `remember` (alias for write), `search` (alias for read), `health`. Options: `user_timezone` on `read()`, `get_context()`, `search()`, and `turn()` for timezone-aware "today"/"yesterday"; `timestamp` on `write()`, `turn()`, and `remember()` for event time; `eval_mode` on `write()`/`remember()` for benchmark responses. Write supports `context_tags`, `session_id`, `memory_type`, `namespace`, `metadata`, `agent_id`. Read supports `memory_types`, `since`, `until`, `response_format` (`packet` | `list` | `llm_context`).

**Response shape:** `ReadResponse` has `memories`, `facts`, `preferences`, `episodes`, `constraints` (when the server has constraint extraction), and `context` (formatted string for LLM injection).

**Server compatibility:** The server supports `delete_all` (admin API key), read filters and `user_timezone`, response formats, write `metadata` and `memory_type`, and session-scoped context. Read filters and `user_timezone` are sent when the server supports them. The server can use LLM-based extraction (constraints, facts, salience, importance) when `FEATURES__USE_LLM_*` flags are enabled; see [UsageDocumentation](../../ProjectPlan/UsageDocumentation.md) § Configuration Reference.

**Session and namespace:** `memory.session(name=...)` (SessionScope) scopes writes/reads/turns to a session via session-scoped routes. `with_namespace(namespace)` returns a `NamespacedClient` (and async `AsyncNamespacedClient`) that injects namespace into write, update, and batch_write, and forwards `user_timezone`/`timestamp` on read/turn helpers.

**Admin & batch:** `batch_write`, `batch_read`, `consolidate`, `run_forgetting`, `reconsolidate`, `admin_consolidate`, `admin_forget`, `with_namespace`, `iter_memories`, `list_tenants`, `get_events`, `component_health`. Dashboard admin (require `CML_ADMIN_API_KEY`): `dashboard_overview`, `dashboard_memories`, `dashboard_memory_detail`, `dashboard_facts`, `dashboard_invalidate_fact`, `dashboard_export_memories`, `dashboard_timeline`, `get_sessions` (active sessions from Redis), `get_rate_limits` (rate-limit usage per API key), `get_request_stats` (hourly request volume), `get_graph_stats`, `graph_overview`, `explore_graph`, `search_graph`, `dashboard_neo4j_config`, `get_config`/`update_config`, `get_labile_status`, `test_retrieval`, `get_jobs`, `bulk_memory_action`, `reset_database`.

**Embedded extras:** `EmbeddedConfig` for storage_mode, embedding/LLM, `auto_consolidate`, `auto_forget`. Export/import: `export_memories`, `import_memories` (and async `export_memories_async`, `import_memories_async`) for migration between embedded and server.

**OpenAI integration:** `CMLOpenAIHelper(memory_client, openai_client)` for memory-augmented chat. Set `OPENAI_MODEL` or `LLM_INTERNAL__MODEL` in `.env`.

```python
from openai import OpenAI
from cml import CognitiveMemoryLayer
from cml.integrations import CMLOpenAIHelper

memory = CognitiveMemoryLayer(api_key="...", base_url="...")
helper = CMLOpenAIHelper(memory, OpenAI())
response = helper.chat("What should I eat tonight?", session_id="s1")
```

**Developer:** `read_safe` (returns empty on connection/timeout), `memory.session(name=...)`, `configure_logging("DEBUG")`, typed models (`py.typed`). Typed exceptions: `AuthenticationError`, `AuthorizationError`, `ValidationError`, `RateLimitError`, `NotFoundError`, `ServerError`, `CMLConnectionError`, `CMLTimeoutError`. The `MemoryProvider` protocol is available for custom backends. See [API Reference](docs/api-reference.md).

**Temporal fidelity:** Optional `timestamp` in `write()`, `turn()`, and `remember()` enables historical data replay for benchmarks, migration, and testing. See [Temporal Fidelity](docs/temporal-fidelity.md).

**Eval mode:** `eval_mode=True` in `write()` or `remember()` returns `eval_outcome` and `eval_reason` (stored/skipped and write-gate reason) for benchmark scripts. See [API Reference — Eval mode](docs/api-reference.md#eval-mode-write-gate).

---

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Configuration](docs/configuration.md)
- [Examples](docs/examples.md)
- [Temporal Fidelity](docs/temporal-fidelity.md)
- [Evaluation Module](docs/evaluation.md) — `cml-eval` CLI and Python API
- [Modeling Module](docs/modeling.md) — `cml-models` CLI and Python API
- [Security policy](../../SECURITY.md)

[GitHub repository](https://github.com/avinash-mall/CognitiveMemoryLayer) — source, issues, server setup

[CHANGELOG](CHANGELOG.md)

---

## Testing

The SDK has **175 tests** (unit, integration, embedded, e2e). From the **repository root**:

```bash
# Run all SDK tests
pytest packages/py-cml/tests -v

# Unit only
pytest packages/py-cml/tests/unit -v

# Integration (requires CML API; set CML_BASE_URL, CML_API_KEY)
pytest packages/py-cml/tests/integration -v

# Embedded (requires embedding/LLM from .env or skip)
pytest packages/py-cml/tests/embedded -v

# E2E (requires CML API)
pytest packages/py-cml/tests/e2e -v
```

Some integration, embedded, and e2e tests skip when the CML server or embedding model is unavailable. See the root [tests/README.md](../../tests/README.md) for skipped-test details.

---

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).


---

## Optional Modules (Eval and Modeling)

Install optional modules depending on your workflow:

```bash
# Evaluation utilities (`cml.eval`, `cml-eval`)
pip install "cognitive-memory-layer[eval]"

# Custom model prep/training (`cml.modeling`, `cml-models`)
pip install "cognitive-memory-layer[modeling]"

# Both modules
pip install "cognitive-memory-layer[eval,modeling]"
```

Each extra installs only its own dependencies. Running `cml-eval` or `cml-models` without the corresponding extra produces a clear error message with install instructions.

**Evaluation CLI** — run LoCoMo-Plus benchmarks, validate outputs, and generate comparison reports:

```bash
cml-eval run-full --repo-root .              # Full pipeline (Docker + ingest + QA + judge)
cml-eval run-locomo --limit-samples 10       # Quick test with 10 samples
cml-eval validate --outputs-dir evaluation/outputs
cml-eval report --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json
cml-eval compare --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json
```

**Modeling CLI** — prepare training data and train custom TF-IDF models:

```bash
cml-models prepare --config packages/models/model_pipeline.toml
cml-models train --config packages/models/model_pipeline.toml --families router,pair
cml-models pipeline --config packages/models/model_pipeline.toml  # prepare + train
```

**Python API** — both modules expose typed dataclass configs for programmatic use:

```python
from cml.eval import LocomoEvalConfig, run_locomo_plus
from cml.modeling import PrepareConfig, TrainConfig, run_pipeline
```

See [Evaluation Module](docs/evaluation.md) and [Modeling Module](docs/modeling.md) for full CLI flags, Python API reference, and dataclass field documentation.
