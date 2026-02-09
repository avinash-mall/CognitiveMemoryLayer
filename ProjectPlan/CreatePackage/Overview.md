# py-cml: Python Package for CognitiveMemoryLayer

## Master Project Plan

---

## 1. Vision

**py-cml** is a pip-installable Python package that lets any Python application use the CognitiveMemoryLayer with a single import. It provides a clean, Pythonic API for all memory operations — write, read, update, forget, consolidate, and more — with both a **client mode** (connecting to a running CML server) and an **embedded mode** (running the full memory system in-process without infrastructure dependencies).

```python
from cml import CognitiveMemoryLayer

memory = CognitiveMemoryLayer(api_key="sk-...", base_url="http://localhost:8000")

# Store a memory
await memory.write("User prefers vegetarian food and lives in Paris.")

# Retrieve relevant memories
result = await memory.read("What does the user eat?")
print(result.context)  # Formatted for LLM injection

# Seamless turn processing
turn = await memory.turn(
    user_message="What should I eat tonight?",
    session_id="session-001"
)
print(turn.memory_context)  # Auto-retrieved, ready to inject
```

---

## 2. Package Identity

| Field | Value |
|:------|:------|
| **Package Name** | `py-cml` |
| **Import Name** | `cml` |
| **PyPI Name** | `py-cml` |
| **GitHub Repo** | Existing `CognitiveMemoryLayer` repo (monorepo) |
| **Package Location** | `packages/py-cml/` subdirectory |
| **License** | GPL-3.0 (matches parent project) |
| **Python** | >= 3.11 |
| **Build System** | Hatchling (PEP 517) |

### 2.1 Monorepo Strategy

The py-cml package lives inside the existing CognitiveMemoryLayer repository as a subdirectory under `packages/py-cml/`. This approach:

- **Shares git history** — all changes tracked in one place
- **Simplifies CI** — one repo, one set of workflows
- **Enables code reuse** — the embedded mode can directly import from `src/` (the CML engine)
- **Single source of truth** — the SDK evolves alongside the server

```
CognitiveMemoryLayer/               # Existing repo (unchanged)
├── src/                             # CML server engine
├── tests/                           # Server tests
├── docker/                          # Server Docker config
├── packages/                        # NEW: publishable packages
│   └── py-cml/                      # The Python SDK package
│       ├── src/cml/                 # Package source
│       ├── tests/                   # SDK tests
│       ├── examples/                # SDK examples
│       ├── docs/                    # SDK docs
│       ├── pyproject.toml           # SDK build config
│       ├── README.md                # SDK README (shown on PyPI)
│       └── CHANGELOG.md             # SDK changelog
├── pyproject.toml                   # Server's existing pyproject.toml
├── README.md                        # Main project README
└── .github/workflows/               # Shared CI (server + SDK)
```

---

## 3. Phase Overview

| Phase | Title | Focus | Dependencies |
|:------|:------|:------|:-------------|
| **Phase 1** | Project Setup & Packaging | Repository structure, build system, CI/CD | None |
| **Phase 2** | Core Client SDK | HTTP client, authentication, connection management | Phase 1 |
| **Phase 3** | Memory Operations API | write, read, update, forget, turn, stats, sessions | Phase 2 |
| **Phase 4** | Embedded Mode | In-process CML engine (no server required) | Phase 2 |
| **Phase 5** | Advanced Features | Consolidation, forgetting, streaming, batch ops | Phase 3 |
| **Phase 6** | Developer Experience | Sync/async support, context managers, error handling, typing | Phase 3 |
| **Phase 7** | Testing & Quality | Unit, integration, e2e tests, CI pipeline | Phase 3, 4 |
| **Phase 8** | Documentation & Publishing | API docs, examples, GitHub release, PyPI publish | All |

---

## 4. Architecture Overview

The SDK lives within the existing CognitiveMemoryLayer repo at `packages/py-cml/`:

```
CognitiveMemoryLayer/                # Existing repo root
├── src/                             # CML server engine (existing, unchanged)
├── tests/                           # Server tests (existing, unchanged)
├── docker/                          # Docker config (existing, unchanged)
├── pyproject.toml                   # Server pyproject.toml (existing)
├── README.md                        # Main project README (existing)
│
├── packages/                        # NEW: publishable packages directory
│   └── py-cml/                      # The Python SDK package
│       ├── src/
│       │   └── cml/                 # Import as `from cml import ...`
│       │       ├── __init__.py      # Public API surface
│       │       ├── client.py        # CognitiveMemoryLayer (HTTP client)
│       │       ├── async_client.py  # AsyncCognitiveMemoryLayer
│       │       ├── embedded.py      # EmbeddedCognitiveMemoryLayer (in-process)
│       │       ├── config.py        # Configuration dataclasses
│       │       ├── models/          # Pydantic models (request/response)
│       │       │   ├── __init__.py
│       │       │   ├── memory.py    # MemoryRecord, MemoryItem, MemoryPacket
│       │       │   ├── requests.py  # WriteRequest, ReadRequest, etc.
│       │       │   ├── responses.py # WriteResponse, ReadResponse, etc.
│       │       │   └── enums.py     # MemoryType, MemoryStatus, etc.
│       │       ├── transport/       # HTTP transport layer
│       │       │   ├── __init__.py
│       │       │   ├── http.py      # httpx-based transport
│       │       │   └── retry.py     # Retry logic with backoff
│       │       ├── exceptions.py    # CMLError hierarchy
│       │       ├── utils/           # Utility functions
│       │       │   ├── __init__.py
│       │       │   └── serialization.py
│       │       └── _version.py      # Version string
│       ├── tests/                   # SDK test suite
│       │   ├── unit/               # Unit tests (mocked)
│       │   ├── integration/        # Integration tests (real server)
│       │   └── conftest.py         # Shared fixtures
│       ├── examples/               # SDK usage examples
│       │   ├── quickstart.py
│       │   ├── chat_with_memory.py
│       │   ├── batch_operations.py
│       │   └── embedded_mode.py
│       ├── docs/                    # SDK documentation
│       │   ├── getting-started.md
│       │   ├── api-reference.md
│       │   ├── configuration.md
│       │   └── examples.md
│       ├── pyproject.toml           # SDK build config (separate from server)
│       ├── README.md                # SDK README (shown on PyPI)
│       ├── CHANGELOG.md             # SDK version history
│       └── LICENSE                  # GPL-3.0 (symlink or copy)
│
└── .github/
    └── workflows/
        ├── test.yml                 # Existing server CI
        ├── py-cml-test.yml          # NEW: SDK CI tests
        ├── py-cml-lint.yml          # NEW: SDK linting
        └── py-cml-publish.yml       # NEW: SDK PyPI publish
```

---

## 5. Public API Surface

### 5.1 Top-Level Imports

```python
from cml import CognitiveMemoryLayer          # Sync client
from cml import AsyncCognitiveMemoryLayer      # Async client
from cml import EmbeddedCognitiveMemoryLayer   # Embedded (no server)

# Models
from cml.models import MemoryRecord, MemoryPacket, MemoryItem
from cml.models import MemoryType, MemoryStatus, MemorySource

# Exceptions
from cml.exceptions import CMLError, AuthenticationError, ConnectionError

# Configuration
from cml.config import CMLConfig
```

### 5.2 Core Methods

| Method | Description | HTTP Endpoint |
|:-------|:------------|:-------------|
| `write(content, **kwargs)` | Store new memory | `POST /api/v1/memory/write` |
| `read(query, **kwargs)` | Retrieve relevant memories | `POST /api/v1/memory/read` |
| `turn(user_message, **kwargs)` | Seamless memory turn | `POST /api/v1/memory/turn` |
| `update(memory_id, **kwargs)` | Update existing memory | `POST /api/v1/memory/update` |
| `forget(memory_ids, **kwargs)` | Forget memories | `POST /api/v1/memory/forget` |
| `stats()` | Get memory statistics | `GET /api/v1/memory/stats` |
| `health()` | Check server health | `GET /api/v1/health` |
| `create_session(**kwargs)` | Create a new session | `POST /api/v1/session/create` |

### 5.3 Advanced Methods

| Method | Description |
|:-------|:------------|
| `consolidate(tenant_id)` | Trigger consolidation cycle |
| `run_forgetting(tenant_id, dry_run)` | Run active forgetting |
| `delete_all()` | Delete all memories (GDPR) |
| `batch_write(items)` | Write multiple memories |
| `batch_read(queries)` | Execute multiple reads |
| `get_session_context(session_id)` | Get full session context |

---

## 6. Configuration Options

```python
from cml import CognitiveMemoryLayer

# Option 1: Direct initialization
memory = CognitiveMemoryLayer(
    api_key="sk-...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
)

# Option 2: Environment variables
# CML_API_KEY=sk-...
# CML_BASE_URL=http://localhost:8000
# CML_TENANT_ID=my-tenant
memory = CognitiveMemoryLayer()  # Auto-reads from env

# Option 3: Config object
from cml.config import CMLConfig
config = CMLConfig(
    api_key="sk-...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
    timeout=30.0,
    max_retries=3,
    retry_delay=1.0,
)
memory = CognitiveMemoryLayer(config=config)
```

---

## 7. Dependencies

### Core (required)
| Package | Purpose |
|:--------|:--------|
| `httpx` | Async/sync HTTP client |
| `pydantic >= 2.0` | Request/response validation |
| `python-dotenv` | Environment variable loading |

### Embedded Mode (optional, `pip install py-cml[embedded]`)
| Package | Purpose |
|:--------|:--------|
| `sqlalchemy[asyncio]` | Database ORM |
| `asyncpg` | PostgreSQL driver |
| `pgvector` | Vector search |
| `neo4j` | Graph database |
| `redis` | Cache |
| `openai` | Embeddings & LLM |
| `sentence-transformers` | Local embeddings |

### Development
| Package | Purpose |
|:--------|:--------|
| `pytest` | Testing |
| `pytest-asyncio` | Async test support |
| `pytest-httpx` | HTTP mocking |
| `ruff` | Linting |
| `mypy` | Type checking |
| `mkdocs-material` | Documentation |

---

## 8. Milestone Deliverables

| Milestone | Version | Deliverables |
|:----------|:--------|:-------------|
| **M1: Alpha** | 0.1.0 | Client SDK with write/read/turn, published to GitHub |
| **M2: Beta** | 0.2.0 | Full API surface, sync + async, embedded mode |
| **M3: RC** | 0.9.0 | Complete test suite, documentation, CI/CD |
| **M4: GA** | 1.0.0 | PyPI publish, stable API, migration guide |

---

## 9. Non-Goals (v1)

- GUI or web dashboard (use the CML server dashboard)
- Direct database administration
- Model training or fine-tuning
- Replacing the CML server for production multi-tenant deployments

---

## 10. Document Index

| Document | Description |
|:---------|:------------|
| [Phase1_ProjectSetup.md](./Phase1_ProjectSetup.md) | Repository structure, build system, CI/CD pipeline |
| [Phase2_CoreClient.md](./Phase2_CoreClient.md) | HTTP client SDK, authentication, connection management |
| [Phase3_MemoryOperations.md](./Phase3_MemoryOperations.md) | Full memory operations API with pseudo-code |
| [Phase4_EmbeddedMode.md](./Phase4_EmbeddedMode.md) | In-process engine, no server required |
| [Phase5_AdvancedFeatures.md](./Phase5_AdvancedFeatures.md) | Consolidation, forgetting, batch ops, streaming |
| [Phase6_DeveloperExperience.md](./Phase6_DeveloperExperience.md) | Sync/async wrappers, context managers, error handling |
| [Phase7_Testing.md](./Phase7_Testing.md) | Test strategy, fixtures, CI pipeline |
| [Phase8_DocumentationPublishing.md](./Phase8_DocumentationPublishing.md) | Docs, examples, GitHub/PyPI publishing |
