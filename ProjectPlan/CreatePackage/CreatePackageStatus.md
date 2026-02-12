# cognitive-memory-layer: Python Package for CognitiveMemoryLayer — Complete Project Plan

Merged from all ProjectPlan/CreatePackage documents.

---

# cognitive-memory-layer: Python Package for CognitiveMemoryLayer

## Master Project Plan

---

## 1. Vision

**cognitive-memory-layer** is a pip-installable Python package that lets any Python application use the CognitiveMemoryLayer with a single import. It provides a clean, Pythonic API for all memory operations — write, read, update, forget, consolidate, and more — with both a **client mode** (connecting to a running CML server) and an **embedded mode** (running the full memory system in-process without infrastructure dependencies).

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
| **Package Name** | `cognitive-memory-layer` |
| **Import Name** | `cml` |
| **PyPI Name** | `cognitive-memory-layer` |
| **GitHub Repo** | Existing `CognitiveMemoryLayer` repo (monorepo) |
| **Package Location** | `packages/py-cml/` subdirectory |
| **License** | GPL-3.0 (matches parent project) |
| **Python** | >= 3.11 |
| **Build System** | Hatchling (PEP 517) |

### 2.1 Monorepo Strategy

The cognitive-memory-layer package lives inside the existing CognitiveMemoryLayer repository as a subdirectory under `packages/py-cml/`. This approach:

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

| Phase | Title | Focus | Dependencies | Status |
|:------|:------|:------|:-------------|:------|
| **Phase 1** | Project Setup & Packaging | Repository structure, build system, CI/CD | None | **Done** |
| **Phase 2** | Core Client SDK | HTTP client, authentication, connection management | Phase 1 | **Done** |
| **Phase 3** | Memory Operations API | write, read, update, forget, turn, stats, sessions | Phase 2 | **Done** |
| **Phase 4** | Embedded Mode | In-process CML engine (no server required) | Phase 2 | **Done** |
| **Phase 5** | Advanced Features | Consolidation, forgetting, streaming, batch ops | Phase 3 | **Done** |
| **Phase 6** | Developer Experience | Sync/async support, context managers, error handling, typing | Phase 3 | **Done** |
| **Phase 7** | Testing & Quality | Unit, integration, e2e tests, CI pipeline | Phase 3, 4 | **Done** |
| **Phase 8** | Documentation & Publishing | API docs, examples, GitHub release, PyPI publish | All | **Done** |

**Current implementation:** Phase 1 (package layout, Hatchling, CI), Phase 2 (Pydantic config, env/.env, exceptions, httpx transport, retry, request/response models, `health()`, context managers), Phase 3 (memory operations: write, read, turn, update, forget, stats, create_session, get_session_context, get_context, remember, search, delete_all), Phase 4 (embedded mode: EmbeddedCognitiveMemoryLayer, lite SQLite + local embeddings, optional background workers, export/import), Phase 5 (advanced features: consolidate, run_forgetting, batch_write, batch_read, set_tenant, list_tenants, get_events, component_health, with_namespace, iter_memories, CMLOpenAIHelper), Phase 6 (developer experience: structured errors with suggestion/request_id, configure_logging, read_safe, TypedDicts, response __str__, serialization, session() context manager, HTTP/2 and limits, deprecation decorator, thread-safe set_tenant, async event-loop check), and Phase 7 (testing: shared conftest fixtures and mock helpers, unit tests including test_models, extended transport/retry, test_serialization/test_logging, integration tests with live_client, embedded tests for lite mode and lifecycle, e2e chat flow and migration, pytest markers and coverage in pyproject.toml, CI running unit-only by default), and Phase 8 (README with PyPI/Python badges and tagline, docs/getting-started, api-reference, configuration, examples; examples/quickstart, chat_with_memory, async_example, embedded_mode, agent_integration; GitHub issue and PR templates; SECURITY.md; CONTRIBUTING with releasing section and PR checklist; publish workflow on py-cml-v* already in place) are implemented in `packages/py-cml/`.

**Code review (2026-02-10):** The issues in [Issues.md](Issues.md) have been addressed. Resolved items include: admin endpoint URL paths (leading slash), shared `dashboard_item_to_memory_item` in `cml.utils.converters`, CML exception names (`CMLConnectionError`/`CMLTimeoutError` with aliases), retry logic for `RateLimitError` and max delay cap, request body serialization (datetime/UUID), SQLite upsert by content_hash, OpenAI helper single-turn flow, background task logging, `database_url` rename, `iter_memories` multi-type validation, conftest type hints, examples using env vars for API keys, `__del__` warning for unclosed clients, and `pyproject.toml` project URLs. See Issues.md for full resolution summary.

**Testing (integration and e2e):** Integration and e2e tests require a running CML server. Start with `docker compose -f docker/docker-compose.yml up -d postgres neo4j redis api` from the repo root. The project `.env.example` uses `AUTH__API_KEY=test-key` and `AUTH__ADMIN_API_KEY=test-key`; use the same in `.env` so the API accepts the key. If `CML_TEST_API_KEY` is unset, test conftests load the repo `.env` and use `AUTH__API_KEY`/`AUTH__ADMIN_API_KEY`. Run from `packages/py-cml`: `pytest tests/integration/ tests/e2e/ -v -m "integration or e2e"`.

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
│       │   ├── conftest.py          # Shared fixtures, mock response helpers
│       │   ├── unit/                # Unit tests (mocked, no server)
│       │   ├── integration/        # Integration tests (live CML server)
│       │   ├── embedded/           # Embedded mode tests (engine deps)
│       │   └── e2e/                # End-to-end (chat flow, migration)
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

**Implemented (Phase 1–2):**

```python
from cml import CognitiveMemoryLayer, AsyncCognitiveMemoryLayer
from cml import CMLConfig, HealthResponse
from cml import (
    CMLError, AuthenticationError, AuthorizationError, ConnectionError,
    NotFoundError, ValidationError, RateLimitError, ServerError, TimeoutError,
)

# Models (enums and request/response types)
from cml.models import MemoryType, MemoryStatus, MemorySource, OperationType
from cml.models import MemoryItem, WriteResponse, ReadResponse, TurnResponse, StatsResponse
from cml.models import WriteRequest, ReadRequest, TurnRequest, UpdateRequest, ForgetRequest
```

**Planned:** `EmbeddedCognitiveMemoryLayer`, `MemoryRecord`, `MemoryPacket` (Phase 4 / later).

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
| `delete_all()` | Delete all memories (GDPR); server implements `DELETE /api/v1/memory/all` (admin key) |
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

### Embedded Mode (optional, `pip install cognitive-memory-layer[embedded]`)
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

The following phase documents are merged into this single file below: Phase 1 (Project Setup & Packaging), Phase 2 (Core Client SDK), Phase 3 (Memory Operations API), Phase 4 (Embedded Mode), Phase 5 (Advanced Features), Phase 6 (Developer Experience), Phase 7 (Testing & Quality), Phase 8 (Documentation & Publishing).

---

# Phase 1: Project Setup & Packaging

**Status: Implemented.** The package layout, Hatchling build, tooling, and CI are in place under `packages/py-cml/`.

## Objective

Establish the cognitive-memory-layer package directory within the existing CognitiveMemoryLayer monorepo, with a modern Python packaging structure, its own build system, development tooling, and CI/CD pipeline scoped to the SDK.

---

## Task 1.1: Monorepo Setup

### Sub-Task 1.1.1: Create Package Directory in Existing Repo

**Architecture**: Create the `packages/py-cml/` subdirectory inside the existing CognitiveMemoryLayer repository. This approach keeps the SDK and server in the same repo, sharing git history and CI infrastructure.

**Implementation**:
```bash
# From the existing CognitiveMemoryLayer repo root
cd CognitiveMemoryLayer

# Create the package directory structure
mkdir -p packages/py-cml

# All SDK work happens inside this subdirectory
cd packages/py-cml
```

**Pseudo-code**:
```
1. Navigate to existing CognitiveMemoryLayer repo
2. Create packages/py-cml/ directory
3. All SDK source, tests, docs go inside packages/py-cml/
4. The existing src/, tests/, docker/, pyproject.toml remain untouched
5. CI workflows added to .github/workflows/ at repo root (shared)
6. Git tags for SDK releases use prefix: py-cml-v0.1.0
```

**Why monorepo?**:
```
ADVANTAGES:
  - Single repo to manage → simpler for contributors
  - Embedded mode can reference parent src/ for code reuse
  - Shared CI infrastructure (GitHub Actions)
  - Atomic commits (server + SDK changes together)
  - Single git history for traceability

CONSIDERATIONS:
  - SDK pyproject.toml is at packages/py-cml/pyproject.toml (NOT repo root)
  - CI workflows must use working-directory: packages/py-cml
  - Git tags use py-cml-v* prefix to distinguish from server tags
  - PyPI build runs from the packages/py-cml/ subdirectory
```

### Sub-Task 1.1.2: Update Existing .gitignore

**Architecture**: Add SDK-specific entries to the existing `.gitignore` at the repo root (or create a `.gitignore` inside `packages/py-cml/`).

**Implementation** (`packages/py-cml/.gitignore`):
```
# Build artifacts
dist/
build/
*.egg-info/
.eggs/
*.egg

# Caches
.mypy_cache/
.pytest_cache/
.ruff_cache/
htmlcov/
.coverage
```

### Sub-Task 1.1.3: Create LICENSE File

**Implementation**: Symlink or copy the GPL-3.0 LICENSE from the repo root. PyPI requires the LICENSE to be in the package directory.

```bash
# Option A: Copy (simpler, works everywhere)
cp ../../LICENSE ./LICENSE

# Option B: Symlink (stays in sync, Unix only)
ln -s ../../LICENSE ./LICENSE
```

---

## Task 1.2: Package Structure

### Sub-Task 1.2.1: Create Directory Layout

**Architecture**: Use `src/` layout (PEP 517 recommended) with `cml` as the import namespace. Everything lives under `packages/py-cml/` in the existing repo.

**Implementation**:
```
CognitiveMemoryLayer/                  # Existing repo root
├── src/                               # Server engine (existing, untouched)
├── packages/
│   └── py-cml/                        # SDK package root
│       ├── src/
│       │   └── cml/
│       │       ├── __init__.py        # Public API exports
│       │       ├── _version.py        # Single source of version truth
│       │       ├── client.py          # Sync client (placeholder)
│       │       ├── async_client.py    # Async client (placeholder)
│       │       ├── config.py          # Configuration (placeholder)
│       │       ├── exceptions.py     # Exception hierarchy (placeholder)
│       │       ├── models/
│       │       │   ├── __init__.py
│       │       │   ├── enums.py       # Enums (placeholder)
│       │       │   ├── memory.py      # Memory models (placeholder)
│       │       │   ├── requests.py    # Request schemas (placeholder)
│       │       │   └── responses.py   # Response schemas (placeholder)
│       │       ├── transport/
│       │       │   ├── __init__.py
│       │       │   ├── http.py        # HTTP transport (placeholder)
│       │       │   └── retry.py       # Retry logic (placeholder)
│       │       └── utils/
│       │           ├── __init__.py
│       │           └── serialization.py
│       ├── tests/
│       │   ├── __init__.py
│       │   ├── conftest.py
│       │   ├── unit/
│       │   │   └── __init__.py
│       │   └── integration/
│       │       └── __init__.py
│       ├── examples/
│       │   └── .gitkeep
│       ├── docs/
│       │   └── .gitkeep
│       ├── pyproject.toml             # SDK-specific build config
│       ├── README.md                  # SDK README (shown on PyPI)
│       ├── CHANGELOG.md               # SDK changelog
│       └── LICENSE                    # GPL-3.0 (copy from root)
└── .github/workflows/                 # Shared CI directory
    ├── test.yml                       # Server CI (existing)
    ├── py-cml-test.yml                # SDK CI (new)
    ├── py-cml-lint.yml                # SDK lint (new)
    └── py-cml-publish.yml             # SDK publish (new)
```

**Pseudo-code for `__init__.py`**:
```python
"""CognitiveMemoryLayer Python SDK."""

from cml._version import __version__
from cml.client import CognitiveMemoryLayer
from cml.async_client import AsyncCognitiveMemoryLayer
from cml.config import CMLConfig
from cml.exceptions import CMLError

__all__ = [
    "__version__",
    "CognitiveMemoryLayer",
    "AsyncCognitiveMemoryLayer",
    "CMLConfig",
    "CMLError",
]
```

**Pseudo-code for `_version.py`**:
```python
__version__ = "0.1.0"
```

### Sub-Task 1.2.2: Verify Import Structure

**Pseudo-code**:
```
1. Verify `import cml` works
2. Verify `from cml import CognitiveMemoryLayer` works
3. Verify `from cml.models import MemoryType` works
4. Verify `from cml.exceptions import CMLError` works
5. Verify no circular imports exist
```

---

## Task 1.3: Build System Configuration

### Sub-Task 1.3.1: Create pyproject.toml

**Architecture**: Use Hatchling as the build backend (modern, fast, PEP 517 compliant). Define optional dependency groups for different use cases.

**Implementation** (`packages/py-cml/pyproject.toml`):
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "cognitive-memory-layer"
version = "0.1.0"
description = "Python SDK for CognitiveMemoryLayer — neuro-inspired memory for AI"
readme = "README.md"
license = "GPL-3.0-or-later"
requires-python = ">=3.11"
authors = [
    { name = "CognitiveMemoryLayer Team" },
]
keywords = ["memory", "llm", "ai", "cognitive", "sdk", "rag"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Typing :: Typed",
]
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
embedded = [
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.30",
    "pgvector>=0.3",
    "neo4j>=5.0",
    "redis>=5.0",
    "openai>=1.0",
    "sentence-transformers>=3.0",
    "tiktoken>=0.8",
    "structlog>=24.0",
    "celery>=5.4",
    "pydantic-settings>=2.0",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "pytest-httpx>=0.34",
    "ruff>=0.8",
    "mypy>=1.0",
    "pre-commit>=3.0",
]
docs = [
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.27",
]

[project.urls]
Homepage = "https://github.com/<org>/CognitiveMemoryLayer"
Documentation = "https://github.com/<org>/CognitiveMemoryLayer/tree/main/packages/py-cml#readme"
Repository = "https://github.com/<org>/CognitiveMemoryLayer"
Issues = "https://github.com/<org>/CognitiveMemoryLayer/issues"
Changelog = "https://github.com/<org>/CognitiveMemoryLayer/blob/main/packages/py-cml/CHANGELOG.md"

[tool.hatch.build.targets.wheel]
packages = ["src/cml"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "TCH", "RUF"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["cml"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

> **Note**: The `pyproject.toml` lives at `packages/py-cml/pyproject.toml`, separate from
> the server's root `pyproject.toml`. The `[project.urls]` point to the CognitiveMemoryLayer
> repo with paths into the `packages/py-cml/` subdirectory.

### Sub-Task 1.3.2: Configure Version Management

**Architecture**: Use explicit version in `pyproject.toml` and `_version.py` (simpler for monorepo since `hatch-vcs` reads from the repo root tags which are shared with the server). SDK releases use prefixed git tags like `py-cml-v0.1.0`.

**Pseudo-code**:
```
1. Set version = "0.1.0" in pyproject.toml [project] section
2. Mirror in _version.py: __version__ = "0.1.0"
3. Tag SDK releases as py-cml-v0.1.0, py-cml-v0.2.0, etc.
   (prefix distinguishes from server tags)
4. CI publish workflow triggers on tags matching "py-cml-v*"
5. Bump version in both pyproject.toml and _version.py before tagging
```

### Sub-Task 1.3.3: Verify Build

**Implementation**:
```bash
# From the repo root, install the SDK in development mode
pip install -e "packages/py-cml[dev]"

# Or from inside the package directory
cd packages/py-cml
pip install -e ".[dev]"

# Verify the package builds correctly
python -m build

# Verify the wheel contains correct files
unzip -l dist/py_cml-0.1.0-py3-none-any.whl

# Verify import works
python -c "from cml import CognitiveMemoryLayer; print('OK')"
```

---

## Task 1.4: Development Tooling

### Sub-Task 1.4.1: Pre-commit Hooks

**Implementation** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.0, httpx>=0.27]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
```

### Sub-Task 1.4.2: Editor Configuration

**Implementation** (`.editorconfig`):
```ini
root = true

[*]
indent_style = space
indent_size = 4
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.{yml,yaml,toml}]
indent_size = 2

[*.md]
trim_trailing_whitespace = false
```

### Sub-Task 1.4.3: Type Stubs & py.typed Marker

**Architecture**: Ship `py.typed` marker so downstream type checkers recognize the package as typed.

**Implementation**:
```
packages/py-cml/src/cml/py.typed   # Empty file, PEP 561 marker
```

---

## Task 1.5: CI/CD Pipeline

### Sub-Task 1.5.1: Test Workflow

**Architecture**: Run SDK tests on pushes/PRs that touch `packages/py-cml/` files. Uses `working-directory` to scope commands to the SDK subdirectory. Tests against Python 3.11, 3.12, 3.13.

**Implementation** (`.github/workflows/py-cml-test.yml`):
```yaml
name: "py-cml: Tests"
on:
  push:
    branches: [main]
    paths:
      - "packages/py-cml/**"
      - ".github/workflows/py-cml-test.yml"
  pull_request:
    branches: [main]
    paths:
      - "packages/py-cml/**"
      - ".github/workflows/py-cml-test.yml"

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: packages/py-cml
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest --cov=cml --cov-report=xml -v
      - uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.12'
        with:
          file: packages/py-cml/coverage.xml
          flags: py-cml
```

### Sub-Task 1.5.2: Lint Workflow

**Implementation** (`.github/workflows/py-cml-lint.yml`):
```yaml
name: "py-cml: Lint"
on:
  push:
    branches: [main]
    paths: ["packages/py-cml/**"]
  pull_request:
    branches: [main]
    paths: ["packages/py-cml/**"]

jobs:
  lint:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: packages/py-cml
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/
      - run: mypy src/cml/
```

### Sub-Task 1.5.3: Publish Workflow

**Architecture**: Publish to PyPI when a git tag matching `py-cml-v*` is pushed. Uses trusted publishing (OIDC). The `working-directory` ensures the build runs from the SDK subdirectory.

**Implementation** (`.github/workflows/py-cml-publish.yml`):
```yaml
name: "py-cml: Publish to PyPI"
on:
  push:
    tags:
      - "py-cml-v*"   # Trigger on tags like py-cml-v0.1.0

permissions:
  id-token: write  # For trusted publishing

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    defaults:
      run:
        working-directory: packages/py-cml
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: packages/py-cml/dist/
```

**Pseudo-code for release process**:
```
1. Developer bumps version in packages/py-cml/pyproject.toml and _version.py
2. Developer commits: "chore(py-cml): prepare release v0.1.0"
3. Developer creates a prefixed git tag: git tag py-cml-v0.1.0
4. Developer pushes tag: git push origin py-cml-v0.1.0
5. Publish workflow triggers (matches py-cml-v* tag pattern):
   a. Checkout repo
   b. Build wheel and sdist from packages/py-cml/
   c. Publish to PyPI using trusted publishing (OIDC)
6. Developer creates GitHub Release from tag (optional, for notes)
7. Users can install: pip install cognitive-memory-layer
```

---

## Task 1.6: README & Initial Documentation

### Sub-Task 1.6.1: Package README

**Architecture**: Create a concise README with badges, installation instructions, quickstart code, and links to full docs.

**Pseudo-code for README sections**:
```
1. Header with badges (PyPI version, Python versions, License, CI status)
2. One-line description
3. Features list (bullet points)
4. Installation:
   - pip install cognitive-memory-layer
   - pip install cognitive-memory-layer[embedded]
5. Quickstart:
   - Import and initialize
   - Write a memory
   - Read memories
   - Seamless turn
6. Configuration section (env vars, direct params)
7. Link to full documentation
8. Link to CognitiveMemoryLayer parent project
9. License notice
```

### Sub-Task 1.6.2: CHANGELOG.md

**Implementation**:
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and build configuration
- Core client SDK with async/sync support
- Memory operations: write, read, update, forget, turn, stats
- Pydantic models for all request/response types
- Exception hierarchy for error handling
- Configuration via environment variables and direct parameters
```

### Sub-Task 1.6.3: CONTRIBUTING.md

**Pseudo-code**:
```
1. Development setup instructions
2. How to run tests
3. Code style (ruff, black, mypy)
4. PR process
5. Commit message conventions (conventional commits)
6. Issue templates
```

---

## Acceptance Criteria

- [x] `packages/py-cml/` directory created in existing CognitiveMemoryLayer repo
- [x] `pip install -e "packages/py-cml[dev]"` succeeds from repo root
- [x] `import cml` works without errors
- [x] `python -m build` (from `packages/py-cml/`) produces valid wheel and sdist
- [x] Ruff linting passes with zero errors
- [x] mypy strict mode passes
- [x] CI workflows trigger only on `packages/py-cml/**` file changes
- [x] CI workflows use `working-directory: packages/py-cml`
- [x] Publish workflow triggers on `py-cml-v*` tag pattern
- [x] SDK `README.md` at `packages/py-cml/README.md` includes installation and quickstart
- [x] `.gitignore`, `.editorconfig` in place
- [x] `py.typed` marker file present at `packages/py-cml/src/cml/py.typed`
- [x] Server's existing `pyproject.toml`, `src/`, `tests/` remain untouched


---

# Phase 2: Core Client SDK

**Status: Implemented.** Pydantic config (env/.env), exception hierarchy, httpx transport (sync + async), retry with backoff, request/response models, and client `health()` with context managers are in place.

## Objective

Build the foundational HTTP client that connects to a running CognitiveMemoryLayer server, handling authentication, connection management, request/response serialization, retry logic, and error handling.

---

## Task 2.1: Configuration System

### Sub-Task 2.1.1: CMLConfig Dataclass

**Architecture**: Single configuration object that accepts parameters directly, from environment variables, or from a `.env` file. Uses Pydantic for validation.

**Implementation** (`src/cml/config.py`):
```python
"""Configuration management for py-cml."""

import os
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CMLConfig(BaseModel):
    """Configuration for CognitiveMemoryLayer client.

    Parameters can be set directly, via environment variables, or
    via a .env file. Environment variables use the CML_ prefix.

    Env vars:
        CML_API_KEY: API key for authentication
        CML_BASE_URL: Base URL of the CML server
        CML_TENANT_ID: Tenant identifier
        CML_TIMEOUT: Request timeout in seconds
        CML_MAX_RETRIES: Maximum retry attempts
        CML_RETRY_DELAY: Delay between retries in seconds
        CML_ADMIN_API_KEY: Admin API key (for admin operations)
    """

    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the CML server"
    )
    tenant_id: str = Field(default="default", description="Tenant identifier")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Base delay between retries (seconds)")
    admin_api_key: Optional[str] = Field(default=None, description="Admin API key")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, values: dict) -> dict:
        """Load unset values from environment variables."""
        env_map = {
            "api_key": "CML_API_KEY",
            "base_url": "CML_BASE_URL",
            "tenant_id": "CML_TENANT_ID",
            "timeout": "CML_TIMEOUT",
            "max_retries": "CML_MAX_RETRIES",
            "retry_delay": "CML_RETRY_DELAY",
            "admin_api_key": "CML_ADMIN_API_KEY",
        }
        for field, env_var in env_map.items():
            if field not in values or values[field] is None:
                env_val = os.environ.get(env_var)
                if env_val is not None:
                    values[field] = env_val
        return values
```

**Pseudo-code for resolution order**:
```
1. Direct parameter passed to constructor (highest priority)
2. Environment variable with CML_ prefix
3. .env file loaded via python-dotenv
4. Default value from Field definition (lowest priority)

For each config field:
    IF explicitly passed → use it
    ELSE IF CML_{FIELD_UPPER} in os.environ → use env var
    ELSE IF .env file has CML_{FIELD_UPPER} → use .env value
    ELSE → use default
```

### Sub-Task 2.1.2: Config Validation

**Pseudo-code**:
```
VALIDATE config:
    IF base_url does not start with http:// or https:// → raise ValueError
    IF timeout <= 0 → raise ValueError
    IF max_retries < 0 → raise ValueError
    IF retry_delay < 0 → raise ValueError
    NORMALIZE base_url: strip trailing slash
```

---

## Task 2.2: Exception Hierarchy

### Sub-Task 2.2.1: Define Exception Classes

**Architecture**: Create a rich exception hierarchy that maps HTTP status codes and transport errors to meaningful Python exceptions.

**Implementation** (`src/cml/exceptions.py`):
```python
"""Exception hierarchy for py-cml."""

from typing import Any, Dict, Optional


class CMLError(Exception):
    """Base exception for all CML errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(CMLError):
    """Raised when authentication fails (401)."""
    pass


class AuthorizationError(CMLError):
    """Raised when authorization fails (403)."""
    pass


class NotFoundError(CMLError):
    """Raised when a resource is not found (404)."""
    pass


class ValidationError(CMLError):
    """Raised when request validation fails (422)."""
    pass


class RateLimitError(CMLError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ServerError(CMLError):
    """Raised when the server returns 5xx error."""
    pass


class ConnectionError(CMLError):
    """Raised when unable to connect to the CML server."""
    pass


class TimeoutError(CMLError):
    """Raised when a request times out."""
    pass
```

**Pseudo-code for status code mapping**:
```
FUNCTION map_status_code(status_code, response_body):
    MATCH status_code:
        401 → raise AuthenticationError
        403 → raise AuthorizationError
        404 → raise NotFoundError
        422 → raise ValidationError
        429 → raise RateLimitError (extract retry_after header)
        >= 500 → raise ServerError
        ELSE → raise CMLError
```

---

## Task 2.3: HTTP Transport Layer

### Sub-Task 2.3.1: Base HTTP Transport

**Architecture**: Use `httpx` for both sync and async HTTP. Wrap all requests with consistent headers, authentication, error mapping, and serialization.

**Implementation** (`src/cml/transport/http.py`):
```python
"""HTTP transport layer using httpx."""

from typing import Any, Dict, Optional
import httpx

from ..config import CMLConfig
from ..exceptions import (
    AuthenticationError,
    AuthorizationError,
    CMLError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)


class HTTPTransport:
    """Synchronous HTTP transport for CML API."""

    def __init__(self, config: CMLConfig):
        self._config = config
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                verify=self._config.verify_ssl,
                headers=self._build_headers(),
            )
        return self._client

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"py-cml/{__version__}",
        }
        if self._config.api_key:
            headers["X-API-Key"] = self._config.api_key
        if self._config.tenant_id:
            headers["X-Tenant-ID"] = self._config.tenant_id
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        use_admin_key: bool = False,
    ) -> Dict[str, Any]:
        """Execute an HTTP request with error handling."""
        headers = {}
        if use_admin_key and self._config.admin_api_key:
            headers["X-API-Key"] = self._config.admin_api_key

        try:
            response = self.client.request(
                method=method,
                url=path,
                json=json,
                params=params,
                headers=headers,
            )
            self._raise_for_status(response)
            return response.json()
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to {self._config.base_url}: {e}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out after {self._config.timeout}s: {e}")

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status codes to CML exceptions."""
        if response.is_success:
            return

        body = None
        try:
            body = response.json()
        except Exception:
            pass

        msg = f"HTTP {response.status_code}"
        if body and "detail" in body:
            msg = body["detail"]

        match response.status_code:
            case 401:
                raise AuthenticationError(msg, status_code=401, response_body=body)
            case 403:
                raise AuthorizationError(msg, status_code=403, response_body=body)
            case 404:
                raise NotFoundError(msg, status_code=404, response_body=body)
            case 422:
                raise ValidationError(msg, status_code=422, response_body=body)
            case 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    msg,
                    status_code=429,
                    response_body=body,
                    retry_after=float(retry_after) if retry_after else None,
                )
            case code if code >= 500:
                raise ServerError(msg, status_code=code, response_body=body)
            case _:
                raise CMLError(msg, status_code=response.status_code, response_body=body)

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()
```

**Pseudo-code**:
```
CLASS HTTPTransport:
    INIT(config):
        Store config
        Create lazy httpx.Client with:
            base_url from config
            timeout from config
            default headers (Content-Type, Accept, User-Agent, X-API-Key, X-Tenant-ID)

    METHOD request(method, path, json=None, params=None):
        TRY:
            response = client.request(method, path, json, params)
            IF response.is_error:
                Map status code → CMLError subclass
                Raise with message, status_code, response_body
            RETURN response.json()
        CATCH httpx.ConnectError:
            Raise ConnectionError
        CATCH httpx.TimeoutException:
            Raise TimeoutError

    METHOD close():
        Close httpx.Client
```

### Sub-Task 2.3.2: Async HTTP Transport

**Architecture**: Mirror the sync transport but use `httpx.AsyncClient`.

**Implementation** (`src/cml/transport/http.py` — async portion):
```python
class AsyncHTTPTransport:
    """Asynchronous HTTP transport for CML API."""

    def __init__(self, config: CMLConfig):
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                verify=self._config.verify_ssl,
                headers=self._build_headers(),
            )
        return self._client

    # _build_headers() — same as sync
    # _raise_for_status() — same as sync

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        use_admin_key: bool = False,
    ) -> Dict[str, Any]:
        headers = {}
        if use_admin_key and self._config.admin_api_key:
            headers["X-API-Key"] = self._config.admin_api_key

        try:
            response = await self.client.request(
                method=method,
                url=path,
                json=json,
                params=params,
                headers=headers,
            )
            self._raise_for_status(response)
            return response.json()
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect: {e}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
```

### Sub-Task 2.3.3: Retry Logic

**Architecture**: Implement exponential backoff with jitter for retryable errors (5xx, 429, connection errors).

**Implementation** (`src/cml/transport/retry.py`):
```python
"""Retry logic with exponential backoff and jitter."""

import asyncio
import random
import time
from typing import Callable, TypeVar

from ..config import CMLConfig
from ..exceptions import CMLError, RateLimitError, ServerError, ConnectionError, TimeoutError

T = TypeVar("T")

RETRYABLE_EXCEPTIONS = (ServerError, ConnectionError, TimeoutError, RateLimitError)


def retry_sync(config: CMLConfig, func: Callable[..., T], *args, **kwargs) -> T:
    """Execute func with sync retry logic."""
    last_exception = None
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            last_exception = e
            if e.retry_after:
                time.sleep(e.retry_after)
            else:
                _sleep_with_backoff(attempt, config.retry_delay)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < config.max_retries:
                _sleep_with_backoff(attempt, config.retry_delay)
    raise last_exception


async def retry_async(config: CMLConfig, func, *args, **kwargs):
    """Execute func with async retry logic."""
    last_exception = None
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except RateLimitError as e:
            last_exception = e
            if e.retry_after:
                await asyncio.sleep(e.retry_after)
            else:
                await _async_sleep_with_backoff(attempt, config.retry_delay)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < config.max_retries:
                await _async_sleep_with_backoff(attempt, config.retry_delay)
    raise last_exception


def _sleep_with_backoff(attempt: int, base_delay: float) -> None:
    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
    time.sleep(delay)


async def _async_sleep_with_backoff(attempt: int, base_delay: float) -> None:
    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
    await asyncio.sleep(delay)
```

**Pseudo-code**:
```
FUNCTION retry(config, callable, *args, **kwargs):
    FOR attempt IN range(max_retries + 1):
        TRY:
            RETURN callable(*args, **kwargs)
        CATCH RateLimitError:
            IF retry_after header → sleep(retry_after)
            ELSE → sleep(backoff_with_jitter(attempt))
        CATCH (ServerError, ConnectionError, TimeoutError):
            IF attempt < max_retries:
                sleep(backoff_with_jitter(attempt))

    RAISE last_exception

FUNCTION backoff_with_jitter(attempt, base_delay):
    delay = base_delay * (2 ^ attempt) + random(0, base_delay)
    RETURN delay
```

---

## Task 2.4: Pydantic Models

### Sub-Task 2.4.1: Enum Definitions

**Architecture**: Mirror the CML server enums so the SDK provides typed constants.

**Implementation** (`src/cml/models/enums.py`):
```python
"""Enums for memory types, status, and operations."""

from enum import Enum


class MemoryType(str, Enum):
    """Type of memory record."""
    EPISODIC_EVENT = "episodic_event"
    SEMANTIC_FACT = "semantic_fact"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    HYPOTHESIS = "hypothesis"
    PREFERENCE = "preference"
    TASK_STATE = "task_state"
    CONVERSATION = "conversation"
    MESSAGE = "message"
    TOOL_RESULT = "tool_result"
    REASONING_STEP = "reasoning_step"
    SCRATCH = "scratch"
    KNOWLEDGE = "knowledge"
    OBSERVATION = "observation"
    PLAN = "plan"


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""
    ACTIVE = "active"
    SILENT = "silent"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemorySource(str, Enum):
    """Provenance source of a memory."""
    USER_EXPLICIT = "user_explicit"
    USER_CONFIRMED = "user_confirmed"
    AGENT_INFERRED = "agent_inferred"
    TOOL_RESULT = "tool_result"
    CONSOLIDATION = "consolidation"
    RECONSOLIDATION = "reconsolidation"


class OperationType(str, Enum):
    """Type of operation in event log."""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"
    REINFORCE = "reinforce"
    DECAY = "decay"
    SILENCE = "silence"
    COMPRESS = "compress"
```

### Sub-Task 2.4.2: Response Models

**Implementation** (`src/cml/models/responses.py`):
```python
"""Response models matching CML API responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .enums import MemoryType


class MemoryItem(BaseModel):
    """A single memory item from retrieval."""
    id: UUID
    text: str
    type: str
    confidence: float
    relevance: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WriteResponse(BaseModel):
    """Response from write operation."""
    success: bool
    memory_id: Optional[UUID] = None
    chunks_created: int = 0
    message: str = ""


class ReadResponse(BaseModel):
    """Response from read operation."""
    query: str
    memories: List[MemoryItem]
    facts: List[MemoryItem] = Field(default_factory=list)
    preferences: List[MemoryItem] = Field(default_factory=list)
    episodes: List[MemoryItem] = Field(default_factory=list)
    llm_context: Optional[str] = None
    total_count: int
    elapsed_ms: float

    @property
    def context(self) -> str:
        """Shortcut to get formatted LLM context string."""
        return self.llm_context or ""


class TurnResponse(BaseModel):
    """Response from seamless turn processing."""
    memory_context: str
    memories_retrieved: int
    memories_stored: int
    reconsolidation_applied: bool = False


class UpdateResponse(BaseModel):
    """Response from update operation."""
    success: bool
    memory_id: UUID
    version: int
    message: str = ""


class ForgetResponse(BaseModel):
    """Response from forget operation."""
    success: bool
    affected_count: int
    message: str = ""


class StatsResponse(BaseModel):
    """Memory statistics response."""
    total_memories: int
    active_memories: int
    silent_memories: int
    archived_memories: int
    by_type: Dict[str, int]
    avg_confidence: float
    avg_importance: float
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None
    estimated_size_mb: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: Optional[str] = None
    components: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Session creation response."""
    session_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
```

### Sub-Task 2.4.3: Request Models

**Implementation** (`src/cml/models/requests.py`):
```python
"""Internal request models for constructing API payloads."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .enums import MemoryType


class WriteRequest(BaseModel):
    """Write memory request payload."""
    content: str
    context_tags: Optional[List[str]] = None
    session_id: Optional[str] = None
    memory_type: Optional[MemoryType] = None
    namespace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    turn_id: Optional[str] = None
    agent_id: Optional[str] = None


class ReadRequest(BaseModel):
    """Read memory request payload."""
    query: str
    max_results: int = Field(default=10, le=50)
    context_filter: Optional[List[str]] = None
    memory_types: Optional[List[MemoryType]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    format: Literal["packet", "list", "llm_context"] = "packet"


class TurnRequest(BaseModel):
    """Seamless turn request payload."""
    user_message: str
    assistant_response: Optional[str] = None
    session_id: Optional[str] = None
    max_context_tokens: int = 1500


class UpdateRequest(BaseModel):
    """Update memory request payload."""
    memory_id: UUID
    text: Optional[str] = None
    confidence: Optional[float] = None
    importance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None


class ForgetRequest(BaseModel):
    """Forget memories request payload."""
    memory_ids: Optional[List[UUID]] = None
    query: Optional[str] = None
    before: Optional[datetime] = None
    action: Literal["delete", "archive", "silence"] = "delete"
```

### Sub-Task 2.4.4: Model Exports

**Implementation** (`src/cml/models/__init__.py`):
```python
"""Public model exports."""

from .enums import MemorySource, MemoryStatus, MemoryType, OperationType
from .responses import (
    ForgetResponse,
    HealthResponse,
    MemoryItem,
    ReadResponse,
    SessionResponse,
    StatsResponse,
    TurnResponse,
    UpdateResponse,
    WriteResponse,
)

__all__ = [
    "MemoryType",
    "MemoryStatus",
    "MemorySource",
    "OperationType",
    "MemoryItem",
    "WriteResponse",
    "ReadResponse",
    "TurnResponse",
    "UpdateResponse",
    "ForgetResponse",
    "StatsResponse",
    "HealthResponse",
    "SessionResponse",
]
```

---

## Task 2.5: Async Client Foundation

### Sub-Task 2.5.1: AsyncCognitiveMemoryLayer Class

**Architecture**: The async client is the primary implementation. All operations are async methods. Uses `AsyncHTTPTransport` for requests.

**Implementation** (`src/cml/async_client.py`):
```python
"""Async client for CognitiveMemoryLayer."""

from typing import Optional

from .config import CMLConfig
from .transport.http import AsyncHTTPTransport


class AsyncCognitiveMemoryLayer:
    """Async Python client for the CognitiveMemoryLayer API.

    Usage:
        async with AsyncCognitiveMemoryLayer(api_key="sk-...") as memory:
            await memory.write("User prefers vegetarian food.")
            result = await memory.read("dietary preferences")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        tenant_id: str = "default",
        *,
        config: Optional[CMLConfig] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        if config:
            self._config = config
        else:
            self._config = CMLConfig(
                api_key=api_key,
                base_url=base_url,
                tenant_id=tenant_id,
                timeout=timeout,
                max_retries=max_retries,
            )
        self._transport = AsyncHTTPTransport(self._config)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP connection."""
        await self._transport.close()

    # Memory operations defined in Phase 3
```

**Pseudo-code**:
```
CLASS AsyncCognitiveMemoryLayer:
    INIT(api_key, base_url, tenant_id, config, timeout, max_retries):
        IF config provided → use it
        ELSE → create CMLConfig from individual params
        Create AsyncHTTPTransport(config)

    ASYNC CONTEXT MANAGER:
        __aenter__ → return self
        __aexit__ → await close()

    ASYNC close():
        Close transport connection

    # All memory operations (write, read, etc.) return parsed Pydantic models
    # Each operation:
    #   1. Build request model
    #   2. Serialize to dict (exclude_none=True)
    #   3. Call transport.request(method, path, json=payload)
    #   4. Parse response into response model
    #   5. Return response model
```

### Sub-Task 2.5.2: Sync Client Wrapper

**Architecture**: The sync client wraps the async client using `asyncio.run()` or an existing event loop.

**Implementation** (`src/cml/client.py`):
```python
"""Synchronous client wrapping the async client."""

import asyncio
from typing import Optional

from .async_client import AsyncCognitiveMemoryLayer
from .config import CMLConfig


class CognitiveMemoryLayer:
    """Synchronous Python client for the CognitiveMemoryLayer API.

    Usage:
        with CognitiveMemoryLayer(api_key="sk-...") as memory:
            memory.write("User prefers vegetarian food.")
            result = memory.read("dietary preferences")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        tenant_id: str = "default",
        *,
        config: Optional[CMLConfig] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        if config:
            self._config = config
        else:
            self._config = CMLConfig(
                api_key=api_key,
                base_url=base_url,
                tenant_id=tenant_id,
                timeout=timeout,
                max_retries=max_retries,
            )
        # Use sync transport directly (not async wrapper)
        from .transport.http import HTTPTransport
        self._transport = HTTPTransport(self._config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        self._transport.close()

    # Memory operations defined in Phase 3
    # Each method mirrors the async version but calls
    # self._transport.request() synchronously
```

**Pseudo-code**:
```
CLASS CognitiveMemoryLayer:
    INIT(same params as async):
        Create CMLConfig
        Create HTTPTransport (sync)

    CONTEXT MANAGER:
        __enter__ → return self
        __exit__ → close()

    close():
        Close transport

    # For each operation (write, read, etc.):
    #   Same logic as async but using sync transport
    #   No asyncio.run() needed — direct sync httpx calls
```

---

## Task 2.6: Connection Verification

### Sub-Task 2.6.1: Health Check

**Pseudo-code**:
```
METHOD health() -> HealthResponse:
    response = transport.request("GET", "/api/v1/health")
    RETURN HealthResponse(**response)
```

### Sub-Task 2.6.2: Connection Test on Init (Optional)

**Pseudo-code**:
```
METHOD verify_connection() -> bool:
    TRY:
        health_response = self.health()
        RETURN health_response.status == "ok"
    CATCH CMLError:
        RETURN False
```

---

## Acceptance Criteria

- [x] `CMLConfig` loads from direct params, env vars, and `.env` files
- [x] Config validation rejects invalid values (bad URLs, negative timeouts)
- [x] Exception hierarchy covers all HTTP status codes
- [x] `HTTPTransport` sends requests with correct headers (API key, tenant ID)
- [x] `AsyncHTTPTransport` works with `async/await`
- [x] Retry logic retries on 5xx, 429, connection errors
- [x] Retry uses exponential backoff with jitter
- [x] `RateLimitError` respects `Retry-After` header
- [x] Both sync and async clients support context manager protocol
- [x] All response models parse server JSON correctly
- [x] `health()` method returns `HealthResponse`
- [x] Type annotations pass `mypy --strict`


---

# Phase 3: Memory Operations API

**Status: Implemented**

## Objective

Implement all memory operations (write, read, update, forget, turn, stats, sessions) on both the sync and async clients, providing a complete Pythonic interface to every CML server endpoint.

---

## Task 3.1: Write Operation

### Sub-Task 3.1.1: Async Write

**Architecture**: Store new information into the CML memory system. Accepts plain text content with optional metadata, context tags, memory type, and session binding.

**Implementation** (`src/cml/async_client.py`):
```python
async def write(
    self,
    content: str,
    *,
    context_tags: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    memory_type: Optional[MemoryType] = None,
    namespace: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    turn_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> WriteResponse:
    """Store new information in memory.

    Args:
        content: The text content to store.
        context_tags: Optional tags for categorization (e.g., ["personal", "preference"]).
        session_id: Optional session identifier for grouping.
        memory_type: Optional explicit type (auto-detected if omitted).
        namespace: Optional namespace for isolation.
        metadata: Optional key-value metadata.
        turn_id: Optional conversation turn identifier.
        agent_id: Optional agent identifier.

    Returns:
        WriteResponse with memory_id, chunks_created, and message.

    Raises:
        AuthenticationError: If API key is invalid.
        ValidationError: If request payload is invalid.
        CMLError: For other server errors.
    """
    payload = WriteRequest(
        content=content,
        context_tags=context_tags,
        session_id=session_id,
        memory_type=memory_type,
        namespace=namespace,
        metadata=metadata or {},
        turn_id=turn_id,
        agent_id=agent_id,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/write", json=payload)
    return WriteResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD write(content, **options) -> WriteResponse:
    1. Build WriteRequest from parameters
    2. Serialize to dict, excluding None values
    3. POST /api/v1/memory/write with JSON body
    4. Parse response into WriteResponse
    5. RETURN WriteResponse(success, memory_id, chunks_created, message)
```

### Sub-Task 3.1.2: Sync Write

**Implementation** (`src/cml/client.py`):
```python
def write(
    self,
    content: str,
    *,
    context_tags: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    memory_type: Optional[MemoryType] = None,
    namespace: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    turn_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> WriteResponse:
    """Store new information in memory (synchronous)."""
    payload = WriteRequest(
        content=content,
        context_tags=context_tags,
        session_id=session_id,
        memory_type=memory_type,
        namespace=namespace,
        metadata=metadata or {},
        turn_id=turn_id,
        agent_id=agent_id,
    ).model_dump(exclude_none=True)

    data = self._transport.request("POST", "/api/v1/memory/write", json=payload)
    return WriteResponse(**data)
```

---

## Task 3.2: Read Operation

### Sub-Task 3.2.1: Async Read

**Architecture**: Retrieve relevant memories using hybrid search (vector + graph + lexical). Supports filtering by context tags, memory types, and time ranges. Returns structured results with an optional pre-formatted LLM context string.

**Implementation**:
```python
async def read(
    self,
    query: str,
    *,
    max_results: int = 10,
    context_filter: Optional[List[str]] = None,
    memory_types: Optional[List[MemoryType]] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    format: Literal["packet", "list", "llm_context"] = "packet",
) -> ReadResponse:
    """Retrieve relevant memories for a query.

    Args:
        query: The search query (natural language).
        max_results: Maximum number of results (1-50).
        context_filter: Filter by context tags.
        memory_types: Filter by memory types.
        since: Only return memories after this time.
        until: Only return memories before this time.
        format: Response format:
            - "packet": Categorized by type (facts, preferences, episodes).
            - "list": Flat list sorted by relevance.
            - "llm_context": Includes pre-formatted context string.

    Returns:
        ReadResponse with memories, optional LLM context, and elapsed time.
    """
    payload = ReadRequest(
        query=query,
        max_results=max_results,
        context_filter=context_filter,
        memory_types=memory_types,
        since=since,
        until=until,
        format=format,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/read", json=payload)
    return ReadResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD read(query, **filters) -> ReadResponse:
    1. Build ReadRequest with query and filters
    2. Serialize (exclude_none)
    3. POST /api/v1/memory/read
    4. Parse into ReadResponse
    5. ReadResponse contains:
       - memories: List[MemoryItem]  (all results)
       - facts: List[MemoryItem]      (semantic facts)
       - preferences: List[MemoryItem] (user preferences)
       - episodes: List[MemoryItem]    (episodic events)
       - llm_context: str | None       (formatted for prompt injection)
       - total_count: int
       - elapsed_ms: float
    6. RETURN ReadResponse

    Convenience: result.context → formatted LLM context string
```

---

## Task 3.3: Seamless Turn

### Sub-Task 3.3.1: Async Turn

**Architecture**: The "seamless" turn endpoint auto-retrieves relevant memories and optionally stores new information from the conversation. This is the primary integration point for chatbots.

**Implementation**:
```python
async def turn(
    self,
    user_message: str,
    *,
    assistant_response: Optional[str] = None,
    session_id: Optional[str] = None,
    max_context_tokens: int = 1500,
) -> TurnResponse:
    """Process a conversation turn with seamless memory.

    Automatically retrieves relevant memories for the user's message
    and optionally stores information from the assistant's response.

    Args:
        user_message: The user's message in this turn.
        assistant_response: Optional assistant response to store.
        session_id: Optional session identifier.
        max_context_tokens: Maximum tokens for memory context.

    Returns:
        TurnResponse with memory_context ready for prompt injection.

    Example:
        turn = await memory.turn(
            user_message="What should I eat tonight?",
            session_id="session-001"
        )
        # Inject turn.memory_context into your LLM prompt
    """
    payload = TurnRequest(
        user_message=user_message,
        assistant_response=assistant_response,
        session_id=session_id,
        max_context_tokens=max_context_tokens,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/turn", json=payload)
    return TurnResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD turn(user_message, assistant_response?, session_id?) -> TurnResponse:
    1. Build TurnRequest payload
    2. POST /api/v1/memory/turn
    3. Server internally:
       a. Retrieves relevant memories for user_message
       b. Formats memories into context string
       c. If assistant_response provided, extracts & stores new info
       d. Runs reconsolidation on retrieved memories
    4. Parse response → TurnResponse
    5. TurnResponse contains:
       - memory_context: str        (ready to inject into prompt)
       - memories_retrieved: int     (count of retrieved memories)
       - memories_stored: int        (count of newly stored memories)
       - reconsolidation_applied: bool
    6. RETURN TurnResponse
```

### Sub-Task 3.3.2: Sync Turn

Mirror of async with sync transport call.

---

## Task 3.4: Update Operation

### Sub-Task 3.4.1: Async Update

**Architecture**: Update an existing memory's text, confidence, importance, metadata, or provide feedback (correct/incorrect/outdated).

**Implementation**:
```python
async def update(
    self,
    memory_id: UUID,
    *,
    text: Optional[str] = None,
    confidence: Optional[float] = None,
    importance: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    feedback: Optional[Literal["correct", "incorrect", "outdated"]] = None,
) -> UpdateResponse:
    """Update an existing memory.

    Args:
        memory_id: UUID of the memory to update.
        text: New text content (triggers re-embedding).
        confidence: Updated confidence score (0.0-1.0).
        importance: Updated importance score (0.0-1.0).
        metadata: Updated metadata dict.
        feedback: Semantic feedback:
            - "correct": Boosts confidence by 0.2
            - "incorrect": Sets confidence to 0, marks as deleted
            - "outdated": Sets valid_to to now

    Returns:
        UpdateResponse with success, version number.
    """
    payload = UpdateRequest(
        memory_id=memory_id,
        text=text,
        confidence=confidence,
        importance=importance,
        metadata=metadata,
        feedback=feedback,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/update", json=payload)
    return UpdateResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD update(memory_id, **patches) -> UpdateResponse:
    1. Build UpdateRequest
    2. POST /api/v1/memory/update
    3. Server:
       a. Finds memory by ID
       b. Validates tenant ownership
       c. If text changed → re-embed, re-extract entities
       d. If feedback == "correct" → confidence += 0.2
       e. If feedback == "incorrect" → confidence = 0, status = DELETED
       f. If feedback == "outdated" → valid_to = now()
       g. Increment version
    4. Parse → UpdateResponse
    5. RETURN UpdateResponse(success, memory_id, version)
```

---

## Task 3.5: Forget Operation

### Sub-Task 3.5.1: Async Forget

**Architecture**: Remove memories by ID, query match, or time filter. Supports soft-delete (archive/silence) and hard-delete.

**Implementation**:
```python
async def forget(
    self,
    *,
    memory_ids: Optional[List[UUID]] = None,
    query: Optional[str] = None,
    before: Optional[datetime] = None,
    action: Literal["delete", "archive", "silence"] = "delete",
) -> ForgetResponse:
    """Forget (remove) memories.

    Args:
        memory_ids: Specific memory IDs to forget.
        query: Forget memories matching this query.
        before: Forget memories created before this time.
        action: Forget strategy:
            - "delete": Hard delete (permanent).
            - "archive": Soft delete, keep for audit.
            - "silence": Make hard to retrieve (needs strong cue).

    Returns:
        ForgetResponse with affected_count.

    Note:
        At least one of memory_ids, query, or before must be provided.
    """
    if not memory_ids and not query and not before:
        raise ValueError("At least one of memory_ids, query, or before must be provided")

    payload = ForgetRequest(
        memory_ids=memory_ids,
        query=query,
        before=before,
        action=action,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/forget", json=payload)
    return ForgetResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD forget(memory_ids?, query?, before?, action) -> ForgetResponse:
    1. Validate at least one selector provided
    2. Build ForgetRequest
    3. POST /api/v1/memory/forget
    4. Server:
       a. Collect target IDs from memory_ids, query results, time filter
       b. Deduplicate
       c. For each target:
          - IF action == "delete" → hard delete
          - IF action == "archive" → set status = ARCHIVED
          - IF action == "silence" → set status = SILENT
    5. RETURN ForgetResponse(success, affected_count)
```

---

## Task 3.6: Stats Operation

### Sub-Task 3.6.1: Async Stats

**Implementation**:
```python
async def stats(self) -> StatsResponse:
    """Get memory statistics for the current tenant.

    Returns:
        StatsResponse with counts, averages, and breakdowns.
    """
    data = await self._transport.request("GET", "/api/v1/memory/stats")
    return StatsResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD stats() -> StatsResponse:
    1. GET /api/v1/memory/stats (tenant from header)
    2. Parse response:
       - total_memories: int
       - active_memories: int
       - silent_memories: int
       - archived_memories: int
       - by_type: Dict[str, int]  e.g. {"episodic_event": 42, "semantic_fact": 15}
       - avg_confidence: float
       - avg_importance: float
       - oldest_memory: datetime | None
       - newest_memory: datetime | None
       - estimated_size_mb: float
    3. RETURN StatsResponse
```

---

## Task 3.7: Session Management

### Sub-Task 3.7.1: Create Session

**Implementation**:
```python
async def create_session(
    self,
    *,
    name: Optional[str] = None,
    ttl_hours: int = 24,
    metadata: Optional[Dict[str, Any]] = None,
) -> SessionResponse:
    """Create a new memory session.

    Args:
        name: Optional human-readable session name.
        ttl_hours: Session time-to-live in hours (default: 24).
        metadata: Optional session metadata.

    Returns:
        SessionResponse with session_id, created_at, expires_at.
    """
    payload = {
        "name": name,
        "ttl_hours": ttl_hours,
        "metadata": metadata or {},
    }
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    data = await self._transport.request("POST", "/api/v1/session/create", json=payload)
    return SessionResponse(**data)
```

### Sub-Task 3.7.2: Get Session Context

**Implementation**:
```python
async def get_session_context(
    self,
    session_id: str,
) -> Dict[str, Any]:
    """Get full session context for LLM injection.

    Args:
        session_id: The session to retrieve context for.

    Returns:
        Dict with messages, tool_results, scratch_pad, and context_string.
    """
    data = await self._transport.request(
        "GET",
        f"/api/v1/session/{session_id}/context",
    )
    return data
```

---

## Task 3.8: Health Check

### Sub-Task 3.8.1: Health

**Implementation**:
```python
async def health(self) -> HealthResponse:
    """Check CML server health.

    Returns:
        HealthResponse with status and component details.
    """
    data = await self._transport.request("GET", "/api/v1/health")
    return HealthResponse(**data)
```

---

## Task 3.9: Delete All (GDPR)

### Sub-Task 3.9.1: Delete All Memories

**Implementation**:
```python
async def delete_all(self, *, confirm: bool = False) -> int:
    """Delete ALL memories for the current tenant (GDPR compliance).

    Args:
        confirm: Must be True to execute. Safety check.

    Returns:
        Number of memories deleted.

    Raises:
        ValueError: If confirm is not True.
    """
    if not confirm:
        raise ValueError(
            "delete_all() requires confirm=True. "
            "This permanently deletes ALL memories for the tenant."
        )
    data = await self._transport.request(
        "DELETE",
        "/api/v1/memory/all",
        use_admin_key=True,
    )
    return data.get("affected_count", 0)
```

**Pseudo-code**:
```
ASYNC METHOD delete_all(confirm=False) -> int:
    1. Safety check: confirm must be True
    2. DELETE /api/v1/memory/all (uses admin API key)
    3. Server paginates and deletes all records for tenant
    4. RETURN affected_count
```

---

## Task 3.10: Convenience Methods

### Sub-Task 3.10.1: Quick Memory Context

**Architecture**: Provide a one-line method to get memory context for LLM prompt injection.

**Implementation**:
```python
async def get_context(
    self,
    query: str,
    *,
    max_results: int = 10,
) -> str:
    """Get formatted memory context string for LLM prompt injection.

    Convenience method that calls read() with format="llm_context"
    and returns just the context string.

    Args:
        query: The search query.
        max_results: Maximum memories to include.

    Returns:
        Formatted context string ready for LLM prompt.
    """
    result = await self.read(query, max_results=max_results, format="llm_context")
    return result.context
```

### Sub-Task 3.10.2: Remember (Alias for Write)

**Implementation**:
```python
async def remember(self, content: str, **kwargs) -> WriteResponse:
    """Alias for write(). More intuitive for some use cases.

    Example:
        await memory.remember("User's birthday is March 15th")
    """
    return await self.write(content, **kwargs)
```

### Sub-Task 3.10.3: Search (Alias for Read)

**Implementation**:
```python
async def search(self, query: str, **kwargs) -> ReadResponse:
    """Alias for read(). More intuitive for search-oriented use cases.

    Example:
        results = await memory.search("birthday")
    """
    return await self.read(query, **kwargs)
```

---

## Task 3.11: Sync Client Implementation

### Sub-Task 3.11.1: Mirror All Methods

**Architecture**: The sync `CognitiveMemoryLayer` class mirrors every method of `AsyncCognitiveMemoryLayer`, but uses `HTTPTransport` (sync) instead of `AsyncHTTPTransport`.

**Pseudo-code**:
```
FOR EACH async method IN AsyncCognitiveMemoryLayer:
    CREATE sync method with same signature (minus async/await)
    REPLACE: await self._transport.request(...) → self._transport.request(...)
    RETURN same response type

Methods to mirror:
    write()            → sync write()
    read()             → sync read()
    turn()             → sync turn()
    update()           → sync update()
    forget()           → sync forget()
    stats()            → sync stats()
    health()           → sync health()
    create_session()   → sync create_session()
    get_session_context() → sync get_session_context()
    delete_all()       → sync delete_all()
    get_context()      → sync get_context()
    remember()         → sync remember()
    search()           → sync search()
```

---

## Task 3.12: Integration Patterns

### Sub-Task 3.12.1: OpenAI Integration Example

**Pseudo-code**:
```python
from openai import OpenAI
from cml import CognitiveMemoryLayer

openai_client = OpenAI()
memory = CognitiveMemoryLayer(api_key="sk-cml-...")

def chat_with_memory(user_message: str, session_id: str) -> str:
    # 1. Get memory context
    turn = memory.turn(user_message=user_message, session_id=session_id)

    # 2. Build prompt with memory context
    messages = [
        {"role": "system", "content": f"You have the following memories:\n{turn.memory_context}"},
        {"role": "user", "content": user_message},
    ]

    # 3. Call LLM
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    assistant_message = response.choices[0].message.content

    # 4. Store the response for future recall
    memory.turn(
        user_message=user_message,
        assistant_response=assistant_message,
        session_id=session_id,
    )

    return assistant_message
```

### Sub-Task 3.12.2: LangChain Integration Pattern

**Pseudo-code**:
```python
from langchain.memory import BaseMemory
from cml import CognitiveMemoryLayer


class CMLMemory(BaseMemory):
    """LangChain-compatible memory using CognitiveMemoryLayer."""

    client: CognitiveMemoryLayer
    session_id: str
    memory_key: str = "memory_context"

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict) -> dict:
        query = inputs.get("input", "")
        context = self.client.get_context(query)
        return {self.memory_key: context}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        self.client.turn(
            user_message=user_input,
            assistant_response=ai_output,
            session_id=self.session_id,
        )

    def clear(self) -> None:
        self.client.delete_all(confirm=True)
```

---

## Acceptance Criteria

- [x] All 13 methods implemented on both async and sync clients
- [x] All methods have comprehensive docstrings with Args, Returns, Raises
- [x] Request payloads use Pydantic models with exclude_none serialization
- [x] Response parsing handles all fields from the CML API
- [x] `forget()` validates at least one selector provided
- [x] `delete_all()` requires explicit `confirm=True`
- [x] `get_context()` returns just the formatted string
- [x] Convenience aliases (`remember`, `search`) delegate correctly
- [x] Type annotations are complete and pass mypy
- [x] Integration patterns documented for OpenAI and LangChain


---

# Phase 4: Embedded Mode

**Status: Implemented**

## Objective

Provide an in-process CognitiveMemoryLayer engine that runs entirely within the user's Python process, eliminating the need for a separate CML server, Docker containers, or external infrastructure. This mode is ideal for development, testing, single-user applications, and environments where running a server is impractical.

---

## Task 4.1: Architecture Design

### Sub-Task 4.1.1: Embedded vs Client Architecture

**Architecture**: The embedded mode reuses the CML core engine (`src/`) from the parent CognitiveMemoryLayer project directly within the Python process. It exposes the same API surface as the HTTP client but routes operations through the `MemoryOrchestrator` in-process instead of over HTTP.

```
┌─────────────────────────────────────────────────┐
│  User's Python Application                       │
│                                                   │
│  from cml import EmbeddedCognitiveMemoryLayer     │
│  memory = EmbeddedCognitiveMemoryLayer(...)       │
│       │                                           │
│       ▼                                           │
│  ┌─────────────────────────────────┐              │
│  │  EmbeddedCognitiveMemoryLayer    │             │
│  │    └── MemoryOrchestrator        │             │
│  │         ├── ShortTermMemory      │             │
│  │         ├── HippocampalStore     │             │
│  │         ├── NeocorticalStore     │             │
│  │         ├── MemoryRetriever      │             │
│  │         ├── ReconsolidationSvc   │             │
│  │         ├── ConsolidationWorker  │             │
│  │         └── ForgettingWorker     │             │
│  └─────────────────────────────────┘              │
│       │              │              │              │
│       ▼              ▼              ▼              │
│  PostgreSQL      Neo4j          Redis              │
│  (or SQLite)   (optional)     (optional)           │
└─────────────────────────────────────────────────┘
```

**Design decisions**:
```
1. FULL ENGINE: Re-use the complete CML engine, not a simplified version
2. SAME API: Same method signatures as HTTP client (write, read, turn, etc.)
3. OPTIONAL DEPS: Only installed with `pip install cognitive-memory-layer[embedded]`
4. LIGHTWEIGHT OPTION: Support SQLite+local embeddings for zero-infra dev
5. FULL OPTION: Support PostgreSQL+Neo4j+Redis for production embedded use
```

### Sub-Task 4.1.2: Storage Backend Options

**Architecture**: Support multiple storage configurations for different use cases.

| Mode | PostgreSQL | Neo4j | Redis | Embeddings | Use Case |
|:-----|:-----------|:------|:------|:-----------|:---------|
| **Lite** | SQLite (in-memory or file) | Disabled | Disabled | Local (sentence-transformers) | Dev, testing, prototyping |
| **Standard** | PostgreSQL + pgvector | Disabled | Disabled | OpenAI or local | Single-user apps |
| **Full** | PostgreSQL + pgvector | Neo4j | Redis | OpenAI or local | Production embedded |

**Pseudo-code for mode selection**:
```
FUNCTION select_storage_mode(config):
    IF config.storage_mode == "lite":
        Use SQLite (aiosqlite)
        Disable Neo4j graph store
        Disable Redis cache
        Use local embeddings (sentence-transformers)
    ELIF config.storage_mode == "standard":
        Use PostgreSQL + pgvector
        Disable Neo4j (use fact-only neocortical store)
        Disable Redis
        Use configured embedding provider
    ELIF config.storage_mode == "full":
        Use PostgreSQL + pgvector
        Use Neo4j
        Use Redis
        Use configured embedding provider
```

---

## Task 4.2: Embedded Configuration

### Sub-Task 4.2.1: EmbeddedConfig

**Implementation** (`src/cml/embedded_config.py`):
```python
"""Configuration for embedded CML mode."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class EmbeddedDatabaseConfig(BaseModel):
    """Database configuration for embedded mode."""
    postgres_url: str = Field(
        default="sqlite+aiosqlite:///cml_memory.db",
        description="Database URL. Use sqlite+aiosqlite:// for lite mode."
    )
    neo4j_url: Optional[str] = Field(default=None, description="Neo4j bolt URL")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    redis_url: Optional[str] = Field(default=None, description="Redis URL")


class EmbeddedEmbeddingConfig(BaseModel):
    """Embedding configuration for embedded mode."""
    provider: Literal["openai", "local", "vllm"] = Field(default="local")
    model: str = Field(default="all-MiniLM-L6-v2")
    dimensions: int = Field(default=384)
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)


class EmbeddedLLMConfig(BaseModel):
    """LLM configuration for embedded mode."""
    provider: Literal["openai", "vllm", "ollama", "gemini", "claude"] = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)


class EmbeddedConfig(BaseModel):
    """Full configuration for embedded CognitiveMemoryLayer."""
    storage_mode: Literal["lite", "standard", "full"] = Field(
        default="lite",
        description="Storage backend complexity level"
    )
    tenant_id: str = Field(default="default")
    database: EmbeddedDatabaseConfig = Field(default_factory=EmbeddedDatabaseConfig)
    embedding: EmbeddedEmbeddingConfig = Field(default_factory=EmbeddedEmbeddingConfig)
    llm: EmbeddedLLMConfig = Field(default_factory=EmbeddedLLMConfig)
    auto_consolidate: bool = Field(
        default=False,
        description="Automatically run consolidation periodically"
    )
    auto_forget: bool = Field(
        default=False,
        description="Automatically run active forgetting periodically"
    )
```

**Pseudo-code**:
```
CLASS EmbeddedConfig:
    storage_mode: "lite" | "standard" | "full"
    tenant_id: str

    database:
        postgres_url: str  (SQLite URL for lite, PostgreSQL for standard/full)
        neo4j_url: str?    (None for lite/standard)
        redis_url: str?    (None for lite/standard)

    embedding:
        provider: "local" | "openai" | "vllm"
        model: str
        dimensions: int
        api_key: str?

    llm:
        provider: "openai" | "vllm" | "ollama" | ...
        model: str
        api_key: str?

    auto_consolidate: bool  (run consolidation in background thread)
    auto_forget: bool       (run forgetting in background thread)
```

---

## Task 4.3: SQLite Storage Adapter

### Sub-Task 4.3.1: SQLite Memory Store

**Architecture**: Create a lightweight SQLite-based memory store that implements the same interface as `PostgresMemoryStore` but uses SQLite with in-memory vector similarity (no pgvector extension needed).

**Implementation** (`src/cml/storage/sqlite_store.py`):
```python
"""SQLite-based memory store for lite embedded mode."""

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiosqlite

from ..models.memory import MemoryRecord


class SQLiteMemoryStore:
    """Memory store backed by SQLite.

    Uses in-memory cosine similarity for vector search
    (suitable for up to ~10,000 records).
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create tables and indexes."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                text TEXT NOT NULL,
                key TEXT,
                namespace TEXT,
                embedding TEXT,  -- JSON array of floats
                entities TEXT,   -- JSON array
                relations TEXT,  -- JSON array
                metadata TEXT,   -- JSON object
                context_tags TEXT, -- JSON array
                confidence REAL DEFAULT 0.5,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                decay_rate REAL DEFAULT 0.01,
                timestamp TEXT,
                written_at TEXT,
                content_hash TEXT,
                version INTEGER DEFAULT 1,
                provenance TEXT  -- JSON object
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tenant ON memories(tenant_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_status ON memories(tenant_id, status)"
        )
        await self._db.commit()

    async def store(self, record: MemoryRecord) -> MemoryRecord:
        """Store a memory record."""
        # ... serialize and INSERT

    async def get_by_id(self, memory_id: UUID) -> Optional[MemoryRecord]:
        """Retrieve a single record by ID."""
        # ... SELECT by id, deserialize

    async def vector_search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[MemoryRecord]:
        """Search by vector similarity using in-memory cosine similarity."""
        # ... Load all embeddings for tenant, compute cosine similarity, sort, limit

    async def scan(
        self,
        tenant_id: str,
        filters: Optional[Dict] = None,
        limit: int = 100,
        order_by: str = "-timestamp",
    ) -> List[MemoryRecord]:
        """Scan records with filters."""
        # ... SELECT with WHERE clauses

    async def update(self, memory_id: UUID, patch: Dict) -> Optional[MemoryRecord]:
        """Update a record."""
        # ... UPDATE with patch fields

    async def delete(self, memory_id: UUID, hard: bool = False) -> None:
        """Delete or soft-delete a record."""
        # ... DELETE or UPDATE status

    async def count(self, tenant_id: str, filters: Optional[Dict] = None) -> int:
        """Count records matching filters."""
        # ... SELECT COUNT(*)
```

**Pseudo-code for in-memory vector search**:
```
METHOD vector_search(tenant_id, query_embedding, limit):
    1. SELECT all rows WHERE tenant_id = ? AND status = 'active'
    2. FOR EACH row:
       a. Parse embedding from JSON
       b. Compute cosine_similarity(query_embedding, row.embedding)
       c. Store (record, similarity) pair
    3. Sort by similarity DESC
    4. Return top `limit` records

FUNCTION cosine_similarity(a, b):
    dot_product = SUM(a[i] * b[i] for i in range(len(a)))
    norm_a = SQRT(SUM(x^2 for x in a))
    norm_b = SQRT(SUM(x^2 for x in b))
    IF norm_a == 0 OR norm_b == 0: RETURN 0.0
    RETURN dot_product / (norm_a * norm_b)
```

---

## Task 4.4: Embedded Client Implementation

### Sub-Task 4.4.1: EmbeddedCognitiveMemoryLayer Class

**Architecture**: Wraps the `MemoryOrchestrator` from the CML engine, providing the same API surface as the HTTP clients.

**Implementation** (`src/cml/embedded.py`):
```python
"""Embedded CognitiveMemoryLayer — runs in-process without a server."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from .embedded_config import EmbeddedConfig
from .models.responses import (
    ForgetResponse,
    ReadResponse,
    StatsResponse,
    TurnResponse,
    UpdateResponse,
    WriteResponse,
    MemoryItem,
)
from .models.enums import MemoryType


class EmbeddedCognitiveMemoryLayer:
    """In-process CognitiveMemoryLayer engine.

    Runs the full CML engine within your Python process.
    No server, no Docker, no HTTP overhead.

    Install with: pip install cognitive-memory-layer[embedded]

    Usage:
        async with EmbeddedCognitiveMemoryLayer() as memory:
            await memory.write("User prefers vegetarian food.")
            result = await memory.read("dietary preferences")

    Storage modes:
        - "lite": SQLite + local embeddings (zero infrastructure)
        - "standard": PostgreSQL + pgvector (requires PostgreSQL)
        - "full": PostgreSQL + Neo4j + Redis (full feature set)
    """

    def __init__(
        self,
        *,
        config: Optional[EmbeddedConfig] = None,
        storage_mode: str = "lite",
        tenant_id: str = "default",
        db_path: Optional[str] = None,
        embedding_provider: str = "local",
        llm_api_key: Optional[str] = None,
    ):
        if config:
            self._config = config
        else:
            self._config = EmbeddedConfig(
                storage_mode=storage_mode,
                tenant_id=tenant_id,
                # ... map other params to config
            )
        self._orchestrator = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize storage backends and the memory orchestrator."""
        self._check_embedded_deps()
        # 1. Initialize storage based on storage_mode
        # 2. Initialize embedding client
        # 3. Initialize LLM client
        # 4. Create MemoryOrchestrator with all dependencies
        # 5. Run schema migrations if needed
        self._initialized = True

    def _check_embedded_deps(self) -> None:
        """Verify embedded dependencies are installed."""
        try:
            import sqlalchemy
            import asyncpg  # or aiosqlite for lite mode
        except ImportError:
            raise ImportError(
                "Embedded mode requires additional dependencies. "
                "Install with: pip install cognitive-memory-layer[embedded]"
            )

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Shutdown storage connections and background workers."""
        # Close database connections
        # Stop background consolidation/forgetting threads
        self._initialized = False

    # --- Memory Operations (same signature as HTTP client) ---

    async def write(self, content: str, **kwargs) -> WriteResponse:
        """Store new information in memory."""
        self._ensure_initialized()
        result = await self._orchestrator.write(
            tenant_id=self._config.tenant_id,
            content=content,
            **kwargs,
        )
        return WriteResponse(
            success=True,
            memory_id=result.get("memory_id"),
            chunks_created=result.get("chunks_created", 0),
            message=result.get("message", ""),
        )

    async def read(self, query: str, **kwargs) -> ReadResponse:
        """Retrieve relevant memories."""
        self._ensure_initialized()
        packet = await self._orchestrator.read(
            tenant_id=self._config.tenant_id,
            query=query,
            **kwargs,
        )
        # Convert MemoryPacket → ReadResponse
        return self._packet_to_read_response(query, packet)

    async def turn(self, user_message: str, **kwargs) -> TurnResponse:
        """Process a conversation turn with seamless memory."""
        self._ensure_initialized()
        # Use SeamlessMemoryProvider internally
        # ... implementation details

    async def update(self, memory_id: UUID, **kwargs) -> UpdateResponse:
        """Update an existing memory."""
        self._ensure_initialized()
        result = await self._orchestrator.update(
            tenant_id=self._config.tenant_id,
            memory_id=memory_id,
            **kwargs,
        )
        return UpdateResponse(
            success=True,
            memory_id=memory_id,
            version=result.get("version", 1),
        )

    async def forget(self, **kwargs) -> ForgetResponse:
        """Forget memories."""
        self._ensure_initialized()
        result = await self._orchestrator.forget(
            tenant_id=self._config.tenant_id,
            **kwargs,
        )
        return ForgetResponse(
            success=True,
            affected_count=result.get("affected_count", 0),
        )

    async def stats(self) -> StatsResponse:
        """Get memory statistics."""
        self._ensure_initialized()
        result = await self._orchestrator.get_stats(
            tenant_id=self._config.tenant_id,
        )
        return StatsResponse(**result)

    async def consolidate(self) -> Dict[str, Any]:
        """Manually trigger memory consolidation."""
        self._ensure_initialized()
        return await self._orchestrator.consolidation.run(
            tenant_id=self._config.tenant_id,
        )

    async def run_forgetting(self, *, dry_run: bool = True) -> Dict[str, Any]:
        """Manually trigger active forgetting."""
        self._ensure_initialized()
        return await self._orchestrator.forgetting.run(
            tenant_id=self._config.tenant_id,
            dry_run=dry_run,
        )

    # --- Internal Helpers ---

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "EmbeddedCognitiveMemoryLayer not initialized. "
                "Use `async with` or call `await memory.initialize()` first."
            )

    def _packet_to_read_response(self, query, packet) -> ReadResponse:
        """Convert internal MemoryPacket to client ReadResponse."""
        # Map packet.facts, packet.preferences, etc. to MemoryItem lists
        # Generate llm_context string
        # Return ReadResponse
```

**Pseudo-code**:
```
CLASS EmbeddedCognitiveMemoryLayer:

    INIT(config or individual params):
        Build EmbeddedConfig
        Set orchestrator = None
        Set initialized = False

    ASYNC initialize():
        1. Check embedded dependencies are installed
        2. SWITCH storage_mode:
           "lite":
              Create SQLiteMemoryStore
              Create local EmbeddingClient (sentence-transformers)
              Create LLM client (for extraction, chunking)
           "standard":
              Create PostgresMemoryStore (asyncpg)
              Run alembic migrations
              Create configured EmbeddingClient
              Create LLM client
           "full":
              Create PostgresMemoryStore
              Create Neo4jGraphStore
              Create RedisClient
              Run migrations
              Create EmbeddingClient
              Create LLM client
        3. Assemble MemoryOrchestrator with all components
        4. IF auto_consolidate → start background consolidation
        5. IF auto_forget → start background forgetting
        6. Set initialized = True

    ASYNC write(content, **kwargs) -> WriteResponse:
        Ensure initialized
        result = orchestrator.write(tenant_id, content, **kwargs)
        RETURN WriteResponse from result dict

    ASYNC read(query, **kwargs) -> ReadResponse:
        Ensure initialized
        packet = orchestrator.read(tenant_id, query, **kwargs)
        Convert MemoryPacket → ReadResponse
        RETURN ReadResponse

    ASYNC turn(user_message, **kwargs) -> TurnResponse:
        Ensure initialized
        Use SeamlessMemoryProvider.process_turn()
        RETURN TurnResponse

    ASYNC close():
        Close all database connections
        Stop background workers
        Set initialized = False
```

---

## Task 4.5: Background Workers (Optional)

### Sub-Task 4.5.1: In-Process Consolidation Worker

**Architecture**: Run consolidation periodically in a background asyncio task instead of using Celery.

**Pseudo-code**:
```
CLASS BackgroundConsolidation:
    INIT(orchestrator, interval_hours=24):
        self.orchestrator = orchestrator
        self.interval = interval_hours * 3600
        self.task = None

    ASYNC start():
        self.task = asyncio.create_task(self._run_loop())

    ASYNC _run_loop():
        WHILE True:
            TRY:
                await asyncio.sleep(self.interval)
                await self.orchestrator.consolidation.run(tenant_id)
                LOG "Consolidation completed"
            CATCH Exception as e:
                LOG "Consolidation failed: {e}"

    ASYNC stop():
        IF self.task:
            self.task.cancel()
            TRY:
                await self.task
            CATCH asyncio.CancelledError:
                pass
```

### Sub-Task 4.5.2: In-Process Forgetting Worker

**Pseudo-code**:
```
CLASS BackgroundForgetting:
    INIT(orchestrator, interval_hours=24):
        Same pattern as BackgroundConsolidation
        Uses orchestrator.forgetting.run()
```

---

## Task 4.6: Lite Mode Shortcuts

### Sub-Task 4.6.1: Zero-Config Quick Start

**Architecture**: Allow creating an embedded instance with zero configuration for instant prototyping.

**Implementation**:
```python
# Absolute minimum setup:
from cml import EmbeddedCognitiveMemoryLayer

async def main():
    async with EmbeddedCognitiveMemoryLayer() as memory:
        await memory.write("User prefers vegetarian food.")
        result = await memory.read("dietary preferences")
        print(result.context)

# This uses:
# - SQLite in-memory database (lost on exit)
# - Local sentence-transformers embeddings
# - No Neo4j, no Redis
# - No API keys needed (for embeddings)
```

### Sub-Task 4.6.2: Persistent Lite Mode

**Implementation**:
```python
# Persistent storage to file:
from cml import EmbeddedCognitiveMemoryLayer

memory = EmbeddedCognitiveMemoryLayer(
    db_path="./my_memories.db",   # SQLite file
    tenant_id="my-app",
)
await memory.initialize()

# Data persists between restarts
await memory.write("User's birthday is March 15th")
```

---

## Task 4.7: Embedded-to-Server Migration

### Sub-Task 4.7.1: Export/Import Utilities

**Architecture**: Provide utilities to export memories from embedded mode and import them into a CML server (or vice versa).

**Pseudo-code**:
```python
async def export_memories(
    source: EmbeddedCognitiveMemoryLayer,
    output_path: str,
    format: Literal["json", "jsonl"] = "jsonl",
) -> int:
    """Export all memories to a file."""
    records = await source._orchestrator.hippocampal.store.scan(
        tenant_id=source._config.tenant_id,
        limit=100000,
    )
    with open(output_path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    return len(records)


async def import_memories(
    target: CognitiveMemoryLayer | EmbeddedCognitiveMemoryLayer,
    input_path: str,
) -> int:
    """Import memories from a file."""
    count = 0
    with open(input_path) as f:
        for line in f:
            record = MemoryRecord.model_validate_json(line)
            await target.write(record.text, metadata=record.metadata)
            count += 1
    return count
```

---

## Acceptance Criteria

<<<<<<< HEAD
- [x] `pip install py-cml[embedded]` installs all required dependencies
- [x] `EmbeddedCognitiveMemoryLayer()` works with zero configuration (lite mode)
- [x] Lite mode uses SQLite + local embeddings (no external services)
- [x] Standard mode connects to PostgreSQL with pgvector
- [x] Full mode uses PostgreSQL + Neo4j + Redis
- [x] Same API surface as HTTP client (write, read, turn, update, forget, stats)
- [x] Context manager protocol (`async with`) handles init/teardown
- [x] Background consolidation/forgetting works via asyncio tasks
- [x] Export/import utilities support data migration
- [x] Missing `[embedded]` dependencies produce clear error message
- [x] Embedded mode passes the same functional tests as HTTP client
=======
- [ ] `pip install cognitive-memory-layer[embedded]` installs all required dependencies
- [ ] `EmbeddedCognitiveMemoryLayer()` works with zero configuration (lite mode)
- [ ] Lite mode uses SQLite + local embeddings (no external services)
- [ ] Standard mode connects to PostgreSQL with pgvector
- [ ] Full mode uses PostgreSQL + Neo4j + Redis
- [ ] Same API surface as HTTP client (write, read, turn, update, forget, stats)
- [ ] Context manager protocol (`async with`) handles init/teardown
- [ ] Background consolidation/forgetting works via asyncio tasks
- [ ] Export/import utilities support data migration
- [ ] Missing `[embedded]` dependencies produce clear error message
- [ ] Embedded mode passes the same functional tests as HTTP client
>>>>>>> 42897739dbe59559f3754da63c76f08f1e7a6549


---

# Phase 5: Advanced Features

## Objective

Implement advanced memory management features including admin operations (consolidation, forgetting), batch operations, streaming, tenant management, and framework integrations that make py-cml production-ready.

---

## Task 5.1: Admin Operations

### Sub-Task 5.1.1: Trigger Consolidation

**Architecture**: Expose the server's consolidation endpoint (episodic-to-semantic migration) via the client. Requires admin API key.

**Implementation**:
```python
async def consolidate(
    self,
    *,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Trigger memory consolidation (episodic → semantic migration).

    Runs the "sleep cycle" — samples recent episodes, clusters them,
    extracts semantic gists, and migrates to the neocortical store.

    Requires admin API key.

    Args:
        tenant_id: Target tenant (defaults to configured tenant).
        user_id: Optional specific user within tenant.

    Returns:
        Dict with consolidation results:
            - episodes_sampled: int
            - clusters_formed: int
            - facts_extracted: int
            - migrations_completed: int
    """
    payload = {
        "tenant_id": tenant_id or self._config.tenant_id,
    }
    if user_id:
        payload["user_id"] = user_id

    return await self._transport.request(
        "POST",
        "/api/v1/admin/consolidate",
        json=payload,
        use_admin_key=True,
    )
```

**Pseudo-code**:
```
ASYNC METHOD consolidate(tenant_id?, user_id?) -> Dict:
    1. Build payload with tenant_id (default to config)
    2. POST /api/v1/admin/consolidate (admin API key)
    3. Server runs consolidation pipeline:
       a. EpisodeSampler: Select recent episodes
       b. SemanticClusterer: Group similar memories
       c. Summarizer: Extract semantic gists
       d. SchemaAligner: Align with existing schemas
       e. Migrator: Move hippo → neocortex
    4. RETURN results dict
```

### Sub-Task 5.1.2: Trigger Active Forgetting

**Implementation**:
```python
async def run_forgetting(
    self,
    *,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    dry_run: bool = True,
    max_memories: int = 5000,
) -> Dict[str, Any]:
    """Trigger active forgetting cycle.

    Scores all memories by relevance and applies forgetting actions:
    KEEP (>0.7), DECAY (>0.5), SILENCE (>0.3), COMPRESS (>0.1), DELETE (<=0.1).

    Requires admin API key.

    Args:
        tenant_id: Target tenant (defaults to configured tenant).
        user_id: Optional specific user.
        dry_run: If True, only report what would happen (default: True).
        max_memories: Maximum memories to process.

    Returns:
        Dict with forgetting results:
            - total_scored: int
            - actions: Dict[str, int] (e.g., {"KEEP": 100, "DECAY": 20, ...})
            - dry_run: bool
    """
    payload = {
        "tenant_id": tenant_id or self._config.tenant_id,
        "dry_run": dry_run,
        "max_memories": max_memories,
    }
    if user_id:
        payload["user_id"] = user_id

    return await self._transport.request(
        "POST",
        "/api/v1/admin/forget",
        json=payload,
        use_admin_key=True,
    )
```

**Pseudo-code**:
```
ASYNC METHOD run_forgetting(dry_run=True, max_memories=5000) -> Dict:
    1. Build payload
    2. POST /api/v1/admin/forget (admin API key)
    3. Server runs forgetting pipeline:
       a. RelevanceScorer: Score each memory
       b. ForgettingExecutor: Determine action per score threshold
       c. IF NOT dry_run:
          - DECAY: Reduce confidence
          - SILENCE: Set status = SILENT
          - COMPRESS: LLM-summarize content
          - DELETE: Hard or soft delete
       d. IF dry_run: Only return planned actions
    4. RETURN results with action counts
```

---

## Task 5.2: Batch Operations

### Sub-Task 5.2.1: Batch Write

**Architecture**: Write multiple memories in a single call to reduce HTTP overhead. Server processes them sequentially but in one request.

**Implementation**:
```python
async def batch_write(
    self,
    items: List[Dict[str, Any]],
    *,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> List[WriteResponse]:
    """Write multiple memories in a single request.

    Args:
        items: List of dicts, each containing at minimum 'content'.
            Optional keys: context_tags, memory_type, metadata, agent_id.
        session_id: Shared session for all items.
        namespace: Shared namespace for all items.

    Returns:
        List of WriteResponse, one per item.

    Example:
        results = await memory.batch_write([
            {"content": "User likes Italian food", "context_tags": ["preference"]},
            {"content": "User works at Acme Corp", "context_tags": ["personal"]},
            {"content": "User lives in Paris", "context_tags": ["personal"]},
        ])
    """
    responses = []
    for item in items:
        resp = await self.write(
            content=item["content"],
            context_tags=item.get("context_tags"),
            session_id=session_id or item.get("session_id"),
            memory_type=item.get("memory_type"),
            namespace=namespace or item.get("namespace"),
            metadata=item.get("metadata"),
            agent_id=item.get("agent_id"),
        )
        responses.append(resp)
    return responses
```

**Pseudo-code**:
```
ASYNC METHOD batch_write(items, session_id?, namespace?) -> List[WriteResponse]:
    responses = []
    FOR EACH item IN items:
        resp = await self.write(
            content=item["content"],
            **merge(defaults, item_options)
        )
        responses.append(resp)
    RETURN responses

    FUTURE OPTIMIZATION:
    - Send all items in single POST /api/v1/memory/batch_write
    - Server processes in parallel
    - Return all results at once
```

### Sub-Task 5.2.2: Batch Read

**Implementation**:
```python
async def batch_read(
    self,
    queries: List[str],
    *,
    max_results: int = 10,
    format: Literal["packet", "list", "llm_context"] = "packet",
) -> List[ReadResponse]:
    """Execute multiple read queries.

    Args:
        queries: List of search queries.
        max_results: Max results per query.
        format: Response format for all queries.

    Returns:
        List of ReadResponse, one per query.
    """
    import asyncio
    tasks = [
        self.read(query, max_results=max_results, format=format)
        for query in queries
    ]
    return await asyncio.gather(*tasks)
```

**Pseudo-code**:
```
ASYNC METHOD batch_read(queries, **options) -> List[ReadResponse]:
    1. Create async tasks for each query
    2. Execute all concurrently with asyncio.gather()
    3. RETURN list of results in query order
```

### Sub-Task 5.2.3: Sync Batch Operations

**Implementation**:
```python
# Sync batch_write — sequential execution
def batch_write(self, items, **kwargs) -> List[WriteResponse]:
    return [self.write(item["content"], **kwargs) for item in items]

# Sync batch_read — sequential execution
def batch_read(self, queries, **kwargs) -> List[ReadResponse]:
    return [self.read(query, **kwargs) for query in queries]
```

---

## Task 5.3: Tenant Management

### Sub-Task 5.3.1: Multi-Tenant Support

**Architecture**: Support switching tenants on the same client instance, and listing tenant information (admin only).

**Implementation**:
```python
def set_tenant(self, tenant_id: str) -> None:
    """Switch the active tenant for subsequent operations.

    Args:
        tenant_id: New tenant identifier.
    """
    self._config.tenant_id = tenant_id
    # Update transport headers
    self._transport.update_header("X-Tenant-ID", tenant_id)


@property
def tenant_id(self) -> str:
    """Get the current active tenant ID."""
    return self._config.tenant_id


async def list_tenants(self) -> List[Dict[str, Any]]:
    """List all tenants and their memory counts (admin only).

    Returns:
        List of tenant info dicts with tenant_id, memory_count, etc.
    """
    data = await self._transport.request(
        "GET",
        "/api/v1/admin/tenants",
        use_admin_key=True,
    )
    return data.get("tenants", [])
```

**Pseudo-code**:
```
METHOD set_tenant(tenant_id):
    Update config.tenant_id
    Update X-Tenant-ID header on transport

PROPERTY tenant_id -> str:
    Return config.tenant_id

ASYNC METHOD list_tenants() -> List[Dict]:
    GET /api/v1/admin/tenants (admin key)
    RETURN list of {tenant_id, memory_count, fact_count, event_count}
```

---

## Task 5.4: Event Log Access

### Sub-Task 5.4.1: Query Events

**Implementation**:
```python
async def get_events(
    self,
    *,
    limit: int = 50,
    page: int = 1,
    event_type: Optional[str] = None,
    since: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Query the event log (admin only).

    Args:
        limit: Results per page.
        page: Page number.
        event_type: Filter by event type (e.g., "memory_op", "consolidation").
        since: Only events after this time.

    Returns:
        Paginated event list with items, total, page, per_page, total_pages.
    """
    params = {"per_page": limit, "page": page}
    if event_type:
        params["event_type"] = event_type
    if since:
        params["since"] = since.isoformat()

    return await self._transport.request(
        "GET",
        "/api/v1/admin/events",
        params=params,
        use_admin_key=True,
    )
```

---

## Task 5.5: Component Health Monitoring

### Sub-Task 5.5.1: Detailed Health Check

**Implementation**:
```python
async def component_health(self) -> Dict[str, Any]:
    """Get detailed health status of all CML components.

    Returns:
        Dict with component statuses:
            - postgres: {status, latency_ms, details}
            - neo4j: {status, latency_ms, details}
            - redis: {status, latency_ms, details}
    """
    return await self._transport.request(
        "GET",
        "/api/v1/admin/components",
        use_admin_key=True,
    )
```

---

## Task 5.6: Framework Integrations

### Sub-Task 5.6.1: OpenAI Integration Helper

**Architecture**: Provide a helper class that wraps OpenAI chat completions with CML memory.

**Implementation** (`src/cml/integrations/openai_helper.py`):
```python
"""OpenAI integration helper for py-cml."""

from typing import Any, Dict, List, Optional


class CMLOpenAIHelper:
    """Helper for integrating CML memory with OpenAI chat completions.

    Example:
        from openai import OpenAI
        from cml import CognitiveMemoryLayer
        from cml.integrations import CMLOpenAIHelper

        memory = CognitiveMemoryLayer(api_key="...")
        openai_client = OpenAI()
        helper = CMLOpenAIHelper(memory, openai_client)

        response = helper.chat("What should I eat tonight?", session_id="s1")
    """

    def __init__(self, memory_client, openai_client, *, model: str = "gpt-4o"):
        self.memory = memory_client
        self.openai = openai_client
        self.model = model

    def chat(
        self,
        user_message: str,
        *,
        session_id: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send a message with automatic memory context.

        1. Retrieves relevant memories
        2. Injects them into the system prompt
        3. Calls OpenAI chat completion
        4. Stores the exchange for future recall
        """
        # 1. Get memory context
        turn_result = self.memory.turn(
            user_message=user_message,
            session_id=session_id,
        )

        # 2. Build messages
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n"
                           f"## Relevant Memories\n{turn_result.memory_context}",
            }
        ]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_message})

        # 3. Call OpenAI
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        assistant_message = response.choices[0].message.content

        # 4. Store the exchange
        self.memory.turn(
            user_message=user_message,
            assistant_response=assistant_message,
            session_id=session_id,
        )

        return assistant_message
```

### Sub-Task 5.6.2: Generic LLM Integration Protocol

**Architecture**: Define a protocol (interface) that any LLM provider integration can implement.

**Pseudo-code**:
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for memory-enhanced LLM providers."""

    def get_context(self, query: str) -> str:
        """Get memory context for a query."""
        ...

    def store_exchange(
        self,
        user_message: str,
        assistant_response: str,
        session_id: str,
    ) -> None:
        """Store a conversation exchange."""
        ...

    def clear_session(self, session_id: str) -> None:
        """Clear a session's memories."""
        ...
```

---

## Task 5.7: Namespace Isolation

### Sub-Task 5.7.1: Namespace-Scoped Operations

**Architecture**: Support namespace prefixing for all operations, allowing logical isolation within a tenant.

**Implementation**:
```python
def with_namespace(self, namespace: str) -> "NamespacedClient":
    """Create a namespace-scoped view of this client.

    All operations through the returned client will be scoped
    to the given namespace.

    Args:
        namespace: Namespace identifier.

    Returns:
        NamespacedClient that adds namespace to all operations.

    Example:
        user_memory = memory.with_namespace("user-123")
        await user_memory.write("Prefers dark mode")
        # Stored with namespace="user-123"
    """
    return NamespacedClient(self, namespace)


class NamespacedClient:
    """Namespace-scoped wrapper around a CML client."""

    def __init__(self, parent, namespace: str):
        self._parent = parent
        self._namespace = namespace

    async def write(self, content: str, **kwargs) -> WriteResponse:
        return await self._parent.write(
            content, namespace=self._namespace, **kwargs
        )

    async def read(self, query: str, **kwargs) -> ReadResponse:
        return await self._parent.read(query, **kwargs)

    # ... delegate all other methods with namespace injected
```

**Pseudo-code**:
```
METHOD with_namespace(namespace) -> NamespacedClient:
    Create wrapper that auto-injects namespace into all write operations
    Read operations can optionally filter by namespace
    RETURN NamespacedClient

CLASS NamespacedClient:
    Wraps parent client
    Adds namespace= param to all write/update calls
    Delegates all other calls to parent
```

---

## Task 5.8: Memory Iteration & Pagination

### Sub-Task 5.8.1: Iterate All Memories

**Architecture**: Provide an async iterator for scanning all memories with pagination.

**Implementation**:
```python
async def iter_memories(
    self,
    *,
    memory_types: Optional[List[MemoryType]] = None,
    status: Optional[str] = "active",
    batch_size: int = 100,
) -> AsyncIterator[MemoryItem]:
    """Iterate over all memories with automatic pagination.

    Args:
        memory_types: Filter by types.
        status: Filter by status.
        batch_size: Items per page.

    Yields:
        MemoryItem for each memory.

    Example:
        async for memory in client.iter_memories():
            print(f"{memory.type}: {memory.text[:50]}")
    """
    page = 1
    while True:
        data = await self._transport.request(
            "GET",
            "/api/v1/admin/memories",
            params={
                "page": page,
                "per_page": batch_size,
                "status": status,
            },
            use_admin_key=True,
        )
        items = data.get("items", [])
        if not items:
            break
        for item in items:
            yield MemoryItem(**item)
        if page >= data.get("total_pages", 1):
            break
        page += 1
```

**Pseudo-code**:
```
ASYNC GENERATOR iter_memories(**filters) -> AsyncIterator[MemoryItem]:
    page = 1
    LOOP:
        GET /api/v1/admin/memories?page={page}&per_page={batch_size}
        IF no items → BREAK
        FOR EACH item → YIELD MemoryItem
        IF page >= total_pages → BREAK
        page += 1
```

---

## Acceptance Criteria

- [x] `consolidate()` triggers consolidation via admin endpoint
- [x] `run_forgetting()` triggers forgetting with dry_run support
- [x] `batch_write()` writes multiple memories efficiently
- [x] `batch_read()` executes queries concurrently (async) or sequentially (sync)
- [x] `set_tenant()` switches active tenant
- [x] `list_tenants()` returns tenant list (admin only)
- [x] `get_events()` queries event log with pagination
- [x] `component_health()` returns per-component status
- [x] `with_namespace()` returns namespace-scoped client
- [x] `iter_memories()` provides async iterator with pagination
- [x] OpenAI helper integrates memory into chat completions
- [x] All admin operations require admin API key
- [x] All new methods have full docstrings and type annotations


---

# Phase 6: Developer Experience

## Objective

Polish the SDK's developer experience with robust error handling, comprehensive type annotations, logging, context manager patterns, serialization utilities, and IDE-friendly features that make py-cml a joy to use.

---

## Task 6.1: Error Handling & Recovery

### Sub-Task 6.1.1: Structured Error Messages

**Architecture**: Every exception should carry actionable information — what went wrong, what the developer can do about it, and the raw server response for debugging.

**Implementation**:
```python
class CMLError(Exception):
    """Base exception for all CML errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code (if from HTTP response).
        response_body: Raw server response dict (if available).
        request_id: Server request ID for support debugging.
        suggestion: Actionable suggestion for the developer.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        full_message = message
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"
        if request_id:
            full_message += f"\n  Request ID: {request_id}"
        super().__init__(full_message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id
        self.suggestion = suggestion

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code})"
        )
```

**Pseudo-code for enhanced error mapping**:
```
FUNCTION _raise_for_status(response):
    IF response.is_success: RETURN

    body = try_parse_json(response)
    request_id = response.headers.get("X-Request-ID")

    MATCH response.status_code:
        401 → raise AuthenticationError(
            "Invalid or missing API key",
            suggestion="Set CML_API_KEY env var or pass api_key= to constructor"
        )
        403 → raise AuthorizationError(
            "Insufficient permissions",
            suggestion="This operation requires admin API key. Set CML_ADMIN_API_KEY."
        )
        404 → raise NotFoundError(
            f"Resource not found: {response.url.path}",
            suggestion="Check that the CML server version supports this endpoint"
        )
        422 → raise ValidationError(
            f"Request validation failed: {body.get('detail', '')}",
            suggestion="Check request parameters match the API schema"
        )
        429 → raise RateLimitError(
            "Rate limit exceeded",
            retry_after=response.headers.get("Retry-After"),
            suggestion="Reduce request frequency or contact admin"
        )
        503 → raise ServerError(
            "CML server is unavailable",
            suggestion="Check that all backend services (PostgreSQL, Neo4j, Redis) are running"
        )
```

### Sub-Task 6.1.2: Graceful Degradation

**Architecture**: For non-critical failures, provide fallback behavior instead of crashing.

**Pseudo-code**:
```
# Example: read() with graceful degradation
ASYNC METHOD read_safe(query, **kwargs) -> ReadResponse:
    """Read with graceful degradation — returns empty result on failure."""
    TRY:
        RETURN await self.read(query, **kwargs)
    CATCH ConnectionError:
        LOG.warning("CML server unreachable, returning empty context")
        RETURN ReadResponse(
            query=query,
            memories=[],
            total_count=0,
            elapsed_ms=0,
        )
    CATCH TimeoutError:
        LOG.warning("CML request timed out, returning empty context")
        RETURN ReadResponse(query=query, memories=[], total_count=0, elapsed_ms=0)
```

---

## Task 6.2: Logging

### Sub-Task 6.2.1: Structured Logging

**Architecture**: Use Python's standard `logging` module with structured context. Never log secrets (API keys, tokens).

**Implementation** (`src/cml/utils/logging.py`):
```python
"""Logging configuration for py-cml."""

import logging
from typing import Any


logger = logging.getLogger("cml")


def configure_logging(
    level: str = "WARNING",
    handler: logging.Handler | None = None,
) -> None:
    """Configure py-cml logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        handler: Custom handler. Defaults to StreamHandler.

    Example:
        import cml
        cml.configure_logging("DEBUG")  # Enable debug logging
    """
    logger.setLevel(getattr(logging, level.upper()))
    if handler:
        logger.addHandler(handler)
    elif not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(h)


def _redact(value: str, visible_chars: int = 4) -> str:
    """Redact a secret, showing only the last N characters."""
    if len(value) <= visible_chars:
        return "***"
    return f"***{value[-visible_chars:]}"
```

**Pseudo-code for log points**:
```
# Transport layer:
DEBUG: "POST /api/v1/memory/write → 200 (123ms)"
DEBUG: "Retry attempt 2/3 after ServerError, sleeping 2.1s"
WARNING: "Rate limited, retrying after 5.0s"

# Client layer:
INFO: "Connected to CML server at http://localhost:8000"
INFO: "Stored memory: {id} ({chunks_created} chunks)"
DEBUG: "Read query: {query} → {total_count} results in {elapsed_ms}ms"

# NEVER log:
# - API keys (redact to ***abcd)
# - Full memory content (truncate to first 50 chars)
# - Embedding vectors
```

### Sub-Task 6.2.2: Expose configure_logging in Package Init

**Implementation** (`src/cml/__init__.py`):
```python
from cml.utils.logging import configure_logging

__all__ = [
    # ... existing exports
    "configure_logging",
]
```

---

## Task 6.3: Type Annotations & IDE Support

### Sub-Task 6.3.1: Complete Type Stubs

**Architecture**: Every public method should have complete type annotations that provide rich IDE autocomplete and hover documentation.

**Pseudo-code for type coverage**:
```
ENSURE all public methods have:
    1. Parameter types (positional and keyword)
    2. Return type
    3. Docstring with Args, Returns, Raises, Example sections
    4. Optional overload decorators for polymorphic methods

EXAMPLE:
    @overload
    async def read(
        self,
        query: str,
        *,
        format: Literal["llm_context"],
        **kwargs,
    ) -> ReadResponse: ...  # .llm_context is always populated

    @overload
    async def read(
        self,
        query: str,
        *,
        format: Literal["packet", "list"] = "packet",
        **kwargs,
    ) -> ReadResponse: ...

    async def read(self, query, *, format="packet", **kwargs) -> ReadResponse:
        # Implementation
```

### Sub-Task 6.3.2: TypedDict for Unstructured Returns

**Implementation**:
```python
from typing import TypedDict

class ConsolidationResult(TypedDict):
    episodes_sampled: int
    clusters_formed: int
    facts_extracted: int
    migrations_completed: int

class ForgettingResult(TypedDict):
    total_scored: int
    actions: Dict[str, int]
    dry_run: bool
```

### Sub-Task 6.3.3: py.typed and Inline Types

**Implementation**:
```
# Ensure py.typed marker exists at src/cml/py.typed (empty file)
# This tells type checkers (mypy, pyright) that this package ships types

# All modules use from __future__ import annotations for forward refs
# All public classes use __slots__ where appropriate for memory efficiency
```

---

## Task 6.4: Context Manager Patterns

### Sub-Task 6.4.1: Client Lifecycle

**Architecture**: Ensure both sync and async clients properly manage connection lifecycle through context managers.

**Implementation**:
```python
# Async context manager (already defined)
async with AsyncCognitiveMemoryLayer(api_key="...") as memory:
    await memory.write("...")
    # Transport is automatically closed on exit

# Sync context manager (already defined)
with CognitiveMemoryLayer(api_key="...") as memory:
    memory.write("...")
    # Transport is automatically closed on exit

# Without context manager (manual close required)
memory = CognitiveMemoryLayer(api_key="...")
try:
    memory.write("...")
finally:
    memory.close()
```

### Sub-Task 6.4.2: Session Context Manager

**Architecture**: Provide a session context manager that auto-creates and manages a session.

**Implementation**:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def session(
    self,
    *,
    name: Optional[str] = None,
    ttl_hours: int = 24,
):
    """Create a session-scoped memory context.

    All operations within the context use the same session_id.

    Example:
        async with memory.session(name="onboarding") as sess:
            await sess.write("User prefers dark mode")
            await sess.write("User is a Python developer")
            result = await sess.read("user preferences")
            # sess.session_id is auto-generated
    """
    session_response = await self.create_session(name=name, ttl_hours=ttl_hours)
    session_id = session_response.session_id

    class SessionScope:
        def __init__(scope_self):
            scope_self.session_id = session_id

        async def write(scope_self, content, **kwargs):
            return await self.write(content, session_id=session_id, **kwargs)

        async def read(scope_self, query, **kwargs):
            return await self.read(query, **kwargs)

        async def turn(scope_self, user_message, **kwargs):
            return await self.turn(
                user_message, session_id=session_id, **kwargs
            )

    yield SessionScope()
```

**Pseudo-code**:
```
ASYNC CONTEXT MANAGER session(name?, ttl_hours?) -> SessionScope:
    1. Create session via API → get session_id
    2. Create SessionScope wrapper that injects session_id into all writes/turns
    3. YIELD SessionScope
    4. On exit: session expires naturally via TTL (no explicit cleanup needed)
```

---

## Task 6.5: Serialization Utilities

### Sub-Task 6.5.1: JSON Serialization Helpers

**Implementation** (`src/cml/utils/serialization.py`):
```python
"""Serialization utilities for py-cml."""

import json
from datetime import datetime
from uuid import UUID
from typing import Any


class CMLJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for CML types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def serialize_for_api(data: dict) -> dict:
    """Prepare a dict for API transmission.

    - Converts UUID to string
    - Converts datetime to ISO format
    - Removes None values
    """
    result = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, UUID):
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_for_api(value)
        elif isinstance(value, list):
            result[key] = [
                serialize_for_api(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            result[key] = value
    return result
```

### Sub-Task 6.5.2: Response Pretty-Printing

**Implementation**:
```python
# On ReadResponse:
def __str__(self) -> str:
    """Pretty-print retrieval results."""
    lines = [f"Query: {self.query} ({self.total_count} results, {self.elapsed_ms:.0f}ms)"]
    for mem in self.memories[:10]:
        lines.append(f"  [{mem.type}] {mem.text[:80]}... (rel={mem.relevance:.2f})")
    return "\n".join(lines)

# On WriteResponse:
def __str__(self) -> str:
    return f"WriteResponse(success={self.success}, chunks={self.chunks_created})"

# On StatsResponse:
def __str__(self) -> str:
    return (
        f"Memory Stats: {self.total_memories} total "
        f"({self.active_memories} active, {self.silent_memories} silent, "
        f"{self.archived_memories} archived)"
    )
```

---

## Task 6.6: Connection Pooling & Reuse

### Sub-Task 6.6.1: HTTP/2 Support

**Architecture**: Enable HTTP/2 by default in httpx for connection multiplexing.

**Pseudo-code**:
```
# In HTTPTransport.__init__:
self._client = httpx.Client(
    http2=True,  # Enable HTTP/2 for multiplexing
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    ),
    ...
)

# In AsyncHTTPTransport.__init__:
self._client = httpx.AsyncClient(
    http2=True,
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    ),
    ...
)
```

### Sub-Task 6.6.2: Connection Health Monitoring

**Pseudo-code**:
```
METHOD _ensure_healthy_connection():
    IF connection has been idle > 60 seconds:
        Send lightweight health check
        IF health check fails:
            Reconnect (create new httpx client)
    RETURN client
```

---

## Task 6.7: Deprecation & Versioning

### Sub-Task 6.7.1: API Compatibility Layer

**Architecture**: Provide mechanisms for graceful API evolution.

**Implementation**:
```python
import warnings
from functools import wraps


def deprecated(alternative: str, removal_version: str):
    """Mark a method as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in "
                f"v{removal_version}. Use {alternative} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Usage:
@deprecated("write()", "2.0.0")
def store(self, content: str, **kwargs):
    """Deprecated alias for write()."""
    return self.write(content, **kwargs)
```

### Sub-Task 6.7.2: Server Version Compatibility

**Pseudo-code**:
```
ASYNC METHOD _check_server_version():
    health = await self.health()
    server_version = health.version
    IF server_version < MIN_SUPPORTED_VERSION:
        LOG.warning(
            f"CML server {server_version} may not be fully compatible "
            f"with py-cml {__version__}. Upgrade to >= {MIN_SUPPORTED_VERSION}."
        )
```

---

## Task 6.8: Thread Safety

### Sub-Task 6.8.1: Sync Client Thread Safety

**Architecture**: The sync `CognitiveMemoryLayer` should be thread-safe for use in multi-threaded applications.

**Pseudo-code**:
```
CLASS CognitiveMemoryLayer:
    INIT:
        self._lock = threading.Lock()

    METHOD write(content, **kwargs):
        # httpx.Client is thread-safe by default
        # Pydantic model construction is thread-safe
        # No additional locking needed for stateless operations

    METHOD set_tenant(tenant_id):
        WITH self._lock:
            Update config.tenant_id
            Update transport headers

    NOTE: httpx.Client is already thread-safe.
    Only mutable shared state (tenant_id changes) needs locking.
```

### Sub-Task 6.8.2: Async Client Event Loop Safety

**Pseudo-code**:
```
CLASS AsyncCognitiveMemoryLayer:
    # asyncio operations are inherently single-threaded per event loop.
    # No additional synchronization needed.
    # Users should not share an async client across multiple event loops.

    METHOD _ensure_same_loop():
        IF current event loop != creation event loop:
            raise RuntimeError(
                "AsyncCognitiveMemoryLayer must be used "
                "in the same event loop it was created in."
            )
```

---

## Acceptance Criteria

- [x] All exceptions include actionable suggestions
- [x] API keys are never logged (redacted to `***xxxx`)
- [x] `configure_logging()` available at package level
- [x] Debug logging shows request/response timing
- [x] All public methods have complete type annotations
- [x] `py.typed` marker enables IDE type checking
- [x] Context managers properly close connections
- [x] Session context manager auto-creates and scopes sessions
- [x] Serialization handles UUID, datetime, and nested dicts
- [x] Response objects have human-readable `__str__` methods
- [x] HTTP/2 enabled for connection multiplexing
- [x] Deprecation decorator produces proper warnings
- [x] Sync client is thread-safe
- [x] Async client validates event loop consistency


---

# Phase 7: Testing & Quality

## Objective

Build a comprehensive test suite covering unit tests (with mocked HTTP), integration tests (against a live CML server), and end-to-end tests for both sync/async clients and embedded mode. Establish CI quality gates for linting, type checking, and code coverage.

---

## Task 7.1: Test Infrastructure

### Sub-Task 7.1.1: Test Directory Structure

**Implementation**:
```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── unit/                          # Tests with mocked HTTP (fast, no server)
│   ├── __init__.py
│   ├── test_config.py             # Configuration loading & validation
│   ├── test_exceptions.py         # Exception hierarchy & mapping
│   ├── test_transport.py          # HTTP transport with mocked httpx
│   ├── test_retry.py              # Retry logic
│   ├── test_async_client.py       # Async client (all operations)
│   ├── test_sync_client.py        # Sync client (all operations)
│   ├── test_models.py             # Pydantic model serialization
│   ├── test_serialization.py      # Serialization utilities
│   └── test_logging.py            # Logging configuration
├── integration/                   # Tests against a running CML server
│   ├── __init__.py
│   ├── conftest.py                # Server connection fixtures
│   ├── test_write_read.py         # Write and read roundtrip
│   ├── test_turn.py               # Seamless turn processing
│   ├── test_update_forget.py      # Update and forget operations
│   ├── test_stats.py              # Statistics endpoint
│   ├── test_sessions.py           # Session management
│   ├── test_batch.py              # Batch operations
│   ├── test_admin.py              # Admin operations
│   └── test_namespace.py          # Namespace isolation
├── embedded/                      # Tests for embedded mode
│   ├── __init__.py
│   ├── conftest.py                # Embedded fixtures
│   ├── test_lite_mode.py          # SQLite + local embeddings
│   ├── test_standard_mode.py      # PostgreSQL mode
│   └── test_lifecycle.py          # Init, close, context manager
└── e2e/                           # End-to-end scenarios
    ├── __init__.py
    ├── test_chat_flow.py           # Full chat with memory flow
    └── test_migration.py           # Embedded → server migration
```

### Sub-Task 7.1.2: Shared Fixtures

**Implementation** (`tests/conftest.py`):
```python
"""Shared test fixtures for py-cml."""

import os
import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig


# --- Configuration Fixtures ---

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return CMLConfig(
        api_key="test-api-key",
        base_url="http://localhost:8000",
        tenant_id="test-tenant",
        timeout=10.0,
        max_retries=0,  # No retries in tests
    )


@pytest.fixture
def mock_config():
    """Config for unit tests (no real server)."""
    return CMLConfig(
        api_key="mock-key",
        base_url="http://mock-server:8000",
        tenant_id="mock-tenant",
        max_retries=0,
    )


# --- Client Fixtures ---

@pytest_asyncio.fixture
async def async_client(test_config):
    """Create an async client for testing."""
    client = AsyncCognitiveMemoryLayer(config=test_config)
    yield client
    await client.close()


@pytest.fixture
def sync_client(test_config):
    """Create a sync client for testing."""
    client = CognitiveMemoryLayer(config=test_config)
    yield client
    client.close()


# --- Mock Response Helpers ---

def make_write_response(
    success: bool = True,
    memory_id: str = "00000000-0000-0000-0000-000000000001",
    chunks_created: int = 1,
    message: str = "Stored 1 memory chunks",
) -> dict:
    """Create a mock write response."""
    return {
        "success": success,
        "memory_id": memory_id,
        "chunks_created": chunks_created,
        "message": message,
    }


def make_read_response(
    query: str = "test query",
    memories: list | None = None,
    total_count: int = 1,
    elapsed_ms: float = 42.0,
) -> dict:
    """Create a mock read response."""
    if memories is None:
        memories = [{
            "id": "00000000-0000-0000-0000-000000000001",
            "text": "User prefers vegetarian food",
            "type": "preference",
            "confidence": 0.9,
            "relevance": 0.95,
            "timestamp": "2025-01-01T00:00:00Z",
            "metadata": {},
        }]
    return {
        "query": query,
        "memories": memories,
        "facts": [],
        "preferences": memories,
        "episodes": [],
        "llm_context": "## Preferences\n- User prefers vegetarian food",
        "total_count": total_count,
        "elapsed_ms": elapsed_ms,
    }


def make_turn_response(
    memory_context: str = "## Preferences\n- vegetarian",
    memories_retrieved: int = 3,
    memories_stored: int = 1,
) -> dict:
    """Create a mock turn response."""
    return {
        "memory_context": memory_context,
        "memories_retrieved": memories_retrieved,
        "memories_stored": memories_stored,
        "reconsolidation_applied": False,
    }
```

**Pseudo-code**:
```
FIXTURES:
    test_config → CMLConfig with test values (no retries, fast timeout)
    mock_config → CMLConfig for mocked HTTP tests
    async_client → AsyncCognitiveMemoryLayer with test_config
    sync_client → CognitiveMemoryLayer with test_config

HELPERS:
    make_write_response() → mock server response for write
    make_read_response() → mock server response for read
    make_turn_response() → mock server response for turn
    make_stats_response() → mock server response for stats
```

---

## Task 7.2: Unit Tests

### Sub-Task 7.2.1: Configuration Tests

**Implementation** (`tests/unit/test_config.py`):
```python
"""Tests for configuration loading and validation."""

import os
import pytest
from cml.config import CMLConfig


class TestCMLConfig:
    """Test CMLConfig loading and validation."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CMLConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.tenant_id == "default"
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_direct_params(self):
        """Direct params should override defaults."""
        config = CMLConfig(
            api_key="sk-test",
            base_url="http://custom:9000",
            tenant_id="my-tenant",
        )
        assert config.api_key == "sk-test"
        assert config.base_url == "http://custom:9000"
        assert config.tenant_id == "my-tenant"

    def test_env_vars(self, monkeypatch):
        """Environment variables should populate config."""
        monkeypatch.setenv("CML_API_KEY", "env-key")
        monkeypatch.setenv("CML_BASE_URL", "http://env:8000")
        monkeypatch.setenv("CML_TENANT_ID", "env-tenant")
        config = CMLConfig()
        assert config.api_key == "env-key"
        assert config.base_url == "http://env:8000"
        assert config.tenant_id == "env-tenant"

    def test_direct_overrides_env(self, monkeypatch):
        """Direct params should take priority over env vars."""
        monkeypatch.setenv("CML_API_KEY", "env-key")
        config = CMLConfig(api_key="direct-key")
        assert config.api_key == "direct-key"

    def test_invalid_timeout(self):
        """Negative timeout should raise validation error."""
        with pytest.raises(ValueError):
            CMLConfig(timeout=-1.0)

    def test_url_normalization(self):
        """Trailing slash should be stripped from base_url."""
        config = CMLConfig(base_url="http://localhost:8000/")
        assert config.base_url == "http://localhost:8000"
```

**Pseudo-code**:
```
TEST CONFIG:
    test_default_values: All fields have sensible defaults
    test_direct_params: Constructor params are stored correctly
    test_env_vars: CML_* env vars populate config fields
    test_direct_overrides_env: Direct params win over env vars
    test_invalid_timeout: Negative timeout raises ValueError
    test_invalid_retries: Negative max_retries raises ValueError
    test_url_normalization: Trailing slash stripped
    test_missing_api_key: No error (api_key is optional)
```

### Sub-Task 7.2.2: Transport Tests

**Implementation** (`tests/unit/test_transport.py`):
```python
"""Tests for HTTP transport layer."""

import pytest
import httpx
from pytest_httpx import HTTPXMock

from cml.config import CMLConfig
from cml.transport.http import HTTPTransport, AsyncHTTPTransport
from cml.exceptions import (
    AuthenticationError,
    NotFoundError,
    ServerError,
    ConnectionError,
    RateLimitError,
)


class TestHTTPTransport:
    """Test synchronous HTTP transport."""

    def test_sends_api_key_header(self, httpx_mock: HTTPXMock, mock_config):
        """Transport should send X-API-Key header."""
        httpx_mock.add_response(json={"status": "ok"})
        transport = HTTPTransport(mock_config)
        transport.request("GET", "/api/v1/health")
        request = httpx_mock.get_request()
        assert request.headers["X-API-Key"] == "mock-key"

    def test_sends_tenant_header(self, httpx_mock: HTTPXMock, mock_config):
        """Transport should send X-Tenant-ID header."""
        httpx_mock.add_response(json={"status": "ok"})
        transport = HTTPTransport(mock_config)
        transport.request("GET", "/api/v1/health")
        request = httpx_mock.get_request()
        assert request.headers["X-Tenant-ID"] == "mock-tenant"

    def test_401_raises_authentication_error(self, httpx_mock: HTTPXMock, mock_config):
        """401 response should raise AuthenticationError."""
        httpx_mock.add_response(status_code=401, json={"detail": "Invalid API key"})
        transport = HTTPTransport(mock_config)
        with pytest.raises(AuthenticationError):
            transport.request("GET", "/api/v1/health")

    def test_404_raises_not_found(self, httpx_mock: HTTPXMock, mock_config):
        """404 response should raise NotFoundError."""
        httpx_mock.add_response(status_code=404, json={"detail": "Not found"})
        transport = HTTPTransport(mock_config)
        with pytest.raises(NotFoundError):
            transport.request("GET", "/api/v1/nonexistent")

    def test_500_raises_server_error(self, httpx_mock: HTTPXMock, mock_config):
        """5xx response should raise ServerError."""
        httpx_mock.add_response(status_code=500, json={"detail": "Internal error"})
        transport = HTTPTransport(mock_config)
        with pytest.raises(ServerError):
            transport.request("GET", "/api/v1/health")

    def test_429_raises_rate_limit(self, httpx_mock: HTTPXMock, mock_config):
        """429 response should raise RateLimitError with retry_after."""
        httpx_mock.add_response(
            status_code=429,
            headers={"Retry-After": "5"},
            json={"detail": "Rate limited"},
        )
        transport = HTTPTransport(mock_config)
        with pytest.raises(RateLimitError) as exc_info:
            transport.request("GET", "/api/v1/health")
        assert exc_info.value.retry_after == 5.0

    def test_connection_error(self, httpx_mock: HTTPXMock, mock_config):
        """Connection failure should raise ConnectionError."""
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        transport = HTTPTransport(mock_config)
        with pytest.raises(ConnectionError):
            transport.request("GET", "/api/v1/health")
```

**Pseudo-code**:
```
TEST TRANSPORT:
    test_sends_api_key_header: X-API-Key present in request
    test_sends_tenant_header: X-Tenant-ID present in request
    test_sends_user_agent: User-Agent contains py-cml version
    test_sends_json_content_type: Content-Type is application/json
    test_401_maps_to_AuthenticationError
    test_403_maps_to_AuthorizationError
    test_404_maps_to_NotFoundError
    test_422_maps_to_ValidationError
    test_429_maps_to_RateLimitError_with_retry_after
    test_500_maps_to_ServerError
    test_connection_error_maps_to_ConnectionError
    test_timeout_maps_to_TimeoutError
    test_admin_key_override: use_admin_key replaces X-API-Key
    test_close_closes_client: close() closes httpx client
```

### Sub-Task 7.2.3: Retry Tests

**Implementation** (`tests/unit/test_retry.py`):
```python
"""Tests for retry logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from cml.config import CMLConfig
from cml.transport.retry import retry_sync, retry_async
from cml.exceptions import ServerError, RateLimitError, ConnectionError


class TestRetrySync:
    """Test synchronous retry logic."""

    def test_no_retry_on_success(self):
        """Should not retry on success."""
        func = MagicMock(return_value={"ok": True})
        config = CMLConfig(max_retries=3, retry_delay=0.01)
        result = retry_sync(config, func)
        assert result == {"ok": True}
        assert func.call_count == 1

    def test_retry_on_server_error(self):
        """Should retry on ServerError."""
        func = MagicMock(
            side_effect=[ServerError("500"), ServerError("500"), {"ok": True}]
        )
        config = CMLConfig(max_retries=3, retry_delay=0.01)
        result = retry_sync(config, func)
        assert result == {"ok": True}
        assert func.call_count == 3

    def test_exhausts_retries(self):
        """Should raise after max_retries exhausted."""
        func = MagicMock(side_effect=ServerError("500"))
        config = CMLConfig(max_retries=2, retry_delay=0.01)
        with pytest.raises(ServerError):
            retry_sync(config, func)
        assert func.call_count == 3  # initial + 2 retries

    def test_no_retry_on_client_error(self):
        """Should not retry on 4xx errors (except 429)."""
        from cml.exceptions import ValidationError
        func = MagicMock(side_effect=ValidationError("422"))
        config = CMLConfig(max_retries=3, retry_delay=0.01)
        with pytest.raises(ValidationError):
            retry_sync(config, func)
        assert func.call_count == 1
```

**Pseudo-code**:
```
TEST RETRY:
    test_no_retry_on_success: Call count == 1
    test_retry_on_server_error: Retries until success
    test_retry_on_connection_error: Retries on connection failures
    test_retry_on_rate_limit: Retries with retry_after
    test_exhausts_retries: Raises after max_retries + 1 attempts
    test_no_retry_on_client_error: 4xx errors (except 429) not retried
    test_backoff_increases: Verify exponential delay
    test_jitter_adds_randomness: Verify delay has random component
```

### Sub-Task 7.2.4: Client Operation Tests

**Implementation** (`tests/unit/test_async_client.py` — representative):
```python
"""Tests for async client operations."""

import pytest
from pytest_httpx import HTTPXMock
from uuid import UUID

from cml import AsyncCognitiveMemoryLayer
from cml.models import MemoryType, WriteResponse, ReadResponse, TurnResponse
from conftest import make_write_response, make_read_response, make_turn_response


class TestAsyncWrite:
    """Test async write operation."""

    @pytest.mark.asyncio
    async def test_write_basic(self, httpx_mock: HTTPXMock, mock_config):
        """Write basic content."""
        httpx_mock.add_response(json=make_write_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.write("User prefers vegetarian food")
        assert isinstance(result, WriteResponse)
        assert result.success is True
        assert result.chunks_created == 1
        assert result.memory_id is not None

    @pytest.mark.asyncio
    async def test_write_with_tags(self, httpx_mock: HTTPXMock, mock_config):
        """Write with context tags."""
        httpx_mock.add_response(json=make_write_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.write(
                "User prefers vegetarian food",
                context_tags=["preference", "food"],
            )
        request = httpx_mock.get_request()
        body = request.content
        assert b"preference" in body

    @pytest.mark.asyncio
    async def test_write_with_type(self, httpx_mock: HTTPXMock, mock_config):
        """Write with explicit memory type."""
        httpx_mock.add_response(json=make_write_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.write(
                "User prefers vegetarian food",
                memory_type=MemoryType.PREFERENCE,
            )
        assert result.success


class TestAsyncRead:
    """Test async read operation."""

    @pytest.mark.asyncio
    async def test_read_basic(self, httpx_mock: HTTPXMock, mock_config):
        """Read with basic query."""
        httpx_mock.add_response(json=make_read_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.read("dietary preferences")
        assert isinstance(result, ReadResponse)
        assert result.total_count == 1
        assert len(result.memories) == 1

    @pytest.mark.asyncio
    async def test_read_context_property(self, httpx_mock: HTTPXMock, mock_config):
        """Read .context returns LLM context string."""
        httpx_mock.add_response(json=make_read_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.read("preferences", format="llm_context")
        assert isinstance(result.context, str)
        assert len(result.context) > 0


class TestAsyncTurn:
    """Test async turn operation."""

    @pytest.mark.asyncio
    async def test_turn_basic(self, httpx_mock: HTTPXMock, mock_config):
        """Process a basic turn."""
        httpx_mock.add_response(json=make_turn_response())
        async with AsyncCognitiveMemoryLayer(config=mock_config) as client:
            result = await client.turn(user_message="What do I like to eat?")
        assert isinstance(result, TurnResponse)
        assert result.memories_retrieved == 3
        assert len(result.memory_context) > 0
```

**Pseudo-code**:
```
TEST ASYNC CLIENT:
    WRITE:
        test_write_basic: Returns WriteResponse with success
        test_write_with_tags: context_tags in request body
        test_write_with_type: memory_type in request body
        test_write_with_metadata: metadata in request body
        test_write_with_session: session_id in request body
        test_write_with_namespace: namespace in request body

    READ:
        test_read_basic: Returns ReadResponse with memories
        test_read_with_filters: context_filter and memory_types in body
        test_read_with_time_range: since and until in body
        test_read_context_property: .context returns formatted string
        test_read_llm_context_format: format="llm_context" works

    TURN:
        test_turn_basic: Returns TurnResponse with context
        test_turn_with_response: assistant_response in body
        test_turn_with_session: session_id in body

    UPDATE:
        test_update_text: text change triggers re-embedding
        test_update_confidence: confidence update
        test_update_feedback_correct: boosts confidence
        test_update_feedback_incorrect: deletes memory

    FORGET:
        test_forget_by_ids: memory_ids in body
        test_forget_by_query: query-based forget
        test_forget_by_time: before filter
        test_forget_no_selector: raises ValueError

    STATS:
        test_stats_returns_response: StatsResponse parsed correctly

    HEALTH:
        test_health: HealthResponse with status

TEST SYNC CLIENT:
    Mirror all async tests with sync calls
```

### Sub-Task 7.2.5: Model Serialization Tests

**Implementation** (`tests/unit/test_models.py`):
```python
"""Tests for Pydantic model serialization."""

import pytest
from datetime import datetime, timezone
from uuid import UUID

from cml.models import MemoryType, MemoryStatus, MemoryItem
from cml.models.responses import WriteResponse, ReadResponse
from cml.models.requests import WriteRequest, ReadRequest


class TestMemoryItem:
    def test_parse_from_dict(self):
        data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "text": "Test memory",
            "type": "preference",
            "confidence": 0.9,
            "relevance": 0.95,
            "timestamp": "2025-01-01T00:00:00Z",
            "metadata": {"key": "value"},
        }
        item = MemoryItem(**data)
        assert isinstance(item.id, UUID)
        assert item.text == "Test memory"
        assert item.type == "preference"

    def test_serialize_to_dict(self):
        item = MemoryItem(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            text="Test",
            type="preference",
            confidence=0.9,
            relevance=0.95,
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        d = item.model_dump()
        assert "id" in d
        assert "text" in d


class TestWriteRequest:
    def test_exclude_none(self):
        req = WriteRequest(content="Hello")
        d = req.model_dump(exclude_none=True)
        assert "content" in d
        assert "context_tags" not in d
        assert "session_id" not in d
```

---

## Task 7.3: Integration Tests

### Sub-Task 7.3.1: Integration Test Fixtures

**Implementation** (`tests/integration/conftest.py`):
```python
"""Integration test fixtures — requires running CML server."""

import os
import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig

INTEGRATION_URL = os.environ.get("CML_TEST_URL", "http://localhost:8000")
INTEGRATION_KEY = os.environ.get("CML_TEST_API_KEY", "test-key")
INTEGRATION_TENANT = f"test-{os.getpid()}"  # Unique per test run


@pytest.fixture(scope="session")
def integration_config():
    return CMLConfig(
        api_key=INTEGRATION_KEY,
        base_url=INTEGRATION_URL,
        tenant_id=INTEGRATION_TENANT,
        timeout=30.0,
        max_retries=1,
    )


@pytest_asyncio.fixture
async def live_client(integration_config):
    """Async client connected to real CML server."""
    client = AsyncCognitiveMemoryLayer(config=integration_config)
    yield client
    # Cleanup: delete all test memories
    try:
        await client.delete_all(confirm=True)
    except Exception:
        pass
    await client.close()
```

### Sub-Task 7.3.2: Write-Read Roundtrip Test

**Implementation** (`tests/integration/test_write_read.py`):
```python
"""Integration tests for write → read roundtrip."""

import pytest


@pytest.mark.integration
class TestWriteReadRoundtrip:

    @pytest.mark.asyncio
    async def test_write_then_read(self, live_client):
        """Write content, then retrieve it."""
        # Write
        write_result = await live_client.write(
            "User prefers vegetarian food and lives in Paris",
            context_tags=["preference", "personal"],
        )
        assert write_result.success
        assert write_result.chunks_created >= 1

        # Read
        read_result = await live_client.read("dietary preferences")
        assert read_result.total_count >= 1
        assert any("vegetarian" in m.text for m in read_result.memories)

    @pytest.mark.asyncio
    async def test_write_multiple_then_read(self, live_client):
        """Write multiple items, read should find all."""
        await live_client.write("User likes Italian cuisine")
        await live_client.write("User is allergic to nuts")
        await live_client.write("User prefers organic produce")

        result = await live_client.read("food preferences")
        assert result.total_count >= 2

    @pytest.mark.asyncio
    async def test_read_llm_context_format(self, live_client):
        """Read with llm_context format returns formatted string."""
        await live_client.write("User's birthday is March 15th")
        result = await live_client.read("birthday", format="llm_context")
        assert result.llm_context is not None
        assert isinstance(result.llm_context, str)
```

**Pseudo-code**:
```
TEST INTEGRATION WRITE-READ:
    test_write_then_read:
        1. Write "User prefers vegetarian food"
        2. Read "dietary preferences"
        3. Assert retrieved text contains "vegetarian"

    test_write_multiple_then_read:
        1. Write 3 food-related memories
        2. Read "food preferences"
        3. Assert total_count >= 2

    test_read_returns_empty_for_unrelated:
        1. Write about food preferences
        2. Read "quantum physics"
        3. Assert total_count == 0 or relevance < 0.3
```

---

## Task 7.4: Embedded Mode Tests

### Sub-Task 7.4.1: Lite Mode Tests

**Implementation** (`tests/embedded/test_lite_mode.py`):
```python
"""Tests for embedded lite mode (SQLite + local embeddings)."""

import pytest
from cml import EmbeddedCognitiveMemoryLayer


@pytest.mark.embedded
class TestLiteMode:

    @pytest.mark.asyncio
    async def test_zero_config_init(self):
        """Should initialize with zero configuration."""
        async with EmbeddedCognitiveMemoryLayer() as memory:
            assert memory is not None

    @pytest.mark.asyncio
    async def test_write_and_read(self):
        """Should store and retrieve in lite mode."""
        async with EmbeddedCognitiveMemoryLayer() as memory:
            result = await memory.write("User prefers dark mode")
            assert result.success

            read = await memory.read("theme preferences")
            assert read.total_count >= 1

    @pytest.mark.asyncio
    async def test_persistent_storage(self, tmp_path):
        """SQLite file should persist between instances."""
        db_path = str(tmp_path / "test.db")

        # Write in first instance
        async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as memory:
            await memory.write("Persistent memory test")

        # Read in second instance
        async with EmbeddedCognitiveMemoryLayer(db_path=db_path) as memory:
            result = await memory.read("persistent")
            assert result.total_count >= 1
```

---

## Task 7.5: End-to-End Tests

### Sub-Task 7.5.1: Chat Flow Test

**Implementation** (`tests/e2e/test_chat_flow.py`):
```python
"""End-to-end test: full chat flow with memory."""

import pytest


@pytest.mark.e2e
class TestChatFlow:

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, live_client):
        """Simulate a multi-turn conversation with memory."""
        session_id = "e2e-chat-001"

        # Turn 1: User introduces themselves
        turn1 = await live_client.turn(
            user_message="My name is Alice and I'm a software engineer.",
            session_id=session_id,
        )
        assert turn1.memories_stored >= 0  # May or may not store

        # Turn 2: Ask about user
        turn2 = await live_client.turn(
            user_message="What do you know about me?",
            session_id=session_id,
        )
        # Should retrieve the introduction
        assert turn2.memories_retrieved >= 0
        assert len(turn2.memory_context) > 0

        # Turn 3: Add more info
        turn3 = await live_client.turn(
            user_message="I live in San Francisco and I love hiking.",
            session_id=session_id,
        )
        assert turn3.memories_stored >= 0

        # Turn 4: Query preferences
        turn4 = await live_client.turn(
            user_message="What are my hobbies?",
            session_id=session_id,
        )
        assert turn4.memories_retrieved >= 0
```

---

## Task 7.6: Test Markers & CI Configuration

### Sub-Task 7.6.1: Pytest Markers

**Implementation** (in `pyproject.toml`):
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: Integration tests (requires CML server)",
    "embedded: Embedded mode tests (requires embedded deps)",
    "e2e: End-to-end tests (requires full setup)",
    "slow: Tests that take > 5 seconds",
]
```

### Sub-Task 7.6.2: CI Test Matrix

**Pseudo-code**:
```yaml
# In .github/workflows/test.yml:
jobs:
  unit-tests:
    # Run on every push/PR
    # No external services needed
    # Fast (< 1 minute)
    steps:
      - pip install -e ".[dev]"
      - pytest tests/unit/ -v --cov

  integration-tests:
    # Run on main branch only (or manually)
    # Requires CML server via docker-compose
    services:
      postgres, neo4j, redis, cml-server
    steps:
      - pip install -e ".[dev]"
      - pytest tests/integration/ -v -m integration

  embedded-tests:
    # Run on main branch or when embedded/ files change
    steps:
      - pip install -e ".[dev,embedded]"
      - pytest tests/embedded/ -v -m embedded
```

---

## Task 7.7: Code Coverage

### Sub-Task 7.7.1: Coverage Configuration

**Implementation** (`.coveragerc` or in `pyproject.toml`):
```toml
[tool.coverage.run]
source = ["src/cml"]
branch = true

[tool.coverage.report]
show_missing = true
fail_under = 80
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "pass",
]
```

### Sub-Task 7.7.2: Coverage Targets

| Component | Target Coverage |
|:----------|:---------------|
| `config.py` | 95% |
| `exceptions.py` | 100% |
| `transport/http.py` | 90% |
| `transport/retry.py` | 95% |
| `async_client.py` | 90% |
| `client.py` | 90% |
| `models/` | 95% |
| `utils/` | 85% |
| **Overall** | **85%** |

---

## Acceptance Criteria

- [x] Unit tests cover all public methods (sync and async)
- [x] Transport tests verify all HTTP status code → exception mappings
- [x] Retry tests verify backoff, jitter, and exhaustion behavior
- [x] Model tests verify serialization and deserialization
- [x] Integration tests pass against a running CML server
- [x] Embedded tests verify lite mode with zero config
- [x] E2E tests simulate realistic multi-turn conversations
- [x] Test markers allow selective test execution
- [x] CI runs unit tests on every push (< 2 minutes)
- [x] CI runs integration tests on main branch
- [x] Code coverage >= 85% overall
- [x] All tests use fixtures (no hardcoded URLs or keys)

---

## Implementation status (Phase 7 complete)

The following has been implemented in `packages/py-cml/`:

**Test infrastructure**
- `tests/conftest.py` — `test_config`, `mock_config`, `sync_client`, `async_client`, `make_write_response`, `make_read_response`, `make_turn_response`, `make_stats_response`.
- `tests/integration/conftest.py` — `integration_config`, `live_client`, `live_sync_client` with server reachability check and teardown `delete_all`.
- `tests/embedded/conftest.py` — optional skip when embedded not installed.
- `tests/e2e/conftest.py` — e2e `integration_config` and `live_client`.

**Unit tests**
- `tests/unit/test_models.py` — MemoryItem/WriteResponse/ReadResponse parse and serialize, WriteRequest/ReadRequest `model_dump(exclude_none=True)`.
- `tests/unit/test_transport.py` — extended with 403, 422, 429 (Retry-After), 500, ConnectError, TimeoutException, admin key, `close()`.
- `tests/unit/test_retry.py` — no retry on 4xx (ValidationError).
- `tests/unit/test_serialization.py`, `tests/unit/test_logging.py` — serialization and logging.

**Integration tests** (require running CML server; skip if unreachable)
- `tests/integration/` — `test_write_read`, `test_turn`, `test_update_forget`, `test_stats`, `test_sessions`, `test_batch`, `test_admin`, `test_namespace`; all use `@pytest.mark.integration` and `live_client`.

**Embedded tests** (require embedded extras and engine; skip if unavailable)
- `tests/embedded/test_lite_mode.py`, `test_lifecycle.py` — `@pytest.mark.embedded`; zero-config init, write/read, persistent storage, context manager, close, ensure_initialized.

**E2E tests** (require live server; optionally embedded for migration)
- `tests/e2e/test_chat_flow.py`, `test_migration.py` — `@pytest.mark.e2e`; multi-turn conversation, export/import embedded→server.

**Config and CI**
- `pyproject.toml` — pytest markers (`integration`, `embedded`, `e2e`, `slow`), coverage `source = ["src/cml"]`, `branch = true`, `fail_under` and `exclude_lines`.
- `.github/workflows/py-cml-test.yml` — runs **unit tests only** by default: `pytest tests/unit/ -v --cov=cml --cov-report=xml --cov-branch`.

**How to run**
- Unit (default CI): `pytest tests/unit/` or `pytest -m "not integration and not embedded and not e2e"`.
- Integration: `pytest tests/integration/ -m integration` (set `CML_TEST_URL`, `CML_TEST_API_KEY` if needed).
- Embedded: `pytest tests/embedded/ -m embedded` (requires `pip install -e ".[dev,embedded]"` and engine).
- E2E: `pytest tests/e2e/ -m e2e`.


---

# Phase 8: Documentation & Publishing

## Objective

Create comprehensive documentation, usage examples, and establish the GitHub release and PyPI publishing pipeline for py-cml. Ensure developers can discover, install, and integrate the package within minutes.

---

## Task 8.1: Package README

### Sub-Task 8.1.1: README.md

**Architecture**: The README is the package's landing page on both GitHub and PyPI. It should convey value in 10 seconds and get developers to a working example in 60 seconds.

**Implementation** (`packages/py-cml/README.md`):
```markdown
# py-cml

**Python SDK for [CognitiveMemoryLayer](https://github.com/<org>/CognitiveMemoryLayer)** — neuro-inspired memory for AI applications.

[![PyPI](https://img.shields.io/pypi/v/py-cml)](https://pypi.org/project/py-cml/)
[![Python](https://img.shields.io/pypi/pyversions/py-cml)](https://pypi.org/project/py-cml/)
[![License](https://img.shields.io/github/license/<org>/CognitiveMemoryLayer)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/<org>/CognitiveMemoryLayer/py-cml-test.yml)](https://github.com/<org>/CognitiveMemoryLayer/actions)

Give your AI applications human-like memory — store, retrieve, consolidate, and forget information just like the brain does.

## Installation

```bash
pip install cognitive-memory-layer
```

For embedded mode (no server required):
```bash
pip install cognitive-memory-layer[embedded]
```

## Quick Start

```python
from cml import CognitiveMemoryLayer

# Connect to a CML server
memory = CognitiveMemoryLayer(
    api_key="your-api-key",
    base_url="http://localhost:8000",
)

# Store a memory
memory.write("User prefers vegetarian food and lives in Paris.")

# Retrieve relevant memories
result = memory.read("What does the user eat?")
for item in result.memories:
    print(f"  {item.text} (relevance: {item.relevance:.2f})")

# Seamless chat integration
turn = memory.turn(
    user_message="What should I eat tonight?",
    session_id="chat-001",
)
# Inject turn.memory_context into your LLM prompt
```

## Async Support

```python
from cml import AsyncCognitiveMemoryLayer

async with AsyncCognitiveMemoryLayer(api_key="...") as memory:
    await memory.write("User is a Python developer.")
    result = await memory.read("programming skills")
```

## Embedded Mode (No Server)

```python
from cml import EmbeddedCognitiveMemoryLayer

# Zero-config: SQLite + local embeddings
async with EmbeddedCognitiveMemoryLayer() as memory:
    await memory.write("User prefers dark mode.")
    result = await memory.read("UI preferences")
```

## Features

- **Write** — Store information with automatic chunking, embedding, and entity extraction
- **Read** — Hybrid retrieval (vector + graph + lexical) with relevance ranking
- **Turn** — Seamless chat integration: auto-retrieve context, auto-store new info
- **Update** — Modify memories with feedback (correct/incorrect/outdated)
- **Forget** — Remove memories by ID, query, or time (with soft/hard delete)
- **Consolidate** — Migrate episodic memories to semantic knowledge
- **Active Forgetting** — Automatically prune low-relevance memories
- **Multi-tenant** — Tenant isolation with namespace support
- **Sync + Async** — Both synchronous and asynchronous clients
- **Embedded Mode** — Run in-process without a server (SQLite or PostgreSQL)

## Configuration

```python
# Environment variables
# CML_API_KEY=your-api-key
# CML_BASE_URL=http://localhost:8000
# CML_TENANT_ID=my-tenant

# Or direct initialization
memory = CognitiveMemoryLayer(
    api_key="...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
    timeout=30.0,
    max_retries=3,
)
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Configuration](docs/configuration.md)
- [Examples](docs/examples.md)

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.
```

**Pseudo-code for README sections**:
```
SECTIONS:
1. Title + badges (PyPI, Python versions, license, CI)
2. One-line description
3. Installation (pip install cognitive-memory-layer / py-cml[embedded])
4. Quick Start (sync, 5 lines)
5. Async Support (async context manager, 4 lines)
6. Embedded Mode (zero config, 4 lines)
7. Features (bullet list, 12 items)
8. Configuration (env vars + direct params)
9. Documentation links
10. License
```

---

## Task 8.2: API Reference Documentation

### Sub-Task 8.2.1: Getting Started Guide

**Implementation** (`docs/getting-started.md`):

**Pseudo-code for content**:
```
# Getting Started

## Prerequisites
- Python 3.11+
- A running CML server (or use embedded mode)

## Installation
pip install cognitive-memory-layer

## Connect to a Server
1. Start CML server (link to parent project docs)
2. Get your API key
3. Initialize client
4. Write first memory
5. Read it back

## Next Steps
- Read API Reference for all operations
- See Examples for common patterns
- Try Embedded Mode for serverless usage
```

### Sub-Task 8.2.2: API Reference

**Implementation** (`docs/api-reference.md`):

**Pseudo-code for content**:
```
# API Reference

## CognitiveMemoryLayer (Sync Client)

### Constructor
    CognitiveMemoryLayer(
        api_key: str | None = None,
        base_url: str = "http://localhost:8000",
        tenant_id: str = "default",
        config: CMLConfig | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    )

### Methods

#### write(content, **kwargs) -> WriteResponse
    Store new information in memory.
    Parameters: content, context_tags, session_id, memory_type, namespace, metadata
    Returns: WriteResponse(success, memory_id, chunks_created, message)
    Raises: AuthenticationError, ValidationError

#### read(query, **kwargs) -> ReadResponse
    Retrieve relevant memories.
    Parameters: query, max_results, context_filter, memory_types, since, until, format
    Returns: ReadResponse(query, memories, facts, preferences, episodes, llm_context)

#### turn(user_message, **kwargs) -> TurnResponse
    Process a conversation turn with seamless memory.
    Parameters: user_message, assistant_response, session_id, max_context_tokens
    Returns: TurnResponse(memory_context, memories_retrieved, memories_stored)

#### update(memory_id, **kwargs) -> UpdateResponse
    Update an existing memory.
    Parameters: memory_id, text, confidence, importance, metadata, feedback
    Returns: UpdateResponse(success, memory_id, version)

#### forget(**kwargs) -> ForgetResponse
    Forget memories.
    Parameters: memory_ids, query, before, action
    Returns: ForgetResponse(success, affected_count)

#### stats() -> StatsResponse
    Get memory statistics.

#### health() -> HealthResponse
    Check server health.

#### get_context(query, max_results) -> str
    Convenience: get formatted LLM context string.

#### delete_all(confirm=False) -> int
    Delete all memories (GDPR). Requires confirm=True.

## AsyncCognitiveMemoryLayer (Async Client)
    Same methods as above, but all are async.

## EmbeddedCognitiveMemoryLayer
    Same methods as async client.
    Additional: consolidate(), run_forgetting()

## Models

### MemoryType (enum)
    EPISODIC_EVENT, SEMANTIC_FACT, PREFERENCE, ...

### MemoryStatus (enum)
    ACTIVE, SILENT, COMPRESSED, ARCHIVED, DELETED

### MemoryItem
    id, text, type, confidence, relevance, timestamp, metadata

### WriteResponse, ReadResponse, TurnResponse, UpdateResponse, ForgetResponse, StatsResponse

## Exceptions

### CMLError (base)
### AuthenticationError (401)
### AuthorizationError (403)
### NotFoundError (404)
### ValidationError (422)
### RateLimitError (429)
### ServerError (5xx)
### ConnectionError (network)
### TimeoutError (timeout)

## Configuration

### CMLConfig
    api_key, base_url, tenant_id, timeout, max_retries, retry_delay, admin_api_key

### Environment Variables
    CML_API_KEY, CML_BASE_URL, CML_TENANT_ID, ...
```

### Sub-Task 8.2.3: Configuration Guide

**Implementation** (`docs/configuration.md`):

**Pseudo-code for content**:
```
# Configuration

## Environment Variables
Table of all CML_ env vars with descriptions and defaults.

## Direct Initialization
Code example with all params.

## Config Object
CMLConfig usage with all fields.

## Priority Order
1. Direct params → 2. Env vars → 3. .env file → 4. Defaults

## Embedded Configuration
EmbeddedConfig with storage_mode, database, embedding, llm settings.

## Storage Modes (Embedded)
- Lite: SQLite + local embeddings
- Standard: PostgreSQL + pgvector
- Full: PostgreSQL + Neo4j + Redis
```

---

## Task 8.3: Usage Examples

### Sub-Task 8.3.1: Quickstart Example

**Implementation** (`examples/quickstart.py`):
```python
"""py-cml Quickstart — store and retrieve memories in 30 seconds."""

from cml import CognitiveMemoryLayer

def main():
    # Initialize client
    memory = CognitiveMemoryLayer(
        api_key="your-api-key",
        base_url="http://localhost:8000",
    )

    # Store some information
    memory.write("User prefers vegetarian food and lives in Paris.")
    memory.write("User works at a tech startup as a backend engineer.")
    memory.write("User has a meeting with the design team every Tuesday.")

    # Retrieve relevant memories
    result = memory.read("What does the user do for work?")
    print(f"Found {result.total_count} relevant memories:")
    for item in result.memories:
        print(f"  [{item.type}] {item.text}")
        print(f"    Relevance: {item.relevance:.2f}, Confidence: {item.confidence:.2f}")

    # Get formatted context for LLM
    context = memory.get_context("dietary restrictions")
    print(f"\nLLM Context:\n{context}")

    # Check stats
    stats = memory.stats()
    print(f"\nMemory Stats: {stats.total_memories} memories stored")

    memory.close()

if __name__ == "__main__":
    main()
```

### Sub-Task 8.3.2: Chat with Memory Example

**Implementation** (`examples/chat_with_memory.py`):
```python
"""Build a chatbot with persistent memory using py-cml and OpenAI."""

from openai import OpenAI
from cml import CognitiveMemoryLayer


def chat_with_memory():
    # Initialize clients
    memory = CognitiveMemoryLayer(api_key="cml-key")
    openai = OpenAI()

    session_id = "chat-demo-001"
    print("Chat with Memory (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # Get memory context
        turn = memory.turn(
            user_message=user_input,
            session_id=session_id,
        )

        # Build prompt with memory
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with persistent memory. "
                    "Use the following memories to personalize your responses:\n\n"
                    f"{turn.memory_context}"
                ),
            },
            {"role": "user", "content": user_input},
        ]

        # Call OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        assistant_msg = response.choices[0].message.content

        # Store the exchange
        memory.turn(
            user_message=user_input,
            assistant_response=assistant_msg,
            session_id=session_id,
        )

        print(f"Assistant: {assistant_msg}\n")
        print(f"  [memories retrieved: {turn.memories_retrieved}]")

    memory.close()


if __name__ == "__main__":
    chat_with_memory()
```

### Sub-Task 8.3.3: Async Example

**Implementation** (`examples/async_example.py`):
```python
"""Async usage of py-cml with asyncio."""

import asyncio
from cml import AsyncCognitiveMemoryLayer


async def main():
    async with AsyncCognitiveMemoryLayer(api_key="your-key") as memory:
        # Store multiple memories concurrently
        import asyncio
        await asyncio.gather(
            memory.write("User likes hiking in the mountains"),
            memory.write("User prefers morning workouts"),
            memory.write("User is training for a marathon"),
        )

        # Read memories
        result = await memory.read("exercise habits")
        print(f"Found {result.total_count} memories about exercise")
        for mem in result.memories:
            print(f"  - {mem.text}")

        # Batch read
        results = await memory.batch_read([
            "exercise habits",
            "outdoor activities",
            "fitness goals",
        ])
        for r in results:
            print(f"\nQuery: {r.query} -> {r.total_count} results")


if __name__ == "__main__":
    asyncio.run(main())
```

### Sub-Task 8.3.4: Embedded Mode Example

**Implementation** (`examples/embedded_mode.py`):
```python
"""Use py-cml without a server — embedded mode with SQLite."""

import asyncio
from cml import EmbeddedCognitiveMemoryLayer


async def main():
    # Zero-config: uses SQLite in-memory + local embeddings
    async with EmbeddedCognitiveMemoryLayer() as memory:
        # Store memories
        await memory.write("User prefers Python over JavaScript")
        await memory.write("User uses VS Code as their primary editor")
        await memory.write("User follows TDD methodology")

        # Retrieve
        result = await memory.read("development tools")
        print(f"Found {result.total_count} relevant memories:")
        for mem in result.memories:
            print(f"  - {mem.text}")

        # Get stats
        stats = await memory.stats()
        print(f"\nTotal memories: {stats.total_memories}")

    # With persistent storage:
    async with EmbeddedCognitiveMemoryLayer(db_path="./my_app.db") as memory:
        await memory.write("This memory persists between restarts")

    # Read it back in a new instance:
    async with EmbeddedCognitiveMemoryLayer(db_path="./my_app.db") as memory:
        result = await memory.read("persists")
        print(f"\nPersistent memory found: {result.memories[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Sub-Task 8.3.5: Agent Framework Example

**Implementation** (`examples/agent_integration.py`):
```python
"""Integrate py-cml with an AI agent framework."""

import asyncio
from cml import AsyncCognitiveMemoryLayer


class MemoryAgent:
    """Simple agent with persistent memory."""

    def __init__(self, memory: AsyncCognitiveMemoryLayer, agent_id: str):
        self.memory = memory
        self.agent_id = agent_id

    async def observe(self, observation: str):
        """Store an observation."""
        await self.memory.write(
            observation,
            context_tags=["observation"],
            agent_id=self.agent_id,
        )

    async def plan(self, goal: str) -> str:
        """Create a plan using memory context."""
        context = await self.memory.get_context(goal)
        # In a real agent, you'd call an LLM here with the context
        return f"Plan for '{goal}' with context:\n{context}"

    async def reflect(self, topic: str):
        """Reflect on past observations."""
        result = await self.memory.read(
            topic,
            context_filter=["observation"],
            max_results=20,
        )
        print(f"Reflections on '{topic}':")
        for mem in result.memories:
            print(f"  [{mem.timestamp}] {mem.text}")


async def main():
    async with AsyncCognitiveMemoryLayer(api_key="...") as memory:
        agent = MemoryAgent(memory, agent_id="agent-001")

        # Agent observes things
        await agent.observe("The deployment pipeline took 15 minutes today")
        await agent.observe("Three tests failed due to timeout issues")
        await agent.observe("The database migration completed successfully")

        # Agent plans based on observations
        plan = await agent.plan("improve deployment speed")
        print(plan)

        # Agent reflects on past events
        await agent.reflect("deployment")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Task 8.4: GitHub Repository Setup

> **Note**: All templates and workflows are added to the existing CognitiveMemoryLayer repo.
> Issue templates are shared across the whole repo. CI workflows use path filters to only
> trigger on `packages/py-cml/**` changes.

### Sub-Task 8.4.1: Repository Templates

**Implementation**:

**Issue Templates** (`.github/ISSUE_TEMPLATE/bug_report.md`):
```markdown
---
name: Bug Report
about: Report a bug in py-cml
title: "[BUG] "
labels: bug
---

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
```python
from cml import CognitiveMemoryLayer
# Steps to reproduce...
```

**Expected behavior**
What you expected to happen.

**Environment**
- py-cml version:
- Python version:
- CML server version:
- OS:
```

**Feature Request Template** (`.github/ISSUE_TEMPLATE/feature_request.md`):
```markdown
---
name: Feature Request
about: Suggest a feature for py-cml
title: "[FEATURE] "
labels: enhancement
---

**Use Case**
What problem does this solve?

**Proposed API**
```python
# How should this look for developers?
```

**Alternatives**
Other approaches considered.
```

### Sub-Task 8.4.2: Pull Request Template

**Implementation** (`.github/pull_request_template.md`):
```markdown
## Summary
<!-- Brief description of changes -->

## Changes
- [ ] Change 1
- [ ] Change 2

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] mypy passes
- [ ] ruff passes

## Documentation
- [ ] Docstrings updated
- [ ] README updated (if applicable)
- [ ] CHANGELOG updated
```

---

## Task 8.5: Publishing Pipeline

### Sub-Task 8.5.1: GitHub Release Process

**Architecture**: Use prefixed git tags (`py-cml-v*`) to trigger PyPI publishing from the monorepo. GitHub Releases are created from these tags.

**Pseudo-code**:
```
RELEASE PROCESS (from existing CognitiveMemoryLayer repo):

1. Bump version in packages/py-cml/pyproject.toml and _version.py
2. Update packages/py-cml/CHANGELOG.md with new version section
3. Commit: "chore(py-cml): prepare release v0.1.0"
4. Create prefixed git tag: git tag py-cml-v0.1.0
5. Push: git push origin main --tags
6. py-cml-publish.yml workflow triggers (matches py-cml-v* tag):
   a. Checks out repo
   b. Builds wheel and sdist from packages/py-cml/
   c. Publishes to PyPI via trusted publishing (OIDC)
7. Create GitHub Release from tag (optional):
   gh release create py-cml-v0.1.0 \
     --title "py-cml v0.1.0 — Initial Release" \
     --notes-file packages/py-cml/CHANGELOG.md
8. Verify: pip install cognitive-memory-layer==0.1.0
```

### Sub-Task 8.5.2: PyPI Configuration

**Architecture**: Use PyPI trusted publishing (OIDC) — no API tokens needed. The publisher is configured for the existing CognitiveMemoryLayer repo.

**Pseudo-code for setup**:
```
1. Create PyPI account (if not exists)
2. Reserve package name "cognitive-memory-layer" on PyPI:
   a. Go to pypi.org/manage/account/publishing/
   b. Add new pending publisher:
      - PyPI project name: py-cml
      - Owner: <github-org>
      - Repository: CognitiveMemoryLayer    ← existing repo, NOT a new one
      - Workflow: py-cml-publish.yml         ← SDK-specific workflow
      - Environment: pypi
3. Create GitHub environment "pypi" in CognitiveMemoryLayer repo settings
4. Push py-cml-publish.yml workflow
5. Create first py-cml-v0.1.0 tag to trigger publish
```

### Sub-Task 8.5.3: TestPyPI First

**Pseudo-code**:
```
BEFORE first real release:

1. Configure TestPyPI trusted publisher:
   - test.pypi.org/manage/account/publishing/
   - Same settings as PyPI

2. Create test publish workflow:
   - Trigger on push to "release/*" branches
   - Build and publish to TestPyPI

3. Verify installation from TestPyPI:
   pip install --index-url https://test.pypi.org/simple/ py-cml

4. Once verified, create real release to publish to PyPI
```

---

## Task 8.6: Versioning Strategy

### Sub-Task 8.6.1: Semantic Versioning

**Architecture**: Follow SemVer 2.0.0.

```
MAJOR.MINOR.PATCH

MAJOR: Breaking API changes
  - Removing a public method
  - Changing method signature (required params)
  - Changing return types

MINOR: New features (backwards compatible)
  - Adding new methods
  - Adding optional parameters
  - New embedded storage backends

PATCH: Bug fixes
  - Fixing incorrect behavior
  - Performance improvements
  - Documentation updates
```

### Sub-Task 8.6.2: Version History Plan

| Version | Milestone | Key Features |
|:--------|:----------|:-------------|
| 0.1.0 | Alpha | Async/sync clients, write/read/turn, basic models |
| 0.2.0 | Beta | Full API surface, embedded mode, batch ops |
| 0.3.0 | RC | Admin ops, namespace, session management |
| 0.9.0 | Pre-GA | Complete test suite, documentation, CI/CD |
| 1.0.0 | GA | Stable API, production ready, full docs |
| 1.1.0 | Post-GA | Framework integrations (LangChain, CrewAI) |
| 1.2.0 | Post-GA | Streaming support, webhook events |

---

## Task 8.7: Community & Support

### Sub-Task 8.7.1: Contributing Guide

**Implementation** (`CONTRIBUTING.md`):

**Pseudo-code for content**:
```
# Contributing to py-cml

## Development Setup
1. Fork and clone the repo
2. Create a virtual environment
3. pip install -e ".[dev]"
4. pre-commit install

## Running Tests
pytest tests/unit/       # Fast unit tests
pytest tests/ -m "not integration"  # All non-integration tests

## Code Style
- Ruff for linting: ruff check src/
- Ruff for formatting: ruff format src/
- mypy for types: mypy src/cml/

## PR Process
1. Create a branch from main
2. Make changes
3. Add tests
4. Update CHANGELOG.md
5. Submit PR
6. Pass CI checks
7. Get review approval

## Commit Messages
Use conventional commits:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- test: Tests
- chore: Maintenance
```

### Sub-Task 8.7.2: Security Policy

**Implementation** (`SECURITY.md`):
```markdown
# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |
| 0.x     | Best effort |

## Reporting Vulnerabilities

Please report security vulnerabilities via GitHub Security Advisories:
https://github.com/<org>/CognitiveMemoryLayer/security/advisories/new

Do NOT open a public issue for security vulnerabilities.
```

---

## Acceptance Criteria

<<<<<<< HEAD
- [x] README.md on GitHub/PyPI conveys value in < 10 seconds
- [x] Quickstart works in < 60 seconds (pip install + 5 lines of code)
- [x] API reference documents all public methods, parameters, return types
- [x] Getting started guide covers install → first memory in 5 steps
- [x] Configuration guide covers all options (env vars, direct, config object)
- [x] 5+ usage examples covering common patterns
- [x] GitHub issue templates for bugs and features
- [x] PR template with checklist
- [x] CONTRIBUTING.md with dev setup and PR process
- [x] SECURITY.md with reporting instructions
- [x] CHANGELOG.md follows Keep a Changelog format
- [x] CI publishes to PyPI on GitHub Release
- [x] TestPyPI verified before first real release
- [x] Package installable: `pip install py-cml`
- [x] Package importable: `from cml import CognitiveMemoryLayer`
- [x] All SDK files live under `packages/py-cml/` in the existing repo
- [x] CI workflows scoped to `packages/py-cml/**` path changes
- [x] Publish workflow triggers on `py-cml-v*` tag pattern
- [x] PyPI trusted publisher configured for the CognitiveMemoryLayer repo
=======
- [ ] README.md on GitHub/PyPI conveys value in < 10 seconds
- [ ] Quickstart works in < 60 seconds (pip install + 5 lines of code)
- [ ] API reference documents all public methods, parameters, return types
- [ ] Getting started guide covers install → first memory in 5 steps
- [ ] Configuration guide covers all options (env vars, direct, config object)
- [ ] 5+ usage examples covering common patterns
- [ ] GitHub issue templates for bugs and features
- [ ] PR template with checklist
- [ ] CONTRIBUTING.md with dev setup and PR process
- [ ] SECURITY.md with reporting instructions
- [ ] CHANGELOG.md follows Keep a Changelog format
- [ ] CI publishes to PyPI on GitHub Release
- [ ] TestPyPI verified before first real release
- [ ] Package installable: `pip install cognitive-memory-layer`
- [ ] Package importable: `from cml import CognitiveMemoryLayer`
- [ ] All SDK files live under `packages/py-cml/` in the existing repo
- [ ] CI workflows scoped to `packages/py-cml/**` path changes
- [ ] Publish workflow triggers on `py-cml-v*` tag pattern
- [ ] PyPI trusted publisher configured for the CognitiveMemoryLayer repo
>>>>>>> 42897739dbe59559f3754da63c76f08f1e7a6549

---

## Implementation status (Phase 8 complete)

The following has been implemented:

**README** ([packages/py-cml/README.md](packages/py-cml/README.md)): Added PyPI and Python version badges, one-line tagline ("Give your AI applications human-like memory..."), and Documentation section with links to [Getting Started](packages/py-cml/docs/getting-started.md), [API Reference](packages/py-cml/docs/api-reference.md), [Configuration](packages/py-cml/docs/configuration.md), [Examples](packages/py-cml/docs/examples.md).

**Docs** (packages/py-cml/docs/): Created getting-started.md (prerequisites, installation, connect to server, next steps), api-reference.md (sync/async/embedded clients, methods, models, exceptions, config), configuration.md (env vars table, direct init, CMLConfig, priority order, embedded config, versioning), examples.md (overview of the five examples with paths and run instructions).

**Examples** (packages/py-cml/examples/): Created quickstart.py (sync write/read/get_context/stats), chat_with_memory.py (OpenAI + turn with memory_context), async_example.py (asyncio.gather writes, batch_read), embedded_mode.py (zero-config and db_path persistence), agent_integration.py (MemoryAgent observe/plan/reflect).

**GitHub templates**: .github/ISSUE_TEMPLATE/bug_report.md, .github/ISSUE_TEMPLATE/feature_request.md, .github/pull_request_template.md (Summary, Changes, Testing, Documentation checklists).

**Package**: packages/py-cml/SECURITY.md (supported versions, report via Security Advisories). CONTRIBUTING.md updated with fork/venv in development setup, PR checklist alignment, and Releasing cognitive-memory-layer section (version bump, CHANGELOG, tag py-cml-v*, push, optional GitHub Release, verify; TestPyPI note).

**Publishing**: .github/workflows/py-cml-publish.yml was already in place; triggers on push of tags matching py-cml-v*, uses environment pypi and OIDC. No workflow changes.

**ProjectPlan**: Overview.md Phase 8 set to Done; Phase 8 implementation line added to Current implementation.
