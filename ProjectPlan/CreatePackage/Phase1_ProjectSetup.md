# Phase 1: Project Setup & Packaging

## Objective

Establish the py-cml package directory within the existing CognitiveMemoryLayer monorepo, with a modern Python packaging structure, its own build system, development tooling, and CI/CD pipeline scoped to the SDK.

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
│       │       ├── exceptions.py      # Exception hierarchy (placeholder)
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
name = "py-cml"
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
7. Users can install: pip install py-cml
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
   - pip install py-cml
   - pip install py-cml[embedded]
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

- [ ] `packages/py-cml/` directory created in existing CognitiveMemoryLayer repo
- [ ] `pip install -e "packages/py-cml[dev]"` succeeds from repo root
- [ ] `import cml` works without errors
- [ ] `python -m build` (from `packages/py-cml/`) produces valid wheel and sdist
- [ ] Ruff linting passes with zero errors
- [ ] mypy strict mode passes
- [ ] CI workflows trigger only on `packages/py-cml/**` file changes
- [ ] CI workflows use `working-directory: packages/py-cml`
- [ ] Publish workflow triggers on `py-cml-v*` tag pattern
- [ ] SDK `README.md` at `packages/py-cml/README.md` includes installation and quickstart
- [ ] `.gitignore`, `.editorconfig` in place
- [ ] `py.typed` marker file present at `packages/py-cml/src/cml/py.typed`
- [ ] Server's existing `pyproject.toml`, `src/`, `tests/` remain untouched
