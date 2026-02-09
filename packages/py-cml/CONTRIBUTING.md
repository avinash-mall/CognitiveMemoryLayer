# Contributing to py-cml

Thanks for your interest in the CognitiveMemoryLayer Python SDK. This guide covers development setup and workflow for the `packages/py-cml` package.

## Development setup

1. Fork and clone the CognitiveMemoryLayer repository (or work in a branch if you have write access).
2. Create a virtual environment (recommended): `python -m venv .venv` then activate it.
3. Install the package in editable mode with dev dependencies.

From the **repository root**:

```bash
pip install -e "packages/py-cml[dev]"
```

Or from inside the package:

```bash
cd packages/py-cml
pip install -e ".[dev]"
```

This installs the package in editable mode with dev dependencies (pytest, ruff, mypy, pre-commit). Run `pre-commit install` to enable hooks.

## Running tests

From `packages/py-cml`:

**Unit tests (default; no server required):**

```bash
pytest tests/unit/ -v
pytest tests/unit/ -v --cov=cml --cov-report=term-missing --cov-branch
```

Unit tests cover config, exceptions, retry logic, transport (including 403/422/429/500 and connection/timeout mapping), models (serialization), client health and memory operations (mocked), serialization, and logging. CI runs only unit tests on every push.

**Integration tests** (require a running CML server):

```bash
export CML_TEST_URL=http://localhost:8000   # optional
export CML_TEST_API_KEY=your-key            # optional
pytest tests/integration/ -v -m integration
```

**Embedded tests** (require `pip install -e ".[dev,embedded]"` and CML engine):

```bash
pytest tests/embedded/ -v -m embedded
```

**E2E tests** (require live server):

```bash
pytest tests/e2e/ -v -m e2e
```

**Run everything except integration/embedded/e2e:**

```bash
pytest -m "not integration and not embedded and not e2e" -v
```

Shared fixtures live in `tests/conftest.py` (`test_config`, `mock_config`, `sync_client`, `async_client`, and mock response helpers). Integration tests use `live_client` from `tests/integration/conftest.py` and skip if the server is unreachable.

## Code style and type checking

- **Ruff** — linting and formatting:
  ```bash
  ruff check src/ tests/
  ruff format src/ tests/
  ```

- **mypy** — type checking (strict):
  ```bash
  mypy src/cml/
  ```

Configuration for ruff and mypy lives in `pyproject.toml`.

## Pre-commit hooks

Install and run pre-commit (from `packages/py-cml` or repo root with config path):

```bash
pre-commit install
pre-commit run --all-files
```

Hooks run ruff (check + format), mypy, and generic checks (trailing whitespace, YAML/TOML, etc.).

## Pull request process

1. Create a branch from `main`, make changes in `packages/py-cml/`.
2. Add or update tests; ensure `pytest tests/unit/`, `mypy src/cml/`, and `ruff check src/ tests/` pass.
3. Update docstrings, README, or CHANGELOG as applicable (see PR template checklist).
4. Open a PR. CI runs py-cml tests and lint when paths under `packages/py-cml/**` or the workflow file change.
5. Get review approval and merge.

## Commit message conventions

- `feat(cml): ...` — new feature
- `fix(cml): ...` — bug fix
- `docs(cml): ...` — documentation
- `test(cml): ...` — tests
- `chore(py-cml): ...` — build, CI, tooling

## Releasing py-cml

Releases are published to PyPI when a tag matching `py-cml-v*` is pushed. The workflow uses PyPI trusted publishing (OIDC); no API tokens in the repo.

1. **Bump version** in `packages/py-cml/pyproject.toml` and `packages/py-cml/src/cml/_version.py`.
2. **Update CHANGELOG.md** — add a new `## [X.Y.Z] - YYYY-MM-DD` section under Unreleased and move entries.
3. **Commit:** e.g. `chore(py-cml): prepare release v0.1.0`
4. **Tag:** `git tag py-cml-v0.1.0` (use the same version number).
5. **Push:** `git push origin main --tags`. The `py-cml-publish.yml` workflow runs and publishes to PyPI.
6. **Optional:** Create a GitHub Release from the tag and paste the CHANGELOG section for that version.
7. **Verify:** `pip install cognitive-memory-layer==0.1.0` and `from cml import CognitiveMemoryLayer`.

**TestPyPI:** Before the first production release, you can configure a trusted publisher for TestPyPI and run the same workflow from a branch (e.g. `release/*`) or trigger a test publish to verify the package builds and installs.

## General repository guidelines

For broader contribution guidelines (code of conduct, issue templates, repository structure), see the root [CONTRIBUTING.md](../../CONTRIBUTING.md) of the CognitiveMemoryLayer repository.
