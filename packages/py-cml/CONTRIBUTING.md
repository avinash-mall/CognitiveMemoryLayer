# Contributing to cognitive-memory-layer

Thanks for your interest in the CognitiveMemoryLayer Python SDK. This guide covers development setup and workflow for the `packages/py-cml` package (published on PyPI as **cognitive-memory-layer**).

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

**Integration and E2E tests** (require a running CML server):

1. From the **repository root**, start the API: `docker compose -f docker/docker-compose.yml up -d postgres neo4j redis api`
2. Use the same API key for server and tests: the repo **.env.example** sets `AUTH__API_KEY=test-key` and `AUTH__ADMIN_API_KEY=test-key`. Copy to `.env` or set these in your `.env` so the API accepts `test-key`. Set `CML_BASE_URL` (or `CML_TEST_URL`) for the client; no hardcoded URLs. Alternatively set `CML_TEST_API_KEY` (and optionally `CML_TEST_URL`) to match your server’s key.
3. From `packages/py-cml`: `pytest tests/integration/ tests/e2e/ -v -m "integration or e2e"`

If `CML_TEST_API_KEY` is unset, the integration and e2e conftests load the repo root `.env` and use `AUTH__API_KEY` / `AUTH__ADMIN_API_KEY`, so one key works for both. If the server is unreachable, tests are skipped.

**Embedded tests** (require `pip install -e ".[dev,embedded]"` and CML engine):

```bash
pytest tests/embedded/ -v -m embedded
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
4. Open a PR. CI runs cognitive-memory-layer tests and lint when paths under `packages/py-cml/**` or the workflow file change.
5. Get review approval and merge.

## Commit message conventions

- `feat(cml): ...` — new feature
- `fix(cml): ...` — bug fix
- `docs(cml): ...` — documentation
- `test(cml): ...` — tests
- `chore(py-cml): ...` — build, CI, tooling

## Publishing the package to PyPI

Releases are published to PyPI when a tag matching `py-cml-v*` is pushed. The workflow (`.github/workflows/py-cml.yml`) uses **PyPI trusted publishing (OIDC)**; no API tokens are stored in the repo.

### Prerequisites (one-time)

- **PyPI:** Add a [trusted publisher](https://docs.pypi.org/trusted-publishers/) (or pending publisher) for the project `cognitive-memory-layer`:
  - **Repository:** `avinash-mall/CognitiveMemoryLayer`
  - **Workflow name:** `py-cml.yml`
  - **Environment name:** `pypi`
- **GitHub:** Create an environment named **`pypi`** under **Settings → Environments** (no secrets required for trusted publishing).

### Before each release

1. **Bump version** in `packages/py-cml/pyproject.toml` and `packages/py-cml/src/cml/_version.py` (e.g. `0.1.0` → `0.1.1`).
2. **Update CHANGELOG.md** — add a `## [X.Y.Z] - YYYY-MM-DD` section and move entries from Unreleased.
3. **Commit and push to `main`:** e.g. `chore(py-cml): prepare release v0.1.1`

### Ways to publish

**Option A — From the command line (recommended)**

From the repository root, after the version bump is pushed to `main`:

```bash
git pull origin main
git tag py-cml-v0.1.0   # use the same version as in pyproject.toml
git push origin py-cml-v0.1.0
```

The tag push triggers the workflow; the **Build and publish to PyPI** job runs and uploads the package. Check **Actions** for the run triggered by the tag and confirm the job succeeded.

**Option B — From GitHub Actions (no local tag needed)**

1. Go to **Actions** → **py-cml: CI and release** → **Run workflow**.
2. Choose branch **main** (or the branch you pushed the version bump to).
3. Enter **Version to release** (e.g. `0.1.0`). Leave it empty only if you just want to run lint/test/build.
4. Click **Run workflow**.
5. **Two runs:** The first run (manual) only runs **Create release tag** and pushes the tag; lint/test/build/publish are skipped. The **tag push triggers a second run** where only **Build and publish to PyPI** runs. In the Actions list, find the run triggered by the tag and confirm that job succeeded.

**Option C — From GitHub Releases**

1. After pushing the version bump to `main`, go to **Releases** → **Draft a new release**.
2. Choose **Tag:** create a new tag `py-cml-v0.1.1` from `main`.
3. Set the release title (e.g. `v0.1.1`) and paste the CHANGELOG section. Publish the release.
4. Creating the tag (via the release) pushes it; the workflow runs and publishes to PyPI.

### After publishing

- **Optional:** If you used Option A or B, create a **GitHub Release** from the new tag and paste the CHANGELOG section.
- **Verify:** `pip install cognitive-memory-layer==0.1.1` then `python -c "from cml import CognitiveMemoryLayer; print('OK')"`.

### Troubleshooting

- **PyPI still shows "0 projects"**  
  Check that the **tag-triggered** run exists and that **Build and publish to PyPI** completed successfully. If using Option A (command line), the run should appear shortly after you push the tag. If using Option B (GitHub Actions) and the second run is missing, use Option A to push the tag from your machine and confirm the run appears.
- **Pending publishers:** Use only the publisher for **workflow `py-cml.yml`** and **environment `pypi`**. If you have a pending publisher for `py-cml-publish.yml`, remove it (that workflow was merged into `py-cml.yml`).

### TestPyPI (optional)

To try the release flow without publishing to production PyPI: add a trusted publisher for **TestPyPI** with the same workflow and environment, then push a tag (e.g. `py-cml-v0.1.0a1`) or use **Run workflow** with a pre-release version. Install with `pip install -i https://test.pypi.org/simple/ cognitive-memory-layer==0.1.0a1`.

## General repository guidelines

For broader contribution guidelines (code of conduct, issue templates, repository structure), see the root [CONTRIBUTING.md](../../CONTRIBUTING.md) of the CognitiveMemoryLayer repository.
