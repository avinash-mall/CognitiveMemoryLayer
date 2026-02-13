# Contributing to Cognitive Memory Layer

Thank you for your interest in contributing. This document outlines how to get set up and submit changes.

## Development setup

1. **Clone and install**

   ```bash
   git clone https://github.com/your-org/CognitiveMemoryLayer.git
   cd CognitiveMemoryLayer
   python -m venv .venv
   source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
   pip install -e ".[dev]"
   ```

2. **Environment**

   Copy `.env.example` to `.env` and fill in your local database URLs and API keys (see [Quick Start](README.md#quick-start) for services).

3. **Tests**

   ```bash
   pytest
   ```

   Run with coverage: `pytest --cov=src --cov-report=html`

## Code standards

- **Style:** Follow the existing style; the project uses standard Python conventions. Format with `ruff format` / `black` if configured.
- **Imports:** Prefer explicit public API. Use `src.*` package exports where available (e.g. `from src.core import get_settings, MemoryRecord`).
- **Types:** Use type hints for new functions and modules where practical.

## Submitting changes

1. **Branch:** Create a feature branch from `main` (e.g. `feature/your-feature` or `fix/issue-description`).
2. **Tests:** Ensure all tests pass and add or update tests for new behavior. From repo root: `pytest tests` (see [tests/README.md](tests/README.md)).
3. **Commit:** Use clear, concise commit messages.
4. **Pull request:** Open a PR against `main`. Describe what changed and why; reference any related issues.
5. **README badges:** If you changed the project version (e.g. `VERSION` file) or added/removed tests, run `python scripts/update_readme_badges.py` from the repo root to refresh the version and test count in the README badges, then commit the updated README(s).

## Project structure

- `src/` — Application code: `core`, `api`, `memory`, `storage`, `retrieval`, `consolidation`, `forgetting`, `reconsolidation`, `extraction`, `utils`.
- `packages/py-cml/` — Python SDK; see [packages/py-cml/CONTRIBUTING.md](packages/py-cml/CONTRIBUTING.md) for dev setup and publishing.
- `tests/` — Pytest tests (unit, integration, e2e); see [tests/README.md](tests/README.md).
- `scripts/` — Dev scripts (e.g. `update_readme_badges.py` to refresh version and test count in README badges).
- `ProjectPlan/` — Design docs and phase plans (e.g. [ActiveCML/Issues.md](ProjectPlan/ActiveCML/Issues.md), [CreatePackage/CreatePackageStatus.md](ProjectPlan/CreatePackage/CreatePackageStatus.md)).

For full architecture and API details, see [README](README.md) and [Usage Documentation](ProjectPlan/UsageDocumentation.md).

## Publishing the Python SDK

To publish the Python SDK, see [packages/py-cml/CONTRIBUTING.md#publishing-the-package-to-pypi](packages/py-cml/CONTRIBUTING.md#publishing-the-package-to-pypi).

## Questions

Open an issue for bugs, feature ideas, or questions. For security concerns, see [SECURITY.md](SECURITY.md).
