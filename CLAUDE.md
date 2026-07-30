# CognitiveMemoryLayer — agent instructions

Neuro-inspired memory system for LLMs. FastAPI server (`src/`) + Python SDK
(`packages/py-cml/`). PostgreSQL+pgvector, Neo4j, Redis. Env config via
pydantic-settings with `__` nesting (see `.env`, untracked).

## `docs/`

Two kinds of file live here.

**Agent working memory.** Read `docs/STATE.md` at the start of a session; keep it
current as you work (active work, load-bearing architecture facts, known issues).
Put anything too big to inline into `docs/<topic>.md` and link it from the STATE.md
notes index. Delete notes when they stop being true — stale memory is worse than none.

**Durable reference docs**, currently `docs/usage.md` (server API, endpoints,
configuration reference — other docs link into its anchors, so don't rename its
headings casually). These are *not* subject to the delete-when-stale rule: fix them
instead. If you add one, list it in the STATE.md notes index so it is discoverable.

## Rules

1. **Fully offline.** Zero CDN/external calls except the configured LLM
   endpoint. No `https://` in built assets — fonts, mermaid, scripts are all
   bundled.
2. **No stubs.** No TODO/FIXME/NotImplementedError in shipped code.
3. **Fix the class, never the case.** A bug arrives as one document, one
   chat, one file — that is the evidence, not the specification. Ship the
   general rule and let the reported case be one instance of it.
4. **Full test after every major change — before committing.**
5. **Ponytail.** Fix root causes, don't over-build, no speculative extras.

## Commands

```bash
# Server (plain uvicorn; docker compose exists but local dev runs bare)
set -a && source .env && set +a
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Full test = hermetic unit suite + live suites against a running server
uv run pytest tests/unit -q                                        # hermetic (LLM off, mock embeddings)
uv run pytest tests/integration tests/e2e packages/py-cml/tests -q # needs server on :8000

# Lint (CI enforces all three)
uv run ruff check src tests packages/py-cml/src
uv run ruff format --check src packages/py-cml/src tests packages/py-cml/tests
uv run mypy packages/py-cml/src/cml
```

`tests/conftest.py` forces `FEATURES__USE_LLM_ENABLED=false` and mock
embeddings so unit tests never need a model server. Integration/e2e/py-cml
suites hit the live server configured by `.env`.
