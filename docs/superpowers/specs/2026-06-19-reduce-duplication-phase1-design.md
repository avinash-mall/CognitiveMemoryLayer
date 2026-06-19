# Reduce Duplication — Phase 1: Safe in-`src/` consolidations

**Date:** 2026-06-19
**Status:** Proposed (awaiting sign-off)
**Goal:** Reduce code duplication with **zero behavior change** and **no package-boundary impact**. This is phase 1 of a 3-phase program (phase 2 = shared API contracts; phase 3 = sync/async client codegen), each shipped separately.

## Background

A repo-wide `symilar` survey found 46 duplicate clusters. The largest (SDK↔server schemas ~217 lines; sync/async client twins ~120 lines) cross the published `py-cml` package boundary and are deferred to phases 2–3. Phase 1 consolidates only duplication that lives **entirely within `src/`** and can be fully verified locally.

## Scope (two consolidations)

### 1. Shared loose-data parsing helpers → `src/utils/parsing.py` (new module)

Three helpers are copy-pasted across `src/`:

- `_strip_markdown_fences` — **verbatim** duplicate in `src/extraction/entity_extractor.py:18` and `src/extraction/relation_extractor.py:19`.
- `_safe_float` / `_safe_int` — two *variants*: `unified_write_extractor.py` (default-returning: `(value, default) -> float`) and `modelpack.py:245` (None-returning: `(value) -> float | None`).

**New module `src/utils/parsing.py`** exposes:

```python
from typing import Any, overload

def strip_markdown_fences(text: str) -> str: ...   # moved verbatim

@overload
def safe_float(value: Any, default: float) -> float: ...
@overload
def safe_float(value: Any, default: None = ...) -> float | None: ...
def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# safe_int: identical shape with int()
```

The `@overload` pair unifies both variants: callers passing a default get a guaranteed `float`; callers passing none get `float | None`.

**Consumers updated to import from the new module:**
- `entity_extractor.py`, `relation_extractor.py` → `strip_markdown_fences` (drop local copies)
- `unified_write_extractor.py` → `safe_float`/`safe_int` (drop the local `_safe_float`/`_safe_int` added during the recent bug-fix)
- `modelpack.py` → `safe_float`/`safe_int` (drop local copies)

**Layering:** `src/utils/parsing.py` is a leaf (stdlib + typing only). `src/extraction/*` already imports `src/utils/*`, and `modelpack.py` is a `src/utils` sibling — no cycles.

**One intentional micro-change:** `modelpack.py`'s helpers currently `except Exception`; the shared version catches `(TypeError, ValueError)`. For the JSON/artifact-metadata values these parse, those are the only realistic failures, so behavior is preserved in practice. Noted for the reviewer.

> Out of scope (phase 3): the `cml/modeling/train.py` copies of `_safe_float`/`_safe_int` live in the separate `py-cml` package and cannot import `src/`.

### 2. `AsyncMicroBatcher` base → `src/utils/micro_batcher.py` (new module)

`_BatchingSpanPredictor` (`store.py`) and `BatchingEmbeddingClient` (`embeddings.py`) share a near-identical wait/coalesce/dispatch/overflow skeleton (`_get_lock`, the enqueue method, `_dispatch_after_wait`, and the queue-management half of `_drain`). They differ only in:
- the actual batch call (sync `predict_spans_batch` in a thread executor vs `await inner.embed_batch`), and
- result distribution (span = tolerant None-fill; embed = strict length-check then raise).

**New generic base `AsyncMicroBatcher[I, R]`** holds the shared machinery and exposes a protected `_submit(items) -> list[R]`. Subclasses implement:
- `async _run_batch(items) -> list[R]` — the actual work
- `_distribute(batch, results)` — defaults to tolerant None-fill (span); `BatchingEmbeddingClient` overrides with the strict length-check variant

**Public APIs are preserved exactly** — each subclass keeps its existing method as a thin wrapper:
- `_BatchingSpanPredictor.predict_batch(texts)` → `return await self._submit(texts)`
- `BatchingEmbeddingClient.embed_batch(texts)` → `return await self._submit(texts)` (plus existing `embed`, `dimensions`)

Subclass-specific state stays in the subclass (`_BatchingSpanPredictor`'s per-instance `ThreadPoolExecutor` + modelpack/task; `BatchingEmbeddingClient`'s `_inner`).

## Risk & mitigation

- **Helpers:** trivial, near-zero risk.
- **Micro-batcher:** this is **perf-critical concurrency code** (a documented throughput optimization). The refactor is mechanical and semantics-preserving, but concurrency bugs hide in subtle places and the full load path isn't runnable here. **Mitigation:** a dedicated `pytest.mark.asyncio` unit test (`tests/unit/test_micro_batcher.py`) exercising coalescing (N concurrent submits → one batch), overflow re-dispatch (>max_batch), result ordering, and exception propagation — runnable with plain asyncio, no heavy deps. If the abstraction turns out to be leaky during implementation, fall back to leaving the two classes separate and report.

## Verification

- `ruff check` + `ruff format --check` (CI paths) — green
- `mypy` (pinned 1.19.1, strict on `src.*`) — green
- New asyncio unit test for the batcher — green
- Functional check of the parsing helpers (bad input → default/None; fences stripped)
- Confirm no remaining `def _strip_markdown_fences` / `def _safe_float` inside `src/` except the shared module

## Out of scope

Phases 2 (shared `cml-contracts`/codegen for SDK↔server schemas) and 3 (unasync sync/async client codegen + modeling cross-package dupes). Each gets its own design.
