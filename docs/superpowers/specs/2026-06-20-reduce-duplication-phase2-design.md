# Reduce Duplication — Phase 2: Shared `cml_contracts` package

**Date:** 2026-06-20
**Status:** Proposed (awaiting sign-off)
**Goal:** Make the SDK (`cml`) and server (`src`) share ONE definition of the API contract (enums + Pydantic models) instead of re-declaring ~217 lines, **with zero breaking changes** to either side's public import surface.

## Why this is non-trivial

A repo-wide survey found the SDK's `cml/models/*` re-declares a large subset of the server's `src/api/schemas.py` + `src/core/enums.py`. An inventory established:

- **4 shared enums** (`MemoryType`, `MemoryStatus`, `MemorySource`, `OperationType`) — **byte-identical** member sets. (`MemoryContext` is server-only and unused elsewhere → stays in `src`.)
- **~25 dashboard/graph/config models** — field-identical on both sides.
- **A set of core models duplicated under *different names* with real drift**: `WriteMemoryResponse`/`WriteResponse` (`eval_outcome` `Literal` vs `str`), `CreateSessionRequest.ttl_hours` (`int|None` vs `int`), `ReadMemoryRequest`/`ReadRequest` (`format` vs `response_format` alias + `ge=1`), `FactItem`/`DashboardFactItem` (timestamps `str` vs `datetime`), plus systematic request-class renames (`WriteMemoryRequest`↔`WriteRequest`, etc.).
- **Packaging:** the published wheel (`cognitive-memory-layer`) ships **only** the `cml` top-level package; `src` does not ship. So the shared package must ship in that wheel to reach pip-install SDK users.

## Package & packaging

- **New package `cml_contracts`** at `packages/py-cml/src/cml_contracts/` (sibling of `cml`, same source root). Modules: `enums.py`, `models.py` (split into `requests.py`/`responses.py` if it grows).
- **Dependencies:** `pydantic` only (already a base dep). It depends on NOTHING in `src` or `cml` — it is the leaf both import.
- **Ship it:** add `"packages/py-cml/src/cml_contracts"` to root `pyproject.toml` → `[tool.hatch.build.targets.wheel] packages`. It then installs alongside `cml` in the wheel.
- **Import availability:** the server Docker has `src` on `PYTHONPATH=/app` and pip-installs the wheel (gets `cml` + `cml_contracts`); the SDK install gets `cml` + `cml_contracts`. Both sides can import `cml_contracts`. (Dev/CI: editable install / container, same as today for `cml`.)

## Canonical content & drift reconciliation

Each duplicated symbol is defined **once** in `cml_contracts`. Drift is reconciled to a value-compatible canonical form (no change to what goes over the wire):

| Pair | Drift | Canonical decision |
|---|---|---|
| `WriteMemoryResponse` / `WriteResponse` | `eval_outcome`: `Literal["stored","skipped"]\|None` vs `str\|None` | `Literal[...]\|None` (server-accurate; values unchanged → SDK back-compatible) |
| `CreateSessionRequest` | `ttl_hours`: `int\|None` vs `int` | `int\|None = 24` |
| `ReadMemoryRequest` / `ReadRequest` | `format` vs `response_format` alias + `ge=1` | field `response_format` with `alias="format"`, `populate_by_name=True`, `ge=1` |
| `FactItem` / `DashboardFactItem` | timestamps `str\|None` vs `datetime\|None` | `str\|None` (server emits isoformat strings on the wire) |
| request-class renames | `WriteMemoryRequest` vs `WriteRequest`, etc. | canonical = **server** name; the SDK name becomes an alias (`WriteRequest = WriteMemoryRequest`) |

## Backward compatibility — NO breaking changes

Existing import paths keep working via re-exports/aliases:

- `src/core/enums.py` → `from cml_contracts.enums import MemoryType, MemoryStatus, MemorySource, OperationType` (re-export); `MemoryContext` stays defined locally. (`from ...core.enums import X` unchanged — 24 server import sites untouched.)
- `cml/models/enums.py` → re-export the 4 from `cml_contracts.enums`.
- `src/api/schemas.py` → import shared models from `cml_contracts`, keep ~28 server-only classes, re-export so `from .schemas import X` is unchanged (3 server import sites untouched).
- `cml/models/responses.py` / `requests.py` → re-export shared models from `cml_contracts`; keep SDK alias names (`WriteRequest = WriteMemoryRequest`). `cml/models/__init__.__all__` (59 names) unchanged.
- `cml/storage/sqlite_store.py:13` currently imports the 3 enums from `src.core.enums` → switch to `cml_contracts.enums` (removes an existing cross-tree import).

## Implementation waves (each independently verified & shippable)

1. **Skeleton + enums** — create `cml_contracts`, move the 4 byte-identical enums, wire re-exports, add the wheel-packages entry. *Lowest risk; proves the packaging end-to-end before moving bulk.*
2. **Identical models** — move the ~25 field-identical dashboard/graph/config classes; re-export both sides.
3. **Renamed/drifted models** — move with the reconciliations above + back-compat aliases.

## Verification

- **Local (runnable here):** `ruff`, `ruff format`, `mypy` (pinned 1.19.1); an import-smoke (`import cml_contracts`, `from src.core.enums import MemoryType`, `from cml.models import WriteResponse`) under the repo path; a guard test asserting enum-member and model-field parity is now trivially true (single source).
- **CI-only (cannot verify locally — FLAGGED):** wheel build (`python -m build`) includes `cml_contracts`; `pip install` of the wheel exposes it; full pytest in Docker; SDK-consumer import. These MUST pass in CI before release.

## Risks

- Changes a **published-package** architecture + the wheel mapping; packaging correctness is only fully verifiable in CI. **Mitigations:** wave 1 proves the wheel mapping before any bulk move; back-compat re-exports/aliases mean zero import breakage on either side; drift reconciliations are value-compatible (nothing changes on the wire).

## Out of scope

Phase 3 (sync/async client codegen + modeling cross-package dupes), its own design.
