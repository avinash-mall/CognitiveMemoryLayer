# py-cml Validation Report: Current Compatibility Status

This report tracks `packages/py-cml` compatibility with the current CML server/API.

Updated: 2026-03-05

---

## Summary

| Area | Status | Notes |
|------|--------|-------|
| Dashboard CSRF header | DONE | `X-Requested-With` added for dashboard state-changing methods |
| `ReadResponse.retrieval_meta` | DONE | Optional field exposed on SDK response model |
| Write content validation | DONE | `WriteRequest.content` enforces 1..100000 characters |
| Streaming read endpoint | DONE | `read_stream()` available in sync/async + namespaced wrappers |
| Session-scoped write route | DONE | `SessionScope.write()` / `AsyncSessionScope.write()` use `/session/{id}/write` |
| Session-scoped read route | DONE | `SessionScope.read()` / `AsyncSessionScope.read()` use `/session/{id}/read` |
| New dashboard/admin endpoints | DONE | Added facts, graph overview, export, and direct admin helpers |
| Namespaced wrapper parity | DONE | `user_timezone` and `timestamp` forwarding aligned with parent clients |
| Embedded response mapping parity | DONE | Embedded read mapping includes constraints/procedures in `memories` |

---

## Implemented in py-cml

### Core parity updates

- `SessionScope.write()` and `AsyncSessionScope.write()` now call `POST /session/{session_id}/write`.
- `NamespacedClient` and `AsyncNamespacedClient` forward:
  - `user_timezone` on read/get_context/search/read_stream
  - `timestamp` and `user_timezone` on turn
- Embedded packet mapping now includes constraints/procedures in `ReadResponse.memories` and sets `ReadResponse.constraints`.

### Added methods (sync + async + namespaced wrappers)

- `dashboard_facts(...)`
- `dashboard_invalidate_fact(fact_id)`
- `dashboard_export_memories(...)`
- `graph_overview(...)`
- `admin_consolidate(user_id)`
- `admin_forget(user_id, dry_run=...)`

### Added models

- `DashboardFactItem`
- `DashboardFactListResponse`

---

## Verification

- Unit tests: `175 passed`
- Lint: `ruff check` passed on updated SDK/doc-related files

---

## Notes

- The server still marks `POST /session/{session_id}/read` as deprecated in favor of `/memory/read`.
- The SDK intentionally keeps session-scoped read wrappers using this route for strict session-path behavior.
