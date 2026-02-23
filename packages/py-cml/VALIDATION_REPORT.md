# py-cml Validation Report: Compatibility with Project Changes

This report validates whether `packages/py-cml` needs updates given the changes made to the CML server and API.

---

## Summary

| Area | Status | Action Required |
|------|--------|-----------------|
| **CSRF header for dashboard POST/PUT/DELETE/PATCH** | **DONE** | Added in transport `_add_dashboard_csrf_if_needed()` |
| **ReadResponse.retrieval_meta** | **DONE** | Added optional field for forward compatibility |
| **WriteRequest content validation** | Done | Already has `min_length=1, max_length=100_000` |
| **Streaming Read Endpoint (E-02)** | **DONE** | Added `read_stream()` to sync and async clients |
| **get_session_context** | OK | Uses GET `/session/{id}/context` (not deprecated) |
| **404/403 handling** | OK | Transport maps 404→NotFoundError, 403→AuthorizationError |
| **API paths** | OK | Dashboard paths unchanged (`/api/v1/dashboard/...`) |

---

## 1. CSRF Header (Required)

**Change:** The server's CSRF middleware requires `X-Requested-With: XMLHttpRequest` for all dashboard POST, PUT, DELETE, and PATCH requests.

**Impact:** The following py-cml methods will **return 403 "Missing CSRF header"** when calling a CML server with the new middleware:

- `consolidate()` — POST `/dashboard/consolidate`
- `run_forgetting()` — POST `/dashboard/forget`
- `reconsolidate()` — POST `/dashboard/reconsolidate`
- `dashboard_bulk_action()` — POST `/dashboard/memories/bulk-action`
- `dashboard_config_update()` — PUT `/dashboard/config`
- `test_retrieval()` — POST `/dashboard/retrieval`
- `reset_database()` — POST `/dashboard/database/reset`

**Fix:** In [packages/py-cml/src/cml/transport/http.py](packages/py-cml/src/cml/transport/http.py), add `X-Requested-With: XMLHttpRequest` to headers when:
- `path.startswith("/dashboard")` and
- `method in ("POST", "PUT", "DELETE", "PATCH")`

Suggested approach: In `_do_request` (and async equivalent), before building the final `headers` dict, add:

```python
if path.startswith("/dashboard") and method in ("POST", "PUT", "DELETE", "PATCH"):
    headers = {**headers, "X-Requested-With": "XMLHttpRequest"}
```

---

## 2. ReadResponse.retrieval_meta (Optional)

**Change:** The server's `ReadMemoryResponse` now includes optional `retrieval_meta: dict | None` (sources completed/timed out, elapsed ms).

**Impact:** Pydantic v2 ignores extra fields by default when parsing, so the SDK will not break. However, users cannot access `retrieval_meta` if the server returns it.

**Fix (optional):** In [packages/py-cml/src/cml/models/responses.py](packages/py-cml/src/cml/models/responses.py), add to `ReadResponse`:

```python
retrieval_meta: dict | None = None  # Server: sources_completed, sources_timed_out, total_elapsed_ms
```

---

## 3. WriteRequest Content Validation (Done)

**Status:** Already implemented. `WriteRequest` has `content: str = Field(..., min_length=1, max_length=100_000)`.

---

## 3.5. Streaming Read Endpoint (Done)

**Change:** The server added a `POST /memory/read/stream` SSE endpoint for progressive rendering (E-02).

**Impact:** The SDK's `read()` method uses the synchronous JSON endpoint and processes the whole result set at once.

**Fix:** Implemented `read_stream()` in `CognitiveMemoryLayer`, `AsyncCognitiveMemoryLayer`, `NamespacedClient`, and `AsyncNamespacedClient`. Uses `httpx.stream()` and yields `MemoryItem` objects incrementally as they arrive from the server.

---

## 4. get_session_context (OK)

**Change:** The server deprecated `POST /session/{session_id}/read` in favor of `/memory/read`.

**Impact:** py-cml's `get_session_context()` uses `GET /session/{session_id}/context`, which is a different endpoint and is **not** deprecated. No change needed.

---

## 5. Exception Handling (OK)

**Change:** The server returns 404 for `MemoryNotFoundError` and 403 for `MemoryAccessDenied` on update.

**Impact:** The transport's `_raise_for_status()` already maps:
- 404 → `NotFoundError`
- 403 → `AuthorizationError`

No change needed.

---

## 6. API Paths (OK)

**Change:** Dashboard routes were split from `dashboard_routes.py` into `src/api/dashboard/` package.

**Impact:** URL paths remain `/api/v1/dashboard/...`. No change needed.

---

## Recommended Implementation Order

1. **CSRF header** — Required; without it, all dashboard state-changing calls fail with 403.
2. **retrieval_meta** — Optional; improves parity and future-proofing.
