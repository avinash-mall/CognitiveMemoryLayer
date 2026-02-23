# API Versioning

## Current Version

The CML API is served at `/api/v1/` and is the only active version.

## Versioning Strategy

CML follows **URL-based versioning** with a semantic versioning commitment:

| Change Type | Example | Version Impact |
|---|---|---|
| New endpoint added | `POST /api/v1/memory/search` | No version bump (additive) |
| New optional field on request/response | `retrieval_meta` on `ReadMemoryResponse` | No version bump (additive) |
| Field removed or renamed | `memory_id` → `id` | **v2** required |
| Endpoint removed or path changed | `/session/{id}/read` removed | **v2** required |
| Request field becomes required | `context_tags` optional → required | **v2** required |

## Backward Compatibility Guarantees

Within a major version (e.g., `v1`):

- **Existing endpoints** will not be removed or have their paths changed.
- **Existing response fields** will not be removed or have their types changed.
- **Existing request fields** will not become required if they were optional.
- **New fields** may be added to responses at any time. Clients should ignore unknown fields.
- **New optional fields** may be added to requests at any time.

## Deprecation Policy

1. Deprecated endpoints are marked with `deprecated=True` in the OpenAPI schema.
2. Deprecated endpoints will continue to function for **at least 6 months** after deprecation.
3. The `Deprecation` HTTP header will be set on responses from deprecated endpoints.
4. The `Sunset` header will indicate the planned removal date.
5. Deprecation notices will be documented in the [CHANGELOG](../CHANGELOG.md).

### Currently Deprecated

| Endpoint | Deprecated Since | Replacement | Sunset Date |
|---|---|---|---|
| `POST /session/{id}/read` | v1.0.0 | `POST /memory/read` | TBD |

## Future Versions

When breaking changes are necessary:

1. A new version prefix (e.g., `/api/v2/`) will be introduced.
2. The previous version will enter maintenance mode (bug fixes only).
3. Both versions will run concurrently for at least 6 months.
4. Migration guides will be published in [docs/](.).
