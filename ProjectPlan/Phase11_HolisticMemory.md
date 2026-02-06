# Phase 11: Holistic Memory Refactoring

**Status:** Completed

## Goal

Remove `MemoryScope` enum and scope-based partitioning. Replace with a unified memory store using contextual tags.

## Summary of Changes

- **Core**: Removed `MemoryScope`; added `MemoryContext` enum for tagging (non-exclusive). Updated `MemoryRecord` / `MemoryRecordCreate` to use `context_tags` and `source_session_id` instead of `scope` / `scope_id` / `user_id`.
- **Database**: Migration `004_remove_scopes_holistic` adds `context_tags` (TEXT[]) and `source_session_id`, drops scope columns. GIN index on `context_tags`.
- **Storage**: `PostgresMemoryStore` and neocortical/hippocampal stores use `tenant_id` only; optional `context_filter` for retrieval.
- **Orchestrator**: All methods use `tenant_id`, `context_tags`, `session_id`; no scope parameters.
- **Retrieval**: `MemoryRetriever` and `HybridRetriever` search across tenant memories; optional `context_filter`.
- **API**: Request/response schemas updated; stats endpoint is `GET /memory/stats` (tenant from auth).

## Architecture

Memory access is **holistic per tenant**: one logical store per tenant. Optional `context_tags` and `source_session_id` support categorization and origin tracking without partitioning.
