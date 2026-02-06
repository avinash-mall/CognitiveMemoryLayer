# Phase 12: Seamless Memory Integration

**Status:** Completed

## Goal

Make memory retrieval automatic and unconscious, like human association.

## Summary of Changes

- **SeamlessMemoryProvider** (`src/memory/seamless_provider.py`): Processes a conversation turn by (1) auto-retrieving relevant context for the user message, (2) optionally storing user/assistant content, (3) running reconsolidation when an assistant response is provided. Returns `SeamlessTurnResult` with `memory_context` ready for LLM injection.
- **QueryClassifier**: Added `recent_context` to `classify()` for context-aware classification of vague queries.
- **MemoryRetriever**: Passes `recent_context` to the classifier when provided.
- **API**: New `POST /memory/turn` endpoint and `ProcessTurnRequest` / `ProcessTurnResponse` schemas.
- **Tools**: Simplified tool descriptions for `memory_write` and `memory_read` (no scope parameters).

## Usage

Use `POST /memory/turn` with `user_message` (and optionally `assistant_response`, `session_id`) to get `memory_context` for the current turn and optional auto-store.
