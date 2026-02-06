# Phase 14: Documentation and Examples Update

**Status:** Completed

## Goal

Update all documentation and examples to reflect the holistic, seamless API.

## Summary of Changes

- **README.md**: Quick start uses holistic write/read and documents `POST /memory/turn`; API table updated; added "Seamless Memory" subsection.
- **UsageDocumentation.md**: Quick start and API reference rewritten without scopes; added "Holistic memory and context tags" and "Seamless Memory"; tool definitions and examples updated; `GET /memory/stats` documented.
- **Examples**: `memory_client.py` updated to holistic API and `process_turn()`; `basic_usage.py`, `chatbot_with_memory.py`, `openai_tool_calling.py`, `anthropic_tool_calling.py`, `langchain_integration.py` updated to remove scope/scope_id and use the new client; examples README updated.
