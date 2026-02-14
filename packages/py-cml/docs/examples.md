# Examples

Runnable scripts live in the repository [examples/](../../../examples/) directory; this page describes SDK usage patterns.

The `examples/` directory contains runnable scripts that use cognitive-memory-layer with a CML server or in embedded mode. **Set in `.env`:** `CML_API_KEY`, `CML_BASE_URL`, and for chat/OpenAI examples `OPENAI_MODEL` or `LLM__MODEL`. No hardcoded URLs or model names in code. The server supports read filters (`memory_types`, `since`, `until`), response formats (`packet`, `list`, `llm_context`), and write `metadata`/`memory_type`. For timezone-aware "today"/"yesterday" retrieval, use `read(..., user_timezone="America/New_York")` or `turn(..., user_timezone="America/New_York")` when the server supports it. For benchmark scripts, use `write(..., eval_mode=True)` to get `eval_outcome` and `eval_reason` in the response; see [API Reference — Eval mode](api-reference.md#eval-mode-write-gate).

## Quickstart

**File:** [examples/quickstart.py](../../../examples/quickstart.py)

Store and retrieve memories in under a minute. Uses the sync client: initialize, write several memories, read by query, call `get_context`, and `stats()`. Run with:

```bash
python examples/quickstart.py
```

## Chat with Memory

**File:** [examples/chat_with_memory.py](../../../examples/chat_with_memory.py)

A simple chatbot that uses OpenAI and cognitive-memory-layer for persistent memory. Each turn retrieves relevant memories and injects them into the system prompt, then stores the exchange. Requires `openai`, a CML server, and `.env` with `CML_BASE_URL`, `CML_API_KEY`, `OPENAI_MODEL` (or `LLM__MODEL`). Type `quit` to exit.

```bash
pip install openai
python examples/chat_with_memory.py
```

## Async Usage

**File:** [examples/async_example.py](../../../examples/async_example.py)

Uses `AsyncCognitiveMemoryLayer` with `asyncio`: concurrent writes with `asyncio.gather`, then `read` and `batch_read`. Demonstrates async context manager and batch operations.

```bash
python examples/async_example.py
```

## Embedded Mode

**File:** [examples/embedded_mode.py](../../../examples/embedded_mode.py)

Run py-cml without a server. Zero-config block uses in-memory SQLite and local embeddings; a second part uses `db_path` for persistent storage and reads back in a new instance. Requires `pip install cognitive-memory-layer[embedded]` and the CML engine from the monorepo for full functionality.

```bash
python examples/embedded_mode.py
```

## Agent Integration

**File:** [examples/agent_integration.py](../../../examples/agent_integration.py)

A minimal agent that observes events, plans using memory context, and reflects on past observations. Uses `AsyncCognitiveMemoryLayer` with `context_tags` and `agent_id`. Illustrates how to plug cognitive-memory-layer into an agent loop.

```bash
python examples/agent_integration.py
```

## Temporal Fidelity (Historical Data)

**File:** [examples/temporal_fidelity.py](../examples/temporal_fidelity.py)

Demonstrates the `timestamp` parameter for storing memories with specific event times, enabling historical data replay and temporal reasoning. Shows:
- Storing historical memories with specific timestamps
- Processing historical conversation turns
- Benchmark evaluation scenarios (Locomo-style)
- Temporal ordering verification

```bash
python examples/temporal_fidelity.py
```

**Quick example:**

```python
from datetime import datetime, timezone
from cml import CognitiveMemoryLayer

memory = CognitiveMemoryLayer(api_key="...", base_url="...")

# Store historical memories with their original timestamps
session_date = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

memory.write(
    "User mentioned they prefer dark mode",
    timestamp=session_date,
    session_id="session_1"
)

# Process historical conversation turns
memory.turn(
    user_message="What's the weather like?",
    assistant_response="It's sunny today!",
    timestamp=session_date,
    session_id="session_1"
)

# When timestamp is omitted, it defaults to "now"
memory.write("This memory gets the current timestamp")
```

This feature is particularly useful for:
- **Benchmark evaluations** (e.g., Locomo) that replay historical conversations with correct temporal ordering
- **Data migration** when importing historical records from other systems
- **Testing** temporal reasoning and memory consolidation over time
