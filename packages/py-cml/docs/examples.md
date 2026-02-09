# Examples

The `examples/` directory contains runnable scripts that use py-cml with a CML server or in embedded mode. Set `CML_API_KEY` and `CML_BASE_URL` (e.g. `http://localhost:8000`) when using client mode.

## Quickstart

**File:** [examples/quickstart.py](../examples/quickstart.py)

Store and retrieve memories in under a minute. Uses the sync client: initialize, write several memories, read by query, call `get_context`, and `stats()`. Run with:

```bash
python examples/quickstart.py
```

## Chat with Memory

**File:** [examples/chat_with_memory.py](../examples/chat_with_memory.py)

A simple chatbot that uses OpenAI and py-cml for persistent memory. Each turn retrieves relevant memories and injects them into the system prompt, then stores the exchange. Requires `openai` and a CML server. Type `quit` to exit.

```bash
pip install openai
python examples/chat_with_memory.py
```

## Async Usage

**File:** [examples/async_example.py](../examples/async_example.py)

Uses `AsyncCognitiveMemoryLayer` with `asyncio`: concurrent writes with `asyncio.gather`, then `read` and `batch_read`. Demonstrates async context manager and batch operations.

```bash
python examples/async_example.py
```

## Embedded Mode

**File:** [examples/embedded_mode.py](../examples/embedded_mode.py)

Run py-cml without a server. Zero-config block uses in-memory SQLite and local embeddings; a second part uses `db_path` for persistent storage and reads back in a new instance. Requires `pip install py-cml[embedded]` and the CML engine from the monorepo for full functionality.

```bash
python examples/embedded_mode.py
```

## Agent Integration

**File:** [examples/agent_integration.py](../examples/agent_integration.py)

A minimal agent that observes events, plans using memory context, and reflects on past observations. Uses `AsyncCognitiveMemoryLayer` with `context_tags` and `agent_id`. Illustrates how to plug py-cml into an agent loop.

```bash
python examples/agent_integration.py
```
