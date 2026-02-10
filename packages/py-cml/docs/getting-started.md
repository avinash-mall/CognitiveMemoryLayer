# Getting Started

## Prerequisites

- Python 3.11+
- A running CML server (for client mode), or use embedded mode to run without a server

## Installation

```bash
pip install cognitive-memory-layer
```

For embedded mode (in-process, no server):

```bash
pip install cognitive-memory-layer[embedded]
```

## First memory in 5 steps

1. **Install** — `pip install cognitive-memory-layer`
2. **Start the CML server** — See the CognitiveMemoryLayer project for server setup (or use embedded mode and skip this). From the repo root: `docker compose -f docker/docker-compose.yml up -d postgres neo4j redis api`.
3. **Get your API key** — From your CML server or dashboard. For local development, the project `.env.example` uses `AUTH__API_KEY=test-key`; copy to `.env` so the server accepts that key.
4. **Create the client** — Set `CML_BASE_URL` and `CML_API_KEY` in `.env`, then `CognitiveMemoryLayer(api_key="...", base_url="...")` (or omit `base_url` to use `CML_BASE_URL` from env).
5. **Write and read** — `memory.write("...")` then `memory.read("query")` or `memory.get_context("query")`

## Connect to a Server (detailed)

1. **Start the CML server** — See the CognitiveMemoryLayer project for server setup.
2. **Get your API key** — From your CML server or dashboard.
3. **Initialize the client** and write your first memory:

```python
from cml import CognitiveMemoryLayer

memory = CognitiveMemoryLayer(
    api_key="your-api-key",
    base_url="http://localhost:8000",  # or set CML_BASE_URL in .env and omit
)
memory.write("User prefers vegetarian food and lives in Paris.")
result = memory.read("What does the user eat?")
for item in result.memories:
    print(f"  {item.text} (relevance: {item.relevance:.2f})")
memory.close()
```

4. Or use a context manager so the client is closed automatically:

```python
with CognitiveMemoryLayer(api_key="...", base_url="http://localhost:8000") as memory:  # or set CML_BASE_URL in .env
    memory.write("User works at a tech startup.")
    result = memory.read("user job")
```

## Testing the package (install and run examples)

From the repo root, with a virtual environment:

1. **Install from source (editable):**  
   `pip install -e packages/py-cml`
2. **Verify:**  
   `python -c "import cml; print(cml.__version__)"`  
   `python -c "from cml import CognitiveMemoryLayer; print('OK')"`
3. **Unit tests (no server):**  
   `cd packages/py-cml && pytest tests/unit -v`
4. **Run quickstart against a local CML server:**  
   - Start the CML API (e.g. Docker: `docker compose -f docker/docker-compose.yml up -d postgres neo4j redis && docker compose -f docker/docker-compose.yml up -d api`).
   - In the repo `.env` set `AUTH__API_KEY=your-api-key` and `OPENAI_API_KEY=sk-your-key` (server uses OpenAI for embeddings by default).
   - From repo root:  
     `python packages/py-cml/examples/quickstart.py`  
   Set `CML_API_KEY`, `CML_BASE_URL`, and (for chat examples) `OPENAI_MODEL` or `LLM__MODEL` in `.env`; examples read these and do not use hardcoded URLs or models.

## Next Steps

- [API Reference](api-reference.md) — All operations and types
- [Examples](examples.md) — Quickstart, chat, async, embedded, agent
- Embedded mode — See README and examples/embedded_mode.py
