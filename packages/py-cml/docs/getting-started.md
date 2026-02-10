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

## Connect to a Server

1. **Start the CML server** — See the CognitiveMemoryLayer project for server setup.
2. **Get your API key** — From your CML server or dashboard.
3. **Initialize the client** and write your first memory:

```python
from cml import CognitiveMemoryLayer

memory = CognitiveMemoryLayer(
    api_key="your-api-key",
    base_url="http://localhost:8000",
)
memory.write("User prefers vegetarian food and lives in Paris.")
result = memory.read("What does the user eat?")
for item in result.memories:
    print(f"  {item.text} (relevance: {item.relevance:.2f})")
memory.close()
```

4. Or use a context manager so the client is closed automatically:

```python
with CognitiveMemoryLayer(api_key="...", base_url="http://localhost:8000") as memory:
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
   The example uses `api_key="your-api-key"` and `base_url="http://localhost:8000"` by default.

## Next Steps

- [API Reference](api-reference.md) — All operations and types
- [Examples](examples.md) — Quickstart, chat, async, embedded, agent
- Embedded mode — See README and examples/embedded_mode.py
