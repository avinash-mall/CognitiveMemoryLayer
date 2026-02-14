# Cognitive Memory Layer - Examples

Working examples for the Cognitive Memory Layer with LLMs and as a standalone service. All examples use the **py-cml** package (`cognitive-memory-layer`).

## Prerequisites

1. **Start the API server** (for non-embedded examples):
   ```bash
   docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
   docker compose -f docker/docker-compose.yml up api
   ```
   The server reads configuration from the project root **`.env`** (copy `.env.example` to `.env`). Set `EMBEDDING__DIMENSIONS` to match your embedding model when using write/read; see main project [.env.example](../.env.example) and [tests/README.md](../tests/README.md).

2. **Install dependencies**:
   ```bash
   pip install -r examples/requirements.txt
   # Or from monorepo: pip install -e packages/py-cml
   ```

3. **Environment variables** (in `.env` or shell):
   ```bash
   AUTH__API_KEY=your-secret-key     # Required for API examples
   CML_BASE_URL=http://localhost:8000  # Or MEMORY_API_URL
   # For LLM examples:
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o
   ANTHROPIC_API_KEY=sk-ant-...
   ```

## Examples Overview

| File | Description |
|------|-------------|
| `quickstart.py` | Minimal intro: write, read, get_context, stats |
| `basic_usage.py` | Full CRUD: write, read, update, forget, stats |
| `chat_with_memory.py` | Simple chatbot: py-cml `turn()` for context + OpenAI |
| `openai_tool_calling.py` | OpenAI function calling: memory_write, memory_read, memory_update, memory_forget |
| `anthropic_tool_calling.py` | Anthropic Claude tool use with same memory tools |
| `langchain_integration.py` | LangChain `BaseMemory` backed by py-cml |
| `async_example.py` | Async usage: concurrent writes, batch_read, pipeline |
| `embedded_mode.py` | Serverless: py-cml embedded with SQLite (no API) |
| `agent_integration.py` | Agent pattern: observe, plan, reflect using memory |
| `standalone_demo.py` | **No py-cml**: raw httpx demo of all API endpoints |
| `openclaw_skill/` | [OpenClaw](https://openclaw.ai/) skill: persistent structured memory (SKILL.md + setup) |

### Quick Start

```bash
python examples/quickstart.py
```

### Simple Chat (py-cml + OpenAI)

```bash
python examples/chat_with_memory.py
```

### Tool-calling chat (OpenAI or Anthropic)

```bash
python examples/openai_tool_calling.py
# or
python examples/anthropic_tool_calling.py
```

### Async Usage

```bash
python examples/async_example.py
```

### Embedded Mode (no server)

```bash
pip install cognitive-memory-layer[embedded]
python examples/embedded_mode.py
```

### Standalone API Demo (no LLM)

```bash
python examples/standalone_demo.py
```

## API Quick Reference (py-cml)

The API is **holistic**: tenant from API key, no explicit scopes. Use `session_id`, `agent_id`, `namespace` for origin tracking.

```python
from cml import CognitiveMemoryLayer

with CognitiveMemoryLayer(
    api_key=os.environ.get("AUTH__API_KEY"),
    base_url="http://localhost:8000",
) as memory:
    # Write
    memory.write("User prefers vegetarian food", session_id="sess-1", memory_type="preference")

    # Read
    result = memory.read("dietary preferences", response_format="llm_context")
    print(result.context)

    # Seamless turn (auto-retrieve + auto-store)
    turn = memory.turn(user_message="What do I like?", session_id="sess-1")
    print(turn.memory_context)  # Inject into LLM prompt

    # Update, forget, stats
    memory.update(memory_id=uuid, feedback="correct")
    memory.forget(query="old address", action="archive")
    stats = memory.stats()
```

## Memory Types

| Type | Use Case |
|------|----------|
| `semantic_fact` | Permanent facts |
| `preference` | User preferences |
| `constraint` | Must-follow rules (allergies, restrictions) |
| `episodic_event` | Specific events |
| `hypothesis` | Uncertain info (needs confirmation) |

## Troubleshooting

- **Could not connect**: Start API with `docker compose -f docker/docker-compose.yml up api`
- **API key required**: Set `AUTH__API_KEY` (or `CML_API_KEY`) in `.env`
- **No memories found**: Try a broader query; verify writes succeeded

## Further Reading

- [Usage Documentation](../ProjectPlan/UsageDocumentation.md)
- [py-cml README](../packages/py-cml/README.md)
- [API Docs](http://localhost:8000/docs) (when server is running)
