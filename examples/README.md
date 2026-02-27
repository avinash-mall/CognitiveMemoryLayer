# Cognitive Memory Layer - Examples

Runnable examples using CML API directly or via `py-cml`.

See:

- Server docs: [UsageDocumentation](../ProjectPlan/UsageDocumentation.md)
- SDK docs: [packages/py-cml/docs](../packages/py-cml/docs/)

## Prerequisites

- Start API:
  - `docker compose -f docker/docker-compose.yml up api`
  - or test-key overlay: `docker compose -f docker/docker-compose.yml -f docker/docker-compose.test-key.yml up api`
- Copy `.env.example` to `.env`
- Set:
  - `AUTH__API_KEY` (server)
  - `CML_API_KEY` (client/examples; usually same value)
  - `CML_BASE_URL` (default `http://localhost:8000`)
- For LLM examples: `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`
- Install deps:
  - `pip install -r examples/requirements.txt`
  - or from repo root: `pip install -e .`

## Examples

| File | Description |
|------|-------------|
| `quickstart.py` | Minimal intro: write, read, get_context, stats |
| `basic_usage.py` | Full CRUD: write, read, update, forget, stats |
| `chat_with_memory.py` | Chatbot with `turn()` + OpenAI |
| `openai_tool_calling.py` | OpenAI tool-calling memory tools |
| `anthropic_tool_calling.py` | Anthropic tool-calling memory tools |
| `langchain_integration.py` | LangChain memory integration |
| `async_example.py` | Async usage and concurrency |
| `embedded_mode.py` | Serverless embedded mode |
| `agent_integration.py` | Agent loop pattern |
| `standalone_demo.py` | Raw `httpx` API demo |
| `api_direct_minimal.py` | Minimal direct API script |
| `api_curl_examples.sh` | Direct API curl examples |

## Run

```bash
python scripts/run_examples.py --all
python scripts/run_examples.py --example quickstart
python scripts/run_examples.py --all --include-llm
```

Quick manual runs:

```bash
python examples/quickstart.py
python examples/chat_with_memory.py
python examples/embedded_mode.py
```

## Troubleshooting

- Connection refused: start API containers.
- Unauthorized: verify `CML_API_KEY` matches server `AUTH__API_KEY`.
- Empty reads: broaden query or add more writes first.
