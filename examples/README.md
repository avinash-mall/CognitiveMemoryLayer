# Cognitive Memory Layer - Examples

Working examples using the **py-cml** package. See [UsageDocumentation](../ProjectPlan/UsageDocumentation.md) for server setup and [packages/py-cml/docs](../packages/py-cml/docs/) for SDK docs.

**Prerequisites:** Start API (`docker compose -f docker/docker-compose.yml up api`, or with test keys without changing `.env`: `docker compose -f docker/docker-compose.yml -f docker/docker-compose.test-key.yml up api`), copy `.env.example` to `.env`, set `AUTH__API_KEY`. For LLM examples: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Install: `pip install -r examples/requirements.txt` or `pip install -e packages/py-cml`.

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
| `api_direct_minimal.py` | **No py-cml**: minimal httpx-only script (health, write, read, turn, stats) |
| `api_curl_examples.sh` | **No py-cml**: curl commands for all memory endpoints |
| `openclaw_skill/` | [OpenClaw](https://openclaw.ai/) skill: persistent structured memory (SKILL.md + setup) |
| `packages/py-cml/examples/temporal_fidelity.py` | Timestamped writes and turns (historical replay, benchmarks) |

## How to run

From the project root you can run each example in turn and get a Pass / Fail / Skip report:

```bash
# Run all non-LLM examples (embedded, quickstart, basic_usage, async, agent_integration, temporal_fidelity, standalone_demo)
python scripts/run_examples.py --all

# Run a single example by name
python scripts/run_examples.py --example embedded_mode
python scripts/run_examples.py --example quickstart

# Include LLM examples (requires OPENAI_* or ANTHROPIC_* keys and CML API)
python scripts/run_examples.py --all --include-llm

# Do not skip when env vars are missing (useful for debugging)
python scripts/run_examples.py --all --no-skip
```

Prerequisites: API up for API-dependent examples; embedded needs `pip install -e "packages/py-cml[embedded]"`. Standalone demo runs non-interactively.

**Quick runs:** `python examples/quickstart.py` | `python examples/chat_with_memory.py` | `python examples/embedded_mode.py` (no server). Direct API: `python examples/api_direct_minimal.py` or `bash examples/api_curl_examples.sh`.

**Troubleshooting:** Could not connect → start API. API key required → set `AUTH__API_KEY` in `.env` (or start API with `docker compose -f docker/docker-compose.yml -f docker/docker-compose.test-key.yml up api` to use test keys). No memories → broaden query. See [UsageDocumentation](../ProjectPlan/UsageDocumentation.md) and [py-cml docs](../packages/py-cml/docs/).
