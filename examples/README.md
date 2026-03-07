# Cognitive Memory Layer Examples

Runnable examples live in:

- `examples/`
- `packages/py-cml/examples/`

Use the runner to see the current catalog:

```bash
python scripts/run_examples.py --list
```

## Prerequisites

- Start the API and backing services for API-based examples.
- Set these in the repo root `.env`:
  - `CML_API_KEY`
  - `CML_BASE_URL`
- Set `CML_ADMIN_API_KEY` for read-only admin examples.
- Set LLM env vars when running LLM-backed examples:
  - OpenAI-compatible examples: `OPENAI_API_KEY` + `OPENAI_MODEL`, or `LLM_INTERNAL__MODEL` + `LLM_INTERNAL__BASE_URL`
  - Anthropic example: `ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL`
- Install dependencies:
  - `pip install -e .`
  - or `pip install -r examples/requirements.txt`
  - embedded example: `pip install -e ".[embedded]"`

## Example Catalog

| Example | Kind | Description |
|---------|------|-------------|
| `quickstart.py` | Python | Minimal sync write/read/stream/context example |
| `basic_usage.py` | Python | Typed writes plus update, stats, and forget |
| `async_example.py` | Python | Async gather writes, batch reads, and SSE streaming |
| `embedded_mode.py` | Python | Embedded mode with in-memory and file-backed storage |
| `agent_integration.py` | Python | Agent loop with graceful `read_safe()` retrieval |
| `bulk_ingestion.py` | Python | Semaphore-limited concurrent ingestion |
| `session_scope.py` | Python | SessionScope write/read/turn and session context |
| `admin_dashboard.py` | Python | Read-only dashboard/admin helper example |
| `chat_with_memory.py` | Python | OpenAI chat loop backed by `memory.turn()` |
| `openai_tool_calling.py` | Python | OpenAI function-calling with memory tools |
| `anthropic_tool_calling.py` | Python | Anthropic tool-use with memory tools |
| `langchain_integration.py` | Python | LangChain conversation chain backed by CML |
| `api_direct_minimal.py` | Python | Minimal direct HTTP example with `httpx` |
| `standalone_demo.py` | Python | Broader direct HTTP walkthrough |
| `streamlit_app.py` | Streamlit | Streamlit UI for chat, write/read, sessions, and optional admin inspection |
| `api_curl_examples.sh` | Shell | Direct `curl` examples |
| `packages/py-cml/examples/temporal_fidelity.py` | Python | Historical timestamp and temporal replay example |

## Runner Usage

```bash
python scripts/run_examples.py --list
python scripts/run_examples.py --all
python scripts/run_examples.py --example quickstart
python scripts/run_examples.py --all --include-llm
python scripts/run_examples.py --kind shell --all
python scripts/run_examples.py --kind streamlit --all
```

Runner behavior:

- Discovers examples automatically from `examples/` and `packages/py-cml/examples/`
- Runs Python, shell, and Streamlit examples when the local environment supports them
- Skips missing env or dependency cases with explicit reasons
- Sets `CML_EXAMPLE_NON_INTERACTIVE=1` so interactive examples can run unattended
- Supports optional scripted inputs through `CML_EXAMPLE_INPUTS`

## Manual Runs

```bash
python examples/quickstart.py
python examples/session_scope.py
python examples/admin_dashboard.py
python packages/py-cml/examples/temporal_fidelity.py
bash examples/api_curl_examples.sh
streamlit run examples/streamlit_app.py
```

## Troubleshooting

- Connection errors: verify the API is running and `CML_BASE_URL` points to it.
- Unauthorized errors: verify `CML_API_KEY` matches server `AUTH__API_KEY`.
- Admin example failures: verify `CML_ADMIN_API_KEY` is configured.
- Empty reads: write a few memories first or broaden the query.
- Embedded example failures: install `.[embedded]` and ensure the embedding backend is available.
