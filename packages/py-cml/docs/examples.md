# Examples

Runnable scripts live in the repository [examples/](../../../examples/) and [packages/py-cml/examples/](../examples/). The runner discovers both locations automatically:

```bash
python scripts/run_examples.py --list
```

Set example env in the repo root `.env`:

- `CML_API_KEY`
- `CML_BASE_URL`
- `CML_ADMIN_API_KEY` for admin examples
- `OPENAI_API_KEY` + `OPENAI_MODEL`, or `LLM_INTERNAL__MODEL` + `LLM_INTERNAL__BASE_URL`, for OpenAI-compatible examples
- `ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL` for the Anthropic example

Interactive examples support unattended runs through `CML_EXAMPLE_NON_INTERACTIVE=1` and optional scripted input via `CML_EXAMPLE_INPUTS`.

## Core SDK Examples

| File | What it demonstrates |
|------|----------------------|
| [examples/quickstart.py](../../../examples/quickstart.py) | Minimal sync write/read/stream/context example |
| [examples/basic_usage.py](../../../examples/basic_usage.py) | Typed writes plus update, stats, and forget |
| [examples/async_example.py](../../../examples/async_example.py) | `AsyncCognitiveMemoryLayer`, `batch_read`, and SSE streaming |
| [examples/embedded_mode.py](../../../examples/embedded_mode.py) | In-process embedded mode with in-memory and file-backed storage |
| [examples/agent_integration.py](../../../examples/agent_integration.py) | Agent loop with graceful `read_safe()` retrieval |
| [examples/bulk_ingestion.py](../../../examples/bulk_ingestion.py) | Semaphore-limited concurrent ingestion |
| [examples/session_scope.py](../../../examples/session_scope.py) | SessionScope write/read/turn plus `get_session_context()` |
| [packages/py-cml/examples/temporal_fidelity.py](../examples/temporal_fidelity.py) | Historical timestamps and temporal replay |

## Admin, Direct API, and UI Examples

| File | What it demonstrates |
|------|----------------------|
| [examples/admin_dashboard.py](../../../examples/admin_dashboard.py) | Read-only dashboard/admin helpers such as overview, facts, retrieval test, and jobs |
| [examples/api_direct_minimal.py](../../../examples/api_direct_minimal.py) | Minimal `httpx` usage without the SDK |
| [examples/standalone_demo.py](../../../examples/standalone_demo.py) | Broader direct HTTP walkthrough |
| [examples/api_curl_examples.sh](../../../examples/api_curl_examples.sh) | Shell `curl` equivalents for core routes |
| [examples/streamlit_app.py](../../../examples/streamlit_app.py) | Streamlit UI for chat, retrieval, sessions, and optional admin inspection |

## LLM Integration Examples

| File | What it demonstrates |
|------|----------------------|
| [examples/chat_with_memory.py](../../../examples/chat_with_memory.py) | OpenAI chat loop backed by `memory.turn()` |
| [examples/openai_tool_calling.py](../../../examples/openai_tool_calling.py) | OpenAI function-calling with memory tools |
| [examples/anthropic_tool_calling.py](../../../examples/anthropic_tool_calling.py) | Anthropic tool-use with memory tools |
| [examples/langchain_integration.py](../../../examples/langchain_integration.py) | LangChain conversation chain backed by CML |

## Runner Usage

```bash
python scripts/run_examples.py --list
python scripts/run_examples.py --all
python scripts/run_examples.py --example quickstart
python scripts/run_examples.py --all --include-llm
python scripts/run_examples.py --kind shell --all
python scripts/run_examples.py --kind streamlit --all
```

## Notes

- `read(..., user_timezone="America/New_York")` and `turn(..., user_timezone="America/New_York")` are demonstrated in the session-scoped example.
- `timestamp=` for historical replay is demonstrated in the temporal fidelity example.
- Admin helpers require `CML_ADMIN_API_KEY`.
