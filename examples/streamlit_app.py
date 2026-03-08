"""Streamlit UI for direct interaction with the CML API."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import streamlit as st

try:
    from _shared import get_env, normalize_base_url
except ModuleNotFoundError:  # pragma: no cover - exercised by streamlit test harness
    from examples._shared import get_env, normalize_base_url

_get_settings: Callable[[], Any] | None
_get_eval_llm_client: Callable[[], Any] | None
try:
    from src.core.config import get_settings as _get_settings
    from src.utils.llm import get_eval_llm_client as _get_eval_llm_client
except Exception:  # pragma: no cover - optional for pure-API startup checks
    _get_settings = None
    _get_eval_llm_client = None

EXAMPLE_META = {
    "name": "streamlit_app",
    "kind": "streamlit",
    "summary": "Streamlit UI for chat, write/read, sessions, and optional admin inspection.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": False,
    "requires_anthropic": False,
    "interactive": False,
    "timeout_sec": 45,
}

st.set_page_config(page_title="CML Studio", page_icon="CML", layout="wide")


def _default_api_url() -> str:
    base_url = get_env("CML_BASE_URL")
    if not base_url:
        return ""
    return normalize_base_url(base_url, api_path=True)


def _turn_context_message(payload: dict) -> str:
    """Render a helpful chat message from /memory/turn response payload."""
    memory_context = str(payload.get("memory_context") or "").strip()
    if memory_context:
        return memory_context

    retrieved = int(payload.get("memories_retrieved") or 0)
    if retrieved == 0:
        return (
            "No matching memories yet. This is expected on a first turn.\n\n"
            "Try writing a memory in the 'Write & Read' tab, then ask a follow-up question."
        )
    return "Turn completed, but memory context was empty."


def _run_async(coro: Any) -> Any:
    """Run an async LLM call without relying on global event loop state."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _chat_model_info() -> str:
    if _get_settings is None:
        return "Chat model: unavailable (server LLM utilities not importable)"
    try:
        settings = _get_settings()
        provider = settings.llm_eval.provider or settings.llm_internal.provider
        model = settings.llm_eval.model or settings.llm_internal.model
        return f"Chat model: {provider}:{model} (from LLM_EVAL__* fallback)"
    except Exception as exc:
        return f"Chat model: unavailable ({exc})"


def _build_chat_prompt(
    *,
    user_message: str,
    memory_context: str,
    chat_history: list[dict],
    max_messages: int = 6,
) -> str:
    recent = chat_history[-max_messages:]
    transcript_lines: list[str] = []
    for message in recent:
        role = str(message.get("role", "assistant")).upper()
        content = str(message.get("content", "")).strip()
        if content:
            transcript_lines.append(f"{role}: {content}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(none)"

    context_text = memory_context.strip() or "(none)"
    return (
        "Retrieved memory context:\n"
        f"{context_text}\n\n"
        "Recent conversation:\n"
        f"{transcript}\n\n"
        "Current user message:\n"
        f"{user_message}"
    )


def _generate_assistant_reply(
    *,
    user_message: str,
    memory_context: str,
    chat_history: list[dict],
) -> str:
    if _get_eval_llm_client is None:
        raise RuntimeError(
            "Eval LLM client is unavailable. Start Streamlit from the repo root with server deps installed."
        )

    llm = _get_eval_llm_client()
    prompt = _build_chat_prompt(
        user_message=user_message,
        memory_context=memory_context,
        chat_history=chat_history,
    )
    response = _run_async(
        llm.complete(
            prompt=prompt,
            temperature=0.2,
            max_tokens=450,
            system_prompt=(
                "You are a helpful assistant in a chat UI. "
                "Use retrieved memory context when relevant. "
                "If context is empty, still answer normally from the current message."
            ),
        )
    )
    text = str(response).strip()
    return text or "I do not have enough information yet. Please provide more detail."


if "base_url" not in st.session_state:
    st.session_state.base_url = _default_api_url()
if "api_key" not in st.session_state:
    st.session_state.api_key = get_env("CML_API_KEY") or ""
if "admin_api_key" not in st.session_state:
    st.session_state.admin_api_key = get_env("CML_ADMIN_API_KEY") or ""
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-demo"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


class CMLHttpClient:
    def __init__(self, *, base_url: str, api_key: str, admin_api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.admin_api_key = admin_api_key
        self.timeout = 60.0

    def _headers(self, *, use_admin: bool = False) -> dict[str, str]:
        api_key = self.admin_api_key if use_admin else self.api_key
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        if use_admin:
            headers["X-Requested-With"] = "XMLHttpRequest"
        return headers

    def get(self, path: str, *, use_admin: bool = False) -> httpx.Response:
        return httpx.get(
            f"{self.base_url}{path}",
            headers=self._headers(use_admin=use_admin),
            timeout=self.timeout,
        )

    def post(self, path: str, payload: dict, *, use_admin: bool = False) -> httpx.Response:
        return httpx.post(
            f"{self.base_url}{path}",
            headers=self._headers(use_admin=use_admin),
            json=payload,
            timeout=self.timeout,
        )


def client() -> CMLHttpClient:
    return CMLHttpClient(
        base_url=st.session_state.base_url,
        api_key=st.session_state.api_key,
        admin_api_key=st.session_state.admin_api_key,
    )


st.title("Cognitive Memory Layer Studio")
st.caption("Direct API demo for chat, retrieval, sessions, and optional admin inspection.")

with st.sidebar:
    st.header("Connection")
    st.session_state.base_url = st.text_input("API base URL", value=st.session_state.base_url)
    st.session_state.api_key = st.text_input(
        "API key",
        value=st.session_state.api_key,
        type="password",
    )
    st.session_state.admin_api_key = st.text_input(
        "Admin API key",
        value=st.session_state.admin_api_key,
        type="password",
    )
    st.session_state.session_id = st.text_input("Session id", value=st.session_state.session_id)
    st.caption(_chat_model_info())

    if st.button("Test /health"):
        try:
            response = client().get("/health")
            st.success(response.json())
        except Exception as exc:
            st.error(str(exc))


chat_tab, browse_tab, session_tab, admin_tab = st.tabs(
    ["Chat", "Write & Read", "Sessions", "Optional Admin"]
)

with chat_tab:
    st.subheader("/memory/turn")
    st.caption("This tab retrieves context via /memory/turn, then generates assistant replies using LLM_EVAL.")
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.write(item["content"])
            if item.get("caption"):
                st.caption(item["caption"])

    if prompt := st.chat_input("Ask the memory layer something"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        try:
            response = client().post(
                "/memory/turn",
                {
                    "user_message": prompt,
                    "session_id": st.session_state.session_id,
                    "max_context_tokens": 1200,
                },
            )
            response.raise_for_status()
            payload = response.json()
            memory_context = str(payload.get("memory_context") or "")
            assistant_text = _generate_assistant_reply(
                user_message=prompt,
                memory_context=memory_context,
                chat_history=st.session_state.chat_history,
            )
            # Store assistant response so later turns can reference it.
            client().post(
                "/memory/write",
                {
                    "content": assistant_text,
                    "session_id": st.session_state.session_id,
                    "context_tags": ["assistant", "conversation", "streamlit"],
                },
            )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "caption": (
                        f"retrieved={payload.get('memories_retrieved')} "
                        f"stored={payload.get('memories_stored')}"
                    ),
                }
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
            st.info(_turn_context_message(payload if "payload" in locals() else {}))

with browse_tab:
    st.subheader("/memory/write")
    write_text = st.text_area("Memory content", "User prefers concise project updates.")
    write_type = st.selectbox(
        "Memory type",
        ["auto", "semantic_fact", "preference", "constraint", "episodic_event", "hypothesis"],
    )
    write_tags = st.text_input("Context tags", value="examples,streamlit")
    if st.button("Write memory"):
        payload = {
            "content": write_text,
            "session_id": st.session_state.session_id,
            "context_tags": [tag.strip() for tag in write_tags.split(",") if tag.strip()],
        }
        if write_type != "auto":
            payload["memory_type"] = write_type
        try:
            response = client().post("/memory/write", payload)
            response.raise_for_status()
            st.success(response.json())
        except Exception as exc:
            st.error(str(exc))

    st.subheader("/memory/read")
    read_query = st.text_input("Read query", value="What does the user prefer?")
    read_format = st.selectbox("Format", ["packet", "list", "llm_context"])
    if st.button("Read memory"):
        try:
            response = client().post(
                "/memory/read",
                {"query": read_query, "max_results": 10, "format": read_format},
            )
            response.raise_for_status()
            st.json(response.json())
        except Exception as exc:
            st.error(str(exc))

with session_tab:
    st.subheader("Session helpers")
    if st.button("Create session"):
        try:
            response = client().post("/session/create", {"ttl_hours": 24, "name": "streamlit-demo"})
            response.raise_for_status()
            payload = response.json()
            st.session_state.session_id = payload["session_id"]
            st.success(payload)
        except Exception as exc:
            st.error(str(exc))

    if st.button("Get session context"):
        try:
            response = client().get(f"/session/{st.session_state.session_id}/context")
            response.raise_for_status()
            st.json(response.json())
        except Exception as exc:
            st.error(str(exc))

    if st.button("Get memory stats"):
        try:
            response = client().get("/memory/stats")
            response.raise_for_status()
            st.json(response.json())
        except Exception as exc:
            st.error(str(exc))

with admin_tab:
    st.subheader("Read-only admin inspection")
    if not st.session_state.admin_api_key:
        st.info("Set CML_ADMIN_API_KEY to enable dashboard overview and retrieval testing.")
    else:
        if st.button("Dashboard overview"):
            try:
                response = client().get("/dashboard/overview", use_admin=True)
                response.raise_for_status()
                st.json(response.json())
            except Exception as exc:
                st.error(str(exc))

        admin_query = st.text_input("Admin retrieval query", value="user preferences")
        if st.button("Dashboard retrieval test"):
            try:
                response = client().post(
                    "/dashboard/retrieval",
                    {
                        "tenant_id": get_env("CML_TENANT_ID") or "default",
                        "query": admin_query,
                        "max_results": 5,
                        "format": "list",
                    },
                    use_admin=True,
                )
                response.raise_for_status()
                st.json(response.json())
            except Exception as exc:
                st.error(str(exc))
