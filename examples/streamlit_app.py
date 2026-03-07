"""Streamlit UI for direct interaction with the CML API."""

from __future__ import annotations

import httpx
import streamlit as st
from _shared import get_env, normalize_base_url

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
            assistant_text = payload.get("memory_context") or "No context returned."
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
