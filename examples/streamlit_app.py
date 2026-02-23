"""
Streamlit Chat App for Cognitive Memory Layer (CML)

Run this via:
streamlit run examples/streamlit_app.py
"""

import os
from io import StringIO

import httpx
import streamlit as st

# ------------------------------------------------------------------------------
# App Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="CML Chat Studio", layout="wide", page_icon="🧠")

# Try loading from .env if available
try:
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Initialize session state for connection config if not present
if "base_url" not in st.session_state:
    st.session_state.base_url = os.environ.get("CML_BASE_URL", "http://localhost:8000/api/v1")
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("CML_API_KEY", "")
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = "streamlit-demo-session"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # For UI display


# CML Client
class CMLClient:
    def __init__(self, base_url: str, api_key: str):
        # ensure no trailing slashes or duplicate /api/v1
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/api/v1"):
            base_url = f"{base_url}/api/v1"
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json", "X-API-Key": api_key}
        self.timeout = 60.0

    def get(self, endpoint: str) -> httpx.Response:
        return httpx.get(f"{self.base_url}{endpoint}", headers=self.headers, timeout=self.timeout)

    def post(self, endpoint: str, payload: dict) -> httpx.Response:
        return httpx.post(
            f"{self.base_url}{endpoint}", headers=self.headers, json=payload, timeout=self.timeout
        )


@st.cache_resource(ttl=60)
def get_client() -> CMLClient:
    return CMLClient(st.session_state.base_url, st.session_state.api_key)


# ------------------------------------------------------------------------------
# Header & Intro
# ------------------------------------------------------------------------------
st.title("🧠 Cognitive Memory Layer - Streamlit Chat Studio")
st.markdown("""
Welcome to the interactive Cognitive Memory Layer (CML) application! This app demonstrates how to integrate
persistent memory into conversational interfaces and manage memory directly.

**Features demonstrated in this app (`/api/v1` routes):**
1. **Chat (`/turn`)**: Sends a message and retrieves fully contextualized memory to augment LLMs, optionally storing the new interaction.
2. **Ingest (`/write`)**: Upload documents (txt, md) that are automatically chunked and written into memory.
3. **Explore (`/read`, `/update`)**: Query memories using multiple formats (e.g., packet, list), and provide feedback (`correct`, `outdated`) to trigger reconsolidation.
4. **Manage Tools (`/forget`, `/session/create`, `/stats`, `/admin/consolidate`)**: Administer the semantic graph, view current stats, create volatile sessions, and consolidate episodic events into semantic facts.

Configure your connection settings in the sidebar!
""")

# ------------------------------------------------------------------------------
# Sidebar Config
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("API Connection", expanded=True):
        st.session_state.base_url = st.text_input("Base URL", value=st.session_state.base_url)
        st.session_state.api_key = st.text_input(
            "API Key", value=st.session_state.api_key, type="password"
        )

        if st.button("Test Connection"):
            try:
                cli = get_client()
                resp = cli.get("/health")
                if resp.status_code == 200:
                    st.success("API healthy! 🟢")
                else:
                    st.error(f"Error {resp.status_code}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    with st.expander("Active Session", expanded=True):
        st.session_state.active_session_id = st.text_input(
            "Session ID", value=st.session_state.active_session_id
        )

        if st.button("New Session (`/create`)"):
            try:
                cli = get_client()
                resp = cli.post("/session/create", {"ttl_hours": 24})
                if resp.status_code == 200:
                    sid = resp.json().get("session_id")
                    st.session_state.active_session_id = sid
                    st.success(f"Created Session: {sid}")
                    st.session_state.chat_history = []
            except Exception as e:
                st.error(str(e))

# ------------------------------------------------------------------------------
# Main Tabs
# ------------------------------------------------------------------------------
tabs = st.tabs(
    [
        "💬 Chat (/turn)",
        "📄 Ingestion (/write)",
        "🔍 Explore (/read & /update)",
        "🛠️ Management (Stats, Forget, Consolidate)",
    ]
)


def display_json(title, data):
    with st.expander(title):
        st.json(data)


# ==============================================================================
# TAB 1: Chat (Process Turn)
# ==============================================================================
with tabs[0]:
    st.markdown("### Chat with Memories")
    st.info(
        "The Chat interface calls `/memory/turn`. It automatically queries the CML for context relevant to your message, injects the memories, and optionally stores the turn in the database."
    )

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metadata"):
                st.caption(msg["metadata"])

    # Chat input
    if prompt := st.chat_input("What would you like to say?"):
        # UI Add user msg
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call /memory/turn
        cli = get_client()
        with st.spinner("Processing turn with Cognitive Memory Layer..."):
            try:
                resp = cli.post(
                    "/memory/turn",
                    {
                        "user_message": prompt,
                        "session_id": st.session_state.active_session_id,
                        "max_context_tokens": 1500,
                    },
                )

                if resp.status_code == 200:
                    data = resp.json()

                    # Simulated Assistant Response (In a real app, you'd pass data["memory_context"] to an LLM here)
                    # For demo purposes, we will just echo back what we found.
                    if data.get("memories_retrieved", 0) > 0:
                        ass_response = (
                            "I have reviewed your memory context to process this request. \n\n"
                        )
                        ass_response += f"**Context Retrieved:**\n```text\n{data.get('memory_context', '')}\n```"
                    else:
                        ass_response = "I couldn't recall anything relevant from my memory regarding that, but I'm ready to help!"

                    meta = f"Retrieved: {data.get('memories_retrieved', 0)} | Stored: {data.get('memories_stored', 0)} | Reconsolidation: {data.get('reconsolidation_applied', False)}"

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": ass_response, "metadata": meta}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(ass_response)
                        st.caption(meta)
                else:
                    st.error(f"Error processing turn: {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect: {e}")

# ==============================================================================
# TAB 2: Ingestion
# ==============================================================================
with tabs[1]:
    st.markdown("### Upload Documents to Memory")
    st.write("Extract text from files and write it into the memory layer using `/memory/write`.")

    uploaded_file = st.file_uploader("Choose a text file (.txt, .md)", type=["txt", "md"])
    mem_type = st.selectbox(
        "Memory Type",
        ["auto", "semantic_fact", "preference", "constraint", "episodic_event", "procedure"],
    )
    tags = st.text_input("Context Tags (comma separated)", "document, upload")

    if st.button("Ingest Document") and uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        content = stringio.read()

        tag_list = [x.strip() for x in tags.split(",") if x.strip()]

        with st.spinner("Ingesting to CML..."):
            try:
                payload = {
                    "content": content,
                    "session_id": st.session_state.active_session_id,
                    "context_tags": tag_list,
                }
                if mem_type != "auto":
                    payload["memory_type"] = mem_type

                cli = get_client()
                resp = cli.post("/memory/write", payload)
                if resp.status_code == 200:
                    st.success(
                        f"Successfully ingested {len(content)} characters! Memory ID: {resp.json().get('memory_id')}"
                    )
                    display_json("Response Payload", resp.json())
                else:
                    st.error(f"Failed to ingest: {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect: {e}")

# ==============================================================================
# TAB 3: Explore & Update
# ==============================================================================
with tabs[2]:
    st.markdown("### Explore Existing Memories")

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Search Query", "What do you know?")
    with col2:
        res_format = st.selectbox("Format", ["packet", "list", "llm_context"])

    if st.button("Read Memory (`/read`)"):
        with st.spinner("Retrieving memories..."):
            try:
                cli = get_client()
                resp = cli.post(
                    "/memory/read", {"query": query, "max_results": 10, "format": res_format}
                )

                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Found {data.get('total_count', 0)} memories ({data.get('elapsed_ms', 0):.2f}ms)"
                    )

                    if res_format == "packet":
                        # Grouped by buckets for packet response
                        buckets = ["facts", "preferences", "constraints", "episodes"]
                        for b in buckets:
                            items = data.get(b, [])
                            if items:
                                st.markdown(f"**{b.title()} ({len(items)})**")
                                for i, m in enumerate(items):
                                    with st.expander(
                                        f"[{m.get('id')[:8]}] {m.get('text', '')[:60]}... (Conf: {m.get('confidence', 0):.2f})"
                                    ):
                                        st.write(f"**Type:** {m.get('type')}")
                                        st.write(f"**Text:** {m.get('text')}")

                                        # Update feature right in the viewer!
                                        fb_col1, fb_col2, fb_col3 = st.columns(3)
                                        if fb_col1.button(
                                            "Mark Correct", key=f"corr_{m.get('id')}"
                                        ):
                                            up_resp = cli.post(
                                                "/memory/update",
                                                {"memory_id": m.get("id"), "feedback": "correct"},
                                            )
                                            if up_resp.status_code == 200:
                                                st.toast("Memory marked correct.")
                                        if fb_col2.button(
                                            "Mark Outdated", key=f"outd_{m.get('id')}"
                                        ):
                                            up_resp = cli.post(
                                                "/memory/update",
                                                {"memory_id": m.get("id"), "feedback": "outdated"},
                                            )
                                            if up_resp.status_code == 200:
                                                st.toast("Memory marked outdated.")
                                        if fb_col3.button("Force Delete", key=f"del_{m.get('id')}"):
                                            # Using forget to delete fully
                                            del_resp = cli.post(
                                                "/memory/forget",
                                                {"memory_ids": [m.get("id")], "action": "delete"},
                                            )
                                            if del_resp.status_code == 200:
                                                st.toast("Memory deleted.")

                    elif res_format == "llm_context":
                        st.markdown("**LLM Context Output**")
                        st.code(data.get("llm_context", ""), language="text")

                    elif res_format == "list":
                        st.write(data.get("memories", []))
                else:
                    st.error(f"Read failed: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# ==============================================================================
# TAB 4: Management Tools
# ==============================================================================
with tabs[3]:
    st.markdown("### Manage CML and Statistics")

    st.subheader("Stats (`/stats`)")
    if st.button("Refresh Stats"):
        cli = get_client()
        resp = cli.get("/memory/stats")
        if resp.status_code == 200:
            stats = resp.json()
            st.metric("Total Memories", stats.get("total_memories", 0))

            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("Active Memories", stats.get("active_memories", 0))
            scol2.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.2f}")
            scol3.metric("Size", f"{stats.get('estimated_size_mb', 0):.2f} MB")

            st.write("Breakdown by type:", stats.get("by_type", {}))
        else:
            st.error(f"Stats error: {resp.text}")

    st.divider()

    st.subheader("Forget / Cleanup (`/forget`)")
    f_query = st.text_input("Forget Query (Leave blank to use action globally)")
    f_action = st.selectbox("Action", ["archive", "silence", "delete"])
    if st.button("Execute Forget"):
        cli = get_client()
        payload = {"action": f_action, "query": f_query}
        resp = cli.post("/memory/forget", payload)
        if resp.status_code == 200:
            st.success(f"Affected {resp.json().get('affected_count')} memories via {f_action}.")
        else:
            st.error(f"Forget error: {resp.text}")

    st.divider()

    st.subheader("Consolidation Pipeline (`/admin/consolidate`)")
    st.info(
        "Triggers the consolidation process that reviews episodes, clusters behaviors, and extracts semantic facts."
    )
    tc_id = st.text_input(
        "Tenant URL slug (for config):", value=st.session_state.api_key
    )  # Usually user_id
    if st.button("Run Data Consolidation"):
        with st.spinner("Consolidation running (this may take a few seconds)..."):
            cli = get_client()
            # Note: route might need adjusting if base_url is strictly /api/v1
            admin_url = cli.base_url.replace("/api/v1", "") + f"/api/v1/admin/consolidate/{tc_id}"
            try:
                resp = httpx.post(admin_url, headers=cli.headers, timeout=cli.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("Consolidation executed successfully!")
                    display_json("Consolidation Report", data)
                else:
                    st.error(f"Consolidation error: {resp.text}")
            except Exception as e:
                st.error(f"Failed: {e}")
