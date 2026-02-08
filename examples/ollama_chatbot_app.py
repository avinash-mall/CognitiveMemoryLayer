"""
Streamlit chatbot with Cognitive Memory Layer – context from memory only.

The assistant does NOT receive conversation history. Each turn uses only:
  - System prompt
  - Current user message
  - Tool calls (memory_read / memory_write) and their results within that turn

Memory reads and writes are shown in the "Memory activity" section so you can
see what was stored and recalled.

Prerequisites:
  - Ollama running (e.g. gpt-oss:20b), memory API running, .env configured.
  - Run: streamlit run examples/ollama_chatbot_app.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from openai import OpenAI
from memory_client import CognitiveMemoryClient


# ---------------------------------------------------------------------------
# Config (from env / .env)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("LLM__MODEL", "gpt-oss:20b")
MEMORY_API_URL = os.environ.get("MEMORY_API_URL", "http://localhost:8000")
MEMORY_API_KEY = os.environ.get("AUTH__API_KEY", "test-api-key")

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Store important information in long-term memory. "
                "Use when the user shares personal information, preferences, or significant facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to store. Be specific and factual."},
                    "memory_type": {
                        "type": "string",
                        "enum": ["semantic_fact", "preference", "constraint", "episodic_event"],
                        "description": "Type: semantic_fact, preference, constraint, episodic_event.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": (
                "Retrieve relevant memories. Call before answering questions "
                "about the user's preferences, history, or personal information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query for what you need."},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful assistant with long-term memory.

You do NOT see past conversation turns. You only see this message and the current user message.
Use memory_read to recall what you know about the user before answering questions about them.
Use memory_write when the user shares important information (name, preferences, facts, constraints).
Be natural and concise. For constraints (e.g. allergies), always remember them."""


def get_session_id():
    return st.session_state.get("session_id", "streamlit-session")


def get_memory_client():
    if "memory_client" not in st.session_state:
        st.session_state.memory_client = CognitiveMemoryClient(
            base_url=MEMORY_API_URL,
            api_key=MEMORY_API_KEY,
            timeout=120.0,
        )
    return st.session_state.memory_client


def get_openai_client():
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=300.0,
        )
    return st.session_state.openai_client


def memory_activity_append(kind: str, detail: dict):
    if "memory_activity" not in st.session_state:
        st.session_state.memory_activity = []
    st.session_state.memory_activity.append({"kind": kind, "detail": detail})


def execute_tool(name: str, arguments: dict, memory_client: CognitiveMemoryClient) -> str:
    session_id = get_session_id()
    if name == "memory_write":
        result = memory_client.write(
            arguments["content"],
            session_id=session_id,
            context_tags=["conversation"],
            memory_type=arguments.get("memory_type"),
        )
        memory_activity_append("write", {
            "content": arguments["content"],
            "memory_type": arguments.get("memory_type"),
            "success": result.success,
            "message": result.message,
        })
        return json.dumps({"success": result.success, "message": result.message})

    if name == "memory_read":
        result = memory_client.read(arguments["query"], format="llm_context")
        text = result.llm_context or "No relevant memories found."
        memory_activity_append("read", {
            "query": arguments["query"],
            "total_count": result.total_count,
            "snippet": (text[:500] + "…") if len(text) > 500 else text,
        })
        return text

    return json.dumps({"error": f"Unknown tool: {name}"})


def chat_turn(user_message: str) -> str:
    """One turn: no conversation history. Only system + user message + tool calls until final reply."""
    memory_client = get_memory_client()
    openai_client = get_openai_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    max_rounds = 8
    for _ in range(max_rounds):
        response = openai_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=MEMORY_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args, memory_client)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            return (msg.content or "").strip()

    return "(Max tool rounds reached.)"


def main():
    st.set_page_config(page_title="Memory Chat", page_icon="🧠", layout="wide")

    st.title("🧠 Memory-powered chatbot")
    st.caption("Context comes from memory only — no conversation history is sent to the model.")

    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    if "memory_activity" not in st.session_state:
        st.session_state.memory_activity = []

    with st.sidebar:
        st.subheader("Session")
        session_id = st.text_input("Session ID", value=get_session_id(), key="session_id_input")
        st.session_state["session_id"] = session_id

        try:
            mem = get_memory_client()
            health = mem.health()
            st.success(f"Memory API: {health.get('status', 'ok')}")
        except Exception as e:
            st.error(f"Memory API: {str(e)[:80]}")

        st.subheader("Memory stats")
        try:
            stats = get_memory_client().stats()
            st.metric("Total memories", stats.total_memories)
            st.metric("Active", stats.active_memories)
        except Exception:
            st.write("—")

        if st.button("Clear memory activity log"):
            st.session_state.memory_activity = []
            st.rerun()

    col_chat, col_memory = st.columns([2, 1])

    with col_chat:
        for entry in st.session_state.chat_display:
            role = entry["role"]
            content = entry["content"]
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)

    with col_memory:
        st.subheader("Memory activity")
        if not st.session_state.memory_activity:
            st.info("Memory reads and writes will appear here.")
        else:
            for i, act in enumerate(reversed(st.session_state.memory_activity[-50:]), 1):
                kind = act["kind"]
                d = act["detail"]
                if kind == "write":
                    with st.expander(f"📥 Write: {d.get('content', '')[:40]}…" if len(d.get("content", "")) > 40 else f"📥 Write: {d.get('content', '')}"):
                        st.write("**Content:**", d.get("content", ""))
                        st.write("**Type:**", d.get("memory_type", "—"))
                        st.write("**Result:**", d.get("message", "—"))
                else:
                    with st.expander(f"📤 Read: “{d.get('query', '')[:35]}…”"):
                        st.write("**Query:**", d.get("query", ""))
                        st.write("**Memories found:**", d.get("total_count", 0))
                        st.write("**Snippet:**")
                        st.text(d.get("snippet", "—"))

    if prompt := st.chat_input("Message (context from memory only)..."):
        st.session_state.chat_display.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = chat_turn(prompt)
            st.session_state.chat_display.append({"role": "assistant", "content": reply})
            st.markdown(reply)
        st.rerun()


if __name__ == "__main__":
    main()
