"""Streamlit chatbot with Cognitive Memory Layer - context from memory only.

Set OLLAMA_BASE_URL, LLM__MODEL, CML_BASE_URL, AUTH__API_KEY in .env.
Run: streamlit run examples/ollama_chatbot_app.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import streamlit as st
from openai import OpenAI
from cml import CognitiveMemoryLayer


OLLAMA_BASE = (os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip()
OLLAMA_MODEL = (os.environ.get("LLM__MODEL") or "").strip()
MEMORY_BASE = (os.environ.get("CML_BASE_URL") or os.environ.get("MEMORY_API_URL") or "").strip()
MEMORY_KEY = os.environ.get("CML_API_KEY") or os.environ.get("AUTH__API_KEY") or ""

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Store important information in long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "memory_type": {"type": "string", "enum": ["semantic_fact", "preference", "constraint", "episodic_event"]},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Retrieve relevant memories.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
]

SYSTEM_PROMPT = """You are a helpful assistant with long-term memory.
You do NOT see past turns. Use memory_read to recall what you know.
Use memory_write when the user shares important information. Be concise."""


def get_session_id():
    return st.session_state.get("session_id", "streamlit-session")


def get_memory():
    if "memory" not in st.session_state:
        st.session_state.memory = CognitiveMemoryLayer(
            api_key=MEMORY_KEY,
            base_url=MEMORY_BASE,
            timeout=120.0,
        )
    return st.session_state.memory


def get_openai():
    if "openai" not in st.session_state:
        st.session_state.openai = OpenAI(base_url=OLLAMA_BASE, api_key="ollama", timeout=300.0)
    return st.session_state.openai


def append_activity(kind: str, detail: dict):
    if "memory_activity" not in st.session_state:
        st.session_state.memory_activity = []
    st.session_state.memory_activity.append({"kind": kind, "detail": detail})


def execute_tool(name: str, args: dict, memory: CognitiveMemoryLayer) -> str:
    sid = get_session_id()
    if name == "memory_write":
        r = memory.write(
            args["content"],
            session_id=sid,
            context_tags=["conversation"],
            memory_type=args.get("memory_type"),
        )
        append_activity("write", {"content": args["content"], "success": r.success})
        return json.dumps({"success": r.success, "message": r.message})
    if name == "memory_read":
        r = memory.read(args["query"], response_format="llm_context")
        text = r.context or "No relevant memories found."
        append_activity("read", {"query": args["query"], "total_count": r.total_count, "snippet": text[:500]})
        return text
    return json.dumps({"error": f"Unknown: {name}"})


def chat_turn(user_message: str) -> str:
    memory = get_memory()
    openai_client = get_openai()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
    for _ in range(8):
        resp = openai_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=MEMORY_TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args, memory)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            return (msg.content or "").strip()
    return "(Max tool rounds)"


def main():
    st.set_page_config(page_title="Memory Chat", page_icon="🧠", layout="wide")
    if not OLLAMA_BASE or not OLLAMA_MODEL or not MEMORY_BASE:
        st.error("Set OLLAMA_BASE_URL, LLM__MODEL, CML_BASE_URL in .env")
        return
    st.title("🧠 Memory-powered chatbot")
    st.caption("Context from memory only — no conversation history sent.")
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    if "memory_activity" not in st.session_state:
        st.session_state.memory_activity = []
    with st.sidebar:
        session_id = st.text_input("Session ID", value=get_session_id(), key="sid")
        st.session_state["session_id"] = session_id
        try:
            mem = get_memory()
            h = mem.health()
            st.success(f"Memory API: {h.status}")
        except Exception as e:
            st.error(str(e)[:80])
        try:
            s = get_memory().stats()
            st.metric("Total memories", s.total_memories)
            st.metric("Active", s.active_memories)
        except Exception:
            st.write("—")
        if st.button("Clear activity log"):
            st.session_state.memory_activity = []
            st.rerun()
    col_chat, col_mem = st.columns([2, 1])
    with col_chat:
        for e in st.session_state.chat_display:
            with st.chat_message(e["role"]):
                st.markdown(e["content"])
    with col_mem:
        st.subheader("Memory activity")
        if not st.session_state.memory_activity:
            st.info("Reads and writes will appear here.")
        else:
            for act in reversed(st.session_state.memory_activity[-50:]):
                d = act["detail"]
                if act["kind"] == "write":
                    st.expander(f"📥 Write: {d.get('content', '')[:40]}…").write(d)
                else:
                    st.expander(f"📤 Read: {d.get('query', '')[:35]}…").write(d)
    if prompt := st.chat_input("Message..."):
        st.session_state.chat_display.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = chat_turn(prompt)
            st.session_state.chat_display.append({"role": "assistant", "content": reply})
            st.markdown(reply)
        st.rerun()


if __name__ == "__main__":
    main()
