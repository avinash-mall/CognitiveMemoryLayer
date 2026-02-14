"""Ollama + Cognitive Memory Layer - non-interactive end-to-end test.

Set OLLAMA_BASE_URL (or OPENAI_BASE_URL), LLM__MODEL, CML_BASE_URL, AUTH__API_KEY in .env.
Run: python examples/ollama_chat_test.py
"""

import json
import os
import sys
import time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI
from cml import CognitiveMemoryLayer

OLLAMA_BASE_URL = (
    os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
).strip()
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
                    "memory_type": {
                        "type": "string",
                        "enum": ["semantic_fact", "preference", "constraint", "episodic_event"],
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
            "description": "Retrieve relevant memories.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


class OllamaMemoryAssistant:
    def __init__(self):
        self.openai = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=300.0)
        self.memory = CognitiveMemoryLayer(
            api_key=MEMORY_KEY,
            base_url=MEMORY_BASE,
            timeout=120.0,
        )
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with long-term memory. Use memory_write and memory_read as needed.",
            }
        ]

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "memory_write":
                r = self.memory.write(
                    args["content"],
                    session_id="ollama-test",
                    context_tags=["conversation"],
                    memory_type=args.get("memory_type"),
                )
                return json.dumps({"success": r.success, "message": r.message})
            if name == "memory_read":
                r = self.memory.read(args["query"], response_format="llm_context")
                return r.context or "No relevant memories found."
            return json.dumps({"error": f"Unknown: {name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        for _ in range(6):
            resp = self.openai.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=self.messages,
                tools=MEMORY_TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    print(f"    [tool] {tc.function.name}({args})")
                    result = self._execute_tool(tc.function.name, args)
                    print(f"    [result] {result[:200]}")
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content
        return "(max tool rounds)"

    def close(self):
        self.memory.close()


def run_test():
    print("\n" + "=" * 60)
    print("  Ollama + Cognitive Memory Layer - E2E Test")
    print("=" * 60 + "\n")
    if not OLLAMA_BASE_URL or not OLLAMA_MODEL or not MEMORY_BASE:
        print("Set OLLAMA_BASE_URL, LLM__MODEL, CML_BASE_URL in .env")
        return
    timeout = float(os.environ.get("MEMORY_API_TIMEOUT", "120"))

    # 1. Health
    print("[1/5] API health...")
    mem = CognitiveMemoryLayer(api_key=MEMORY_KEY, base_url=MEMORY_BASE, timeout=timeout)
    try:
        h = mem.health()
        print(f"  API: {h.status}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    finally:
        mem.close()

    # 2. Direct write/read
    print("\n[2/5] Direct write + read...")
    mem = CognitiveMemoryLayer(api_key=MEMORY_KEY, base_url=MEMORY_BASE, timeout=timeout)
    try:
        mem.write(
            "Test user enjoys mountain biking on weekends",
            session_id="ollama-test",
            context_tags=["test"],
            memory_type="preference",
        )
        r = mem.read("weekend hobbies", response_format="llm_context")
        print(f"  Read: {r.total_count} memories")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    finally:
        mem.close()

    # 3. Chat
    print("\n[3/5] Chat with tool calling...")
    assistant = OllamaMemoryAssistant()
    conv = [
        "Hi! My name is Alex and I'm a software engineer at a startup.",
        "I prefer Python for backend work, and I'm allergic to shellfish.",
        "What do you know about me?",
        "What dietary restrictions should you remember?",
    ]
    try:
        for i, user_msg in enumerate(conv, 1):
            print(f"  Turn {i}: User: {user_msg}")
            t0 = time.time()
            reply = assistant.chat(user_msg)
            print(f"  Assistant ({time.time() - t0:.1f}s): {reply}\n")
    finally:
        assistant.close()

    # 4. Verify
    print("[4/5] Verify persisted memories...")
    mem = CognitiveMemoryLayer(api_key=MEMORY_KEY, base_url=MEMORY_BASE, timeout=timeout)
    try:
        r = mem.read("Alex software engineer", response_format="llm_context")
        print(f"  Alex: {r.total_count} memories")
        r2 = mem.read("allergies dietary", response_format="llm_context")
        print(f"  Dietary: {r2.total_count} memories")
    finally:
        mem.close()

    # 5. Stats
    print("\n[5/5] Stats...")
    mem = CognitiveMemoryLayer(api_key=MEMORY_KEY, base_url=MEMORY_BASE, timeout=timeout)
    try:
        s = mem.stats()
        print(f"  Total: {s.total_memories}, Active: {s.active_memories}")
    finally:
        mem.close()
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run_test()
