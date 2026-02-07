"""
Ollama Chat Test - Cognitive Memory Layer

Automated end-to-end test that exercises the full memory pipeline using
local Ollama models:
  - LLM:       gpt-oss:20b   (chat + tool calling)
  - Embedding: mxbai-embed-large (1024-dim vectors)

The script is NON-INTERACTIVE: it runs a scripted conversation, prints
every tool call and response, and verifies that memories were stored and
retrieved correctly.

Prerequisites:
    1. Ollama running locally with gpt-oss:20b and mxbai-embed-large
    2. Infrastructure services:
       docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
    3. API server running:
       uvicorn src.api.app:app --host 0.0.0.0 --port 8000
    4. .env configured for Ollama (see .env.example)

Usage:
    cd examples
    python ollama_chat_test.py
"""

import json
import os
import sys
import time

# Ensure UTF-8 output on Windows terminals that default to cp1252.
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow running from the examples/ directory
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from memory_client import CognitiveMemoryClient


# ---------------------------------------------------------------------------
# Configuration - read from environment / .env, with sensible local defaults
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("LLM__MODEL", "gpt-oss:20b")
MEMORY_API_URL = os.environ.get("MEMORY_API_URL", "http://localhost:8000")
MEMORY_API_KEY = os.environ.get("AUTH__API_KEY", "test-api-key")


# ---------------------------------------------------------------------------
# Memory tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------
MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Store important information in long-term memory. "
                "Use when the user shares personal information, preferences, "
                "or significant facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to store. Be specific and factual.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": [
                            "semantic_fact",
                            "preference",
                            "constraint",
                            "episodic_event",
                        ],
                        "description": (
                            "Type of memory. 'semantic_fact' for facts, "
                            "'preference' for preferences, "
                            "'constraint' for rules/restrictions."
                        ),
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
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what information you need.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Assistant class (adapted from openai_tool_calling.py for Ollama)
# ---------------------------------------------------------------------------
class OllamaMemoryAssistant:
    """Chat assistant backed by local Ollama with persistent memory."""

    def __init__(self):
        self.openai = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=300.0,  # local models can be slow on first load
        )
        self.memory = CognitiveMemoryClient(
            base_url=MEMORY_API_URL,
            api_key=MEMORY_API_KEY,
            timeout=120.0,
        )
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with long-term memory.\n\n"
                    "Guidelines:\n"
                    "1. Use memory_write when the user shares important information "
                    "(name, preferences, facts about themselves, constraints).\n"
                    "2. Use memory_read BEFORE answering questions about the user.\n"
                    "3. Be natural and conversational.\n"
                    "4. For constraints (allergies, restrictions), ALWAYS remember them."
                ),
            }
        ]

    # -- tool execution -----------------------------------------------------
    def _execute_tool(self, name: str, arguments: dict) -> str:
        try:
            if name == "memory_write":
                result = self.memory.write(
                    arguments["content"],
                    session_id="ollama-test",
                    context_tags=["conversation"],
                    memory_type=arguments.get("memory_type"),
                )
                return json.dumps({"success": result.success, "message": result.message})

            elif name == "memory_read":
                result = self.memory.read(
                    arguments["query"],
                    format="llm_context",
                )
                return result.llm_context or "No relevant memories found."

            return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # -- chat turn ----------------------------------------------------------
    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        max_rounds = 6  # safety limit on tool-call loops
        for _ in range(max_rounds):
            response = self.openai.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=self.messages,
                tools=MEMORY_TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                # Record assistant message with tool calls
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
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
            else:
                # Final text reply
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content

        return "(max tool-call rounds reached)"

    def close(self):
        self.memory.close()


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_test():
    separator("Ollama + Cognitive Memory Layer - End-to-End Test")

    # Longer timeout for local models (first call loads the model into RAM).
    api_timeout = float(os.environ.get("MEMORY_API_TIMEOUT", "120"))

    # ---- 1. Health check --------------------------------------------------
    print("[1/5] Checking API health...")
    client = CognitiveMemoryClient(
        base_url=MEMORY_API_URL, api_key=MEMORY_API_KEY, timeout=api_timeout,
    )
    try:
        health = client.health()
        print(f"  API status: {health['status']}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Make sure the API server is running:")
        print("    uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
        return
    finally:
        client.close()

    # ---- 2. Test direct memory write + read (no LLM) ---------------------
    separator("Direct Memory API Test (no LLM)")
    client = CognitiveMemoryClient(
        base_url=MEMORY_API_URL, api_key=MEMORY_API_KEY, timeout=api_timeout,
    )
    try:
        print("[2/5] Writing a test memory...")
        wr = client.write(
            "Test user enjoys mountain biking on weekends",
            session_id="ollama-test",
            context_tags=["test"],
            memory_type="preference",
        )
        print(f"  Write: success={wr.success}  id={wr.memory_id}")

        print("  Reading back...")
        rd = client.read("weekend hobbies", format="llm_context")
        print(f"  Read: {rd.total_count} memories found")
        if rd.llm_context:
            for line in rd.llm_context.strip().splitlines()[:5]:
                print(f"    {line}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        return
    finally:
        client.close()

    # ---- 3. Chat conversation with tool calling --------------------------
    separator("Chat Conversation with Tool Calling")
    print(f"  Model:  {OLLAMA_MODEL}")
    print(f"  Ollama: {OLLAMA_BASE_URL}\n")

    assistant = OllamaMemoryAssistant()

    conversation = [
        "Hi! My name is Alex and I'm a software engineer at a startup.",
        "I prefer Python for backend work, and I'm allergic to shellfish.",
        "What do you know about me?",
        "What dietary restrictions should you remember for me?",
    ]

    try:
        for i, user_msg in enumerate(conversation, 1):
            print(f"[3/5] Turn {i}/{len(conversation)}")
            print(f"  User: {user_msg}")
            t0 = time.time()
            reply = assistant.chat(user_msg)
            elapsed = time.time() - t0
            print(f"  Assistant ({elapsed:.1f}s): {reply}\n")
    except Exception as e:
        print(f"  CHAT FAILED: {e}")
        import traceback; traceback.print_exc()
    finally:
        assistant.close()

    # ---- 4. Verify memories persisted ------------------------------------
    separator("Verify Persisted Memories")
    client = CognitiveMemoryClient(
        base_url=MEMORY_API_URL, api_key=MEMORY_API_KEY, timeout=api_timeout,
    )
    try:
        print("[4/5] Querying stored memories about 'Alex'...")
        rd = client.read("Alex software engineer", format="llm_context")
        print(f"  Found {rd.total_count} memories")
        if rd.llm_context:
            for line in rd.llm_context.strip().splitlines()[:8]:
                print(f"    {line}")

        print("\n  Querying for dietary restrictions...")
        rd2 = client.read("allergies dietary restrictions", format="llm_context")
        print(f"  Found {rd2.total_count} memories")
        if rd2.llm_context:
            for line in rd2.llm_context.strip().splitlines()[:5]:
                print(f"    {line}")
    except Exception as e:
        print(f"  VERIFY FAILED: {e}")
        import traceback; traceback.print_exc()
    finally:
        client.close()

    # ---- 5. Stats --------------------------------------------------------
    separator("Memory Stats")
    client = CognitiveMemoryClient(
        base_url=MEMORY_API_URL, api_key=MEMORY_API_KEY, timeout=api_timeout,
    )
    try:
        print("[5/5] Fetching memory statistics...")
        stats = client.stats()
        print(f"  Total memories:  {stats.total_memories}")
        print(f"  Active memories: {stats.active_memories}")
        print(f"  By type:         {stats.by_type}")
        print(f"  Avg confidence:  {stats.avg_confidence:.2f}")
    except Exception as e:
        print(f"  STATS FAILED: {e}")
    finally:
        client.close()

    separator("TEST COMPLETE")


if __name__ == "__main__":
    run_test()
