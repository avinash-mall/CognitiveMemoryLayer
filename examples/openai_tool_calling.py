"""OpenAI tool calling with Cognitive Memory Layer.

Set AUTH__API_KEY, CML_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL in .env.
"""

import json
import os
from pathlib import Path
from typing import Optional
from uuid import UUID

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI
from cml import CognitiveMemoryLayer

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
                        "enum": [
                            "episodic_event",
                            "semantic_fact",
                            "preference",
                            "constraint",
                            "hypothesis",
                        ],
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
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": "Update or provide feedback on a memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "feedback": {"type": "string", "enum": ["correct", "incorrect", "outdated"]},
                },
                "required": ["memory_id", "feedback"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_forget",
            "description": "Forget information when the user requests deletion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "action": {"type": "string", "enum": ["delete", "archive"]},
                },
                "required": ["query"],
            },
        },
    },
]


class MemoryEnabledAssistant:
    def __init__(self, session_id: str, model: Optional[str] = None):
        self.session_id = session_id
        self.model = (
            model or os.environ.get("OPENAI_MODEL") or os.environ.get("LLM__MODEL") or ""
        ).strip()
        base_url = (
            os.environ.get("CML_BASE_URL")
            or os.environ.get("MEMORY_API_URL")
            or "http://localhost:8000"
        ).strip()
        self.openai = OpenAI()
        self.memory = CognitiveMemoryLayer(
            api_key=os.environ.get("CML_API_KEY") or os.environ.get("AUTH__API_KEY"),
            base_url=base_url,
        )
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with long-term memory. Use memory_read before answering about the user; memory_write for new info; memory_update for corrections; memory_forget when asked to forget.",
            }
        ]

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "memory_write":
                r = self.memory.write(
                    args["content"],
                    session_id=self.session_id,
                    context_tags=["conversation"],
                    memory_type=args.get("memory_type"),
                )
                return json.dumps({"success": r.success, "message": r.message})
            if name == "memory_read":
                r = self.memory.read(args["query"], response_format="llm_context")
                return r.context or "No relevant memories found."
            if name == "memory_update":
                r = self.memory.update(
                    memory_id=UUID(args["memory_id"]),
                    feedback=args["feedback"],
                )
                return json.dumps({"success": r.success, "version": r.version})
            if name == "memory_forget":
                r = self.memory.forget(
                    query=args["query"],
                    action=args.get("action", "archive"),
                )
                return json.dumps({"affected_count": r.affected_count})
            return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        while True:
            resp = self.openai.chat.completions.create(
                model=self.model,
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
                    print(f"  [Tool: {tc.function.name}] {args}")
                    result = self._execute_tool(tc.function.name, args)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content

    def close(self):
        self.memory.close()


def main():
    print("=" * 60)
    print("OpenAI Tool Calling + Cognitive Memory Layer")
    print("=" * 60)
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env")
        return
    if not (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM__MODEL")):
        print("Set OPENAI_MODEL or LLM__MODEL in .env")
        return
    assistant = MemoryEnabledAssistant(session_id="openai-demo")
    try:
        print("Assistant: Hello! I can remember things. What would you like to tell me?\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "bye"):
                print("\nAssistant: Goodbye!")
                break
            if not user_input:
                continue
            print(f"\nAssistant: {assistant.chat(user_input)}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
