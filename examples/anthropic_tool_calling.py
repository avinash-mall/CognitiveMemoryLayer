"""Anthropic Claude tool use with Cognitive Memory Layer.

Set CML_API_KEY, CML_BASE_URL, ANTHROPIC_API_KEY in .env.
"""

import json
import os
from pathlib import Path
from typing import Any, cast
from uuid import UUID

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from anthropic import Anthropic

from cml import CognitiveMemoryLayer

MEMORY_TOOLS = [
    {
        "name": "memory_write",
        "description": "Store important information in long-term memory.",
        "input_schema": {
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
    {
        "name": "memory_read",
        "description": "Retrieve relevant memories.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "memory_update",
        "description": "Update or provide feedback on a memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "feedback": {"type": "string", "enum": ["correct", "incorrect", "outdated"]},
            },
            "required": ["memory_id", "feedback"],
        },
    },
    {
        "name": "memory_forget",
        "description": "Forget information when the user requests deletion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "action": {"type": "string", "enum": ["delete", "archive", "silence"]},
            },
            "required": ["query"],
        },
    },
]


class ClaudeMemoryAssistant:
    def __init__(self, session_id: str, model: str = "claude-sonnet-4-20250514"):
        self.session_id = session_id
        self.model = model
        base_url = (os.environ.get("CML_BASE_URL") or "").strip() or "http://localhost:8000"
        self.anthropic = Anthropic()
        self.memory = CognitiveMemoryLayer(
            api_key=os.environ.get("CML_API_KEY"),
            base_url=base_url,
        )
        self.messages: list[dict[str, Any]] = []
        self.system = (
            "You are a helpful assistant with long-term memory. Use memory tools as needed."
        )

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "memory_write":
                r = self.memory.write(
                    args["content"],
                    session_id=self.session_id,
                    context_tags=["conversation"],
                    memory_type=args.get("memory_type"),
                )
                return json.dumps({"success": r.success})
            if name == "memory_read":
                read_r = self.memory.read(args["query"], response_format="llm_context")
                return read_r.context or "No relevant memories found."
            if name == "memory_update":
                update_r = self.memory.update(
                    memory_id=UUID(args["memory_id"]), feedback=args["feedback"]
                )
                return json.dumps({"success": update_r.success})
            if name == "memory_forget":
                forget_r = self.memory.forget(
                    query=args["query"], action=args.get("action", "archive")
                )
                return json.dumps({"affected_count": forget_r.affected_count})
            return json.dumps({"error": f"Unknown: {name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        while True:
            resp = self.anthropic.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system,
                tools=cast("Any", MEMORY_TOOLS),
                messages=cast("Any", self.messages),
            )
            if resp.stop_reason == "tool_use":
                self.messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        print(f"  [Tool: {block.name}] {block.input}")
                        result = self._execute_tool(block.name, cast("dict[str, Any]", block.input))
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": result}
                        )
                self.messages.append({"role": "user", "content": tool_results})
            else:
                text = "".join(getattr(b, "text", "") for b in resp.content if hasattr(b, "text"))
                self.messages.append({"role": "assistant", "content": resp.content})
                return text

    def close(self):
        self.memory.close()


def main():
    print("=" * 60)
    print("Anthropic Claude Tool Use + Cognitive Memory Layer")
    print("=" * 60)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in .env")
        return
    assistant = ClaudeMemoryAssistant(session_id="claude-demo")
    try:
        print("Claude: Hello! I can remember things. What would you like to share?\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "bye"):
                print("\nClaude: Goodbye!")
                break
            if not user_input:
                continue
            print(f"\nClaude: {assistant.chat(user_input)}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        assistant.close()


if __name__ == "__main__":
    main()

