"""Anthropic tool-calling example backed by Cognitive Memory Layer."""

from __future__ import annotations

import json
from typing import Any, cast

from _shared import (
    anthropic_settings,
    build_cml_config,
    explain_connection_failure,
    iter_user_inputs,
    print_header,
)
from anthropic import Anthropic

from cml import CognitiveMemoryLayer

EXAMPLE_META = {
    "name": "anthropic_tool_calling",
    "kind": "python",
    "summary": "Anthropic tool-use loop that stores and retrieves memory.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": False,
    "requires_anthropic": True,
    "interactive": True,
    "timeout_sec": 120,
}

MEMORY_TOOLS = [
    {
        "name": "memory_write",
        "description": "Store explicit user information in long-term memory.",
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
        "description": "Retrieve relevant memory context before answering.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


class ClaudeMemoryAssistant:
    def __init__(self, session_id: str) -> None:
        settings = anthropic_settings()
        self.model = settings["model"]
        self.anthropic = Anthropic(api_key=settings["api_key"])
        self.memory = CognitiveMemoryLayer(config=build_cml_config(timeout=60.0))
        self.session_id = session_id
        self.messages: list[dict[str, Any]] = []
        self.system = (
            "You are a helpful assistant with memory tools. "
            "Use memory_write to store durable user facts and memory_read before answering."
        )

    def close(self) -> None:
        self.memory.close()

    def _run_tool(self, name: str, payload: dict[str, Any]) -> str:
        if name == "memory_write":
            response = self.memory.write(
                payload["content"],
                session_id=self.session_id,
                context_tags=["conversation"],
                memory_type=payload.get("memory_type"),
            )
            return json.dumps({"success": response.success, "memory_id": str(response.memory_id)})

        if name == "memory_read":
            read_resp = self.memory.read(payload["query"], response_format="llm_context")
            return read_resp.context or "No relevant memory found."

        return json.dumps({"error": f"Unknown tool: {name}"})

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        while True:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system,
                tools=cast("Any", MEMORY_TOOLS),
                messages=cast("Any", self.messages),
            )
            if response.stop_reason != "tool_use":
                text = "".join(
                    getattr(block, "text", "")
                    for block in response.content
                    if hasattr(block, "text")
                )
                self.messages.append({"role": "assistant", "content": response.content})
                return text

            self.messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = self._run_tool(block.name, cast("dict[str, Any]", block.input))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})


def main() -> int:
    print_header("CML Anthropic Tool Calling")
    assistant = ClaudeMemoryAssistant(session_id="examples-anthropic-tool")
    try:
        for user_input in iter_user_inputs(
            "You: ",
            default_inputs=[
                "Use memory_write to store this preference: I prefer oat milk in coffee.",
                "Use memory_read before answering: what coffee preference do I have?",
                "quit",
            ],
        ):
            if user_input.lower() in {"quit", "exit"}:
                break
            print(f"Claude: {assistant.chat(user_input)}\n")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1
    finally:
        assistant.close()


if __name__ == "__main__":
    raise SystemExit(main())
