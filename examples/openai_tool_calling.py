"""OpenAI tool-calling example backed by Cognitive Memory Layer."""

from __future__ import annotations

import json
from typing import Any, cast

from _shared import (
    build_cml_config,
    explain_connection_failure,
    iter_user_inputs,
    openai_settings,
    print_header,
)
from openai import OpenAI

from cml import CognitiveMemoryLayer

EXAMPLE_META = {
    "name": "openai_tool_calling",
    "kind": "python",
    "summary": "OpenAI function-calling loop that stores and retrieves memory.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": True,
    "requires_anthropic": False,
    "interactive": True,
    "timeout_sec": 120,
}

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Store explicit user information in long-term memory.",
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
            "description": "Retrieve relevant memory context before answering.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


class MemoryEnabledAssistant:
    def __init__(self, session_id: str) -> None:
        settings = openai_settings()
        self.model = str(settings["model"])
        client_kwargs = {"api_key": settings["api_key"]}
        if settings["base_url"] is not None:
            client_kwargs["base_url"] = settings["base_url"]
        self.openai = OpenAI(**client_kwargs)
        self.memory = CognitiveMemoryLayer(config=build_cml_config(timeout=60.0))
        self.session_id = session_id
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with memory tools. "
                    "Use memory_write to store durable user facts and memory_read before answering."
                ),
            }
        ]

    def close(self) -> None:
        self.memory.close()

    def _run_tool(self, name: str, arguments: str) -> str:
        payload = json.loads(arguments)
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
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=cast("Any", self.messages),
                tools=cast("Any", MEMORY_TOOLS),
                tool_choice="auto",
            )
            message = response.choices[0].message
            if not message.tool_calls:
                assistant_text = message.content or ""
                self.messages.append({"role": "assistant", "content": assistant_text})
                return assistant_text

            self.messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,  # type: ignore[union-attr]
                                "arguments": call.function.arguments,  # type: ignore[union-attr]
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )
            for call in message.tool_calls:
                tool_result = self._run_tool(
                    call.function.name,  # type: ignore[union-attr]
                    call.function.arguments,  # type: ignore[union-attr]
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_result,
                    }
                )


def main() -> int:
    print_header("CML OpenAI Tool Calling")
    assistant = MemoryEnabledAssistant(session_id="examples-openai-tool")
    try:
        for user_input in iter_user_inputs(
            "You: ",
            default_inputs=[
                "Use memory_write to store this preference: I prefer tea over coffee.",
                "Use memory_read before answering: what drink do I prefer?",
                "quit",
            ],
        ):
            if user_input.lower() in {"quit", "exit"}:
                break
            print(f"Assistant: {assistant.chat(user_input)}\n")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1
    finally:
        assistant.close()


if __name__ == "__main__":
    raise SystemExit(main())
