"""OpenAI chat loop that uses /memory/turn for retrieval and persistence."""

from __future__ import annotations

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
    "name": "chat_with_memory",
    "kind": "python",
    "summary": "OpenAI chat loop backed by memory.turn and explicit assistant writes.",
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


def main() -> int:
    print_header("CML Chat With Memory")
    openai_config = openai_settings()
    client_kwargs = {"api_key": openai_config["api_key"]}
    if openai_config["base_url"] is not None:
        client_kwargs["base_url"] = openai_config["base_url"]
    openai_client = OpenAI(**client_kwargs)
    session_id = "examples-chat"

    try:
        with CognitiveMemoryLayer(config=build_cml_config(timeout=60.0)) as memory:
            for user_input in iter_user_inputs(
                "You: ",
                default_inputs=[
                    "Remember that I prefer green tea over coffee.",
                    "What drink do I prefer?",
                    "quit",
                ],
            ):
                if user_input.lower() in {"quit", "exit"}:
                    break

                turn = memory.turn(
                    user_message=user_input, session_id=session_id, max_context_tokens=800
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant with persistent memory. "
                            "Use the provided memory context when it is relevant.\n\n"
                            f"{turn.memory_context}"
                        ),
                    },
                    {"role": "user", "content": user_input},
                ]
                completion = openai_client.chat.completions.create(
                    model=str(openai_config["model"]),
                    messages=cast("Any", messages),
                )
                assistant_text = completion.choices[0].message.content or ""
                memory.write(
                    assistant_text,
                    session_id=session_id,
                    context_tags=["assistant", "conversation"],
                )
                print(f"Assistant: {assistant_text}")
                print(f"Retrieved memories: {turn.memories_retrieved}\n")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
