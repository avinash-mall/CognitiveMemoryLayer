"""Simple chatbot with persistent memory (py-cml + OpenAI).

Set CML_API_KEY, CML_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL in .env.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI

from cml import CognitiveMemoryLayer


def chat_with_memory():
    base_url = (os.environ.get("CML_BASE_URL") or "").strip() or "http://localhost:8000"
    # Prefer LLM_INTERNAL__* when set (e.g. .env lines 20-22)
    model = (
        os.environ.get("LLM_INTERNAL__MODEL") or os.environ.get("OPENAI_MODEL") or ""
    ).strip()
    if not model:
        raise SystemExit("Set OPENAI_MODEL (or LLM_INTERNAL__MODEL) in .env")
    llm_base = (os.environ.get("LLM_INTERNAL__BASE_URL") or "").strip()
    if llm_base:
        api_key = (
            os.environ.get("LLM_INTERNAL__API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy"
        )
        openai_client = OpenAI(base_url=llm_base, api_key=api_key)
    else:
        openai_client = OpenAI()
    session_id = "chat-demo-001"
    print("Chat with Memory (type 'quit' to exit)\n")

    with CognitiveMemoryLayer(
        api_key=os.environ.get("CML_API_KEY"),
        base_url=base_url,
    ) as memory:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                break

            # Retrieve memory context and store user message (one turn call)
            turn = memory.turn(
                user_message=user_input,
                session_id=session_id,
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant with persistent memory. "
                        "Use the following memories to personalize your responses:\n\n"
                        f"{turn.memory_context}"
                    ),
                },
                {"role": "user", "content": user_input},
            ]

            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
            )
            assistant_msg = response.choices[0].message.content

            memory.write(
                assistant_msg,
                session_id=session_id,
                context_tags=["conversation", "assistant"],
            )

            print(f"Assistant: {assistant_msg}\n")
            print(f"  [memories retrieved: {turn.memories_retrieved}]")


if __name__ == "__main__":
    chat_with_memory()
