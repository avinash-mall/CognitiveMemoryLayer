"""Simple chatbot with persistent memory (py-cml + OpenAI).

Set AUTH__API_KEY, CML_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL in .env.
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
    base_url = (os.environ.get("CML_BASE_URL") or os.environ.get("MEMORY_API_URL") or "").strip()
    model = (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM__MODEL") or "").strip()
    if not base_url or not model:
        raise SystemExit("Set CML_BASE_URL and OPENAI_MODEL (or LLM__MODEL) in .env")
    memory = CognitiveMemoryLayer(
        api_key=os.environ.get("CML_API_KEY") or os.environ.get("AUTH__API_KEY"),
        base_url=base_url,
    )
    openai_client = OpenAI()

    session_id = "chat-demo-001"
    print("Chat with Memory (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # Get memory context
        turn = memory.turn(
            user_message=user_input,
            session_id=session_id,
        )

        # Build prompt with memory
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

        # Call OpenAI
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        assistant_msg = response.choices[0].message.content

        # Store the exchange
        memory.turn(
            user_message=user_input,
            assistant_response=assistant_msg,
            session_id=session_id,
        )

        print(f"Assistant: {assistant_msg}\n")
        print(f"  [memories retrieved: {turn.memories_retrieved}]")

    memory.close()


if __name__ == "__main__":
    chat_with_memory()
