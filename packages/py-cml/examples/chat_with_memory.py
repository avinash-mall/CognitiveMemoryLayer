"""Build a chatbot with persistent memory using py-cml and OpenAI."""

from openai import OpenAI

from cml import CognitiveMemoryLayer


def chat_with_memory():
    # Initialize clients
    memory = CognitiveMemoryLayer(api_key="cml-key", base_url="http://localhost:8000")
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
            model="gpt-4o-mini",
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
