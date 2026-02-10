"""py-cml Quickstart — store and retrieve memories in 30 seconds."""

import os

from cml import CognitiveMemoryLayer


def main():
    # Prefer CML_API_KEY env var; set it or pass api_key= explicitly
    memory = CognitiveMemoryLayer(
        api_key=os.environ.get("CML_API_KEY", "your-api-key"),
        base_url=os.environ.get("CML_BASE_URL", "http://localhost:8000"),
    )
    memory.write("User prefers vegetarian food and lives in Paris.")
    memory.write("User works at a tech startup as a backend engineer.")
    memory.write("User has a meeting with the design team every Tuesday.")
    result = memory.read("What does the user do for work?")
    print(f"Found {result.total_count} relevant memories:")
    for item in result.memories:
        print(f"  [{item.type}] {item.text}")
        print(f"    Relevance: {item.relevance:.2f}, Confidence: {item.confidence:.2f}")
    context = memory.get_context("dietary restrictions")
    print(f"\nLLM Context:\n{context}")
    stats = memory.stats()
    print(f"\nMemory Stats: {stats.total_memories} memories stored")
    memory.close()


if __name__ == "__main__":
    main()
