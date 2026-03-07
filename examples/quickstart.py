"""Minimal write/read/stream/context example for py-cml."""

from __future__ import annotations

from _shared import build_cml_config, explain_connection_failure, print_header

from cml import CognitiveMemoryLayer

EXAMPLE_META = {
    "name": "quickstart",
    "kind": "python",
    "summary": "Minimal sync write/read/stream/context example.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": False,
    "requires_anthropic": False,
    "interactive": False,
    "timeout_sec": 60,
}


def main() -> int:
    print_header("CML Quickstart")
    try:
        with CognitiveMemoryLayer(config=build_cml_config()) as memory:
            session_id = "examples-quickstart"
            for text in (
                "User prefers vegetarian food and lives in Paris.",
                "User works as a backend engineer at a startup.",
                "User has a design sync every Tuesday afternoon.",
            ):
                memory.write(text, session_id=session_id)

            result = memory.read("What does the user do for work?")
            print(f"Read returned {result.total_count} memories in {result.elapsed_ms:.1f}ms")
            for item in result.memories[:3]:
                print(f"  - [{item.type}] {item.text}")

            print("\nStreaming a broader query:")
            for item in memory.read_stream("Tell me about the user", max_results=3):
                print(f"  - {item.text}")

            context = memory.get_context("dietary preferences", max_results=3)
            print(f"\nContext snippet: {context[:180]}")

            stats = memory.stats()
            print(f"\nStats: {stats.total_memories} total memories")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
