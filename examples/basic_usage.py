"""CRUD-oriented py-cml example with typed writes and filtered reads."""

from __future__ import annotations

from uuid import UUID

from _shared import build_cml_config, explain_connection_failure, print_header

from cml import CognitiveMemoryLayer
from cml.models import MemoryType

EXAMPLE_META = {
    "name": "basic_usage",
    "kind": "python",
    "summary": "Typed writes plus read, update, stats, and forget flows.",
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
    print_header("CML Basic Usage")
    session_id = "examples-basic-usage"
    try:
        with CognitiveMemoryLayer(config=build_cml_config()) as memory:
            writes = [
                (
                    "The user is Alice and works as a software engineer.",
                    MemoryType.SEMANTIC_FACT,
                    ["identity"],
                ),
                (
                    "The user prefers Python for backend work.",
                    MemoryType.PREFERENCE,
                    ["preferences"],
                ),
                (
                    "The user is allergic to shellfish and should not receive seafood recommendations.",
                    MemoryType.CONSTRAINT,
                    ["medical", "constraints"],
                ),
                (
                    "On 2026-02-15 the user said they want to learn Rust this year.",
                    MemoryType.EPISODIC_EVENT,
                    ["timeline"],
                ),
                (
                    "The user might be interested in machine learning.",
                    MemoryType.HYPOTHESIS,
                    ["inference"],
                ),
            ]

            for text, memory_type, tags in writes:
                response = memory.write(
                    text,
                    session_id=session_id,
                    context_tags=tags,
                    memory_type=memory_type,
                )
                print(f"Stored {memory_type.value}: success={response.success}")

            filtered = memory.read(
                "What programming languages does the user like?",
                memory_types=[MemoryType.PREFERENCE, MemoryType.CONSTRAINT],
            )
            print(f"\nFiltered read returned {filtered.total_count} memories")
            for item in filtered.memories:
                print(f"  - [{item.type}] {item.text}")

            llm_ready = memory.read("Summarize the user", response_format="llm_context")
            print(f"\nLLM context snippet: {(llm_ready.context or '')[:200]}")
            if llm_ready.constraints:
                print("Constraints:")
                for constraint in llm_ready.constraints[:3]:
                    print(f"  - {constraint.text}")

            hypothesis = memory.read("machine learning")
            if hypothesis.memories:
                memory_id = hypothesis.memories[0].id
                if not isinstance(memory_id, UUID):
                    memory_id = UUID(str(memory_id))
                updated = memory.update(memory_id=memory_id, feedback="correct")
                print(f"\nUpdated hypothesis version -> {updated.version}")

            stats = memory.stats()
            print(f"\nStats by type: {stats.by_type}")

            forgotten = memory.forget(query="learn Rust", action="archive")
            print(f"Archived {forgotten.affected_count} matching memories")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
