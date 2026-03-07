"""Session-scoped workflow example with timestamps and timezone-aware reads."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from _shared import build_cml_config, explain_connection_failure, print_header

from cml import CognitiveMemoryLayer
from cml.models import MemoryType

EXAMPLE_META = {
    "name": "session_scope",
    "kind": "python",
    "summary": "SessionScope write/read/turn plus session context and timezone-aware retrieval.",
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
    print_header("CML Session Scope")
    historical_time = datetime.now(UTC) - timedelta(days=2)
    try:
        with (
            CognitiveMemoryLayer(config=build_cml_config(timeout=60.0)) as memory,
            memory.session(name="examples-session-scope") as session,
        ):
            session.write(
                "User prefers to receive project updates in the morning.",
                memory_type=MemoryType.PREFERENCE,
                context_tags=["schedule"],
                timestamp=historical_time,
            )
            session.write(
                "User is traveling to New York next week.",
                memory_type=MemoryType.EPISODIC_EVENT,
                context_tags=["travel"],
            )

            read = session.read(
                "How should I communicate project updates?",
                response_format="llm_context",
                user_timezone="America/New_York",
            )
            print(f"Session read count: {read.total_count}")
            print(f"Context: {(read.context or '')[:200]}")

            turn = session.turn(
                "What should you remember about my schedule?",
                assistant_response="You prefer morning project updates.",
                timestamp=datetime.now(UTC),
                user_timezone="America/New_York",
            )
            print(f"Turn retrieved={turn.memories_retrieved} stored={turn.memories_stored}")

            context = memory.get_session_context(session.session_id)
            print(f"Session context messages: {len(context.messages)}")
            print(f"Session id: {session.session_id}")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
