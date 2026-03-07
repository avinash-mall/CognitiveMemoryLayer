"""Broader direct HTTP walkthrough for the CML API."""

from __future__ import annotations

import httpx
from _shared import (
    api_base_url,
    api_headers,
    explain_connection_failure,
    get_env,
    is_non_interactive,
    print_header,
)

EXAMPLE_META = {
    "name": "standalone_demo",
    "kind": "python",
    "summary": "Broader direct HTTP walkthrough for health, write, read, turn, sessions, stats, and forget.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": False,
    "requires_anthropic": False,
    "interactive": False,
    "timeout_sec": 120,
}

TIMEOUT = 60.0


def maybe_pause(message: str) -> None:
    if not is_non_interactive():
        input(message)


def main() -> int:
    print_header("CML Standalone Demo")
    base_url = api_base_url()
    headers = api_headers()
    session_id = "examples-standalone"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            health = client.get(f"{base_url}/health", headers=headers)
            health.raise_for_status()
            print(f"Health: {health.json().get('status')}")

            maybe_pause("Press Enter to write sample memories...")
            for payload in (
                {
                    "content": "User prefers dark mode in all applications.",
                    "session_id": session_id,
                    "memory_type": "preference",
                    "context_tags": ["preferences"],
                },
                {
                    "content": "User is allergic to penicillin.",
                    "session_id": session_id,
                    "memory_type": "constraint",
                    "context_tags": ["medical", "constraints"],
                },
                {
                    "content": "User recently started a new backend engineering role.",
                    "session_id": session_id,
                    "memory_type": "episodic_event",
                    "context_tags": ["career"],
                },
            ):
                response = client.post(f"{base_url}/memory/write", headers=headers, json=payload)
                response.raise_for_status()
                print(f"Wrote memory: {payload['content']}")

            maybe_pause("Press Enter to run retrieval...")
            read = client.post(
                f"{base_url}/memory/read",
                headers=headers,
                json={"query": "What preferences and constraints should I know about?", "format": "packet"},
            )
            read.raise_for_status()
            read_data = read.json()
            print(f"Read returned {read_data.get('total_count')} memories")
            for item in read_data.get("memories", [])[:5]:
                print(f"  - [{item.get('type')}] {item.get('text')}")

            llm_context = client.post(
                f"{base_url}/memory/read",
                headers=headers,
                json={"query": "Summarize the user", "format": "llm_context"},
            )
            llm_context.raise_for_status()
            print(f"\nLLM context snippet: {llm_context.json().get('llm_context', '')[:220]}")

            maybe_pause("Press Enter to run a /memory/turn request...")
            turn = client.post(
                f"{base_url}/memory/turn",
                headers=headers,
                json={
                    "user_message": "What do you know about my preferences?",
                    "session_id": session_id,
                    "max_context_tokens": 500,
                },
            )
            turn.raise_for_status()
            turn_data = turn.json()
            print(
                f"Turn: retrieved={turn_data.get('memories_retrieved')} stored={turn_data.get('memories_stored')}"
            )

            maybe_pause("Press Enter to inspect session context...")
            context = client.get(f"{base_url}/session/{session_id}/context", headers=headers)
            context.raise_for_status()
            context_data = context.json()
            print(f"Session messages: {len(context_data.get('messages', []))}")

            maybe_pause("Press Enter to create a new session...")
            created = client.post(
                f"{base_url}/session/create",
                headers=headers,
                json={"name": "examples-standalone", "ttl_hours": 24},
            )
            created.raise_for_status()
            print(f"Created session: {created.json().get('session_id')}")

            stats = client.get(f"{base_url}/memory/stats", headers=headers)
            stats.raise_for_status()
            print(f"Stats total memories: {stats.json().get('total_memories')}")

            forgotten = client.post(
                f"{base_url}/memory/forget",
                headers=headers,
                json={"query": "backend engineering role", "action": "archive"},
            )
            forgotten.raise_for_status()
            print(f"Archived memories: {forgotten.json().get('affected_count')}")

        if get_env("CML_ADMIN_API_KEY"):
            print("\nAdmin key detected. See examples/admin_dashboard.py for admin workflows.")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
