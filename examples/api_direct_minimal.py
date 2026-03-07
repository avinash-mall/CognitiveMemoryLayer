"""Minimal direct HTTP example that talks to the CML API with httpx."""

from __future__ import annotations

import httpx
from _shared import api_base_url, api_headers, explain_connection_failure, print_header

EXAMPLE_META = {
    "name": "api_direct_minimal",
    "kind": "python",
    "summary": "Minimal direct HTTP example using health, write, read, turn, and stats.",
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
    print_header("CML Direct API Minimal")
    base_url = api_base_url()
    headers = api_headers()
    try:
        with httpx.Client(timeout=30.0) as client:
            health = client.get(f"{base_url}/health", headers=headers)
            health.raise_for_status()
            print(f"Health: {health.json().get('status')}")

            write = client.post(
                f"{base_url}/memory/write",
                headers=headers,
                json={
                    "content": "User prefers Python and works as a backend engineer.",
                    "session_id": "examples-api-direct",
                    "context_tags": ["preferences", "career"],
                },
            )
            write.raise_for_status()
            print(f"Write memory_id: {write.json().get('memory_id')}")

            read = client.post(
                f"{base_url}/memory/read",
                headers=headers,
                json={"query": "user job preferences", "max_results": 3, "format": "packet"},
            )
            read.raise_for_status()
            read_data = read.json()
            print(f"Read count: {read_data.get('total_count')}")
            for item in read_data.get("memories", [])[:3]:
                print(f"  - [{item.get('type')}] {item.get('text')}")

            turn = client.post(
                f"{base_url}/memory/turn",
                headers=headers,
                json={
                    "user_message": "What programming language do I like?",
                    "session_id": "examples-api-direct",
                    "max_context_tokens": 400,
                },
            )
            turn.raise_for_status()
            turn_data = turn.json()
            print(
                f"Turn: retrieved={turn_data.get('memories_retrieved')} stored={turn_data.get('memories_stored')}"
            )

            stats = client.get(f"{base_url}/memory/stats", headers=headers)
            stats.raise_for_status()
            stats_data = stats.json()
            print(
                f"Stats: total={stats_data.get('total_memories')} active={stats_data.get('active_memories')}"
            )
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
