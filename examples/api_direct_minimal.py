"""Minimal direct API example — no py-cml, httpx only.

Set AUTH__API_KEY and CML_BASE_URL (default http://localhost:8000) in .env.
Demonstrates: health, write, read (packet + llm_context), turn, stats.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import httpx

BASE_URL = (
    os.environ.get("CML_BASE_URL") or os.environ.get("MEMORY_API_URL") or ""
).strip() or "http://localhost:8000"
API_URL = f"{BASE_URL.rstrip('/')}/api/v1"
API_KEY = os.environ.get("AUTH__API_KEY") or os.environ.get("CML_API_KEY") or ""
HEADERS = {"Content-Type": "application/json", "X-API-Key": API_KEY}
TIMEOUT = 30.0


def main():
    if not API_KEY:
        print("Error: Set AUTH__API_KEY or CML_API_KEY in .env")
        return 1

    with httpx.Client(timeout=TIMEOUT) as client:
        # Health
        r = client.get(f"{API_URL}/health", headers=HEADERS)
        if r.status_code != 200:
            print(
                f"API not ready: {r.status_code}. Start with: docker compose -f docker/docker-compose.yml up api"
            )
            return 1
        print("Health OK:", r.json().get("status"))

        # Write
        r = client.post(
            f"{API_URL}/memory/write",
            headers=HEADERS,
            json={
                "content": "User prefers Python and works as a backend engineer.",
                "session_id": "api-direct-minimal",
                "context_tags": ["preference", "career"],
            },
        )
        assert r.status_code == 200
        print("Write OK:", r.json().get("memory_id"))

        # Read (packet)
        r = client.post(
            f"{API_URL}/memory/read",
            headers=HEADERS,
            json={"query": "user job preferences", "max_results": 5, "format": "packet"},
        )
        assert r.status_code == 200
        data = r.json()
        print(f"Read OK: {data.get('total_count')} memories, {data.get('elapsed_ms', 0):.1f}ms")
        for m in data.get("memories", [])[:3]:
            print(f"  - [{m.get('type')}] {m.get('text', '')[:50]}...")

        # Read (llm_context)
        r = client.post(
            f"{API_URL}/memory/read",
            headers=HEADERS,
            json={"query": "user preferences", "format": "llm_context"},
        )
        assert r.status_code == 200
        ctx = r.json().get("llm_context", "")
        print(f"LLM context ({len(ctx)} chars):", ctx[:150] + "..." if len(ctx) > 150 else ctx)

        # Turn
        r = client.post(
            f"{API_URL}/memory/turn",
            headers=HEADERS,
            json={
                "user_message": "What programming language do I like?",
                "session_id": "api-direct-minimal",
                "max_context_tokens": 500,
            },
        )
        assert r.status_code == 200
        t = r.json()
        print(
            f"Turn OK: retrieved {t.get('memories_retrieved')}, stored {t.get('memories_stored')}"
        )

        # Stats
        r = client.get(f"{API_URL}/memory/stats", headers=HEADERS)
        assert r.status_code == 200
        s = r.json()
        print(f"Stats: {s.get('total_memories')} total, {s.get('active_memories')} active")

    print("Done.")
    return 0


if __name__ == "__main__":
    exit(main())
