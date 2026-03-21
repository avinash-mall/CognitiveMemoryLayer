"""Live API integration coverage for eval ingestion helpers."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import requests

from evaluation.scripts.eval_locomo_plus import _cml_write


def _wait_for_memory(
    *,
    base_url: str,
    headers: dict[str, str],
    query: str,
    expected_text: str,
    timeout_seconds: float = 10.0,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict = {}
    while time.monotonic() < deadline:
        response = requests.post(
            f"{base_url}/api/v1/memory/read",
            headers=headers,
            json={"query": query},
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        memories = payload.get("memories") or []
        if any(expected_text in str(memory.get("text", "")) for memory in memories):
            return payload
        time.sleep(0.5)
    raise AssertionError(
        f"Timed out waiting for query '{query}' to retrieve '{expected_text}'. "
        f"Last payload: {last_payload}"
    )


def test_api_ingestion(live_api_test_context):
    historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC).isoformat()

    _cml_write(
        base_url=live_api_test_context["base_url"],
        api_key=live_api_test_context["api_key"],
        tenant_id=live_api_test_context["tenant_id"],
        content="I moved to New York in January 2023.",
        session_id=live_api_test_context["session_id"],
        metadata={"test": True, "source": "integration"},
        turn_id=f"turn-{live_api_test_context['session_id']}",
        timestamp=historical_date,
    )

    payload = _wait_for_memory(
        base_url=live_api_test_context["base_url"],
        headers=live_api_test_context["headers"],
        query="New York",
        expected_text="New York",
    )

    assert payload["total_count"] >= 1
    assert any(
        "New York" in str(memory.get("text", ""))
        and str(memory.get("timestamp", "")).startswith("2023-01-15")
        for memory in payload.get("memories", [])
    )
