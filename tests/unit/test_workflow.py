import os
from datetime import UTC, datetime

import requests


def _server_reachable() -> bool:
    url = os.environ.get("CML_BASE_URL", "")
    if not url:
        return False
    try:
        return requests.get(f"{url.rstrip('/')}/api/v1/health", timeout=2).status_code == 200
    except Exception:
        return False


def test_api():
    assert _server_reachable(), "CML API server is not running"
    base_url = os.environ.get("CML_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("CML_API_KEY", "")
    tenant_id = "test-temporal-range-tenant"
    historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC).isoformat()
    headers = {"X-Tenant-Id": tenant_id, "X-API-Key": api_key}

    print(f"Writing memory with timestamp {historical_date}...")
    write_resp = requests.post(
        f"{base_url}/api/v1/memory/write",
        headers=headers,
        json={
            "content": "I moved to New York in January 2023.",
            "session_id": "test-session",
            "timestamp": historical_date,
        },
    )
    print(f"Write response: {write_resp.status_code} {write_resp.text}")

    import time

    time.sleep(2)

    print("Reading memories back...")
    read_resp = requests.post(
        f"{base_url}/api/v1/memory/read", headers=headers, json={"query": "New York"}
    )
    print(f"Read response: {read_resp.status_code}")
    assert read_resp.status_code == 200
    data = read_resp.json()
    assert data["total_count"] >= 1


if __name__ == "__main__":
    test_api()
