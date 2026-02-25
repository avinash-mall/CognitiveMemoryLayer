import os
from datetime import UTC, datetime

import pytest
import requests


def _server_reachable() -> bool:
    url = os.environ.get("CML_BASE_URL", "")
    if not url:
        return False
    try:
        return requests.get(f"{url.rstrip('/')}/api/v1/health", timeout=2).status_code == 200
    except Exception:
        return False


@pytest.mark.e2e
@pytest.mark.skipif(not _server_reachable(), reason="CML API server not running")
def test_api():
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
    if read_resp.status_code == 200:
        data = read_resp.json()
        print("Facts:")
        for f in data.get("facts", []):
            print(f"- {f.get('text')} (timestamp: {f.get('timestamp')})")

        print("Memories:")
        for m in data.get("memories", []):
            print(f"- {m.get('text')} (timestamp: {m.get('timestamp')})")


if __name__ == "__main__":
    test_api()
