from datetime import UTC, datetime

import requests


def test_api():
    tenant_id = "test-temporal-range-tenant"
    historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC).isoformat()
    headers = {"X-Tenant-Id": tenant_id, "X-API-Key": "test-key"}

    # 1. Write the memory
    print(f"Writing memory with timestamp {historical_date}...")
    write_resp = requests.post(
        "http://localhost:8000/api/v1/memory/write",
        headers=headers,
        json={
            "content": "I moved to New York in January 2023.",
            "session_id": "test-session",
            "timestamp": historical_date,
        },
    )
    print(f"Write response: {write_resp.status_code} {write_resp.text}")

    # 2. Wait for jobs
    import time

    time.sleep(2)

    # 3. Read back
    print("Reading memories back...")
    read_resp = requests.post(
        "http://localhost:8000/api/v1/memory/read", headers=headers, json={"query": "New York"}
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
