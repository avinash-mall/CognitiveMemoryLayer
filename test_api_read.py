import asyncio
import os
import requests

def get_tenant_memories(tenant_id):
    print(f"Reading memories for {tenant_id}")
    url = "http://localhost:8000/api/v1/memory/read"
    headers = {"X-Tenant-Id": tenant_id, "Authorization": "Bearer test-key"}
    data = {
        "query": "New York",
        "format": "categorized"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        data = response.json()
        print("Facts:")
        for f in data.get('facts', []):
            print(f"Fact: {f.get('text')}, Timestamp: {f.get('timestamp')}")
        print("Memories:")
        for m in data.get('memories', []):
            print(f"Memory: {m.get('text')}, Timestamp: {m.get('timestamp')}")
    else:
        print(f"Failed to read memories: {response.text}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        get_tenant_memories(sys.argv[1])
    else:
        print("Please provide a tenant ID as an argument.")
