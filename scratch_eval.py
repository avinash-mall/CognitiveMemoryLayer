import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

headers = {"X-API-Key": "test-key", "X-Tenant-ID": "lp-0"}
payload = {"query": "Tell me about my constraints and plans.", "format": "llm_context", "max_results": int(os.environ.get("MAX_RESULTS", 25))}

start = time.time()
try:
    resp = requests.post("http://localhost:8000/api/v1/memory/read", json=payload, headers=headers, timeout=10)
    duration = time.time() - start
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Context length: {len(data.get('llm_context', ''))}")
    else:
        print(resp.text)
except Exception as e:
    duration = time.time() - start
    print(f"Error: {e}")
print(f"Time: {duration:.3f}s")
