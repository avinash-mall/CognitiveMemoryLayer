import os
import pytest
from datetime import UTC, datetime
from fastapi.testclient import TestClient

from app.main import app
from app.storage.postgres.store import PostgresMemoryStore
from app.memory.neocortical.schemas import FactCategory

@pytest.mark.asyncio
async def test_timestamp_preservation():
    client = TestClient(app)
    
    # 1. Ingest via API
    tenant_id = "test-tenant-timestamp"
    session_id = "test-session-timestamp"
    historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC).isoformat()
    
    response = client.post(
        "/api/v1/memory/write",
        headers={"X-Tenant-Id": tenant_id, "Authorization": "Bearer test-admin-key"},
        json={
            "content": "I moved to New York in January 2023.",
            "session_id": session_id,
            "timestamp": historical_date
        }
    )
    assert response.status_code == 200
    
    db = app.state.db
    episodic = PostgresMemoryStore(db)
    
    # Check episodic store
    records = await episodic.scan(tenant_id)
    print("\n--- Episodic Records ---")
    for r in records:
        print(f"ID: {r.id}, Timestamp: {r.timestamp}")
        assert r.timestamp.year == 2023
        
    print("\n--- Neocortical Facts ---")
    orchestrator = app.state.orchestrator
    for cat in FactCategory:
        facts = await orchestrator.neocortical.facts.get_facts_by_category(tenant_id, cat)
        for f in facts:
            print(f"Fact: {f.key} = {f.value}, valid_from: {f.valid_from}")
            assert f.valid_from.year == 2023
