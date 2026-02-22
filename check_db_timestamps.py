import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from datetime import UTC, datetime

from storage.connection import DatabaseManager
from storage.postgres.store import PostgresMemoryStore
from memory.neocortical.schemas import FactCategory

async def check_db():
    db = DatabaseManager()
    
    # We just ingested tenant: test-tenant-d0445b31-d922-44fe-ab8d-b01a9fea374d
    tenant_id = "test-tenant-d0445b31-d922-44fe-ab8d-b01a9fea374d"
    
    try:
        episodic_store = PostgresMemoryStore(db)
        
        print(f"Checking Episodic memories for {tenant_id}")
        records = await episodic_store.scan(tenant_id)
        for r in records:
            print(f"Episodic Record ID: {r.id}, Timestamp: {r.timestamp}")
            
        print("\nChecking Neocortical facts for {tenant_id}")
        # Need to query facts via SQL explicitly since we don't have orchestrator loaded
        async with db.postgres_session_factory() as session:
            from sqlalchemy import select
            from storage.models import SemanticFactModel
            q = select(SemanticFactModel).where(SemanticFactModel.tenant_id == tenant_id)
            result = await session.execute(q)
            for row in result.scalars().all():
                print(f"Fact Key: {row.key}, Valid From: {row.valid_from}, Created At: {row.created_at}")

    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(check_db())
