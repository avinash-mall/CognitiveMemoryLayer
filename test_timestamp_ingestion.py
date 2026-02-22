import asyncio
import os
import uuid
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from datetime import UTC, datetime, timedelta

from core.enums import MemoryType
from memory.orchestrator import MemoryOrchestrator
from storage.connection import DatabaseManager


async def test_timestamp_ingestion():
    db = DatabaseManager()
    
    # Needs a real setup of stores like in tests
    from storage.postgres.store import PostgresMemoryStore
    from utils.embeddings import EmbeddingClient
    from retrieval.llm import LLMClient
    
    os.environ["STORAGE__EMBEDDING_DIM"] = "384"
    os.environ["LLM__PROVIDER"] = "openai"
    os.environ["LLM__MODEL"] = "gpt-4o-mini"
    os.environ["LLM__API_KEY"] = "sk-proj-test"
    
    episodic_store = PostgresMemoryStore(db)
    embedding_client = EmbeddingClient()
    llm_client = LLMClient()
    
    orchestrator = await MemoryOrchestrator.create_lite(
        episodic_store=episodic_store,
        embedding_client=embedding_client,
        llm_client=llm_client,
    )
    
    tenant_id = f"test-tenant-{uuid.uuid4()}"
    session_id = f"test-session-{uuid.uuid4()}"
    
    historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC)
    
    print(f"Ingesting memory with timestamp: {historical_date}")
    
    await orchestrator.write(
        tenant_id=tenant_id,
        content="I moved to New York in January 2023.",
        session_id=session_id,
        timestamp=historical_date,
    )
    
    print("Memory ingested. Checking database...")
    
    # Check episodic store
    records = await episodic_store.scan(tenant_id)
    for r in records:
        print(f"Episodic Record ID: {r.id}, Timestamp: {r.timestamp}")
        
    print("\nChecking neocortical store for facts...")
    from memory.neocortical.schemas import FactCategory
    profile = await orchestrator.neocortical.get_tenant_profile(tenant_id)
    print(f"Profile: {profile}")
    for cat in FactCategory:
        facts = await orchestrator.neocortical.facts.get_facts_by_category(tenant_id, cat)
        for f in facts:
            print(f"Fact: {f.key} = {f.value}, valid_from: {f.valid_from}")

    db.close()

if __name__ == "__main__":
    asyncio.run(test_timestamp_ingestion())
