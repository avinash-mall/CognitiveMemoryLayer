import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from src.memory.orchestrator import MemoryOrchestrator
from src.storage.connection import DatabaseManager


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_timestamp_ingestion():
    import os

    from src.core.config import get_settings

    settings = get_settings()
    provider = settings.embedding_internal.provider
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not configured for embedding provider")

    from src.utils.embeddings import get_embedding_client
    from src.utils.llm import get_internal_llm_client

    embedding_client = get_embedding_client()
    llm_client = get_internal_llm_client()

    db = await DatabaseManager.create()

    from src.storage.postgres import PostgresMemoryStore

    episodic_store = PostgresMemoryStore(db.pg_session_factory)

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
    from src.memory.neocortical.schemas import FactCategory

    profile = await orchestrator.neocortical.get_tenant_profile(tenant_id)
    print(f"Profile: {profile}")
    for cat in FactCategory:
        facts = await orchestrator.neocortical.facts.get_facts_by_category(tenant_id, cat)
        for f in facts:
            print(f"Fact: {f.key} = {f.value}, valid_from: {f.valid_from}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(test_timestamp_ingestion())
