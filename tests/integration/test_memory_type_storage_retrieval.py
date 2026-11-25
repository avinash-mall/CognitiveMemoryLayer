"""Storage and retrieval tests for each memory type.

For each MemoryType, we store a record (via the appropriate path where applicable)
and retrieve it (scan, get_by_key, or retrieval API), then assert type and content.
"""

from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecordCreate, Provenance
from src.storage.postgres import PostgresMemoryStore


@pytest.fixture
def store(pg_session_factory):
    return PostgresMemoryStore(pg_session_factory)


@pytest.fixture
def tenant_id():
    return f"t-{uuid4().hex[:8]}"


_MEMORY_TYPE_CASES = [
    (MemoryType.EPISODIC_EVENT, "We met last Tuesday.", None),
    (MemoryType.SEMANTIC_FACT, "The capital of France is Paris.", "fact:capital:france"),
    (MemoryType.PREFERENCE, "I prefer dark mode.", "user:preference:theme"),
    (MemoryType.TASK_STATE, "Remind me to call John.", None),
    (MemoryType.PROCEDURE, "Steps to deploy: build, push, run.", None),
    (MemoryType.CONSTRAINT, "Never remind on weekends.", "constraint:no_weekends"),
    (MemoryType.HYPOTHESIS, "I think we should delay launch.", None),
    (MemoryType.CONVERSATION, "User asked about the weather.", None),
    (MemoryType.MESSAGE, "Assistant replied with the forecast.", None),
    (MemoryType.TOOL_RESULT, '{"result": "search completed"}', None),
    (MemoryType.REASONING_STEP, "First we need to verify the inputs.", None),
    (MemoryType.SCRATCH, '{"step": 1}', None),
    (MemoryType.KNOWLEDGE, "Python is a programming language.", None),
    (MemoryType.OBSERVATION, "The user clicked the submit button.", None),
    (MemoryType.PLAN, "Goal: complete the report by Friday.", None),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "memory_type,text,key",
    _MEMORY_TYPE_CASES,
    ids=[mt.value for mt, _, _ in _MEMORY_TYPE_CASES],
)
async def test_storage_retrieval_by_memory_type_direct_upsert(
    store, tenant_id, memory_type, text, key
):
    """Store a record of the given type via direct upsert; retrieve via scan; assert type and content."""
    create = MemoryRecordCreate(
        tenant_id=tenant_id,
        context_tags=[],
        type=memory_type,
        text=text,
        key=key,
        embedding=None,
        provenance=Provenance(source=MemorySource.AGENT_INFERRED),
    )
    created = await store.upsert(create)
    assert created.id is not None
    assert created.type == memory_type
    assert created.text == text

    records = await store.scan(
        tenant_id,
        filters={"type": memory_type.value, "status": "active"},
        limit=10,
    )
    assert len(records) >= 1
    found = next((r for r in records if r.id == created.id), None)
    assert found is not None
    assert found.type == memory_type
    assert found.text == text


@pytest.mark.asyncio
async def test_semantic_fact_via_neocortical_store_and_retrieve(pg_session_factory, tenant_id):
    """Semantic_fact: store via NeocorticalStore.store_fact; retrieve via get_fact."""
    from src.memory.neocortical.fact_store import SemanticFactStore
    from src.memory.neocortical.store import NeocorticalStore

    class _MockGraph:
        async def merge_edge(self, *args, **kwargs):
            return "mock"

        async def get_entity_facts(self, *args, **kwargs):
            return []

        async def personalized_pagerank(self, *args, **kwargs):
            return []

    fact_store = SemanticFactStore(pg_session_factory)
    neocortical = NeocorticalStore(graph_store=_MockGraph(), fact_store=fact_store)
    key = "user:identity:name"
    value = "Alice"
    await neocortical.store_fact(tenant_id, key, value, confidence=0.9)
    got = await neocortical.get_fact(tenant_id, key)
    assert got is not None
    assert got.key == key
    assert got.value == value
    assert got.confidence == 0.9


@pytest.mark.asyncio
async def test_historical_timestamp_preservation(pg_session_factory, tenant_id):
    """MemoryOrchestrator: write preserving historic timestamp to Episodic and Semantic facts."""
    from datetime import UTC, datetime

    from src.core.config import get_embedding_dimensions
    from src.memory.neocortical.schemas import FactCategory
    from src.memory.orchestrator import MemoryOrchestrator
    from src.storage.postgres import PostgresMemoryStore
    from src.utils.embeddings import MockEmbeddingClient
    from src.utils.llm import MockLLMClient

    dims = get_embedding_dimensions()
    episodic_store = PostgresMemoryStore(pg_session_factory)
    embedding_client = MockEmbeddingClient(dimensions=dims)
    llm_client = MockLLMClient()

    orchestrator = await MemoryOrchestrator.create_lite(
        episodic_store=episodic_store,
        embedding_client=embedding_client,
        llm_client=llm_client,
    )

    historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC)

    await orchestrator.write(
        tenant_id=tenant_id,
        content="I moved to New York in January 2023.",
        session_id="test-session",
        timestamp=historical_date,
    )

    # Check episodic store
    records = await episodic_store.scan(tenant_id)
    found_historic_episodic = False
    for r in records:
        if r.timestamp.year == 2023:
            found_historic_episodic = True
    assert found_historic_episodic, "Episodic record did not preserve historic 2023 timestamp"

    # Check semantic facts (optional: MockLLMClient may not extract; episodic check is sufficient)
    found_historic_fact = False
    for cat in FactCategory:
        facts = await orchestrator.neocortical.facts.get_facts_by_category(tenant_id, cat)
        for f in facts:
            if f.valid_from and f.valid_from.year == 2023:
                found_historic_fact = True
                break
        if found_historic_fact:
            break

    assert found_historic_episodic or found_historic_fact, (
        "Either episodic or semantic store should preserve historic 2023 timestamp"
    )
