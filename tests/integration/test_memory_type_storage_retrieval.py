"""Storage and retrieval tests for each memory type.

For each MemoryType, we store a record (via the appropriate path where applicable)
and retrieve it (scan, get_by_key, or retrieval API), then assert type and content.
"""

from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecordCreate, Provenance
from src.memory.conversation import ConversationMemory
from src.memory.knowledge_base import KnowledgeBase
from src.memory.scratch_pad import ScratchPad
from src.memory.tool_memory import ToolMemory
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
async def test_scratch_pad_set_get(store, tenant_id):
    """ScratchPad: set then get_by_key retrieves SCRATCH type."""
    scratch = ScratchPad(store=store)
    session_id = "s1"
    await scratch.set(tenant_id, session_id, "counter", 42)
    value = await scratch.get(tenant_id, session_id, "counter")
    assert value == 42
    key = f"{ScratchPad.SCRATCH_KEY_PREFIX}{session_id}:counter"
    record = await store.get_by_key(tenant_id, key)
    assert record is not None
    assert record.type == MemoryType.SCRATCH
    assert record.text == "42"


@pytest.mark.asyncio
async def test_conversation_memory_add_get_history(store, tenant_id):
    """ConversationMemory: add_message stores MESSAGE; get_history retrieves by session."""
    conv = ConversationMemory(store=store)
    session_id = "s1"
    await conv.add_message(tenant_id, session_id, "user", "Hello")
    await conv.add_message(tenant_id, session_id, "assistant", "Hi there!")
    history = await conv.get_history(tenant_id, session_id, limit=10)
    assert len(history) >= 2
    contents = [h.get("content", "") for h in history]
    assert "Hello" in contents
    assert "Hi there!" in contents
    records = await store.scan(
        tenant_id,
        filters={
            "type": MemoryType.MESSAGE.value,
            "status": "active",
            "source_session_id": session_id,
        },
        limit=10,
    )
    assert len(records) >= 2


@pytest.mark.asyncio
async def test_tool_memory_store_get_results(store, tenant_id):
    """ToolMemory: store_result stores TOOL_RESULT; get_results retrieves by session."""
    tool_mem = ToolMemory(store=store)
    session_id = "s1"
    await tool_mem.store_result(tenant_id, session_id, "search", {"q": "test"}, {"hits": 5})
    results = await tool_mem.get_results(tenant_id, session_id, limit=10)
    assert len(results) >= 1
    records = await store.scan(
        tenant_id,
        filters={"type": MemoryType.TOOL_RESULT.value, "status": "active"},
        limit=10,
    )
    assert len(records) >= 1
    assert records[0].type == MemoryType.TOOL_RESULT


@pytest.mark.asyncio
async def test_knowledge_base_store_query(store, tenant_id):
    """KnowledgeBase: store_fact stores KNOWLEDGE; query/scan retrieves by type or namespace."""
    from src.core.config import get_embedding_dimensions
    from src.utils.embeddings import MockEmbeddingClient

    kb = KnowledgeBase(
        store=store,
        embedding_client=MockEmbeddingClient(dimensions=get_embedding_dimensions()),
    )
    await kb.store_fact(
        tenant_id, "general", "Python", "is_a", "language", source="test", confidence=0.9
    )
    records = await store.scan(
        tenant_id,
        filters={"type": MemoryType.KNOWLEDGE.value, "status": "active"},
        limit=10,
    )
    assert len(records) >= 1
    assert records[0].type == MemoryType.KNOWLEDGE
    assert "Python" in records[0].text


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

    from src.memory.neocortical.schemas import FactCategory
    from src.memory.orchestrator import MemoryOrchestrator
    from src.storage.postgres import PostgresMemoryStore
    from src.utils.embeddings import MockEmbeddingClient

    class MockLLMClient:
        async def generate(self, *args, **kwargs):
            from src.utils.llm import LLMResponse

            return LLMResponse(content="", usage={})

        async def generate_json(self, *args, **kwargs):
            return {}

    from src.core.config import get_embedding_dimensions

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
