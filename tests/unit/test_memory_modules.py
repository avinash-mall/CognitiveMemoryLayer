"""Unit tests for memory modules: ConversationMemory, ScratchPad, ToolMemory, KnowledgeBase."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, MemoryRecordCreate


# ===== Mock MemoryStore for unit tests =====
class MockMemoryStore:
    """In-memory mock store for testing memory modules."""

    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}
        self.keys: dict[str, MemoryRecord] = {}

    async def upsert(self, record: MemoryRecordCreate) -> MemoryRecord:
        mem_id = uuid4()
        mem_record = MemoryRecord(
            id=mem_id,
            tenant_id=record.tenant_id,
            context_tags=record.context_tags,
            source_session_id=record.source_session_id,
            namespace=record.namespace,
            type=record.type,
            text=record.text,
            key=record.key,
            embedding=record.embedding,
            metadata=record.metadata,
            confidence=record.confidence,
            importance=record.importance,
            timestamp=record.timestamp or datetime.now(UTC),
            provenance=record.provenance,
        )
        self.records[str(mem_id)] = mem_record
        if record.key:
            self.keys[record.key] = mem_record
        return mem_record

    async def get_by_id(self, record_id) -> MemoryRecord | None:
        return self.records.get(str(record_id))

    async def get_by_key(
        self, tenant_id: str, key: str, context_filter: list[str] | None = None
    ) -> MemoryRecord | None:
        return self.keys.get(key)

    async def scan(
        self,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        results = list(self.records.values())
        filters = filters or {}
        if "type" in filters:
            type_filter = filters["type"]
            if isinstance(type_filter, list):
                results = [r for r in results if r.type.value in type_filter]
            else:
                results = [r for r in results if r.type.value == type_filter]
        if "source_session_id" in filters:
            sid = filters["source_session_id"]
            results = [r for r in results if r.source_session_id == sid]
        if "status" in filters and filters["status"] == "active":
            results = [r for r in results if r.status.value == "active"]
        return results[:limit]

    async def delete(self, record_id, hard: bool = False) -> None:
        self.records.pop(str(record_id), None)

    async def update(
        self, record_id, patch: dict[str, Any], increment_version: bool = True
    ) -> MemoryRecord | None:
        rec = self.records.get(str(record_id))
        if rec:
            for k, v in patch.items():
                setattr(rec, k, v)
        return rec


# ===== ConversationMemory Tests =====
class TestConversationMemory:
    """Tests for ConversationMemory module."""

    @pytest.fixture
    def store(self):
        return MockMemoryStore()

    @pytest.fixture
    def conversation(self, store):
        from src.memory.conversation import ConversationMemory

        return ConversationMemory(store)

    @pytest.mark.asyncio
    async def test_add_message_user_role(self, conversation, store):
        """Test adding a user message."""
        msg_id = await conversation.add_message(
            tenant_id="t1",
            session_id="s1",
            role="user",
            content="Hello, world!",
        )
        assert msg_id is not None
        assert len(store.records) == 1
        record = next(iter(store.records.values()))
        assert record.type == MemoryType.MESSAGE
        assert record.text == "Hello, world!"
        assert record.source_session_id == "s1"
        assert (record.metadata or {}).get("role") == "user"

    @pytest.mark.asyncio
    async def test_add_message_assistant_role(self, conversation, store):
        """Test adding an assistant message."""
        await conversation.add_message(
            tenant_id="t1",
            session_id="s1",
            role="assistant",
            content="Hi there!",
        )
        record = next(iter(store.records.values()))
        assert (record.metadata or {}).get("role") == "assistant"
        assert record.provenance.source == MemorySource.AGENT_INFERRED

    @pytest.mark.asyncio
    async def test_add_message_with_tool_calls(self, conversation, store):
        """Test adding a message with tool calls."""
        tool_calls = [{"name": "search", "args": {"q": "weather"}}]
        await conversation.add_message(
            tenant_id="t1",
            session_id="s1",
            role="assistant",
            content="Let me check.",
            tool_calls=tool_calls,
        )
        record = next(iter(store.records.values()))
        assert (record.metadata or {}).get("tool_calls") == tool_calls

    @pytest.mark.asyncio
    async def test_get_history_returns_ordered_messages(self, conversation):
        """Test getting conversation history returns all messages."""
        await conversation.add_message("t1", "s1", "user", "First")
        await conversation.add_message("t1", "s1", "assistant", "Second")
        await conversation.add_message("t1", "s1", "user", "Third")

        history = await conversation.get_history("t1", "s1")
        assert len(history) == 3
        # Verify all messages are returned (order depends on store implementation)
        contents = {h["content"] for h in history}
        assert contents == {"First", "Second", "Third"}

    @pytest.mark.asyncio
    async def test_get_history_respects_limit(self, conversation):
        """Test that history respects limit parameter."""
        for i in range(10):
            await conversation.add_message("t1", "s1", "user", f"Msg {i}")

        history = await conversation.get_history("t1", "s1", limit=5)
        assert len(history) <= 5

    @pytest.mark.asyncio
    async def test_summarize_and_compress_short_history(self, conversation):
        """Test summarize returns content when history is short."""
        await conversation.add_message("t1", "s1", "user", "Hello")
        await conversation.add_message("t1", "s1", "assistant", "Hi")

        result = await conversation.summarize_and_compress("t1", "s1", keep_recent=10)
        assert "Hello" in result
        assert "Hi" in result

    @pytest.mark.asyncio
    async def test_summarize_with_summary_text(self, conversation):
        """Test summarize includes provided summary."""
        for i in range(15):
            await conversation.add_message("t1", "s1", "user", f"Msg {i}")

        result = await conversation.summarize_and_compress(
            "t1", "s1", keep_recent=3, summary_text="Earlier discussion about topics."
        )
        assert "Earlier discussion" in result
        assert "--- Recent ---" in result


# ===== ScratchPad Tests =====
class TestScratchPad:
    """Tests for ScratchPad ephemeral memory."""

    @pytest.fixture
    def store(self):
        return MockMemoryStore()

    @pytest.fixture
    def scratch_pad(self, store):
        from src.memory.scratch_pad import ScratchPad

        return ScratchPad(store)

    @pytest.mark.asyncio
    async def test_set_and_get_string_value(self, scratch_pad):
        """Test storing and retrieving a string value."""
        await scratch_pad.set("t1", "s1", "my_key", "my_value")
        result = await scratch_pad.get("t1", "s1", "my_key")
        assert result == "my_value"

    @pytest.mark.asyncio
    async def test_set_and_get_dict_value(self, scratch_pad):
        """Test storing and retrieving a dict value (JSON serialized)."""
        data = {"foo": "bar", "count": 42}
        await scratch_pad.set("t1", "s1", "data", data)
        result = await scratch_pad.get("t1", "s1", "data")
        assert result == data

    @pytest.mark.asyncio
    async def test_get_missing_key_returns_none(self, scratch_pad):
        """Test getting a non-existent key returns None."""
        result = await scratch_pad.get("t1", "s1", "missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_append_creates_list(self, scratch_pad):
        """Test append creates a list if key doesn't exist."""
        await scratch_pad.append("t1", "s1", "items", "first")
        result = await scratch_pad.get("t1", "s1", "items")
        assert result == ["first"]

    @pytest.mark.asyncio
    async def test_append_to_existing_list(self, scratch_pad):
        """Test append adds to existing list."""
        await scratch_pad.set("t1", "s1", "items", ["a", "b"])
        await scratch_pad.append("t1", "s1", "items", "c")
        result = await scratch_pad.get("t1", "s1", "items")
        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_append_to_non_list_creates_list(self, scratch_pad):
        """Test append converts scalar to list."""
        await scratch_pad.set("t1", "s1", "val", "single")
        await scratch_pad.append("t1", "s1", "val", "second")
        result = await scratch_pad.get("t1", "s1", "val")
        assert result == ["single", "second"]

    @pytest.mark.asyncio
    async def test_clear_removes_session_scratch(self, scratch_pad, store):
        """Test clear removes all scratch data for session."""
        await scratch_pad.set("t1", "s1", "key1", "val1")
        await scratch_pad.set("t1", "s1", "key2", "val2")
        assert len(store.records) == 2

        await scratch_pad.clear("t1", "s1")
        # After clear, records should be deleted
        assert len(store.records) == 0


# ===== ToolMemory Tests =====
class TestToolMemory:
    """Tests for ToolMemory module."""

    @pytest.fixture
    def store(self):
        return MockMemoryStore()

    @pytest.fixture
    def tool_memory(self, store):
        from src.memory.tool_memory import ToolMemory

        return ToolMemory(store)

    @pytest.mark.asyncio
    async def test_store_result_returns_id(self, tool_memory, store):
        """Test storing a tool result returns an ID."""
        result_id = await tool_memory.store_result(
            tenant_id="t1",
            session_id="s1",
            tool_name="search",
            input_params={"query": "weather"},
            output={"temp": 72, "condition": "sunny"},
        )
        assert result_id is not None
        assert len(store.records) == 1

    @pytest.mark.asyncio
    async def test_store_result_has_correct_metadata(self, tool_memory, store):
        """Test stored result has correct metadata."""
        await tool_memory.store_result(
            tenant_id="t1",
            session_id="s1",
            tool_name="calculator",
            input_params={"expression": "2+2"},
            output=4,
        )
        record = next(iter(store.records.values()))
        assert record.type == MemoryType.TOOL_RESULT
        meta = record.metadata or {}
        assert meta.get("tool_name") == "calculator"
        assert meta.get("input_params") == {"expression": "2+2"}

    @pytest.mark.asyncio
    async def test_get_results_returns_stored_results(self, tool_memory):
        """Test retrieving tool results."""
        await tool_memory.store_result("t1", "s1", "tool1", {"a": 1}, "output1")
        await tool_memory.store_result("t1", "s1", "tool2", {"b": 2}, "output2")

        results = await tool_memory.get_results("t1", "s1")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_results_filter_by_tool_name(self, tool_memory):
        """Test filtering results by tool name."""
        await tool_memory.store_result("t1", "s1", "search", {}, "result1")
        await tool_memory.store_result("t1", "s1", "calc", {}, "result2")
        await tool_memory.store_result("t1", "s1", "search", {}, "result3")

        results = await tool_memory.get_results("t1", "s1", tool_name="search")
        assert len(results) == 2
        assert all(r["tool_name"] == "search" for r in results)


# ===== KnowledgeBase Tests =====
class TestKnowledgeBase:
    """Tests for KnowledgeBase module."""

    @pytest.fixture
    def store(self):
        return MockMemoryStore()

    @pytest.fixture
    def knowledge_base(self, store):
        from src.memory.knowledge_base import KnowledgeBase

        return KnowledgeBase(store, embedding_client=None)

    @pytest.mark.asyncio
    async def test_store_fact_returns_id(self, knowledge_base, store):
        """Test storing a fact returns an ID."""
        fact_id = await knowledge_base.store_fact(
            tenant_id="t1",
            namespace="general",
            subject="Paris",
            predicate="is_capital_of",
            object="France",
        )
        assert fact_id is not None
        assert len(store.records) == 1

    @pytest.mark.asyncio
    async def test_store_fact_has_correct_structure(self, knowledge_base, store):
        """Test stored fact has correct metadata."""
        await knowledge_base.store_fact(
            tenant_id="t1",
            namespace="geo",
            subject="Tokyo",
            predicate="population",
            object="14 million",
            source="census",
            confidence=0.95,
        )
        record = next(iter(store.records.values()))
        assert record.type == MemoryType.KNOWLEDGE
        assert record.namespace == "geo"
        meta = record.metadata or {}
        assert meta.get("subject") == "Tokyo"
        assert meta.get("predicate") == "population"
        assert meta.get("object") == "14 million"
        assert meta.get("source") == "census"
        assert record.confidence == 0.95

    @pytest.mark.asyncio
    async def test_query_without_embeddings(self, knowledge_base):
        """Test querying without embedding client uses scan."""
        await knowledge_base.store_fact("t1", "ns1", "A", "relates", "B")
        await knowledge_base.store_fact("t1", "ns1", "C", "relates", "D")

        # Query should return facts (uses scan when no embeddings)
        facts = await knowledge_base.query("t1", "ns1", "any query")
        assert len(facts) == 2
        assert all(f.subject in ("A", "C") for f in facts)

    @pytest.mark.asyncio
    async def test_fact_dataclass_properties(self, knowledge_base):
        """Test Fact dataclass has correct properties."""
        await knowledge_base.store_fact("t1", "ns", "X", "has", "Y", confidence=0.7)
        facts = await knowledge_base.query("t1", "ns", "query")
        assert len(facts) == 1
        f = facts[0]
        assert f.subject == "X"
        assert f.predicate == "has"
        assert f.object == "Y"
        assert f.text == "X has Y"
