"""Unit tests for SeamlessMemoryProvider and MemoryOrchestrator."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import (
    EntityMention,
    MemoryPacket,
    MemoryRecord,
    Provenance,
    Relation,
    RetrievedMemory,
)


def _make_memory_record(
    text: str = "Test memory",
    mem_type: MemoryType = MemoryType.SEMANTIC_FACT,
    confidence: float = 0.8,
) -> MemoryRecord:
    """Create a test memory record."""
    return MemoryRecord(
        id=uuid4(),
        tenant_id="t1",
        context_tags=[],
        type=mem_type,
        text=text,
        confidence=confidence,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=datetime.now(UTC),
    )


def _make_retrieved_memory(
    text: str = "Test memory",
    relevance: float = 0.8,
    mem_type: MemoryType = MemoryType.SEMANTIC_FACT,
) -> RetrievedMemory:
    """Create a test retrieved memory."""
    return RetrievedMemory(
        record=_make_memory_record(text, mem_type),
        relevance_score=relevance,
        retrieval_source="vector",
    )


class TestSeamlessMemoryProvider:
    """Tests for SeamlessMemoryProvider."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orch = MagicMock()

        # Mock read to return a memory packet (accept user_timezone and other kwargs)
        async def mock_read(
            tenant_id,
            query,
            max_results=10,
            context_filter=None,
            memory_types=None,
            since=None,
            until=None,
            user_timezone=None,
            **kwargs,
        ):
            return MemoryPacket(
                query=query,
                facts=[_make_retrieved_memory("User likes coffee", 0.85)],
                preferences=[
                    _make_retrieved_memory("Dark mode preference", 0.7, MemoryType.PREFERENCE)
                ],
            )

        orch.read = AsyncMock(side_effect=mock_read)

        # Mock write to return a result dict
        async def mock_write(**kwargs):
            return {"chunks_created": 1, "memory_id": str(uuid4())}

        orch.write = AsyncMock(side_effect=mock_write)

        # Mock reconsolidation
        recon_result = MagicMock()
        recon_result.memories_processed = 0
        recon_result.operations_applied = []
        orch.reconsolidation = MagicMock()
        orch.reconsolidation.process_turn = AsyncMock(return_value=recon_result)

        return orch

    @pytest.fixture
    def seamless(self, mock_orchestrator):
        """Create SeamlessMemoryProvider with mock."""
        from src.memory.seamless_provider import SeamlessMemoryProvider

        return SeamlessMemoryProvider(
            orchestrator=mock_orchestrator,
            max_context_tokens=500,
            auto_store=True,
            relevance_threshold=0.3,
        )

    @pytest.mark.asyncio
    async def test_process_turn_retrieves_context(self, seamless, mock_orchestrator):
        """Test that process_turn calls read to retrieve context."""
        result = await seamless.process_turn(
            tenant_id="t1",
            user_message="What do I like?",
        )

        mock_orchestrator.read.assert_called_once()
        assert result.memory_context != ""
        assert len(result.injected_memories) >= 1

    @pytest.mark.asyncio
    async def test_process_turn_stores_user_message(self, seamless, mock_orchestrator):
        """Test that auto_store saves the user message."""
        result = await seamless.process_turn(
            tenant_id="t1",
            user_message="I prefer tea now",
            session_id="s1",
        )

        # write should be called at least once for user message
        assert mock_orchestrator.write.call_count >= 1
        assert result.stored_count >= 1

    @pytest.mark.asyncio
    async def test_process_turn_stores_assistant_response(self, seamless, mock_orchestrator):
        """Test that assistant response is stored when provided."""
        result = await seamless.process_turn(
            tenant_id="t1",
            user_message="Hello",
            assistant_response="Hi there!",
            session_id="s1",
        )

        # write should be called twice (user msg + assistant response)
        assert mock_orchestrator.write.call_count >= 2
        assert result.stored_count >= 2

    @pytest.mark.asyncio
    async def test_process_turn_filters_by_relevance(self, seamless):
        """Test that memories below relevance threshold are filtered."""
        # Set high threshold
        seamless.relevance_threshold = 0.9

        result = await seamless.process_turn(
            tenant_id="t1",
            user_message="Query",
        )

        # With 0.9 threshold, the 0.85 and 0.7 memories should be filtered
        assert len(result.injected_memories) == 0

    @pytest.mark.asyncio
    async def test_process_turn_returns_turn_result(self, seamless):
        """Test that process_turn returns expected SeamlessTurnResult."""
        result = await seamless.process_turn(
            tenant_id="t1",
            user_message="Test message",
        )

        assert hasattr(result, "memory_context")
        assert hasattr(result, "injected_memories")
        assert hasattr(result, "stored_count")
        assert hasattr(result, "reconsolidation_applied")

    @pytest.mark.asyncio
    async def test_process_turn_no_auto_store(self, mock_orchestrator):
        """Test that auto_store=False skips storage."""
        from src.memory.seamless_provider import SeamlessMemoryProvider

        seamless = SeamlessMemoryProvider(
            orchestrator=mock_orchestrator,
            auto_store=False,
        )

        result = await seamless.process_turn(
            tenant_id="t1",
            user_message="Do not store this",
        )

        mock_orchestrator.write.assert_not_called()
        assert result.stored_count == 0

    @pytest.mark.asyncio
    async def test_format_for_injection_respects_max_tokens(self, seamless):
        """Test that context string respects token limit."""
        result = await seamless.process_turn(
            tenant_id="t1",
            user_message="Test",
        )

        # max_context_tokens=500, roughly 4 chars per token = 2000 chars
        assert len(result.memory_context) <= 2000


class TestMemoryOrchestrator:
    """Tests for MemoryOrchestrator (with mocked dependencies)."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mocked dependencies for orchestrator."""
        deps = {
            "short_term": MagicMock(),
            "hippocampal": MagicMock(),
            "neocortical": MagicMock(),
            "retriever": MagicMock(),
            "reconsolidation": MagicMock(),
            "consolidation": MagicMock(),
            "forgetting": MagicMock(),
            "scratch_pad": MagicMock(),
            "conversation": MagicMock(),
            "tool_memory": MagicMock(),
            "knowledge_base": MagicMock(),
        }

        # Setup async mocks
        deps["short_term"].ingest_turn = AsyncMock(
            return_value={
                "tokens_buffered": 10,
                "chunks_created": 1,
                "all_chunks": [],
                "chunks_for_encoding": [],
            }
        )
        deps["hippocampal"].encode_batch = AsyncMock(return_value=[])
        # Setup store mock for update operations and scan (used by get_session_context)
        deps["hippocampal"].store = MagicMock()
        deps["hippocampal"].store.get_by_id = AsyncMock(return_value=None)
        deps["hippocampal"].store.update = AsyncMock(return_value=None)
        deps["hippocampal"].store.scan = AsyncMock(return_value=[])
        deps["retriever"].retrieve = AsyncMock(
            return_value=MemoryPacket(
                query="test",
                facts=[],
            )
        )
        deps["scratch_pad"].get = AsyncMock(return_value=None)
        deps["conversation"].get_history = AsyncMock(return_value=[])
        deps["tool_memory"].get_results = AsyncMock(return_value=[])

        return deps

    @pytest.fixture
    def orchestrator(self, mock_dependencies):
        """Create orchestrator with mocked dependencies."""
        from src.memory.orchestrator import MemoryOrchestrator

        return MemoryOrchestrator(**mock_dependencies)

    @pytest.mark.asyncio
    async def test_write_ingests_turn(self, orchestrator, mock_dependencies):
        """Test that write calls short_term.ingest_turn."""
        result = await orchestrator.write(
            tenant_id="t1",
            content="I like pizza",
            session_id="s1",
        )

        mock_dependencies["short_term"].ingest_turn.assert_called_once()
        assert "chunks_created" in result

    @pytest.mark.asyncio
    async def test_write_stores_via_hippocampal(self, orchestrator, mock_dependencies):
        """Test that write calls hippocampal.encode_batch when chunks exist."""
        # Setup to return chunks for encoding
        mock_dependencies["short_term"].ingest_turn = AsyncMock(
            return_value={
                "tokens_buffered": 10,
                "chunks_created": 1,
                "all_chunks": [],
                "chunks_for_encoding": [MagicMock()],  # Non-empty to trigger encoding
            }
        )

        await orchestrator.write(
            tenant_id="t1",
            content="Important information",
        )

        mock_dependencies["hippocampal"].encode_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_uses_retriever(self, orchestrator, mock_dependencies):
        """Test that read calls retriever.retrieve."""
        result = await orchestrator.read(
            tenant_id="t1",
            query="What do I like?",
            max_results=5,
        )

        mock_dependencies["retriever"].retrieve.assert_called_once()
        assert isinstance(result, MemoryPacket)

    @pytest.mark.asyncio
    async def test_get_session_context_builds_context(self, orchestrator, mock_dependencies):
        """Test that get_session_context returns expected structure."""
        result = await orchestrator.get_session_context(
            tenant_id="t1",
            session_id="s1",
        )

        assert "messages" in result
        assert "tool_results" in result
        assert "context_string" in result

    @pytest.mark.asyncio
    async def test_update_handles_feedback(self, orchestrator, mock_dependencies):
        """Test update with feedback parameter."""
        mem_id = uuid4()

        # Setup the hippocampal.store mock for update
        record = _make_memory_record()
        record.tenant_id = "t1"  # Must match tenant_id in test
        updated_record = _make_memory_record()
        updated_record.version = 2

        mock_dependencies["hippocampal"].store.get_by_id = AsyncMock(return_value=record)
        mock_dependencies["hippocampal"].store.update = AsyncMock(return_value=updated_record)

        result = await orchestrator.update(
            tenant_id="t1",
            memory_id=mem_id,
            feedback="correct",
        )

        assert result is not None
        assert "version" in result

    # ------------------------------------------------------------------
    # Neo4j graph sync tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_write_calls_sync_to_graph(self, orchestrator, mock_dependencies):
        """After a successful write, _sync_to_graph pushes entities + relations to Neo4j."""
        entity = EntityMention(text="Paris", normalized="Paris", entity_type="LOCATION")
        relation = Relation(subject="user", predicate="lives_in", object="Paris", confidence=0.9)
        stored_record = _make_memory_record(text="I live in Paris")
        stored_record.entities = [entity]
        stored_record.relations = [relation]

        # STM returns chunks so encode_batch is invoked
        mock_dependencies["short_term"].ingest_turn = AsyncMock(
            return_value={
                "tokens_buffered": 10,
                "chunks_created": 1,
                "all_chunks": [],
                "chunks_for_encoding": [MagicMock()],
            }
        )
        mock_dependencies["hippocampal"].encode_batch = AsyncMock(return_value=[stored_record])

        # Setup neocortical graph mock
        mock_dependencies["neocortical"].graph = MagicMock()
        mock_dependencies["neocortical"].graph.merge_node = AsyncMock(return_value="node-1")
        mock_dependencies["neocortical"].store_relations_batch = AsyncMock(return_value=["edge-1"])

        result = await orchestrator.write(tenant_id="t1", content="I live in Paris")

        assert result["chunks_created"] == 1

        # Entity node should have been merged
        mock_dependencies["neocortical"].graph.merge_node.assert_called_once_with(
            tenant_id="t1",
            scope_id="t1",
            entity="Paris",
            entity_type="LOCATION",
        )

        # Relation edge should have been merged
        mock_dependencies["neocortical"].store_relations_batch.assert_called_once_with(
            tenant_id="t1",
            relations=[relation],
            evidence_ids=[str(stored_record.id)],
        )

    @pytest.mark.asyncio
    async def test_write_graph_sync_failure_does_not_break_write(
        self, orchestrator, mock_dependencies
    ):
        """If Neo4j sync fails, the write still succeeds (Postgres already committed)."""
        entity = EntityMention(text="Berlin", normalized="Berlin", entity_type="LOCATION")
        stored_record = _make_memory_record(text="I visited Berlin")
        stored_record.entities = [entity]
        stored_record.relations = []

        mock_dependencies["short_term"].ingest_turn = AsyncMock(
            return_value={
                "tokens_buffered": 5,
                "chunks_created": 1,
                "all_chunks": [],
                "chunks_for_encoding": [MagicMock()],
            }
        )
        mock_dependencies["hippocampal"].encode_batch = AsyncMock(return_value=[stored_record])

        # Make Neo4j merge_node explode
        mock_dependencies["neocortical"].graph = MagicMock()
        mock_dependencies["neocortical"].graph.merge_node = AsyncMock(
            side_effect=RuntimeError("Neo4j connection refused")
        )

        # Write should still succeed despite graph failure
        result = await orchestrator.write(tenant_id="t1", content="I visited Berlin")

        assert result["chunks_created"] == 1
        assert result["memory_id"] == stored_record.id

    @pytest.mark.asyncio
    async def test_write_relation_sync_failure_does_not_break_write(
        self, orchestrator, mock_dependencies
    ):
        """If relation sync to Neo4j fails, the write still returns successfully."""
        relation = Relation(subject="user", predicate="works_at", object="ACME", confidence=0.8)
        stored_record = _make_memory_record(text="I work at ACME")
        stored_record.entities = []
        stored_record.relations = [relation]

        mock_dependencies["short_term"].ingest_turn = AsyncMock(
            return_value={
                "tokens_buffered": 5,
                "chunks_created": 1,
                "all_chunks": [],
                "chunks_for_encoding": [MagicMock()],
            }
        )
        mock_dependencies["hippocampal"].encode_batch = AsyncMock(return_value=[stored_record])

        # Make store_relations_batch explode
        mock_dependencies["neocortical"].graph = MagicMock()
        mock_dependencies["neocortical"].graph.merge_node = AsyncMock(return_value="ok")
        mock_dependencies["neocortical"].store_relations_batch = AsyncMock(
            side_effect=RuntimeError("Neo4j timeout")
        )

        result = await orchestrator.write(tenant_id="t1", content="I work at ACME")

        assert result["chunks_created"] == 1
        assert result["memory_id"] == stored_record.id

    @pytest.mark.asyncio
    async def test_write_no_entities_or_relations_skips_graph_sync(
        self, orchestrator, mock_dependencies
    ):
        """When stored records have no entities or relations, graph sync is a no-op."""
        stored_record = _make_memory_record(text="Hello world")
        stored_record.entities = []
        stored_record.relations = []

        mock_dependencies["short_term"].ingest_turn = AsyncMock(
            return_value={
                "tokens_buffered": 5,
                "chunks_created": 1,
                "all_chunks": [],
                "chunks_for_encoding": [MagicMock()],
            }
        )
        mock_dependencies["hippocampal"].encode_batch = AsyncMock(return_value=[stored_record])

        mock_dependencies["neocortical"].graph = MagicMock()
        mock_dependencies["neocortical"].graph.merge_node = AsyncMock()
        mock_dependencies["neocortical"].store_relations_batch = AsyncMock()

        result = await orchestrator.write(tenant_id="t1", content="Hello world")

        assert result["chunks_created"] == 1
        # Neither merge_node nor store_relations_batch should be called
        mock_dependencies["neocortical"].graph.merge_node.assert_not_called()
        mock_dependencies["neocortical"].store_relations_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_eval_mode_also_syncs_to_graph(self, orchestrator, mock_dependencies):
        """In eval_mode, graph sync still runs before the eval result is returned."""
        entity = EntityMention(text="Tokyo", normalized="Tokyo", entity_type="LOCATION")
        stored_record = _make_memory_record(text="I traveled to Tokyo")
        stored_record.entities = [entity]
        stored_record.relations = []

        gate_result = {"decision": "store", "reason": "significant"}

        mock_dependencies["short_term"].ingest_turn = AsyncMock(
            return_value={
                "tokens_buffered": 5,
                "chunks_created": 1,
                "all_chunks": [],
                "chunks_for_encoding": [MagicMock()],
            }
        )
        mock_dependencies["hippocampal"].encode_batch = AsyncMock(
            return_value=([stored_record], [gate_result])
        )

        mock_dependencies["neocortical"].graph = MagicMock()
        mock_dependencies["neocortical"].graph.merge_node = AsyncMock(return_value="node-1")
        mock_dependencies["neocortical"].store_relations_batch = AsyncMock()

        result = await orchestrator.write(
            tenant_id="t1", content="I traveled to Tokyo", eval_mode=True
        )

        assert result["eval_outcome"] == "stored"
        mock_dependencies["neocortical"].graph.merge_node.assert_called_once_with(
            tenant_id="t1",
            scope_id="t1",
            entity="Tokyo",
            entity_type="LOCATION",
        )


class TestOrchestratorFactory:
    """Tests for MemoryOrchestrator.create() factory wiring."""

    @pytest.mark.asyncio
    async def test_create_wires_entity_and_relation_extractors(self):
        """orchestrator.create() should inject EntityExtractor and RelationExtractor
        into HippocampalStore so entities and relations are extracted at write time."""
        from src.extraction.entity_extractor import EntityExtractor
        from src.extraction.relation_extractor import RelationExtractor
        from src.memory.orchestrator import MemoryOrchestrator

        mock_db = MagicMock()
        mock_db.pg_session = MagicMock()
        mock_db.neo4j_driver = MagicMock()
        mock_db.redis = None

        with (
            patch("src.memory.orchestrator.get_llm_client") as mock_llm,
            patch("src.memory.orchestrator.get_embedding_client") as mock_emb,
            patch("src.memory.orchestrator.Neo4jGraphStore"),
            patch("src.memory.orchestrator.PostgresMemoryStore"),
            patch("src.memory.orchestrator.SemanticFactStore"),
        ):
            mock_llm.return_value = MagicMock()
            mock_emb_instance = MagicMock()
            mock_emb_instance.dimensions = 1024
            mock_emb.return_value = mock_emb_instance

            orch = await MemoryOrchestrator.create(mock_db)

            # HippocampalStore should have real extractors, not None
            assert orch.hippocampal.entity_extractor is not None
            assert orch.hippocampal.relation_extractor is not None
            assert isinstance(orch.hippocampal.entity_extractor, EntityExtractor)
            assert isinstance(orch.hippocampal.relation_extractor, RelationExtractor)
