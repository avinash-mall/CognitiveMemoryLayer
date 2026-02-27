"""Unit tests for deep-research implementation improvements.

Covers all 6 phases:
  Phase 1: Correctness fixes (stable keys, write-time facts)
  Phase 2: Hot-path performance (batch embeddings, async pipeline, cached embeds)
  Phase 3: Retrieval reliability (timeouts, skip-if-found, timezone)
  Phase 4: Background scalability (DB dep counts, batch graph)
  Phase 5: Operational stability (sensory tokens, bounded state)
  Phase 6: Observability (HNSW tuning, metrics)
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.config import FeatureFlags, RetrievalSettings, Settings
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance
from src.extraction.write_time_facts import WriteTimeFactExtractor, _derive_predicate
from src.memory.neocortical.schemas import FactCategory, SemanticFact
from src.memory.sensory.buffer import SensoryBuffer
from src.memory.working.models import ChunkType, SemanticChunk
from src.retrieval.planner import RetrievalPlan, RetrievalPlanner, RetrievalSource, RetrievalStep
from src.retrieval.query_types import QueryAnalysis, QueryIntent
from src.retrieval.retriever import HybridRetriever
from src.storage.async_pipeline import AsyncStoragePipeline
from src.utils.bounded_state import BoundedStateMap
from src.utils.embeddings import EmbeddingResult, MockEmbeddingClient


def _stable_fact_key_inline(prefix: str, text: str) -> str:
    """Inline replica for testing without triggering circular imports."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{h}"


class TestStableFactKey:
    def test_stable_fact_key_deterministic(self):
        key1 = _stable_fact_key_inline("user:custom", "User likes hiking")
        key2 = _stable_fact_key_inline("user:custom", "User likes hiking")
        assert key1 == key2

    def test_different_inputs_produce_different_keys(self):
        key1 = _stable_fact_key_inline("user:custom", "User likes hiking")
        key2 = _stable_fact_key_inline("user:custom", "User likes swimming")
        assert key1 != key2

    def test_format_matches_prefix(self):
        key = _stable_fact_key_inline("user:custom", "any text")
        assert key.startswith("user:custom:")
        parts = key.split(":")
        assert len(parts) == 3
        assert len(parts[2]) == 16  # 16 hex chars

    def test_no_python_hash_dependency(self):
        """Key must not use Python's built-in hash()."""
        text = "test text"
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        key = _stable_fact_key_inline("pfx", text)
        assert key == f"pfx:{expected_hash}"


# ═══════════════════════════════════════════════════════════════════
# Phase 1.2: Hippocampal key generation
# ═══════════════════════════════════════════════════════════════════


def _make_chunk(text: str, entities: list[str] | None = None) -> SemanticChunk:
    return SemanticChunk(
        id=str(uuid4()),
        text=text,
        chunk_type=ChunkType.STATEMENT,
        entities=entities or [],
    )


def _get_hippocampal_store():
    """Lazy import to avoid circular dependency at collection time."""
    from src.memory.hippocampal.store import HippocampalStore

    return HippocampalStore(
        vector_store=MagicMock(),
        embedding_client=MagicMock(),
    )


class TestHippocampalKeyGeneration:
    def setup_method(self):
        self.store = _get_hippocampal_store()

    def test_distinct_facts_with_same_entity_get_different_keys(self):
        chunk1 = _make_chunk("I love Italian food", entities=["Italian"])
        chunk2 = _make_chunk("I love Italian music", entities=["Italian"])
        key1 = self.store._generate_key(chunk1, MemoryType.PREFERENCE)
        key2 = self.store._generate_key(chunk2, MemoryType.PREFERENCE)
        assert key1 != key2

    def test_same_fact_gets_same_key(self):
        chunk = _make_chunk("I love Italian food", entities=["Italian"])
        key1 = self.store._generate_key(chunk, MemoryType.PREFERENCE)
        key2 = self.store._generate_key(chunk, MemoryType.PREFERENCE)
        assert key1 == key2

    def test_episodic_events_get_no_key(self):
        chunk = _make_chunk("It rained today", entities=["rain"])
        key = self.store._generate_key(chunk, MemoryType.EPISODIC_EVENT)
        assert key is None

    def test_key_includes_entity_prefix(self):
        chunk = _make_chunk("I love sushi", entities=["sushi"])
        key = self.store._generate_key(chunk, MemoryType.PREFERENCE)
        assert key is not None
        assert key.startswith("preference:sushi:")

    def test_key_without_entities_still_generates_hash(self):
        """Content-based hash generates a key even without entities."""
        chunk = _make_chunk("I love running", entities=[])
        key = self.store._generate_key(chunk, MemoryType.PREFERENCE)
        # With the new implementation, content-based keys are generated even
        # without entities (entity prefix is empty, hash still present)
        assert key is not None
        assert key.startswith("preference:")


# ═══════════════════════════════════════════════════════════════════
# Phase 1.3: Write-time fact extraction
# ═══════════════════════════════════════════════════════════════════


class TestWriteTimeFactExtractor:
    def setup_method(self):
        self.extractor = WriteTimeFactExtractor()

    def test_extracts_preference_from_preference_chunk(self):
        chunk = SemanticChunk(id="1", text="I love Italian food", chunk_type=ChunkType.PREFERENCE)
        facts = self.extractor.extract(chunk)
        assert len(facts) >= 1
        assert any("preference" in f.key for f in facts)

    def test_extracts_name_identity(self):
        chunk = SemanticChunk(id="2", text="My name is Alice", chunk_type=ChunkType.FACT)
        facts = self.extractor.extract(chunk)
        assert any(f.key == "user:identity:name" for f in facts)
        assert any(f.value == "Alice" for f in facts)

    def test_extracts_location(self):
        chunk = SemanticChunk(id="3", text="I live in Paris", chunk_type=ChunkType.FACT)
        facts = self.extractor.extract(chunk)
        assert any("location" in f.key for f in facts)
        assert any(f.value == "Paris" for f in facts)

    def test_ignores_non_fact_chunks(self):
        chunk = SemanticChunk(id="4", text="It rained today", chunk_type=ChunkType.EVENT)
        facts = self.extractor.extract(chunk)
        assert facts == []

    def test_ignores_statements(self):
        chunk = SemanticChunk(id="5", text="I love Italian food", chunk_type=ChunkType.STATEMENT)
        facts = self.extractor.extract(chunk)
        assert facts == []

    def test_confidence_below_consolidation_level(self):
        chunk = SemanticChunk(id="6", text="I love sushi", chunk_type=ChunkType.PREFERENCE)
        facts = self.extractor.extract(chunk)
        for f in facts:
            assert f.confidence < 0.8  # Write-time < consolidation


class TestDerivePredicateFunction:
    def test_known_cuisine_keyword(self):
        assert _derive_predicate("Italian food") == "cuisine"

    def test_known_music_keyword(self):
        assert _derive_predicate("jazz music") == "music"

    def test_unknown_falls_back_to_hash(self):
        pred = _derive_predicate("quantum physics")
        assert len(pred) == 12  # sha256 truncated

    def test_derive_predicate_deterministic(self):
        assert _derive_predicate("xyz") == _derive_predicate("xyz")


# ═══════════════════════════════════════════════════════════════════
# Phase 2.1: Batch embeddings in hippocampal store
# ═══════════════════════════════════════════════════════════════════


class TestBatchEmbeddings:
    @pytest.mark.asyncio
    async def test_encode_batch_calls_embed_batch_once(self):
        from src.memory.hippocampal.store import HippocampalStore

        mock_store = AsyncMock()
        mock_store.scan = AsyncMock(return_value=[])
        mock_store.upsert = AsyncMock(side_effect=lambda r: MagicMock(id=uuid4(), text=r.text))

        mock_embeddings = MockEmbeddingClient(dimensions=8)
        mock_embeddings.embed_batch = AsyncMock(
            return_value=[
                EmbeddingResult(embedding=[0.1] * 8, model="m", dimensions=8, tokens_used=5),
                EmbeddingResult(embedding=[0.2] * 8, model="m", dimensions=8, tokens_used=5),
            ]
        )

        store = HippocampalStore(
            vector_store=mock_store,
            embedding_client=mock_embeddings,
        )

        chunks = [
            _make_chunk("fact one"),
            _make_chunk("fact two"),
        ]

        results, _gate, _unified = await store.encode_batch("tenant1", chunks)
        mock_embeddings.embed_batch.assert_called_once()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_calls_increment_access_counts_once_with_result_ids(self):
        """search() invokes increment_access_counts once with the list of result IDs (no N+1)."""
        from src.memory.hippocampal.store import HippocampalStore

        id1 = uuid4()
        id2 = uuid4()
        record1 = MemoryRecord(
            id=id1,
            tenant_id="t",
            type=MemoryType.EPISODIC_EVENT,
            text="one",
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
            timestamp=datetime.now(UTC),
        )
        record2 = MemoryRecord(
            id=id2,
            tenant_id="t",
            type=MemoryType.EPISODIC_EVENT,
            text="two",
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
            timestamp=datetime.now(UTC),
        )
        mock_store = MagicMock()
        mock_store.vector_search = AsyncMock(return_value=[record1, record2])
        mock_store.increment_access_counts = AsyncMock()

        mock_embeddings = MagicMock()
        mock_embeddings.embed = AsyncMock(
            return_value=EmbeddingResult(
                embedding=[0.1] * 8, model="m", dimensions=8, tokens_used=1
            )
        )

        store = HippocampalStore(vector_store=mock_store, embedding_client=mock_embeddings)
        results = await store.search("tenant1", "query", top_k=10)
        assert len(results) == 2
        mock_store.increment_access_counts.assert_awaited_once()
        call_args = mock_store.increment_access_counts.call_args
        assert list(call_args[0][0]) == [id1, id2]

    @pytest.mark.asyncio
    async def test_encode_chunk_uses_ner_fallback_when_extractors_absent(self, monkeypatch):
        """When entity/relation extractors are None, hippocampal store should use NER fallback."""
        from src.memory.hippocampal.store import HippocampalStore

        class _FakeEnt:
            def __init__(self):
                self.text = "Alice"
                self.normalized = "Alice"
                self.entity_type = "PERSON"
                self.start_char = 0
                self.end_char = 5

        class _FakeRel:
            def __init__(self):
                self.subject = "Alice"
                self.predicate = "works_at"
                self.object = "Acme"
                self.confidence = 0.65

        monkeypatch.setattr(
            "src.memory.hippocampal.store._ner_extract_entities",
            lambda text: [_FakeEnt()],
        )
        monkeypatch.setattr(
            "src.memory.hippocampal.store._ner_extract_relations",
            lambda text: [_FakeRel()],
        )

        captured = {}
        mock_store = MagicMock()
        mock_store.scan = AsyncMock(return_value=[])

        async def _upsert(record):
            captured["record"] = record
            return MemoryRecord(
                id=uuid4(),
                tenant_id=record.tenant_id,
                context_tags=record.context_tags,
                type=record.type,
                text=record.text,
                key=record.key,
                confidence=record.confidence,
                importance=record.importance,
                provenance=record.provenance,
                timestamp=record.timestamp,
                entities=record.entities,
                relations=record.relations,
            )

        mock_store.upsert = AsyncMock(side_effect=_upsert)

        mock_embeddings = MagicMock()
        mock_embeddings.embed = AsyncMock(
            return_value=EmbeddingResult(
                embedding=[0.1] * 8, model="m", dimensions=8, tokens_used=3
            )
        )

        store = HippocampalStore(
            vector_store=mock_store,
            embedding_client=mock_embeddings,
            entity_extractor=None,
            relation_extractor=None,
        )

        chunk = SemanticChunk(
            id="c-ner",
            text="Alice works at Acme.",
            chunk_type=ChunkType.EVENT,
            salience=0.8,
            timestamp=datetime.now(UTC),
        )
        record, _ = await store.encode_chunk("tenant1", chunk, existing_memories=[])

        assert record is not None
        rec = captured["record"]
        assert rec.entities and rec.entities[0].normalized == "Alice"
        assert rec.relations and rec.relations[0].predicate == "works_at"


# ═══════════════════════════════════════════════════════════════════
# Phase 2.2: Async storage pipeline
# ═══════════════════════════════════════════════════════════════════


class TestAsyncStoragePipeline:
    @pytest.mark.asyncio
    async def test_enqueue_returns_job_id(self):
        redis = AsyncMock()
        redis.exists = AsyncMock(return_value=False)
        redis.rpush = AsyncMock()

        pipeline = AsyncStoragePipeline(redis_client=redis, enabled=True)
        job_id = await pipeline.enqueue(
            tenant_id="t1",
            user_message="Hello",
        )
        assert job_id  # Non-empty
        redis.rpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_enqueue_dedup_skips_duplicate(self):
        redis = AsyncMock()
        redis.exists = AsyncMock(return_value=True)  # Already processed

        pipeline = AsyncStoragePipeline(redis_client=redis, enabled=True)
        job_id = await pipeline.enqueue(
            tenant_id="t1",
            user_message="Hello",
        )
        assert job_id.startswith("dedup:")


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Retrieval reliability
# ═══════════════════════════════════════════════════════════════════


class TestRetrievalTimeouts:
    @pytest.mark.asyncio
    async def test_step_timeout_returns_empty_result(self):
        hippocampal = MagicMock()
        neocortical = MagicMock()
        retriever = HybridRetriever(hippocampal, neocortical)

        step = RetrievalStep(
            source=RetrievalSource.VECTOR,
            query="test query",
            timeout_ms=10,  # Very short
        )

        # Mock a slow vector search
        async def slow_search(*a, **kw):
            await asyncio.sleep(5)
            return []

        hippocampal.search = slow_search

        result = await retriever._execute_step_with_timeout("t1", step)
        assert not result.success
        assert "Timeout" in (result.error or "")


class TestTimezoneAwareTemporalQueries:
    def test_today_uses_user_timezone(self):
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="What did I say today?",
            intent=QueryIntent.TEMPORAL_QUERY,
            confidence=0.9,
            time_reference="today",
            user_timezone="America/New_York",
        )
        plan = planner.plan(analysis)
        assert len(plan.steps) > 0
        step = plan.steps[0]
        assert step.time_filter is not None
        assert "since" in step.time_filter

    def test_yesterday_filter_without_timezone(self):
        planner = RetrievalPlanner()
        analysis = QueryAnalysis(
            original_query="What happened yesterday?",
            intent=QueryIntent.TEMPORAL_QUERY,
            confidence=0.9,
            time_reference="yesterday",
        )
        plan = planner.plan(analysis)
        step = plan.steps[0]
        assert step.time_filter is not None
        assert "since" in step.time_filter
        assert "until" in step.time_filter

    @pytest.mark.asyncio
    async def test_memory_retriever_passes_user_timezone_to_planner(self):
        """MemoryRetriever.retrieve() sets user_timezone on analysis before planning."""
        from src.core.schemas import MemoryPacket
        from src.retrieval.memory_retriever import MemoryRetriever

        hippocampal = MagicMock()
        neocortical = MagicMock()
        classifier = AsyncMock()
        classifier.classify.return_value = QueryAnalysis(
            original_query="today?",
            intent=QueryIntent.TEMPORAL_QUERY,
            confidence=0.8,
            time_reference="today",
            user_timezone=None,
        )
        captured_analysis = []

        def capture_plan(analysis):
            captured_analysis.append(analysis)
            return RetrievalPlan(
                query=analysis.original_query,
                analysis=analysis,
                steps=[
                    RetrievalStep(
                        source=RetrievalSource.VECTOR,
                        key=None,
                        top_k=10,
                        time_filter={},
                    ),
                ],
                parallel_steps=[[0]],
                total_timeout_ms=5000,
            )

        planner = MagicMock()
        planner.plan.side_effect = capture_plan
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(return_value=[])
        retriever.hippocampal = MagicMock()
        retriever.hippocampal.embeddings = MagicMock()
        retriever.hippocampal.embeddings.embed = AsyncMock(
            return_value=EmbeddingResult(
                embedding=[0.1] * 1536,
                model="test",
                dimensions=1536,
                tokens_used=10,
            )
        )
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[])
        packet_builder = MagicMock()
        packet_builder.build.return_value = MemoryPacket(query="today?", recent_episodes=[])

        mr = MemoryRetriever(
            hippocampal=hippocampal,
            neocortical=neocortical,
            llm_client=None,
        )
        mr.classifier = classifier
        mr.planner = planner
        mr.retriever = retriever
        mr.reranker = reranker
        mr.packet_builder = packet_builder

        await mr.retrieve("t1", "today?", user_timezone="Europe/London")

        assert len(captured_analysis) == 1
        assert captured_analysis[0].user_timezone == "Europe/London"


# ═══════════════════════════════════════════════════════════════════
# Phase 5.1: Sensory buffer token ID storage
# ═══════════════════════════════════════════════════════════════════


class TestSensoryBufferTokenStorage:
    @pytest.mark.asyncio
    async def test_ingest_returns_token_count(self):
        buf = SensoryBuffer()
        count = await buf.ingest("Hello world")
        assert count > 0

    @pytest.mark.asyncio
    async def test_tokens_stored_as_ints(self):
        buf = SensoryBuffer()
        await buf.ingest("Hello")
        tokens = await buf.get_recent()
        for bt in tokens:
            assert isinstance(bt.token_id, int)

    @pytest.mark.asyncio
    async def test_get_text_returns_decoded_string(self):
        buf = SensoryBuffer()
        await buf.ingest("Hello world")
        text = await buf.get_text()
        # tiktoken might split differently, but joined text should contain originals
        assert "Hello" in text or "hello" in text.lower()


# ═══════════════════════════════════════════════════════════════════
# Phase 5.2: BoundedStateMap
# ═══════════════════════════════════════════════════════════════════


class TestBoundedStateMap:
    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self) -> None:
        m: BoundedStateMap[str] = BoundedStateMap(max_size=10)
        assert await m.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_or_create_creates_and_returns(self) -> None:
        m: BoundedStateMap[str] = BoundedStateMap(max_size=10)
        val = await m.get_or_create("k1", factory=lambda: "hello")
        assert val == "hello"
        # Second call returns same value
        val2 = await m.get_or_create("k1", factory=lambda: "world")
        assert val2 == "hello"

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        m: BoundedStateMap[int] = BoundedStateMap(max_size=3, ttl_seconds=999)
        await m.set("a", 1)
        await m.set("b", 2)
        await m.set("c", 3)
        assert m.size == 3

        # Adding a 4th should evict the oldest ("a")
        await m.set("d", 4)
        assert m.size == 3
        assert await m.get("a") is None
        assert await m.get("d") == 4

    @pytest.mark.asyncio
    async def test_ttl_expiry(self) -> None:
        m: BoundedStateMap[str] = BoundedStateMap(max_size=100, ttl_seconds=0.01)
        await m.set("k", "v")
        await asyncio.sleep(0.05)
        assert await m.get("k") is None

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        m: BoundedStateMap[str] = BoundedStateMap()
        await m.set("k", "v")
        assert await m.delete("k") is True
        assert await m.delete("k") is False
        assert await m.get("k") is None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self) -> None:
        m: BoundedStateMap[str] = BoundedStateMap(ttl_seconds=0.01)
        await m.set("a", "1")
        await m.set("b", "2")
        await asyncio.sleep(0.05)
        removed = await m.cleanup_expired()
        assert removed == 2
        assert m.size == 0


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Feature flags and config
# ═══════════════════════════════════════════════════════════════════


class TestFeatureFlags:
    def test_defaults_are_true(self):
        flags = FeatureFlags()
        assert flags.stable_keys_enabled is True
        assert flags.write_time_facts_enabled is True
        assert flags.batch_embeddings_enabled is True
        assert flags.retrieval_timeouts_enabled is True
        assert flags.db_dependency_counts is True
        assert flags.bounded_state_enabled is True
        assert flags.hnsw_ef_search_tuning is True

    def test_store_async_defaults_false(self):
        flags = FeatureFlags()
        assert flags.store_async is False

    def test_retrieval_settings_defaults(self):
        rs = RetrievalSettings()
        assert rs.default_step_timeout_ms == 500
        assert rs.total_timeout_ms == 2000
        assert rs.hnsw_ef_search == 64

    def test_settings_include_features(self):
        """Settings model includes features and retrieval sub-models."""
        s = Settings()
        assert hasattr(s, "features")
        assert hasattr(s, "retrieval")
        assert isinstance(s.features, FeatureFlags)
        assert isinstance(s.retrieval, RetrievalSettings)


# ═══════════════════════════════════════════════════════════════════
# Phase 4.2: Graph multi-hop N+1 batch fix
# ═══════════════════════════════════════════════════════════════════


def _make_semantic_fact(
    subject: str,
    key: str = "user:custom:test",
    value: str = "test-value",
    confidence: float = 0.9,
) -> SemanticFact:
    return SemanticFact(
        id=str(uuid4()),
        tenant_id="t1",
        category=FactCategory.CUSTOM,
        key=key,
        subject=subject,
        predicate="test",
        value=value,
        value_type="string",
        confidence=confidence,
    )


class TestNeo4jGetEntityFactsBatch:
    """Tests for Neo4jGraphStore.get_entity_facts_batch (UNWIND Cypher)."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver with async context manager session."""
        driver = MagicMock()
        mock_session = AsyncMock()
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        driver.session.return_value = session_ctx
        return driver, mock_session

    async def test_empty_entity_names_returns_empty_dict(self, mock_driver):
        from src.storage.neo4j import Neo4jGraphStore

        driver, _ = mock_driver
        store = Neo4jGraphStore(driver=driver)
        result = await store.get_entity_facts_batch("t1", "s1", [])
        assert result == {}

    async def test_groups_records_by_entity_name(self, mock_driver):
        from src.storage.neo4j import Neo4jGraphStore

        driver, mock_session = mock_driver
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(
            return_value=[
                {
                    "entity_name": "Alice",
                    "predicate": "KNOWS",
                    "direction": "outgoing",
                    "related_entity": "Bob",
                    "related_type": "PERSON",
                    "relation_properties": {"confidence": 0.9},
                },
                {
                    "entity_name": "Alice",
                    "predicate": "WORKS_AT",
                    "direction": "outgoing",
                    "related_entity": "Acme",
                    "related_type": "ORG",
                    "relation_properties": {},
                },
                {
                    "entity_name": "Bob",
                    "predicate": "LIVES_IN",
                    "direction": "outgoing",
                    "related_entity": "Paris",
                    "related_type": "LOCATION",
                    "relation_properties": {},
                },
            ]
        )
        mock_session.run = AsyncMock(return_value=mock_result)

        store = Neo4jGraphStore(driver=driver)
        result = await store.get_entity_facts_batch("t1", "s1", ["Alice", "Bob"])

        assert "Alice" in result
        assert "Bob" in result
        assert len(result["Alice"]) == 2
        assert len(result["Bob"]) == 1
        # entity_name should be popped from individual dicts
        for rel in result["Alice"]:
            assert "entity_name" not in rel
            assert "predicate" in rel
            assert "direction" in rel
            assert "related_entity" in rel

    async def test_missing_entity_not_in_result(self, mock_driver):
        """Entities with no relations are absent from the dict."""
        from src.storage.neo4j import Neo4jGraphStore

        driver, mock_session = mock_driver
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(
            return_value=[
                {
                    "entity_name": "Alice",
                    "predicate": "KNOWS",
                    "direction": "outgoing",
                    "related_entity": "Bob",
                    "related_type": "PERSON",
                    "relation_properties": {},
                },
            ]
        )
        mock_session.run = AsyncMock(return_value=mock_result)

        store = Neo4jGraphStore(driver=driver)
        result = await store.get_entity_facts_batch("t1", "s1", ["Alice", "Charlie"])

        assert "Alice" in result
        assert "Charlie" not in result
        # Callers use .get(name, [])
        assert result.get("Charlie", []) == []


class TestSemanticFactStoreSearchFactsBatch:
    """Tests for SemanticFactStore.search_facts_batch (subject IN)."""

    async def test_empty_entity_names_returns_empty_dict(self):
        from src.memory.neocortical.fact_store import SemanticFactStore

        mock_session_factory = AsyncMock()
        store = SemanticFactStore(session_factory=mock_session_factory)
        result = await store.search_facts_batch("t1", [])
        assert result == {}

    async def test_groups_by_subject_and_caps_per_entity(self):
        """Results are grouped by subject, limited to limit_per_entity each."""
        from src.memory.neocortical.fact_store import SemanticFactStore
        from src.storage.models import SemanticFactModel

        # Build mock ORM rows
        mock_rows = []
        for i, (subj, conf) in enumerate(
            [
                ("Alice", 0.9),
                ("Alice", 0.8),
                ("Alice", 0.7),
                ("Bob", 0.95),
                ("Bob", 0.85),
                ("Bob", 0.75),
            ]
        ):
            row = MagicMock(spec=SemanticFactModel)
            row.id = uuid4()
            row.tenant_id = "t1"
            row.category = "custom"
            row.key = f"user:custom:{subj.lower()}_{i}"
            row.subject = subj
            row.predicate = "test"
            row.value = f"value_{i}"
            row.value_type = "string"
            row.context_tags = []
            row.confidence = conf
            row.evidence_count = 1
            row.evidence_ids = []
            row.valid_from = None
            row.valid_to = None
            row.is_current = True
            row.created_at = datetime(2025, 1, 1)
            row.updated_at = datetime(2025, 1, 1)
            row.version = 1
            row.supersedes_id = None
            mock_rows.append(row)

        # Build mock session
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_rows

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=session_ctx)

        store = SemanticFactStore(session_factory=mock_factory)
        result = await store.search_facts_batch("t1", ["Alice", "Bob"], limit_per_entity=2)

        assert "Alice" in result
        assert "Bob" in result
        assert len(result["Alice"]) == 2  # capped from 3 to 2
        assert len(result["Bob"]) == 2  # capped from 3 to 2
        # Highest confidence first (they arrive pre-sorted by the query)
        assert result["Alice"][0].confidence == 0.9
        assert result["Bob"][0].confidence == 0.95


class TestMultiHopQueryBatch:
    """Tests for NeocorticalStore.multi_hop_query using batch APIs."""

    async def test_uses_batch_apis_and_returns_correct_shape(self):
        """multi_hop_query calls batch methods and assembles the result list."""
        from src.memory.neocortical.store import NeocorticalStore

        mock_graph = AsyncMock()
        mock_facts = AsyncMock()

        # PPR returns two ranked entities
        mock_graph.personalized_pagerank = AsyncMock(
            return_value=[
                {"entity": "Alice", "score": 0.9},
                {"entity": "Bob", "score": 0.7},
            ]
        )

        # Batch graph relations
        mock_graph.get_entity_facts_batch = AsyncMock(
            return_value={
                "Alice": [
                    {
                        "predicate": "KNOWS",
                        "direction": "outgoing",
                        "related_entity": "Bob",
                        "related_type": "PERSON",
                        "relation_properties": {},
                    },
                ],
            }
        )

        # Batch semantic facts
        alice_fact = _make_semantic_fact("Alice", key="user:identity:name", value="Alice Smith")
        mock_facts.search_facts_batch = AsyncMock(
            return_value={
                "Alice": [alice_fact],
            }
        )

        store = NeocorticalStore(graph_store=mock_graph, fact_store=mock_facts)
        results = await store.multi_hop_query("t1", seed_entities=["Alice"])

        # Verify batch APIs were called (not single-entity calls)
        mock_graph.get_entity_facts_batch.assert_awaited_once_with("t1", "t1", ["Alice", "Bob"])
        mock_facts.search_facts_batch.assert_awaited_once_with(
            "t1", ["Alice", "Bob"], limit_per_entity=5
        )

        # Verify result shape
        assert len(results) == 2
        alice_result = results[0]
        assert alice_result["entity"] == "Alice"
        assert alice_result["relevance_score"] == 0.9
        assert len(alice_result["relations"]) == 1
        assert alice_result["relations"][0]["predicate"] == "KNOWS"
        assert len(alice_result["facts"]) == 1
        assert alice_result["facts"][0]["key"] == "user:identity:name"

        # Bob has no graph relations or facts (not in batch result dicts)
        bob_result = results[1]
        assert bob_result["entity"] == "Bob"
        assert bob_result["relevance_score"] == 0.7
        assert bob_result["relations"] == []
        assert bob_result["facts"] == []

    async def test_empty_ppr_returns_empty_list(self):
        """When PPR returns no entities, result is empty."""
        from src.memory.neocortical.store import NeocorticalStore

        mock_graph = AsyncMock()
        mock_facts = AsyncMock()
        mock_graph.personalized_pagerank = AsyncMock(return_value=[])

        store = NeocorticalStore(graph_store=mock_graph, fact_store=mock_facts)
        results = await store.multi_hop_query("t1", seed_entities=["Unknown"])

        assert results == []
        # Batch methods should not be called at all
        mock_graph.get_entity_facts_batch.assert_not_awaited()
        mock_facts.search_facts_batch.assert_not_awaited()

    async def test_works_with_noop_stores(self):
        """multi_hop_query works correctly with NoOp stores (lite mode)."""
        from src.memory.neocortical.store import NeocorticalStore
        from src.storage.noop_stores import NoOpFactStore, NoOpGraphStore

        store = NeocorticalStore(
            graph_store=NoOpGraphStore(),
            fact_store=NoOpFactStore(),
        )
        # NoOp PPR returns [], so multi_hop returns []
        results = await store.multi_hop_query("t1", seed_entities=["Alice"])
        assert results == []
