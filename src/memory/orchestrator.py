"""Memory orchestrator: coordinates all memory operations."""

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ..consolidation.worker import ConsolidationWorker
from ..core.enums import MemoryStatus, MemoryType
from ..core.exceptions import MemoryAccessDenied, MemoryNotFoundError
from ..core.schemas import MemoryPacket
from ..extraction.fact_extractor import LLMFactExtractor
from ..forgetting.worker import ForgettingWorker
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from ..memory.short_term import ShortTermMemory, ShortTermMemoryConfig
from ..reconsolidation.service import ReconsolidationService
from ..storage.base import MemoryStoreBase
from ..storage.noop_stores import NoOpFactStore, NoOpGraphStore
from ._write_pipeline import WritePathConfig, WritePipelineMixin

if TYPE_CHECKING:
    from ..memory.neocortical.fact_store import SemanticFactStore
    from ..retrieval.memory_retriever import MemoryRetriever
    from ..storage.connection import DatabaseManager
    from ..storage.neo4j import Neo4jGraphStore
    from ..storage.postgres import PostgresMemoryStore
else:
    try:
        from ..memory.neocortical.fact_store import SemanticFactStore
        from ..storage.connection import DatabaseManager
        from ..storage.neo4j import Neo4jGraphStore
        from ..storage.postgres import PostgresMemoryStore
    except ImportError:
        SemanticFactStore = None
        DatabaseManager = Any
        Neo4jGraphStore = None
        PostgresMemoryStore = None
from ..utils.embeddings import CachedEmbeddings, EmbeddingClient, get_embedding_client
from ..utils.llm import LLMClient, get_internal_llm_client
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

_SESSION_CONTEXT_LIMIT = 50  # Max memories to return in session context


class MemoryOrchestrator(WritePipelineMixin):
    """
    Main orchestrator for all memory operations.
    Coordinates short-term, hippocampal, neocortical, retrieval, reconsolidation,
    consolidation, and forgetting.
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        retriever: "MemoryRetriever",
        reconsolidation: ReconsolidationService,
        consolidation: ConsolidationWorker,
        forgetting: ForgettingWorker,
        dmn_store: Any | None = None,
        procedural_engine: Any | None = None,
        windowed_consolidator: Any | None = None,
        dopamine_rpe: Any | None = None,
        preplay_simulator: Any | None = None,
    ):
        self.short_term = short_term
        self.hippocampal = hippocampal
        self.neocortical = neocortical
        self.retriever = retriever
        self.reconsolidation = reconsolidation
        self.consolidation = consolidation
        self.forgetting = forgetting
        self.dmn_store = dmn_store
        self.procedural_engine = procedural_engine
        self.windowed_consolidator = windowed_consolidator
        self.dopamine_rpe = dopamine_rpe
        self.preplay_simulator = preplay_simulator

    @classmethod
    async def create(cls, db_manager: DatabaseManager) -> "MemoryOrchestrator":
        """Factory method to create orchestrator with all dependencies."""
        from ..core.config import get_settings

        settings = get_settings()
        internal_llm = None
        if settings.features.use_llm_enabled:
            import os

            cfg_llm = settings.llm_internal
            provider = (cfg_llm.provider or "").lower()
            has_key = bool(cfg_llm.api_key or os.environ.get("OPENAI_API_KEY", ""))
            if provider in ("openai", "gemini", "claude", "anthropic") and not has_key:
                logger.warning(
                    "internal_llm_missing_api_key_fallback",
                    extra={"provider": provider, "model": cfg_llm.model},
                )
            else:
                internal_llm = get_internal_llm_client()
        embedding_client: EmbeddingClient = get_embedding_client()

        # Phase 2.3: wrap with Redis cache when available and enabled
        redis_client = getattr(db_manager, "redis", None)
        if redis_client and settings.features.cached_embeddings_enabled:
            embedding_client = CachedEmbeddings(
                client=embedding_client,
                redis_client=redis_client,
                ttl_seconds=86400,
            )

        episodic_store = PostgresMemoryStore(db_manager.pg_session)
        graph_store = Neo4jGraphStore(db_manager.neo4j_driver)
        fact_store = SemanticFactStore(db_manager.pg_session)
        neocortical = NeocorticalStore(graph_store, fact_store)

        from ..storage.neo4j import initialize_graph_schema

        try:
            await initialize_graph_schema(graph_store)
        except Exception as exc:
            logger.warning("neo4j_schema_init_failed", extra={"error": str(exc)})

        short_term_config = ShortTermMemoryConfig()
        short_term = ShortTermMemory(config=short_term_config)
        from ..extraction.unified_write_extractor import UnifiedWritePathExtractor

        unified_extractor = UnifiedWritePathExtractor(internal_llm) if internal_llm else None
        hippocampal = HippocampalStore(
            vector_store=episodic_store,
            embedding_client=embedding_client,
            unified_extractor=unified_extractor,
        )

        from ..consolidation.windowed_consolidator import WindowedMicroSleepConsolidator
        from ..memory.dmn import PersistentDMNStore
        from ..memory.procedural import ProceduralMemoryEngine
        from ..reconsolidation.dopamine_rpe import DopamineRPEEngine
        from ..retrieval.memory_retriever import MemoryRetriever
        from ..retrieval.preplay import HippocampalPreplaySimulator

        dmn_store = PersistentDMNStore(pg_session_factory=db_manager.pg_session)
        procedural_engine = ProceduralMemoryEngine(
            embedding_client=embedding_client,
            pg_session_factory=db_manager.pg_session,
        )

        retriever = MemoryRetriever(
            hippocampal=hippocampal,
            neocortical=neocortical,
            llm_client=internal_llm,
            short_term=short_term,
            dmn_store=dmn_store,
            procedural_engine=procedural_engine,
        )

        raw_wsize = getattr(settings.features, "windowed_consolidation_turn_size", 18)
        wsize = raw_wsize if isinstance(raw_wsize, int) else 18
        windowed_consolidator = WindowedMicroSleepConsolidator(
            episodic_store=episodic_store,
            neocortical_store=neocortical,
            embedding_client=embedding_client,
            llm_client=internal_llm,
            dmn_store=dmn_store,
            window_size=wsize,
        )
        dopamine_rpe = DopamineRPEEngine(
            episodic_store=episodic_store,
            neocortical_store=neocortical,
            procedural_engine=procedural_engine,
            embedding_client=embedding_client,
        )
        preplay_simulator = HippocampalPreplaySimulator(
            retriever=retriever,
            dmn_store=dmn_store,
            procedural_engine=procedural_engine,
            llm_client=internal_llm,
        )

        reconsolidation = ReconsolidationService(
            memory_store=episodic_store,
            llm_client=internal_llm,
            fact_extractor=LLMFactExtractor(internal_llm) if internal_llm is not None else None,
            redis_client=getattr(db_manager, "redis", None),
        )

        consolidation = ConsolidationWorker(
            episodic_store=episodic_store,
            neocortical_store=neocortical,
            llm_client=internal_llm,
        )

        forgetting = ForgettingWorker(
            store=episodic_store,
            compression_llm_client=internal_llm,
        )

        return cls(
            short_term=short_term,
            hippocampal=hippocampal,
            neocortical=neocortical,
            retriever=retriever,
            reconsolidation=reconsolidation,
            consolidation=consolidation,
            forgetting=forgetting,
            dmn_store=dmn_store,
            procedural_engine=procedural_engine,
            windowed_consolidator=windowed_consolidator,
            dopamine_rpe=dopamine_rpe,
            preplay_simulator=preplay_simulator,
        )

    @classmethod
    async def create_lite(
        cls,
        episodic_store: MemoryStoreBase,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient | None = None,
    ) -> "MemoryOrchestrator":
        """Factory for embedded lite mode: SQLite-backed Neocortical + Hypergraph + DMN parity."""
        graph_store: Any
        fact_store: Any
        is_sqlite_mem = type(episodic_store).__name__ == "SQLiteMemoryStore"
        if is_sqlite_mem:
            try:
                from cml.storage.sqlite_neocortical import SQLiteFactStore, SQLiteGraphStore

                graph_store = SQLiteGraphStore(episodic_store)
                fact_store = SQLiteFactStore(episodic_store)
            except Exception:
                graph_store = NoOpGraphStore()
                fact_store = NoOpFactStore()
        else:
            graph_store = NoOpGraphStore()
            fact_store = NoOpFactStore()
        neocortical = NeocorticalStore(graph_store, fact_store)

        short_term_config = ShortTermMemoryConfig(min_salience_for_encoding=0.0)
        short_term = ShortTermMemory(config=short_term_config)
        from ..core.config import get_settings

        settings = get_settings()
        runtime_llm = llm_client if settings.features.use_llm_enabled else None

        from ..consolidation.windowed_consolidator import WindowedMicroSleepConsolidator
        from ..extraction.unified_write_extractor import UnifiedWritePathExtractor
        from ..memory.dmn import PersistentDMNStore
        from ..memory.procedural import ProceduralMemoryEngine
        from ..reconsolidation.dopamine_rpe import DopamineRPEEngine
        from ..retrieval.memory_retriever import MemoryRetriever
        from ..retrieval.preplay import HippocampalPreplaySimulator

        hippocampal = HippocampalStore(
            vector_store=episodic_store,
            embedding_client=embedding_client,
            unified_extractor=UnifiedWritePathExtractor(runtime_llm) if runtime_llm else None,
        )

        sq_ref = episodic_store if is_sqlite_mem else None
        dmn_store = PersistentDMNStore(sqlite_store=sq_ref)
        procedural_engine = ProceduralMemoryEngine(
            embedding_client=embedding_client,
            sqlite_store=sq_ref,
        )

        retriever = MemoryRetriever(
            hippocampal=hippocampal,
            neocortical=neocortical,
            llm_client=runtime_llm,
            short_term=short_term,
            dmn_store=dmn_store,
            procedural_engine=procedural_engine,
        )

        raw_wsize = getattr(settings.features, "windowed_consolidation_turn_size", 18)
        wsize = raw_wsize if isinstance(raw_wsize, int) else 18
        windowed_consolidator = WindowedMicroSleepConsolidator(
            episodic_store=episodic_store,
            neocortical_store=neocortical,
            embedding_client=embedding_client,
            llm_client=runtime_llm,
            dmn_store=dmn_store,
            window_size=wsize,
        )
        dopamine_rpe = DopamineRPEEngine(
            episodic_store=episodic_store,
            neocortical_store=neocortical,
            procedural_engine=procedural_engine,
            embedding_client=embedding_client,
        )
        preplay_simulator = HippocampalPreplaySimulator(
            retriever=retriever,
            dmn_store=dmn_store,
            procedural_engine=procedural_engine,
            llm_client=runtime_llm,
        )

        reconsolidation = ReconsolidationService(
            memory_store=episodic_store,
            llm_client=runtime_llm,
            fact_extractor=LLMFactExtractor(runtime_llm) if runtime_llm is not None else None,
            redis_client=None,
        )

        consolidation = ConsolidationWorker(
            episodic_store=episodic_store,
            neocortical_store=neocortical,
            llm_client=runtime_llm,
        )

        forgetting = ForgettingWorker(
            store=episodic_store,
            compression_llm_client=runtime_llm,
        )

        return cls(
            short_term=short_term,
            hippocampal=hippocampal,
            neocortical=neocortical,
            retriever=retriever,
            reconsolidation=reconsolidation,
            consolidation=consolidation,
            forgetting=forgetting,
            dmn_store=dmn_store,
            procedural_engine=procedural_engine,
            windowed_consolidator=windowed_consolidator,
            dopamine_rpe=dopamine_rpe,
            preplay_simulator=preplay_simulator,
        )

    async def write(
        self,
        tenant_id: str,
        content: str,
        context_tags: list[str] | None = None,
        session_id: str | None = None,
        memory_type: Any | None = None,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
        agent_id: str | None = None,
        namespace: str | None = None,
        timestamp: datetime | None = None,
        eval_mode: bool = False,
        media: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Write new information (plus optional multimodal media) to memory. Holistic: tenant-only."""
        import time as _time

        _t0 = _time.perf_counter()

        raw_media = media or ((metadata or {}).get("media") if isinstance(metadata, dict) else None)
        if raw_media and isinstance(raw_media, list):
            try:
                from ..core.config import get_settings as _get_mm_settings

                if getattr(_get_mm_settings().features, "perirhinal_multimodal_enabled", True):
                    from ..memory.sensory.perirhinal import PerirhinalSensoryStream

                    descriptors = await PerirhinalSensoryStream.extract_descriptors(
                        raw_media, llm_client=getattr(self.retriever, "llm_client", None)
                    )
                    if descriptors:
                        content = PerirhinalSensoryStream.enrich_content_with_sensory(
                            content, descriptors
                        )
                        metadata = dict(metadata or {})
                        metadata["perirhinal_descriptors"] = descriptors
            except Exception as exc:
                logger.debug("perirhinal_write_enrich_failed", extra={"error": str(exc)})

        chunks_for_encoding = await self._phase_ingest(
            tenant_id, content, session_id, turn_id, timestamp
        )
        _t_ingest = _time.perf_counter()

        if not chunks_for_encoding:
            return self._empty_write_response(eval_mode)

        # Resolve memory_type override if provided (deferred import for circular dep)
        from ..core.enums import MemoryType as _MemoryType

        _memory_type_override = None
        if memory_type is not None:
            if isinstance(memory_type, _MemoryType):
                _memory_type_override = memory_type
            else:
                try:
                    _memory_type_override = (
                        _MemoryType(memory_type)
                        if isinstance(memory_type, str)
                        else _MemoryType(memory_type.value)
                    )
                except (ValueError, AttributeError):
                    logger.warning(
                        "invalid_memory_type_override",
                        extra={"memory_type": str(memory_type), "tenant_id": tenant_id},
                    )

        # Resolve all feature flags once (S-02: no repeated get_settings() in phases)
        from ..core.config import get_settings as _get_settings

        wpc = WritePathConfig.from_settings(
            _get_settings(),
            has_unified_extractor=getattr(self.hippocampal, "unified_extractor", None) is not None,
        )
        # eval_mode skips the *batched* unified extraction and constraint deactivation.
        # Note it does not skip LLM enrichment: encode_batch re-runs unified extraction
        # per chunk when unified_results is None, so eval mode substitutes N per-chunk
        # extract() calls for one extract_batch(). Deactivation stays skipped because it
        # is the one phase here that makes extra LLM calls (detect_supersession).
        if not eval_mode:
            unified_results = await self._phase_unified_extraction(chunks_for_encoding, wpc)
        else:
            unified_results = None
        _t_unified = _time.perf_counter()

        if not eval_mode:
            await self._phase_deactivate_constraints(
                tenant_id, chunks_for_encoding, unified_results, wpc
            )
        _t_deactivate = _time.perf_counter()

        (
            stored,
            gate_results,
            unified_results,
            stored_chunks,
        ) = await self._phase_encode_and_store(
            tenant_id=tenant_id,
            chunks=chunks_for_encoding,
            context_tags=context_tags,
            session_id=session_id,
            agent_id=agent_id,
            namespace=namespace,
            metadata=metadata,
            memory_type_override=_memory_type_override,
            eval_mode=eval_mode,
            unified_results=unified_results,
        )
        _t_encode = _time.perf_counter()

        # These run on the eval path too. They used to be skipped as "low-quality
        # neocortical data", but _sync_to_graph is the only writer of Neo4j entities and
        # relations in the codebase, so skipping it left the graph empty for every eval
        # tenant — and the LoCoMo-Plus multi-hop score was therefore measured against
        # nothing. Same for the fact prong: no write-time facts meant _retrieve_facts had
        # nothing to return. None of the three calls an LLM; the latency argument for
        # skipping them does not hold.
        graph_task = asyncio.create_task(self._sync_to_graph(tenant_id, stored)) if stored else None
        # stored_chunks is positionally aligned with stored / unified_results
        # (one entry per stored record), so stored[idx] always corresponds to
        # chunks[idx] in both phases — even when the write gate skipped some
        # chunks or an upsert was dropped.
        _facts_coro = self._phase_write_time_facts(
            tenant_id,
            stored_chunks,
            stored,
            unified_results,
            timestamp,
            wpc,
        )
        _constraints_coro = self._phase_write_constraints(
            tenant_id, stored_chunks, stored, unified_results, timestamp, wpc
        )
        await asyncio.gather(_facts_coro, _constraints_coro)
        _t_facts = _time.perf_counter()
        _t_constraints = _t_facts

        # Awaited here, not inside the block that created it: an un-awaited task is
        # GC-eligible mid-flight, which is how graph writes would silently vanish.
        if graph_task is not None:
            try:
                await asyncio.wait_for(graph_task, timeout=2.0)
            except Exception as exc:
                logger.warning("graph_sync_incomplete", extra={"error": str(exc)})
        _t_graph = _time.perf_counter()

        logger.info(
            "write_phase_timing",
            extra={
                "tenant_id": tenant_id,
                "ingest_ms": round((_t_ingest - _t0) * 1000, 1),
                "unified_ms": round((_t_unified - _t_ingest) * 1000, 1),
                "deactivate_ms": round((_t_deactivate - _t_unified) * 1000, 1),
                "encode_ms": round((_t_encode - _t_deactivate) * 1000, 1),
                "facts_ms": round((_t_facts - _t_encode) * 1000, 1),
                "constraints_ms": round((_t_constraints - _t_facts) * 1000, 1),
                "graph_ms": round((_t_graph - _t_constraints) * 1000, 1),
                "total_ms": round((_t_graph - _t0) * 1000, 1),
            },
        )
        return self._build_write_response(stored, gate_results, eval_mode)

    async def write_batch(
        self,
        tenant_id: str,
        turns: list[dict[str, Any]],
        eval_mode: bool = False,
    ) -> dict[str, Any]:
        """Write multiple turns to memory in a single call.

        Each item in *turns* is a dict with keys matching WriteMemoryRequest:
        content, session_id, metadata, turn_id, timestamp (optional).
        Returns aggregate stats.

        Turns are processed concurrently in micro-batches so that the
        BatchingEmbeddingClient can coalesce GPU encode calls.
        """
        import time as _time

        _t0 = _time.perf_counter()
        total_stored = 0
        total_chunks = 0

        valid_turns = [t for t in turns if t.get("content")]

        # Process in concurrent micro-batches of 8 so embedding coalescing
        # can batch GPU calls while not overwhelming DB connections.
        _micro_batch = 8
        for batch_start in range(0, len(valid_turns), _micro_batch):
            batch = valid_turns[batch_start : batch_start + _micro_batch]
            coros = [
                self.write(
                    tenant_id=tenant_id,
                    content=turn["content"],
                    session_id=turn.get("session_id"),
                    metadata=turn.get("metadata"),
                    turn_id=turn.get("turn_id"),
                    timestamp=turn.get("timestamp"),
                    eval_mode=eval_mode,
                )
                for turn in batch
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException):
                    logger.warning("write_batch_turn_error", error=str(r))
                    continue
                total_chunks += r.get("chunks_created", 0)
                if r.get("memory_id"):
                    total_stored += 1

        elapsed_ms = round((_time.perf_counter() - _t0) * 1000, 1)
        logger.info(
            "write_batch_timing",
            extra={
                "tenant_id": tenant_id,
                "turns": len(turns),
                "total_ms": elapsed_ms,
                "avg_ms": round(elapsed_ms / max(len(turns), 1), 1),
            },
        )
        return {
            "success": True,
            "turns_processed": len(valid_turns),
            "chunks_created": total_chunks,
            "message": f"Batch wrote {len(valid_turns)} turns ({elapsed_ms:.0f}ms)",
        }

    async def read(
        self,
        tenant_id: str,
        query: str,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        user_timezone: str | None = None,
        source_session_id: str | None = None,
    ) -> MemoryPacket:
        """Retrieve relevant memories. Holistic: tenant-only."""
        return await self.retriever.retrieve(
            tenant_id=tenant_id,
            query=query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            user_timezone=user_timezone,
            source_session_id=source_session_id,
        )

    async def update(
        self,
        tenant_id: str,
        memory_id: UUID,
        text: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        metadata: dict | None = None,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing memory. Holistic: tenant-only."""
        record = await self.hippocampal.store.get_by_id(memory_id)
        if not record:
            raise MemoryNotFoundError(memory_id)
        if record.tenant_id != tenant_id:
            raise MemoryAccessDenied("Memory does not belong to tenant")

        patch: dict[str, Any] = {}
        if text is not None:
            patch["text"] = text
            emb_result = await self.hippocampal.embeddings.embed(text)
            patch["embedding"] = emb_result.embedding
        if confidence is not None:
            patch["confidence"] = confidence
        if importance is not None:
            patch["importance"] = importance
        if metadata is not None:
            patch["metadata"] = metadata

        if feedback == "correct":
            patch["confidence"] = min(
                1.0, (record.confidence if confidence is None else confidence) + 0.2
            )
        elif feedback == "incorrect":
            patch["confidence"] = 0.0
            patch["status"] = MemoryStatus.DELETED.value
        elif feedback == "outdated":
            patch["valid_to"] = datetime.now(UTC)

        result = await self.hippocampal.store.update(memory_id, patch)
        return {
            "version": result.version if result else 1,
        }

    async def forget(
        self,
        tenant_id: str,
        memory_ids: list[UUID] | None = None,
        query: str | None = None,
        before: datetime | None = None,
        action: str = "delete",
    ) -> dict[str, Any]:
        """Forget memories. Holistic: tenant-only.

        Collects all target IDs into a set first to avoid double-counting when
        multiple criteria overlap (MED-19).
        """
        target_ids: set[UUID] = set()

        def owns(record) -> bool:
            return record.tenant_id == tenant_id

        # Collect IDs from explicit memory_ids (batch fetch)
        if memory_ids:
            records = await self.hippocampal.store.get_by_ids_batch(memory_ids)
            for record in records:
                if owns(record):
                    target_ids.add(record.id)

        # Collect IDs from query-based search using direct vector similarity so
        # forget(query=...) only deletes genuine matches and never wipes unrelated
        # memories promoted by retrieval RRF floors or constraint injection.
        if query:
            embeddings = getattr(self.hippocampal, "embeddings", None)
            if embeddings is not None and hasattr(self.hippocampal.store, "vector_search"):
                emb_result = await embeddings.embed(query)
                matched = await self.hippocampal.store.vector_search(
                    tenant_id,
                    emb_result.embedding,
                    top_k=100,
                    filters={"status": MemoryStatus.ACTIVE.value},
                    min_similarity=0.55,
                )
                for record in matched:
                    if owns(record):
                        target_ids.add(record.id)
            else:
                packet = await self.retriever.retrieve(tenant_id, query=query, max_results=100)
                ids_from_query = [
                    mem.record.id for mem in packet.all_memories if mem.relevance_score >= 0.85
                ]
                if ids_from_query:
                    records = await self.hippocampal.store.get_by_ids_batch(ids_from_query)
                    for record in records:
                        if owns(record):
                            target_ids.add(record.id)

        # Collect IDs by time filter (cursor-based pagination)
        if before:
            from src.storage.utils import naive_utc

            before_utc = naive_utc(before)
            offset = 0
            while True:
                records = await self.hippocampal.store.scan(
                    tenant_id,
                    filters={"status": MemoryStatus.ACTIVE.value},
                    limit=500,
                    offset=offset,
                )
                if not records:
                    break
                for r in records:
                    rec_ts = naive_utc(r.timestamp)
                    if rec_ts is not None and before_utc is not None and rec_ts < before_utc:
                        target_ids.add(r.id)
                offset += len(records)

        # Apply forget action to deduplicated set
        if action == "delete":
            for mid in target_ids:
                await self.hippocampal.store.delete(mid, hard=True)
        elif action == "archive":
            for mid in target_ids:
                await self.hippocampal.store.update(
                    mid, {"status": MemoryStatus.ARCHIVED.value}, increment_version=False
                )
        elif action == "silence":
            for mid in target_ids:
                await self.hippocampal.store.update(
                    mid, {"status": MemoryStatus.SILENT.value}, increment_version=False
                )
        else:
            for mid in target_ids:
                await self.hippocampal.store.delete(mid, hard=False)

        return {"affected_count": len(target_ids)}

    async def get_session_context(
        self,
        tenant_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Get full session context for LLM (messages, tool_results, scratch_pad, context_string).

        When session_id is provided, scopes retrieval to memories from that
        session only (via source_session_id filter) instead of returning all
        tenant memories.
        """
        if session_id:
            # Scoped retrieval: only memories belonging to this session
            filters: dict[str, Any] = {
                "status": MemoryStatus.ACTIVE.value,
                "source_session_id": session_id,
            }
            records = await self.hippocampal.store.scan(
                tenant_id,
                filters=filters,
                order_by="-timestamp",
                limit=_SESSION_CONTEXT_LIMIT,
            )
            # Build items directly from records
            messages = []
            tool_results = []
            scratch_pad = []
            for r in records:
                t = r.type.value if hasattr(r.type, "value") else str(r.type)
                item = {
                    "id": r.id,
                    "text": r.text,
                    "type": t,
                    "confidence": r.confidence,
                    "relevance": 1.0,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata or {},
                }
                if t in ("message", "conversation"):
                    messages.append(item)
                elif t == "tool_result":
                    tool_results.append(item)
                elif t == "scratch":
                    scratch_pad.append(item)
                else:
                    messages.append(item)

            from ..core.schemas import RetrievedMemory
            from ..retrieval.packet_builder import MemoryPacketBuilder

            # Build a minimal packet for context string generation
            retrieved = [
                RetrievedMemory(record=r, relevance_score=1.0, retrieval_source="session")
                for r in records
            ]
            builder = MemoryPacketBuilder()
            packet = builder.build(retrieved, query="")
            context_string = builder.to_llm_context(packet, max_tokens=4000)
        else:
            # Fallback: tenant-wide retrieval when no session specified
            packet = await self.retriever.retrieve(
                tenant_id=tenant_id,
                query="",
                max_results=50,
            )
            messages = []
            tool_results = []
            scratch_pad = []
            for m in packet.all_memories:
                t = m.record.type.value if hasattr(m.record.type, "value") else str(m.record.type)
                item = {
                    "id": m.record.id,
                    "text": m.record.text,
                    "type": t,
                    "confidence": m.record.confidence,
                    "relevance": m.relevance_score,
                    "timestamp": m.record.timestamp,
                    "metadata": m.record.metadata or {},
                }
                if t in ("message", "conversation"):
                    messages.append(item)
                elif t == "tool_result":
                    tool_results.append(item)
                elif t == "scratch":
                    scratch_pad.append(item)
                else:
                    messages.append(item)
            from ..retrieval.packet_builder import MemoryPacketBuilder

            builder = MemoryPacketBuilder()
            context_string = builder.to_llm_context(packet, max_tokens=4000)

        return {
            "messages": messages,
            "tool_results": tool_results,
            "scratch_pad": scratch_pad,
            "context_string": context_string,
        }

    async def delete_all(
        self,
        tenant_id: str,
    ) -> int:
        """Delete all memories for a tenant (GDPR). Holistic: tenant-only. Paginates until none left."""
        affected = 0
        while True:
            records = await self.hippocampal.store.scan(tenant_id, limit=1000)
            if not records:
                break
            for r in records:
                await self.hippocampal.store.delete(r.id, hard=True)
                affected += 1
        return affected

    async def get_stats(
        self,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Get memory statistics for tenant. Holistic: tenant-only.

        Uses count queries per status and per type for accurate results even
        when the tenant has more than 1,000 records (MED-18).
        """
        total = await self.hippocampal.store.count(tenant_id)
        active = await self.hippocampal.store.count(
            tenant_id,
            filters={"status": MemoryStatus.ACTIVE.value},
        )
        silent = await self.hippocampal.store.count(
            tenant_id,
            filters={"status": MemoryStatus.SILENT.value},
        )
        archived = await self.hippocampal.store.count(
            tenant_id,
            filters={"status": MemoryStatus.ARCHIVED.value},
        )

        # Count per type using dedicated count queries (not limited to first N records)
        by_type: dict[str, int] = {}
        for mt in MemoryType:
            cnt = await self.hippocampal.store.count(tenant_id, filters={"type": mt.value})
            if cnt > 0:
                by_type[mt.value] = cnt

        # Sample a limited set for aggregate statistics
        records = await self.hippocampal.store.scan(tenant_id, limit=1000, order_by="-timestamp")
        timestamps = [r.timestamp for r in records if r.timestamp]
        confidences = [r.confidence for r in records]
        importances = [r.importance for r in records]

        return {
            "total_memories": total,
            "active_memories": active,
            "silent_memories": silent,
            "archived_memories": archived,
            "by_type": by_type,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "avg_importance": sum(importances) / len(importances) if importances else 0.0,
            "oldest_memory": min(timestamps) if timestamps else None,
            "newest_memory": max(timestamps) if timestamps else None,
            "estimated_size_mb": total * 0.001,
        }

    async def consolidate_windowed(
        self,
        tenant_id: str,
        *,
        session_id: str | None = None,
        max_episodes: int = 2000,
        use_llm: bool | None = None,
    ) -> dict[str, int]:
        """Run Windowed Micro-Sleep Consolidation (Hyperedges + Proustian Cues + Conway + DMN)."""
        from ..consolidation.windowed_consolidator import WindowedMicroSleepConsolidator
        from ..core.config import get_settings

        if not getattr(get_settings().features, "windowed_consolidation_enabled", True):
            return {
                "episodes_scanned": 0,
                "windows_processed": 0,
                "hyperedges_created": 0,
                "proustian_cues_created": 0,
                "chapter_gists_created": 0,
                "facts_upserted": 0,
            }

        consolidator = self.windowed_consolidator
        if consolidator is None:
            consolidator = WindowedMicroSleepConsolidator(
                episodic_store=self.hippocampal.store,
                neocortical_store=self.neocortical,
                embedding_client=self.hippocampal.embeddings,
                llm_client=getattr(self.retriever, "llm_client", None),
                dmn_store=self.dmn_store,
            )
            self.windowed_consolidator = consolidator

        # Allow explicit on-demand System-2 LLM consolidation even when bulk write ran with use_llm=false
        prev_llm = consolidator.llm_client
        if use_llm is True and prev_llm is None:
            try:
                consolidator.llm_client = get_internal_llm_client()
            except Exception:
                pass
        elif use_llm is False:
            consolidator.llm_client = None

        try:
            return await consolidator.consolidate_tenant(
                tenant_id=tenant_id,
                session_id=session_id,
                max_episodes=max_episodes,
            )
        finally:
            if use_llm is False:
                consolidator.llm_client = prev_llm

    async def feedback(
        self,
        tenant_id: str,
        *,
        outcome_reward: float,
        expected_reward: float = 0.5,
        query: str | None = None,
        memory_ids: list[str] | None = None,
        correction_text: str | None = None,
        preferred_steps: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply Dopaminergic Reward Prediction Error (RPE) feedback & Procedural compilation."""
        from ..memory.procedural import ProceduralMemoryEngine
        from ..reconsolidation.dopamine_rpe import DopamineRPEEngine

        if self.dopamine_rpe is None:
            proc = self.procedural_engine or ProceduralMemoryEngine(
                embedding_client=self.hippocampal.embeddings
            )
            self.procedural_engine = proc
            self.dopamine_rpe = DopamineRPEEngine(
                episodic_store=self.hippocampal.store,
                neocortical_store=self.neocortical,
                procedural_engine=proc,
                embedding_client=self.hippocampal.embeddings,
            )

        res = await self.dopamine_rpe.process_feedback(
            tenant_id=tenant_id,
            outcome_reward=outcome_reward,
            expected_reward=expected_reward,
            query=query,
            memory_ids=memory_ids,
            correction_text=correction_text,
            preferred_steps=preferred_steps,
            session_id=session_id,
        )
        return res.to_dict()

    async def simulate(
        self,
        tenant_id: str,
        scenario: str,
        *,
        max_results: int = 12,
        context_filter: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute Hippocampal Forward Preplay / Counterfactual Simulation."""
        from ..retrieval.preplay import HippocampalPreplaySimulator

        if self.preplay_simulator is None:
            self.preplay_simulator = HippocampalPreplaySimulator(
                retriever=self.retriever,
                dmn_store=self.dmn_store,
                procedural_engine=self.procedural_engine,
                llm_client=getattr(self.retriever, "llm_client", None),
            )
        verdict = await self.preplay_simulator.simulate(
            tenant_id=tenant_id,
            scenario=scenario,
            max_results=max_results,
            context_filter=context_filter,
        )
        return verdict.to_dict()
