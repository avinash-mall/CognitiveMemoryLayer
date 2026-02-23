"""Memory orchestrator: coordinates all memory operations."""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ..consolidation.worker import ConsolidationWorker
from ..core.enums import MemoryStatus
from ..core.schemas import MemoryPacket
from ..extraction.entity_extractor import EntityExtractor
from ..extraction.fact_extractor import LLMFactExtractor
from ..extraction.relation_extractor import RelationExtractor
from ..forgetting.worker import ForgettingWorker
from ..memory.conversation import ConversationMemory
from ..memory.hippocampal.store import HippocampalStore
from ..memory.knowledge_base import KnowledgeBase
from ..memory.neocortical.fact_store import SemanticFactStore
from ..memory.neocortical.store import NeocorticalStore
from ..memory.scratch_pad import ScratchPad
from ..memory.short_term import ShortTermMemory, ShortTermMemoryConfig
from ..memory.tool_memory import ToolMemory
from ..reconsolidation.service import ReconsolidationService

if TYPE_CHECKING:
    from ..retrieval.memory_retriever import MemoryRetriever
from ..storage.base import MemoryStoreBase
from ..storage.connection import DatabaseManager
from ..storage.neo4j import Neo4jGraphStore
from ..storage.noop_stores import NoOpFactStore, NoOpGraphStore
from ..storage.postgres import PostgresMemoryStore
from ..utils.embeddings import CachedEmbeddings, EmbeddingClient, get_embedding_client
from ..utils.llm import LLMClient, get_internal_llm_client

logger = logging.getLogger(__name__)

try:
    from ..core.enums import MemoryType
except ImportError:
    MemoryType = None  # type: ignore


class MemoryOrchestrator:
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
        scratch_pad: ScratchPad,
        conversation: ConversationMemory,
        tool_memory: ToolMemory,
        knowledge_base: KnowledgeBase,
    ):
        self.short_term = short_term
        self.hippocampal = hippocampal
        self.neocortical = neocortical
        self.retriever = retriever
        self.reconsolidation = reconsolidation
        self.consolidation = consolidation
        self.forgetting = forgetting
        self.scratch_pad = scratch_pad
        self.conversation = conversation
        self.tool_memory = tool_memory
        self.knowledge_base = knowledge_base

    @classmethod
    async def create(cls, db_manager: DatabaseManager) -> "MemoryOrchestrator":
        """Factory method to create orchestrator with all dependencies."""
        from ..core.config import get_settings

        settings = get_settings()
        internal_llm = get_internal_llm_client() if settings.features.use_llm_enabled else None
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

        short_term_config = ShortTermMemoryConfig()
        short_term = ShortTermMemory(config=short_term_config)
        entity_extractor = (
            EntityExtractor(internal_llm)
            if internal_llm and settings.features.use_llm_enabled
            else None
        )
        relation_extractor = (
            RelationExtractor(internal_llm)
            if internal_llm and settings.features.use_llm_enabled
            else None
        )
        from ..extraction.unified_write_extractor import UnifiedWritePathExtractor

        _use_unified = settings.features.use_llm_enabled and (
            settings.features.use_llm_constraint_extractor
            or settings.features.use_llm_write_time_facts
            or settings.features.use_llm_salience_refinement
            or settings.features.use_llm_pii_redaction
            or settings.features.use_llm_write_gate_importance
        )
        if _use_unified:
            entity_extractor = None
            relation_extractor = None

        unified_extractor = (
            UnifiedWritePathExtractor(internal_llm) if _use_unified and internal_llm else None
        )
        hippocampal = HippocampalStore(
            vector_store=episodic_store,
            embedding_client=embedding_client,
            entity_extractor=entity_extractor,
            relation_extractor=relation_extractor,
            unified_extractor=unified_extractor,
        )

        from ..retrieval.memory_retriever import MemoryRetriever

        retriever = MemoryRetriever(
            hippocampal=hippocampal,
            neocortical=neocortical,
            llm_client=internal_llm,
        )

        reconsolidation = ReconsolidationService(
            memory_store=episodic_store,
            llm_client=internal_llm,
            fact_extractor=LLMFactExtractor(internal_llm),
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

        scratch_pad = ScratchPad(store=episodic_store)
        conversation = ConversationMemory(store=episodic_store)
        tool_memory = ToolMemory(store=episodic_store)
        knowledge_base = KnowledgeBase(store=episodic_store, embedding_client=embedding_client)

        return cls(
            short_term=short_term,
            hippocampal=hippocampal,
            neocortical=neocortical,
            retriever=retriever,
            reconsolidation=reconsolidation,
            consolidation=consolidation,
            forgetting=forgetting,
            scratch_pad=scratch_pad,
            conversation=conversation,
            tool_memory=tool_memory,
            knowledge_base=knowledge_base,
        )

    @classmethod
    async def create_lite(
        cls,
        episodic_store: MemoryStoreBase,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
    ) -> "MemoryOrchestrator":
        """Factory for embedded lite mode: no PostgreSQL/Neo4j/Redis; uses provided episodic store and clients."""
        graph_store = NoOpGraphStore()
        fact_store = NoOpFactStore()
        neocortical = NeocorticalStore(graph_store, fact_store)

        short_term_config = ShortTermMemoryConfig(min_salience_for_encoding=0.0)
        short_term = ShortTermMemory(config=short_term_config)
        hippocampal = HippocampalStore(
            vector_store=episodic_store,
            embedding_client=embedding_client,
        )

        from ..retrieval.memory_retriever import MemoryRetriever

        retriever = MemoryRetriever(
            hippocampal=hippocampal,
            neocortical=neocortical,
            llm_client=llm_client,
        )

        reconsolidation = ReconsolidationService(
            memory_store=episodic_store,
            llm_client=llm_client,
            fact_extractor=LLMFactExtractor(llm_client),
            redis_client=None,
        )

        consolidation = ConsolidationWorker(
            episodic_store=episodic_store,
            neocortical_store=neocortical,
            llm_client=llm_client,
        )

        forgetting = ForgettingWorker(
            store=episodic_store,
            compression_llm_client=llm_client,
        )

        scratch_pad = ScratchPad(store=episodic_store)
        conversation = ConversationMemory(store=episodic_store)
        tool_memory = ToolMemory(store=episodic_store)
        knowledge_base = KnowledgeBase(store=episodic_store, embedding_client=embedding_client)

        return cls(
            short_term=short_term,
            hippocampal=hippocampal,
            neocortical=neocortical,
            retriever=retriever,
            reconsolidation=reconsolidation,
            consolidation=consolidation,
            forgetting=forgetting,
            scratch_pad=scratch_pad,
            conversation=conversation,
            tool_memory=tool_memory,
            knowledge_base=knowledge_base,
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
    ) -> dict[str, Any]:
        """Write new information to memory. Holistic: tenant-only."""
        stm_result = await self.short_term.ingest_turn(
            tenant_id=tenant_id,
            scope_id=session_id or tenant_id,
            text=content,
            turn_id=turn_id,
            role="user",
            timestamp=timestamp,
        )

        chunks_for_encoding = stm_result.get("chunks_for_encoding", [])

        if not chunks_for_encoding:
            out = {
                "memory_id": None,
                "chunks_created": 0,
                "message": "No significant information to store",
            }
            if eval_mode:
                out["eval_outcome"] = "skipped"
                out["eval_reason"] = "No significant information to store"
            return out

        # Resolve memory_type override if provided
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
                    pass  # Invalid memory_type; let write gate decide

        # Run unified extraction first when LLM path enabled (for deactivation + encode_batch)
        from ..core.config import get_settings as _get_settings

        _settings = _get_settings()
        _unified_results: list | None = None
        _use_unified = (
            _settings.features.use_llm_enabled
            and getattr(self.hippocampal, "unified_extractor", None) is not None
            and (
                _settings.features.use_llm_constraint_extractor
                or _settings.features.use_llm_write_time_facts
                or _settings.features.use_llm_salience_refinement
                or _settings.features.use_llm_pii_redaction
                or _settings.features.use_llm_write_gate_importance
            )
        )
        if (
            _use_unified
            and hasattr(self.hippocampal, "unified_extractor")
            and self.hippocampal.unified_extractor
        ):
            # Use extract_batch for a single LLM call covering all chunks instead of
            # N parallel extract() calls (1 LLM call regardless of chunk count).
            _unified_results = await self.hippocampal.unified_extractor.extract_batch(
                chunks_for_encoding
            )

        # Deactivate previous episodic constraints by fact key before writing (supersession)
        if _settings.features.constraint_extraction_enabled and chunks_for_encoding:
            if hasattr(self.hippocampal, "deactivate_constraints_by_key"):
                from ..extraction.constraint_extractor import ConstraintExtractor, ConstraintObject
                from ..memory.neocortical.schemas import FactCategory

                _fact_keys = set()
                _new_constraints: list = []
                if (
                    _unified_results
                    and _settings.features.use_llm_enabled
                    and _settings.features.use_llm_constraint_extractor
                ):
                    for _ur in _unified_results:
                        if _ur and hasattr(_ur, "constraints"):
                            for _c in _ur.constraints:
                                _fact_keys.add(ConstraintExtractor.constraint_fact_key(_c))
                                _new_constraints.append(_c)
                else:
                    _extractor = ConstraintExtractor()
                    for _chunk in chunks_for_encoding:
                        for _c in _extractor.extract(_chunk):
                            _fact_keys.add(ConstraintExtractor.constraint_fact_key(_c))
                            _new_constraints.append(_c)
                for _fk in _fact_keys:
                    await self.hippocampal.deactivate_constraints_by_key(tenant_id, _fk)
                # BUG-05: Deactivate overlapping constraints (different scope, same type)
                _cat_map = {
                    "goal": FactCategory.GOAL,
                    "value": FactCategory.VALUE,
                    "state": FactCategory.STATE,
                    "causal": FactCategory.CAUSAL,
                    "policy": FactCategory.POLICY,
                }
                for _nc in _new_constraints:
                    _cat = _cat_map.get((_nc.constraint_type or "").lower())
                    if not _cat:
                        continue
                    try:
                        _existing = await self.neocortical.facts.get_facts_by_category(
                            tenant_id, _cat, current_only=True
                        )
                        for _old in _existing:
                            if _old.key in _fact_keys:
                                continue
                            _old_obj = ConstraintObject(
                                constraint_type=_cat.value,
                                subject="user",
                                description=str(_old.value),
                                scope=getattr(_old, "context_tags", None) or [],
                            )
                            if await ConstraintExtractor.detect_supersession(
                                _old_obj,
                                _nc,
                                llm_client=getattr(
                                    getattr(self.hippocampal, "unified_extractor", None),
                                    "llm_client",
                                    None,
                                ),
                            ):
                                await self.neocortical.facts.deactivate_fact(_old.id)
                                await self.hippocampal.deactivate_constraints_by_key(
                                    tenant_id, _old.key
                                )
                                await self.neocortical.facts.invalidate_fact(
                                    tenant_id, _old.key, reason="superseded"
                                )
                    except Exception as e:
                        logger.warning(
                            "supersession_check_failed",
                            extra={"constraint_type": _nc.constraint_type, "error": str(e)},
                            exc_info=True,
                        )

        result = await self.hippocampal.encode_batch(
            tenant_id=tenant_id,
            chunks=chunks_for_encoding,
            context_tags=context_tags,
            source_session_id=session_id,
            agent_id=agent_id,
            namespace=namespace,
            timestamp=timestamp,
            request_metadata=metadata if metadata else None,
            memory_type_override=_memory_type_override,
            return_gate_results=eval_mode,
            unified_results=_unified_results,
        )

        stored, gate_results, unified_results = result

        # Sync extracted entities and relations to Neo4j knowledge graph
        if stored:
            await self._sync_to_graph(tenant_id, stored)

        # Phase 1.3: Write-time fact extraction — populate semantic store immediately
        from ..core.config import get_settings

        settings = get_settings()
        if settings.features.write_time_facts_enabled and chunks_for_encoding:
            try:
                evidence = [str(stored[0].id)] if stored else []
                # Use unified extractor facts when LLM path enabled, else rule-based
                if (
                    settings.features.use_llm_enabled
                    and settings.features.use_llm_write_time_facts
                    and unified_results
                ):
                    for ur in unified_results:
                        if ur and hasattr(ur, "facts"):
                            for fact in ur.facts:
                                try:
                                    await self.neocortical.store_fact(
                                        tenant_id=tenant_id,
                                        key=fact.key,
                                        value=fact.value,
                                        confidence=fact.confidence,
                                        evidence_ids=evidence,
                                        valid_from=timestamp,
                                    )
                                except Exception:
                                    logger.warning(
                                        "write_time_fact_store_failed",
                                        extra={"tenant_id": tenant_id, "fact_key": fact.key},
                                        exc_info=True,
                                    )
                else:
                    from ..extraction.write_time_facts import WriteTimeFactExtractor

                    extractor = WriteTimeFactExtractor()
                    for chunk in chunks_for_encoding:
                        extracted_facts = extractor.extract(chunk)
                        for fact in extracted_facts:
                            try:
                                await self.neocortical.store_fact(
                                    tenant_id=tenant_id,
                                    key=fact.key,
                                    value=fact.value,
                                    confidence=fact.confidence,
                                    evidence_ids=evidence,
                                    valid_from=timestamp,
                                )
                            except Exception:
                                logger.warning(
                                    "write_time_fact_store_failed",
                                    extra={"tenant_id": tenant_id, "fact_key": fact.key},
                                    exc_info=True,
                                )
            except Exception:
                logger.warning("write_time_facts_skipped", exc_info=True)

        # Cognitive: Write-time constraint extraction — store constraints as semantic facts
        if settings.features.constraint_extraction_enabled and chunks_for_encoding:
            try:
                from ..extraction.constraint_extractor import ConstraintExtractor

                constraint_extractor = ConstraintExtractor()
                constraints_stored = 0
                # Use unified extractor constraints when LLM path enabled, else rule-based
                if (
                    settings.features.use_llm_enabled
                    and settings.features.use_llm_constraint_extractor
                    and unified_results
                ):
                    for ur in unified_results:
                        if ur and hasattr(ur, "constraints"):
                            for constraint in ur.constraints:
                                try:
                                    evidence_c = [str(stored[0].id)] if stored else []
                                    fact_key = ConstraintExtractor.constraint_fact_key(constraint)
                                    await self.neocortical.store_fact(
                                        tenant_id=tenant_id,
                                        key=fact_key,
                                        value=constraint.description,
                                        confidence=constraint.confidence,
                                        evidence_ids=evidence_c,
                                        context_tags=constraint.scope,  # BUG-05: persist scope for supersession
                                        valid_from=timestamp,
                                    )
                                    constraints_stored += 1
                                except Exception:
                                    logger.warning(
                                        "constraint_fact_store_failed",
                                        extra={
                                            "tenant_id": tenant_id,
                                            "constraint_type": constraint.constraint_type,
                                        },
                                        exc_info=True,
                                    )
                else:
                    for chunk in chunks_for_encoding:
                        extracted_constraints = constraint_extractor.extract(chunk)
                        for constraint in extracted_constraints:
                            try:
                                evidence_c = [str(stored[0].id)] if stored else []
                                fact_key = ConstraintExtractor.constraint_fact_key(constraint)
                                await self.neocortical.store_fact(
                                    tenant_id=tenant_id,
                                    key=fact_key,
                                    value=constraint.description,
                                    confidence=constraint.confidence,
                                    evidence_ids=evidence_c,
                                    context_tags=constraint.scope,  # BUG-05: persist scope for supersession
                                    valid_from=timestamp,
                                )
                                constraints_stored += 1
                            except Exception:
                                logger.warning(
                                    "constraint_fact_store_failed",
                                    extra={
                                        "tenant_id": tenant_id,
                                        "constraint_type": constraint.constraint_type,
                                    },
                                    exc_info=True,
                                )
                if constraints_stored > 0:
                    logger.info(
                        "constraints_extracted",
                        extra={
                            "tenant_id": tenant_id,
                            "count": constraints_stored,
                        },
                    )
            except Exception:
                logger.warning("constraint_extraction_skipped", exc_info=True)

        if eval_mode:
            n_stored = len(stored)
            n_skipped = sum(1 for g in gate_results if g.get("decision") == "skip")
            if n_stored == 0:
                eval_outcome = "skipped"
                eval_reason = (
                    gate_results[0].get("reason", "unknown") if gate_results else "all skipped"
                )
            else:
                eval_outcome = "stored"
                if n_skipped:
                    reasons = [
                        g.get("reason", "") for g in gate_results if g.get("decision") == "skip"
                    ]
                    eval_reason = f"{n_stored} stored, {n_skipped} skipped: " + (
                        reasons[0] if reasons else ""
                    )
                else:
                    eval_reason = f"{n_stored} chunk(s) stored"
            return {
                "memory_id": stored[0].id if stored else None,
                "chunks_created": n_stored,
                "message": f"Stored {n_stored} memory chunks",
                "eval_outcome": eval_outcome,
                "eval_reason": eval_reason,
            }
        else:
            return {
                "memory_id": stored[0].id if stored else None,
                "chunks_created": len(stored),
                "message": f"Stored {len(stored)} memory chunks",
            }

    async def _sync_to_graph(
        self,
        tenant_id: str,
        records: list,
    ) -> None:
        """Sync extracted entities and relations from stored records to Neo4j.

        Failures are logged but never propagated — the Postgres write has
        already succeeded and must not be rolled back because of a graph issue.
        """
        for record in records:
            # 1. Merge entity nodes so the graph contains discoverable vertices
            for entity in getattr(record, "entities", None) or []:
                try:
                    await self.neocortical.graph.merge_node(
                        tenant_id=tenant_id,
                        scope_id=tenant_id,
                        entity=entity.normalized,
                        entity_type=entity.entity_type,
                    )
                except Exception:
                    logger.warning(
                        "neo4j_entity_sync_failed",
                        extra={
                            "tenant_id": tenant_id,
                            "entity": getattr(entity, "normalized", "?"),
                            "memory_id": str(record.id),
                        },
                        exc_info=True,
                    )

            # 2. Merge relation edges
            relations = getattr(record, "relations", None) or []
            if relations:
                try:
                    await self.neocortical.store_relations_batch(
                        tenant_id=tenant_id,
                        relations=relations,
                        evidence_ids=[str(record.id)],
                    )
                except Exception:
                    logger.warning(
                        "neo4j_relation_sync_failed",
                        extra={
                            "tenant_id": tenant_id,
                            "memory_id": str(record.id),
                            "relation_count": len(relations),
                        },
                        exc_info=True,
                    )

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
            raise ValueError(f"Memory {memory_id} not found")
        if record.tenant_id != tenant_id:
            raise ValueError("Memory does not belong to tenant")

        patch: dict[str, Any] = {}
        if text is not None:
            patch["text"] = text
            # Re-embed and re-extract entities when text changes
            emb_result = await self.hippocampal.embeddings.embed(text)
            patch["embedding"] = emb_result.embedding
            if self.hippocampal.entity_extractor:
                entities = await self.hippocampal.entity_extractor.extract(text)
                patch["entities"] = [e.model_dump() for e in entities]
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
        hard = action == "delete"
        target_ids: set[UUID] = set()

        def owns(record) -> bool:
            return record.tenant_id == tenant_id

        # Collect IDs from explicit memory_ids
        if memory_ids:
            for mid in memory_ids:
                record = await self.hippocampal.store.get_by_id(mid)
                if record and owns(record):
                    target_ids.add(mid)

        # Collect IDs from query-based search
        if query:
            packet = await self.retriever.retrieve(tenant_id, query=query, max_results=100)
            for mem in packet.all_memories:
                rid = mem.record.id
                record = await self.hippocampal.store.get_by_id(rid)
                if record and owns(record):
                    target_ids.add(rid)

        # Collect IDs by time filter
        if before:
            records = await self.hippocampal.store.scan(
                tenant_id, filters={"status": MemoryStatus.ACTIVE.value}, limit=500
            )
            for r in records:
                if r.timestamp and r.timestamp < before:
                    target_ids.add(r.id)

        # Delete deduplicated set
        for mid in target_ids:
            await self.hippocampal.store.delete(mid, hard=hard)

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
                limit=50,
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
        if MemoryType is not None:
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
