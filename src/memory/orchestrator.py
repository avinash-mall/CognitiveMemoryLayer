"""Memory orchestrator: coordinates all memory operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..core.enums import MemoryStatus
from ..core.schemas import MemoryPacket
from ..consolidation.worker import ConsolidationWorker
from ..forgetting.worker import ForgettingWorker
from ..memory.conversation import ConversationMemory
from ..memory.hippocampal.store import HippocampalStore
from ..memory.knowledge_base import KnowledgeBase
from ..memory.neocortical.fact_store import SemanticFactStore
from ..memory.neocortical.store import NeocorticalStore
from ..memory.scratch_pad import ScratchPad
from ..memory.short_term import ShortTermMemory
from ..memory.tool_memory import ToolMemory
from ..extraction.fact_extractor import LLMFactExtractor
from ..reconsolidation.service import ReconsolidationService
from ..retrieval.memory_retriever import MemoryRetriever
from ..storage.connection import DatabaseManager
from ..storage.neo4j import Neo4jGraphStore
from ..storage.postgres import PostgresMemoryStore
from ..utils.embeddings import OpenAIEmbeddings
from ..utils.llm import get_llm_client

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
        retriever: MemoryRetriever,
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
        llm_client = get_llm_client()
        embedding_client = OpenAIEmbeddings()

        episodic_store = PostgresMemoryStore(db_manager.pg_session)
        graph_store = Neo4jGraphStore(db_manager.neo4j_driver)
        fact_store = SemanticFactStore(db_manager.pg_session)
        neocortical = NeocorticalStore(graph_store, fact_store)

        short_term = ShortTermMemory(llm_client=llm_client)
        hippocampal = HippocampalStore(
            vector_store=episodic_store,
            embedding_client=embedding_client,
        )

        retriever = MemoryRetriever(
            hippocampal=hippocampal,
            neocortical=neocortical,
            llm_client=llm_client,
        )

        reconsolidation = ReconsolidationService(
            memory_store=episodic_store,
            llm_client=llm_client,
            fact_extractor=LLMFactExtractor(llm_client),
        )

        consolidation = ConsolidationWorker(
            episodic_store=episodic_store,
            neocortical_store=neocortical,
            llm_client=llm_client,
        )

        forgetting = ForgettingWorker(store=episodic_store)

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
        context_tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write new information to memory. Holistic: tenant-only."""
        stm_result = await self.short_term.ingest_turn(
            tenant_id=tenant_id,
            scope_id=session_id or tenant_id,
            text=content,
            turn_id=turn_id,
            role="user",
        )

        chunks_for_encoding = stm_result.get("chunks_for_encoding", [])

        if not chunks_for_encoding:
            return {
                "memory_id": None,
                "chunks_created": 0,
                "message": "No significant information to store",
            }

        stored = await self.hippocampal.encode_batch(
            tenant_id=tenant_id,
            chunks=chunks_for_encoding,
            context_tags=context_tags,
            source_session_id=session_id,
            agent_id=agent_id,
            namespace=namespace,
        )

        return {
            "memory_id": stored[0].id if stored else None,
            "chunks_created": len(stored),
            "message": f"Stored {len(stored)} memory chunks",
        }

    async def read(
        self,
        tenant_id: str,
        query: str,
        max_results: int = 10,
        context_filter: Optional[List[str]] = None,
        memory_types: Optional[List[Any]] = None,
        time_filter: Optional[Dict] = None,
    ) -> MemoryPacket:
        """Retrieve relevant memories. Holistic: tenant-only."""
        return await self.retriever.retrieve(
            tenant_id=tenant_id,
            query=query,
            max_results=max_results,
            context_filter=context_filter,
        )

    async def update(
        self,
        tenant_id: str,
        memory_id: UUID,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict] = None,
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing memory. Holistic: tenant-only."""
        record = await self.hippocampal.store.get_by_id(memory_id)
        if not record:
            raise ValueError(f"Memory {memory_id} not found")
        if record.tenant_id != tenant_id:
            raise ValueError("Memory does not belong to tenant")

        patch: Dict[str, Any] = {}
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
            patch["valid_to"] = datetime.utcnow()

        result = await self.hippocampal.store.update(memory_id, patch)
        return {
            "version": result.version if result else 1,
        }

    async def forget(
        self,
        tenant_id: str,
        memory_ids: Optional[List[UUID]] = None,
        query: Optional[str] = None,
        before: Optional[datetime] = None,
        action: str = "delete",
    ) -> Dict[str, Any]:
        """Forget memories. Holistic: tenant-only."""
        affected = 0
        hard = action == "delete"

        def owns(record) -> bool:
            return record.tenant_id == tenant_id

        if memory_ids:
            for mid in memory_ids:
                record = await self.hippocampal.store.get_by_id(mid)
                if record and owns(record):
                    await self.hippocampal.store.delete(mid, hard=hard)
                    affected += 1

        if query:
            packet = await self.retriever.retrieve(tenant_id, query=query, max_results=100)
            for mem in packet.all_memories:
                rid = mem.record.id
                record = await self.hippocampal.store.get_by_id(rid)
                if record and owns(record):
                    await self.hippocampal.store.delete(rid, hard=hard)
                    affected += 1

        if before:
            records = await self.hippocampal.store.scan(
                tenant_id, filters={"status": MemoryStatus.ACTIVE.value}, limit=500
            )
            for r in records:
                if r.timestamp and r.timestamp < before:
                    await self.hippocampal.store.delete(r.id, hard=hard)
                    affected += 1

        return {"affected_count": affected}

    async def get_session_context(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get full session context for LLM (messages, tool_results, scratch_pad, context_string). Holistic: tenant-only."""
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
        """Delete all memories for a tenant (GDPR). Holistic: tenant-only."""
        records = await self.hippocampal.store.scan(tenant_id, limit=10000)
        affected = 0
        for r in records:
            await self.hippocampal.store.delete(r.id, hard=True)
            affected += 1
        return affected

    async def get_stats(
        self,
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Get memory statistics for tenant. Holistic: tenant-only."""
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

        by_type: Dict[str, int] = {}
        records = await self.hippocampal.store.scan(tenant_id, limit=1000)
        for r in records:
            t = r.type.value if hasattr(r.type, "value") else str(r.type)
            by_type[t] = by_type.get(t, 0) + 1

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
