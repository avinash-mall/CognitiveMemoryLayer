"""Hippocampal store: episodic memory with write gate, embedding, and vector store."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...core.enums import MemoryStatus, MemorySource, MemoryType
from ...core.schemas import (
    EntityMention,
    MemoryRecord,
    MemoryRecordCreate,
    Provenance,
    Relation,
)
from ...extraction.entity_extractor import EntityExtractor
from ...extraction.relation_extractor import RelationExtractor
from ...storage.base import MemoryStoreBase
from ...utils.embeddings import EmbeddingClient
from ..working.models import SemanticChunk
from .redactor import PIIRedactor
from .write_gate import WriteDecision, WriteGate


class HippocampalStore:
    """
    Fast episodic memory store.
    Coordinates write gate, redaction, embedding, extraction, and vector store.
    """

    def __init__(
        self,
        vector_store: MemoryStoreBase,
        embedding_client: EmbeddingClient,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        write_gate: Optional[WriteGate] = None,
        redactor: Optional[PIIRedactor] = None,
    ) -> None:
        self.store = vector_store
        self.embeddings = embedding_client
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.write_gate = write_gate or WriteGate()
        self.redactor = redactor or PIIRedactor()

    async def encode_chunk(
        self,
        tenant_id: str,
        chunk: SemanticChunk,
        context_tags: Optional[List[str]] = None,
        source_session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        existing_memories: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        memory_type_override: Optional[MemoryType] = None,
    ) -> Optional[MemoryRecord]:
        gate_result = self.write_gate.evaluate(chunk, existing_memories=existing_memories)
        if gate_result.decision == WriteDecision.SKIP:
            return None

        text = chunk.text
        if gate_result.redaction_required:
            redaction_result = self.redactor.redact(text)
            text = redaction_result.redacted_text

        embedding_result = await self.embeddings.embed(text)

        entities: List[EntityMention] = []
        if self.entity_extractor:
            entities = await self.entity_extractor.extract(text)
        elif chunk.entities:
            entities = [
                EntityMention(text=e, normalized=e, entity_type="CONCEPT") for e in chunk.entities
            ]

        relations: List[Relation] = []
        if self.relation_extractor:
            entity_texts = [e.normalized for e in entities]
            relations = await self.relation_extractor.extract(text, entities=entity_texts)

        # Use caller-provided memory_type override, else fall back to gate classification
        memory_type = memory_type_override or (
            gate_result.memory_types[0] if gate_result.memory_types else MemoryType.EPISODIC_EVENT
        )
        key = self._generate_key(chunk, memory_type)

        # Merge request-level metadata with system metadata; request metadata wins on conflict
        system_metadata: Dict[str, Any] = {
            "chunk_type": chunk.chunk_type.value,
            "source_turn_id": chunk.source_turn_id,
            "source_role": chunk.source_role,
        }
        if request_metadata:
            merged_metadata = {**system_metadata, **request_metadata}
        else:
            merged_metadata = system_metadata

        record = MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=context_tags or [],
            source_session_id=source_session_id,
            agent_id=agent_id,
            namespace=namespace,
            type=memory_type,
            text=text,
            key=key,
            embedding=embedding_result.embedding,
            entities=entities,
            relations=relations,
            metadata=merged_metadata,
            timestamp=timestamp or chunk.timestamp,
            confidence=chunk.confidence,
            importance=gate_result.importance,
            provenance=Provenance(
                source=MemorySource.AGENT_INFERRED,
                evidence_refs=([chunk.source_turn_id] if chunk.source_turn_id else []),
                model_version=embedding_result.model,
            ),
        )
        stored = await self.store.upsert(record)
        return stored

    async def encode_batch(
        self,
        tenant_id: str,
        chunks: List[SemanticChunk],
        context_tags: Optional[List[str]] = None,
        source_session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        namespace: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        memory_type_override: Optional[MemoryType] = None,
    ) -> List[MemoryRecord]:
        existing = await self.store.scan(
            tenant_id,
            filters={"status": MemoryStatus.ACTIVE.value},
            limit=50,
            order_by="-timestamp",
        )
        existing_dicts = [{"text": m.text} for m in existing]
        results: List[MemoryRecord] = []
        for chunk in chunks:
            record = await self.encode_chunk(
                tenant_id,
                chunk,
                context_tags=context_tags,
                source_session_id=source_session_id,
                agent_id=agent_id,
                existing_memories=existing_dicts,
                namespace=namespace,
                timestamp=timestamp,
                request_metadata=request_metadata,
                memory_type_override=memory_type_override,
            )
            if record:
                results.append(record)
                existing_dicts.append({"text": record.text})
        return results

    async def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 10,
        context_filter: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        query_embedding = await self.embeddings.embed(query)
        results = await self.store.vector_search(
            tenant_id,
            embedding=query_embedding.embedding,
            top_k=top_k,
            context_filter=context_filter,
            filters=filters,
        )
        # Batch update access tracking: atomic increment to avoid lost update (BUG-02)
        now = datetime.now(timezone.utc)
        for record in results:
            record.access_count += 1
            record.last_accessed_at = now
        if results:
            if hasattr(self.store, "increment_access_counts"):
                await self.store.increment_access_counts([r.id for r in results], now)
            else:
                import asyncio

                await asyncio.gather(
                    *[
                        self.store.update(
                            record.id,
                            {
                                "access_count": record.access_count,
                                "last_accessed_at": now,
                            },
                            increment_version=False,
                        )
                        for record in results
                    ]
                )
        return results

    async def get_recent(
        self,
        tenant_id: str,
        limit: int = 20,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[MemoryRecord]:
        filters: Dict[str, Any] = {"status": MemoryStatus.ACTIVE.value}
        if memory_types:
            filters["type"] = [t.value for t in memory_types]
        return await self.store.scan(
            tenant_id,
            filters=filters,
            order_by="-timestamp",
            limit=limit,
        )

    def _generate_key(self, chunk: SemanticChunk, memory_type: MemoryType) -> Optional[str]:
        if memory_type not in (
            MemoryType.PREFERENCE,
            MemoryType.SEMANTIC_FACT,
        ):
            return None
        if chunk.entities:
            return f"{memory_type.value}:{chunk.entities[0].lower()}"
        return None
