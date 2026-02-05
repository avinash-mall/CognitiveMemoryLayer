"""General-purpose knowledge storage (not user-specific)."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.enums import MemoryScope, MemorySource, MemoryType
from ..core.schemas import MemoryRecordCreate, Provenance
from ..storage.base import MemoryStoreBase
from ..utils.embeddings import EmbeddingClient


@dataclass
class Fact:
    """A single knowledge fact."""

    id: Any
    subject: str
    predicate: str
    object: str
    source: Optional[str]
    confidence: float
    text: str


class KnowledgeBase:
    """
    General-purpose knowledge storage (not user-specific).
    Facts are scoped by namespace; supports semantic query via embeddings.
    """

    def __init__(
        self,
        store: MemoryStoreBase,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> None:
        self.store = store
        self.embeddings = embedding_client

    async def store_fact(
        self,
        tenant_id: str,
        namespace: str,
        subject: str,
        predicate: str,
        object: str,
        source: Optional[str] = None,
        confidence: float = 0.8,
    ) -> Any:
        """Store a general fact in the given namespace. Returns record id."""
        text = f"{subject} {predicate} {object}"
        key = f"kb:{namespace}:{subject}:{predicate}"
        meta: Dict[str, Any] = {"subject": subject, "predicate": predicate, "object": object}
        if source:
            meta["source"] = source
        record = MemoryRecordCreate(
            tenant_id=tenant_id,
            scope=MemoryScope.NAMESPACE,
            scope_id=namespace,
            user_id=None,
            namespace=namespace,
            type=MemoryType.KNOWLEDGE,
            text=text,
            key=key,
            embedding=None,
            metadata=meta,
            confidence=confidence,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
        )
        if self.embeddings:
            emb = await self.embeddings.embed(text)
            record.embedding = emb.embedding
        created = await self.store.upsert(record)
        return created.id

    async def query(
        self,
        tenant_id: str,
        namespace: str,
        query: str,
        max_results: int = 10,
    ) -> List[Fact]:
        """Query the knowledge base by semantic similarity (requires embedding_client)."""
        if not self.embeddings:
            records = await self.store.scan(
                tenant_id=tenant_id,
                user_id=namespace,
                filters={"status": "active", "type": MemoryType.KNOWLEDGE.value},
                limit=max_results,
            )
            return [
                Fact(
                    id=r.id,
                    subject=(r.metadata or {}).get("subject", ""),
                    predicate=(r.metadata or {}).get("predicate", ""),
                    object=(r.metadata or {}).get("object", ""),
                    source=(r.metadata or {}).get("source"),
                    confidence=r.confidence,
                    text=r.text,
                )
                for r in records
            ]
        emb_result = await self.embeddings.embed(query)
        records = await self.store.vector_search(
            tenant_id=tenant_id,
            user_id=namespace,
            embedding=emb_result.embedding,
            top_k=max_results,
            filters={"type": MemoryType.KNOWLEDGE.value},
        )
        return [
            Fact(
                id=r.id,
                subject=(r.metadata or {}).get("subject", ""),
                predicate=(r.metadata or {}).get("predicate", ""),
                object=(r.metadata or {}).get("object", ""),
                source=(r.metadata or {}).get("source"),
                confidence=r.confidence,
                text=r.text,
            )
            for r in records
        ]
