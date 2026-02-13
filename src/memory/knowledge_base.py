"""General-purpose knowledge storage (not user-specific)."""

from dataclasses import dataclass
from typing import Any

from ..core.enums import MemorySource, MemoryType
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
    source: str | None
    confidence: float
    text: str


class KnowledgeBase:
    """
    General-purpose knowledge storage (not user-specific).
    Facts are tagged by namespace; supports semantic query via embeddings.
    Holistic: tenant-only.
    """

    def __init__(
        self,
        store: MemoryStoreBase,
        embedding_client: EmbeddingClient | None = None,
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
        source: str | None = None,
        confidence: float = 0.8,
    ) -> Any:
        """Store a general fact in the given namespace. Returns record id."""
        text = f"{subject} {predicate} {object}"
        key = f"kb:{namespace}:{subject}:{predicate}"
        meta: dict[str, Any] = {"subject": subject, "predicate": predicate, "object": object}
        if source:
            meta["source"] = source
        record = MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=["world", "knowledge", f"namespace:{namespace}"],
            source_session_id=None,
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
    ) -> list[Fact]:
        """Query the knowledge base by semantic similarity (requires embedding_client). Holistic: filter by context_tags."""
        ns_tag = f"namespace:{namespace}"
        if not self.embeddings:
            records = await self.store.scan(
                tenant_id=tenant_id,
                filters={
                    "status": "active",
                    "type": MemoryType.KNOWLEDGE.value,
                    "context_tags": [ns_tag],
                },
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
            embedding=emb_result.embedding,
            top_k=max_results,
            context_filter=[ns_tag],
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
