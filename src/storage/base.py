"""Abstract storage interfaces for memory and graph backends."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from ..core.schemas import MemoryRecord, MemoryRecordCreate


class MemoryStoreBase(ABC):
    """Abstract base for memory storage backends."""

    @abstractmethod
    async def upsert(self, record: MemoryRecordCreate) -> MemoryRecord:
        """Insert or update a memory record."""
        ...

    @abstractmethod
    async def get_by_id(self, record_id: UUID) -> MemoryRecord | None:
        """Get a single record by ID."""
        ...

    @abstractmethod
    async def get_by_key(
        self,
        tenant_id: str,
        key: str,
        context_filter: list[str] | None = None,
    ) -> MemoryRecord | None:
        """Get a record by its unique key (for facts/preferences). Holistic: tenant-only."""
        ...

    @abstractmethod
    async def delete(self, record_id: UUID, hard: bool = False) -> bool:
        """Delete (soft or hard) a record."""
        ...

    @abstractmethod
    async def update(
        self,
        record_id: UUID,
        patch: dict[str, Any],
        increment_version: bool = True,
    ) -> MemoryRecord | None:
        """Partial update with optimistic locking."""
        ...

    @abstractmethod
    async def vector_search(
        self,
        tenant_id: str,
        embedding: list[float],
        top_k: int = 10,
        context_filter: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[MemoryRecord]:
        """Search by vector similarity. Holistic: tenant-only, optional context_tags filter."""
        ...

    @abstractmethod
    async def scan(
        self,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """Scan records with filters. Holistic: tenant-only."""
        ...

    @abstractmethod
    async def count(
        self,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Count records matching filters. Holistic: tenant-only."""
        ...

    @abstractmethod
    async def delete_by_filter(
        self,
        tenant_id: str,
        filters: dict[str, Any],
    ) -> int:
        """Delete records matching filters. Holistic: tenant-only. Returns count of deleted records."""
        ...


class GraphStoreBase(ABC):
    """Abstract base for knowledge graph storage."""

    @abstractmethod
    async def merge_node(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Create or update a node, return node ID."""
        ...

    @abstractmethod
    async def merge_edge(
        self,
        tenant_id: str,
        scope_id: str,
        subject: str,
        predicate: str,
        object: str,
        properties: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Create or update an edge."""
        ...

    @abstractmethod
    async def get_neighbors(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes up to max_depth."""
        ...

    @abstractmethod
    async def personalized_pagerank(
        self,
        tenant_id: str,
        scope_id: str,
        seed_entities: list[str],
        top_k: int = 20,
        damping: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Run PPR from seed entities."""
        ...
