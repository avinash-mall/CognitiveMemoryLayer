"""No-op storage implementations for embedded lite mode (no Neo4j/Postgres facts)."""

from datetime import datetime
from typing import Any

from ..memory.neocortical.schemas import FactCategory, SemanticFact
from .base import GraphStoreBase


class NoOpGraphStore(GraphStoreBase):
    """Graph store that does nothing; used when Neo4j is not available (lite mode)."""

    async def merge_node(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> str:
        return "noop-node"

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
        return "noop-edge"

    async def get_neighbors(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        return []

    async def personalized_pagerank(
        self,
        tenant_id: str,
        scope_id: str,
        seed_entities: list[str],
        top_k: int = 20,
        damping: float = 0.85,
    ) -> list[dict[str, Any]]:
        return []

    async def get_entity_facts(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
    ) -> list[dict[str, Any]]:
        """Neo4jGraphStore also has this method; return empty for no-op."""
        return []

    async def get_entity_facts_batch(
        self,
        tenant_id: str,
        scope_id: str,
        entity_names: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Batch variant; return empty dict for no-op."""
        return {}


class NoOpFactStore:
    """Fact store that does nothing; used when semantic facts DB is not available (lite mode)."""

    async def upsert_fact(
        self,
        tenant_id: str,
        key: str,
        value: Any,
        confidence: float = 0.8,
        evidence_ids: list[str] | None = None,
        valid_from: datetime | None = None,
        context_tags: list[str] | None = None,
    ) -> SemanticFact:
        parts = key.split(":")
        subject = parts[0] if len(parts) > 0 else "unknown"
        predicate = parts[-1] if len(parts) > 1 else key
        return SemanticFact(
            id="noop-fact",
            tenant_id=tenant_id,
            category=FactCategory.CUSTOM,
            key=key,
            subject=subject,
            predicate=predicate,
            value=value,
            value_type="string",
            context_tags=context_tags or [],
            confidence=confidence,
        )

    async def get_fact(
        self,
        tenant_id: str,
        key: str,
        include_historical: bool = False,
    ) -> SemanticFact | None:
        return None

    async def get_facts_by_category(
        self,
        tenant_id: str,
        category: FactCategory,
        current_only: bool = True,
        limit: int = 50,
    ) -> list[SemanticFact]:
        return []

    async def get_tenant_profile(self, tenant_id: str) -> dict[str, Any]:
        return {}

    async def search_facts(
        self,
        tenant_id: str,
        query: str,
        limit: int = 20,
    ) -> list[SemanticFact]:
        return []

    async def search_facts_batch(
        self,
        tenant_id: str,
        entity_names: list[str],
        limit_per_entity: int = 5,
    ) -> dict[str, list[SemanticFact]]:
        """Batch variant; return empty dict for no-op."""
        return {}
