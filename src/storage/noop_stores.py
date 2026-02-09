"""No-op storage implementations for embedded lite mode (no Neo4j/Postgres facts)."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import GraphStoreBase
from ..memory.neocortical.schemas import FactCategory, SemanticFact


class NoOpGraphStore(GraphStoreBase):
    """Graph store that does nothing; used when Neo4j is not available (lite mode)."""

    async def merge_node(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> str:
        return "noop-node"

    async def merge_edge(
        self,
        tenant_id: str,
        scope_id: str,
        subject: str,
        predicate: str,
        object: str,
        properties: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> str:
        return "noop-edge"

    async def get_neighbors(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        return []

    async def personalized_pagerank(
        self,
        tenant_id: str,
        scope_id: str,
        seed_entities: List[str],
        top_k: int = 20,
        damping: float = 0.85,
    ) -> List[Dict[str, Any]]:
        return []

    async def get_entity_facts(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
    ) -> List[Dict[str, Any]]:
        """Neo4jGraphStore also has this method; return empty for no-op."""
        return []


class NoOpFactStore:
    """Fact store that does nothing; used when semantic facts DB is not available (lite mode)."""

    async def upsert_fact(
        self,
        tenant_id: str,
        key: str,
        value: Any,
        confidence: float = 0.8,
        evidence_ids: Optional[List[str]] = None,
        valid_from: Optional[datetime] = None,
        context_tags: Optional[List[str]] = None,
    ) -> SemanticFact:
        now = datetime.now(timezone.utc)
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
    ) -> Optional[SemanticFact]:
        return None

    async def get_facts_by_category(
        self,
        tenant_id: str,
        category: FactCategory,
        current_only: bool = True,
    ) -> List[SemanticFact]:
        return []

    async def get_tenant_profile(self, tenant_id: str) -> Dict[str, Any]:
        return {}

    async def search_facts(
        self,
        tenant_id: str,
        query: str,
        limit: int = 20,
    ) -> List[SemanticFact]:
        return []
