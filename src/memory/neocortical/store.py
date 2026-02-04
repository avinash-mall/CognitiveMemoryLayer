"""Neocortical store: semantic memory facade (graph + fact store)."""
from typing import Any, Dict, List, Optional

from ...core.schemas import Relation
from ...storage.neo4j import Neo4jGraphStore
from .fact_store import SemanticFactStore
from .schemas import SemanticFact


class NeocorticalStore:
    """
    Semantic memory store combining knowledge graph and structured fact store.
    """

    def __init__(self, graph_store: Neo4jGraphStore, fact_store: SemanticFactStore):
        self.graph = graph_store
        self.facts = fact_store

    async def store_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        value: Any,
        confidence: float = 0.8,
        evidence_ids: Optional[List[str]] = None,
    ) -> SemanticFact:
        """Store a semantic fact; optionally sync to graph if relation-like."""
        fact = await self.facts.upsert_fact(
            tenant_id, user_id, key, value, confidence, evidence_ids
        )
        if ":" in key and isinstance(value, str):
            await self._sync_fact_to_graph(tenant_id, user_id, fact)
        return fact

    async def store_relation(
        self,
        tenant_id: str,
        user_id: str,
        relation: Relation,
        evidence_ids: Optional[List[str]] = None,
    ) -> str:
        """Store a relation in the knowledge graph."""
        edge_id = await self.graph.merge_edge(
            tenant_id,
            user_id,
            subject=relation.subject,
            predicate=relation.predicate,
            object=relation.object,
            properties={
                "confidence": relation.confidence,
                "evidence_ids": evidence_ids or [],
            },
        )
        return edge_id

    async def store_relations_batch(
        self,
        tenant_id: str,
        user_id: str,
        relations: List[Relation],
        evidence_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Store multiple relations."""
        return [
            await self.store_relation(tenant_id, user_id, rel, evidence_ids)
            for rel in relations
        ]

    async def get_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
    ) -> Optional[SemanticFact]:
        """Get a fact by key."""
        return await self.facts.get_fact(tenant_id, user_id, key)

    async def get_user_profile(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """Get structured user profile."""
        return await self.facts.get_user_profile(tenant_id, user_id)

    async def query_entity(
        self,
        tenant_id: str,
        user_id: str,
        entity: str,
    ) -> Dict[str, Any]:
        """Get all information about an entity (graph + facts)."""
        graph_facts = await self.graph.get_entity_facts(tenant_id, user_id, entity)
        fact_results = await self.facts.search_facts(tenant_id, user_id, entity, limit=20)
        return {
            "entity": entity,
            "relations": graph_facts,
            "facts": [
                {"key": f.key, "value": f.value, "confidence": f.confidence}
                for f in fact_results
            ],
        }

    async def multi_hop_query(
        self,
        tenant_id: str,
        user_id: str,
        seed_entities: List[str],
        max_hops: int = 3,
    ) -> List[Dict[str, Any]]:
        """Multi-hop reasoning from seed entities via Personalized PageRank."""
        related = await self.graph.personalized_pagerank(
            tenant_id, user_id, seed_entities=seed_entities, top_k=20
        )
        results = []
        for item in related[:10]:
            entity_info = await self.query_entity(tenant_id, user_id, item["entity"])
            entity_info["relevance_score"] = item.get("score", 0)
            results.append(entity_info)
        return results

    async def find_schema_match(
        self,
        tenant_id: str,
        user_id: str,
        candidate_fact: str,
    ) -> Optional[SemanticFact]:
        """Find if a candidate fact matches existing schema/knowledge."""
        matches = await self.facts.search_facts(
            tenant_id, user_id, candidate_fact, limit=5
        )
        if matches:
            best = max(matches, key=lambda f: f.confidence)
            if best.confidence > 0.5:
                return best
        return None

    async def text_search(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search semantic memory by text."""
        facts = await self.facts.search_facts(tenant_id, user_id, query, limit)
        return [
            {
                "type": "fact",
                "key": f.key,
                "value": f.value,
                "confidence": f.confidence,
                "category": f.category.value,
            }
            for f in facts
        ]

    async def _sync_fact_to_graph(
        self,
        tenant_id: str,
        user_id: str,
        fact: SemanticFact,
    ) -> None:
        """Sync a fact to the knowledge graph as a relation."""
        await self.graph.merge_edge(
            tenant_id,
            user_id,
            subject=fact.subject,
            predicate=fact.predicate,
            object=str(fact.value),
            properties={
                "confidence": fact.confidence,
                "fact_id": fact.id,
            },
        )
