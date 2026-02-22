"""Neocortical store: semantic memory facade (graph + fact store)."""

from typing import Any

from ...core.schemas import Relation
from ...storage.neo4j import Neo4jGraphStore
from .fact_store import SemanticFactStore
from .schemas import SemanticFact
from datetime import datetime


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
        key: str,
        value: Any,
        confidence: float = 0.8,
        evidence_ids: list[str] | None = None,
        context_tags: list[str] | None = None,
        valid_from: datetime | None = None,
    ) -> SemanticFact:
        """Store a semantic fact; optionally sync to graph if relation-like. Holistic: tenant-only."""
        fact = await self.facts.upsert_fact(
            tenant_id, key, value, confidence, evidence_ids, context_tags=context_tags, valid_from=valid_from
        )
        if ":" in key and isinstance(value, str):
            await self._sync_fact_to_graph(tenant_id, fact)
        return fact

    async def store_relation(
        self,
        tenant_id: str,
        relation: Relation,
        evidence_ids: list[str] | None = None,
    ) -> str:
        """Store a relation in the knowledge graph. Holistic: tenant as partition."""
        edge_id = await self.graph.merge_edge(
            tenant_id,
            tenant_id,  # scope_id = tenant_id for holistic (one partition per tenant)
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
        relations: list[Relation],
        evidence_ids: list[str] | None = None,
    ) -> list[str]:
        """Store multiple relations."""
        return [await self.store_relation(tenant_id, rel, evidence_ids) for rel in relations]

    async def get_fact(
        self,
        tenant_id: str,
        key: str,
    ) -> SemanticFact | None:
        """Get a fact by key. Holistic: tenant-only."""
        return await self.facts.get_fact(tenant_id, key)

    async def get_tenant_profile(self, tenant_id: str) -> dict[str, Any]:
        """Get structured profile for tenant. Holistic: tenant-only."""
        return await self.facts.get_tenant_profile(tenant_id)

    async def query_entity(
        self,
        tenant_id: str,
        entity: str,
    ) -> dict[str, Any]:
        """Get all information about an entity (graph + facts). Holistic: tenant-only."""
        graph_facts = await self.graph.get_entity_facts(tenant_id, tenant_id, entity)
        fact_results = await self.facts.search_facts(tenant_id, entity, limit=20)
        return {
            "entity": entity,
            "relations": graph_facts,
            "facts": [
                {"key": f.key, "value": f.value, "confidence": f.confidence} for f in fact_results
            ],
        }

    async def multi_hop_query(
        self,
        tenant_id: str,
        seed_entities: list[str],
        max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """Multi-hop reasoning from seed entities via Personalized PageRank.

        Uses single-query batching to avoid the N+1 pattern:
        one Cypher ``UNWIND`` call for all graph relations and one
        SQL ``subject IN (...)`` call for all semantic facts.

        Holistic: tenant-only.
        """
        related = await self.graph.personalized_pagerank(
            tenant_id, tenant_id, seed_entities=seed_entities, top_k=20
        )
        entity_names = [item["entity"] for item in related[:10]]
        if not entity_names:
            return []

        # Single Cypher query for all entities (replaces N individual calls)
        graph_by_entity = await self.graph.get_entity_facts_batch(
            tenant_id, tenant_id, entity_names
        )

        # Single SQL query for all entities (replaces N individual calls)
        facts_by_entity = await self.facts.search_facts_batch(
            tenant_id, entity_names, limit_per_entity=5
        )

        results: list[dict[str, Any]] = []
        for name in entity_names:
            score = next(
                (item.get("score", 0) for item in related if item["entity"] == name),
                0,
            )
            results.append(
                {
                    "entity": name,
                    "relations": graph_by_entity.get(name, []),
                    "facts": [
                        {"key": f.key, "value": f.value, "confidence": f.confidence}
                        for f in facts_by_entity.get(name, [])
                    ],
                    "relevance_score": score,
                }
            )
        return results

    async def find_schema_match(
        self,
        tenant_id: str,
        candidate_fact: str,
    ) -> SemanticFact | None:
        """Find if a candidate fact matches existing schema/knowledge. Holistic: tenant-only."""
        matches = await self.facts.search_facts(tenant_id, candidate_fact, limit=5)
        if matches:
            best = max(matches, key=lambda f: f.confidence)
            if best.confidence > 0.5:
                return best
        return None

    async def text_search(
        self,
        tenant_id: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search semantic memory by text. Holistic: tenant-only."""
        facts = await self.facts.search_facts(tenant_id, query, limit)
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

    async def _sync_fact_to_graph(self, tenant_id: str, fact: SemanticFact) -> None:
        """Sync a fact to the knowledge graph as a relation. Partition by tenant."""
        await self.graph.merge_edge(
            tenant_id,
            tenant_id,
            subject=fact.subject,
            predicate=fact.predicate,
            object=str(fact.value),
            properties={
                "confidence": fact.confidence,
                "fact_id": fact.id,
            },
        )
