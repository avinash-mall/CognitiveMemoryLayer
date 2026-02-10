"""Neo4j knowledge graph store for semantic memory."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ClientError as Neo4jClientError

from .base import GraphStoreBase
from ..core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Graph node (entity) representation."""

    id: str
    entity: str
    entity_type: str
    properties: Dict[str, Any]
    tenant_id: str
    scope_id: str
    created_at: datetime
    updated_at: datetime


@dataclass
class GraphEdge:
    """Graph edge (relation) representation."""

    id: str
    source_id: str
    target_id: str
    predicate: str
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime


_REL_TYPE_ALLOWLIST = re.compile(r"^[A-Za-z0-9_\s]+$")


def _sanitize_rel_type(predicate: str) -> str:
    """Sanitize predicate for Neo4j relationship type (SEC-01: strict allowlist, reject invalid)."""
    if not _REL_TYPE_ALLOWLIST.match(predicate):
        raise ValueError(
            "Invalid relationship type: only alphanumeric characters, underscores, and spaces allowed"
        )
    sanitized = "".join(
        c if c.isalnum() or c == "_" else "_" for c in predicate.upper().replace(" ", "_")
    )
    # Strip leading/trailing underscores and collapse runs
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    if not sanitized or not sanitized.replace("_", "").isalnum():
        return "RELATED_TO"
    return sanitized


def _validate_max_depth(max_depth: int) -> int:
    """Validate max_depth is a positive integer within safe bounds."""
    if not isinstance(max_depth, int) or max_depth < 1:
        return 1
    return min(max_depth, 10)  # Cap at 10 to prevent runaway traversals


class Neo4jGraphStore(GraphStoreBase):
    """
    Neo4j-based knowledge graph for semantic memory.
    Stores entities as nodes and relations as edges.

    Note: The scope_id parameter is used as the identity key in the graph.
    Graph data is partitioned by tenant_id and scope_id.
    """

    def __init__(self, driver: Optional[Any] = None):
        if driver is not None:
            self.driver = driver
        else:
            settings = get_settings()
            self.driver = AsyncGraphDatabase.driver(
                settings.database.neo4j_url,
                auth=(settings.database.neo4j_user, settings.database.neo4j_password),
            )

    async def close(self) -> None:
        await self.driver.close()

    async def merge_node(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """Create or update a node. Uses MERGE to avoid duplicates."""
        properties = properties or {}
        if namespace is not None:
            properties["namespace"] = namespace
        now = datetime.now(timezone.utc).isoformat()

        query = """
        MERGE (n:Entity {
            tenant_id: $tenant_id,
            scope_id: $scope_id,
            entity: $entity,
            entity_type: $entity_type
        })
        ON CREATE SET
            n.created_at = $now,
            n.updated_at = $now,
            n += $properties
        ON MATCH SET
            n.updated_at = $now,
            n += $properties
        RETURN elementId(n) AS node_id
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                tenant_id=tenant_id,
                scope_id=scope_id,
                entity=entity,
                entity_type=entity_type,
                properties=properties,
                now=now,
            )
            record = await result.single()
            return record["node_id"] if record else ""

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
        """Create or update an edge between two nodes. Creates nodes if they don't exist."""
        target = object  # Avoid shadowing built-in 'object'
        properties = properties or {}
        if namespace is not None:
            properties["namespace"] = namespace
        rel_type = _sanitize_rel_type(predicate)
        confidence = properties.get("confidence", 0.8)
        now = datetime.now(timezone.utc).isoformat()

        query = f"""
        MERGE (s:Entity {{
            tenant_id: $tenant_id,
            scope_id: $scope_id,
            entity: $subject
        }})
        ON CREATE SET s.created_at = $now, s.entity_type = 'UNKNOWN'

        MERGE (o:Entity {{
            tenant_id: $tenant_id,
            scope_id: $scope_id,
            entity: $target
        }})
        ON CREATE SET o.created_at = $now, o.entity_type = 'UNKNOWN'

        MERGE (s)-[r:{rel_type}]->(o)
        ON CREATE SET
            r.created_at = $now,
            r.updated_at = $now,
            r.confidence = $confidence,
            r += $properties
        ON MATCH SET
            r.updated_at = $now,
            r.access_count = coalesce(r.access_count, 0) + 1,
            r += $properties

        RETURN elementId(r) AS edge_id
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                tenant_id=tenant_id,
                scope_id=scope_id,
                subject=subject,
                target=target,
                properties=properties,
                confidence=confidence,
                now=now,
            )
            record = await result.single()
            return record["edge_id"] if record else ""

    async def get_neighbors(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes up to max_depth hops."""
        max_depth = _validate_max_depth(max_depth)
        fallback_query = f"""
        MATCH path = (start:Entity {{
            tenant_id: $tenant_id,
            scope_id: $scope_id,
            entity: $entity
        }})-[*1..{max_depth}]-(neighbor:Entity)
        WHERE neighbor.tenant_id = $tenant_id AND neighbor.scope_id = $scope_id
        RETURN DISTINCT neighbor.entity AS entity,
               neighbor.entity_type AS entity_type,
               properties(neighbor) AS properties
        LIMIT 100
        """

        async with self.driver.session() as session:
            try:
                apoc_query = """
                MATCH (start:Entity {
                    tenant_id: $tenant_id,
                    scope_id: $scope_id,
                    entity: $entity
                })
                CALL apoc.path.subgraphNodes(start, {
                    maxLevel: $max_depth,
                    relationshipFilter: null,
                    labelFilter: '+Entity'
                }) YIELD node
                WHERE node.tenant_id = $tenant_id AND node.scope_id = $scope_id
                RETURN node.entity AS entity,
                       node.entity_type AS entity_type,
                       properties(node) AS properties
                """
                result = await session.run(
                    apoc_query,
                    tenant_id=tenant_id,
                    scope_id=scope_id,
                    entity=entity,
                    max_depth=max_depth,
                )
            except Neo4jClientError:
                # APOC not available; fall back to plain Cypher traversal
                logger.debug("APOC unavailable, falling back to Cypher traversal")
                result = await session.run(
                    fallback_query,
                    tenant_id=tenant_id,
                    scope_id=scope_id,
                    entity=entity,
                )
            records = await result.data()
            return list(records) if records else []

    async def personalized_pagerank(
        self,
        tenant_id: str,
        scope_id: str,
        seed_entities: List[str],
        top_k: int = 20,
        damping: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """Run Personalized PageRank from seed entities. Falls back to multi-hop if GDS unavailable."""
        fallback_query = """
        MATCH (seed:Entity)
        WHERE seed.tenant_id = $tenant_id
          AND seed.scope_id = $scope_id
          AND seed.entity IN $seeds

        MATCH path = (seed)-[*1..3]-(related:Entity)
        WHERE related.tenant_id = $tenant_id AND related.scope_id = $scope_id

        WITH related,
             min(length(path)) AS min_distance,
             count(path) AS path_count

        RETURN related.entity AS entity,
               related.entity_type AS entity_type,
               1.0 / (min_distance + 1) * path_count AS score,
               properties(related) AS properties
        ORDER BY score DESC
        LIMIT $top_k
        """

        async with self.driver.session() as session:
            try:
                node_query = (
                    "MATCH (n:Entity) WHERE n.tenant_id = $tenant_id AND n.scope_id = $scope_id "
                    "RETURN id(n) AS id"
                )
                rel_query = (
                    "MATCH (n1:Entity)-[r]->(n2:Entity) WHERE n1.tenant_id = $tenant_id "
                    "AND n1.scope_id = $scope_id RETURN id(n1) AS source, id(n2) AS target"
                )
                gds_query = f"""
                MATCH (source:Entity)
                WHERE source.tenant_id = $tenant_id
                  AND source.scope_id = $scope_id
                  AND source.entity IN $seeds

                CALL gds.pageRank.stream({{
                    nodeQuery: '{node_query}',
                    relationshipQuery: '{rel_query}',
                    dampingFactor: $damping,
                    sourceNodes: collect(source)
                }})
                YIELD nodeId, score

                MATCH (n:Entity) WHERE id(n) = nodeId
                RETURN n.entity AS entity,
                       n.entity_type AS entity_type,
                       score,
                       properties(n) AS properties
                ORDER BY score DESC
                LIMIT $top_k
                """
                result = await session.run(
                    gds_query,
                    tenant_id=tenant_id,
                    scope_id=scope_id,
                    seeds=seed_entities,
                    damping=damping,
                    top_k=top_k,
                )
            except Neo4jClientError:
                # GDS not available; fall back to multi-hop heuristic
                logger.debug("GDS unavailable, falling back to multi-hop heuristic")
                result = await session.run(
                    fallback_query,
                    tenant_id=tenant_id,
                    scope_id=scope_id,
                    seeds=seed_entities,
                    top_k=top_k,
                )
            records = await result.data()
            return list(records) if records else []

    async def get_entity_facts(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
    ) -> List[Dict[str, Any]]:
        """Get all facts (relations) about an entity."""
        query = """
        MATCH (e:Entity {
            tenant_id: $tenant_id,
            scope_id: $scope_id,
            entity: $entity
        })-[r]-(other:Entity)
        RETURN type(r) AS predicate,
               CASE
                   WHEN startNode(r) = e THEN 'outgoing'
                   ELSE 'incoming'
               END AS direction,
               other.entity AS related_entity,
               other.entity_type AS related_type,
               properties(r) AS relation_properties
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                tenant_id=tenant_id,
                scope_id=scope_id,
                entity=entity,
            )
            records = await result.data()
            return list(records) if records else []

    async def search_by_pattern(
        self,
        tenant_id: str,
        scope_id: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        limit: int = 50,
    ) -> List[Tuple[str, str, str, Dict]]:
        """Search for triples matching a pattern. None values are wildcards."""
        target = object  # Avoid shadowing built-in 'object'
        conditions = ["s.tenant_id = $tenant_id", "s.scope_id = $scope_id"]
        params: Dict[str, Any] = {"tenant_id": tenant_id, "scope_id": scope_id, "limit": limit}

        if subject:
            conditions.append("s.entity = $subject")
            params["subject"] = subject
        if target:
            conditions.append("o.entity = $target")
            params["target"] = target

        rel_pattern = "[r]" if not predicate else f"[r:{_sanitize_rel_type(predicate)}]"
        query = f"""
        MATCH (s:Entity)-{rel_pattern}->(o:Entity)
        WHERE {' AND '.join(conditions)}
        RETURN s.entity AS subject,
               type(r) AS predicate,
               o.entity AS object,
               properties(r) AS properties
        LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [
                (r["subject"], r["predicate"], r["object"], r.get("properties") or {})
                for r in (records or [])
            ]

    async def delete_entity(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        cascade: bool = True,
    ) -> int:
        """Delete an entity node (and optionally its edges)."""
        if cascade:
            query = """
            MATCH (n:Entity {
                tenant_id: $tenant_id,
                scope_id: $scope_id,
                entity: $entity
            })
            DETACH DELETE n
            RETURN count(n) AS deleted_count
            """
        else:
            query = """
            MATCH (n:Entity {
                tenant_id: $tenant_id,
                scope_id: $scope_id,
                entity: $entity
            })
            DELETE n
            RETURN count(n) AS deleted_count
            """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                tenant_id=tenant_id,
                scope_id=scope_id,
                entity=entity,
            )
            record = await result.single()
            return record["deleted_count"] if record else 0


async def initialize_graph_schema(store: Neo4jGraphStore) -> None:
    """Initialize Neo4j constraints and indexes."""
    async with store.driver.session() as session:
        await session.run("""
            CREATE CONSTRAINT entity_unique IF NOT EXISTS
            FOR (n:Entity)
            REQUIRE (n.tenant_id, n.scope_id, n.entity) IS UNIQUE
        """)
        await session.run("""
            CREATE INDEX entity_type_idx IF NOT EXISTS
            FOR (n:Entity)
            ON (n.tenant_id, n.scope_id, n.entity_type)
        """)
        await session.run("""
            CREATE INDEX entity_time_idx IF NOT EXISTS
            FOR (n:Entity)
            ON (n.tenant_id, n.scope_id, n.updated_at)
        """)
