"""Neo4j knowledge graph store for semantic memory."""

import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ClientError as Neo4jClientError

from ..core.config import get_settings
from ..utils.logging_config import get_logger
from .base import GraphStoreBase

logger = get_logger(__name__)


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

    def __init__(self, driver: Any | None = None):
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
        properties: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Create or update a node. Uses MERGE to avoid duplicates."""
        properties = properties or {}
        if namespace is not None:
            properties["namespace"] = namespace
        now = datetime.now(UTC).isoformat()

        # Node identity is (tenant_id, scope_id, entity) — matching the unique
        # constraint and merge_edge(). entity_type is a mutable property, NOT part
        # of identity; including it here would fork nodes and violate the
        # constraint when an edge first created the node as 'UNKNOWN'.
        query = """
        MERGE (n:Entity {
            tenant_id: $tenant_id,
            scope_id: $scope_id,
            entity: $entity
        })
        ON CREATE SET
            n.created_at = $now,
            n.updated_at = $now,
            n += $properties,
            n.entity_type = $entity_type
        ON MATCH SET
            n.updated_at = $now,
            n += $properties,
            n.entity_type = coalesce($entity_type, n.entity_type)
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

    async def merge_nodes_batch(
        self,
        tenant_id: str,
        scope_id: str,
        nodes: list[dict[str, Any]],
    ) -> int:
        """Batch-merge entity nodes using a single UNWIND query."""
        if not nodes:
            return 0
        now = datetime.now(UTC).isoformat()
        params = [
            {
                "entity": n["entity"],
                "entity_type": n["entity_type"],
                "properties": n.get("properties") or {},
            }
            for n in nodes
        ]

        # Identity is (tenant_id, scope_id, entity) only — see merge_node().
        query = """
        UNWIND $batch AS item
        MERGE (n:Entity {
            tenant_id: $tenant_id,
            scope_id: $scope_id,
            entity: item.entity
        })
        ON CREATE SET
            n.created_at = $now,
            n.updated_at = $now,
            n += item.properties,
            n.entity_type = item.entity_type
        ON MATCH SET
            n.updated_at = $now,
            n += item.properties,
            n.entity_type = coalesce(item.entity_type, n.entity_type)
        RETURN count(n) AS cnt
        """

        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    tenant_id=tenant_id,
                    scope_id=scope_id,
                    batch=params,
                    now=now,
                )
                record = await result.single()
                return record["cnt"] if record else len(nodes)
        except Exception:
            count = 0
            for n in nodes:
                await self.merge_node(
                    tenant_id=tenant_id,
                    scope_id=scope_id,
                    entity=n["entity"],
                    entity_type=n["entity_type"],
                    properties=n.get("properties"),
                )
                count += 1
            return count

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
        """Create or update an edge between two nodes. Creates nodes if they don't exist."""
        target = object  # Avoid shadowing built-in 'object'
        properties = properties or {}
        if namespace is not None:
            properties["namespace"] = namespace
        rel_type = _sanitize_rel_type(predicate)
        confidence = properties.get("confidence", 0.8)
        now = datetime.now(UTC).isoformat()

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

        MERGE (s)-[r:`{rel_type}`]->(o)
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

    async def merge_edges_batch(
        self,
        tenant_id: str,
        scope_id: str,
        edges: list[dict[str, Any]],
    ) -> list[str]:
        """Batch-merge multiple edges in a single session.

        Edges are grouped by sanitised relationship type, and each group is
        written with a single UNWIND Cypher statement (Neo4j does not support
        dynamic relationship types inside UNWIND).
        """
        if not edges:
            return []

        from collections import defaultdict

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            rel_type = _sanitize_rel_type(edge.get("predicate", "RELATED_TO"))
            props = dict(edge.get("properties", {}) or {})
            groups[rel_type].append(
                {
                    "subject": edge["subject"],
                    "object": edge["object"],
                    "confidence": props.pop("confidence", 0.8),
                    "properties": props,
                }
            )

        now = datetime.now(UTC).isoformat()
        all_ids: list[str] = []

        async with self.driver.session() as session:
            for rel_type, batch in groups.items():
                query = f"""
                UNWIND $batch AS edge
                MERGE (s:Entity {{
                    tenant_id: $tenant_id,
                    scope_id: $scope_id,
                    entity: edge.subject
                }})
                ON CREATE SET s.created_at = $now, s.entity_type = 'UNKNOWN'

                MERGE (o:Entity {{
                    tenant_id: $tenant_id,
                    scope_id: $scope_id,
                    entity: edge.object
                }})
                ON CREATE SET o.created_at = $now, o.entity_type = 'UNKNOWN'

                MERGE (s)-[r:`{rel_type}`]->(o)
                ON CREATE SET
                    r.created_at = $now,
                    r.updated_at = $now,
                    r.confidence = edge.confidence,
                    r += edge.properties
                ON MATCH SET
                    r.updated_at = $now,
                    r.access_count = coalesce(r.access_count, 0) + 1,
                    r += edge.properties

                RETURN elementId(r) AS edge_id
                """
                result = await session.run(
                    query,
                    tenant_id=tenant_id,
                    scope_id=scope_id,
                    batch=batch,
                    now=now,
                )
                records = await result.data()
                all_ids.extend(r["edge_id"] for r in records if "edge_id" in r)

        return all_ids

    async def get_neighbors(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
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
        seed_entities: list[str],
        top_k: int = 20,
        damping: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Run Personalized PageRank from seed entities. Falls back to multi-hop if GDS unavailable.

        The fallback is what actually runs on any deployment without the GDS plugin —
        including this one — so treat it as the primary path, not a safety net. It is a
        path-count proximity score, not PageRank, which is where the unbounded scores
        (315, 744 observed) come from.

        Depth is 2, not 3, on measured evidence: on a real tenant, depth 3 reached 504
        entities against depth 2's 502 — two more — while counting roughly 63x as many
        paths (max raw score 117153 vs 1849) and taking 2.5x as long, which pushed the
        prong past its 2s step budget so it timed out and contributed nothing. The extra
        hop buys reachability that is already there. This also matches the retrieval
        literature, where hop depth contributes far less than cue quality and text
        grounding.
        """
        fallback_query = """
        MATCH (seed:Entity)
        WHERE seed.tenant_id = $tenant_id
          AND seed.scope_id = $scope_id
          AND seed.entity IN $seeds

        MATCH path = (seed)-[*1..2]-(related:Entity)
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

        # GDS 2.x removed anonymous projection: gds.pageRank.stream takes a graph *name*,
        # not a {nodeQuery, relationshipQuery} map. The 1.x form this used to send failed
        # with "Type mismatch: expected String but was Map" on every call, so the fallback
        # ran even where GDS was installed. The projection is therefore mandatory, and is
        # dropped again below — a leaked projection holds its nodes in heap for the life
        # of the database.
        # gds.graph.project.cypher warns as deprecated in 2.13 in favour of the
        # gds.graph.project aggregation form. Kept deliberately: the aggregation form
        # projects only nodes that appear in a relationship, so an isolated seed entity
        # would vanish from the graph and take its own PPR mass with it. Revisit when a
        # GDS version actually removes this, and handle isolated seeds explicitly then.
        graph_name = f"ppr-{uuid4().hex}"
        node_query = (
            "MATCH (n:Entity) WHERE n.tenant_id = $tenant_id AND n.scope_id = $scope_id "
            "RETURN id(n) AS id"
        )
        rel_query = (
            "MATCH (n1:Entity)-[r]->(n2:Entity) WHERE n1.tenant_id = $tenant_id "
            "AND n1.scope_id = $scope_id RETURN id(n1) AS source, id(n2) AS target"
        )

        async with self.driver.session() as session:
            try:
                await session.run(
                    "CALL gds.graph.project.cypher($graph_name, $node_query, $rel_query, "
                    "{parameters: {tenant_id: $tenant_id, scope_id: $scope_id}}) "
                    "YIELD graphName RETURN graphName",
                    graph_name=graph_name,
                    node_query=node_query,
                    rel_query=rel_query,
                    tenant_id=tenant_id,
                    scope_id=scope_id,
                )
                try:
                    result = await session.run(
                        """
                        MATCH (source:Entity)
                        WHERE source.tenant_id = $tenant_id
                          AND source.scope_id = $scope_id
                          AND source.entity IN $seeds
                        WITH collect(source) AS sources

                        CALL gds.pageRank.stream($graph_name, {
                            dampingFactor: $damping,
                            sourceNodes: sources
                        })
                        YIELD nodeId, score

                        MATCH (n:Entity) WHERE id(n) = nodeId
                        RETURN n.entity AS entity,
                               n.entity_type AS entity_type,
                               score,
                               properties(n) AS properties
                        ORDER BY score DESC
                        LIMIT $top_k
                        """,
                        graph_name=graph_name,
                        tenant_id=tenant_id,
                        scope_id=scope_id,
                        seeds=seed_entities,
                        damping=damping,
                        top_k=top_k,
                    )
                    records = await result.data()
                finally:
                    # Best-effort: a failed drop must not mask the result or the error
                    # that got us here, but it must still be attempted on both paths.
                    try:
                        await session.run(
                            "CALL gds.graph.drop($graph_name, false) YIELD graphName "
                            "RETURN graphName",
                            graph_name=graph_name,
                        )
                    except Exception:
                        logger.warning("gds_projection_drop_failed", extra={"graph": graph_name})
                return list(records) if records else []
            except Neo4jClientError:
                # GDS not installed, or the projection was refused. Either way the
                # fallback is a path-count proximity score, NOT PageRank — which is why
                # its scores are unbounded and its docstring used to lie.
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
    ) -> list[dict[str, Any]]:
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

    async def get_entity_facts_batch(
        self,
        tenant_id: str,
        scope_id: str,
        entity_names: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Get relations for multiple entities in one Cypher round-trip.

        Uses ``UNWIND`` to match all requested entities in a single query,
        then groups the results by entity name in Python.

        Returns a mapping ``entity_name -> list[relation_dict]`` where each
        relation dict has the same shape as :meth:`get_entity_facts` output
        (``predicate``, ``direction``, ``related_entity``, ``related_type``,
        ``relation_properties``).  Entities with no relations are absent
        from the dict; callers should use ``.get(name, [])``.
        """
        if not entity_names:
            return {}

        query = """
        UNWIND $entity_names AS entity_name
        MATCH (e:Entity {
            tenant_id: $tenant_id,
            scope_id: $scope_id,
            entity: entity_name
        })-[r]-(other:Entity)
        RETURN entity_name,
               type(r) AS predicate,
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
                entity_names=entity_names,
            )
            records = await result.data()

        # Group records by entity_name
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records or []:
            name = record.pop("entity_name", None)
            if name is not None:
                grouped.setdefault(name, []).append(record)
        return grouped

    async def search_by_pattern(
        self,
        tenant_id: str,
        scope_id: str,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        limit: int = 50,
    ) -> list[tuple[str, str, str, dict]]:
        """Search for triples matching a pattern. None values are wildcards."""
        target = object  # Avoid shadowing built-in 'object'
        conditions = ["s.tenant_id = $tenant_id", "s.scope_id = $scope_id"]
        params: dict[str, Any] = {"tenant_id": tenant_id, "scope_id": scope_id, "limit": limit}

        if subject:
            conditions.append("s.entity = $subject")
            params["subject"] = subject
        if target:
            conditions.append("o.entity = $target")
            params["target"] = target

        rel_pattern = "[r]" if not predicate else f"[r:`{_sanitize_rel_type(predicate)}`]"
        query = f"""
        MATCH (s:Entity)-{rel_pattern}->(o:Entity)
        WHERE {" AND ".join(conditions)}
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
    """Initialize Neo4j constraints and indexes.

    Tries to create a unique constraint (best-case for correctness).
    Falls back to a plain RANGE index if duplicates already exist.
    """
    async with store.driver.session() as session:
        # Try unique constraint first; fall back to plain index if duplicates exist.
        try:
            await session.run("""
                CREATE CONSTRAINT entity_unique IF NOT EXISTS
                FOR (n:Entity)
                REQUIRE (n.tenant_id, n.scope_id, n.entity) IS UNIQUE
            """)
        except Exception:
            await session.run("""
                CREATE INDEX entity_lookup IF NOT EXISTS
                FOR (n:Entity)
                ON (n.tenant_id, n.scope_id, n.entity)
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
