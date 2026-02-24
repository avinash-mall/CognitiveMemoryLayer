"""Dashboard knowledge graph routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ...core.config import get_settings
from ...storage.connection import DatabaseManager
from ..auth import AuthContext, require_admin_permission
from ._shared import (
    GraphEdgeInfo,
    GraphExploreResponse,
    GraphNeo4jConfigResponse,
    GraphNodeInfo,
    GraphSearchResponse,
    GraphSearchResult,
    GraphStatsResponse,
    _get_db,
    logger,
)

router = APIRouter()


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def dashboard_graph_stats(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Knowledge graph statistics from Neo4j: total nodes, edges, entity type distribution, and tenants with graph data."""
    if not db.neo4j_driver:
        return GraphStatsResponse()
    try:
        async with db.neo4j_session() as session:
            r1 = await session.run("MATCH (n:Entity) RETURN count(n) AS cnt")
            rec1 = await r1.single()
            total_nodes = rec1["cnt"] if rec1 else 0

            r2 = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            rec2 = await r2.single()
            total_edges = rec2["cnt"] if rec2 else 0

            r3 = await session.run(
                "MATCH (n:Entity) RETURN n.entity_type AS t, count(*) AS c ORDER BY c DESC LIMIT 20"
            )
            entity_types = {rec["t"]: rec["c"] async for rec in r3 if rec["t"]}

            r4 = await session.run("MATCH (n:Entity) RETURN DISTINCT n.tenant_id AS tid")
            tenants_with_graph = [rec["tid"] async for rec in r4 if rec["tid"]]

        return GraphStatsResponse(
            total_nodes=total_nodes,
            total_edges=total_edges,
            entity_types=entity_types,
            tenants_with_graph=sorted(tenants_with_graph),
        )
    except Exception as e:
        logger.error("dashboard_graph_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/overview", response_model=GraphExploreResponse)
async def dashboard_graph_overview(
    tenant_id: str = Query(..., description="Tenant ID"),
    scope_id: str | None = Query(None, description="Scope ID (defaults to tenant_id)"),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Sample subgraph for a tenant: finds the highest-degree entity and returns its 2-hop neighborhood. Used to auto-load the graph on page visit."""
    if not db.neo4j_driver:
        return GraphExploreResponse()
    scope_id = scope_id or tenant_id
    try:
        async with db.neo4j_session() as session:
            r0 = await session.run(
                """
                MATCH (n:Entity {tenant_id: $tenant_id, scope_id: $scope_id})
                WITH n, COUNT { (n)--() } AS deg
                ORDER BY deg DESC
                LIMIT 1
                RETURN n.entity AS entity
                """,
                tenant_id=tenant_id,
                scope_id=scope_id,
            )
            rec0 = await r0.single()
            if not rec0 or not rec0["entity"]:
                return GraphExploreResponse()

            entity = rec0["entity"]
            depth = 2

            query = f"""
            MATCH path = (start:Entity {{
                tenant_id: $tenant_id, scope_id: $scope_id, entity: $entity
            }})-[*1..{min(depth, 5)}]-(neighbor:Entity)
            WHERE neighbor.tenant_id = $tenant_id AND neighbor.scope_id = $scope_id
            UNWIND relationships(path) AS rel
            WITH DISTINCT neighbor, rel, startNode(rel) AS sn, endNode(rel) AS en
            RETURN
                collect(DISTINCT {{
                    entity: neighbor.entity,
                    entity_type: neighbor.entity_type,
                    properties: properties(neighbor)
                }}) AS neighbors,
                collect(DISTINCT {{
                    source: sn.entity,
                    target: en.entity,
                    predicate: type(rel),
                    confidence: coalesce(rel.confidence, 0),
                    properties: properties(rel)
                }}) AS rels
            """
            result = await session.run(query, tenant_id=tenant_id, scope_id=scope_id, entity=entity)
            record = await result.single()

            nodes: list[GraphNodeInfo] = [
                GraphNodeInfo(id=entity, entity=entity, entity_type="center")
            ]
            edges: list[GraphEdgeInfo] = []
            seen_nodes = {entity}

            if record:
                for n in record["neighbors"] or []:
                    ent = n.get("entity", "")
                    if ent and ent not in seen_nodes:
                        seen_nodes.add(ent)
                        props = dict(n.get("properties") or {})
                        props.pop("tenant_id", None)
                        props.pop("scope_id", None)
                        nodes.append(
                            GraphNodeInfo(
                                id=ent,
                                entity=ent,
                                entity_type=n.get("entity_type", "unknown"),
                                properties=props,
                            )
                        )
                seen_edges = set()
                for r in record["rels"] or []:
                    src = r.get("source", "")
                    tgt = r.get("target", "")
                    pred = r.get("predicate", "RELATED_TO")
                    edge_key = f"{src}-{pred}-{tgt}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        props = dict(r.get("properties") or {})
                        props.pop("created_at", None)
                        props.pop("updated_at", None)
                        edges.append(
                            GraphEdgeInfo(
                                source=src,
                                target=tgt,
                                predicate=pred,
                                confidence=float(r.get("confidence", 0)),
                                properties=props,
                            )
                        )

        return GraphExploreResponse(nodes=nodes, edges=edges, center_entity=entity)
    except Exception as e:
        logger.error("dashboard_graph_overview_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/explore", response_model=GraphExploreResponse)
async def dashboard_graph_explore(
    tenant_id: str = Query(..., description="Tenant ID"),
    entity: str = Query(..., description="Center entity name"),
    scope_id: str | None = Query(None, description="Scope ID (defaults to tenant_id)"),
    depth: int = Query(2, ge=1, le=5),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Explore the neighborhood of a specific entity in the knowledge graph. Specify tenant, entity name, scope, and depth (1-5). Returns nodes and edges for visualization."""
    if not db.neo4j_driver:
        return GraphExploreResponse(center_entity=entity)
    scope_id = scope_id or tenant_id
    try:
        async with db.neo4j_session() as session:
            query = f"""
            MATCH path = (start:Entity {{
                tenant_id: $tenant_id, scope_id: $scope_id, entity: $entity
            }})-[*1..{min(depth, 5)}]-(neighbor:Entity)
            WHERE neighbor.tenant_id = $tenant_id AND neighbor.scope_id = $scope_id
            UNWIND relationships(path) AS rel
            WITH DISTINCT neighbor, rel, startNode(rel) AS sn, endNode(rel) AS en
            RETURN
                collect(DISTINCT {{
                    entity: neighbor.entity,
                    entity_type: neighbor.entity_type,
                    properties: properties(neighbor)
                }}) AS neighbors,
                collect(DISTINCT {{
                    source: sn.entity,
                    target: en.entity,
                    predicate: type(rel),
                    confidence: coalesce(rel.confidence, 0),
                    properties: properties(rel)
                }}) AS rels
            """
            result = await session.run(query, tenant_id=tenant_id, scope_id=scope_id, entity=entity)
            record = await result.single()

            nodes: list[GraphNodeInfo] = [
                GraphNodeInfo(id=entity, entity=entity, entity_type="center")
            ]
            edges: list[GraphEdgeInfo] = []
            seen_nodes = {entity}

            if record:
                for n in record["neighbors"] or []:
                    ent = n.get("entity", "")
                    if ent and ent not in seen_nodes:
                        seen_nodes.add(ent)
                        props = dict(n.get("properties") or {})
                        props.pop("tenant_id", None)
                        props.pop("scope_id", None)
                        nodes.append(
                            GraphNodeInfo(
                                id=ent,
                                entity=ent,
                                entity_type=n.get("entity_type", "unknown"),
                                properties=props,
                            )
                        )
                seen_edges = set()
                for r in record["rels"] or []:
                    src = r.get("source", "")
                    tgt = r.get("target", "")
                    pred = r.get("predicate", "RELATED_TO")
                    edge_key = f"{src}-{pred}-{tgt}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        props = dict(r.get("properties") or {})
                        props.pop("created_at", None)
                        props.pop("updated_at", None)
                        edges.append(
                            GraphEdgeInfo(
                                source=src,
                                target=tgt,
                                predicate=pred,
                                confidence=float(r.get("confidence", 0)),
                                properties=props,
                            )
                        )

        return GraphExploreResponse(nodes=nodes, edges=edges, center_entity=entity)
    except Exception as e:
        logger.error("dashboard_graph_explore_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/search", response_model=GraphSearchResponse)
async def dashboard_graph_search(
    query: str = Query(..., min_length=1, description="Entity name pattern"),
    tenant_id: str | None = Query(None),
    limit: int = Query(25, ge=1, le=100),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Search entities in the knowledge graph by name pattern (case-insensitive contains). Returns matching entities with type and tenant."""
    if not db.neo4j_driver:
        return GraphSearchResponse()
    try:
        async with db.neo4j_session() as session:
            cypher = """
            MATCH (n:Entity)
            WHERE toLower(n.entity) CONTAINS toLower($pattern)
            """
            params: dict[str, Any] = {"pattern": query, "lim": limit}
            if tenant_id:
                cypher += " AND n.tenant_id = $tenant_id"
                params["tenant_id"] = tenant_id
            cypher += " RETURN n.entity AS entity, n.entity_type AS entity_type, n.tenant_id AS tid, n.scope_id AS sid LIMIT $lim"
            result = await session.run(cypher, **params)
            results = [
                GraphSearchResult(
                    entity=rec["entity"],
                    entity_type=rec["entity_type"] or "",
                    tenant_id=rec["tid"] or "",
                    scope_id=rec["sid"] or "",
                )
                async for rec in result
            ]
        return GraphSearchResponse(results=results)
    except Exception as e:
        logger.error("dashboard_graph_search_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/neo4j-config", response_model=GraphNeo4jConfigResponse)
async def dashboard_graph_neo4j_config(
    auth: AuthContext = Depends(require_admin_permission),
):
    """Return Neo4j connection config for browser (neovis.js). Admin-only. Use DATABASE__NEO4J_BROWSER_URL when Neo4j is not reachable at DATABASE__NEO4J_URL from the browser (e.g. Docker: bolt://localhost:7687)."""
    settings = get_settings()
    db = settings.database
    server_url = db.neo4j_browser_url or db.neo4j_url
    return GraphNeo4jConfigResponse(
        server_url=server_url,
        server_user=db.neo4j_user,
        server_password=db.neo4j_password or "",
    )
