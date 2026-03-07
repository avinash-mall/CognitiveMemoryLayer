from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.dashboard import graph_routes

from .dashboard_support import ADMIN_AUTH, NeoResultStub, NeoSessionStub, make_db


@pytest.mark.asyncio
async def test_dashboard_graph_stats_returns_empty_without_driver() -> None:
    db, _ = make_db(neo_session=None)
    result = await graph_routes.dashboard_graph_stats(auth=ADMIN_AUTH, db=db)
    assert result.total_nodes == 0
    assert result.total_edges == 0
    assert result.tenants_with_graph == []


@pytest.mark.asyncio
async def test_dashboard_graph_stats_returns_counts_and_sorted_tenants() -> None:
    neo_session = NeoSessionStub(
        [
            NeoResultStub(single={"cnt": 4}),
            NeoResultStub(single={"cnt": 7}),
            NeoResultStub(items=[{"t": "person", "c": 3}, {"t": "city", "c": 1}]),
            NeoResultStub(items=[{"tid": "tenant-b"}, {"tid": "tenant-a"}]),
        ]
    )
    db, _ = make_db(neo_session=neo_session)

    result = await graph_routes.dashboard_graph_stats(auth=ADMIN_AUTH, db=db)

    assert result.total_nodes == 4
    assert result.total_edges == 7
    assert result.entity_types == {"person": 3, "city": 1}
    assert result.tenants_with_graph == ["tenant-a", "tenant-b"]


@pytest.mark.asyncio
async def test_dashboard_graph_stats_raises_http_500_on_error() -> None:
    neo_session = NeoSessionStub([RuntimeError("neo failed")])
    db, _ = make_db(neo_session=neo_session)

    with pytest.raises(HTTPException, match="neo failed"):
        await graph_routes.dashboard_graph_stats(auth=ADMIN_AUTH, db=db)


@pytest.mark.asyncio
async def test_dashboard_graph_overview_returns_empty_when_center_missing() -> None:
    neo_session = NeoSessionStub([NeoResultStub(single={"entity": None})])
    db, _ = make_db(neo_session=neo_session)

    result = await graph_routes.dashboard_graph_overview(
        tenant_id="tenant-a",
        scope_id=None,
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result.center_entity is None
    assert result.nodes == []
    assert result.edges == []


@pytest.mark.asyncio
async def test_dashboard_graph_overview_dedupes_and_strips_props() -> None:
    neo_session = NeoSessionStub(
        [
            NeoResultStub(single={"entity": "Alice"}),
            NeoResultStub(
                single={
                    "neighbors": [
                        {
                            "entity": "Paris",
                            "entity_type": "city",
                            "properties": {"tenant_id": "tenant-a", "scope_id": "tenant-a", "rank": 1},
                        },
                        {
                            "entity": "Paris",
                            "entity_type": "city",
                            "properties": {"rank": 1},
                        },
                    ],
                    "rels": [
                        {
                            "source": "Alice",
                            "target": "Paris",
                            "predicate": "LIVES_IN",
                            "confidence": 0.9,
                            "properties": {"created_at": "x", "updated_at": "y", "weight": 2},
                        },
                        {
                            "source": "Alice",
                            "target": "Paris",
                            "predicate": "LIVES_IN",
                            "confidence": 0.9,
                            "properties": {"weight": 2},
                        },
                    ],
                }
            ),
        ]
    )
    db, _ = make_db(neo_session=neo_session)

    result = await graph_routes.dashboard_graph_overview(
        tenant_id="tenant-a",
        scope_id=None,
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result.center_entity == "Alice"
    assert [node.entity for node in result.nodes] == ["Alice", "Paris"]
    assert result.nodes[1].properties == {"rank": 1}
    assert len(result.edges) == 1
    assert result.edges[0].properties == {"weight": 2}


@pytest.mark.asyncio
async def test_dashboard_graph_explore_respects_depth_cap_and_dedupes() -> None:
    neo_session = NeoSessionStub(
        [
            NeoResultStub(
                single={
                    "neighbors": [
                        {"entity": "Bob", "entity_type": "person", "properties": {"tenant_id": "tenant-a"}},
                        {"entity": "Bob", "entity_type": "person", "properties": {}},
                    ],
                    "rels": [
                        {
                            "source": "Alice",
                            "target": "Bob",
                            "predicate": "KNOWS",
                            "confidence": 0.7,
                            "properties": {"updated_at": "now"},
                        }
                    ],
                }
            )
        ]
    )
    db, _ = make_db(neo_session=neo_session)

    result = await graph_routes.dashboard_graph_explore(
        tenant_id="tenant-a",
        entity="Alice",
        scope_id=None,
        depth=9,
        auth=ADMIN_AUTH,
        db=db,
    )

    query = neo_session.calls[0][0][0]
    assert "*1..5" in query
    assert result.center_entity == "Alice"
    assert [node.entity for node in result.nodes] == ["Alice", "Bob"]
    assert result.edges[0].predicate == "KNOWS"


@pytest.mark.asyncio
async def test_dashboard_graph_search_supports_tenant_filtered_and_global() -> None:
    neo_session = NeoSessionStub(
        [
            NeoResultStub(items=[{"entity": "Alice", "entity_type": "person", "tid": "tenant-a", "sid": "scope-a"}]),
            NeoResultStub(items=[{"entity": "Global", "entity_type": "topic", "tid": "tenant-b", "sid": "scope-b"}]),
        ]
    )
    db, _ = make_db(neo_session=neo_session)

    tenant_result = await graph_routes.dashboard_graph_search(
        query="ali",
        tenant_id="tenant-a",
        limit=5,
        auth=ADMIN_AUTH,
        db=db,
    )
    global_result = await graph_routes.dashboard_graph_search(
        query="glo",
        tenant_id=None,
        limit=5,
        auth=ADMIN_AUTH,
        db=db,
    )

    tenant_query = neo_session.calls[0][0][0]
    global_query = neo_session.calls[1][0][0]
    assert "n.tenant_id = $tenant_id" in tenant_query
    assert "n.tenant_id = $tenant_id" not in global_query
    assert tenant_result.results[0].tenant_id == "tenant-a"
    assert global_result.results[0].entity == "Global"


@pytest.mark.asyncio
async def test_dashboard_graph_neo4j_config_falls_back_to_primary_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_routes,
        "get_settings",
        lambda: SimpleNamespace(
            database=SimpleNamespace(
                neo4j_browser_url="",
                neo4j_url="bolt://neo4j:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
            )
        ),
    )

    result = await graph_routes.dashboard_graph_neo4j_config(auth=ADMIN_AUTH)

    assert result.server_url == "bolt://neo4j:7687"
    assert result.server_user == "neo4j"
    assert result.server_password == "password"
