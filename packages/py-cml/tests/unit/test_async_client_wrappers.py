"""Additional async client coverage for dashboard/admin wrappers and stream helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from cml import AsyncCognitiveMemoryLayer
from cml.async_client import AsyncNamespacedClient, AsyncSessionScope
from cml.config import CMLConfig
from cml.exceptions import ConnectionError, TimeoutError
from cml.models import MemoryType, TurnResponse, WriteResponse


async def _yield_items(items: list[dict[str, object]]) -> AsyncIterator[dict[str, object]]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_async_client_read_safe_and_stream_helpers(
    cml_config: CMLConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncCognitiveMemoryLayer(config=cml_config)

    async def _read_connection_error(*args: object, **kwargs: object):
        raise ConnectionError("down")

    monkeypatch.setattr(client, "read", _read_connection_error)
    empty = await client.read_safe("query")
    assert empty.total_count == 0

    async def _read_timeout(*args: object, **kwargs: object):
        raise TimeoutError("slow")

    monkeypatch.setattr(client, "read", _read_timeout)
    empty = await client.read_safe("query")
    assert empty.total_count == 0

    item_id = str(uuid4())

    async def _stream(path: str, *, json_body: dict[str, object] | None = None):
        assert path == "/memory/read/stream"
        assert json_body is not None
        async for item in _yield_items(
            [
                {
                    "id": item_id,
                    "text": "streamed",
                    "type": "semantic_fact",
                    "confidence": 0.9,
                    "relevance": 0.8,
                    "timestamp": "2025-01-01T00:00:00Z",
                    "metadata": {},
                }
            ]
        ):
            yield item

    client._transport.stream_sse = _stream  # type: ignore[method-assign]
    items = [item async for item in client.read_stream("stream query")]
    assert len(items) == 1
    assert str(items[0].id) == item_id


@pytest.mark.asyncio
async def test_async_client_dashboard_and_admin_wrappers_build_expected_requests(
    cml_config: CMLConfig,
) -> None:
    client = AsyncCognitiveMemoryLayer(config=cml_config)
    memory_id = uuid4()
    calls: list[tuple[str, str, dict[str, object]]] = []

    async def _request(method: str, path: str, **kwargs: object) -> dict[str, object]:
        calls.append((method, path, dict(kwargs)))
        if path == "/memory/stats":
            return {
                "total_memories": 3,
                "active_memories": 2,
                "silent_memories": 1,
                "archived_memories": 0,
                "by_type": {"semantic_fact": 3},
                "avg_confidence": 0.8,
                "avg_importance": 0.7,
                "oldest_memory": None,
                "newest_memory": None,
                "estimated_size_mb": 0.1,
            }
        if path == "/session/create":
            return {
                "session_id": "sess-1",
                "created_at": "2025-01-01T00:00:00Z",
                "expires_at": "2025-01-02T00:00:00Z",
            }
        if path == "/session/sess-1/context":
            return {"session_id": "sess-1", "messages": [], "tool_results": [], "scratch_pad": []}
        if path == "/memory/read":
            return {
                "query": "food",
                "memories": [],
                "facts": [],
                "preferences": [],
                "episodes": [],
                "constraints": [],
                "llm_context": "ctx",
                "total_count": 0,
                "elapsed_ms": 1.0,
            }
        if path == "/memory/all":
            return {"affected_count": 0}
        if path == "/dashboard/tenants":
            return {
                "tenants": [
                    {"tenant_id": "tenant-a", "memory_count": 1, "fact_count": 0, "event_count": 0}
                ]
            }
        if path == "/dashboard/overview":
            return {}
        if path == "/dashboard/memories":
            return {
                "items": [
                    {
                        "id": str(memory_id),
                        "tenant_id": "tenant-a",
                        "type": "semantic_fact",
                        "status": "active",
                        "text": "hello",
                        "confidence": 0.8,
                        "importance": 0.5,
                        "timestamp": "2025-01-01T00:00:00Z",
                    }
                ],
                "total": 1,
                "page": 1,
                "per_page": 25,
                "total_pages": 1,
            }
        if path == f"/dashboard/memories/{memory_id}":
            return {
                "id": str(memory_id),
                "tenant_id": "tenant-a",
                "type": "semantic_fact",
                "status": "active",
                "text": "hello",
                "context_tags": [],
                "confidence": 0.8,
                "importance": 0.5,
                "access_count": 0,
                "decay_rate": 0.01,
                "labile": False,
                "version": 1,
                "related_events": [],
            }
        if path == "/dashboard/facts":
            return {"items": [], "total": 0}
        if path.startswith("/dashboard/facts/"):
            return {"success": True}
        if path == "/dashboard/export/memories":
            return [{"id": str(memory_id), "text": "hello"}]
        if path == "/dashboard/events":
            return {"items": [], "total": 0, "page": 1, "per_page": 50, "total_pages": 1}
        if path == "/dashboard/timeline":
            return {"points": [], "total": 0}
        if path == "/dashboard/components":
            return {"components": [{"name": "PostgreSQL", "status": "ok"}]}
        if path == "/dashboard/sessions":
            return {"sessions": [], "total_active": 0, "total_memories_with_session": 0}
        if path == "/dashboard/ratelimits":
            return {"entries": [], "configured_rpm": 60}
        if path == "/dashboard/request-stats":
            return {"points": [], "total_last_24h": 0}
        if path == "/dashboard/graph/stats":
            return {
                "total_nodes": 1,
                "total_edges": 0,
                "entity_types": {},
                "tenants_with_graph": [],
            }
        if path == "/dashboard/graph/overview":
            return {"nodes": [], "edges": [], "center_entity": None}
        if path == "/dashboard/graph/explore":
            return {"nodes": [], "edges": [], "center_entity": "pizza"}
        if path == "/dashboard/graph/search":
            return {"results": []}
        if path == "/dashboard/graph/neo4j-config":
            return {
                "server_url": "bolt://neo4j:7687",
                "server_user": "neo4j",
                "server_password": "password",
            }
        if path == "/dashboard/config":
            if method == "GET":
                return {"sections": []}
            return {"success": True}
        if path == "/dashboard/labile":
            return {
                "tenants": [],
                "total_db_labile": 0,
                "total_redis_scopes": 0,
                "total_redis_sessions": 0,
                "total_redis_memories": 0,
            }
        if path == "/dashboard/retrieval":
            return {"query": "food", "results": [], "total_count": 0, "elapsed_ms": 0.0}
        if path == "/dashboard/jobs":
            return {"items": [], "total": 0}
        if path == "/dashboard/database/reset":
            return {"success": True}
        if path == "/dashboard/memories/bulk-action":
            return {"success": True}
        if path == "/dashboard/consolidate":
            return {"status": "completed"}
        if path == "/dashboard/forget":
            return {"status": "completed"}
        if path == "/dashboard/reconsolidate":
            return {"status": "completed"}
        if path == "/admin/consolidate/user-1":
            return {"status": "consolidation_completed"}
        if path == "/admin/forget/user-1":
            return {"status": "forgetting_completed"}
        if path == "/memory/write":
            return {
                "success": True,
                "memory_id": str(memory_id),
                "chunks_created": 1,
                "message": "stored",
            }
        raise AssertionError(f"Unhandled path: {path}")

    client._transport.request = _request  # type: ignore[method-assign]

    assert (await client.stats()).total_memories == 3
    assert (await client.create_session(name="demo")).session_id == "sess-1"
    assert (await client.get_session_context("sess-1")).session_id == "sess-1"
    assert await client.delete_all(confirm=True) == 0
    assert await client.get_context("food") == "ctx"
    assert (await client.remember("hello")).success is True
    assert (await client.search("food")).query == "food"
    await client.consolidate()
    await client.run_forgetting()
    await client.reconsolidate()
    await client.admin_consolidate("user-1")
    await client.admin_forget("user-1", dry_run=False)
    assert (await client.list_tenants()).tenants[0].tenant_id == "tenant-a"
    assert (await client.dashboard_overview()).total_memories == 0
    assert (await client.dashboard_memories()).items[0].id == memory_id
    assert (await client.dashboard_memory_detail(memory_id)).id == memory_id
    assert (await client.dashboard_facts()).total == 0
    assert (await client.dashboard_invalidate_fact("fact-1"))["success"] is True
    assert (await client.dashboard_export_memories())[0]["id"] == str(memory_id)
    assert (await client.get_events()).total == 0
    assert (await client.dashboard_timeline()).total == 0
    assert (await client.component_health()).components[0].name == "PostgreSQL"
    assert (await client.get_sessions()).total_active == 0
    assert (await client.get_rate_limits()).configured_rpm == 60
    assert (await client.get_request_stats()).total_last_24h == 0
    assert (await client.get_graph_stats()).total_nodes == 1
    assert (await client.graph_overview()).center_entity is None
    assert (await client.explore_graph(entity="pizza")).center_entity == "pizza"
    assert (await client.search_graph("pizza")).results == []
    assert (await client.dashboard_neo4j_config()).server_user == "neo4j"
    assert (await client.get_config()).sections == []
    assert (await client.update_config({"debug": True}))["success"] is True
    assert (await client.get_labile_status()).total_db_labile == 0
    assert (await client.test_retrieval("food")).query == "food"
    assert (await client.get_jobs()).total == 0
    assert (await client.reset_database(confirm=True))["success"] is True
    assert (await client.bulk_memory_action([memory_id], "archive"))["success"] is True

    called_paths = [path for _, path, _ in calls]
    assert "/dashboard/database/reset" in called_paths
    assert "/dashboard/memories/bulk-action" in called_paths
    assert "/dashboard/retrieval" in called_paths
    assert any(
        path == "/dashboard/config" and kwargs.get("json") == {"updates": {"debug": True}}
        for _, path, kwargs in calls
    )


@pytest.mark.asyncio
async def test_async_client_batch_iter_and_namespace_wrappers(
    cml_config: CMLConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncCognitiveMemoryLayer(config=cml_config)
    memory_id = uuid4()
    page_calls: list[int] = []

    async def _request(method: str, path: str, **kwargs: object) -> dict[str, object]:
        if path == "/memory/write":
            return {
                "success": True,
                "memory_id": str(uuid4()),
                "chunks_created": 1,
                "message": "ok",
            }
        if path == "/memory/read":
            return {
                "query": str(kwargs["json"]["query"]),
                "memories": [],
                "facts": [],
                "preferences": [],
                "episodes": [],
                "constraints": [],
                "total_count": 0,
                "elapsed_ms": 1.0,
            }
        if path == "/dashboard/memories":
            page = int(kwargs["params"]["page"])
            page_calls.append(page)
            if page == 1:
                return {
                    "items": [
                        {
                            "id": str(memory_id),
                            "tenant_id": "tenant-a",
                            "type": "semantic_fact",
                            "status": "active",
                            "text": "hello",
                            "confidence": 0.8,
                            "importance": 0.5,
                            "timestamp": "2025-01-01T00:00:00Z",
                        }
                    ],
                    "total": 2,
                    "page": 1,
                    "per_page": 1,
                    "total_pages": 2,
                }
            return {"items": [], "total": 2, "page": 2, "per_page": 1, "total_pages": 2}
        if path == "/session/sess-1/write":
            return {
                "success": True,
                "memory_id": str(memory_id),
                "chunks_created": 1,
                "message": "ok",
            }
        if path == "/session/sess-1/read":
            return {
                "query": "food",
                "memories": [],
                "facts": [],
                "preferences": [],
                "episodes": [],
                "constraints": [],
                "total_count": 0,
                "elapsed_ms": 1.0,
            }
        raise AssertionError(f"Unexpected path {path}")

    client._transport.request = _request  # type: ignore[method-assign]
    writes = await client.batch_write([{"content": "one"}, {"content": "two"}], namespace="ns-1")
    assert len(writes) == 2
    reads = await client.batch_read(["a", "b"], response_format="list")
    assert len(reads) == 2
    items = [item async for item in client.iter_memories(batch_size=1)]
    assert len(items) == 1
    assert page_calls == [1, 2]

    await client.set_tenant("tenant-b")
    assert client.tenant_id == "tenant-b"

    session = AsyncSessionScope(client, "sess-1")
    assert isinstance(await session.write("hello"), WriteResponse)
    assert (await session.read("food")).query == "food"

    async def _turn(*args: object, **kwargs: object) -> TurnResponse:
        return TurnResponse(
            memory_context="",
            memories_retrieved=0,
            memories_stored=0,
            reconsolidation_applied=False,
        )

    monkeypatch.setattr(client, "turn", _turn)
    await session.turn("hello", assistant_response="world")

    async def _write(*args: object, **kwargs: object) -> WriteResponse:
        return WriteResponse(success=True, memory_id=None, chunks_created=0, message="ok")

    monkeypatch.setattr(client, "write", _write)
    await session.remember("hello again")

    namespace_client = AsyncNamespacedClient(client, "ns-1")

    async def _constant(value: object):
        return value

    monkeypatch.setattr(client, "write", lambda *args, **kwargs: _constant("write-ok"))
    monkeypatch.setattr(client, "read", lambda *args, **kwargs: _constant("read-ok"))
    monkeypatch.setattr(client, "dashboard_overview", lambda *args, **kwargs: _constant("overview"))
    monkeypatch.setattr(
        client, "dashboard_memory_detail", lambda *args, **kwargs: _constant("detail")
    )
    monkeypatch.setattr(client, "dashboard_facts", lambda *args, **kwargs: _constant("facts"))
    monkeypatch.setattr(
        client, "dashboard_invalidate_fact", lambda *args, **kwargs: _constant("invalidate")
    )
    monkeypatch.setattr(
        client, "dashboard_export_memories", lambda *args, **kwargs: _constant("export")
    )
    monkeypatch.setattr(client, "get_events", lambda *args, **kwargs: _constant("events"))
    monkeypatch.setattr(client, "dashboard_timeline", lambda *args, **kwargs: _constant("timeline"))
    monkeypatch.setattr(client, "component_health", lambda *args, **kwargs: _constant("components"))
    monkeypatch.setattr(client, "get_sessions", lambda *args, **kwargs: _constant("sessions"))
    monkeypatch.setattr(client, "get_rate_limits", lambda *args, **kwargs: _constant("limits"))
    monkeypatch.setattr(client, "get_request_stats", lambda *args, **kwargs: _constant("stats"))
    monkeypatch.setattr(client, "get_graph_stats", lambda *args, **kwargs: _constant("graph-stats"))
    monkeypatch.setattr(
        client, "graph_overview", lambda *args, **kwargs: _constant("graph-overview")
    )
    monkeypatch.setattr(client, "explore_graph", lambda *args, **kwargs: _constant("explore"))
    monkeypatch.setattr(client, "search_graph", lambda *args, **kwargs: _constant("search"))
    monkeypatch.setattr(
        client, "dashboard_neo4j_config", lambda *args, **kwargs: _constant("neo4j")
    )
    monkeypatch.setattr(client, "get_config", lambda *args, **kwargs: _constant("config"))
    monkeypatch.setattr(client, "update_config", lambda *args, **kwargs: _constant("updated"))
    monkeypatch.setattr(client, "get_labile_status", lambda *args, **kwargs: _constant("labile"))
    monkeypatch.setattr(client, "test_retrieval", lambda *args, **kwargs: _constant("retrieval"))
    monkeypatch.setattr(client, "get_jobs", lambda *args, **kwargs: _constant("jobs"))
    monkeypatch.setattr(client, "reset_database", lambda *args, **kwargs: _constant("reset"))
    monkeypatch.setattr(client, "bulk_memory_action", lambda *args, **kwargs: _constant("bulk"))
    monkeypatch.setattr(
        client, "admin_consolidate", lambda *args, **kwargs: _constant("admin-consolidate")
    )
    monkeypatch.setattr(client, "admin_forget", lambda *args, **kwargs: _constant("admin-forget"))

    assert await namespace_client.write("hello") == "write-ok"
    assert await namespace_client.read("query") == "read-ok"
    assert await namespace_client.dashboard_overview() == "overview"
    assert await namespace_client.dashboard_memory_detail(memory_id) == "detail"
    assert await namespace_client.dashboard_facts() == "facts"
    assert await namespace_client.dashboard_invalidate_fact("fact-1") == "invalidate"
    assert await namespace_client.dashboard_export_memories() == "export"
    assert await namespace_client.get_events() == "events"
    assert await namespace_client.dashboard_timeline() == "timeline"
    assert await namespace_client.component_health() == "components"
    assert await namespace_client.get_sessions() == "sessions"
    assert await namespace_client.get_rate_limits() == "limits"
    assert await namespace_client.get_request_stats() == "stats"
    assert await namespace_client.get_graph_stats() == "graph-stats"
    assert await namespace_client.graph_overview() == "graph-overview"
    assert await namespace_client.explore_graph(entity="pizza") == "explore"
    assert await namespace_client.search_graph("pizza") == "search"
    assert await namespace_client.dashboard_neo4j_config() == "neo4j"
    assert await namespace_client.get_config() == "config"
    assert await namespace_client.update_config({"debug": True}) == "updated"
    assert await namespace_client.get_labile_status() == "labile"
    assert await namespace_client.test_retrieval("food") == "retrieval"
    assert await namespace_client.get_jobs() == "jobs"
    assert await namespace_client.reset_database(confirm=True) == "reset"
    assert await namespace_client.bulk_memory_action([memory_id], "delete") == "bulk"
    assert await namespace_client.admin_consolidate("user-1") == "admin-consolidate"
    assert await namespace_client.admin_forget("user-1", dry_run=False) == "admin-forget"


@pytest.mark.asyncio
async def test_async_client_reset_database_requires_confirm(cml_config: CMLConfig) -> None:
    client = AsyncCognitiveMemoryLayer(config=cml_config)
    with pytest.raises(ValueError, match="confirm=True"):
        await client.reset_database()


@pytest.mark.asyncio
async def test_async_client_iter_memories_rejects_multiple_types(cml_config: CMLConfig) -> None:
    client = AsyncCognitiveMemoryLayer(config=cml_config)
    with pytest.raises(ValueError, match="at most one memory type"):
        _ = [
            item
            async for item in client.iter_memories(
                memory_types=[MemoryType.PREFERENCE, MemoryType.SEMANTIC_FACT]
            )
        ]
