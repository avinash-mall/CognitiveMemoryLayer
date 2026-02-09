"""Unit tests for Phase 5 advanced features (admin, batch, tenant, events, health, namespace, iter_memories, OpenAI helper)."""

from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

from cml import CognitiveMemoryLayer, NamespacedClient
from cml.config import CMLConfig
from cml.integrations import CMLOpenAIHelper
from cml.models import MemoryItem, ReadResponse, WriteResponse

# ---- Admin operations ----


def test_consolidate_calls_dashboard_consolidate() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000", tenant_id="t1")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "status": "completed",
            "episodes_sampled": 10,
            "clusters_formed": 2,
            "gists_extracted": 3,
        }
    )
    out = client.consolidate()
    assert out["episodes_sampled"] == 10
    client._transport.request.assert_called_once()
    call = client._transport.request.call_args
    assert call[0] == ("POST", "dashboard/consolidate")
    assert call[1]["json"]["tenant_id"] == "t1"
    assert call[1]["use_admin_key"] is True


def test_consolidate_with_user_id() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000", tenant_id="t1")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"status": "completed", "episodes_sampled": 0}
    )
    client.consolidate(tenant_id="t2", user_id="u1")
    call = client._transport.request.call_args
    assert call[1]["json"]["tenant_id"] == "t2"
    assert call[1]["json"]["user_id"] == "u1"


def test_run_forgetting_calls_dashboard_forget() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000", tenant_id="t1")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "status": "completed",
            "dry_run": True,
            "memories_scanned": 100,
            "operations_applied": 0,
        }
    )
    out = client.run_forgetting(dry_run=True, max_memories=500)
    assert out["memories_scanned"] == 100
    call = client._transport.request.call_args
    assert call[0] == ("POST", "dashboard/forget")
    assert call[1]["json"]["tenant_id"] == "t1"
    assert call[1]["json"]["dry_run"] is True
    assert call[1]["json"]["max_memories"] == 500
    assert call[1]["use_admin_key"] is True


# ---- Batch operations ----


def test_batch_write_sequential() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    ids = [str(uuid4()), str(uuid4())]
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            {"success": True, "memory_id": ids[0], "chunks_created": 1, "message": ""},
            {"success": True, "memory_id": ids[1], "chunks_created": 1, "message": ""},
        ]
    )
    results = client.batch_write(
        [{"content": "first"}, {"content": "second", "context_tags": ["tag"]}],
        namespace="ns1",
    )
    assert len(results) == 2
    assert all(isinstance(r, WriteResponse) for r in results)
    assert results[0].memory_id is not None
    assert results[1].memory_id is not None
    assert client._transport.request.call_count == 2
    assert client._transport.request.call_args_list[1][1]["json"]["namespace"] == "ns1"


def test_batch_read_sequential() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "query": "q",
            "memories": [],
            "total_count": 0,
            "elapsed_ms": 1.0,
        }
    )
    results = client.batch_read(["q1", "q2"], max_results=5)
    assert len(results) == 2
    assert all(isinstance(r, ReadResponse) for r in results)
    assert client._transport.request.call_count == 2


# ---- Tenant management ----


def test_set_tenant_updates_config_and_closes_transport() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000", tenant_id="t1")
    client = CognitiveMemoryLayer(config=config)
    client._transport.close = MagicMock()  # type: ignore[method-assign]
    client.set_tenant("t2")
    assert client._config.tenant_id == "t2"
    client._transport.close.assert_called_once()


def test_tenant_id_property() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000", tenant_id="my-tenant")
    client = CognitiveMemoryLayer(config=config)
    assert client.tenant_id == "my-tenant"


def test_list_tenants_calls_dashboard_tenants() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "tenants": [
                {"tenant_id": "t1", "memory_count": 10, "fact_count": 2, "event_count": 5},
            ]
        }
    )
    tenants = client.list_tenants()
    assert len(tenants) == 1
    assert tenants[0]["tenant_id"] == "t1"
    assert tenants[0]["memory_count"] == 10
    call = client._transport.request.call_args
    assert call[0] == ("GET", "dashboard/tenants")
    assert call[1]["use_admin_key"] is True


# ---- Event log ----


def test_get_events_calls_dashboard_events() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "items": [],
            "total": 0,
            "page": 2,
            "per_page": 25,
            "total_pages": 1,
        }
    )
    out = client.get_events(limit=25, page=2, event_type="memory_op")
    assert out["page"] == 2
    assert out["per_page"] == 25
    call = client._transport.request.call_args
    assert call[0] == ("GET", "dashboard/events")
    assert call[1]["params"]["per_page"] == 25
    assert call[1]["params"]["page"] == 2
    assert call[1]["params"]["event_type"] == "memory_op"
    assert call[1]["use_admin_key"] is True


def test_get_events_with_since() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"items": [], "total": 0, "page": 1, "per_page": 50, "total_pages": 1}
    )
    since = datetime(2025, 1, 1, 12, 0, 0)
    client.get_events(since=since)
    call = client._transport.request.call_args
    assert "since" in call[1]["params"]


# ---- Component health ----


def test_component_health_calls_dashboard_components() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "components": [
                {"name": "PostgreSQL", "status": "ok", "latency_ms": 1.0},
            ]
        }
    )
    out = client.component_health()
    assert "components" in out
    assert out["components"][0]["name"] == "PostgreSQL"
    call = client._transport.request.call_args
    assert call[0] == ("GET", "dashboard/components")
    assert call[1]["use_admin_key"] is True


# ---- Namespace isolation ----


def test_with_namespace_returns_namespaced_client() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    wrapped = client.with_namespace("user-123")
    assert isinstance(wrapped, NamespacedClient)
    assert wrapped._namespace == "user-123"
    assert wrapped._parent is client


def test_namespaced_client_write_injects_namespace() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "success": True,
            "memory_id": str(uuid4()),
            "chunks_created": 1,
            "message": "",
        }
    )
    ns = client.with_namespace("ns1")
    ns.write("hello")
    call = client._transport.request.call_args
    assert call[1]["json"]["namespace"] == "ns1"


def test_namespaced_client_read_delegates() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"query": "q", "memories": [], "total_count": 0, "elapsed_ms": 1.0}
    )
    ns = client.with_namespace("ns1")
    ns.read("query")
    client._transport.request.assert_called_once()
    assert client._transport.request.call_args[0] == ("POST", "/memory/read")


# ---- Memory iteration ----


def test_iter_memories_one_page() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    mid = uuid4()
    ts = "2025-01-01T12:00:00Z"
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={
            "items": [
                {
                    "id": str(mid),
                    "text": "a memory",
                    "type": "semantic_fact",
                    "confidence": 0.9,
                    "importance": 0.8,
                    "timestamp": ts,
                }
            ],
            "total": 1,
            "page": 1,
            "per_page": 100,
            "total_pages": 1,
        }
    )
    items = list(client.iter_memories(batch_size=100))
    assert len(items) == 1
    assert isinstance(items[0], MemoryItem)
    assert items[0].text == "a memory"
    assert items[0].type == "semantic_fact"
    call = client._transport.request.call_args
    assert call[0] == ("GET", "dashboard/memories")
    assert call[1]["params"]["per_page"] == 100
    assert call[1]["params"]["status"] == "active"
    assert call[1]["use_admin_key"] is True


def test_iter_memories_empty() -> None:
    config = CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    client = CognitiveMemoryLayer(config=config)
    client._transport.request = MagicMock(  # type: ignore[method-assign]
        return_value={"items": [], "total": 0, "page": 1, "per_page": 100, "total_pages": 1}
    )
    items = list(client.iter_memories())
    assert len(items) == 0


# ---- OpenAI helper ----


def test_openai_helper_chat_flow() -> None:
    memory = CognitiveMemoryLayer(
        config=CMLConfig(api_key="sk-test", base_url="http://localhost:8000")
    )
    memory._transport.request = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            {
                "memory_context": "No relevant memories.",
                "memories_retrieved": 0,
                "memories_stored": 0,
            },
            {"memory_context": "", "memories_retrieved": 0, "memories_stored": 1},
        ]
    )
    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Pizza is a good choice!"))]
    )
    helper = CMLOpenAIHelper(memory, fake_openai, model="gpt-4o")
    reply = helper.chat("What should I eat?", session_id="s1")
    assert reply == "Pizza is a good choice!"
    assert memory._transport.request.call_count == 2
    create_call = fake_openai.chat.completions.create.call_args
    assert create_call[1]["model"] == "gpt-4o"
    messages = create_call[1]["messages"]
    assert messages[0]["role"] == "system"
    assert "Relevant Memories" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "What should I eat?"
