from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.api import routes
from src.core.exceptions import MemoryNotFoundError

AUTH = SimpleNamespace(tenant_id="tenant-a")


def _request(*, eval_mode: bool = False, redis: object | None = None) -> SimpleNamespace:
    headers = {"X-Eval-Mode": "true"} if eval_mode else {}
    app_state = SimpleNamespace(db=SimpleNamespace(redis=redis))
    return SimpleNamespace(headers=headers, app=SimpleNamespace(state=app_state))


def _memory_record(
    *,
    memory_id: str | None = None,
    text: str = "memory text",
    memory_type: str = "semantic_fact",
    metadata: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=memory_id or str(uuid4()),
        text=text,
        type=SimpleNamespace(value=memory_type),
        confidence=0.9,
        timestamp=datetime.now(UTC),
        metadata=metadata or {},
    )


def _retrieved_memory(
    *,
    memory_id: str | None = None,
    text: str = "memory text",
    memory_type: str = "semantic_fact",
) -> SimpleNamespace:
    memory_id = memory_id or str(uuid4())
    return SimpleNamespace(
        record=_memory_record(memory_id=memory_id, text=text, memory_type=memory_type),
        relevance_score=0.75,
    )


def _packet(*, memories: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    items = list(memories or [_retrieved_memory()])
    return SimpleNamespace(
        all_memories=items,
        facts=[item for item in items if item.record.type.value == "semantic_fact"],
        preferences=[item for item in items if item.record.type.value == "preference"],
        recent_episodes=[item for item in items if item.record.type.value == "episodic_event"],
        constraints=[item for item in items if item.record.type.value == "constraint"],
        retrieval_meta={"source": "test"},
    )


class _RedisSetexStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    async def setex(self, key: str, ttl_seconds: int, payload: str) -> None:
        self.calls.append((key, ttl_seconds, payload))


@pytest.mark.asyncio
async def test_write_memory_returns_eval_fields_in_eval_mode() -> None:
    orchestrator = SimpleNamespace(
        write=AsyncMock(
            return_value={
                "memory_id": str(uuid4()),
                "chunks_created": 2,
                "message": "Stored",
                "eval_outcome": "stored",
                "eval_reason": "novel",
            }
        )
    )

    result = await routes.write_memory(
        request=_request(eval_mode=True),
        body=routes.WriteMemoryRequest(content="hello", metadata={"topic": "food"}),
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result.success is True
    assert result.eval_outcome == "stored"
    assert result.eval_reason == "novel"
    assert orchestrator.write.await_args.kwargs["eval_mode"] is True


@pytest.mark.asyncio
async def test_write_memory_rejects_overlong_content() -> None:
    orchestrator = SimpleNamespace(write=AsyncMock())

    with pytest.raises(HTTPException, match="Content exceeds maximum length"):
        await routes.write_memory(
            request=_request(),
            body=SimpleNamespace(
                content="x" * 100_001,
                context_tags=None,
                session_id=None,
                memory_type=None,
                metadata={},
                turn_id=None,
                agent_id=None,
                namespace=None,
                timestamp=None,
            ),
            auth=AUTH,
            orchestrator=orchestrator,
        )


@pytest.mark.asyncio
async def test_process_turn_uses_seamless_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = SimpleNamespace(
        memory_context="context",
        injected_memories=[object(), object()],
        stored_count=1,
        reconsolidation_applied=True,
    )

    class _Provider:
        def __init__(self, orchestrator: object, max_context_tokens: int, auto_store: bool) -> None:
            assert max_context_tokens == 123
            assert auto_store is True

        async def process_turn(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["tenant_id"] == "tenant-a"
            assert kwargs["user_timezone"] == "Europe/London"
            return expected

    monkeypatch.setattr(routes, "SeamlessMemoryProvider", _Provider)

    result = await routes.process_turn(
        body=routes.ProcessTurnRequest(
            user_message="hello",
            assistant_response="world",
            session_id="sess-1",
            max_context_tokens=123,
            user_timezone="Europe/London",
        ),
        auth=AUTH,
        orchestrator=SimpleNamespace(),
    )

    assert result.memory_context == "context"
    assert result.memories_retrieved == 2
    assert result.memories_stored == 1
    assert result.reconsolidation_applied is True


@pytest.mark.asyncio
async def test_read_memory_list_format_omits_categorized_buckets() -> None:
    orchestrator = SimpleNamespace(read=AsyncMock(return_value=_packet()))

    result = await routes.read_memory(
        body=routes.ReadMemoryRequest(query="pizza", format="list"),
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result.query == "pizza"
    assert result.total_count == 1
    assert result.facts == []
    assert result.preferences == []
    assert result.episodes == []
    assert result.constraints == []


@pytest.mark.asyncio
async def test_read_memory_llm_context_uses_packet_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = SimpleNamespace(read=AsyncMock(return_value=_packet()))

    class _Builder:
        def to_llm_context(self, packet: object, max_tokens: int) -> str:
            assert max_tokens == 2000
            assert packet is not None
            return "LLM context"

    monkeypatch.setattr("src.retrieval.packet_builder.MemoryPacketBuilder", _Builder)

    result = await routes.read_memory(
        body=routes.ReadMemoryRequest(query="pizza", format="llm_context"),
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result.llm_context == "LLM context"
    assert result.facts[0].text == "memory text"


@pytest.mark.asyncio
async def test_update_memory_returns_result() -> None:
    memory_id = uuid4()
    orchestrator = SimpleNamespace(update=AsyncMock(return_value={"version": 3}))

    result = await routes.update_memory(
        body=routes.UpdateMemoryRequest(memory_id=memory_id, text="updated"),
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result.success is True
    assert result.version == 3


@pytest.mark.asyncio
async def test_update_memory_returns_http_404_for_value_errors() -> None:
    orchestrator = SimpleNamespace(update=AsyncMock(side_effect=ValueError("not found")))

    with pytest.raises(HTTPException, match="not found"):
        await routes.update_memory(
            body=routes.UpdateMemoryRequest(memory_id=uuid4()),
            auth=AUTH,
            orchestrator=orchestrator,
        )


@pytest.mark.asyncio
async def test_update_memory_re_raises_memory_not_found() -> None:
    orchestrator = SimpleNamespace(
        update=AsyncMock(side_effect=MemoryNotFoundError(memory_id="mem-404"))
    )

    with pytest.raises(MemoryNotFoundError):
        await routes.update_memory(
            body=routes.UpdateMemoryRequest(memory_id=uuid4()),
            auth=AUTH,
            orchestrator=orchestrator,
        )


@pytest.mark.asyncio
async def test_forget_memory_returns_success() -> None:
    orchestrator = SimpleNamespace(forget=AsyncMock(return_value={"affected_count": 2}))

    result = await routes.forget_memory(
        body=routes.ForgetRequest(query="obsolete"),
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result.success is True
    assert result.affected_count == 2


@pytest.mark.asyncio
async def test_forget_memory_wraps_errors() -> None:
    orchestrator = SimpleNamespace(forget=AsyncMock(side_effect=RuntimeError("forget failed")))

    with pytest.raises(HTTPException, match="Internal server error"):
        await routes.forget_memory(
            body=routes.ForgetRequest(query="obsolete"),
            auth=AUTH,
            orchestrator=orchestrator,
        )


@pytest.mark.asyncio
async def test_get_memory_stats_returns_payload() -> None:
    orchestrator = SimpleNamespace(
        get_stats=AsyncMock(
            return_value={
                "total_memories": 5,
                "active_memories": 4,
                "silent_memories": 1,
                "archived_memories": 0,
                "by_type": {
                    "semantic_fact": 2,
                    "preference": 1,
                    "constraint": 1,
                    "episodic_event": 1,
                },
                "avg_confidence": 0.8,
                "avg_importance": 0.6,
                "estimated_size_mb": 1.0,
            }
        )
    )

    result = await routes.get_memory_stats(auth=AUTH, orchestrator=orchestrator)

    assert result.total_memories == 5
    assert result.active_memories == 4
    assert result.by_type["constraint"] == 1


@pytest.mark.asyncio
async def test_get_memory_stats_wraps_errors() -> None:
    orchestrator = SimpleNamespace(get_stats=AsyncMock(side_effect=RuntimeError("stats failed")))

    with pytest.raises(HTTPException, match="Internal server error"):
        await routes.get_memory_stats(auth=AUTH, orchestrator=orchestrator)


@pytest.mark.asyncio
async def test_create_session_persists_to_redis_when_available() -> None:
    redis = _RedisSetexStub()

    result = await routes.create_session(
        request=_request(redis=redis),
        body=routes.CreateSessionRequest(ttl_hours=2),
        auth=AUTH,
    )

    assert result.session_id
    assert len(redis.calls) == 1
    key, ttl_seconds, payload = redis.calls[0]
    assert key.startswith("session:")
    assert ttl_seconds == 7200
    doc = json.loads(payload)
    assert doc["tenant_id"] == "tenant-a"


@pytest.mark.asyncio
async def test_create_session_without_redis_still_returns_response() -> None:
    result = await routes.create_session(
        request=_request(redis=None),
        body=routes.CreateSessionRequest(ttl_hours=1),
        auth=AUTH,
    )

    assert result.session_id
    assert result.expires_at > result.created_at


@pytest.mark.asyncio
async def test_session_write_uses_path_session_and_defaults_context_tag() -> None:
    orchestrator = SimpleNamespace(
        write=AsyncMock(return_value={"memory_id": str(uuid4()), "chunks_created": 1})
    )

    result = await routes.session_write(
        request=_request(eval_mode=True),
        session_id="sess-123",
        body=routes.WriteMemoryRequest(content="hello"),
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result.success is True
    kwargs = orchestrator.write.await_args.kwargs
    assert kwargs["session_id"] == "sess-123"
    assert kwargs["context_tags"] == ["conversation"]
    assert kwargs["eval_mode"] is True


@pytest.mark.asyncio
async def test_session_read_forwards_source_session_id() -> None:
    fact_id = str(uuid4())
    constraint_id = str(uuid4())
    memories = [
        _retrieved_memory(memory_id=fact_id, memory_type="semantic_fact"),
        _retrieved_memory(memory_id=constraint_id, memory_type="constraint"),
    ]
    orchestrator = SimpleNamespace(read=AsyncMock(return_value=_packet(memories=memories)))

    result = await routes.session_read(
        session_id="sess-123",
        body=routes.ReadMemoryRequest(query="q"),
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result.total_count == 2
    assert orchestrator.read.await_args.kwargs["source_session_id"] == "sess-123"
    assert str(result.constraints[0].id) == constraint_id


@pytest.mark.asyncio
async def test_session_context_maps_messages_tools_and_scratchpad() -> None:
    now = datetime.now(UTC)
    ctx = {
        "messages": [
            {
                "id": str(uuid4()),
                "text": "hello",
                "type": "episodic_event",
                "confidence": 0.7,
                "relevance": 0.8,
                "timestamp": now,
            }
        ],
        "tool_results": [],
        "scratch_pad": [],
        "context_string": "formatted context",
    }
    orchestrator = SimpleNamespace(get_session_context=AsyncMock(return_value=ctx))

    result = await routes.session_context(
        session_id="sess-123",
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result.session_id == "sess-123"
    assert result.messages[0].text == "hello"
    assert result.context_string == "formatted context"


@pytest.mark.asyncio
async def test_delete_all_memories_returns_affected_count() -> None:
    orchestrator = SimpleNamespace(delete_all=AsyncMock(return_value=12))

    result = await routes.delete_all_memories(auth=AUTH, orchestrator=orchestrator)

    assert result == {"affected_count": 12}


@pytest.mark.asyncio
async def test_delete_all_memories_wraps_errors() -> None:
    orchestrator = SimpleNamespace(delete_all=AsyncMock(side_effect=RuntimeError("delete failed")))

    with pytest.raises(HTTPException, match="Internal server error"):
        await routes.delete_all_memories(auth=AUTH, orchestrator=orchestrator)


@pytest.mark.asyncio
async def test_read_memory_stream_emits_data_and_done_events() -> None:
    mem_one = str(uuid4())
    mem_two = str(uuid4())
    packet = _packet(
        memories=[_retrieved_memory(memory_id=mem_one), _retrieved_memory(memory_id=mem_two)]
    )
    orchestrator = SimpleNamespace(read=AsyncMock(return_value=packet))

    response = await routes.read_memory_stream(
        body=routes.ReadMemoryRequest(query="pizza"),
        auth=AUTH,
        orchestrator=orchestrator,
    )

    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    body = "".join(chunks)

    assert f'"id":"{mem_one}"' in body or f'"id": "{mem_one}"' in body
    assert f'"id":"{mem_two}"' in body or f'"id": "{mem_two}"' in body
    assert "event: done" in body
    assert '"total_count": 2' in body


@pytest.mark.asyncio
async def test_read_memory_stream_emits_error_events() -> None:
    orchestrator = SimpleNamespace(read=AsyncMock(side_effect=RuntimeError("stream failed")))

    response = await routes.read_memory_stream(
        body=routes.ReadMemoryRequest(query="pizza"),
        auth=AUTH,
        orchestrator=orchestrator,
    )

    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    body = "".join(chunks)

    assert "event: error" in body
    assert "Internal server error" in body
