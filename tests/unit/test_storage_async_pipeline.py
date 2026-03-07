from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.storage.async_pipeline as async_pipeline
from src.storage.async_pipeline import (
    _PROCESSED_PREFIX,
    _PROCESSED_TTL,
    _QUEUE_KEY,
    AsyncStoragePipeline,
)


@pytest.mark.asyncio
async def test_process_next_returns_false_when_queue_is_empty() -> None:
    redis = AsyncMock()
    redis.blpop = AsyncMock(return_value=None)
    pipeline = AsyncStoragePipeline(redis_client=redis, enabled=True)
    orchestrator = SimpleNamespace(write=AsyncMock())

    assert await pipeline.process_next(orchestrator) is False
    orchestrator.write.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_next_skips_already_processed_jobs() -> None:
    payload = {"idempotency_key": "abc123"}
    redis = AsyncMock()
    redis.blpop = AsyncMock(return_value=(_QUEUE_KEY, json.dumps(payload)))
    redis.exists = AsyncMock(return_value=True)
    pipeline = AsyncStoragePipeline(redis_client=redis, enabled=True)
    orchestrator = SimpleNamespace(write=AsyncMock())

    assert await pipeline.process_next(orchestrator) is True
    orchestrator.write.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_next_writes_user_and_assistant_messages() -> None:
    payload = {
        "tenant_id": "tenant-a",
        "user_message": "hello",
        "assistant_response": "world",
        "session_id": "sess-1",
        "timestamp": datetime.now(UTC).isoformat(),
        "context_tags_user": ["conversation"],
        "context_tags_assistant": ["assistant"],
        "idempotency_key": "job-1",
    }
    redis = AsyncMock()
    redis.blpop = AsyncMock(return_value=(_QUEUE_KEY, json.dumps(payload)))
    redis.exists = AsyncMock(return_value=False)
    redis.setex = AsyncMock()
    pipeline = AsyncStoragePipeline(redis_client=redis, enabled=True)
    orchestrator = SimpleNamespace(write=AsyncMock())

    assert await pipeline.process_next(orchestrator) is True

    assert orchestrator.write.await_count == 2
    first_call = orchestrator.write.await_args_list[0].kwargs
    second_call = orchestrator.write.await_args_list[1].kwargs
    assert first_call["content"] == "hello"
    assert first_call["context_tags"] == ["conversation"]
    assert second_call["content"] == "world"
    assert second_call["context_tags"] == ["assistant"]
    redis.setex.assert_awaited_once_with(f"{_PROCESSED_PREFIX}job-1", _PROCESSED_TTL, "1")


@pytest.mark.asyncio
async def test_process_next_requeues_retryable_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "job_id": "job-2",
        "tenant_id": "tenant-a",
        "user_message": "hello",
        "assistant_response": None,
        "idempotency_key": "job-2",
        "_retry": 0,
    }
    redis = AsyncMock()
    redis.blpop = AsyncMock(return_value=(_QUEUE_KEY, json.dumps(payload)))
    redis.exists = AsyncMock(return_value=False)
    redis.rpush = AsyncMock()
    pipeline = AsyncStoragePipeline(redis_client=redis, enabled=True)
    orchestrator = SimpleNamespace(write=AsyncMock(side_effect=RuntimeError("boom")))
    logger = MagicMock()
    monkeypatch.setattr(async_pipeline, "logger", logger)

    assert await pipeline.process_next(orchestrator) is True

    redis.rpush.assert_awaited_once()
    requeued = json.loads(redis.rpush.await_args.args[1])
    assert requeued["_retry"] == 1
    logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_process_next_logs_dead_letter_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "job_id": "job-3",
        "tenant_id": "tenant-a",
        "user_message": "hello",
        "assistant_response": None,
        "idempotency_key": "job-3",
        "_retry": 2,
    }
    redis = AsyncMock()
    redis.blpop = AsyncMock(return_value=(_QUEUE_KEY, json.dumps(payload)))
    redis.exists = AsyncMock(return_value=False)
    redis.rpush = AsyncMock()
    pipeline = AsyncStoragePipeline(redis_client=redis, enabled=True)
    orchestrator = SimpleNamespace(write=AsyncMock(side_effect=RuntimeError("boom")))
    logger = MagicMock()
    monkeypatch.setattr(async_pipeline, "logger", logger)

    assert await pipeline.process_next(orchestrator) is True

    redis.rpush.assert_not_awaited()
    logger.error.assert_called_once()
