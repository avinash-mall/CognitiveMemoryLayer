"""Async storage pipeline: decouple writes from the hot response path.

When ``STORE_ASYNC`` is enabled, writes are enqueued to Redis and
processed by a background worker.  This removes storage latency from
the user-facing ``/memory/turn`` response.

The pipeline provides idempotency via content-hash dedup keys to
prevent double-writes on retries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..memory.orchestrator import MemoryOrchestrator

logger = logging.getLogger(__name__)

_QUEUE_KEY = "cml:storage:queue"
_PROCESSED_PREFIX = "cml:storage:processed:"
_PROCESSED_TTL = 3600  # 1-hour dedup window
_MAX_RETRIES = 3


@dataclass
class StorageJob:
    """A queued storage job."""

    job_id: str
    tenant_id: str
    user_message: str
    assistant_response: str | None
    session_id: str | None
    turn_id: str | None
    timestamp: str | None  # ISO format
    context_tags_user: list[str]
    context_tags_assistant: list[str]
    idempotency_key: str
    _retry: int = 0


class AsyncStoragePipeline:
    """Enqueue storage operations for background processing.

    When ``enabled`` is *True*, :meth:`enqueue` pushes a job to
    a Redis list and returns immediately, removing storage latency
    from the turn response path.
    """

    def __init__(self, redis_client: Any, enabled: bool = False) -> None:
        self.redis = redis_client
        self.enabled = enabled

    async def enqueue(
        self,
        tenant_id: str,
        user_message: str,
        assistant_response: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        timestamp: datetime | None = None,
        context_tags_user: list[str] | None = None,
        context_tags_assistant: list[str] | None = None,
    ) -> str:
        """Enqueue a storage job.  Returns ``job_id``."""
        content_hash = hashlib.sha256(
            f"{tenant_id}:{user_message}:{assistant_response}:{turn_id}".encode()
        ).hexdigest()[:32]

        # Idempotency check
        dedup_key = f"{_PROCESSED_PREFIX}{content_hash}"
        if await self.redis.exists(dedup_key):
            return f"dedup:{content_hash}"

        job = StorageJob(
            job_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_message=user_message,
            assistant_response=assistant_response,
            session_id=session_id,
            turn_id=turn_id,
            timestamp=timestamp.isoformat() if timestamp else None,
            context_tags_user=context_tags_user or ["conversation", "user_input"],
            context_tags_assistant=context_tags_assistant or ["conversation", "assistant_response"],
            idempotency_key=content_hash,
        )

        await self.redis.rpush(_QUEUE_KEY, json.dumps(asdict(job)))
        return job.job_id

    async def process_next(self, orchestrator: "MemoryOrchestrator") -> bool:
        """Process the next job from the queue.

        Returns *True* if a job was processed (or de-duped), *False* if
        the queue was empty after waiting for the timeout.
        """
        raw = await self.redis.blpop(_QUEUE_KEY, timeout=5)
        if not raw:
            return False

        job_data: dict[str, Any] = json.loads(raw[1])
        idempotency_key = job_data.get("idempotency_key", "")
        dedup_key = f"{_PROCESSED_PREFIX}{idempotency_key}"

        if await self.redis.exists(dedup_key):
            return True  # Already processed — skip

        try:
            ts_str = job_data.get("timestamp")
            ts = datetime.fromisoformat(ts_str) if ts_str else None

            await orchestrator.write(
                tenant_id=job_data["tenant_id"],
                content=job_data["user_message"],
                session_id=job_data.get("session_id"),
                context_tags=job_data.get("context_tags_user", []),
                timestamp=ts,
            )

            assistant_response = job_data.get("assistant_response")
            if assistant_response:
                await orchestrator.write(
                    tenant_id=job_data["tenant_id"],
                    content=assistant_response,
                    session_id=job_data.get("session_id"),
                    context_tags=job_data.get("context_tags_assistant", []),
                    timestamp=ts,
                )

            await self.redis.setex(dedup_key, _PROCESSED_TTL, "1")
        except Exception:
            retry_count = job_data.get("_retry", 0) + 1
            if retry_count < _MAX_RETRIES:
                job_data["_retry"] = retry_count
                await self.redis.rpush(_QUEUE_KEY, json.dumps(job_data))
                logger.warning(
                    "async_storage_retry",
                    extra={"job_id": job_data.get("job_id"), "retry": retry_count},
                )
            else:
                logger.error(
                    "async_storage_dead_letter",
                    extra={"job_id": job_data.get("job_id")},
                    exc_info=True,
                )

        return True
