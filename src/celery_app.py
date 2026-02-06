"""Celery application and background tasks (e.g. active forgetting)."""

import asyncio
import threading
from typing import Any, Dict

from celery import Celery

from .core.config import get_settings


def _make_celery() -> Celery:
    settings = get_settings()
    return Celery(
        "cognitive_memory_layer",
        broker=settings.database.redis_url,
        backend=settings.database.redis_url,
        include=["src.celery_app"],
    )


app = _make_celery()
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

# --- Persistent event loop per worker thread (MED-13/14) ---
_thread_local = threading.local()


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent event loop for this worker thread."""
    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_local.loop = loop
    return loop


def _run_async(coro):
    """Run an async coroutine on the persistent per-thread event loop."""
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(coro)


def _get_all_tenant_user_pairs() -> list[tuple[str, str]]:
    """Discover all tenant (and user) pairs from the memory store for fan-out."""
    from sqlalchemy import distinct, select

    from .storage.connection import DatabaseManager
    from .storage.models import MemoryRecordModel

    async def _fetch():
        db = DatabaseManager.get_instance()
        pairs = []
        async with db.pg_session_factory() as session:
            r = await session.execute(select(distinct(MemoryRecordModel.tenant_id)))
            for row in r.scalars().all():
                tid = row
                if tid:
                    pairs.append((tid, tid))  # user_id same as tenant in holistic model
        return pairs

    return _run_async(_fetch())


@app.task(name="src.celery_app.fan_out_forgetting")
def fan_out_forgetting() -> None:
    """Discover all tenants and dispatch individual forgetting tasks."""
    for tenant_id, user_id in _get_all_tenant_user_pairs():
        run_forgetting_task.delay(tenant_id, user_id)


# Beat schedule: run forgetting daily via fan-out
app.conf.beat_schedule = {
    "forgetting-daily": {
        "task": "src.celery_app.fan_out_forgetting",
        "schedule": 86400.0,  # 24 hours in seconds
        "options": {"queue": "forgetting"},
    },
}
app.conf.task_routes = {
    "src.celery_app.run_forgetting_task": {"queue": "forgetting"},
    "src.celery_app.fan_out_forgetting": {"queue": "forgetting"},
}


@app.task(name="src.celery_app.run_forgetting_task", bind=True)
def run_forgetting_task(
    self: Any,
    tenant_id: str,
    user_id: str,
    dry_run: bool = False,
    max_memories: int = 5000,
) -> Dict[str, Any]:
    """
    Celery task: run active forgetting for a tenant/user.
    Call from API or Beat; runs in worker process with async bridge.
    Uses persistent event loop to enable connection pool reuse.
    """

    async def _run() -> Dict[str, Any]:
        # Create DB manager inside async context to ensure connections
        # are bound to the correct event loop (MED-14)
        from .forgetting.worker import ForgettingWorker
        from .storage.connection import DatabaseManager
        from .storage.postgres import PostgresMemoryStore

        db = DatabaseManager.get_instance()
        store = PostgresMemoryStore(db.pg_session_factory)
        worker = ForgettingWorker(store)

        report = await worker.run_forgetting(
            tenant_id=tenant_id,
            user_id=user_id,
            max_memories=max_memories,
            dry_run=dry_run,
        )
        return {
            "tenant_id": report.tenant_id,
            "user_id": report.user_id,
            "memories_scanned": report.memories_scanned,
            "memories_scored": report.memories_scored,
            "operations_planned": report.result.operations_planned,
            "operations_applied": report.result.operations_applied,
            "deleted": report.result.deleted,
            "decayed": report.result.decayed,
            "silenced": report.result.silenced,
            "compressed": report.result.compressed,
            "duplicates_found": report.duplicates_found,
            "elapsed_seconds": report.elapsed_seconds,
            "errors": report.result.errors,
        }

    return _run_async(_run())
