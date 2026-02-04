"""Celery application and background tasks (e.g. active forgetting)."""
import asyncio
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
# Beat schedule: run forgetting daily per user (in production, iterate over registered users)
app.conf.beat_schedule = {
    "forgetting-daily": {
        "task": "src.celery_app.run_forgetting_task",
        "schedule": 86400.0,  # 24 hours in seconds
        "options": {"queue": "forgetting"},
    },
}
app.conf.task_routes = {
    "src.celery_app.run_forgetting_task": {"queue": "forgetting"},
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
    """
    from .forgetting.worker import ForgettingWorker
    from .storage.connection import DatabaseManager
    from .storage.postgres import PostgresMemoryStore

    db = DatabaseManager.get_instance()
    store = PostgresMemoryStore(db.pg_session_factory)
    worker = ForgettingWorker(store)

    async def _run() -> Dict[str, Any]:
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

    return asyncio.run(_run())
