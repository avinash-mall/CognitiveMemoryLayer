"""Unit tests for Celery forgetting task and beat schedule."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("celery")

import src.celery_app as celery_app
from src.celery_app import app, run_forgetting_task


def _run_coro(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _reset_celery_thread_state():
    for attr in ("loop", "db_manager"):
        if hasattr(celery_app._thread_local, attr):
            value = getattr(celery_app._thread_local, attr)
            if attr == "loop" and value is not None and not value.is_closed():
                value.close()
            delattr(celery_app._thread_local, attr)
    yield
    for attr in ("loop", "db_manager"):
        if hasattr(celery_app._thread_local, attr):
            value = getattr(celery_app._thread_local, attr)
            if attr == "loop" and value is not None and not value.is_closed():
                value.close()
            delattr(celery_app._thread_local, attr)


class TestCeleryForgettingTask:
    def test_task_registered(self):
        """Forgetting task is registered on the Celery app."""
        task_name = "src.celery_app.run_forgetting_task"
        assert app.tasks.get(task_name) is not None

    def test_task_has_correct_name(self):
        """Task is bound with expected name."""
        assert run_forgetting_task.name == "src.celery_app.run_forgetting_task"

    def test_beat_schedule_includes_forgetting(self):
        """Beat schedule runs fan-out daily; fan-out dispatches run_forgetting_task per tenant."""
        assert "forgetting-daily" in (app.conf.beat_schedule or {})
        entry = app.conf.beat_schedule["forgetting-daily"]
        assert entry["task"] == "src.celery_app.fan_out_forgetting"
        assert entry["schedule"] == 86400.0

    def test_get_or_create_event_loop_reuses_and_replaces_closed_loop(self):
        loop1 = celery_app._get_or_create_event_loop()
        loop2 = celery_app._get_or_create_event_loop()

        assert loop1 is loop2

        loop1.close()
        loop3 = celery_app._get_or_create_event_loop()

        assert loop3 is not loop1
        loop3.close()

    def test_run_async_uses_persistent_loop(self, monkeypatch: pytest.MonkeyPatch):
        seen = {}

        class FakeLoop:
            def run_until_complete(self, coro):
                seen["result"] = _run_coro(coro)
                return seen["result"]

        monkeypatch.setattr(celery_app, "_get_or_create_event_loop", lambda: FakeLoop())

        async def _coro() -> str:
            return "done"

        assert celery_app._run_async(_coro()) == "done"
        assert seen["result"] == "done"

    def test_get_or_create_db_manager_caches_manager_per_thread(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        created: list[object] = []

        class FakeDB:
            pg_session_factory = object()

        async def _create():
            db = FakeDB()
            created.append(db)
            return db

        monkeypatch.setattr("src.storage.connection.DatabaseManager.create", _create)

        db1 = _run_coro(celery_app._get_or_create_db_manager())
        db2 = _run_coro(celery_app._get_or_create_db_manager())
        assert db1 is db2
        assert len(created) == 1

        celery_app._thread_local.db_manager = SimpleNamespace(pg_session_factory=None)
        db3 = _run_coro(celery_app._get_or_create_db_manager())
        assert db3 is created[-1]
        assert len(created) == 2

    def test_get_configured_summarizer_backend_handles_provider_switches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        non_hf_settings = SimpleNamespace(
            summarizer_internal=SimpleNamespace(provider="mock"),
        )
        monkeypatch.setattr(celery_app, "get_settings", lambda: non_hf_settings)
        assert celery_app._get_configured_summarizer_backend() is None

        hf_settings = SimpleNamespace(
            summarizer_internal=SimpleNamespace(
                provider="huggingface",
                model="summ-model",
                task="summarization",
                max_input_chars=10,
                max_output_chars=20,
                min_length=5,
                max_length=15,
                device="cpu",
            )
        )
        monkeypatch.setattr(celery_app, "get_settings", lambda: hf_settings)
        monkeypatch.setattr(
            "src.utils.hf_summarizer.get_hf_summarizer",
            lambda **kwargs: kwargs,
        )

        backend = celery_app._get_configured_summarizer_backend()
        assert backend["model"] == "summ-model"
        assert backend["device"] == "cpu"

    def test_get_all_tenant_user_pairs_filters_blank_tenants(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class SessionCM:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def execute(self, _query):
                return SimpleNamespace(
                    scalars=lambda: SimpleNamespace(all=lambda: ["tenant-a", "", None, "tenant-b"])
                )

        async def _get_db():
            return SimpleNamespace(pg_session_factory=lambda: SessionCM())

        monkeypatch.setattr(celery_app, "_get_or_create_db_manager", _get_db)
        monkeypatch.setattr(celery_app, "_run_async", lambda coro: _run_coro(coro))

        assert celery_app._get_all_tenant_user_pairs() == [
            ("tenant-a", "tenant-a"),
            ("tenant-b", "tenant-b"),
        ]

    def test_fan_out_forgetting_dispatches_one_task_per_tenant(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            celery_app,
            "_get_all_tenant_user_pairs",
            lambda: [("tenant-a", "tenant-a"), ("tenant-b", "tenant-b")],
        )
        monkeypatch.setattr(
            celery_app.run_forgetting_task,
            "delay",
            lambda tenant_id, user_id: calls.append((tenant_id, user_id)),
        )

        celery_app.fan_out_forgetting()

        assert calls == [("tenant-a", "tenant-a"), ("tenant-b", "tenant-b")]

    def test_run_forgetting_task_returns_worker_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        report = SimpleNamespace(
            tenant_id="tenant-a",
            user_id="tenant-a",
            memories_scanned=12,
            memories_scored=10,
            duplicates_found=1,
            elapsed_seconds=0.75,
            result=SimpleNamespace(
                operations_planned=5,
                operations_applied=4,
                deleted=1,
                decayed=2,
                silenced=1,
                compressed=0,
                errors=[],
            ),
        )
        captured: dict[str, object] = {}

        class FakeWorker:
            def __init__(self, store, compression_summarizer=None):
                captured["store"] = store
                captured["summarizer"] = compression_summarizer

            async def run_forgetting(self, **kwargs):
                captured["kwargs"] = kwargs
                return report

        async def _get_db():
            return SimpleNamespace(pg_session_factory="factory")

        monkeypatch.setattr(celery_app, "_get_or_create_db_manager", _get_db)
        monkeypatch.setattr(celery_app, "_get_configured_summarizer_backend", lambda: "backend")
        monkeypatch.setattr(celery_app, "_run_async", lambda coro: _run_coro(coro))
        monkeypatch.setattr(
            "src.storage.postgres.PostgresMemoryStore",
            lambda factory: ("store", factory),
        )
        monkeypatch.setattr("src.forgetting.worker.ForgettingWorker", FakeWorker)

        result = celery_app.run_forgetting_task.run(
            "tenant-a",
            "tenant-a",
            dry_run=True,
            max_memories=25,
        )

        assert result["tenant_id"] == "tenant-a"
        assert result["operations_applied"] == 4
        assert captured["store"] == ("store", "factory")
        assert captured["summarizer"] == "backend"
        assert captured["kwargs"] == {
            "tenant_id": "tenant-a",
            "user_id": "tenant-a",
            "max_memories": 25,
            "dry_run": True,
        }
