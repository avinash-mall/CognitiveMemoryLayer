"""Unit tests for Phase 8: Celery forgetting task."""

from src.celery_app import app, run_forgetting_task


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
