from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.dashboard import models_routes


@pytest.mark.asyncio
async def test_dashboard_models_status_returns_modelpack_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modelpack = SimpleNamespace(
        available=True,
        _models={"family-b": object(), "family-a": object()},
        _task_models={"task-b": object(), "task-a": object()},
        _load_errors=["load warning"],
        models_dir="/tmp/models",
    )
    monkeypatch.setattr("src.utils.modelpack.get_modelpack_runtime", lambda: modelpack)

    result = await models_routes.dashboard_models_status(auth=SimpleNamespace(tenant_id="tenant-a"))

    assert result.available is True
    assert result.families == ["family-a", "family-b"]
    assert result.task_models == ["task-a", "task-b"]
    assert result.load_errors == ["load warning"]
    assert result.models_dir == "/tmp/models"


@pytest.mark.asyncio
async def test_dashboard_models_status_handles_modelpack_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise():
        raise RuntimeError("modelpack boom")

    monkeypatch.setattr("src.utils.modelpack.get_modelpack_runtime", _raise)

    result = await models_routes.dashboard_models_status(auth=SimpleNamespace(tenant_id="tenant-a"))

    assert result.available is False
    assert "modelpack boom" in result.load_errors[0]
