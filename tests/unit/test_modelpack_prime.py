"""Tests for startup priming of the cached modelpack runtime."""

from __future__ import annotations

import sys
from types import ModuleType

from src.utils.modelpack import ModelPackRuntime, prime_modelpack_runtime


def test_modelpack_runtime_resets_loaded_runtime_state(monkeypatch, tmp_path) -> None:
    class _DummyRuntimeModel:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset_runtime_state(self) -> None:
            self.reset_calls += 1

    model = _DummyRuntimeModel()
    joblib = ModuleType("joblib")
    joblib.load = lambda _path: {"model": model}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joblib", joblib)
    monkeypatch.setenv("CML_MODELS_AUTO_DOWNLOAD", "false")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "constraint_dimension_model.joblib").write_text("placeholder", encoding="utf-8")

    runtime = ModelPackRuntime(models_dir=models_dir)

    assert runtime.available is True
    assert model.reset_calls == 1


def test_prime_modelpack_runtime_forwards_fail_fast_flag(monkeypatch) -> None:
    calls: list[bool] = []

    class _DummyRuntime:
        def prime(self, *, fail_on_bootstrap_error: bool = False):
            calls.append(fail_on_bootstrap_error)
            return self

    runtime = _DummyRuntime()
    monkeypatch.setattr("src.utils.modelpack.get_modelpack_runtime", lambda: runtime)

    assert prime_modelpack_runtime(fail_on_bootstrap_error=True) is runtime
    assert calls == [True]
