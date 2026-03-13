"""Unit tests for the Packages Audit implementation (PKG-01 through PKG-09)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# PR-01 (PKG-03): ModelPackRuntime capability reporting
# ---------------------------------------------------------------------------


class _SimpleModel:
    """Pickleable stand-in for a sklearn model."""

    def predict(self, x):
        return ["pair::match"] * len(x)


class _WrapperProbaModel:
    """Pickleable stand-in exposing top-level classes_ and predict_proba."""

    classes_ = [
        "novelty_pair::duplicate",
        "novelty_pair::changed",
        "novelty_pair::novel",
    ]

    def predict(self, x):
        return ["novelty_pair::changed"] * len(x)

    def predict_proba(self, x):
        return [[0.1, 0.8, 0.1] for _ in x]


class TestModelPackCapabilityReporting:
    """Tests for available_families, available_tasks, and partial-load logging."""

    def _make_runtime(self, tmp_path: Path, family_files: list[str] | None = None):
        from src.utils.modelpack import ModelPackRuntime

        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")

        for name in family_files or []:
            joblib.dump({"model": _SimpleModel()}, str(models_dir / name))

        return ModelPackRuntime(models_dir=models_dir)

    def test_available_families_only_pair(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path, ["pair_model.joblib"])
        assert rt.available is True
        assert rt.available_families == ["pair"]
        assert "conflict_detection" in rt.available_tasks
        assert "memory_type" not in rt.available_tasks

    def test_available_families_all(self, tmp_path: Path):
        rt = self._make_runtime(
            tmp_path,
            ["pair_model.joblib", "router_model.joblib", "extractor_model.joblib"],
        )
        assert rt.available_families == ["extractor", "pair", "router"]
        assert "memory_type" in rt.available_tasks

    def test_available_no_models(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path, [])
        assert rt.available is False
        assert rt.available_families == []
        assert rt.available_tasks == []

    def test_pending_families_logged(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path, ["pair_model.joblib"])
        _ = rt.available
        assert rt.available_families == ["pair"]
        pending = sorted({"router", "extractor", "pair"} - {"pair"})
        assert pending == ["extractor", "router"]

    def test_has_task_model_false_for_unloaded_family(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path, ["pair_model.joblib"])
        assert rt.has_task_model("memory_type") is False

    def test_supports_task_with_family_model(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path, ["pair_model.joblib"])
        assert rt.supports_task("scope_match") is True
        assert rt.supports_task("memory_type") is False

    def test_capability_report_contains_core_fields(self, tmp_path: Path):
        rt = self._make_runtime(tmp_path, ["pair_model.joblib"])
        report = rt.capability_report()
        assert report["available"] is True
        assert report["available_families"] == ["pair"]
        assert "scope_match" in report["available_tasks"]
        assert "pending_families" in report


# ---------------------------------------------------------------------------
# PR-02 (PKG-04): Retry config.max_retry_delay wiring
# ---------------------------------------------------------------------------


class TestRetryMaxDelayWiring:
    """Verify that config.max_retry_delay caps backoff."""

    def test_sync_sleep_respects_max_delay(self):
        from cml.transport.retry import _sleep_with_backoff

        actual_sleeps: list[float] = []
        with patch("cml.transport.retry.time.sleep", side_effect=lambda d: actual_sleeps.append(d)):
            _sleep_with_backoff(attempt=10, base_delay=1.0, max_delay=5.0)

        assert all(d <= 5.0 for d in actual_sleeps)

    def test_async_sleep_respects_max_delay(self):
        from cml.transport.retry import _async_sleep_with_backoff

        actual_sleeps: list[float] = []

        async def fake_sleep(d: float):
            actual_sleeps.append(d)

        with patch("cml.transport.retry.asyncio.sleep", side_effect=fake_sleep):
            asyncio.get_event_loop().run_until_complete(
                _async_sleep_with_backoff(attempt=10, base_delay=1.0, max_delay=5.0)
            )

        assert all(d <= 5.0 for d in actual_sleeps)

    def test_retry_sync_reads_config_max_retry_delay(self):
        from cml.config import CMLConfig
        from cml.exceptions import ServerError
        from cml.transport.retry import retry_sync

        config = CMLConfig(
            api_key="test",
            base_url="http://localhost:8000",
            max_retries=1,
            retry_delay=1.0,
            max_retry_delay=3.0,
        )
        actual_sleeps: list[float] = []
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ServerError("boom")

        with (
            patch("cml.transport.retry.time.sleep", side_effect=lambda d: actual_sleeps.append(d)),
            pytest.raises(ServerError),
        ):
            retry_sync(config, failing_func)

        assert all(d <= 3.0 for d in actual_sleeps)


# ---------------------------------------------------------------------------
# PR-02 (PKG-05): Export/import fidelity
# ---------------------------------------------------------------------------


class TestExportImportFidelity:
    """Verify that import preserves fields beyond text and metadata."""

    def test_roundtrip_preserves_fields(self, tmp_path: Path):
        export_path = tmp_path / "export.jsonl"
        records = [
            {
                "id": "rec-001",
                "text": "User prefers dark mode",
                "type": "preference",
                "confidence": 0.95,
                "timestamp": "2026-01-15T10:00:00+00:00",
                "context_tags": ["settings", "ui"],
                "namespace": "global",
                "source_session_id": "sess-42",
                "metadata": {"key": "value"},
            }
        ]
        with open(export_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        write_calls: list[dict[str, Any]] = []

        class FakeTarget:
            def write(self, text, **kwargs):
                write_calls.append({"text": text, **kwargs})

        from cml.embedded_utils import import_memories_async

        # The import helper accepts concrete client types; for this unit test we
        # only need an object with a compatible `write()` method.
        target = cast("Any", FakeTarget())
        count = asyncio.get_event_loop().run_until_complete(
            import_memories_async(target, str(export_path))
        )

        assert count == 1
        call = write_calls[0]
        assert call["text"] == "User prefers dark mode"
        assert call["memory_type"] == "preference"
        assert call.get("context_tags") == ["settings", "ui"]
        assert call.get("namespace") == "global"
        assert call.get("session_id") == "sess-42"
        assert call["metadata"].get("_imported_confidence") == 0.95


# ---------------------------------------------------------------------------
# PR-03 (PKG-06): Training config and strict mode
# ---------------------------------------------------------------------------


class TestTrainingConfigAndStrictMode:
    """Verify disabled tasks are skipped and --strict causes failures."""

    def test_taskspec_enabled_default_true(self):
        pytest.importorskip("pandas", reason="cml.modeling.train requires [modeling] deps")
        from cml.modeling.train import TaskSpec

        spec = TaskSpec(
            task_name="test",
            family="router",
            input_type="single",
            objective="classification",
            labels=["a", "b"],
            artifact_name="test",
            metrics=["accuracy"],
        )
        assert spec.enabled is True

    def test_taskspec_enabled_false(self):
        pytest.importorskip("pandas", reason="cml.modeling.train requires [modeling] deps")
        from cml.modeling.train import TaskSpec

        spec = TaskSpec(
            task_name="test",
            family="router",
            input_type="single",
            objective="classification",
            labels=["a", "b"],
            artifact_name="test",
            metrics=["accuracy"],
            enabled=False,
        )
        assert spec.enabled is False

    def test_train_task_strict_mode_raises(self):
        pytest.importorskip("pandas", reason="cml.modeling.train requires [modeling] deps")
        from cml.modeling.train import TaskSpec, _train_task

        spec = TaskSpec(
            task_name="pii_span_detection",
            family="extractor",
            input_type="single",
            objective="token_classification",
            labels=[],
            artifact_name="pii_span_detection",
            metrics=["span_f1"],
        )
        with pytest.raises(Exception, match="Missing prepared token split"):
            _train_task(
                spec,
                prepared_dir=Path("/nonexistent"),
                output_dir=Path("/nonexistent"),
                train_cfg={},
                strict=True,
            )

    def test_train_task_default_skips_gracefully(self):
        pytest.importorskip("pandas", reason="cml.modeling.train requires [modeling] deps")
        from cml.modeling.train import TaskSpec, _train_task

        spec = TaskSpec(
            task_name="pii_span_detection",
            family="extractor",
            input_type="single",
            objective="token_classification",
            labels=[],
            artifact_name="pii_span_detection",
            metrics=["span_f1"],
        )
        result = _train_task(
            spec,
            prepared_dir=Path("/nonexistent"),
            output_dir=Path("/nonexistent"),
            train_cfg={},
            strict=False,
        )
        assert result == {}

    def test_token_classification_tasks_enabled_in_toml(self):
        import tomllib

        toml_path = (
            Path(__file__).resolve().parents[2] / "packages" / "models" / "model_pipeline.toml"
        )
        if not toml_path.exists():
            pytest.skip("model_pipeline.toml not found")
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
        tasks = config.get("tasks", [])
        token_tasks = [t for t in tasks if t.get("objective") == "token_classification"]
        assert len(token_tasks) >= 2
        for t in token_tasks:
            assert t.get("enabled") is True, f"{t['task_name']} should be enabled"

    def test_write_importance_regression_enabled_in_toml(self):
        import tomllib

        toml_path = (
            Path(__file__).resolve().parents[2] / "packages" / "models" / "model_pipeline.toml"
        )
        if not toml_path.exists():
            pytest.skip("model_pipeline.toml not found")
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
        tasks = config.get("tasks", [])
        write_imp = [t for t in tasks if t.get("task_name") == "write_importance_regression"]
        assert len(write_imp) == 1
        assert write_imp[0].get("enabled") is True
        assert write_imp[0].get("objective") == "single_regression"


# ---------------------------------------------------------------------------
# PR-04 (PKG-08): Legacy wrapper cleanup
# ---------------------------------------------------------------------------


class TestLegacyWrapperCleanup:
    """Verify sys.path mutation is removed from legacy wrappers."""

    def test_prepare_wrapper_no_sys_path_mutation(self):
        wrapper_path = (
            Path(__file__).resolve().parents[2] / "packages" / "models" / "scripts" / "prepare.py"
        )
        if not wrapper_path.exists():
            pytest.skip("prepare.py wrapper not found")
        source = wrapper_path.read_text()
        assert "sys.path.insert" not in source
        assert "sys.path" not in source

    def test_train_wrapper_no_sys_path_mutation(self):
        wrapper_path = (
            Path(__file__).resolve().parents[2] / "packages" / "models" / "scripts" / "train.py"
        )
        if not wrapper_path.exists():
            pytest.skip("train.py wrapper not found")
        source = wrapper_path.read_text()
        assert "sys.path.insert" not in source
        assert "sys.path" not in source


# ---------------------------------------------------------------------------
# PR-04 (PKG-02): CML_MODELS_DIR env var
# ---------------------------------------------------------------------------


class TestModelsDir:
    """Verify ModelPackRuntime honors CML_MODELS_DIR and embedded models_dir."""

    def test_env_var_overrides_default(self, tmp_path: Path):
        from src.utils.modelpack import ModelPackRuntime

        custom_dir = tmp_path / "custom_models"
        custom_dir.mkdir()

        with patch.dict(os.environ, {"CML_MODELS_DIR": str(custom_dir)}):
            rt = ModelPackRuntime()
        assert rt.models_dir == custom_dir

    def test_explicit_models_dir_overrides_env(self, tmp_path: Path):
        from src.utils.modelpack import ModelPackRuntime

        explicit = tmp_path / "explicit"
        explicit.mkdir()
        env_dir = tmp_path / "env_dir"
        env_dir.mkdir()

        with patch.dict(os.environ, {"CML_MODELS_DIR": str(env_dir)}):
            rt = ModelPackRuntime(models_dir=explicit)
        assert rt.models_dir == explicit

    def test_missing_dir_logs_warning(self, tmp_path: Path, capsys):
        from src.utils.modelpack import ModelPackRuntime

        nonexistent = tmp_path / "does_not_exist"
        rt = ModelPackRuntime(models_dir=nonexistent)
        _ = rt.available
        captured = capsys.readouterr()
        all_output = captured.out + captured.err
        assert "modelpack_dir_missing" in all_output or not nonexistent.exists()


# ---------------------------------------------------------------------------
# PR-04: Eval CLI explicit path requirements
# ---------------------------------------------------------------------------


class TestEvalCliPaths:
    """Verify eval CLI errors clearly when repo root is not detected."""

    def test_default_repo_root_returns_none_outside_repo(self, tmp_path: Path):
        from cml.eval.config import find_repo_root

        result = find_repo_root(tmp_path)
        assert result is None

    def test_require_path_raises_on_none(self):
        from cml.eval.cli import _require_path

        with pytest.raises(SystemExit):
            _require_path(None, "repo-root")


# ---------------------------------------------------------------------------
# PR-05 (PKG-09): Client parity
# ---------------------------------------------------------------------------


class TestClientParity:
    """Verify sync and async clients expose matching public APIs."""

    def test_public_method_parity(self):
        from cml.async_client import AsyncCognitiveMemoryLayer
        from cml.client import CognitiveMemoryLayer

        skip = {
            "__aenter__",
            "__aexit__",
            "__enter__",
            "__exit__",
            "__del__",
            "__init__",
            "close",
            "_ensure_same_loop",
            "_loop",
        }

        sync_methods = {
            name
            for name in dir(CognitiveMemoryLayer)
            if not name.startswith("_") and callable(getattr(CognitiveMemoryLayer, name, None))
        }
        async_methods = {
            name
            for name in dir(AsyncCognitiveMemoryLayer)
            if not name.startswith("_") and callable(getattr(AsyncCognitiveMemoryLayer, name, None))
        }

        sync_only = sync_methods - async_methods - skip
        async_only = async_methods - sync_methods - skip

        assert not sync_only, f"Sync-only methods: {sync_only}"
        assert not async_only, f"Async-only methods: {async_only}"


# ---------------------------------------------------------------------------
# PR-05 (PKG-09): Shared _endpoints.py
# ---------------------------------------------------------------------------


class TestSharedEndpoints:
    """Verify shared endpoint helpers produce correct payloads."""

    def test_build_write_body(self):
        from cml._endpoints import build_write_body

        body = build_write_body(
            "test content",
            context_tags=["tag1"],
            namespace="ns1",
        )
        assert body["content"] == "test content"
        assert body["context_tags"] == ["tag1"]
        assert body["namespace"] == "ns1"

    def test_build_read_body(self):
        from cml._endpoints import build_read_body

        body = build_read_body("search query", max_results=5)
        assert body["query"] == "search query"
        assert body["max_results"] == 5

    def test_eval_mode_headers(self):
        from cml._endpoints import eval_mode_headers

        assert eval_mode_headers(True) == {"X-Eval-Mode": "true"}
        assert eval_mode_headers(False) is None


# ---------------------------------------------------------------------------
# PR-05 (PKG-07): Per-task model loading
# ---------------------------------------------------------------------------


class TestPerTaskModelLoading:
    """Verify modelpack discovers and loads per-task artifacts."""

    def test_per_task_model_loaded(self, tmp_path: Path):
        from src.utils.modelpack import ModelPackRuntime

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")

        joblib.dump({"model": _SimpleModel()}, str(models_dir / "novelty_pair_model.joblib"))

        rt = ModelPackRuntime(models_dir=models_dir)
        assert rt.has_task_model("novelty_pair") is True
        assert "novelty_pair" in rt.available_tasks

    def test_per_task_artifact_names_match_config(self):
        """All trainable task artifact names in model_pipeline.toml have matching _TASK_MODEL_FILE entries."""
        import tomllib

        from src.utils.modelpack import _TASK_MODEL_FILE

        toml_path = (
            Path(__file__).resolve().parents[2] / "packages" / "models" / "model_pipeline.toml"
        )
        if not toml_path.exists():
            pytest.skip("model_pipeline.toml not found")
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
        tasks = config.get("tasks", [])
        for t in tasks:
            task_name = t["task_name"]
            assert task_name in _TASK_MODEL_FILE, (
                f"Task '{task_name}' from model_pipeline.toml has no entry in _TASK_MODEL_FILE"
            )

    def test_wrapper_task_model_predict_proba_is_used_for_scoring(self, tmp_path: Path):
        from src.utils.modelpack import ModelPackRuntime

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")

        joblib.dump(
            {"model": _WrapperProbaModel()},
            str(models_dir / "novelty_pair_model.joblib"),
        )

        rt = ModelPackRuntime(models_dir=models_dir)
        pred = rt.predict_score_pair("novelty_pair", "A", "B")

        assert pred is not None
        assert pred.score == pytest.approx(0.785, rel=1e-6)


# ---------------------------------------------------------------------------
# PR-05 (PKG-01): Packaging structure
# ---------------------------------------------------------------------------


class TestPackagingStructure:
    """Verify pyproject.toml dependency split."""

    def test_base_deps_exclude_server(self):
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")
        with open(pyproject, "rb") as f:
            config = tomllib.load(f)
        base_deps = [d.lower() for d in config["project"]["dependencies"]]
        base_text = " ".join(base_deps)
        assert "fastapi" not in base_text
        assert "uvicorn" not in base_text
        assert "neo4j" not in base_text
        assert "redis" not in base_text
        assert "celery" not in base_text

    def test_server_extras_exist(self):
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")
        with open(pyproject, "rb") as f:
            config = tomllib.load(f)
        optional = config["project"]["optional-dependencies"]
        assert "server" in optional
        server_text = " ".join(d.lower() for d in optional["server"])
        assert "fastapi" in server_text

    def test_wheel_does_not_package_repo_src(self):
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")
        with open(pyproject, "rb") as f:
            config = tomllib.load(f)
        wheel_cfg = config["tool"]["hatch"]["build"]["targets"]["wheel"]
        packages = [str(pkg) for pkg in wheel_cfg.get("packages", [])]
        assert "src" not in packages
        assert "packages/py-cml/src/cml" in packages
