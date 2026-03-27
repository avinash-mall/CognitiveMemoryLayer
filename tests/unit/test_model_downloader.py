"""Unit tests for src/utils/model_downloader.py."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.utils.model_downloader import (
    DEFAULT_HF_REPO_ID,
    _needs_download,
    _permission_error_message,
    ensure_models,
)

_ARTIFACTS = ("manifest.json", "router_model.joblib", "extractor_model.joblib", "pair_model.joblib")


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def populated_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    for name in _ARTIFACTS:
        (d / name).write_text("{}")
    return d


class TestNeedsDownload:
    def test_missing_dir(self, empty_dir: Path) -> None:
        assert _needs_download(empty_dir) is True

    def test_empty_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "models"
        d.mkdir()
        assert _needs_download(d) is True

    def test_partial_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "models"
        d.mkdir()
        (d / "manifest.json").write_text("{}")
        assert _needs_download(d) is True

    def test_complete_dir(self, populated_dir: Path) -> None:
        assert _needs_download(populated_dir) is False


class TestEnsureModels:
    def test_already_present_skips_download(self, populated_dir: Path) -> None:
        assert ensure_models(populated_dir) is True

    def test_disabled_via_env(self, empty_dir: Path) -> None:
        with patch.dict("os.environ", {"CML_MODELS_AUTO_DOWNLOAD": "false"}):
            assert ensure_models(empty_dir) is False

    def test_missing_huggingface_hub_returns_false(self, empty_dir: Path) -> None:
        """When huggingface_hub is not installed, ensure_models returns False."""
        # Temporarily remove huggingface_hub from sys.modules to simulate import error
        saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = None  # type: ignore[assignment]
        try:
            # Should not crash regardless of whether huggingface_hub is installed
            ensure_models(empty_dir)
        finally:
            if saved is not None:
                sys.modules["huggingface_hub"] = saved
            else:
                sys.modules.pop("huggingface_hub", None)

    def test_successful_download(self, empty_dir: Path) -> None:
        """Mocks snapshot_download to create files and verify success."""
        mock_module = ModuleType("huggingface_hub")

        def fake_download(**kwargs: object) -> None:
            d = Path(str(kwargs["local_dir"]))
            d.mkdir(parents=True, exist_ok=True)
            for name in _ARTIFACTS:
                (d / name).write_text("{}")

        mock_module.snapshot_download = fake_download  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            result = ensure_models(empty_dir)
        assert result is True
        assert (empty_dir / "manifest.json").exists()

    def test_download_failure_returns_false(self, empty_dir: Path) -> None:
        mock_module = ModuleType("huggingface_hub")
        mock_module.snapshot_download = MagicMock(side_effect=RuntimeError("network error"))  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            result = ensure_models(empty_dir)
        assert result is False

    def test_download_failure_raises_when_requested(self, empty_dir: Path) -> None:
        mock_module = ModuleType("huggingface_hub")
        mock_module.snapshot_download = MagicMock(side_effect=RuntimeError("network error"))  # type: ignore[attr-defined]

        with (
            patch.dict(sys.modules, {"huggingface_hub": mock_module}),
            pytest.raises(RuntimeError, match="Unable to download CML model artifacts"),
        ):
            ensure_models(empty_dir, raise_on_failure=True)

    def test_permission_failure_returns_false(self, empty_dir: Path) -> None:
        mock_module = ModuleType("huggingface_hub")
        mock_module.snapshot_download = MagicMock(  # type: ignore[attr-defined]
            side_effect=PermissionError(13, "Permission denied", "/tmp/models/config.json")
        )

        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            result = ensure_models(empty_dir)
        assert result is False

    def test_permission_failure_raises_with_actionable_message(self, empty_dir: Path) -> None:
        mock_module = ModuleType("huggingface_hub")
        mock_module.snapshot_download = MagicMock(  # type: ignore[attr-defined]
            side_effect=PermissionError(13, "Permission denied", "/tmp/models/config.json")
        )

        with (
            patch.dict(sys.modules, {"huggingface_hub": mock_module}),
            pytest.raises(RuntimeError, match="bind-mounted from the host"),
        ):
            ensure_models(empty_dir, raise_on_failure=True)

    def test_disabled_auto_download_does_not_raise_when_requested(self, empty_dir: Path) -> None:
        with patch.dict("os.environ", {"CML_MODELS_AUTO_DOWNLOAD": "false"}):
            assert ensure_models(empty_dir, raise_on_failure=True) is False

    def test_force_redownload(self, tmp_path: Path) -> None:
        d = tmp_path / "models"
        d.mkdir()
        for name in _ARTIFACTS:
            (d / name).write_text("{}")

        download_called = False
        mock_module = ModuleType("huggingface_hub")

        def fake_download(**kwargs: object) -> None:
            nonlocal download_called
            download_called = True

        mock_module.snapshot_download = fake_download  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"huggingface_hub": mock_module}):
            ensure_models(d, force=True)
        assert download_called is True

    def test_default_repo_id(self) -> None:
        assert DEFAULT_HF_REPO_ID == "avinashm/CognitiveMemoryLayer-models"

    def test_custom_repo_via_env(self, tmp_path: Path) -> None:
        d = tmp_path / "models"
        d.mkdir()
        for name in _ARTIFACTS:
            (d / name).write_text("{}")
        with patch.dict("os.environ", {"CML_MODELS_HF_REPO": "custom/repo"}):
            assert ensure_models(d) is True


def test_permission_error_message_includes_fix_hint(tmp_path: Path) -> None:
    message = _permission_error_message(tmp_path / "models", target="/tmp/models/config.json")

    assert "/tmp/models/config.json" in message
    assert "sudo chown -R $(id -u):$(id -g) packages/models/trained_models" in message
