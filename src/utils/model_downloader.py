"""Auto-download CML model weights from Hugging Face Hub when not found locally."""

from __future__ import annotations

import os
from itertools import chain
from pathlib import Path

from .logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_HF_REPO_ID = "avinashm/CognitiveMemoryLayer-models"

# Minimum files expected after a successful download.
_EXPECTED_ARTIFACTS = (
    "manifest.json",
    "router_model.joblib",
    "extractor_model.joblib",
    "pair_model.joblib",
)

_HOST_BIND_MOUNT_PERMISSION_HINT = (
    "If this path is bind-mounted from the host, ensure the host files are writable by "
    "the user running CML (for local Docker Compose from the repo root: "
    "sudo chown -R $(id -u):$(id -g) packages/models/trained_models)."
)


def _needs_download(models_dir: Path) -> bool:
    """Return True when the models directory is missing or lacks key artifacts."""
    if not models_dir.exists():
        return True
    # If we can find the manifest + at least the three family models, skip download.
    return any(not (models_dir / artifact).exists() for artifact in _EXPECTED_ARTIFACTS)


def _first_unwritable_directory(models_dir: Path) -> Path | None:
    """Return the first existing directory under models_dir that is not writable."""
    if not models_dir.exists():
        return None

    for path in chain((models_dir,), (p for p in models_dir.rglob("*") if p.is_dir())):
        if not os.access(path, os.W_OK | os.X_OK):
            return path
    return None


def _permission_error_message(models_dir: Path, *, target: str | Path | None = None) -> str:
    target_str = str(target or models_dir)
    return (
        f"Model artifacts directory is not writable at {target_str}. "
        f"{_HOST_BIND_MOUNT_PERMISSION_HINT}"
    )


def ensure_models(
    models_dir: Path,
    *,
    repo_id: str | None = None,
    token: str | None = None,
    force: bool = False,
    raise_on_failure: bool = False,
) -> bool:
    """Download model weights from HF Hub if they are not present locally.

    Returns True if models are available (already existed or were downloaded).
    Returns False if download was skipped or failed.
    """
    repo_id = repo_id or os.environ.get("CML_MODELS_HF_REPO", DEFAULT_HF_REPO_ID)
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not force and not _needs_download(models_dir):
        logger.info(
            "models_already_present",
            extra={"models_dir": str(models_dir)},
        )
        return True

    # Check if auto-download is explicitly disabled.
    if os.environ.get("CML_MODELS_AUTO_DOWNLOAD", "true").lower() in ("0", "false", "no"):
        logger.info(
            "models_auto_download_disabled",
            extra={"models_dir": str(models_dir)},
        )
        return not _needs_download(models_dir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        logger.warning(
            "huggingface_hub_not_installed",
            extra={
                "hint": "pip install huggingface_hub  # or pip install cognitive-memory-layer[server]",
            },
        )
        if raise_on_failure:
            raise RuntimeError(
                "Model auto-download requested but huggingface_hub is not installed"
            ) from exc
        return not _needs_download(models_dir)

    unwritable_dir = _first_unwritable_directory(models_dir)
    if unwritable_dir is not None:
        error_message = _permission_error_message(models_dir, target=unwritable_dir)
        logger.warning(
            "models_dir_not_writable",
            extra={
                "models_dir": str(models_dir),
                "path": str(unwritable_dir),
                "hint": _HOST_BIND_MOUNT_PERMISSION_HINT,
            },
        )
        if raise_on_failure:
            raise RuntimeError(error_message)
        return not _needs_download(models_dir)

    logger.info(
        "models_downloading",
        extra={"repo_id": repo_id, "models_dir": str(models_dir)},
    )

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(models_dir),
            token=token,
            # Only download model artifacts, skip dataset/training files.
            allow_patterns=["*.joblib", "*.json", "*.safetensors", "*.txt", "*.bin"],
            ignore_patterns=["*.parquet", "*.csv", "*.arrow", "*.md"],
        )
        logger.info(
            "models_downloaded",
            extra={"repo_id": repo_id, "models_dir": str(models_dir)},
        )
        return True
    except Exception as exc:
        error_message = str(exc)
        if isinstance(exc, PermissionError):
            error_message = _permission_error_message(
                models_dir,
                target=getattr(exc, "filename", None) or models_dir,
            )
        logger.warning(
            "models_download_failed",
            extra={"repo_id": repo_id, "error": error_message},
        )
        if raise_on_failure:
            raise RuntimeError(
                f"Unable to download CML model artifacts into {models_dir}: {error_message}"
            ) from exc
        return not _needs_download(models_dir)
