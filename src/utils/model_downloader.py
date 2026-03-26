"""Auto-download CML model weights from Hugging Face Hub when not found locally."""

from __future__ import annotations

import os
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


def _needs_download(models_dir: Path) -> bool:
    """Return True when the models directory is missing or lacks key artifacts."""
    if not models_dir.exists():
        return True
    # If we can find the manifest + at least the three family models, skip download.
    return any(not (models_dir / artifact).exists() for artifact in _EXPECTED_ARTIFACTS)


def ensure_models(
    models_dir: Path,
    *,
    repo_id: str | None = None,
    token: str | None = None,
    force: bool = False,
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
    except ImportError:
        logger.warning(
            "huggingface_hub_not_installed",
            extra={
                "hint": "pip install huggingface_hub  # or pip install cognitive-memory-layer[server]",
            },
        )
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
        logger.warning(
            "models_download_failed",
            extra={"repo_id": repo_id, "error": str(exc)},
        )
        return not _needs_download(models_dir)
