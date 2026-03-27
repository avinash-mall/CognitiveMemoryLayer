"""Helpers for resolving runtime device placement for local HF-backed models."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL_RUNTIME_DEVICE = "cpu"
_ALLOWED_MODEL_RUNTIME_DEVICES = frozenset({"auto", "cpu", "cuda"})


def get_model_runtime_device_preference() -> str:
    """Return the configured runtime device preference for modelpack inference."""
    raw_value = os.environ.get("CML_MODELS_DEVICE", DEFAULT_MODEL_RUNTIME_DEVICE)
    normalized = str(raw_value).strip().lower()
    if normalized in _ALLOWED_MODEL_RUNTIME_DEVICES:
        return normalized
    return DEFAULT_MODEL_RUNTIME_DEVICE


def resolve_runtime_device_name(
    torch_like: Any,
    *,
    preference: str | None = None,
) -> str:
    """Resolve a runtime device name from env/config and torch CUDA availability."""
    selected = str(preference or get_model_runtime_device_preference()).strip().lower()
    if selected not in _ALLOWED_MODEL_RUNTIME_DEVICES:
        selected = DEFAULT_MODEL_RUNTIME_DEVICE
    if selected == "cpu":
        return "cpu"
    if torch_like is None or getattr(torch_like, "cuda", None) is None:
        return "cpu"
    try:
        if bool(torch_like.cuda.is_available()):
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"
