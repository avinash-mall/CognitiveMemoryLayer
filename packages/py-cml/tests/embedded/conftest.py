"""Embedded test fixtures. Skip if embedded/engine deps not installed."""

from __future__ import annotations

from pathlib import Path

import pytest

# Load repo-root .env first so all tests read EMBEDDING_INTERNAL__*, LLM_INTERNAL__*, etc. (no HuggingFace default).
try:
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parents[4]
    load_dotenv(_repo_root / ".env")
except ImportError:
    pass

import os

os.environ["EMBEDDING_INTERNAL__PROVIDER"] = "mock"

try:
    from src.core.config import get_settings
    from src.utils.embeddings import clear_embedding_client_cache

    get_settings.cache_clear()
    clear_embedding_client_cache()
except Exception:
    pass

# Optional: skip entire embedded dir if embedded extras not installed.
# Actual engine (src.memory.orchestrator) may still be missing; tests skip on first use.
try:
    from cml.embedded import EmbeddedCognitiveMemoryLayer
except ImportError:
    EmbeddedCognitiveMemoryLayer = None  # type: ignore[misc, assignment]


def _embedded_available() -> bool:
    """True if EmbeddedCognitiveMemoryLayer can be imported."""
    return EmbeddedCognitiveMemoryLayer is not None


@pytest.fixture
def embedded_skip():
    """Raise if embedded not available (no skip; test runs and fails)."""
    if not _embedded_available():
        raise ImportError(
            "Embedded mode not installed (pip install cognitive-memory-layer[embedded])"
        )
