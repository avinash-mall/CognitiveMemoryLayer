"""Embedded test fixtures. Skip if embedded/engine deps not installed."""

from __future__ import annotations

import pytest

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
    """Skip test if embedded not available."""
    if not _embedded_available():
        pytest.skip("Embedded mode not installed (pip install cognitive-memory-layer[embedded])")
