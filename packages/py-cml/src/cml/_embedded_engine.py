"""Internal bridge for embedded engine imports.

This keeps `cml.embedded` decoupled from direct `src.*` imports so the public
package surface can stay stable even while the in-repo engine remains the
implementation behind embedded mode.
"""

from __future__ import annotations

from typing import Any


def ensure_engine_available() -> None:
    try:
        import aiosqlite  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Embedded mode requires aiosqlite. Install with: pip install cognitive-memory-layer[embedded]"
        ) from exc
    try:
        from src.memory.orchestrator import MemoryOrchestrator  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'Embedded mode requires the CML engine. From repo root: pip install -e ".[embedded]".'
        ) from exc


def import_memory_orchestrator() -> Any:
    from src.memory.orchestrator import MemoryOrchestrator

    return MemoryOrchestrator


def import_seamless_provider() -> Any:
    from src.memory.seamless_provider import SeamlessMemoryProvider

    return SeamlessMemoryProvider


def import_packet_builder() -> Any:
    from src.retrieval.packet_builder import MemoryPacketBuilder

    return MemoryPacketBuilder


def get_embedding_client() -> Any:
    from src.utils.embeddings import get_embedding_client as _get_embedding_client

    return _get_embedding_client()


def import_settings() -> Any:
    from src.core.config import get_settings

    return get_settings


def import_llm_client() -> Any:
    from src.utils.llm import OpenAICompatibleClient

    return OpenAICompatibleClient
