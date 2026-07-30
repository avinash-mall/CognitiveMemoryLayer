"""Embedded-mode dependency check.

Only the availability check lives here. Six `import_*` pass-through wrappers were
removed: each one lazily imported a `src.*` symbol and returned it, which did not
actually decouple `cml.embedded` from the engine — it still imports `src.*`, just
through one more call. The call sites now do the lazy import inline, which is both
shorter and honest about the dependency.
"""

from __future__ import annotations


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
