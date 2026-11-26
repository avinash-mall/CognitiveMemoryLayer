"""Embedded-mode dependency check.

Only the availability check lives here. Six `import_*` pass-through wrappers were
removed: each one lazily imported a `src.*` symbol and returned it, which did not
actually decouple `cml.embedded` from the engine — it still imports `src.*`, just
through one more call. The call sites now do the lazy import inline, which is both
shorter and honest about the dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _try_add_monorepo_root() -> bool:
    """If running from an editable monorepo checkout, add the repo root to sys.path."""
    # packages/py-cml/src/cml/_embedded_engine.py -> parents[4] is repo root
    try:
        repo_root = Path(__file__).resolve().parents[4]
    except IndexError:
        return False
    if (repo_root / "src" / "memory" / "orchestrator.py").is_file():
        root_str = str(repo_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return True
    return False


def ensure_engine_available() -> None:
    try:
        import aiosqlite  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Embedded mode requires aiosqlite. Install with: pip install cognitive-memory-layer[embedded]"
        ) from exc
    try:
        from src.memory.orchestrator import MemoryOrchestrator
    except ImportError as exc:
        if _try_add_monorepo_root():
            try:
                from src.memory.orchestrator import MemoryOrchestrator  # noqa: F401

                return
            except ImportError:
                pass
        raise ImportError(
            "Embedded mode requires the CML engine from a monorepo checkout. "
            'Clone the repository and install from repo root: pip install -e ".[embedded]".'
        ) from exc
