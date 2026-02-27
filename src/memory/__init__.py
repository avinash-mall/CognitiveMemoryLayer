"""Memory components: sensory, working, hippocampal, neocortical, orchestrator."""

__all__ = [
    "HippocampalStore",
    "MemoryOrchestrator",
    "NeocorticalStore",
    "ShortTermMemory",
]


def __getattr__(name: str):
    """Lazy exports to avoid package-level import cycles."""
    if name == "HippocampalStore":
        from .hippocampal.store import HippocampalStore

        return HippocampalStore
    if name == "NeocorticalStore":
        from .neocortical.store import NeocorticalStore

        return NeocorticalStore
    if name == "MemoryOrchestrator":
        from .orchestrator import MemoryOrchestrator

        return MemoryOrchestrator
    if name == "ShortTermMemory":
        from .short_term import ShortTermMemory

        return ShortTermMemory
    raise AttributeError(name)
