"""Consolidation: episodic to neocortical migration and summarization."""

__all__ = [
    "ConsolidationReport",
    "ConsolidationWorker",
]


def __getattr__(name: str):
    """Lazy export to avoid import cycles during package initialization."""
    if name in {"ConsolidationReport", "ConsolidationWorker"}:
        from .worker import ConsolidationReport, ConsolidationWorker

        return {
            "ConsolidationReport": ConsolidationReport,
            "ConsolidationWorker": ConsolidationWorker,
        }[name]
    raise AttributeError(name)
