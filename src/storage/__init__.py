"""Storage layer: PostgreSQL, Neo4j, Redis, and connection management."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .connection import DatabaseManager
    from .models import Base, MemoryRecordModel, SemanticFactModel
    from .postgres import PostgresMemoryStore

__all__ = [
    "Base",
    "DatabaseManager",
    "MemoryRecordModel",
    "PostgresMemoryStore",
    "SemanticFactModel",
]


def __getattr__(name: str) -> Any:
    if name == "DatabaseManager":
        from .connection import DatabaseManager

        return DatabaseManager
    if name in ("Base", "MemoryRecordModel", "SemanticFactModel"):
        from . import models

        return getattr(models, name)
    if name == "PostgresMemoryStore":
        from .postgres import PostgresMemoryStore

        return PostgresMemoryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
