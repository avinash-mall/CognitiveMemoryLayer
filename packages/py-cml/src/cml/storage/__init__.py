"""Storage implementations for embedded mode."""

__all__ = ["SQLiteMemoryStore"]


def __getattr__(name: str) -> object:
    if name == "SQLiteMemoryStore":
        from cml.storage.sqlite_store import SQLiteMemoryStore

        return SQLiteMemoryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
