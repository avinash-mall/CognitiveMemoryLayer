"""Core enums for memory types, status, and operations.

The shared enums (``MemoryType``, ``MemoryStatus``, ``MemorySource``,
``OperationType``) are re-exported from the single-source-of-truth
``cml_contracts`` package so the server and SDK can never drift. The
server-internal ``MemoryContext`` stays defined here.
"""

from enum import StrEnum

from cml_contracts.enums import (
    MemorySource,
    MemoryStatus,
    MemoryType,
    OperationType,
)


class MemoryContext(StrEnum):
    """Context tags for memory retrieval (non-exclusive, can have multiple)."""

    PERSONAL = "personal"  # About the user
    WORLD = "world"  # General world knowledge
    CONVERSATION = "conversation"  # Recent dialogue
    TASK = "task"  # Task-related
    PROCEDURAL = "procedural"  # How-to knowledge


__all__ = [
    "MemoryContext",
    "MemorySource",
    "MemoryStatus",
    "MemoryType",
    "OperationType",
]
