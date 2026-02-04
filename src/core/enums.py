"""Core enums for memory types, status, and operations."""
from enum import Enum


class MemoryType(str, Enum):
    """Type of memory record."""

    EPISODIC_EVENT = "episodic_event"
    SEMANTIC_FACT = "semantic_fact"
    PREFERENCE = "preference"
    TASK_STATE = "task_state"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    HYPOTHESIS = "hypothesis"


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""

    ACTIVE = "active"
    SILENT = "silent"  # Hard to retrieve, needs strong cue
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemorySource(str, Enum):
    """Provenance source of a memory."""

    USER_EXPLICIT = "user_explicit"  # User directly stated
    USER_CONFIRMED = "user_confirmed"  # User confirmed inference
    AGENT_INFERRED = "agent_inferred"  # Agent extracted/inferred
    TOOL_RESULT = "tool_result"  # From tool execution
    CONSOLIDATION = "consolidation"  # From consolidation process
    RECONSOLIDATION = "reconsolidation"  # Updated after retrieval


class OperationType(str, Enum):
    """Type of operation in event log."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"
    REINFORCE = "reinforce"
    DECAY = "decay"
    SILENCE = "silence"
    COMPRESS = "compress"
