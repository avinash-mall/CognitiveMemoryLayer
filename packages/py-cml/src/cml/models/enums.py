"""Enums for memory types, status, and operations."""

from enum import StrEnum


class MemoryType(StrEnum):
    """Type of memory record."""

    EPISODIC_EVENT = "episodic_event"
    SEMANTIC_FACT = "semantic_fact"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    HYPOTHESIS = "hypothesis"
    PREFERENCE = "preference"
    TASK_STATE = "task_state"
    CONVERSATION = "conversation"
    MESSAGE = "message"
    TOOL_RESULT = "tool_result"
    REASONING_STEP = "reasoning_step"
    SCRATCH = "scratch"
    KNOWLEDGE = "knowledge"
    OBSERVATION = "observation"
    PLAN = "plan"


class MemoryStatus(StrEnum):
    """Lifecycle status of a memory."""

    ACTIVE = "active"
    SILENT = "silent"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemorySource(StrEnum):
    """Provenance source of a memory."""

    USER_EXPLICIT = "user_explicit"
    USER_CONFIRMED = "user_confirmed"
    AGENT_INFERRED = "agent_inferred"
    TOOL_RESULT = "tool_result"
    CONSOLIDATION = "consolidation"
    RECONSOLIDATION = "reconsolidation"


class OperationType(StrEnum):
    """Type of operation in event log."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"
    REINFORCE = "reinforce"
    DECAY = "decay"
    SILENCE = "silence"
    COMPRESS = "compress"
