"""Core enums for memory types, status, and operations."""

from enum import Enum


class MemoryContext(str, Enum):
    """Context tags for memory retrieval (non-exclusive, can have multiple)."""

    PERSONAL = "personal"  # About the user
    WORLD = "world"  # General world knowledge
    CONVERSATION = "conversation"  # Recent dialogue
    TASK = "task"  # Task-related
    PROCEDURAL = "procedural"  # How-to knowledge


class MemoryType(str, Enum):
    """Type of memory record."""

    # Existing
    EPISODIC_EVENT = "episodic_event"
    SEMANTIC_FACT = "semantic_fact"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    HYPOTHESIS = "hypothesis"
    # Generalized (was user-specific)
    PREFERENCE = "preference"
    TASK_STATE = "task_state"
    # General-purpose types
    CONVERSATION = "conversation"  # Chat message/turn
    MESSAGE = "message"  # Single message
    TOOL_RESULT = "tool_result"  # Output from tool execution
    REASONING_STEP = "reasoning_step"  # Chain-of-thought step
    SCRATCH = "scratch"  # Temporary working memory
    KNOWLEDGE = "knowledge"  # General world knowledge
    OBSERVATION = "observation"  # Agent observations
    PLAN = "plan"  # Agent plans/goals


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
