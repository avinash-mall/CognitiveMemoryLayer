"""Tests for cml.models.enums."""

from __future__ import annotations

import pytest

from cml.models.enums import (
    MemorySource,
    MemoryStatus,
    MemoryType,
    OperationType,
)


def test_memory_type_values_are_strings() -> None:
    """MemoryType enum values are strings (StrEnum)."""
    assert isinstance(MemoryType.PREFERENCE.value, str)
    assert MemoryType.PREFERENCE.value == "preference"
    assert MemoryType.SEMANTIC_FACT.value == "semantic_fact"
    assert MemoryType.EPISODIC_EVENT.value == "episodic_event"


def test_memory_type_common_values() -> None:
    """Common memory types exist and have expected values."""
    assert MemoryType.PREFERENCE == "preference"
    assert MemoryType.CONVERSATION == "conversation"
    assert MemoryType.MESSAGE == "message"
    assert MemoryType.KNOWLEDGE == "knowledge"


def test_memory_status_values() -> None:
    """MemoryStatus has active, silent, archived, deleted."""
    assert MemoryStatus.ACTIVE.value == "active"
    assert MemoryStatus.SILENT.value == "silent"
    assert MemoryStatus.ARCHIVED.value == "archived"
    assert MemoryStatus.DELETED.value == "deleted"


def test_memory_source_values() -> None:
    """MemorySource has expected provenance values."""
    assert MemorySource.USER_EXPLICIT.value == "user_explicit"
    assert MemorySource.AGENT_INFERRED.value == "agent_inferred"
    assert MemorySource.CONSOLIDATION.value == "consolidation"


def test_operation_type_values() -> None:
    """OperationType has add, update, delete, etc."""
    assert OperationType.ADD.value == "add"
    assert OperationType.UPDATE.value == "update"
    assert OperationType.DELETE.value == "delete"
    assert OperationType.NOOP.value == "noop"


def test_memory_type_members_count() -> None:
    """MemoryType has expected number of members (non-empty)."""
    members = list(MemoryType)
    assert len(members) >= 10
    assert all(isinstance(m.value, str) for m in members)
