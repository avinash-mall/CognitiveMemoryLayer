"""Unit tests for core exception hierarchy."""

import pytest

from src.core.exceptions import (
    CognitiveMemoryError,
    MemoryAccessDenied,
    MemoryNotFoundError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Core exception inheritance."""

    def test_memory_not_found_inherits_from_cognitive_memory_error(self):
        assert issubclass(MemoryNotFoundError, CognitiveMemoryError)

    def test_memory_access_denied_inherits_from_cognitive_memory_error(self):
        assert issubclass(MemoryAccessDenied, CognitiveMemoryError)

    def test_validation_error_inherits_from_cognitive_memory_error(self):
        assert issubclass(ValidationError, CognitiveMemoryError)


class TestMemoryNotFoundError:
    """MemoryNotFoundError message formatting."""

    def test_message_without_memory_id(self):
        e = MemoryNotFoundError()
        assert e.memory_id is None
        assert str(e) == "Memory not found"

    def test_message_with_custom_message_only(self):
        e = MemoryNotFoundError(message="No such record")
        assert e.memory_id is None
        assert str(e) == "No such record"

    def test_message_with_memory_id(self):
        e = MemoryNotFoundError(memory_id="abc-123")
        assert e.memory_id == "abc-123"
        assert str(e) == "Memory not found: abc-123"

    def test_message_with_memory_id_and_custom_message(self):
        e = MemoryNotFoundError(memory_id="xyz", message="Record missing")
        assert e.memory_id == "xyz"
        assert str(e) == "Record missing: xyz"


class TestExceptionRaising:
    """Exceptions can be raised and caught."""

    def test_catch_as_cognitive_memory_error(self):
        with pytest.raises(CognitiveMemoryError):
            raise ValidationError("bad input")
