"""Unit tests for core exception hierarchy."""

import pytest

from src.core.exceptions import (
    CognitiveMemoryError,
    ConfigurationError,
    ConsolidationError,
    DuplicateMemoryError,
    EmbeddingError,
    ExtractionError,
    ForgettingError,
    MemoryAccessDenied,
    MemoryNotFoundError,
    ReconsolidationError,
    StorageConnectionError,
    StorageError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Core exception inheritance."""

    def test_storage_error_inherits_from_cognitive_memory_error(self):
        assert issubclass(StorageError, CognitiveMemoryError)

    def test_storage_connection_error_inherits_from_storage_error(self):
        assert issubclass(StorageConnectionError, StorageError)
        assert issubclass(StorageConnectionError, CognitiveMemoryError)

    def test_memory_not_found_inherits_from_cognitive_memory_error(self):
        assert issubclass(MemoryNotFoundError, CognitiveMemoryError)

    def test_duplicate_memory_inherits_from_cognitive_memory_error(self):
        assert issubclass(DuplicateMemoryError, CognitiveMemoryError)

    def test_memory_access_denied_inherits_from_cognitive_memory_error(self):
        assert issubclass(MemoryAccessDenied, CognitiveMemoryError)

    def test_validation_error_inherits_from_cognitive_memory_error(self):
        assert issubclass(ValidationError, CognitiveMemoryError)

    def test_configuration_error_inherits_from_cognitive_memory_error(self):
        assert issubclass(ConfigurationError, CognitiveMemoryError)

    def test_embedding_error_inherits_from_cognitive_memory_error(self):
        assert issubclass(EmbeddingError, CognitiveMemoryError)

    def test_extraction_error_inherits_from_cognitive_memory_error(self):
        assert issubclass(ExtractionError, CognitiveMemoryError)

    def test_consolidation_error_inherits_from_cognitive_memory_error(self):
        assert issubclass(ConsolidationError, CognitiveMemoryError)

    def test_forgetting_error_inherits_from_cognitive_memory_error(self):
        assert issubclass(ForgettingError, CognitiveMemoryError)

    def test_reconsolidation_error_inherits_from_cognitive_memory_error(self):
        assert issubclass(ReconsolidationError, CognitiveMemoryError)


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

    def test_storage_connection_error_attributes(self):
        e = StorageConnectionError("Connection refused")
        assert str(e) == "Connection refused"
