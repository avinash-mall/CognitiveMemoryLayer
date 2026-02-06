"""Custom exception hierarchy for the Cognitive Memory Layer."""


class CognitiveMemoryError(Exception):
    """Base exception for all Cognitive Memory Layer errors."""

    pass


# --- Storage errors ---


class StorageError(CognitiveMemoryError):
    """Base for storage-related errors."""

    pass


class StorageConnectionError(StorageError):
    """Failed to connect to a storage backend (Postgres, Neo4j, Redis)."""

    pass


# --- Memory errors ---


class MemoryNotFoundError(CognitiveMemoryError):
    """Requested memory record does not exist."""

    def __init__(self, memory_id=None, message: str = "Memory not found"):
        self.memory_id = memory_id
        super().__init__(f"{message}: {memory_id}" if memory_id else message)


class DuplicateMemoryError(CognitiveMemoryError):
    """Attempted to create a memory that already exists (by content hash)."""

    pass


class MemoryAccessDenied(CognitiveMemoryError):
    """Caller does not have permission to access the requested memory."""

    pass


# --- Validation errors ---


class ValidationError(CognitiveMemoryError):
    """Input validation failed."""

    pass


class ConfigurationError(CognitiveMemoryError):
    """Application configuration is invalid or missing required values."""

    pass


# --- Processing errors ---


class EmbeddingError(CognitiveMemoryError):
    """Failed to compute embeddings (API failure, bad input, etc.)."""

    pass


class ExtractionError(CognitiveMemoryError):
    """Entity/relation extraction failed."""

    pass


class ConsolidationError(CognitiveMemoryError):
    """Error during memory consolidation."""

    pass


class ForgettingError(CognitiveMemoryError):
    """Error during active forgetting process."""

    pass


class ReconsolidationError(CognitiveMemoryError):
    """Error during belief revision / reconsolidation."""

    pass
