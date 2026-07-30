"""Custom exception hierarchy for the Cognitive Memory Layer.

ponytail: only the classes something actually raises live here. `src/api/app.py`
maps them to status codes. Add a class when you add the `raise`, not before —
nine speculative subclasses were removed for never being raised anywhere.
"""


class CognitiveMemoryError(Exception):
    """Base exception for all Cognitive Memory Layer errors."""

    pass


class MemoryNotFoundError(CognitiveMemoryError):
    """Requested memory record does not exist."""

    def __init__(self, memory_id=None, message: str = "Memory not found"):
        self.memory_id = memory_id
        super().__init__(f"{message}: {memory_id}" if memory_id else message)


class MemoryAccessDenied(CognitiveMemoryError):
    """Caller does not have permission to access the requested memory."""

    pass


class ValidationError(CognitiveMemoryError):
    """Input validation failed."""

    pass
