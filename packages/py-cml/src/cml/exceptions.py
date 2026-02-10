"""Exception hierarchy for cognitive-memory-layer."""

from __future__ import annotations

from typing import Any


class CMLError(Exception):
    """Base exception for all CML errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code (if from HTTP response).
        response_body: Raw server response dict (if available).
        request_id: Server request ID for support debugging.
        suggestion: Actionable suggestion for the developer.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: dict[str, Any] | None = None,
        request_id: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        full_message = message
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"
        if request_id:
            full_message += f"\n  Request ID: {request_id}"
        super().__init__(full_message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id
        self.suggestion = suggestion

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self.message!r}, status_code={self.status_code})"
        )


class AuthenticationError(CMLError):
    """Raised when authentication fails (401)."""

    pass


class AuthorizationError(CMLError):
    """Raised when authorization fails (403)."""

    pass


class NotFoundError(CMLError):
    """Raised when a resource is not found (404)."""

    pass


class ValidationError(CMLError):
    """Raised when request validation fails (422)."""

    pass


class RateLimitError(CMLError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        status_code: int | None = None,
        response_body: dict[str, Any] | None = None,
        request_id: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            response_body=response_body,
            request_id=request_id,
            suggestion=suggestion,
        )
        self.retry_after = retry_after


class ServerError(CMLError):
    """Raised when the server returns 5xx error."""

    pass


class ConnectionError(CMLError):
    """Raised when unable to connect to the CML server (CML-specific, not builtin)."""

    pass


class TimeoutError(CMLError):
    """Raised when a request times out (CML-specific, not builtin)."""

    pass


__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "CMLError",
    "ConnectionError",
    "NotFoundError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "ValidationError",
]
