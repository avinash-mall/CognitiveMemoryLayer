"""Tests for exception hierarchy."""

from __future__ import annotations

from cml.exceptions import (
    AuthenticationError,
    CMLError,
    CMLConnectionError,
    CMLTimeoutError,
    ConnectionError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)


def test_cml_error_attributes() -> None:
    """CMLError stores status_code and response_body."""
    err = CMLError(
        "Something failed",
        status_code=500,
        response_body={"detail": "Internal error"},
    )
    assert str(err) == "Something failed"
    assert err.status_code == 500
    assert err.response_body == {"detail": "Internal error"}


def test_cml_error_defaults() -> None:
    """CMLError works with message only."""
    err = CMLError("Bad request")
    assert err.status_code is None
    assert err.response_body is None


def test_rate_limit_error_retry_after() -> None:
    """RateLimitError stores retry_after."""
    err = RateLimitError("Too many requests", retry_after=60.0)
    assert err.retry_after == 60.0


def test_exception_inheritance() -> None:
    """All CML exceptions inherit from CMLError."""
    assert issubclass(AuthenticationError, CMLError)
    assert issubclass(ValidationError, CMLError)
    assert issubclass(RateLimitError, CMLError)
    assert issubclass(CMLConnectionError, CMLError)
    assert issubclass(CMLTimeoutError, CMLError)


def test_cml_connection_error() -> None:
    """CMLConnectionError is the SDK connection error (not builtin)."""
    err = CMLConnectionError("Connection refused")
    assert isinstance(err, CMLError)
    assert "Connection refused" in str(err)


def test_cml_timeout_error() -> None:
    """CMLTimeoutError is the SDK timeout error (not builtin)."""
    err = CMLTimeoutError("Request timed out")
    assert isinstance(err, CMLError)
    assert "Request timed out" in str(err)


def test_connection_error_alias() -> None:
    """ConnectionError from cml.exceptions is alias for CMLConnectionError."""
    assert ConnectionError is CMLConnectionError


def test_timeout_error_alias() -> None:
    """TimeoutError from cml.exceptions is alias for CMLTimeoutError."""
    assert TimeoutError is CMLTimeoutError
