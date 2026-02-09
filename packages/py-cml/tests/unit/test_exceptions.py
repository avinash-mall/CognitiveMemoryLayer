"""Tests for exception hierarchy."""

from cml.exceptions import (
    AuthenticationError,
    CMLError,
    RateLimitError,
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
