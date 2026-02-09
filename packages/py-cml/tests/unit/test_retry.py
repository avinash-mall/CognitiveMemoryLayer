"""Tests for retry logic."""

import pytest

from cml.config import CMLConfig
from cml.exceptions import ConnectionError, ServerError, ValidationError
from cml.transport.retry import retry_sync


def test_retry_sync_no_retry_on_4xx() -> None:
    """retry_sync does not retry on 4xx (e.g. ValidationError); call count is 1."""
    config = CMLConfig(max_retries=3, retry_delay=0.01)
    calls = []

    def fail_validation() -> str:
        calls.append(1)
        raise ValidationError("Invalid request", status_code=422)

    with pytest.raises(ValidationError):
        retry_sync(config, fail_validation)
    assert len(calls) == 1


def test_retry_sync_succeeds_first_try() -> None:
    """retry_sync returns immediately on success."""
    config = CMLConfig(max_retries=2)
    calls = []

    def ok() -> str:
        calls.append(1)
        return "ok"

    result = retry_sync(config, ok)
    assert result == "ok"
    assert len(calls) == 1


def test_retry_sync_retries_then_succeeds() -> None:
    """retry_sync retries on retryable exception then succeeds."""
    config = CMLConfig(max_retries=2, retry_delay=0.01)
    calls = []

    def fail_twice() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ServerError("Server error", status_code=503)
        return "ok"

    result = retry_sync(config, fail_twice)
    assert result == "ok"
    assert len(calls) == 3


def test_retry_sync_raises_after_exhausted() -> None:
    """retry_sync raises last exception when retries exhausted."""
    config = CMLConfig(max_retries=2, retry_delay=0.01)

    def always_fail() -> None:
        raise ConnectionError("Connection refused")

    with pytest.raises(ConnectionError, match="Connection refused"):
        retry_sync(config, always_fail)
