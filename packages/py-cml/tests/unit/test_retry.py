"""Tests for retry logic."""

from __future__ import annotations

import pytest

from cml.config import CMLConfig
from cml.exceptions import (
    ConnectionError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from cml.transport.retry import MAX_RETRY_DELAY, retry_async, retry_sync


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


def test_retry_sync_rate_limit_retries_then_succeeds() -> None:
    """retry_sync retries on RateLimitError (with retry_after) then succeeds."""
    config = CMLConfig(max_retries=2, retry_delay=0.01)
    calls = []

    def rate_limited_then_ok() -> str:
        calls.append(1)
        if len(calls) < 2:
            raise RateLimitError("Too many requests", retry_after=0.01)
        return "ok"

    result = retry_sync(config, rate_limited_then_ok)
    assert result == "ok"
    assert len(calls) == 2


def test_retry_sync_rate_limit_exhausted_raises() -> None:
    """retry_sync raises RateLimitError on last attempt without sleeping again."""
    config = CMLConfig(max_retries=1, retry_delay=0.01)
    calls = []

    def always_rate_limited() -> str:
        calls.append(1)
        raise RateLimitError("Too many requests", retry_after=0.01)

    with pytest.raises(RateLimitError, match="Too many requests"):
        retry_sync(config, always_rate_limited)
    assert len(calls) == 2  # initial + 1 retry, then raise on 2nd attempt


def test_retry_sync_max_retry_delay_constant() -> None:
    """MAX_RETRY_DELAY is set to cap backoff sleep."""
    assert MAX_RETRY_DELAY == 60.0


@pytest.mark.asyncio
async def test_retry_async_succeeds_first_try() -> None:
    """retry_async returns immediately on success."""
    config = CMLConfig(max_retries=2)
    calls = []

    async def ok() -> str:
        calls.append(1)
        return "ok"

    result = await retry_async(config, ok)
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retry_async_retries_then_succeeds() -> None:
    """retry_async retries on ServerError then succeeds."""
    config = CMLConfig(max_retries=2, retry_delay=0.01)
    calls = []

    async def fail_twice() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ServerError("Service unavailable", status_code=503)
        return "ok"

    result = await retry_async(config, fail_twice)
    assert result == "ok"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_retry_async_no_retry_on_validation_error() -> None:
    """retry_async does not retry on ValidationError."""
    config = CMLConfig(max_retries=3, retry_delay=0.01)
    calls = []

    async def fail_validation() -> str:
        calls.append(1)
        raise ValidationError("Bad request", status_code=422)

    with pytest.raises(ValidationError):
        await retry_async(config, fail_validation)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retry_async_rate_limit_exhausted_raises() -> None:
    """retry_async raises RateLimitError when retries exhausted."""
    config = CMLConfig(max_retries=1, retry_delay=0.01)
    calls = []

    async def always_rate_limited() -> str:
        calls.append(1)
        raise RateLimitError("Rate limited", retry_after=0.01)

    with pytest.raises(RateLimitError):
        await retry_async(config, always_rate_limited)
    assert len(calls) == 2
