"""Retry logic with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from cml.config import CMLConfig
from cml.exceptions import (
    ConnectionError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from cml.utils.logging import logger

T = TypeVar("T")

RETRYABLE_EXCEPTIONS = (ServerError, ConnectionError, TimeoutError, RateLimitError)


def retry_sync(config: CMLConfig, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute func with sync retry logic."""
    last_exception: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            last_exception = e
            if e.retry_after is not None:
                logger.warning("Rate limited, retrying after %.1fs", e.retry_after)
                time.sleep(e.retry_after)
            else:
                delay = _sleep_with_backoff(attempt, config.retry_delay)
                logger.debug(
                    "Retry attempt %s/%s after %s, slept %.1fs",
                    attempt + 1,
                    config.max_retries,
                    type(e).__name__,
                    delay,
                )
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < config.max_retries:
                delay = _sleep_with_backoff(attempt, config.retry_delay)
                logger.debug(
                    "Retry attempt %s/%s after %s, sleeping %.1fs",
                    attempt + 1,
                    config.max_retries,
                    type(e).__name__,
                    delay,
                )
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("retry loop exited without attempt or exception")


async def retry_async(
    config: CMLConfig,
    func: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute async func with retry logic."""
    last_exception: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except RateLimitError as e:
            last_exception = e
            if e.retry_after is not None:
                logger.warning("Rate limited, retrying after %.1fs", e.retry_after)
                await asyncio.sleep(e.retry_after)
            else:
                delay = await _async_sleep_with_backoff(attempt, config.retry_delay)
                logger.debug(
                    "Retry attempt %s/%s after %s, slept %.1fs",
                    attempt + 1,
                    config.max_retries,
                    type(e).__name__,
                    delay,
                )
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < config.max_retries:
                delay = await _async_sleep_with_backoff(attempt, config.retry_delay)
                logger.debug(
                    "Retry attempt %s/%s after %s, sleeping %.1fs",
                    attempt + 1,
                    config.max_retries,
                    type(e).__name__,
                    delay,
                )
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("retry loop exited without attempt or exception")


def _sleep_with_backoff(attempt: int, base_delay: float) -> float:
    delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
    time.sleep(delay)
    return delay


async def _async_sleep_with_backoff(attempt: int, base_delay: float) -> float:
    delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
    await asyncio.sleep(delay)
    return delay
