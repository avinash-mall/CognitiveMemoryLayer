"""Timing utilities for performance measurement."""

import time
from contextlib import contextmanager

from .logging_config import get_logger

logger = get_logger(__name__)


@contextmanager
def timed(operation: str, warn_ms: float | None = None):
    """Context manager that logs elapsed time for an operation.

    Args:
        operation: Human-readable label for the timed block.
        warn_ms: If set, emit a warning when elapsed time exceeds this threshold (ms).

    Usage::

        with timed("vector_search", warn_ms=200):
            results = await store.vector_search(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_method = logger.info
        if warn_ms is not None and elapsed_ms > warn_ms:
            log_method = logger.warning
        log_method(
            "operation_timed",
            operation=operation,
            elapsed_ms=round(elapsed_ms, 2),
        )
