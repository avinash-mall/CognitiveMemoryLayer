"""Prometheus metrics for memory operations and retrieval (Phase 10)."""

import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from prometheus_client import Counter, Gauge, Histogram

# Memory operation counters
MEMORY_WRITES = Counter(
    "memory_writes_total",
    "Total memory write operations",
    ["tenant_id", "status"],
)

MEMORY_READS = Counter(
    "memory_reads_total",
    "Total memory read operations",
    ["tenant_id"],
)

# Retrieval latency histogram
RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds",
    "Retrieval operation latency",
    ["tenant_id"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Current memory count gauge (updated by callers)
MEMORY_COUNT = Gauge(
    "memory_count",
    "Current memory count per user",
    ["tenant_id", "user_id", "type"],
)

F = TypeVar("F", bound=Callable[..., Any])


def track_retrieval_latency(tenant_id: str = "default") -> Callable[[F], F]:
    """Decorator to track retrieval operation latency."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                RETRIEVAL_LATENCY.labels(tenant_id=tenant_id).observe(time.perf_counter() - start)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                RETRIEVAL_LATENCY.labels(tenant_id=tenant_id).observe(time.perf_counter() - start)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
