"""Prometheus metrics for memory operations and retrieval (Phase 10)."""

import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from prometheus_client import Counter, Gauge, Histogram

# ── Memory operation counters ───────────────────────────────────────

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

# ── Retrieval latency histogram ─────────────────────────────────────

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

# ── Phase 6.2: Per-step retrieval metrics ───────────────────────────

RETRIEVAL_STEP_DURATION = Histogram(
    "cml_retrieval_step_duration_ms",
    "Duration of a single retrieval step in milliseconds",
    ["source"],
    buckets=[5, 10, 25, 50, 100, 200, 500, 1000, 2000],
)

RETRIEVAL_STEP_RESULT_COUNT = Histogram(
    "cml_retrieval_step_result_count",
    "Number of items returned by a retrieval step",
    ["source"],
    buckets=[0, 1, 3, 5, 10, 20, 50],
)

RETRIEVAL_TIMEOUT_COUNT = Counter(
    "cml_retrieval_timeout_total",
    "Number of retrieval steps that timed out",
    ["source"],
)

RETRIEVAL_STEP_FAILURES = Counter(
    "cml_retrieval_step_failures_total",
    "Number of retrieval steps that failed with an exception",
    ["source"],
)

# â”€â”€ DB Pool observability (A-06) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DB_POOL_CHECKED_OUT = Gauge(
    "cml_db_pool_checked_out",
    "Number of currently checked-out SQL connections",
)

DB_POOL_CHECKOUTS_TOTAL = Counter(
    "cml_db_pool_checkouts_total",
    "Total SQL pool checkout events",
)

DB_POOL_CHECKINS_TOTAL = Counter(
    "cml_db_pool_checkins_total",
    "Total SQL pool checkin events",
)

DB_POOL_INVALIDATIONS_TOTAL = Counter(
    "cml_db_pool_invalidations_total",
    "Total SQL pool invalidation events",
)

# ── Phase 6.3: Fact hit rate tracking ───────────────────────────────

FACT_HIT_RATE = Counter(
    "cml_retrieval_fact_hit_total",
    "Queries answered from semantic fact store vs vector fallback",
    ["intent", "hit"],
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
