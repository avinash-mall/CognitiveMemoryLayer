"""Prometheus metrics for memory operations and retrieval (Phase 10)."""

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
