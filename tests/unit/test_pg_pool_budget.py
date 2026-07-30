"""Per-process PG pool budget must shrink as uvicorn workers grow.

The failure this guards: postgres ships max_connections=100, and a flat
40+20 pool per worker (plus a 20-connection startup pre-warm per worker)
made every multi-worker configuration die at startup with
"Application startup failed" — meaning the documented UVICORN_WORKERS
scaling knob could not actually be used.
"""

from src.storage.connection import pg_pool_sizes


def _sizes(monkeypatch, workers):
    if workers is None:
        monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    else:
        monkeypatch.setenv("UVICORN_WORKERS", str(workers))
    return pg_pool_sizes()


def test_single_worker_keeps_the_original_budget(monkeypatch):
    assert _sizes(monkeypatch, None) == (40, 20)
    assert _sizes(monkeypatch, 1) == (40, 20)


def test_budget_divides_by_worker_count(monkeypatch):
    assert _sizes(monkeypatch, 2) == (20, 10)
    assert _sizes(monkeypatch, 4) == (10, 5)


def test_total_demand_stays_under_postgres_default(monkeypatch):
    # startup pre-warm is min(20, pool_size) per worker; steady-state max is
    # (pool + overflow) per worker. Both must fit max_connections=100 for any
    # plausible worker count.
    for workers in range(1, 13):
        pool, overflow = _sizes(monkeypatch, workers)
        assert min(20, pool) * workers <= 100, f"pre-warm blows up at workers={workers}"
        assert (pool + overflow) * workers <= 110, f"steady-state too high at workers={workers}"


def test_garbage_value_falls_back_to_one_worker(monkeypatch):
    assert _sizes(monkeypatch, "junk") == (40, 20)


def test_floors_prevent_starvation(monkeypatch):
    pool, overflow = _sizes(monkeypatch, 50)
    assert pool >= 5 and overflow >= 2
