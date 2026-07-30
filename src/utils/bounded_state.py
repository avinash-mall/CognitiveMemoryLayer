"""LRU + TTL bounded state map for in-process memory state.

Replaces unbounded ``dict[str, T]`` in :class:`WorkingMemoryManager`
and :class:`SensoryBufferManager`, preventing memory leaks in
long-running servers and reducing global lock contention.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class BoundedStateMap(Generic[T]):
    """Bounded LRU + TTL in-process state map.

    * **LRU eviction** — oldest entry removed when ``max_size`` is exceeded.
    * **TTL expiry** — entries older than ``ttl_seconds`` are lazily pruned.
    Methods are **synchronous**. asyncio is single-threaded and no operation here
    awaits, so a coroutine can never be preempted mid-update — the ``asyncio.Lock``
    this class used to hold could not be contended and guarded nothing, while
    forcing ~8 ``await``s onto the read/write hot path. If a mutating operation
    ever needs to await, the lock has to come back with it.

    ponytail: not ``cachetools.TTLCache``. That evicts FIFO-by-expiry, whereas
    ``get`` here does ``move_to_end`` for true LRU — swapping would let a
    long-lived active session be evicted ahead of an idle newer one.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 1800.0,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        # value → (item, created_timestamp)
        self._data: OrderedDict[str, tuple[T, float]] = OrderedDict()

    # ── Public API ──────────────────────────────────────────────────

    def get(self, key: str) -> T | None:
        """Return the value for *key*, or ``None`` if expired / missing."""
        entry = self._data.get(key)
        if entry is None:
            return None
        value, created_at = entry
        if time.time() - created_at > self._ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def get_or_create(self, key: str, factory: Callable[[], T]) -> T:
        """Return existing value or create one with *factory*."""
        entry = self._data.get(key)
        if entry is not None:
            value, created_at = entry
            if time.time() - created_at <= self._ttl:
                self._data.move_to_end(key)
                return value
            del self._data[key]

        value = factory()
        self._data[key] = (value, time.time())
        self._evict_overflow()
        return value

    def set(self, key: str, value: T) -> None:
        """Set a value (create or update)."""
        self._data[key] = (value, time.time())
        self._data.move_to_end(key)
        self._evict_overflow()

    def delete(self, key: str) -> bool:
        """Remove *key*.  Returns ``True`` if it existed."""
        if key in self._data:
            del self._data[key]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired entries.  Returns count removed."""
        now = time.time()
        to_remove = [k for k, (_, ts) in self._data.items() if now - ts > self._ttl]
        for k in to_remove:
            del self._data[k]
        return len(to_remove)

    @property
    def size(self) -> int:
        return len(self._data)

    # ── Internal ────────────────────────────────────────────────────

    def _evict_overflow(self) -> None:
        """Evict oldest entries while over capacity (caller holds lock)."""
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)
