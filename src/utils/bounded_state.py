"""LRU + TTL bounded state map for in-process memory state.

Replaces unbounded ``dict[str, T]`` in :class:`WorkingMemoryManager`
and :class:`SensoryBufferManager`, preventing memory leaks in
long-running servers and reducing global lock contention.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class BoundedStateMap(Generic[T]):
    """Bounded LRU + TTL in-process state map.

    * **LRU eviction** — oldest entry removed when ``max_size`` is exceeded.
    * **TTL expiry** — entries older than ``ttl_seconds`` are lazily pruned.
    * **Thread-safe** for ``asyncio`` concurrent access via a single lock
      (kept lightweight because all mutating operations are O(1)).
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
        self._lock = asyncio.Lock()

    # ── Public API ──────────────────────────────────────────────────

    async def get(self, key: str) -> T | None:
        """Return the value for *key*, or ``None`` if expired / missing."""
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, created_at = entry
            if time.time() - created_at > self._ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    async def get_or_create(self, key: str, factory: Callable[[], T]) -> T:
        """Return existing value or create one with *factory*."""
        async with self._lock:
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

    async def set(self, key: str, value: T) -> None:
        """Set a value (create or update)."""
        async with self._lock:
            self._data[key] = (value, time.time())
            self._data.move_to_end(key)
            self._evict_overflow()

    async def delete(self, key: str) -> bool:
        """Remove *key*.  Returns ``True`` if it existed."""
        async with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    async def cleanup_expired(self) -> int:
        """Remove all expired entries.  Returns count removed."""
        now = time.time()
        async with self._lock:
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
