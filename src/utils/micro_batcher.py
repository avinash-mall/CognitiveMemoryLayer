"""Generic async micro-batcher.

Coalesces concurrent single/few-item async calls into batched backend calls,
trading a small fixed latency (``max_wait``) for far fewer/larger backend calls.
Shared by the DeBERTa span predictor and the batching embedding client.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")  # input item
R = TypeVar("R")  # result


class AsyncMicroBatcher(ABC, Generic[T, R]):
    """Wait/coalesce/dispatch/overflow machinery shared by batching clients.

    Concurrent ``_submit`` calls arriving within ``max_wait`` seconds are merged
    into a single ``_run_batch`` call (capped at ``max_batch``; overflow is
    re-dispatched immediately). Subclasses implement ``_run_batch`` (the actual
    backend call); ``_distribute`` defaults to tolerant None-fill but may be
    overridden by subclasses with a strict 1:1 result contract.

    Thread-safety: asyncio-only. State is created lazily so instances are safe
    to construct outside a running event loop (e.g. at module import time).
    """

    def __init__(self, max_wait_s: float, max_batch: int) -> None:
        self._max_wait = max_wait_s
        self._max_batch = max_batch
        self._lock: asyncio.Lock | None = None
        self._pending: list[tuple[T, asyncio.Future[R]]] = []
        self._dispatch_task: asyncio.Task[None] | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _submit(self, items: list[T]) -> list[R]:
        """Queue ``items``, await their results, and return them in order."""
        if not items:
            return []
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[R]] = [loop.create_future() for _ in items]
        lock = self._get_lock()
        async with lock:
            for item, fut in zip(items, futures, strict=True):
                self._pending.append((item, fut))
            if self._dispatch_task is None or self._dispatch_task.done():
                self._dispatch_task = loop.create_task(self._dispatch_after_wait())
        return list(await asyncio.gather(*futures))

    async def _dispatch_after_wait(self) -> None:
        await asyncio.sleep(self._max_wait)
        await self._drain()

    async def _drain(self) -> None:
        lock = self._get_lock()
        async with lock:
            if not self._pending:
                return
            # Cap batch size to prevent super-batches; overflow stays in
            # _pending and is re-dispatched immediately after this batch.
            batch = self._pending[: self._max_batch]
            self._pending = self._pending[self._max_batch :]
            if self._pending:
                loop = asyncio.get_running_loop()
                self._dispatch_task = loop.create_task(self._drain())
            else:
                self._dispatch_task = None

        items = [item for item, _ in batch]
        try:
            results = await self._run_batch(items)
        except Exception as exc:
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return
        self._distribute(batch, results)

    @abstractmethod
    async def _run_batch(self, items: list[T]) -> list[R]:
        """Run the backend call for one coalesced batch of items."""

    def _distribute(self, batch: list[tuple[T, asyncio.Future[R]]], results: list[R]) -> None:
        """Resolve each future from ``results`` (tolerant: None-fill when short)."""
        for i, (_, fut) in enumerate(batch):
            if not fut.done():
                fut.set_result(results[i] if i < len(results) else None)  # type: ignore[arg-type]
