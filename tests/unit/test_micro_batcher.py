"""Unit tests for the shared AsyncMicroBatcher coordination logic.

Pure asyncio — no heavy deps. Verifies the wait/coalesce/dispatch/overflow
machinery shared by the DeBERTa span predictor and the embedding client.
"""

import asyncio

from src.utils.micro_batcher import AsyncMicroBatcher


class _RecordingBatcher(AsyncMicroBatcher[int, int]):
    """Doubles each int and records the batches it was asked to run."""

    def __init__(self, max_wait_s: float = 0.02, max_batch: int = 4) -> None:
        super().__init__(max_wait_s, max_batch)
        self.batches: list[list[int]] = []

    async def _run_batch(self, items: list[int]) -> list[int]:
        self.batches.append(list(items))
        return [x * 2 for x in items]

    async def submit(self, items: list[int]) -> list[int]:
        return await self._submit(items)


async def test_coalesces_concurrent_submits_into_one_batch():
    b = _RecordingBatcher(max_wait_s=0.05, max_batch=10)
    results = await asyncio.gather(*[b.submit([i]) for i in range(5)])
    assert results == [[0], [2], [4], [6], [8]]
    # Five single-item submits arriving within max_wait coalesce into ONE batch.
    assert len(b.batches) == 1
    assert sorted(b.batches[0]) == [0, 1, 2, 3, 4]


async def test_preserves_order_within_a_submit():
    b = _RecordingBatcher(max_wait_s=0.02, max_batch=10)
    assert await b.submit([3, 1, 2]) == [6, 2, 4]


async def test_overflow_is_redispatched_in_capped_batches():
    b = _RecordingBatcher(max_wait_s=0.05, max_batch=2)
    results = await asyncio.gather(*[b.submit([i]) for i in range(5)])
    assert results == [[0], [2], [4], [6], [8]]
    assert all(len(batch) <= 2 for batch in b.batches)
    assert sum(len(batch) for batch in b.batches) == 5


async def test_empty_submit_returns_empty_without_backend_call():
    b = _RecordingBatcher()
    assert await b.submit([]) == []
    assert b.batches == []


async def test_exception_propagates_to_every_future_in_batch():
    class _Boom(_RecordingBatcher):
        async def _run_batch(self, items: list[int]) -> list[int]:
            raise RuntimeError("backend down")

    b = _Boom(max_wait_s=0.05, max_batch=10)
    settled = await asyncio.gather(*[b.submit([i]) for i in range(3)], return_exceptions=True)
    assert len(settled) == 3
    assert all(isinstance(r, RuntimeError) and str(r) == "backend down" for r in settled)


async def test_default_distribute_none_fills_short_results():
    class _Short(_RecordingBatcher):
        async def _run_batch(self, items: list[int]) -> list[int]:
            return [items[0] * 2]  # one result regardless of batch size

    b = _Short(max_wait_s=0.05, max_batch=10)
    results = await asyncio.gather(b.submit([1]), b.submit([2]), b.submit([3]))
    flat = [r[0] for r in results]
    assert flat.count(None) == 2  # tolerant fill for the two missing results
    assert sum(1 for v in flat if v is not None) == 1


if __name__ == "__main__":  # allow running without pytest (bare venv)
    import inspect

    async def _main() -> None:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and inspect.iscoroutinefunction(fn):
                await fn()
                print(f"ok: {name}")

    asyncio.run(_main())
    print("ALL MICRO-BATCHER TESTS PASSED")
