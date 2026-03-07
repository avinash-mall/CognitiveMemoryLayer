from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

ADMIN_AUTH = SimpleNamespace(tenant_id="tenant-a")


class ResultStub:
    def __init__(
        self,
        *,
        scalar: Any = None,
        one_or_none: Any = None,
        all_rows: list[Any] | None = None,
        scalar_rows: list[Any] | None = None,
    ) -> None:
        self._scalar = scalar
        self._one_or_none = one_or_none
        self._all_rows = all_rows or []
        self._scalar_rows = scalar_rows or []

    def scalar(self) -> Any:
        return self._scalar

    def one_or_none(self) -> Any:
        return self._one_or_none

    def scalar_one_or_none(self) -> Any:
        return self._one_or_none

    def all(self) -> list[Any]:
        return list(self._all_rows)

    def scalars(self) -> SimpleNamespace:
        return SimpleNamespace(all=lambda: list(self._scalar_rows))


class SessionStub:
    def __init__(self, results: list[ResultStub] | None = None) -> None:
        self._results = list(results or [])
        self.execute_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.added: list[Any] = []
        self.commits = 0

    async def execute(self, *args: Any, **kwargs: Any) -> ResultStub:
        self.execute_calls.append((args, kwargs))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def add(self, value: Any) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.commits += 1


class AsyncCM:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def __aenter__(self) -> Any:
        return self.value

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class NeoResultStub:
    def __init__(self, *, single: Any = None, items: list[Any] | None = None) -> None:
        self._single = single
        self._items = items or []

    async def single(self) -> Any:
        return self._single

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            for item in self._items:
                yield item

        return _gen()


class NeoSessionStub:
    def __init__(self, results: list[NeoResultStub | Exception]) -> None:
        self._results = list(results)
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def run(self, *args: Any, **kwargs: Any) -> NeoResultStub:
        self.calls.append((args, kwargs))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class RedisStub:
    def __init__(
        self,
        *,
        scan_results: list[tuple[int, list[str | bytes]]] | None = None,
        values: dict[str, Any] | None = None,
        lrange_values: dict[str, list[str | bytes]] | None = None,
        ttl_values: dict[str, int] | None = None,
        info_value: dict[str, Any] | None = None,
        db_size: int = 0,
    ) -> None:
        self.scan_results = list(scan_results or [(0, [])])
        self.values = values or {}
        self.lrange_values = lrange_values or {}
        self.ttl_values = ttl_values or {}
        self.info_value = info_value or {}
        self.db_size = db_size

    async def scan(self, cursor: int, match: str, count: int) -> tuple[int, list[str | bytes]]:
        _ = cursor, match, count
        return self.scan_results.pop(0)

    async def ttl(self, key: str) -> int:
        return self.ttl_values.get(key, -1)

    async def get(self, key: str) -> Any:
        return self.values.get(key)

    async def lrange(self, key: str, start: int, end: int) -> list[str | bytes]:
        _ = start, end
        return self.lrange_values.get(key, [])

    async def ping(self) -> bool:
        return True

    async def dbsize(self) -> int:
        return self.db_size

    async def info(self, section: str) -> dict[str, Any]:
        _ = section
        return self.info_value


def make_db(
    *,
    pg_results: list[ResultStub] | None = None,
    session: SessionStub | None = None,
    neo_session: NeoSessionStub | None = None,
    redis: RedisStub | None = None,
    neo4j_driver: object | None = object(),
) -> tuple[SimpleNamespace, SessionStub]:
    pg_session = session or SessionStub(pg_results)
    db = SimpleNamespace(
        pg_session=lambda: AsyncCM(pg_session),
        neo4j_session=(lambda: AsyncCM(neo_session)) if neo_session is not None else None,
        neo4j_driver=neo4j_driver if neo_session is not None else None,
        redis=redis,
    )
    return db, pg_session
