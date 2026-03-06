"""Probe package entrypoints against embedded or live CML surfaces.

This script is package-focused rather than retrieval-focused. It exercises the
public `cml` interfaces the way a package consumer would:

- `embedded`: in-process embedded client
- `live-sync`: sync HTTP client
- `live-async`: async HTTP client

Examples:

    python scripts/package_surface_probe.py embedded --write "User likes tea" --query tea
    python scripts/package_surface_probe.py live-sync --base-url http://localhost:8000 --api-key test-key --write "User likes tea" --query tea
    python scripts/package_surface_probe.py live-async --base-url http://localhost:8000 --api-key test-key --write "User likes tea" --query tea --turn "What should I drink?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


def _normalize_bool_env(name: str) -> None:
    value = os.environ.get(name)
    if value is None:
        return
    normalized = value.strip().lower()
    valid = {"1", "0", "true", "false", "yes", "no", "on", "off"}
    if normalized not in valid:
        os.environ[name] = "false"


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@dataclass
class ProbeResult:
    mode: str
    writes: list[Any]
    read: Any | None
    turn: Any | None


async def _run_embedded(args: argparse.Namespace) -> ProbeResult:
    from cml import EmbeddedCognitiveMemoryLayer

    timestamp = _parse_timestamp(args.timestamp)
    async with EmbeddedCognitiveMemoryLayer(db_path=args.db_path) as memory:
        writes = []
        for text in args.write:
            writes.append(
                await memory.write(
                    text,
                    session_id=args.session_id,
                    timestamp=timestamp,
                    metadata={"probe_mode": "embedded"},
                )
            )
        read = None
        if args.query:
            read = await memory.read(
                args.query,
                max_results=args.max_results,
                user_timezone=args.user_timezone,
            )
        turn = None
        if args.turn:
            turn = await memory.turn(
                user_message=args.turn,
                assistant_response=args.assistant_response,
                session_id=args.session_id,
                user_timezone=args.user_timezone,
                timestamp=timestamp,
            )
    return ProbeResult(mode="embedded", writes=writes, read=read, turn=turn)


def _run_live_sync(args: argparse.Namespace) -> ProbeResult:
    from cml import CognitiveMemoryLayer

    timestamp = _parse_timestamp(args.timestamp)
    with CognitiveMemoryLayer(
        api_key=args.api_key,
        base_url=args.base_url,
        tenant_id=args.tenant_id,
    ) as memory:
        writes = []
        for text in args.write:
            writes.append(
                memory.write(
                    text,
                    session_id=args.session_id,
                    timestamp=timestamp,
                    metadata={"probe_mode": "live-sync"},
                )
            )
        read = None
        if args.query:
            read = memory.read(
                args.query,
                max_results=args.max_results,
                user_timezone=args.user_timezone,
            )
        turn = None
        if args.turn:
            turn = memory.turn(
                user_message=args.turn,
                assistant_response=args.assistant_response,
                session_id=args.session_id,
                user_timezone=args.user_timezone,
                timestamp=timestamp,
            )
    return ProbeResult(mode="live-sync", writes=writes, read=read, turn=turn)


async def _run_live_async(args: argparse.Namespace) -> ProbeResult:
    from cml import AsyncCognitiveMemoryLayer

    timestamp = _parse_timestamp(args.timestamp)
    async with AsyncCognitiveMemoryLayer(
        api_key=args.api_key,
        base_url=args.base_url,
        tenant_id=args.tenant_id,
    ) as memory:
        writes = []
        for text in args.write:
            writes.append(
                await memory.write(
                    text,
                    session_id=args.session_id,
                    timestamp=timestamp,
                    metadata={"probe_mode": "live-async"},
                )
            )
        read = None
        if args.query:
            read = await memory.read(
                args.query,
                max_results=args.max_results,
                user_timezone=args.user_timezone,
            )
        turn = None
        if args.turn:
            turn = await memory.turn(
                user_message=args.turn,
                assistant_response=args.assistant_response,
                session_id=args.session_id,
                user_timezone=args.user_timezone,
                timestamp=timestamp,
            )
    return ProbeResult(mode="live-async", writes=writes, read=read, turn=turn)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe package entrypoints for CML.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def _add_shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--write", action="append", default=[], help="Memory text to write.")
        subparser.add_argument("--query", default="", help="Query to read after writes.")
        subparser.add_argument("--turn", default="", help="Optional user turn to send.")
        subparser.add_argument(
            "--assistant-response",
            default=None,
            help="Optional assistant response to persist with --turn.",
        )
        subparser.add_argument("--session-id", default=None)
        subparser.add_argument("--user-timezone", default=None)
        subparser.add_argument("--timestamp", default=None, help="ISO-8601 timestamp.")
        subparser.add_argument("--max-results", type=int, default=5)
        subparser.add_argument(
            "--expect-min-memories",
            type=int,
            default=0,
            help="Fail when read total_count is below this value.",
        )

    embedded = subparsers.add_parser("embedded", help="Run against EmbeddedCognitiveMemoryLayer.")
    _add_shared(embedded)
    embedded.add_argument("--db-path", default=None)

    live_sync = subparsers.add_parser("live-sync", help="Run against sync HTTP client.")
    _add_shared(live_sync)
    live_sync.add_argument("--base-url", required=True)
    live_sync.add_argument("--api-key", required=True)
    live_sync.add_argument("--tenant-id", default="default")

    live_async = subparsers.add_parser("live-async", help="Run against async HTTP client.")
    _add_shared(live_async)
    live_async.add_argument("--base-url", required=True)
    live_async.add_argument("--api-key", required=True)
    live_async.add_argument("--tenant-id", default="default")

    return parser


def main(argv: list[str] | None = None) -> int:
    _normalize_bool_env("DEBUG")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "embedded":
        result = asyncio.run(_run_embedded(args))
    elif args.mode == "live-sync":
        result = _run_live_sync(args)
    elif args.mode == "live-async":
        result = asyncio.run(_run_live_async(args))
    else:
        parser.error(f"Unknown mode: {args.mode}")
        return 2

    payload = _to_jsonable(asdict(result))
    print(json.dumps(payload, indent=2))

    expected = max(0, int(args.expect_min_memories))
    if expected > 0 and result.read is not None:
        total = int(getattr(result.read, "total_count", 0) or 0)
        if total < expected:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
