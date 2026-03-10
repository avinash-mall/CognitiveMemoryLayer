"""Probe package entrypoints against embedded or live CML surfaces.

This script is package-focused rather than retrieval-focused. It exercises the
public `cml` interfaces the way a package consumer would:

- `embedded`: in-process embedded client
- `live-sync`: sync HTTP client
- `live-async`: async HTTP client

Examples:

    python scripts/package_surface_probe.py embedded --write "User likes tea" --query tea
    python scripts/package_surface_probe.py live-sync --write "User likes tea" --query tea --expect-min-stored 1 --expect-min-memories 1
    python scripts/package_surface_probe.py live-async --write "User likes tea" --query tea --turn "What should I drink?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _shared import default_tenant_id, load_repo_env, normalize_bool_env, normalize_cml_base_url

MEMORY_TYPE_CHOICES = (
    "episodic_event",
    "semantic_fact",
    "procedure",
    "constraint",
    "hypothesis",
    "preference",
    "task_state",
    "conversation",
    "message",
    "tool_result",
    "reasoning_step",
    "scratch",
    "knowledge",
    "observation",
    "plan",
)


def _resolve_memory_type(value: str | None) -> Any | None:
    if not value:
        return None
    from cml.models.enums import MemoryType

    return MemoryType(value)


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
    tenant_id: str
    base_url: str | None
    stored_chunks: int
    writes: list[Any]
    read: Any | None
    turn: Any | None


async def _run_embedded(args: argparse.Namespace) -> ProbeResult:
    from cml import EmbeddedCognitiveMemoryLayer

    timestamp = _parse_timestamp(args.timestamp)
    memory_type = _resolve_memory_type(args.memory_type)
    async with EmbeddedCognitiveMemoryLayer(
        db_path=args.db_path, tenant_id=args.tenant_id
    ) as memory:
        writes = []
        for text in args.write:
            writes.append(
                await memory.write(
                    text,
                    session_id=args.session_id,
                    memory_type=memory_type,
                    timestamp=timestamp,
                    metadata={"probe_mode": "embedded"},
                    eval_mode=True,
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
    return ProbeResult(
        mode="embedded",
        tenant_id=args.tenant_id,
        base_url=None,
        stored_chunks=sum(int(getattr(item, "chunks_created", 0) or 0) for item in writes),
        writes=writes,
        read=read,
        turn=turn,
    )


def _run_live_sync(args: argparse.Namespace) -> ProbeResult:
    from cml import CognitiveMemoryLayer

    timestamp = _parse_timestamp(args.timestamp)
    memory_type = _resolve_memory_type(args.memory_type)
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
                    memory_type=memory_type,
                    timestamp=timestamp,
                    metadata={"probe_mode": "live-sync"},
                    eval_mode=True,
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
    return ProbeResult(
        mode="live-sync",
        tenant_id=args.tenant_id,
        base_url=args.base_url,
        stored_chunks=sum(int(getattr(item, "chunks_created", 0) or 0) for item in writes),
        writes=writes,
        read=read,
        turn=turn,
    )


async def _run_live_async(args: argparse.Namespace) -> ProbeResult:
    from cml import AsyncCognitiveMemoryLayer

    timestamp = _parse_timestamp(args.timestamp)
    memory_type = _resolve_memory_type(args.memory_type)
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
                    memory_type=memory_type,
                    timestamp=timestamp,
                    metadata={"probe_mode": "live-async"},
                    eval_mode=True,
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
    return ProbeResult(
        mode="live-async",
        tenant_id=args.tenant_id,
        base_url=args.base_url,
        stored_chunks=sum(int(getattr(item, "chunks_created", 0) or 0) for item in writes),
        writes=writes,
        read=read,
        turn=turn,
    )


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
        subparser.add_argument(
            "--memory-type",
            choices=MEMORY_TYPE_CHOICES,
            default=None,
            help="Optional explicit memory type for all --write entries.",
        )
        subparser.add_argument("--max-results", type=int, default=5)
        subparser.add_argument(
            "--expect-min-memories",
            type=int,
            default=0,
            help="Fail when read total_count is below this value.",
        )
        subparser.add_argument(
            "--expect-min-stored",
            type=int,
            default=0,
            help="Fail when the sum of chunks_created across writes is below this value.",
        )
        subparser.add_argument(
            "--tenant-id",
            default=None,
            help="Tenant to use. Defaults to a per-run isolated tenant.",
        )

    embedded = subparsers.add_parser("embedded", help="Run against EmbeddedCognitiveMemoryLayer.")
    _add_shared(embedded)
    embedded.add_argument("--db-path", default=None)

    live_sync = subparsers.add_parser("live-sync", help="Run against sync HTTP client.")
    _add_shared(live_sync)
    live_sync.add_argument(
        "--base-url",
        default=None,
        help="CML base URL. Defaults to CML_BASE_URL from the repo .env.",
    )
    live_sync.add_argument(
        "--api-key",
        default=None,
        help="CML API key. Defaults to CML_API_KEY or AUTH__API_KEY from the repo .env.",
    )

    live_async = subparsers.add_parser("live-async", help="Run against async HTTP client.")
    _add_shared(live_async)
    live_async.add_argument(
        "--base-url",
        default=None,
        help="CML base URL. Defaults to CML_BASE_URL from the repo .env.",
    )
    live_async.add_argument(
        "--api-key",
        default=None,
        help="CML API key. Defaults to CML_API_KEY or AUTH__API_KEY from the repo .env.",
    )

    return parser


def finalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if not getattr(args, "tenant_id", None):
        args.tenant_id = default_tenant_id(f"package-probe-{args.mode}")

    if args.mode == "embedded":
        return args

    args.base_url = normalize_cml_base_url(args.base_url or os.environ.get("CML_BASE_URL"))
    args.api_key = args.api_key or os.environ.get("CML_API_KEY") or os.environ.get("AUTH__API_KEY")
    if not args.base_url:
        parser.error(f"{args.mode} requires --base-url or CML_BASE_URL")
    if not args.api_key:
        parser.error(f"{args.mode} requires --api-key or CML_API_KEY/AUTH__API_KEY")
    return args


def main(argv: list[str] | None = None) -> int:
    load_repo_env()
    normalize_bool_env("DEBUG")
    parser = build_parser()
    args = finalize_args(parser.parse_args(argv), parser)

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

    expected_stored = max(0, int(args.expect_min_stored))
    if expected_stored > 0 and result.stored_chunks < expected_stored:
        return 1

    expected = max(0, int(args.expect_min_memories))
    if expected > 0 and result.read is not None:
        total = int(getattr(result.read, "total_count", 0) or 0)
        if total < expected:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
