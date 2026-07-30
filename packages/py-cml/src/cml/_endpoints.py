"""Shared endpoint helpers for sync and async clients.

Centralises URL construction, request-body building, and response parsing
so both ``client.py`` and ``async_client.py`` use identical logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from cml.models import (
    ReadRequest,
    WriteRequest,
)


def build_write_body(
    content: str,
    *,
    context_tags: list[str] | None = None,
    session_id: str | None = None,
    memory_type: Any | None = None,
    namespace: str | None = None,
    metadata: dict[str, Any] | None = None,
    turn_id: str | None = None,
    agent_id: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    return WriteRequest(
        content=content,
        context_tags=context_tags,
        session_id=session_id,
        memory_type=memory_type,
        namespace=namespace,
        metadata=metadata or {},
        turn_id=turn_id,
        agent_id=agent_id,
        timestamp=timestamp,
    ).model_dump(exclude_none=True, mode="json")


def build_read_body(
    query: str,
    *,
    max_results: int = 10,
    context_filter: list[str] | None = None,
    memory_types: list[Any] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    response_format: Literal["packet", "list", "llm_context"] = "packet",
    user_timezone: str | None = None,
) -> dict[str, Any]:
    return ReadRequest(
        query=query,
        max_results=max_results,
        context_filter=context_filter,
        memory_types=memory_types,
        since=since,
        until=until,
        format=response_format,
        user_timezone=user_timezone,
    ).model_dump(exclude_none=True, by_alias=True, mode="json")


def eval_mode_headers(eval_mode: bool) -> dict[str, str] | None:
    return {"X-Eval-Mode": "true"} if eval_mode else None
