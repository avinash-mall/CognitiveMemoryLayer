"""Shared endpoint helpers for sync and async clients.

Centralises URL construction, request-body building, and response parsing
so both ``client.py`` and ``async_client.py`` use identical logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from cml.models import (
    ReadRequest,
    TurnRequest,
    UpdateRequest,
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


def build_turn_body(
    user_message: str,
    *,
    assistant_response: str | None = None,
    session_id: str | None = None,
    max_context_tokens: int = 1500,
    timestamp: datetime | None = None,
    user_timezone: str | None = None,
) -> dict[str, Any]:
    return TurnRequest(
        user_message=user_message,
        assistant_response=assistant_response,
        session_id=session_id,
        max_context_tokens=max_context_tokens,
        timestamp=timestamp,
        user_timezone=user_timezone,
    ).model_dump(exclude_none=True, mode="json")


def build_update_body(
    memory_id: UUID,
    *,
    text: str | None = None,
    confidence: float | None = None,
    importance: float | None = None,
    metadata: dict[str, Any] | None = None,
    feedback: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (url_path, body) for the update endpoint."""
    body = UpdateRequest(
        memory_id=memory_id,
        text=text,
        confidence=confidence,
        importance=importance,
        metadata=metadata,
        feedback=feedback,
    ).model_dump(exclude_none=True, mode="json")
    return f"/memory/{memory_id}", body


def build_forget_body(
    *,
    before: datetime | None = None,
    memory_type: Any | None = None,
    namespace: str | None = None,
    context_tags: list[str] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    # The server supports additional optional fields (memory_type/namespace/context_tags/dry_run)
    # beyond the minimal typed `ForgetRequest` model used by this SDK.
    payload: dict[str, Any] = {"dry_run": dry_run}
    if before is not None:
        payload["before"] = before
    if memory_type is not None:
        payload["memory_type"] = memory_type
    if namespace is not None:
        payload["namespace"] = namespace
    if context_tags is not None:
        payload["context_tags"] = context_tags
    return payload


def build_create_session_body(
    *,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    context_tags: list[str] | None = None,
) -> dict[str, Any]:
    # The server accepts `session_id` and `context_tags`, but the typed SDK request model
    # is intentionally smaller. Build the JSON payload directly.
    payload: dict[str, Any] = {"metadata": metadata or {}}
    if session_id is not None:
        payload["session_id"] = session_id
    if context_tags is not None:
        payload["context_tags"] = context_tags
    return payload


def eval_mode_headers(eval_mode: bool) -> dict[str, str] | None:
    return {"X-Eval-Mode": "true"} if eval_mode else None
