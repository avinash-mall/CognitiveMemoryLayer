"""Shared endpoint helpers for sync and async clients.

Centralises validation, URL query parameters, request-body construction, and
fallback response helpers so ``client.py`` and ``async_client.py`` stay thin,
type-safe, and free of duplicated logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from cml.models import (
    CreateSessionRequest,
    FeedbackRequest,
    ForgetRequest,
    ReadRequest,
    ReadResponse,
    SimulateRequest,
    TurnRequest,
    UpdateRequest,
    WindowConsolidateRequest,
    WriteRequest,
)


def eval_mode_headers(eval_mode: bool) -> dict[str, str] | None:
    return {"X-Eval-Mode": "true"} if eval_mode else None


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
    media: list[dict[str, Any]] | None = None,
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
        media=media,
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


def build_read_stream_body(
    query: str,
    *,
    max_results: int = 10,
    context_filter: list[str] | None = None,
    memory_types: list[Any] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    user_timezone: str | None = None,
) -> dict[str, Any]:
    return ReadRequest(
        query=query,
        max_results=max_results,
        context_filter=context_filter,
        memory_types=memory_types,
        since=since,
        until=until,
        user_timezone=user_timezone,
    ).model_dump(exclude_none=True, by_alias=True, mode="json")


def empty_read_response(query: str) -> ReadResponse:
    return ReadResponse(query=query, memories=[], total_count=0, elapsed_ms=0.0)


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
) -> dict[str, Any]:
    return UpdateRequest(
        memory_id=memory_id,
        text=text,
        confidence=confidence,
        importance=importance,
        metadata=metadata,
        feedback=feedback,
    ).model_dump(exclude_none=True, mode="json")


def build_forget_body(
    *,
    memory_ids: list[UUID] | None = None,
    query: str | None = None,
    before: datetime | None = None,
    action: Literal["delete", "archive", "silence"] = "delete",
) -> dict[str, Any]:
    if not memory_ids and not query and before is None:
        raise ValueError("At least one of memory_ids, query, or before must be provided")
    return ForgetRequest(
        memory_ids=memory_ids,
        query=query,
        before=before,
        action=action,
    ).model_dump(exclude_none=True, mode="json")


def build_create_session_body(
    *,
    name: str | None = None,
    ttl_hours: int = 24,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return CreateSessionRequest(
        name=name,
        ttl_hours=ttl_hours,
        metadata=metadata or {},
    ).model_dump(exclude_none=True, mode="json")


def require_delete_all_confirmation(confirm: bool) -> None:
    if not confirm:
        raise ValueError("delete_all requires confirm=True to avoid accidental data loss")


def require_reset_confirmation(confirm: bool) -> dict[str, bool]:
    if not confirm:
        raise ValueError("You must pass confirm=True to reset the database.")
    return {"confirm": True}


def build_consolidate_windowed_body(
    *,
    session_id: str | None = None,
    max_episodes: int = 2000,
    use_llm: bool | None = None,
) -> dict[str, Any]:
    return WindowConsolidateRequest(
        session_id=session_id,
        max_episodes=max_episodes,
        use_llm=use_llm,
    ).model_dump(exclude_none=True, mode="json")


def build_feedback_body(
    outcome_reward: float,
    *,
    expected_reward: float = 0.5,
    query: str | None = None,
    memory_ids: list[str] | None = None,
    correction_text: str | None = None,
    preferred_steps: list[str] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    return FeedbackRequest(
        outcome_reward=outcome_reward,
        expected_reward=expected_reward,
        query=query,
        memory_ids=memory_ids,
        correction_text=correction_text,
        preferred_steps=preferred_steps,
        session_id=session_id,
    ).model_dump(exclude_none=True, mode="json")


def build_simulate_body(
    scenario: str,
    *,
    max_results: int = 12,
    context_filter: list[str] | None = None,
) -> dict[str, Any]:
    return SimulateRequest(
        scenario=scenario,
        max_results=max_results,
        context_filter=context_filter,
    ).model_dump(exclude_none=True, mode="json")


def build_tenant_user_payload(
    default_tenant_id: str,
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"tenant_id": tenant_id or default_tenant_id}
    if user_id is not None:
        payload["user_id"] = user_id
    return payload


def build_forgetting_payload(
    default_tenant_id: str,
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
    dry_run: bool = True,
    max_memories: int = 5000,
) -> dict[str, Any]:
    payload = build_tenant_user_payload(default_tenant_id, tenant_id=tenant_id, user_id=user_id)
    payload["dry_run"] = dry_run
    payload["max_memories"] = max_memories
    return payload


def validate_batch_write_items(items: list[dict[str, Any]]) -> None:
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "content" not in item:
            raise ValueError(
                f"Each item must be a dict with a 'content' key; item at index {i} is invalid"
            )


def optional_tenant_params(tenant_id: str | None) -> dict[str, Any]:
    return {"tenant_id": tenant_id} if tenant_id is not None else {}


def build_dashboard_memories_params(
    *,
    tenant_id: str | None = None,
    page: int = 1,
    per_page: int = 25,
    type: str | None = None,
    status: str | None = None,
    search: str | None = None,
    sort_by: str = "written_at",
    order: str = "desc",
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "page": page,
        "per_page": per_page,
        "sort_by": sort_by,
        "order": order,
    }
    if tenant_id:
        params["tenant_id"] = tenant_id
    if type:
        params["type"] = type
    if status:
        params["status"] = status
    if search:
        params["search"] = search
    return params


def build_dashboard_facts_params(
    *,
    tenant_id: str | None = None,
    category: str | None = None,
    current_only: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "current_only": current_only,
        "limit": limit,
        "offset": offset,
    }
    if tenant_id is not None:
        params["tenant_id"] = tenant_id
    if category is not None:
        params["category"] = category
    return params


def build_bulk_action_payload(
    memory_ids: list[UUID], action: Literal["archive", "silence", "delete"]
) -> dict[str, Any]:
    return {"memory_ids": [str(m) for m in memory_ids], "action": action}


def build_timeline_params(
    *,
    tenant_id: str | None = None,
    days: int = 30,
    metric: str = "memories",
) -> dict[str, Any]:
    params: dict[str, Any] = {"days": days, "metric": metric}
    if tenant_id:
        params["tenant_id"] = tenant_id
    return params


def build_graph_overview_params(
    default_tenant_id: str,
    *,
    tenant_id: str | None = None,
    scope_id: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"tenant_id": tenant_id or default_tenant_id}
    if scope_id is not None:
        params["scope_id"] = scope_id
    return params


def build_explore_graph_params(
    default_tenant_id: str,
    *,
    tenant_id: str | None = None,
    entity: str,
    scope_id: str = "default",
    depth: int = 2,
) -> dict[str, Any]:
    return {
        "tenant_id": tenant_id or default_tenant_id,
        "entity": entity,
        "scope_id": scope_id,
        "depth": depth,
    }


def build_search_graph_params(
    query: str,
    *,
    tenant_id: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    params: dict[str, Any] = {"query": query, "limit": limit}
    if tenant_id is not None:
        params["tenant_id"] = tenant_id
    return params


def build_test_retrieval_payload(
    default_tenant_id: str,
    query: str,
    *,
    tenant_id: str | None = None,
    max_results: int = 10,
    context_filter: list[str] | None = None,
    memory_types: list[str] | None = None,
    response_format: Literal["packet", "list", "llm_context"] = "list",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tenant_id": tenant_id or default_tenant_id,
        "query": query,
        "max_results": max_results,
        "format": response_format,
    }
    if context_filter is not None:
        payload["context_filter"] = context_filter
    if memory_types is not None:
        payload["memory_types"] = memory_types
    return payload


def build_jobs_params(
    *,
    tenant_id: str | None = None,
    job_type: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if tenant_id is not None:
        params["tenant_id"] = tenant_id
    if job_type is not None:
        params["job_type"] = job_type
    return params


def validate_iter_memory_types(memory_types: list[Any] | None) -> None:
    if memory_types and len(memory_types) > 1:
        raise ValueError(
            "iter_memories() supports at most one memory type; "
            f"got {len(memory_types)}. Pass a single type or None."
        )


def build_iter_memories_params(
    *,
    page: int,
    batch_size: int,
    status: str | None,
    memory_types: list[Any] | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"page": page, "per_page": batch_size}
    if status is not None:
        params["status"] = status
    if memory_types:
        params["type"] = memory_types[0].value
    return params
