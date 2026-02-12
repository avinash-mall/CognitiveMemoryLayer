"""Asynchronous client for the CognitiveMemoryLayer API."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal, cast
from uuid import UUID

from cml.config import CMLConfig
from cml.exceptions import ConnectionError, TimeoutError
from cml.models import (
    CreateSessionRequest,
    ForgetRequest,
    ForgetResponse,
    HealthResponse,
    MemoryItem,
    MemoryType,
    ReadRequest,
    ReadResponse,
    SessionContextResponse,
    SessionResponse,
    StatsResponse,
    TurnRequest,
    TurnResponse,
    UpdateRequest,
    UpdateResponse,
    WriteRequest,
    WriteResponse,
)
from cml.transport import AsyncHTTPTransport
from cml.utils.converters import dashboard_item_to_memory_item
from cml.utils.logging import logger


class AsyncCognitiveMemoryLayer:
    """Asynchronous client for the CognitiveMemoryLayer memory API.

    Usage:
        async with AsyncCognitiveMemoryLayer(api_key="sk-...") as memory:
            health = await memory.health()
            await memory.write("User prefers dark mode")
            result = await memory.read("user preferences")
    """

    _loop: asyncio.AbstractEventLoop | None

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:8000",
        tenant_id: str = "default",
        *,
        config: CMLConfig | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        verify_ssl: bool = True,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = CMLConfig(
                api_key=api_key,
                base_url=base_url,
                tenant_id=tenant_id,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                verify_ssl=verify_ssl,
            )
        self._transport = AsyncHTTPTransport(self._config)
        self._closed = False
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            with contextlib.suppress(Exception):
                logger.warning(
                    "AsyncCognitiveMemoryLayer was not closed; use 'async with' or call close() to avoid connection leaks"
                )

    async def __aenter__(self) -> AsyncCognitiveMemoryLayer:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP connection."""
        await self._transport.close()
        self._closed = True

    def _ensure_same_loop(self) -> None:
        """Raise if the current event loop is not the one this client was created in."""
        if self._loop is not None and asyncio.get_running_loop() != self._loop:
            raise RuntimeError(
                "AsyncCognitiveMemoryLayer must be used in the same event loop it was created in."
            )

    async def health(self) -> HealthResponse:
        """Check server health. Returns HealthResponse with status, version, components."""
        self._ensure_same_loop()
        data = await self._transport.request("GET", "/health")
        return HealthResponse(**data)

    async def write(
        self,
        content: str,
        *,
        context_tags: list[str] | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
        agent_id: str | None = None,
        timestamp: datetime | None = None,
        eval_mode: bool = False,
    ) -> WriteResponse:
        """Store a memory.

        Args:
            content: Text to store.
            context_tags: Optional tags for filtering.
            session_id: Optional session to associate.
            memory_type: Optional type (fact, preference, episode, etc.).
            namespace: Optional namespace.
            metadata: Optional key-value metadata.
            turn_id: Optional turn identifier.
            agent_id: Optional agent identifier.
            timestamp: Optional event timestamp (defaults to now).
            eval_mode: If True, send X-Eval-Mode header; server returns eval_outcome and eval_reason (stored/skipped and write-gate reason). Useful for benchmarks.

        Returns:
            WriteResponse with success, memory_id, chunks_created; when eval_mode=True, also eval_outcome and eval_reason.
        """
        self._ensure_same_loop()
        body = WriteRequest(
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
        extra = {"X-Eval-Mode": "true"} if eval_mode else None
        data = await self._transport.request("POST", "/memory/write", json=body, extra_headers=extra)
        return WriteResponse(**data)

    async def read(
        self,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        response_format: Literal["packet", "list", "llm_context"] = "packet",
    ) -> ReadResponse:
        """Retrieve memories by query.

        Args:
            query: Search query.
            max_results: Max items to return (1-50).
            context_filter: Optional context tags filter.
            memory_types: Optional types to include.
            since: Optional start of time range.
            until: Optional end of time range.
            response_format: "packet", "list", or "llm_context".

        Returns:
            ReadResponse with memories, facts, preferences, episodes, context.
        """
        self._ensure_same_loop()
        body = ReadRequest(
            query=query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            format=response_format,  # alias for response_format in ReadRequest
        ).model_dump(exclude_none=True, by_alias=True, mode="json")
        data = await self._transport.request("POST", "/memory/read", json=body)
        return ReadResponse(**data)

    async def read_safe(
        self,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        response_format: Literal["packet", "list", "llm_context"] = "packet",
    ) -> ReadResponse:
        """Read with graceful degradation — returns empty result on connection/timeout failure."""
        self._ensure_same_loop()
        try:
            return await self.read(
                query,
                max_results=max_results,
                context_filter=context_filter,
                memory_types=memory_types,
                since=since,
                until=until,
                response_format=response_format,
            )
        except ConnectionError:
            logger.warning("CML server unreachable, returning empty context", exc_info=False)
            return ReadResponse(
                query=query,
                memories=[],
                total_count=0,
                elapsed_ms=0.0,
            )
        except TimeoutError:
            logger.warning("CML request timed out, returning empty context", exc_info=False)
            return ReadResponse(
                query=query,
                memories=[],
                total_count=0,
                elapsed_ms=0.0,
            )

    async def turn(
        self,
        user_message: str,
        *,
        assistant_response: str | None = None,
        session_id: str | None = None,
        max_context_tokens: int = 1500,
        timestamp: datetime | None = None,
    ) -> TurnResponse:
        """Process a conversational turn (retrieve + store in one call).

        Args:
            user_message: User message for this turn.
            assistant_response: Optional assistant reply to store.
            session_id: Optional session id.
            max_context_tokens: Max tokens for retrieved context.
            timestamp: Optional event timestamp (defaults to now).

        Returns:
            TurnResponse with memory_context, counts, reconsolidation flag.
        """
        self._ensure_same_loop()
        body = TurnRequest(
            user_message=user_message,
            assistant_response=assistant_response,
            session_id=session_id,
            max_context_tokens=max_context_tokens,
            timestamp=timestamp,
        ).model_dump(exclude_none=True, mode="json")
        data = await self._transport.request("POST", "/memory/turn", json=body)
        return TurnResponse(**data)

    async def update(
        self,
        memory_id: UUID,
        *,
        text: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        feedback: str | None = None,
    ) -> UpdateResponse:
        """Update an existing memory.

        Args:
            memory_id: ID of the memory to update.
            text: New text (optional).
            confidence: New confidence (optional).
            importance: New importance (optional).
            metadata: New or merged metadata (optional).
            feedback: Optional feedback label: "correct", "incorrect", or "outdated".

        Returns:
            UpdateResponse with success, memory_id, version.
        """
        self._ensure_same_loop()
        body = UpdateRequest(
            memory_id=memory_id,
            text=text,
            confidence=confidence,
            importance=importance,
            metadata=metadata,
            feedback=feedback,
        ).model_dump(exclude_none=True, mode="json")
        data = await self._transport.request("POST", "/memory/update", json=body)
        return UpdateResponse(**data)

    async def forget(
        self,
        *,
        memory_ids: list[UUID] | None = None,
        query: str | None = None,
        before: datetime | None = None,
        action: Literal["delete", "archive", "silence"] = "delete",
    ) -> ForgetResponse:
        """Forget (delete, archive, or silence) memories.

        Args:
            memory_ids: Optional list of memory IDs to forget.
            query: Optional query to match memories to forget.
            before: Optional cutoff datetime; forget memories before this time.
            action: "delete", "archive", or "silence".

        Returns:
            ForgetResponse with success and affected_count.

        Raises:
            ValueError: If none of memory_ids, query, or before is provided.
        """
        if not memory_ids and not query and before is None:
            raise ValueError("At least one of memory_ids, query, or before must be provided")
        self._ensure_same_loop()
        body = ForgetRequest(
            memory_ids=memory_ids,
            query=query,
            before=before,
            action=action,
        ).model_dump(exclude_none=True, mode="json")
        data = await self._transport.request("POST", "/memory/forget", json=body)
        return ForgetResponse(**data)

    async def stats(self) -> StatsResponse:
        """Get memory statistics for the current tenant.

        Returns:
            StatsResponse with counts, averages, date range, size.
        """
        self._ensure_same_loop()
        data = await self._transport.request("GET", "/memory/stats")
        return StatsResponse(**data)

    async def create_session(
        self,
        *,
        name: str | None = None,
        ttl_hours: int = 24,
        metadata: dict[str, Any] | None = None,
    ) -> SessionResponse:
        """Create a new memory session.

        Args:
            name: Optional session name.
            ttl_hours: Time-to-live in hours (default 24).
            metadata: Optional metadata.

        Returns:
            SessionResponse with session_id, created_at, expires_at.
        """
        self._ensure_same_loop()
        body = CreateSessionRequest(
            name=name,
            ttl_hours=ttl_hours,
            metadata=metadata or {},
        ).model_dump(exclude_none=True, mode="json")
        data = await self._transport.request("POST", "/session/create", json=body)
        return SessionResponse(**data)

    async def get_session_context(self, session_id: str) -> SessionContextResponse:
        """Get full context for a session (messages, tool_results, scratch_pad).

        Args:
            session_id: Session ID.

        Returns:
            SessionContextResponse with messages, tool_results, scratch_pad, context_string.
        """
        self._ensure_same_loop()
        data = await self._transport.request("GET", f"/session/{session_id}/context")
        return SessionContextResponse(**data)

    @asynccontextmanager
    async def session(
        self,
        *,
        name: str | None = None,
        ttl_hours: int = 24,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[AsyncSessionScope]:
        """Create a session-scoped memory context. All operations use the same session_id."""
        self._ensure_same_loop()
        session_response = await self.create_session(
            name=name, ttl_hours=ttl_hours, metadata=metadata
        )
        scope = AsyncSessionScope(self, session_response.session_id)
        try:
            yield scope
        finally:
            pass

    async def delete_all(self, *, confirm: bool = False) -> int:
        """Delete all memories for the current tenant. Requires confirm=True.

        Note:
            The server may not yet expose DELETE /api/v1/memory/all; the call
            may return 404 until the endpoint is added.

        Args:
            confirm: Must be True to execute; otherwise ValueError is raised.

        Returns:
            Number of affected memories.

        Raises:
            ValueError: If confirm is not True.
        """
        if not confirm:
            raise ValueError("delete_all requires confirm=True to avoid accidental data loss")
        self._ensure_same_loop()
        data = await self._transport.request("DELETE", "/memory/all", use_admin_key=True)
        return cast("int", data.get("affected_count", 0))

    async def get_context(
        self,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> str:
        """Convenience: retrieve memories formatted as LLM context string.

        Args:
            query: Search query.
            max_results: Max items (1-50).
            context_filter: Optional context tags.
            memory_types: Optional types.
            since: Optional start of time range.
            until: Optional end of time range.

        Returns:
            Formatted context string (result.context).
        """
        self._ensure_same_loop()
        result = await self.read(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            response_format="llm_context",
        )
        return result.context

    async def remember(
        self,
        content: str,
        *,
        context_tags: list[str] | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
        agent_id: str | None = None,
        timestamp: datetime | None = None,
        eval_mode: bool = False,
    ) -> WriteResponse:
        """Alias for write(): store a memory."""
        self._ensure_same_loop()
        return await self.write(
            content,
            context_tags=context_tags,
            session_id=session_id,
            memory_type=memory_type,
            namespace=namespace,
            metadata=metadata,
            turn_id=turn_id,
            agent_id=agent_id,
            timestamp=timestamp,
            eval_mode=eval_mode,
        )

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        response_format: Literal["packet", "list", "llm_context"] = "packet",
    ) -> ReadResponse:
        """Alias for read(): retrieve memories by query."""
        self._ensure_same_loop()
        return await self.read(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            response_format=response_format,
        )

    # ---- Phase 5: Admin operations ----

    async def consolidate(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Trigger memory consolidation (episodic to semantic migration). Requires admin API key."""
        self._ensure_same_loop()
        payload: dict[str, Any] = {
            "tenant_id": tenant_id or self._config.tenant_id,
        }
        if user_id is not None:
            payload["user_id"] = user_id
        return await self._transport.request(
            "POST",
            "/dashboard/consolidate",
            json=payload,
            use_admin_key=True,
        )

    async def run_forgetting(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        dry_run: bool = True,
        max_memories: int = 5000,
    ) -> dict[str, Any]:
        """Trigger active forgetting cycle. Requires admin API key."""
        self._ensure_same_loop()
        payload: dict[str, Any] = {
            "tenant_id": tenant_id or self._config.tenant_id,
            "dry_run": dry_run,
            "max_memories": max_memories,
        }
        if user_id is not None:
            payload["user_id"] = user_id
        return await self._transport.request(
            "POST",
            "/dashboard/forget",
            json=payload,
            use_admin_key=True,
        )

    # ---- Phase 5: Batch operations ----

    async def batch_write(
        self,
        items: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ) -> list[WriteResponse]:
        """Write multiple memories (sequential)."""
        self._ensure_same_loop()
        for i, item in enumerate(items):
            if not isinstance(item, dict) or "content" not in item:
                raise ValueError(
                    f"Each item must be a dict with a 'content' key; item at index {i} is invalid"
                )
        result: list[WriteResponse] = []
        for item in items:
            resp = await self.write(
                content=item["content"],
                context_tags=item.get("context_tags"),
                session_id=session_id or item.get("session_id"),
                memory_type=item.get("memory_type"),
                namespace=namespace or item.get("namespace"),
                metadata=item.get("metadata"),
                agent_id=item.get("agent_id"),
            )
            result.append(resp)
        return result

    async def batch_read(
        self,
        queries: list[str],
        *,
        max_results: int = 10,
        response_format: Literal["packet", "list", "llm_context"] = "packet",
    ) -> list[ReadResponse]:
        """Execute multiple read queries concurrently."""
        self._ensure_same_loop()
        tasks = [
            self.read(q, max_results=max_results, response_format=response_format) for q in queries
        ]
        return list(await asyncio.gather(*tasks))

    # ---- Phase 5: Tenant management ----

    async def set_tenant(self, tenant_id: str) -> None:
        """Switch the active tenant for subsequent operations."""
        self._ensure_same_loop()
        self._config.tenant_id = tenant_id
        await self._transport.close()

    @property
    def tenant_id(self) -> str:
        """Current active tenant ID."""
        return self._config.tenant_id

    async def list_tenants(self) -> list[dict[str, Any]]:
        """List all tenants and their memory counts (admin only)."""
        self._ensure_same_loop()
        data = await self._transport.request(
            "GET",
            "/dashboard/tenants",
            use_admin_key=True,
        )
        return cast("list[dict[str, Any]]", data.get("tenants", []))

    # ---- Phase 5: Event log ----

    async def get_events(
        self,
        *,
        limit: int = 50,
        page: int = 1,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Query the event log (admin only)."""
        self._ensure_same_loop()
        params: dict[str, Any] = {"per_page": limit, "page": page}
        if event_type is not None:
            params["event_type"] = event_type
        if since is not None:
            params["since"] = since.isoformat()
        return await self._transport.request(
            "GET",
            "/dashboard/events",
            params=params,
            use_admin_key=True,
        )

    # ---- Phase 5: Component health ----

    async def component_health(self) -> dict[str, Any]:
        """Get detailed health status of all CML components (admin only)."""
        self._ensure_same_loop()
        return await self._transport.request(
            "GET",
            "/dashboard/components",
            use_admin_key=True,
        )

    # ---- Phase 5: Namespace isolation ----

    def with_namespace(self, namespace: str) -> AsyncNamespacedClient:
        """Create a namespace-scoped view of this client."""
        return AsyncNamespacedClient(self, namespace)

    # ---- Phase 5: Memory iteration ----

    async def iter_memories(
        self,
        *,
        memory_types: list[MemoryType] | None = None,
        status: str | None = "active",
        batch_size: int = 100,
    ) -> AsyncIterator[MemoryItem]:
        """Iterate over all memories with automatic pagination (admin only).

        Raises:
            ValueError: If more than one memory type is requested (API supports single type only).
        """
        if memory_types and len(memory_types) > 1:
            raise ValueError(
                "iter_memories() supports at most one memory type; "
                f"got {len(memory_types)}. Pass a single type or None."
            )
        self._ensure_same_loop()
        page = 1
        while True:
            params: dict[str, Any] = {
                "page": page,
                "per_page": batch_size,
            }
            if status is not None:
                params["status"] = status
            if memory_types:
                if len(memory_types) > 1:
                    raise ValueError(
                        f"iter_memories currently supports at most one memory_types filter; got {len(memory_types)} types"
                    )
                params["type"] = memory_types[0].value
            data = await self._transport.request(
                "GET",
                "/dashboard/memories",
                params=params,
                use_admin_key=True,
            )
            items = data.get("items", [])
            if not items:
                break
            for raw in items:
                yield dashboard_item_to_memory_item(raw)
            if page >= data.get("total_pages", 1):
                break
            page += 1


class AsyncSessionScope:
    """Session-scoped wrapper: injects session_id into write/turn/remember."""

    def __init__(self, parent: AsyncCognitiveMemoryLayer, session_id: str) -> None:
        self._parent = parent
        self.session_id = session_id

    async def write(
        self,
        content: str,
        *,
        context_tags: list[str] | None = None,
        memory_type: MemoryType | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
        agent_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> WriteResponse:
        return await self._parent.write(
            content,
            context_tags=context_tags,
            session_id=self.session_id,
            memory_type=memory_type,
            namespace=namespace,
            metadata=metadata,
            turn_id=turn_id,
            agent_id=agent_id,
            timestamp=timestamp,
        )

    async def read(
        self,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        response_format: Literal["packet", "list", "llm_context"] = "packet",
    ) -> ReadResponse:
        return await self._parent.read(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            response_format=response_format,
        )

    async def turn(
        self,
        user_message: str,
        *,
        assistant_response: str | None = None,
        max_context_tokens: int = 1500,
        timestamp: datetime | None = None,
    ) -> TurnResponse:
        return await self._parent.turn(
            user_message,
            assistant_response=assistant_response,
            session_id=self.session_id,
            max_context_tokens=max_context_tokens,
            timestamp=timestamp,
        )

    async def remember(
        self,
        content: str,
        *,
        context_tags: list[str] | None = None,
        memory_type: MemoryType | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
        agent_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> WriteResponse:
        return await self.write(
            content,
            context_tags=context_tags,
            memory_type=memory_type,
            namespace=namespace,
            metadata=metadata,
            turn_id=turn_id,
            agent_id=agent_id,
            timestamp=timestamp,
        )


class AsyncNamespacedClient:
    """Namespace-scoped wrapper around an async CML client."""

    def __init__(self, parent: AsyncCognitiveMemoryLayer, namespace: str) -> None:
        self._parent = parent
        self._namespace = namespace

    async def write(
        self,
        content: str,
        *,
        context_tags: list[str] | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
        agent_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> WriteResponse:
        return await self._parent.write(
            content,
            context_tags=context_tags,
            session_id=session_id,
            memory_type=memory_type,
            namespace=namespace or self._namespace,
            metadata=metadata,
            turn_id=turn_id,
            agent_id=agent_id,
            timestamp=timestamp,
        )

    async def read(
        self,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        response_format: Literal["packet", "list", "llm_context"] = "packet",
    ) -> ReadResponse:
        return await self._parent.read(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            response_format=response_format,
        )

    async def turn(
        self,
        user_message: str,
        *,
        assistant_response: str | None = None,
        session_id: str | None = None,
        max_context_tokens: int = 1500,
    ) -> TurnResponse:
        return await self._parent.turn(
            user_message,
            assistant_response=assistant_response,
            session_id=session_id,
            max_context_tokens=max_context_tokens,
        )

    async def update(
        self,
        memory_id: UUID,
        *,
        text: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        feedback: str | None = None,
    ) -> UpdateResponse:
        return await self._parent.update(
            memory_id,
            text=text,
            confidence=confidence,
            importance=importance,
            metadata=metadata,
            feedback=feedback,
        )

    async def forget(
        self,
        *,
        memory_ids: list[UUID] | None = None,
        query: str | None = None,
        before: datetime | None = None,
        action: Literal["delete", "archive", "silence"] = "delete",
    ) -> ForgetResponse:
        return await self._parent.forget(
            memory_ids=memory_ids,
            query=query,
            before=before,
            action=action,
        )

    async def stats(self) -> StatsResponse:
        return await self._parent.stats()

    async def create_session(
        self,
        *,
        name: str | None = None,
        ttl_hours: int = 24,
        metadata: dict[str, Any] | None = None,
    ) -> SessionResponse:
        return await self._parent.create_session(
            name=name,
            ttl_hours=ttl_hours,
            metadata=metadata,
        )

    async def get_session_context(self, session_id: str) -> SessionContextResponse:
        return await self._parent.get_session_context(session_id)

    async def delete_all(self, *, confirm: bool = False) -> int:
        return await self._parent.delete_all(confirm=confirm)

    async def get_context(
        self,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> str:
        return await self._parent.get_context(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
        )

    async def remember(
        self,
        content: str,
        *,
        context_tags: list[str] | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
        agent_id: str | None = None,
    ) -> WriteResponse:
        return await self.write(
            content,
            context_tags=context_tags,
            session_id=session_id,
            memory_type=memory_type,
            namespace=namespace,
            metadata=metadata,
            turn_id=turn_id,
            agent_id=agent_id,
        )

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        response_format: Literal["packet", "list", "llm_context"] = "packet",
    ) -> ReadResponse:
        return await self.read(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            response_format=response_format,
        )

    async def consolidate(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._parent.consolidate(tenant_id=tenant_id, user_id=user_id)

    async def run_forgetting(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        dry_run: bool = True,
        max_memories: int = 5000,
    ) -> dict[str, Any]:
        return await self._parent.run_forgetting(
            tenant_id=tenant_id,
            user_id=user_id,
            dry_run=dry_run,
            max_memories=max_memories,
        )

    async def batch_write(
        self,
        items: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ) -> list[WriteResponse]:
        return await self._parent.batch_write(
            items,
            session_id=session_id,
            namespace=namespace or self._namespace,
        )

    async def batch_read(
        self,
        queries: list[str],
        *,
        max_results: int = 10,
        response_format: Literal["packet", "list", "llm_context"] = "packet",
    ) -> list[ReadResponse]:
        return await self._parent.batch_read(
            queries,
            max_results=max_results,
            response_format=response_format,
        )

    async def set_tenant(self, tenant_id: str) -> None:
        return await self._parent.set_tenant(tenant_id)

    @property
    def tenant_id(self) -> str:
        return self._parent.tenant_id

    async def list_tenants(self) -> list[dict[str, Any]]:
        return await self._parent.list_tenants()

    async def get_events(
        self,
        *,
        limit: int = 50,
        page: int = 1,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        return await self._parent.get_events(
            limit=limit,
            page=page,
            event_type=event_type,
            since=since,
        )

    async def component_health(self) -> dict[str, Any]:
        return await self._parent.component_health()

    async def iter_memories(
        self,
        *,
        memory_types: list[MemoryType] | None = None,
        status: str | None = "active",
        batch_size: int = 100,
    ) -> AsyncIterator[MemoryItem]:
        async for item in self._parent.iter_memories(
            memory_types=memory_types,
            status=status,
            batch_size=batch_size,
        ):
            yield item
