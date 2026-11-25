"""Asynchronous client for the CognitiveMemoryLayer API."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal, cast
from uuid import UUID

from cml import _endpoints as ep
from cml.config import CMLConfig
from cml.exceptions import ConnectionError, TimeoutError
from cml.models import (
    DashboardComponentsResponse,
    DashboardConfigResponse,
    DashboardFactListResponse,
    DashboardJobsResponse,
    DashboardLabileResponse,
    DashboardMemoryDetail,
    DashboardMemoryListResponse,
    DashboardOverview,
    DashboardRateLimitsResponse,
    DashboardRetrievalResponse,
    DashboardSessionsResponse,
    DashboardTenantsResponse,
    DashboardTimelineResponse,
    FeedbackResponse,
    ForgetResponse,
    GraphExploreResponse,
    GraphNeo4jConfigResponse,
    GraphSearchResponse,
    GraphStatsResponse,
    HealthResponse,
    MemoryItem,
    MemoryType,
    ReadResponse,
    RequestStatsResponse,
    SessionContextResponse,
    SessionResponse,
    SimulateResponse,
    StatsResponse,
    TurnResponse,
    UpdateResponse,
    WindowConsolidateResponse,
    WriteResponse,
)
from cml.transport import AsyncHTTPTransport
from cml.utils.converters import dashboard_item_to_memory_item
from cml.utils.logging import logger


class AsyncCognitiveMemoryLayer:
    """Asynchronous client for the CognitiveMemoryLayer memory API."""

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
        self._config = config or CMLConfig(
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
        """Check server health."""
        self._ensure_same_loop()
        return HealthResponse(**(await self._transport.request("GET", "/health")))

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
        media: list[dict[str, Any]] | None = None,
    ) -> WriteResponse:
        """Store a memory (with optional Perirhinal multimodal attachments)."""
        self._ensure_same_loop()
        body = ep.build_write_body(
            content,
            context_tags=context_tags,
            session_id=session_id,
            memory_type=memory_type,
            namespace=namespace,
            metadata=metadata,
            turn_id=turn_id,
            agent_id=agent_id,
            timestamp=timestamp,
            media=media,
        )
        data = await self._transport.request(
            "POST", "/memory/write", json=body, extra_headers=ep.eval_mode_headers(eval_mode)
        )
        return WriteResponse(**data)

    async def _write_session(
        self,
        session_id: str,
        content: str,
        *,
        context_tags: list[str] | None = None,
        memory_type: MemoryType | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
        agent_id: str | None = None,
        timestamp: datetime | None = None,
        eval_mode: bool = False,
    ) -> WriteResponse:
        """Store a memory using the session-scoped write endpoint."""
        self._ensure_same_loop()
        body = ep.build_write_body(
            content,
            context_tags=context_tags,
            memory_type=memory_type,
            namespace=namespace,
            metadata=metadata,
            turn_id=turn_id,
            agent_id=agent_id,
            timestamp=timestamp,
        )
        data = await self._transport.request(
            "POST",
            f"/session/{session_id}/write",
            json=body,
            extra_headers=ep.eval_mode_headers(eval_mode),
        )
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
        user_timezone: str | None = None,
    ) -> ReadResponse:
        """Retrieve memories by query."""
        self._ensure_same_loop()
        body = ep.build_read_body(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            response_format=response_format,
            user_timezone=user_timezone,
        )
        return ReadResponse(**(await self._transport.request("POST", "/memory/read", json=body)))

    async def _read_session(
        self,
        session_id: str,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        response_format: Literal["packet", "list", "llm_context"] = "packet",
        user_timezone: str | None = None,
    ) -> ReadResponse:
        """Retrieve memories scoped to a specific session id."""
        self._ensure_same_loop()
        body = ep.build_read_body(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            response_format=response_format,
            user_timezone=user_timezone,
        )
        return ReadResponse(
            **(await self._transport.request("POST", f"/session/{session_id}/read", json=body))
        )

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
        user_timezone: str | None = None,
    ) -> ReadResponse:
        """Read with graceful degradation on connection/timeout failure."""
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
                user_timezone=user_timezone,
            )
        except ConnectionError:
            logger.warning("CML server unreachable, returning empty context", exc_info=False)
            return ep.empty_read_response(query)
        except TimeoutError:
            logger.warning("CML request timed out, returning empty context", exc_info=False)
            return ep.empty_read_response(query)

    async def read_stream(
        self,
        query: str,
        *,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        user_timezone: str | None = None,
    ) -> AsyncIterator[MemoryItem]:
        """Stream memories incrementally via Server-Sent Events (SSE)."""
        self._ensure_same_loop()
        body = ep.build_read_stream_body(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            user_timezone=user_timezone,
        )
        async for data in self._transport.stream_sse("/memory/read/stream", json_body=body):
            yield MemoryItem(**data)

    async def turn(
        self,
        user_message: str,
        *,
        assistant_response: str | None = None,
        session_id: str | None = None,
        max_context_tokens: int = 1500,
        timestamp: datetime | None = None,
        user_timezone: str | None = None,
    ) -> TurnResponse:
        """Process a conversational turn (retrieve + store in one call)."""
        self._ensure_same_loop()
        body = ep.build_turn_body(
            user_message,
            assistant_response=assistant_response,
            session_id=session_id,
            max_context_tokens=max_context_tokens,
            timestamp=timestamp,
            user_timezone=user_timezone,
        )
        return TurnResponse(**(await self._transport.request("POST", "/memory/turn", json=body)))

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
        """Update an existing memory."""
        self._ensure_same_loop()
        body = ep.build_update_body(
            memory_id,
            text=text,
            confidence=confidence,
            importance=importance,
            metadata=metadata,
            feedback=feedback,
        )
        return UpdateResponse(
            **(await self._transport.request("POST", "/memory/update", json=body))
        )

    async def forget(
        self,
        *,
        memory_ids: list[UUID] | None = None,
        query: str | None = None,
        before: datetime | None = None,
        action: Literal["delete", "archive", "silence"] = "delete",
    ) -> ForgetResponse:
        """Forget (delete, archive, or silence) memories."""
        self._ensure_same_loop()
        body = ep.build_forget_body(
            memory_ids=memory_ids, query=query, before=before, action=action
        )
        return ForgetResponse(
            **(await self._transport.request("POST", "/memory/forget", json=body))
        )

    async def stats(self) -> StatsResponse:
        """Get memory statistics for the current tenant."""
        self._ensure_same_loop()
        return StatsResponse(**(await self._transport.request("GET", "/memory/stats")))

    async def create_session(
        self,
        *,
        name: str | None = None,
        ttl_hours: int = 24,
        metadata: dict[str, Any] | None = None,
    ) -> SessionResponse:
        """Create a new memory session."""
        self._ensure_same_loop()
        body = ep.build_create_session_body(name=name, ttl_hours=ttl_hours, metadata=metadata)
        return SessionResponse(
            **(await self._transport.request("POST", "/session/create", json=body))
        )

    async def get_session_context(self, session_id: str) -> SessionContextResponse:
        """Get full context for a session."""
        self._ensure_same_loop()
        return SessionContextResponse(
            **(await self._transport.request("GET", f"/session/{session_id}/context"))
        )

    @asynccontextmanager
    async def session(
        self,
        *,
        name: str | None = None,
        ttl_hours: int = 24,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[AsyncSessionScope]:
        """Create a session-scoped memory context."""
        self._ensure_same_loop()
        session_response = await self.create_session(
            name=name, ttl_hours=ttl_hours, metadata=metadata
        )
        yield AsyncSessionScope(self, session_response.session_id)

    async def delete_all(self, *, confirm: bool = False) -> int:
        """Delete all memories for the current tenant (requires confirm=True)."""
        ep.require_delete_all_confirmation(confirm)
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
        user_timezone: str | None = None,
    ) -> str:
        """Retrieve memories formatted as an LLM context string."""
        self._ensure_same_loop()
        result = await self.read(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            user_timezone=user_timezone,
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
        user_timezone: str | None = None,
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
            user_timezone=user_timezone,
        )

    # ---- Whole-Brain v3.0 operations ----

    async def consolidate_windowed(
        self,
        *,
        session_id: str | None = None,
        max_episodes: int = 2000,
        use_llm: bool | None = None,
    ) -> WindowConsolidateResponse:
        """Run Windowed Micro-Sleep Consolidation (Hyperedges, Proustian Cues, Conway, DMN)."""
        self._ensure_same_loop()
        body = ep.build_consolidate_windowed_body(
            session_id=session_id, max_episodes=max_episodes, use_llm=use_llm
        )
        return WindowConsolidateResponse(
            **(await self._transport.request("POST", "/memory/consolidate_windowed", json=body))
        )

    async def feedback(
        self,
        outcome_reward: float,
        *,
        expected_reward: float = 0.5,
        query: str | None = None,
        memory_ids: list[str] | None = None,
        correction_text: str | None = None,
        preferred_steps: list[str] | None = None,
        session_id: str | None = None,
    ) -> FeedbackResponse:
        """Submit Dopaminergic Reward Prediction Error (RPE) & Procedural feedback."""
        self._ensure_same_loop()
        body = ep.build_feedback_body(
            outcome_reward,
            expected_reward=expected_reward,
            query=query,
            memory_ids=memory_ids,
            correction_text=correction_text,
            preferred_steps=preferred_steps,
            session_id=session_id,
        )
        return FeedbackResponse(
            **(await self._transport.request("POST", "/memory/feedback", json=body))
        )

    async def simulate(
        self,
        scenario: str,
        *,
        max_results: int = 12,
        context_filter: list[str] | None = None,
    ) -> SimulateResponse:
        """Execute Hippocampal Forward Preplay / Counterfactual Simulation."""
        self._ensure_same_loop()
        body = ep.build_simulate_body(
            scenario, max_results=max_results, context_filter=context_filter
        )
        return SimulateResponse(
            **(await self._transport.request("POST", "/memory/simulate", json=body))
        )

    # ---- Admin & Dashboard operations ----

    async def consolidate(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Trigger sleep-cycle consolidation (requires admin API key)."""
        self._ensure_same_loop()
        payload = ep.build_tenant_user_payload(
            self._config.tenant_id, tenant_id=tenant_id, user_id=user_id
        )
        return await self._transport.request(
            "POST", "/dashboard/consolidate", json=payload, use_admin_key=True
        )

    async def run_forgetting(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        dry_run: bool = True,
        max_memories: int = 5000,
    ) -> dict[str, Any]:
        """Trigger active forgetting cycle (requires admin API key)."""
        self._ensure_same_loop()
        payload = ep.build_forgetting_payload(
            self._config.tenant_id,
            tenant_id=tenant_id,
            user_id=user_id,
            dry_run=dry_run,
            max_memories=max_memories,
        )
        return await self._transport.request(
            "POST", "/dashboard/forget", json=payload, use_admin_key=True
        )

    async def reconsolidate(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Release all labile state for a tenant (requires admin API key)."""
        self._ensure_same_loop()
        payload = ep.build_tenant_user_payload(
            self._config.tenant_id, tenant_id=tenant_id, user_id=user_id
        )
        return await self._transport.request(
            "POST", "/dashboard/reconsolidate", json=payload, use_admin_key=True
        )

    async def admin_consolidate(self, user_id: str) -> dict[str, Any]:
        """Trigger admin consolidate endpoint directly for a specific user."""
        self._ensure_same_loop()
        return await self._transport.request(
            "POST", f"/admin/consolidate/{user_id}", use_admin_key=True
        )

    async def admin_forget(self, user_id: str, *, dry_run: bool = True) -> dict[str, Any]:
        """Trigger admin forgetting endpoint directly for a specific user."""
        self._ensure_same_loop()
        return await self._transport.request(
            "POST", f"/admin/forget/{user_id}", params={"dry_run": dry_run}, use_admin_key=True
        )

    async def batch_write(
        self,
        items: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ) -> list[WriteResponse]:
        """Write multiple memories sequentially."""
        self._ensure_same_loop()
        ep.validate_batch_write_items(items)
        return [
            await self.write(
                content=item["content"],
                context_tags=item.get("context_tags"),
                session_id=session_id or item.get("session_id"),
                memory_type=item.get("memory_type"),
                namespace=namespace or item.get("namespace"),
                metadata=item.get("metadata"),
                agent_id=item.get("agent_id"),
            )
            for item in items
        ]

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

    async def set_tenant(self, tenant_id: str) -> None:
        """Switch the active tenant for subsequent operations."""
        self._ensure_same_loop()
        self._config.tenant_id = tenant_id
        await self._transport.close()

    @property
    def tenant_id(self) -> str:
        """Current active tenant ID."""
        return self._config.tenant_id

    async def list_tenants(self) -> DashboardTenantsResponse:
        """List all tenants and their memory counts (admin only)."""
        self._ensure_same_loop()
        return DashboardTenantsResponse(
            **(await self._transport.request("GET", "/dashboard/tenants", use_admin_key=True))
        )

    async def dashboard_overview(self, *, tenant_id: str | None = None) -> DashboardOverview:
        """Get comprehensive dashboard overview stats (admin only)."""
        self._ensure_same_loop()
        data = await self._transport.request(
            "GET",
            "/dashboard/overview",
            params=ep.optional_tenant_params(tenant_id),
            use_admin_key=True,
        )
        return DashboardOverview(**data)

    async def dashboard_memories(
        self,
        *,
        tenant_id: str | None = None,
        page: int = 1,
        per_page: int = 25,
        type: str | None = None,
        status: str | None = None,
        search: str | None = None,
        sort_by: str = "written_at",
        order: str = "desc",
    ) -> DashboardMemoryListResponse:
        """Get a paginated list of memories across the system (admin only)."""
        self._ensure_same_loop()
        params = ep.build_dashboard_memories_params(
            tenant_id=tenant_id,
            page=page,
            per_page=per_page,
            type=type,
            status=status,
            search=search,
            sort_by=sort_by,
            order=order,
        )
        return DashboardMemoryListResponse(
            **(
                await self._transport.request(
                    "GET", "/dashboard/memories", params=params, use_admin_key=True
                )
            )
        )

    async def dashboard_facts(
        self,
        *,
        tenant_id: str | None = None,
        category: str | None = None,
        current_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> DashboardFactListResponse:
        """Get semantic facts from dashboard API (admin only)."""
        self._ensure_same_loop()
        params = ep.build_dashboard_facts_params(
            tenant_id=tenant_id,
            category=category,
            current_only=current_only,
            limit=limit,
            offset=offset,
        )
        return DashboardFactListResponse(
            **(
                await self._transport.request(
                    "GET", "/dashboard/facts", params=params, use_admin_key=True
                )
            )
        )

    async def dashboard_invalidate_fact(self, fact_id: str) -> dict[str, Any]:
        """Invalidate a semantic fact by id (admin only)."""
        self._ensure_same_loop()
        return await self._transport.request(
            "POST", f"/dashboard/facts/{fact_id}/invalidate", use_admin_key=True
        )

    async def dashboard_export_memories(
        self, *, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Export memories as JSON (admin only)."""
        self._ensure_same_loop()
        data = await self._transport.request(
            "GET",
            "/dashboard/export/memories",
            params=ep.optional_tenant_params(tenant_id),
            use_admin_key=True,
        )
        return cast("list[dict[str, Any]]", data)

    async def dashboard_memory_detail(self, memory_id: UUID) -> DashboardMemoryDetail:
        """Get full detail for a single memory record (admin only)."""
        self._ensure_same_loop()
        return DashboardMemoryDetail(
            **(
                await self._transport.request(
                    "GET", f"/dashboard/memories/{memory_id}", use_admin_key=True
                )
            )
        )

    async def bulk_memory_action(
        self, memory_ids: list[UUID], action: Literal["archive", "silence", "delete"]
    ) -> dict[str, Any]:
        """Perform bulk actions on memories (admin only)."""
        self._ensure_same_loop()
        return await self._transport.request(
            "POST",
            "/dashboard/memories/bulk-action",
            json=ep.build_bulk_action_payload(memory_ids, action),
            use_admin_key=True,
        )

    async def dashboard_timeline(
        self,
        *,
        tenant_id: str | None = None,
        days: int = 30,
        metric: str = "memories",
    ) -> DashboardTimelineResponse:
        """Get timeline data for charts (admin only)."""
        self._ensure_same_loop()
        params = ep.build_timeline_params(tenant_id=tenant_id, days=days, metric=metric)
        return DashboardTimelineResponse(
            **(
                await self._transport.request(
                    "GET", "/dashboard/timeline", params=params, use_admin_key=True
                )
            )
        )

    async def component_health(self) -> DashboardComponentsResponse:
        """Get detailed health status of all CML components (admin only)."""
        self._ensure_same_loop()
        return DashboardComponentsResponse(
            **(await self._transport.request("GET", "/dashboard/components", use_admin_key=True))
        )

    async def get_sessions(self, *, tenant_id: str | None = None) -> DashboardSessionsResponse:
        """List active sessions and memory counts per session (admin only)."""
        self._ensure_same_loop()
        return DashboardSessionsResponse(
            **(
                await self._transport.request(
                    "GET",
                    "/dashboard/sessions",
                    params=ep.optional_tenant_params(tenant_id),
                    use_admin_key=True,
                )
            )
        )

    async def get_rate_limits(self) -> DashboardRateLimitsResponse:
        """Get current rate-limit usage per key (admin only)."""
        self._ensure_same_loop()
        return DashboardRateLimitsResponse(
            **(await self._transport.request("GET", "/dashboard/ratelimits", use_admin_key=True))
        )

    async def get_request_stats(self, *, hours: int = 24) -> RequestStatsResponse:
        """Get hourly request counts (admin only)."""
        self._ensure_same_loop()
        return RequestStatsResponse(
            **(
                await self._transport.request(
                    "GET", "/dashboard/request-stats", params={"hours": hours}, use_admin_key=True
                )
            )
        )

    async def get_graph_stats(self) -> GraphStatsResponse:
        """Get knowledge graph statistics from Neo4j (admin only)."""
        self._ensure_same_loop()
        return GraphStatsResponse(
            **(await self._transport.request("GET", "/dashboard/graph/stats", use_admin_key=True))
        )

    async def graph_overview(
        self,
        *,
        tenant_id: str | None = None,
        scope_id: str | None = None,
    ) -> GraphExploreResponse:
        """Get graph overview for a tenant (admin only)."""
        self._ensure_same_loop()
        params = ep.build_graph_overview_params(
            self._config.tenant_id, tenant_id=tenant_id, scope_id=scope_id
        )
        return GraphExploreResponse(
            **(
                await self._transport.request(
                    "GET", "/dashboard/graph/overview", params=params, use_admin_key=True
                )
            )
        )

    async def explore_graph(
        self,
        *,
        tenant_id: str | None = None,
        entity: str,
        scope_id: str = "default",
        depth: int = 2,
    ) -> GraphExploreResponse:
        """Explore the neighborhood of an entity in the knowledge graph (admin only)."""
        self._ensure_same_loop()
        params = ep.build_explore_graph_params(
            self._config.tenant_id,
            tenant_id=tenant_id,
            entity=entity,
            scope_id=scope_id,
            depth=depth,
        )
        return GraphExploreResponse(
            **(
                await self._transport.request(
                    "GET", "/dashboard/graph/explore", params=params, use_admin_key=True
                )
            )
        )

    async def search_graph(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        limit: int = 25,
    ) -> GraphSearchResponse:
        """Search entities by name pattern in the knowledge graph (admin only)."""
        self._ensure_same_loop()
        params = ep.build_search_graph_params(query, tenant_id=tenant_id, limit=limit)
        return GraphSearchResponse(
            **(
                await self._transport.request(
                    "GET", "/dashboard/graph/search", params=params, use_admin_key=True
                )
            )
        )

    async def dashboard_neo4j_config(self) -> GraphNeo4jConfigResponse:
        """Get Neo4j connection configuration for the frontend graph visualizer (admin only)."""
        self._ensure_same_loop()
        return GraphNeo4jConfigResponse(
            **(
                await self._transport.request(
                    "GET", "/dashboard/graph/neo4j-config", use_admin_key=True
                )
            )
        )

    async def get_config(self) -> DashboardConfigResponse:
        """Get application configuration snapshot with secrets masked (admin only)."""
        self._ensure_same_loop()
        return DashboardConfigResponse(
            **(await self._transport.request("GET", "/dashboard/config", use_admin_key=True))
        )

    async def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Update editable config settings at runtime (admin only)."""
        self._ensure_same_loop()
        return await self._transport.request(
            "PUT", "/dashboard/config", json={"updates": updates}, use_admin_key=True
        )

    async def get_labile_status(self, *, tenant_id: str | None = None) -> DashboardLabileResponse:
        """Get labile memory / reconsolidation status (admin only)."""
        self._ensure_same_loop()
        return DashboardLabileResponse(
            **(
                await self._transport.request(
                    "GET",
                    "/dashboard/labile",
                    params=ep.optional_tenant_params(tenant_id),
                    use_admin_key=True,
                )
            )
        )

    async def test_retrieval(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        max_results: int = 10,
        context_filter: list[str] | None = None,
        memory_types: list[str] | None = None,
        response_format: Literal["packet", "list", "llm_context"] = "list",
    ) -> DashboardRetrievalResponse:
        """Test memory retrieval via the dashboard API (admin only)."""
        self._ensure_same_loop()
        payload = ep.build_test_retrieval_payload(
            self._config.tenant_id,
            query,
            tenant_id=tenant_id,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            response_format=response_format,
        )
        return DashboardRetrievalResponse(
            **(
                await self._transport.request(
                    "POST", "/dashboard/retrieval", json=payload, use_admin_key=True
                )
            )
        )

    async def get_jobs(
        self,
        *,
        tenant_id: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
    ) -> DashboardJobsResponse:
        """List recent consolidation/forgetting job history (admin only)."""
        self._ensure_same_loop()
        params = ep.build_jobs_params(tenant_id=tenant_id, job_type=job_type, limit=limit)
        return DashboardJobsResponse(
            **(
                await self._transport.request(
                    "GET", "/dashboard/jobs", params=params, use_admin_key=True
                )
            )
        )

    async def reset_database(self, confirm: bool = False) -> dict[str, Any]:
        """Reset the entire database, dropping all content (admin only)."""
        body = ep.require_reset_confirmation(confirm)
        self._ensure_same_loop()
        return await self._transport.request(
            "POST", "/dashboard/database/reset", json=body, use_admin_key=True
        )

    def with_namespace(self, namespace: str) -> AsyncNamespacedClient:
        """Create a namespace-scoped view of this client."""
        return AsyncNamespacedClient(self, namespace)

    async def iter_memories(
        self,
        *,
        memory_types: list[MemoryType] | None = None,
        status: str | None = "active",
        batch_size: int = 100,
    ) -> AsyncIterator[MemoryItem]:
        """Iterate over all memories with automatic pagination (admin only)."""
        ep.validate_iter_memory_types(memory_types)
        self._ensure_same_loop()
        page = 1
        while True:
            params = ep.build_iter_memories_params(
                page=page,
                batch_size=batch_size,
                status=status,
                memory_types=memory_types,
            )
            data = await self._transport.request(
                "GET", "/dashboard/memories", params=params, use_admin_key=True
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
    """Session-scoped wrapper for write/read/turn operations."""

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
        return await self._parent._write_session(
            self.session_id,
            content,
            context_tags=context_tags,
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
        user_timezone: str | None = None,
    ) -> ReadResponse:
        return await self._parent._read_session(
            self.session_id,
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            response_format=response_format,
            user_timezone=user_timezone,
        )

    async def turn(
        self,
        user_message: str,
        *,
        assistant_response: str | None = None,
        max_context_tokens: int = 1500,
        timestamp: datetime | None = None,
        user_timezone: str | None = None,
    ) -> TurnResponse:
        return await self._parent.turn(
            user_message,
            assistant_response=assistant_response,
            session_id=self.session_id,
            max_context_tokens=max_context_tokens,
            timestamp=timestamp,
            user_timezone=user_timezone,
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
            timestamp=timestamp,
        )

    async def batch_write(
        self,
        items: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ) -> list[WriteResponse]:
        return await self._parent.batch_write(
            items, session_id=session_id, namespace=namespace or self._namespace
        )

    async def read(self, query: str, **kwargs: Any) -> ReadResponse:
        return await self._parent.read(query, **kwargs)

    async def search(self, query: str, **kwargs: Any) -> ReadResponse:
        return await self.read(query, **kwargs)

    def read_stream(self, query: str, **kwargs: Any) -> AsyncIterator[MemoryItem]:
        return self._parent.read_stream(query, **kwargs)

    async def turn(self, user_message: str, **kwargs: Any) -> TurnResponse:
        return await self._parent.turn(user_message, **kwargs)

    async def update(self, memory_id: UUID, **kwargs: Any) -> UpdateResponse:
        return await self._parent.update(memory_id, **kwargs)

    async def forget(self, **kwargs: Any) -> ForgetResponse:
        return await self._parent.forget(**kwargs)

    async def stats(self) -> StatsResponse:
        return await self._parent.stats()

    async def create_session(self, **kwargs: Any) -> SessionResponse:
        return await self._parent.create_session(**kwargs)

    async def get_session_context(self, session_id: str) -> SessionContextResponse:
        return await self._parent.get_session_context(session_id)

    async def delete_all(self, *, confirm: bool = False) -> int:
        return await self._parent.delete_all(confirm=confirm)

    async def get_context(self, query: str, **kwargs: Any) -> str:
        return await self._parent.get_context(query, **kwargs)

    async def consolidate_windowed(self, **kwargs: Any) -> WindowConsolidateResponse:
        return await self._parent.consolidate_windowed(**kwargs)

    async def feedback(self, outcome_reward: float, **kwargs: Any) -> FeedbackResponse:
        return await self._parent.feedback(outcome_reward, **kwargs)

    async def simulate(self, scenario: str, **kwargs: Any) -> SimulateResponse:
        return await self._parent.simulate(scenario, **kwargs)

    async def consolidate(self, **kwargs: Any) -> dict[str, Any]:
        return await self._parent.consolidate(**kwargs)

    async def run_forgetting(self, **kwargs: Any) -> dict[str, Any]:
        return await self._parent.run_forgetting(**kwargs)

    async def reconsolidate(self, **kwargs: Any) -> dict[str, Any]:
        return await self._parent.reconsolidate(**kwargs)

    async def batch_read(self, queries: list[str], **kwargs: Any) -> list[ReadResponse]:
        return await self._parent.batch_read(queries, **kwargs)

    async def set_tenant(self, tenant_id: str) -> None:
        await self._parent.set_tenant(tenant_id)

    @property
    def tenant_id(self) -> str:
        return self._parent.tenant_id

    async def list_tenants(self) -> DashboardTenantsResponse:
        return await self._parent.list_tenants()

    async def dashboard_overview(self, *, tenant_id: str | None = None) -> DashboardOverview:
        return await self._parent.dashboard_overview(tenant_id=tenant_id)

    async def dashboard_memories(self, **kwargs: Any) -> DashboardMemoryListResponse:
        return await self._parent.dashboard_memories(**kwargs)

    async def dashboard_memory_detail(self, memory_id: UUID) -> DashboardMemoryDetail:
        return await self._parent.dashboard_memory_detail(memory_id)

    async def dashboard_facts(self, **kwargs: Any) -> DashboardFactListResponse:
        return await self._parent.dashboard_facts(**kwargs)

    async def dashboard_invalidate_fact(self, fact_id: str) -> dict[str, Any]:
        return await self._parent.dashboard_invalidate_fact(fact_id)

    async def dashboard_export_memories(
        self, *, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        return await self._parent.dashboard_export_memories(tenant_id=tenant_id)

    async def dashboard_timeline(self, **kwargs: Any) -> DashboardTimelineResponse:
        return await self._parent.dashboard_timeline(**kwargs)

    async def component_health(self) -> DashboardComponentsResponse:
        return await self._parent.component_health()

    def iter_memories(self, **kwargs: Any) -> AsyncIterator[MemoryItem]:
        return self._parent.iter_memories(**kwargs)

    async def get_sessions(self, *, tenant_id: str | None = None) -> DashboardSessionsResponse:
        return await self._parent.get_sessions(tenant_id=tenant_id)

    async def get_rate_limits(self) -> DashboardRateLimitsResponse:
        return await self._parent.get_rate_limits()

    async def get_request_stats(self, *, hours: int = 24) -> RequestStatsResponse:
        return await self._parent.get_request_stats(hours=hours)

    async def get_graph_stats(self) -> GraphStatsResponse:
        return await self._parent.get_graph_stats()

    async def graph_overview(self, **kwargs: Any) -> GraphExploreResponse:
        return await self._parent.graph_overview(**kwargs)

    async def explore_graph(self, **kwargs: Any) -> GraphExploreResponse:
        return await self._parent.explore_graph(**kwargs)

    async def search_graph(self, query: str, **kwargs: Any) -> GraphSearchResponse:
        return await self._parent.search_graph(query, **kwargs)

    async def dashboard_neo4j_config(self) -> GraphNeo4jConfigResponse:
        return await self._parent.dashboard_neo4j_config()

    async def get_config(self) -> DashboardConfigResponse:
        return await self._parent.get_config()

    async def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        return await self._parent.update_config(updates)

    async def get_labile_status(self, *, tenant_id: str | None = None) -> DashboardLabileResponse:
        return await self._parent.get_labile_status(tenant_id=tenant_id)

    async def test_retrieval(self, query: str, **kwargs: Any) -> DashboardRetrievalResponse:
        return await self._parent.test_retrieval(query, **kwargs)

    async def get_jobs(self, **kwargs: Any) -> DashboardJobsResponse:
        return await self._parent.get_jobs(**kwargs)

    async def reset_database(self, confirm: bool = False) -> dict[str, Any]:
        return await self._parent.reset_database(confirm=confirm)

    async def bulk_memory_action(
        self, memory_ids: list[UUID], action: Literal["archive", "silence", "delete"]
    ) -> dict[str, Any]:
        return await self._parent.bulk_memory_action(memory_ids, action)

    async def admin_consolidate(self, user_id: str) -> dict[str, Any]:
        return await self._parent.admin_consolidate(user_id)

    async def admin_forget(self, user_id: str, *, dry_run: bool = True) -> dict[str, Any]:
        return await self._parent.admin_forget(user_id, dry_run=dry_run)
