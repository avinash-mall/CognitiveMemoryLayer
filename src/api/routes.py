"""API routes for memory operations. Holistic: tenant-only, no scopes."""

import json
import traceback
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.config import get_settings
from ..memory.orchestrator import MemoryOrchestrator
from ..memory.seamless_provider import SeamlessMemoryProvider
from ..utils.metrics import MEMORY_READS, MEMORY_WRITES
from .auth import AuthContext, get_auth_context, require_admin_permission, require_write_permission
from .schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    ForgetRequest,
    ForgetResponse,
    MemoryItem,
    MemoryStats,
    ProcessTurnRequest,
    ProcessTurnResponse,
    ReadMemoryRequest,
    ReadMemoryResponse,
    SessionContextResponse,
    UpdateMemoryRequest,
    UpdateMemoryResponse,
    WriteMemoryRequest,
    WriteMemoryResponse,
)

logger = structlog.get_logger()
router = APIRouter(tags=["memory"])


def get_orchestrator(request: Request) -> MemoryOrchestrator:
    """Get memory orchestrator from app state."""
    return request.app.state.orchestrator


def _safe_500_detail(e: Exception) -> str:
    """SEC-04: avoid leaking internal details in 500 responses unless debug."""
    return str(e) if get_settings().debug else "Internal server error"


def _to_memory_item(mem) -> MemoryItem:
    """Convert a retrieved memory (with .record, .relevance_score) to MemoryItem."""
    t = mem.record.type
    type_str = t.value if hasattr(t, "value") else str(t)
    return MemoryItem(
        id=mem.record.id,
        text=mem.record.text,
        type=type_str,
        confidence=mem.record.confidence,
        relevance=mem.relevance_score,
        timestamp=mem.record.timestamp,
        metadata=mem.record.metadata or {},
    )


# ---- General Memory API ----


@router.post("/memory/write", response_model=WriteMemoryResponse)
async def write_memory(
    request: Request,
    body: WriteMemoryRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Store new information in memory. Holistic: tenant-only. Set X-Eval-Mode: true for eval outcome/reason."""
    eval_mode = (request.headers.get("X-Eval-Mode") or "").strip().lower() in ("true", "1", "yes")
    try:
        result = await orchestrator.write(
            tenant_id=auth.tenant_id,
            content=body.content,
            context_tags=body.context_tags,
            session_id=body.session_id,
            memory_type=body.memory_type,
            metadata=body.metadata,
            turn_id=body.turn_id,
            agent_id=body.agent_id,
            namespace=body.namespace,
            timestamp=body.timestamp,
            eval_mode=eval_mode,
        )
        MEMORY_WRITES.labels(tenant_id=auth.tenant_id, status="success").inc()
        return WriteMemoryResponse(
            success=True,
            memory_id=result.get("memory_id"),
            chunks_created=result.get("chunks_created", 0),
            message=result.get("message", "Memory stored successfully"),
            eval_outcome=result.get("eval_outcome"),
            eval_reason=result.get("eval_reason"),
        )
    except Exception as e:
        MEMORY_WRITES.labels(tenant_id=auth.tenant_id, status="error").inc()
        logger.error(
            "memory_write_failed",
            tenant_id=auth.tenant_id,
            error=str(e),
            traceback=traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


@router.post("/memory/turn", response_model=ProcessTurnResponse)
async def process_turn(
    body: ProcessTurnRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """
    Process a conversation turn with seamless memory:
    - Auto-retrieves relevant context for the user message
    - Auto-stores salient information (user message and optional assistant response)
    - Returns formatted memory context for LLM injection
    """
    try:
        provider = SeamlessMemoryProvider(
            orchestrator,
            max_context_tokens=body.max_context_tokens,
            auto_store=True,
        )
        result = await provider.process_turn(
            tenant_id=auth.tenant_id,
            user_message=body.user_message,
            assistant_response=body.assistant_response,
            session_id=body.session_id,
            timestamp=body.timestamp,
        )
        return ProcessTurnResponse(
            memory_context=result.memory_context,
            memories_retrieved=len(result.injected_memories),
            memories_stored=result.stored_count,
            reconsolidation_applied=result.reconsolidation_applied,
        )
    except Exception as e:
        logger.exception("process_turn_failed", tenant_id=auth.tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


@router.post("/memory/read", response_model=ReadMemoryResponse)
async def read_memory(
    body: ReadMemoryRequest,
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Retrieve relevant memories for a query. Holistic: tenant-only."""
    MEMORY_READS.labels(tenant_id=auth.tenant_id).inc()
    start = datetime.now(UTC)
    try:
        memory_type_values = (
            [mt.value if hasattr(mt, "value") else str(mt) for mt in body.memory_types]
            if body.memory_types
            else None
        )
        packet = await orchestrator.read(
            tenant_id=auth.tenant_id,
            query=body.query,
            max_results=body.max_results,
            context_filter=body.context_filter,
            memory_types=memory_type_values,
            since=body.since,
            until=body.until,
        )

        elapsed_ms = (datetime.now(UTC) - start).total_seconds() * 1000

        all_memories = [_to_memory_item(m) for m in packet.all_memories]

        if body.format == "list":
            # "list" format: flat list only, no categorized buckets
            facts = []
            preferences = []
            episodes = []
        else:
            facts = [_to_memory_item(m) for m in packet.facts]
            preferences = [_to_memory_item(m) for m in packet.preferences]
            episodes = [_to_memory_item(m) for m in packet.recent_episodes]

        llm_context = None
        if body.format == "llm_context":
            from ..retrieval.packet_builder import MemoryPacketBuilder

            builder = MemoryPacketBuilder()
            llm_context = builder.to_llm_context(packet, max_tokens=2000)

        return ReadMemoryResponse(
            query=body.query,
            memories=all_memories,
            facts=facts,
            preferences=preferences,
            episodes=episodes,
            llm_context=llm_context,
            total_count=len(all_memories),
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:
        logger.exception("read_memory_failed", tenant_id=auth.tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


@router.post("/memory/update", response_model=UpdateMemoryResponse)
async def update_memory(
    body: UpdateMemoryRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Update an existing memory. Supports feedback for reconsolidation. Holistic: tenant-only."""
    try:
        result = await orchestrator.update(
            tenant_id=auth.tenant_id,
            memory_id=body.memory_id,
            text=body.text,
            confidence=body.confidence,
            importance=body.importance,
            metadata=body.metadata,
            feedback=body.feedback,
        )
        return UpdateMemoryResponse(
            success=True,
            memory_id=body.memory_id,
            version=result.get("version", 1),
            message="Memory updated successfully",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("update_memory_failed", tenant_id=auth.tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


@router.post("/memory/forget", response_model=ForgetResponse)
async def forget_memory(
    body: ForgetRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Forget (delete/archive/silence) memories. Holistic: tenant-only."""
    try:
        result = await orchestrator.forget(
            tenant_id=auth.tenant_id,
            memory_ids=body.memory_ids,
            query=body.query,
            before=body.before,
            action=body.action,
        )
        return ForgetResponse(
            success=True,
            affected_count=result.get("affected_count", 0),
            message=f"{result.get('affected_count', 0)} memories {body.action}d",
        )
    except Exception as e:
        logger.exception("forget_memory_failed", tenant_id=auth.tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


@router.get("/memory/stats", response_model=MemoryStats)
async def get_memory_stats(
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Get memory statistics for tenant. Holistic: tenant-only."""
    try:
        stats = await orchestrator.get_stats(tenant_id=auth.tenant_id)
        return MemoryStats(**stats)
    except Exception as e:
        logger.exception("get_memory_stats_failed", tenant_id=auth.tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


# ---- Session-based API (convenience for scope=SESSION) ----


@router.post("/session/create", response_model=CreateSessionResponse)
async def create_session(
    request: Request,
    body: CreateSessionRequest,
    auth: AuthContext = Depends(require_write_permission),
):
    """Create a new memory session. Returns session_id for subsequent calls. Persisted in Redis."""
    session_id = str(uuid4())
    now = datetime.now(UTC)
    ttl_hours = body.ttl_hours if body.ttl_hours is not None else 24
    expires_at = now + timedelta(hours=ttl_hours)
    ttl_seconds = max(1, int(ttl_hours * 3600))
    if (
        hasattr(request.app.state, "db")
        and request.app.state.db
        and getattr(request.app.state.db, "redis", None)
    ):
        payload = json.dumps(
            {
                "tenant_id": auth.tenant_id,
                "created_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
            }
        )
        await request.app.state.db.redis.setex(f"session:{session_id}", ttl_seconds, payload)
    return CreateSessionResponse(
        session_id=session_id,
        created_at=now,
        expires_at=expires_at,
    )


@router.post("/session/{session_id}/write", response_model=WriteMemoryResponse)
async def session_write(
    request: Request,
    session_id: str,
    body: WriteMemoryRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Write to memory with session_id for origin tracking. Set X-Eval-Mode: true for eval outcome/reason."""
    eval_mode = (request.headers.get("X-Eval-Mode") or "").strip().lower() in ("true", "1", "yes")
    try:
        result = await orchestrator.write(
            tenant_id=auth.tenant_id,
            content=body.content,
            context_tags=body.context_tags or ["conversation"],
            session_id=session_id,
            memory_type=body.memory_type,
            metadata=body.metadata,
            turn_id=body.turn_id,
            agent_id=body.agent_id,
            namespace=body.namespace,
            timestamp=body.timestamp,
            eval_mode=eval_mode,
        )
        MEMORY_WRITES.labels(tenant_id=auth.tenant_id, status="success").inc()
        return WriteMemoryResponse(
            success=True,
            memory_id=result.get("memory_id"),
            chunks_created=result.get("chunks_created", 0),
            message=result.get("message", "Memory stored successfully"),
            eval_outcome=result.get("eval_outcome"),
            eval_reason=result.get("eval_reason"),
        )
    except Exception as e:
        MEMORY_WRITES.labels(tenant_id=auth.tenant_id, status="error").inc()
        logger.exception("session_write_failed", tenant_id=auth.tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


@router.post("/session/{session_id}/read", response_model=ReadMemoryResponse)
async def session_read(
    session_id: str,
    body: ReadMemoryRequest,
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Read from memory. Holistic: tenant-only (session_id kept for API compatibility)."""
    MEMORY_READS.labels(tenant_id=auth.tenant_id).inc()
    start = datetime.now(UTC)
    try:
        memory_type_values = (
            [mt.value if hasattr(mt, "value") else str(mt) for mt in body.memory_types]
            if body.memory_types
            else None
        )
        packet = await orchestrator.read(
            tenant_id=auth.tenant_id,
            query=body.query,
            max_results=body.max_results,
            context_filter=body.context_filter,
            memory_types=memory_type_values,
            since=body.since,
            until=body.until,
        )
        elapsed_ms = (datetime.now(UTC) - start).total_seconds() * 1000

        all_memories = [_to_memory_item(m) for m in packet.all_memories]

        if body.format == "list":
            facts = []
            preferences = []
            episodes = []
        else:
            facts = [_to_memory_item(m) for m in packet.facts]
            preferences = [_to_memory_item(m) for m in packet.preferences]
            episodes = [_to_memory_item(m) for m in packet.recent_episodes]

        llm_context = None
        if body.format == "llm_context":
            from ..retrieval.packet_builder import MemoryPacketBuilder

            builder = MemoryPacketBuilder()
            llm_context = builder.to_llm_context(packet, max_tokens=2000)

        return ReadMemoryResponse(
            query=body.query,
            memories=all_memories,
            facts=facts,
            preferences=preferences,
            episodes=episodes,
            llm_context=llm_context,
            total_count=len(all_memories),
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:
        logger.exception("session_read_failed", tenant_id=auth.tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


@router.get("/session/{session_id}/context", response_model=SessionContextResponse)
async def session_context(
    session_id: str,
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Get full session context for LLM injection."""
    try:
        ctx = await orchestrator.get_session_context(
            tenant_id=auth.tenant_id,
            session_id=session_id,
        )

        def to_memory_item(d):
            return MemoryItem(
                id=d["id"],
                text=d["text"],
                type=d["type"],
                confidence=d["confidence"],
                relevance=d["relevance"],
                timestamp=d["timestamp"],
                metadata=d.get("metadata") or {},
            )

        return SessionContextResponse(
            session_id=session_id,
            messages=[to_memory_item(m) for m in ctx.get("messages", [])],
            tool_results=[to_memory_item(t) for t in ctx.get("tool_results", [])],
            scratch_pad=[to_memory_item(s) for s in ctx.get("scratch_pad", [])],
            context_string=ctx.get("context_string", ""),
        )
    except Exception as e:
        logger.exception("session_context_failed", tenant_id=auth.tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


@router.delete("/memory/all")
async def delete_all_memories(
    auth: AuthContext = Depends(require_admin_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Delete all memories for the authenticated tenant. Admin-only (GDPR)."""
    try:
        affected = await orchestrator.delete_all(tenant_id=auth.tenant_id)
        return {"affected_count": affected}
    except Exception as e:
        logger.exception("delete_all_failed", tenant_id=auth.tenant_id, error=str(e))
        raise HTTPException(status_code=500, detail=_safe_500_detail(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}
