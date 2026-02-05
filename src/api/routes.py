"""API routes for memory operations."""
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.enums import MemoryScope
from ..utils.metrics import MEMORY_READS, MEMORY_WRITES
from .auth import AuthContext, get_auth_context, require_write_permission
from .schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    ForgetRequest,
    ForgetResponse,
    MemoryItem,
    MemoryStats,
    ReadMemoryRequest,
    ReadMemoryResponse,
    SessionContextResponse,
    UpdateMemoryRequest,
    UpdateMemoryResponse,
    WriteMemoryRequest,
    WriteMemoryResponse,
)
from ..memory.orchestrator import MemoryOrchestrator

router = APIRouter(tags=["memory"])


def get_orchestrator(request: Request) -> MemoryOrchestrator:
    """Get memory orchestrator from app state."""
    return request.app.state.orchestrator


# ---- General Memory API ----

@router.post("/memory/write", response_model=WriteMemoryResponse)
async def write_memory(
    body: WriteMemoryRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """
    Store new information in memory.
    Requires scope (SESSION, AGENT, NAMESPACE, GLOBAL) and scope_id.
    """
    try:
        result = await orchestrator.write(
            tenant_id=auth.tenant_id,
            content=body.content,
            memory_type=body.memory_type,
            metadata=body.metadata,
            turn_id=body.turn_id,
            agent_id=body.agent_id,
            scope=body.scope,
            scope_id=body.scope_id,
            namespace=body.namespace,
        )
        MEMORY_WRITES.labels(tenant_id=auth.tenant_id, status="success").inc()
        return WriteMemoryResponse(
            success=True,
            memory_id=result.get("memory_id"),
            chunks_created=result.get("chunks_created", 0),
            message="Memory stored successfully",
        )
    except Exception as e:
        MEMORY_WRITES.labels(tenant_id=auth.tenant_id, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/read", response_model=ReadMemoryResponse)
async def read_memory(
    body: ReadMemoryRequest,
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """
    Retrieve relevant memories for a query.
    Requires scope and scope_id to identify the memory space.
    """
    MEMORY_READS.labels(tenant_id=auth.tenant_id).inc()
    start = datetime.utcnow()
    try:
        packet = await orchestrator.read(
            tenant_id=auth.tenant_id,
            scope=body.scope,
            scope_id=body.scope_id,
            query=body.query,
            max_results=body.max_results,
            memory_types=body.memory_types,
            time_filter={"since": body.since, "until": body.until} if body.since or body.until else None,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000

        def to_item(mem):
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

        all_memories = [to_item(m) for m in packet.all_memories]
        facts = [to_item(m) for m in packet.facts]
        preferences = [to_item(m) for m in packet.preferences]
        episodes = [to_item(m) for m in packet.recent_episodes]

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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/update", response_model=UpdateMemoryResponse)
async def update_memory(
    body: UpdateMemoryRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Update an existing memory. Supports feedback for reconsolidation."""
    try:
        result = await orchestrator.update(
            tenant_id=auth.tenant_id,
            scope=body.scope,
            scope_id=body.scope_id,
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/forget", response_model=ForgetResponse)
async def forget_memory(
    body: ForgetRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Forget (delete/archive/silence) memories within a scope."""
    try:
        result = await orchestrator.forget(
            tenant_id=auth.tenant_id,
            scope=body.scope,
            scope_id=body.scope_id,
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats/{scope}/{scope_id}", response_model=MemoryStats)
async def get_memory_stats(
    scope: MemoryScope,
    scope_id: str,
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Get memory statistics for a scope."""
    try:
        stats = await orchestrator.get_stats(
            tenant_id=auth.tenant_id,
            scope_id=scope_id,
        )
        return MemoryStats(scope=scope, scope_id=scope_id, **stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Session-based API (convenience for scope=SESSION) ----

@router.post("/session/create", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    auth: AuthContext = Depends(require_write_permission),
):
    """Create a new memory session. Returns session_id for subsequent calls."""
    session_id = str(uuid4())
    now = datetime.utcnow()
    return CreateSessionResponse(
        session_id=session_id,
        created_at=now,
        expires_at=now,
    )


@router.post("/session/{session_id}/write", response_model=WriteMemoryResponse)
async def session_write(
    session_id: str,
    body: WriteMemoryRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Write to session memory (scope=SESSION)."""
    try:
        result = await orchestrator.write(
            tenant_id=auth.tenant_id,
            content=body.content,
            memory_type=body.memory_type,
            metadata=body.metadata,
            turn_id=body.turn_id,
            agent_id=body.agent_id,
            scope=MemoryScope.SESSION,
            scope_id=session_id,
            namespace=body.namespace,
        )
        MEMORY_WRITES.labels(tenant_id=auth.tenant_id, status="success").inc()
        return WriteMemoryResponse(
            success=True,
            memory_id=result.get("memory_id"),
            chunks_created=result.get("chunks_created", 0),
            message="Memory stored successfully",
        )
    except Exception as e:
        MEMORY_WRITES.labels(tenant_id=auth.tenant_id, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/read", response_model=ReadMemoryResponse)
async def session_read(
    session_id: str,
    body: ReadMemoryRequest,
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Read from session memory."""
    MEMORY_READS.labels(tenant_id=auth.tenant_id).inc()
    start = datetime.utcnow()
    try:
        packet = await orchestrator.read(
            tenant_id=auth.tenant_id,
            scope=MemoryScope.SESSION,
            scope_id=session_id,
            query=body.query,
            max_results=body.max_results,
            memory_types=body.memory_types,
            time_filter={"since": body.since, "until": body.until} if body.since or body.until else None,
        )
        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000

        def to_item(mem):
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

        all_memories = [to_item(m) for m in packet.all_memories]
        facts = [to_item(m) for m in packet.facts]
        preferences = [to_item(m) for m in packet.preferences]
        episodes = [to_item(m) for m in packet.recent_episodes]
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
