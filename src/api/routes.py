"""API routes for memory operations."""
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request

from ..utils.metrics import MEMORY_READS, MEMORY_WRITES
from .auth import AuthContext, get_auth_context, require_write_permission
from .schemas import (
    ForgetRequest,
    ForgetResponse,
    MemoryItem,
    MemoryStats,
    ReadMemoryRequest,
    ReadMemoryResponse,
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


@router.post("/memory/write", response_model=WriteMemoryResponse)
async def write_memory(
    body: WriteMemoryRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """
    Store new information in memory.
    Processes through short-term memory, write gate, and stores if important.
    """
    try:
        result = await orchestrator.write(
            tenant_id=auth.tenant_id,
            user_id=body.user_id,
            content=body.content,
            memory_type=body.memory_type,
            metadata=body.metadata,
            turn_id=body.turn_id,
            agent_id=body.agent_id,
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
    Performs hybrid retrieval: semantic search, fact lookup, graph traversal.
    """
    MEMORY_READS.labels(tenant_id=auth.tenant_id).inc()
    start = datetime.utcnow()
    try:
        packet = await orchestrator.read(
            tenant_id=auth.tenant_id,
            user_id=body.user_id,
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
            user_id=body.user_id,
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
    """Forget (delete/archive/silence) memories."""
    try:
        result = await orchestrator.forget(
            tenant_id=auth.tenant_id,
            user_id=body.user_id,
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


@router.get("/memory/stats/{user_id}", response_model=MemoryStats)
async def get_memory_stats(
    user_id: str,
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Get memory statistics for a user."""
    try:
        stats = await orchestrator.get_stats(
            tenant_id=auth.tenant_id,
            user_id=user_id,
        )
        return MemoryStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
