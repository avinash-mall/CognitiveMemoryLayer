"""Embedded CognitiveMemoryLayer — runs in-process without a server."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from cml.embedded_config import EmbeddedConfig
from cml.models import (
    ForgetResponse,
    MemoryItem,
    ReadResponse,
    SessionContextResponse,
    SessionResponse,
    StatsResponse,
    TurnResponse,
    UpdateResponse,
    WriteResponse,
)
from cml.models.enums import MemoryType
from cml.utils.logging import logger

CONSOLIDATION_INTERVAL_SEC = 24 * 3600  # 24 hours
FORGETTING_INTERVAL_SEC = 24 * 3600  # 24 hours


async def _consolidation_loop(orchestrator: Any, tenant_id: str) -> None:
    """Background task: run consolidation periodically."""
    while True:
        await asyncio.sleep(CONSOLIDATION_INTERVAL_SEC)
        try:
            await orchestrator.consolidation.consolidate(tenant_id=tenant_id, user_id=tenant_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Consolidation failed: %s", e)


async def _forgetting_loop(orchestrator: Any, tenant_id: str) -> None:
    """Background task: run forgetting periodically (dry_run=False)."""
    while True:
        await asyncio.sleep(FORGETTING_INTERVAL_SEC)
        try:
            await orchestrator.forgetting.run_forgetting(
                tenant_id=tenant_id, user_id=tenant_id, dry_run=False
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Forgetting loop failed: %s", e)


def _check_embedded_deps() -> None:
    """Raise if embedded dependencies are not available."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Embedded mode requires aiosqlite. Install with: pip install cognitive-memory-layer[embedded]"
        ) from e
    try:
        from src.memory.orchestrator import (  # type: ignore
            MemoryOrchestrator,  # noqa: F401
        )
    except ImportError as e:
        raise ImportError(
            "Embedded mode requires the CML engine. "
            "From repo root: pip install -e . then pip install -e packages/py-cml[embedded]."
        ) from e


def _retrieved_to_memory_item(rec: Any) -> MemoryItem:
    """Map engine RetrievedMemory to cognitive-memory-layer MemoryItem."""
    r = rec.record
    return MemoryItem(
        id=r.id,
        text=r.text,
        type=r.type.value if hasattr(r.type, "value") else str(r.type),
        confidence=r.confidence,
        relevance=rec.relevance_score,
        timestamp=r.timestamp,
        metadata=r.metadata or {},
    )


def _packet_to_read_response(query: str, packet: Any, elapsed_ms: float = 0.0) -> ReadResponse:
    """Map engine MemoryPacket to ReadResponse."""
    facts = [_retrieved_to_memory_item(m) for m in packet.facts]
    preferences = [_retrieved_to_memory_item(m) for m in packet.preferences]
    episodes = [_retrieved_to_memory_item(m) for m in packet.recent_episodes]
    all_items = facts + preferences + episodes
    try:
        from src.retrieval.packet_builder import (  # type: ignore
            MemoryPacketBuilder,
        )

        builder = MemoryPacketBuilder()
        llm_context = builder.to_llm_context(packet, max_tokens=4000)
    except Exception as e:
        logger.warning(
            "Failed to import MemoryPacketBuilder, using fallback context: %s",
            e,
        )
        llm_context = "\n".join(f"- {m.text}" for m in all_items[:20])
    return ReadResponse(
        query=query,
        memories=all_items,
        facts=facts,
        preferences=preferences,
        episodes=episodes,
        llm_context=llm_context,
        total_count=len(packet.all_memories),
        elapsed_ms=elapsed_ms,
    )


class EmbeddedCognitiveMemoryLayer:
    """In-process CognitiveMemoryLayer engine. No server, no HTTP."""

    def __init__(
        self,
        *,
        config: EmbeddedConfig | None = None,
        storage_mode: str = "lite",
        tenant_id: str = "default",
        db_path: str | None = None,
        embedding_provider: str = "local",
        llm_api_key: str | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = EmbeddedConfig(
                storage_mode=storage_mode,  # type: ignore[arg-type]
                tenant_id=tenant_id,
            )
            if db_path is not None:
                self._config.database.database_url = (
                    f"sqlite+aiosqlite:///{db_path}"
                    if not db_path.startswith("sqlite")
                    else db_path
                )
            if embedding_provider != "local":
                self._config.embedding.provider = embedding_provider  # type: ignore[assignment]
            if llm_api_key is not None:
                self._config.llm.api_key = llm_api_key
        self._orchestrator: Any = None
        self._sqlite_store: Any = None
        self._initialized = False
        self._session_store: dict[str, dict[str, Any]] = {}
        self._background_tasks: list[asyncio.Task[Any]] = []

    async def initialize(self) -> None:
        """Initialize storage and the memory orchestrator."""
        _check_embedded_deps()
        if self._config.storage_mode != "lite":
            raise NotImplementedError(
                "Only storage_mode='lite' is implemented. "
                "Standard/full require the server or additional wiring."
            )
        from cml.storage.sqlite_store import SQLiteMemoryStore

        db_url = self._config.database.database_url
        if db_url.startswith("sqlite+aiosqlite://"):
            path = db_url.replace("sqlite+aiosqlite:///", "").strip("/") or ":memory:"
        else:
            path = ":memory:"
        self._sqlite_store = SQLiteMemoryStore(db_path=path)
        await self._sqlite_store.initialize()

        from src.utils.embeddings import LocalEmbeddings  # type: ignore
        from src.utils.llm import OpenAICompatibleClient  # type: ignore

        # When running in repo, use project LLM settings from env so tests use local Ollama etc.
        try:
            from src.core.config import get_settings  # type: ignore

            s = get_settings()
            if s.llm.base_url:
                self._config.llm.base_url = s.llm.base_url
                self._config.llm.model = s.llm.model
                self._config.llm.provider = s.llm.provider
        except Exception:
            pass

        embedding_client = LocalEmbeddings(model_name=self._config.embedding.model)
        llm_client = OpenAICompatibleClient(
            model=self._config.llm.model,
            base_url=self._config.llm.base_url,
            api_key=self._config.llm.api_key or "dummy",
        )

        from src.memory.orchestrator import MemoryOrchestrator

        self._orchestrator = await MemoryOrchestrator.create_lite(
            self._sqlite_store,
            embedding_client,
            llm_client,
        )
        self._initialized = True

        if self._config.auto_consolidate:
            task = asyncio.create_task(
                _consolidation_loop(self._orchestrator, self._config.tenant_id)
            )
            self._background_tasks.append(task)
        if self._config.auto_forget:
            task = asyncio.create_task(_forgetting_loop(self._orchestrator, self._config.tenant_id))
            self._background_tasks.append(task)

    def _ensure_initialized(self) -> None:
        if not self._initialized or self._orchestrator is None:
            raise RuntimeError(
                "EmbeddedCognitiveMemoryLayer not initialized. "
                "Use `async with` or call `await memory.initialize()` first."
            )

    async def close(self) -> None:
        """Shutdown storage and workers."""
        import contextlib

        for t in self._background_tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._background_tasks.clear()
        if self._sqlite_store is not None and hasattr(self._sqlite_store, "close"):
            await self._sqlite_store.close()
        self._orchestrator = None
        self._sqlite_store = None
        self._initialized = False

    async def __aenter__(self) -> EmbeddedCognitiveMemoryLayer:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

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
    ) -> WriteResponse:
        """Store a memory."""
        self._ensure_initialized()
        result = await self._orchestrator.write(
            tenant_id=self._config.tenant_id,
            content=content,
            context_tags=context_tags,
            session_id=session_id,
            memory_type=memory_type,
            metadata=metadata,
            turn_id=turn_id,
            agent_id=agent_id,
            namespace=namespace,
        )
        mid = result.get("memory_id")
        return WriteResponse(
            success=True,
            memory_id=UUID(str(mid)) if mid else None,
            chunks_created=result.get("chunks_created", 0),
            message=result.get("message", ""),
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
        format: str = "packet",
    ) -> ReadResponse:
        """Retrieve memories by query."""
        self._ensure_initialized()
        t0 = time.perf_counter()
        packet = await self._orchestrator.read(
            tenant_id=self._config.tenant_id,
            query=query,
            max_results=max_results,
            context_filter=context_filter,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return _packet_to_read_response(query, packet, elapsed_ms)

    async def turn(
        self,
        user_message: str,
        *,
        assistant_response: str | None = None,
        session_id: str | None = None,
        max_context_tokens: int = 1500,
    ) -> TurnResponse:
        """Process a conversational turn."""
        self._ensure_initialized()
        from src.memory.seamless_provider import (  # type: ignore
            SeamlessMemoryProvider,
        )

        provider = SeamlessMemoryProvider(
            self._orchestrator,
            max_context_tokens=max_context_tokens,
        )
        result = await provider.process_turn(
            tenant_id=self._config.tenant_id,
            user_message=user_message,
            assistant_response=assistant_response,
            session_id=session_id,
        )
        return TurnResponse(
            memory_context=result.memory_context,
            memories_retrieved=len(result.injected_memories),
            memories_stored=result.stored_count,
            reconsolidation_applied=result.reconsolidation_applied,
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
        """Update an existing memory."""
        self._ensure_initialized()
        result = await self._orchestrator.update(
            tenant_id=self._config.tenant_id,
            memory_id=memory_id,
            text=text,
            confidence=confidence,
            importance=importance,
            metadata=metadata,
            feedback=feedback,
        )
        return UpdateResponse(
            success=True,
            memory_id=memory_id,
            version=result.get("version", 1),
        )

    async def forget(
        self,
        *,
        memory_ids: list[UUID] | None = None,
        query: str | None = None,
        before: datetime | None = None,
        action: str = "delete",
    ) -> ForgetResponse:
        """Forget memories."""
        if not memory_ids and not query and before is None:
            raise ValueError("At least one of memory_ids, query, or before must be provided")
        self._ensure_initialized()
        result = await self._orchestrator.forget(
            tenant_id=self._config.tenant_id,
            memory_ids=memory_ids,
            query=query,
            before=before,
            action=action,
        )
        return ForgetResponse(
            success=True,
            affected_count=result.get("affected_count", 0),
        )

    async def stats(self) -> StatsResponse:
        """Get memory statistics."""
        self._ensure_initialized()
        result = await self._orchestrator.get_stats(tenant_id=self._config.tenant_id)
        return StatsResponse(**result)

    async def create_session(
        self,
        *,
        name: str | None = None,
        ttl_hours: int = 24,
        metadata: dict[str, Any] | None = None,
    ) -> SessionResponse:
        """Create a session (lite: in-memory)."""
        from datetime import timedelta
        from uuid import uuid4

        session_id = str(uuid4())
        now = datetime.now(UTC)
        self._session_store[session_id] = {
            "name": name,
            "created_at": now,
            "metadata": metadata or {},
        }
        expires_at = now + timedelta(hours=ttl_hours)
        return SessionResponse(
            session_id=session_id,
            created_at=now,
            expires_at=expires_at,
        )

    async def get_session_context(self, session_id: str) -> SessionContextResponse:
        """Get session context (lite: recent memories as context)."""
        self._ensure_initialized()
        ctx = await self._orchestrator.get_session_context(
            tenant_id=self._config.tenant_id,
            session_id=session_id,
        )

        def _to_memory_item(m: dict[str, Any]) -> MemoryItem:
            mid = m.get("id")
            if isinstance(mid, str):
                mid = UUID(mid)
            elif not isinstance(mid, UUID):
                mid = UUID(str(mid)) if mid else UUID(int=0)
            ts = m.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif ts is None:
                ts = datetime.now(UTC)
            return MemoryItem(
                id=mid,
                text=m.get("text", ""),
                type=m.get("type", "episodic_event"),
                confidence=float(m.get("confidence", 0.5)),
                relevance=float(m.get("relevance", 0.0)),
                timestamp=ts,
                metadata=m.get("metadata") or {},
            )

        messages = [_to_memory_item(m) for m in ctx.get("messages", [])]
        return SessionContextResponse(
            session_id=session_id,
            messages=messages,
            tool_results=[],
            scratch_pad=[],
            context_string=ctx.get("context_string", ""),
        )

    async def delete_all(self, *, confirm: bool = False) -> int:
        """Delete all memories for the tenant."""
        if not confirm:
            raise ValueError("delete_all requires confirm=True to avoid accidental data loss")
        self._ensure_initialized()
        n: int = await self._orchestrator.delete_all(tenant_id=self._config.tenant_id)
        return n

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
        """Convenience: return LLM context string."""
        result = await self.read(
            query,
            max_results=max_results,
            context_filter=context_filter,
            memory_types=memory_types,
            since=since,
            until=until,
            format="llm_context",
        )
        return result.context

    async def remember(self, content: str, **kwargs: Any) -> WriteResponse:
        """Alias for write()."""
        return await self.write(content, **kwargs)

    async def search(self, query: str, **kwargs: Any) -> ReadResponse:
        """Alias for read()."""
        return await self.read(query, **kwargs)
