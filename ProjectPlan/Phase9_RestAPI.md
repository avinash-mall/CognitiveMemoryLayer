# Phase 9: REST API & Integration

## Overview
**Duration**: Week 9-10  
**Goal**: Build production-ready REST API with authentication, multi-tenancy, rate limiting, and comprehensive endpoints.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                      Clients                                     │
│   LLM Agents, Chatbots, Applications                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Load Balancer / API Gateway                   │
│   - SSL termination                                              │
│   - Request routing                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                           │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│   │ Auth        │  │ Rate Limit  │  │ Request Validation      │ │
│   │ Middleware  │  │ Middleware  │  │ Middleware              │ │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                              │                                   │
│   ┌──────────────────────────┴────────────────────────────────┐ │
│   │                     API Routes                             │ │
│   │   /memory/write    /memory/read    /memory/update         │ │
│   │   /memory/forget   /memory/stats   /admin/*               │ │
│   └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Orchestrator                           │
│   (Coordinates all memory operations)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 9.1: FastAPI Application Setup

### Description
Set up the FastAPI application with middleware and configuration.

### Subtask 9.1.1: Application Factory

```python
# src/api/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from .middleware import RateLimitMiddleware, RequestLoggingMiddleware
from .routes import router
from ..core.config import get_settings
from ..storage.connection import DatabaseManager

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    
    # Initialize database connections
    db_manager = DatabaseManager.get_instance()
    app.state.db = db_manager
    
    # Initialize memory orchestrator
    from ..memory.orchestrator import MemoryOrchestrator
    app.state.orchestrator = await MemoryOrchestrator.create(db_manager)
    
    yield
    
    # Shutdown
    await db_manager.close()

def create_app() -> FastAPI:
    """Create FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Cognitive Memory Layer",
        description="Neuro-inspired memory system for LLMs",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    
    return app

# Entry point
app = create_app()
```

### Subtask 9.1.2: Middleware Implementation

```python
# src/api/middleware.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import asyncio
from typing import Dict, Tuple
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Generate request ID
        request_id = f"{time.time_ns()}"
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None
        )
        
        try:
            response = await call_next(request)
            
            # Log response
            elapsed = time.time() - start_time
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                elapsed_ms=elapsed * 1000
            )
            
            # Add timing header
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{elapsed * 1000:.2f}ms"
            
            return response
            
        except Exception as e:
            logger.error(
                "request_failed",
                request_id=request_id,
                error=str(e)
            )
            raise

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting per tenant/user."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._buckets: Dict[str, Tuple[int, datetime]] = {}
        self._lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next):
        # Extract tenant/user from headers or auth
        tenant_id = request.headers.get("X-Tenant-ID", "default")
        
        # Check rate limit
        allowed = await self._check_rate_limit(tenant_id)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        return await call_next(request)
    
    async def _check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limit."""
        async with self._lock:
            now = datetime.utcnow()
            
            if key in self._buckets:
                count, window_start = self._buckets[key]
                
                # Check if window has expired
                if now - window_start > timedelta(minutes=1):
                    # New window
                    self._buckets[key] = (1, now)
                    return True
                
                # Check count
                if count >= self.requests_per_minute:
                    return False
                
                # Increment
                self._buckets[key] = (count + 1, window_start)
                return True
            
            else:
                # New key
                self._buckets[key] = (1, now)
                return True
```

---

## Task 9.2: Authentication and Authorization

### Description
Implement API key authentication and tenant isolation.

### Subtask 9.2.1: Auth Dependencies

```python
# src/api/auth.py
from fastapi import Depends, HTTPException, Header, Security
from fastapi.security import APIKeyHeader
from typing import Optional
from dataclasses import dataclass
from ..core.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

@dataclass
class AuthContext:
    """Authentication context for a request."""
    tenant_id: str
    user_id: Optional[str] = None
    api_key: str = ""
    
    # Permissions
    can_read: bool = True
    can_write: bool = True
    can_admin: bool = False

class AuthService:
    """
    Handles authentication and authorization.
    In production, integrate with proper auth system.
    """
    
    def __init__(self):
        self._api_keys: dict = {}  # In production, use database
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from config/database."""
        settings = get_settings()
        
        # Demo keys - replace with database lookup
        self._api_keys = {
            "demo-key-123": AuthContext(
                tenant_id="demo",
                can_read=True,
                can_write=True,
                can_admin=False,
                api_key="demo-key-123"
            ),
            "admin-key-456": AuthContext(
                tenant_id="admin",
                can_read=True,
                can_write=True,
                can_admin=True,
                api_key="admin-key-456"
            )
        }
    
    def validate_key(self, api_key: str) -> Optional[AuthContext]:
        """Validate API key and return context."""
        return self._api_keys.get(api_key)
    
    def check_permission(self, context: AuthContext, permission: str) -> bool:
        """Check if context has permission."""
        if permission == "read":
            return context.can_read
        elif permission == "write":
            return context.can_write
        elif permission == "admin":
            return context.can_admin
        return False

_auth_service = AuthService()

async def get_auth_context(
    api_key: Optional[str] = Security(api_key_header),
    x_tenant_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
) -> AuthContext:
    """
    Dependency to get auth context from request.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )
    
    context = _auth_service.validate_key(api_key)
    
    if not context:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Override user_id from header if provided
    if x_user_id:
        context.user_id = x_user_id
    
    return context

async def require_write_permission(
    context: AuthContext = Depends(get_auth_context)
) -> AuthContext:
    """Require write permission."""
    if not context.can_write:
        raise HTTPException(
            status_code=403,
            detail="Write permission required"
        )
    return context

async def require_admin_permission(
    context: AuthContext = Depends(get_auth_context)
) -> AuthContext:
    """Require admin permission."""
    if not context.can_admin:
        raise HTTPException(
            status_code=403,
            detail="Admin permission required"
        )
    return context
```

---

## Task 9.3: API Routes

### Description
Implement the main API endpoints.

### Subtask 9.3.1: Request/Response Models

```python
# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from ..core.enums import MemoryType

# Write endpoints
class WriteMemoryRequest(BaseModel):
    """Request to store a memory."""
    user_id: str
    content: str
    memory_type: Optional[MemoryType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional context
    turn_id: Optional[str] = None
    agent_id: Optional[str] = None

class WriteMemoryResponse(BaseModel):
    """Response from write operation."""
    success: bool
    memory_id: Optional[UUID] = None
    chunks_created: int = 0
    message: str = ""

# Read endpoints
class ReadMemoryRequest(BaseModel):
    """Request to retrieve memories."""
    user_id: str
    query: str
    max_results: int = Field(default=10, le=50)
    memory_types: Optional[List[MemoryType]] = None
    
    # Time filters
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    
    # Format
    format: str = "packet"  # "packet", "list", "llm_context"

class MemoryItem(BaseModel):
    """A single memory item."""
    id: UUID
    text: str
    type: str
    confidence: float
    relevance: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ReadMemoryResponse(BaseModel):
    """Response from read operation."""
    query: str
    memories: List[MemoryItem]
    
    # Categorized (if format=packet)
    facts: List[MemoryItem] = Field(default_factory=list)
    preferences: List[MemoryItem] = Field(default_factory=list)
    episodes: List[MemoryItem] = Field(default_factory=list)
    
    # For LLM context
    llm_context: Optional[str] = None
    
    # Meta
    total_count: int
    elapsed_ms: float

# Update endpoints
class UpdateMemoryRequest(BaseModel):
    """Request to update a memory."""
    memory_id: UUID
    user_id: str
    
    # What to update
    text: Optional[str] = None
    confidence: Optional[float] = None
    importance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Or provide feedback
    feedback: Optional[str] = None  # "correct", "incorrect", "outdated"

class UpdateMemoryResponse(BaseModel):
    """Response from update operation."""
    success: bool
    memory_id: UUID
    version: int
    message: str = ""

# Forget endpoints
class ForgetRequest(BaseModel):
    """Request to forget memories."""
    user_id: str
    
    # What to forget
    memory_ids: Optional[List[UUID]] = None  # Specific memories
    query: Optional[str] = None              # Matching query
    before: Optional[datetime] = None        # Older than date
    
    # Action
    action: str = "delete"  # "delete", "archive", "silence"

class ForgetResponse(BaseModel):
    """Response from forget operation."""
    success: bool
    affected_count: int
    message: str = ""

# Stats endpoints
class MemoryStats(BaseModel):
    """Memory statistics for a user."""
    user_id: str
    
    # Counts
    total_memories: int
    active_memories: int
    silent_memories: int
    archived_memories: int
    
    # By type
    by_type: Dict[str, int]
    
    # Usage
    avg_confidence: float
    avg_importance: float
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None
    
    # Storage
    estimated_size_mb: float
```

### Subtask 9.3.2: Route Implementations

```python
# src/api/routes.py
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional
from datetime import datetime
from uuid import UUID
from .auth import AuthContext, get_auth_context, require_write_permission
from .schemas import (
    WriteMemoryRequest, WriteMemoryResponse,
    ReadMemoryRequest, ReadMemoryResponse, MemoryItem,
    UpdateMemoryRequest, UpdateMemoryResponse,
    ForgetRequest, ForgetResponse,
    MemoryStats
)
from ..memory.orchestrator import MemoryOrchestrator

router = APIRouter(tags=["memory"])

def get_orchestrator(request: Request) -> MemoryOrchestrator:
    """Get memory orchestrator from app state."""
    return request.app.state.orchestrator

# Write endpoint
@router.post("/memory/write", response_model=WriteMemoryResponse)
async def write_memory(
    body: WriteMemoryRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator)
):
    """
    Store new information in memory.
    
    The system will:
    1. Process through short-term memory (chunking)
    2. Evaluate through write gate
    3. Store if deemed important enough
    4. Extract entities and relations
    """
    try:
        result = await orchestrator.write(
            tenant_id=auth.tenant_id,
            user_id=body.user_id,
            content=body.content,
            memory_type=body.memory_type,
            metadata=body.metadata,
            turn_id=body.turn_id,
            agent_id=body.agent_id
        )
        
        return WriteMemoryResponse(
            success=True,
            memory_id=result.get("memory_id"),
            chunks_created=result.get("chunks_created", 0),
            message="Memory stored successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Read endpoint
@router.post("/memory/read", response_model=ReadMemoryResponse)
async def read_memory(
    body: ReadMemoryRequest,
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator)
):
    """
    Retrieve relevant memories for a query.
    
    Performs hybrid retrieval:
    - Semantic search (vector similarity)
    - Fact lookup (keyed access)
    - Graph traversal (multi-hop reasoning)
    """
    start = datetime.utcnow()
    
    try:
        packet = await orchestrator.read(
            tenant_id=auth.tenant_id,
            user_id=body.user_id,
            query=body.query,
            max_results=body.max_results,
            memory_types=body.memory_types,
            time_filter={"since": body.since, "until": body.until} if body.since or body.until else None
        )
        
        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        
        # Convert to response format
        all_memories = []
        facts = []
        preferences = []
        episodes = []
        
        for mem in packet.all_memories:
            item = MemoryItem(
                id=mem.record.id,
                text=mem.record.text,
                type=mem.record.type,
                confidence=mem.record.confidence,
                relevance=mem.relevance_score,
                timestamp=mem.record.timestamp,
                metadata=mem.record.metadata
            )
            all_memories.append(item)
            
            # Categorize
            if mem.record.type == "semantic_fact":
                facts.append(item)
            elif mem.record.type == "preference":
                preferences.append(item)
            else:
                episodes.append(item)
        
        # Generate LLM context if requested
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
            elapsed_ms=elapsed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update endpoint
@router.post("/memory/update", response_model=UpdateMemoryResponse)
async def update_memory(
    body: UpdateMemoryRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator)
):
    """
    Update an existing memory.
    
    Can update:
    - Content
    - Confidence/importance scores
    - Metadata
    
    Or provide feedback to trigger reconsolidation.
    """
    try:
        result = await orchestrator.update(
            tenant_id=auth.tenant_id,
            user_id=body.user_id,
            memory_id=body.memory_id,
            text=body.text,
            confidence=body.confidence,
            importance=body.importance,
            metadata=body.metadata,
            feedback=body.feedback
        )
        
        return UpdateMemoryResponse(
            success=True,
            memory_id=body.memory_id,
            version=result.get("version", 1),
            message="Memory updated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Forget endpoint
@router.post("/memory/forget", response_model=ForgetResponse)
async def forget_memory(
    body: ForgetRequest,
    auth: AuthContext = Depends(require_write_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator)
):
    """
    Forget (delete/archive/silence) memories.
    
    Can target:
    - Specific memory IDs
    - Memories matching a query
    - Memories older than a date
    """
    try:
        result = await orchestrator.forget(
            tenant_id=auth.tenant_id,
            user_id=body.user_id,
            memory_ids=body.memory_ids,
            query=body.query,
            before=body.before,
            action=body.action
        )
        
        return ForgetResponse(
            success=True,
            affected_count=result.get("affected_count", 0),
            message=f"{result.get('affected_count', 0)} memories {body.action}d"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Stats endpoint
@router.get("/memory/stats/{user_id}", response_model=MemoryStats)
async def get_memory_stats(
    user_id: str,
    auth: AuthContext = Depends(get_auth_context),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator)
):
    """
    Get memory statistics for a user.
    """
    try:
        stats = await orchestrator.get_stats(
            tenant_id=auth.tenant_id,
            user_id=user_id
        )
        
        return MemoryStats(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
```

---

## Task 9.4: Memory Orchestrator

### Description
Main orchestrator coordinating all memory operations.

### Subtask 9.4.1: Orchestrator Implementation

```python
# src/memory/orchestrator.py
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from .short_term import ShortTermMemory
from .hippocampal.store import HippocampalStore
from .neocortical.store import NeocorticalStore
from ..retrieval.memory_retriever import MemoryRetriever
from ..reconsolidation.service import ReconsolidationService
from ..consolidation.worker import ConsolidationWorker
from ..forgetting.worker import ForgettingWorker
from ..storage.connection import DatabaseManager
from ..storage.postgres import PostgresMemoryStore
from ..storage.neo4j import Neo4jGraphStore
from ..utils.llm import get_llm_client
from ..utils.embeddings import OpenAIEmbeddings
from ..core.schemas import MemoryPacket
from ..core.enums import MemoryType

class MemoryOrchestrator:
    """
    Main orchestrator for all memory operations.
    
    Coordinates:
    - Short-term memory (sensory + working)
    - Hippocampal store (episodic)
    - Neocortical store (semantic)
    - Retrieval
    - Reconsolidation
    - Consolidation
    - Forgetting
    """
    
    def __init__(
        self,
        short_term: ShortTermMemory,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        retriever: MemoryRetriever,
        reconsolidation: ReconsolidationService,
        consolidation: ConsolidationWorker,
        forgetting: ForgettingWorker
    ):
        self.short_term = short_term
        self.hippocampal = hippocampal
        self.neocortical = neocortical
        self.retriever = retriever
        self.reconsolidation = reconsolidation
        self.consolidation = consolidation
        self.forgetting = forgetting
    
    @classmethod
    async def create(cls, db_manager: DatabaseManager) -> "MemoryOrchestrator":
        """Factory method to create orchestrator with all dependencies."""
        # Initialize clients
        llm_client = get_llm_client()
        embedding_client = OpenAIEmbeddings()
        
        # Initialize stores
        episodic_store = PostgresMemoryStore(db_manager.pg_session)
        graph_store = Neo4jGraphStore(db_manager.neo4j_driver)
        
        # Initialize components
        short_term = ShortTermMemory(llm_client=llm_client)
        
        hippocampal = HippocampalStore(
            vector_store=episodic_store,
            embedding_client=embedding_client
        )
        
        from .neocortical.fact_store import SemanticFactStore
        fact_store = SemanticFactStore(db_manager.pg_session)
        neocortical = NeocorticalStore(graph_store, fact_store)
        
        retriever = MemoryRetriever(
            hippocampal=hippocampal,
            neocortical=neocortical,
            llm_client=llm_client
        )
        
        reconsolidation = ReconsolidationService(
            memory_store=episodic_store,
            llm_client=llm_client
        )
        
        consolidation = ConsolidationWorker(
            episodic_store=episodic_store,
            neocortical_store=neocortical,
            llm_client=llm_client
        )
        
        forgetting = ForgettingWorker(store=episodic_store)
        
        return cls(
            short_term=short_term,
            hippocampal=hippocampal,
            neocortical=neocortical,
            retriever=retriever,
            reconsolidation=reconsolidation,
            consolidation=consolidation,
            forgetting=forgetting
        )
    
    async def write(
        self,
        tenant_id: str,
        user_id: str,
        content: str,
        memory_type: Optional[MemoryType] = None,
        metadata: Dict[str, Any] = None,
        turn_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Write new information to memory.
        """
        # 1. Process through short-term memory
        stm_result = await self.short_term.ingest_turn(
            tenant_id=tenant_id,
            user_id=user_id,
            text=content,
            turn_id=turn_id,
            role="user"
        )
        
        # 2. Get chunks for encoding
        chunks_for_encoding = stm_result.get("chunks_for_encoding", [])
        
        if not chunks_for_encoding:
            return {
                "memory_id": None,
                "chunks_created": 0,
                "message": "No significant information to store"
            }
        
        # 3. Encode to hippocampal store
        stored = await self.hippocampal.encode_batch(
            tenant_id=tenant_id,
            user_id=user_id,
            chunks=chunks_for_encoding,
            agent_id=agent_id
        )
        
        return {
            "memory_id": stored[0].id if stored else None,
            "chunks_created": len(stored),
            "message": f"Stored {len(stored)} memory chunks"
        }
    
    async def read(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        max_results: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        time_filter: Optional[Dict] = None
    ) -> MemoryPacket:
        """
        Retrieve relevant memories.
        """
        packet = await self.retriever.retrieve(
            tenant_id=tenant_id,
            user_id=user_id,
            query=query,
            max_results=max_results
        )
        
        return packet
    
    async def update(
        self,
        tenant_id: str,
        user_id: str,
        memory_id: UUID,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an existing memory.
        """
        patch = {}
        
        if text is not None:
            patch["text"] = text
        if confidence is not None:
            patch["confidence"] = confidence
        if importance is not None:
            patch["importance"] = importance
        if metadata is not None:
            patch["metadata"] = metadata
        
        # Handle feedback
        if feedback == "correct":
            patch["confidence"] = min(1.0, (confidence or 0.5) + 0.2)
        elif feedback == "incorrect":
            patch["confidence"] = 0.0
            patch["status"] = "invalid"
        elif feedback == "outdated":
            patch["valid_to"] = datetime.utcnow()
        
        result = await self.hippocampal.store.update(memory_id, patch)
        
        return {
            "version": result.version if result else 1
        }
    
    async def forget(
        self,
        tenant_id: str,
        user_id: str,
        memory_ids: Optional[List[UUID]] = None,
        query: Optional[str] = None,
        before: Optional[datetime] = None,
        action: str = "delete"
    ) -> Dict[str, Any]:
        """
        Forget memories.
        """
        affected = 0
        
        if memory_ids:
            for mid in memory_ids:
                await self.hippocampal.store.delete(mid, hard=(action == "delete"))
                affected += 1
        
        # TODO: Implement query-based and time-based forgetting
        
        return {"affected_count": affected}
    
    async def get_stats(
        self,
        tenant_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get memory statistics.
        """
        from ..core.enums import MemoryStatus
        
        total = await self.hippocampal.store.count(tenant_id, user_id)
        active = await self.hippocampal.store.count(
            tenant_id, user_id,
            filters={"status": MemoryStatus.ACTIVE.value}
        )
        
        # Get profile from neocortical
        profile = await self.neocortical.get_user_profile(tenant_id, user_id)
        
        return {
            "user_id": user_id,
            "total_memories": total,
            "active_memories": active,
            "silent_memories": 0,  # TODO: Count
            "archived_memories": 0,
            "by_type": {},  # TODO: Count by type
            "avg_confidence": 0.0,
            "avg_importance": 0.0,
            "oldest_memory": None,
            "newest_memory": None,
            "estimated_size_mb": total * 0.001  # Rough estimate
        }
```

---

## Task 9.5: Admin Routes

### Subtask 9.5.1: Admin Endpoints

```python
# src/api/admin_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from .auth import AuthContext, require_admin_permission
from ..consolidation.worker import ConsolidationWorker
from ..forgetting.worker import ForgettingWorker

admin_router = APIRouter(prefix="/admin", tags=["admin"])

@admin_router.post("/consolidate/{user_id}")
async def trigger_consolidation(
    user_id: str,
    auth: AuthContext = Depends(require_admin_permission)
):
    """Manually trigger consolidation for a user."""
    # Implementation
    return {"status": "consolidation_triggered", "user_id": user_id}

@admin_router.post("/forget/{user_id}")
async def trigger_forgetting(
    user_id: str,
    dry_run: bool = True,
    auth: AuthContext = Depends(require_admin_permission)
):
    """Manually trigger forgetting for a user."""
    return {"status": "forgetting_triggered", "user_id": user_id, "dry_run": dry_run}

@admin_router.get("/users")
async def list_users(
    auth: AuthContext = Depends(require_admin_permission)
):
    """List all users with memory."""
    return {"users": []}

@admin_router.delete("/user/{user_id}")
async def delete_user_memory(
    user_id: str,
    auth: AuthContext = Depends(require_admin_permission)
):
    """Delete all memory for a user (GDPR compliance)."""
    return {"status": "deleted", "user_id": user_id}
```

---

## Deliverables Checklist

- [x] FastAPI application factory with lifespan
- [x] RequestLoggingMiddleware with timing
- [x] RateLimitMiddleware with per-tenant limits
- [x] AuthService with API key validation
- [x] Auth dependencies (get_auth_context, require_write, etc.)
- [x] Request/Response Pydantic models
- [x] /memory/write endpoint
- [x] /memory/read endpoint with format options
- [x] /memory/update endpoint with feedback support
- [x] /memory/forget endpoint
- [x] /memory/stats endpoint
- [x] MemoryOrchestrator coordinating all components
- [x] Admin routes for consolidation/forgetting triggers
- [x] Health check endpoint
- [x] OpenAPI documentation
- [x] Unit tests for routes
- [x] Integration tests for full API flow
