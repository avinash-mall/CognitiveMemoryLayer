"""
Cognitive Memory Layer - Python Client

A simple client for interacting with the Cognitive Memory Layer REST API.
This client can be used directly or integrated with LLM tool calling.

Usage:
    Set AUTH__API_KEY in your environment, or pass api_key when creating the client.

    from memory_client import CognitiveMemoryClient
    import os

    client = CognitiveMemoryClient(api_key=os.environ.get("AUTH__API_KEY", "your-key"))

    # Store a memory (holistic: tenant from auth; optional context_tags, session_id)
    result = client.write("User prefers vegetarian food", session_id="session-123", context_tags=["preference"])

    # Retrieve memories
    memories = client.read("What food does the user like?")
"""

import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class MemoryItem:
    """A single memory item from the API."""
    id: str
    text: str
    type: str
    confidence: float
    relevance: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class MemoryReadResult:
    """Result from a memory read operation."""
    query: str
    memories: List[MemoryItem]
    facts: List[MemoryItem]
    preferences: List[MemoryItem]
    episodes: List[MemoryItem]
    llm_context: Optional[str]
    total_count: int
    elapsed_ms: float


@dataclass
class MemoryWriteResult:
    """Result from a memory write operation."""
    success: bool
    memory_id: Optional[str]
    chunks_created: int
    message: str


@dataclass
class ProcessTurnResult:
    """Result from process_turn (seamless memory)."""
    memory_context: str
    memories_retrieved: int
    memories_stored: int
    reconsolidation_applied: bool


@dataclass
class MemoryStats:
    """Memory statistics for the tenant."""
    total_memories: int
    active_memories: int
    by_type: Dict[str, int]
    avg_confidence: float


class CognitiveMemoryClient:
    """
    Python client for the Cognitive Memory Layer API.
    Holistic memory: tenant from X-Tenant-ID or API key; no scopes.

    Example:
        import os
        client = CognitiveMemoryClient(
            base_url="http://localhost:8000",
            api_key=os.environ.get("AUTH__API_KEY", "")
        )
        client.write("The user lives in Paris", session_id="session-123", context_tags=["personal"])
        result = client.read("Where does the user live?", format="llm_context")
        print(result.llm_context)
        # Seamless turn (auto-retrieve + auto-store):
        turn = client.process_turn("What do I like?", session_id="session-123")
        print(turn.memory_context)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize the memory client.

        Args:
            base_url: API server URL
            api_key: API key (default: AUTH__API_KEY from environment)
            timeout: Request timeout in seconds
        """
        import os
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("AUTH__API_KEY", "")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _headers(self, tenant_id: Optional[str] = None) -> Dict[str, str]:
        """Get request headers with authentication."""
        h = {"Content-Type": "application/json", "X-API-Key": self.api_key}
        if tenant_id:
            h["X-Tenant-ID"] = tenant_id
        return h
    
    def _request(
        self,
        method: str,
        endpoint: str,
        tenant_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an API request."""
        url = f"{self.base_url}/api/v1{endpoint}"
        headers = self._headers(tenant_id=tenant_id)
        if "headers" in kwargs:
            headers = {**headers, **kwargs.pop("headers")}
        response = self._client.request(
            method, url, headers=headers, **kwargs
        )
        if not response.is_success:
            body = response.text
            try:
                data = response.json()
                detail = data.get("detail", body)
            except Exception:
                detail = body
            msg = f"{response.status_code} {response.reason_phrase}"
            if detail:
                msg += f" — {detail}"
            raise httpx.HTTPStatusError(msg, request=response.request, response=response)
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._request("GET", "/health")
    
    def write(
        self,
        content: str,
        context_tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        namespace: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> MemoryWriteResult:
        """
        Store new information in memory. Holistic: tenant from header or API key.

        Args:
            content: The information to store
            context_tags: Optional tags (e.g. personal, conversation)
            session_id: Optional session ID for origin tracking
            memory_type: Type of memory (episodic_event, semantic_fact, preference, etc.)
            metadata: Additional metadata
            turn_id: Conversation turn identifier
            agent_id: Agent that created this memory
            namespace: Optional namespace
            tenant_id: Optional tenant (default from API key config)
        """
        payload = {"content": content}
        if context_tags:
            payload["context_tags"] = context_tags
        if session_id:
            payload["session_id"] = session_id
        if memory_type:
            payload["memory_type"] = memory_type
        if metadata:
            payload["metadata"] = metadata
        if turn_id:
            payload["turn_id"] = turn_id
        if agent_id:
            payload["agent_id"] = agent_id
        if namespace:
            payload["namespace"] = namespace
        data = self._request("POST", "/memory/write", tenant_id=tenant_id, json=payload)
        return MemoryWriteResult(
            success=data.get("success", False),
            memory_id=data.get("memory_id"),
            chunks_created=data.get("chunks_created", 0),
            message=data.get("message", "")
        )

    def process_turn(
        self,
        user_message: str,
        assistant_response: Optional[str] = None,
        session_id: Optional[str] = None,
        max_context_tokens: int = 1500,
        tenant_id: Optional[str] = None,
    ) -> ProcessTurnResult:
        """
        Seamless memory: auto-retrieve context for user message and optionally auto-store.
        Returns memory_context ready to inject into LLM prompt.
        """
        payload = {
            "user_message": user_message,
            "max_context_tokens": max_context_tokens,
        }
        if assistant_response is not None:
            payload["assistant_response"] = assistant_response
        if session_id:
            payload["session_id"] = session_id
        data = self._request("POST", "/memory/turn", tenant_id=tenant_id, json=payload)
        return ProcessTurnResult(
            memory_context=data.get("memory_context", ""),
            memories_retrieved=data.get("memories_retrieved", 0),
            memories_stored=data.get("memories_stored", 0),
            reconsolidation_applied=data.get("reconsolidation_applied", False),
        )
    
    def read(
        self,
        query: str,
        max_results: int = 10,
        context_filter: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        format: str = "packet",
        tenant_id: Optional[str] = None,
    ) -> MemoryReadResult:
        """
        Retrieve relevant memories for a query. Holistic: tenant from auth.
        """
        payload = {
            "query": query,
            "max_results": max_results,
            "format": format,
        }
        if context_filter:
            payload["context_filter"] = context_filter
        if memory_types:
            payload["memory_types"] = memory_types
        if since:
            payload["since"] = since.isoformat()
        if until:
            payload["until"] = until.isoformat()
        data = self._request("POST", "/memory/read", tenant_id=tenant_id, json=payload)
        
        def parse_item(item: Dict) -> MemoryItem:
            return MemoryItem(
                id=item["id"],
                text=item["text"],
                type=item["type"],
                confidence=item["confidence"],
                relevance=item["relevance"],
                timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
                metadata=item.get("metadata", {})
            )
        
        return MemoryReadResult(
            query=data["query"],
            memories=[parse_item(m) for m in data.get("memories", [])],
            facts=[parse_item(m) for m in data.get("facts", [])],
            preferences=[parse_item(m) for m in data.get("preferences", [])],
            episodes=[parse_item(m) for m in data.get("episodes", [])],
            llm_context=data.get("llm_context"),
            total_count=data.get("total_count", 0),
            elapsed_ms=data.get("elapsed_ms", 0)
        )
    
    def update(
        self,
        memory_id: str,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        feedback: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing memory or provide feedback. Holistic: tenant from auth."""
        payload = {"memory_id": memory_id}
        if text:
            payload["text"] = text
        if confidence is not None:
            payload["confidence"] = confidence
        if feedback:
            payload["feedback"] = feedback
        return self._request("POST", "/memory/update", tenant_id=tenant_id, json=payload)
    
    def forget(
        self,
        memory_ids: Optional[List[str]] = None,
        query: Optional[str] = None,
        before: Optional[datetime] = None,
        action: str = "delete",
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forget (delete/archive/silence) memories. Holistic: tenant from auth."""
        payload = {"action": action}
        if memory_ids:
            payload["memory_ids"] = memory_ids
        if query:
            payload["query"] = query
        if before:
            payload["before"] = before.isoformat()
        return self._request("POST", "/memory/forget", tenant_id=tenant_id, json=payload)

    def stats(self, tenant_id: Optional[str] = None) -> MemoryStats:
        """Get memory statistics for the tenant."""
        data = self._request("GET", "/memory/stats", tenant_id=tenant_id)
        return MemoryStats(
            total_memories=data["total_memories"],
            active_memories=data["active_memories"],
            by_type=data.get("by_type", {}),
            avg_confidence=data.get("avg_confidence", 0.0)
        )
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Async version of the client
class AsyncCognitiveMemoryClient:
    """
    Async Python client for the Cognitive Memory Layer API.
    
    Example:
        async with AsyncCognitiveMemoryClient() as client:
            result = await client.write("session", "sess-123", "User likes jazz music")
            memories = await client.read("session", "sess-123", "music preferences")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        import os
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("AUTH__API_KEY", "")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    def _headers(self, tenant_id: Optional[str] = None) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "X-API-Key": self.api_key}
        if tenant_id:
            h["X-Tenant-ID"] = tenant_id
        return h

    async def _request(
        self,
        method: str,
        endpoint: str,
        tenant_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1{endpoint}"
        response = await self._client.request(
            method, url, headers=self._headers(tenant_id=tenant_id), **kwargs
        )
        response.raise_for_status()
        return response.json()

    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")

    async def write(
        self,
        content: str,
        context_tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        namespace: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> MemoryWriteResult:
        payload = {"content": content}
        if context_tags:
            payload["context_tags"] = context_tags
        if session_id:
            payload["session_id"] = session_id
        if memory_type:
            payload["memory_type"] = memory_type
        if metadata:
            payload["metadata"] = metadata
        if turn_id:
            payload["turn_id"] = turn_id
        if agent_id:
            payload["agent_id"] = agent_id
        if namespace:
            payload["namespace"] = namespace
        data = await self._request("POST", "/memory/write", tenant_id=tenant_id, json=payload)
        return MemoryWriteResult(
            success=data.get("success", False),
            memory_id=data.get("memory_id"),
            chunks_created=data.get("chunks_created", 0),
            message=data.get("message", "")
        )

    async def read(
        self,
        query: str,
        max_results: int = 10,
        context_filter: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        format: str = "packet",
        tenant_id: Optional[str] = None,
    ) -> MemoryReadResult:
        payload = {"query": query, "max_results": max_results, "format": format}
        if context_filter:
            payload["context_filter"] = context_filter
        if memory_types:
            payload["memory_types"] = memory_types
        if since:
            payload["since"] = since.isoformat()
        if until:
            payload["until"] = until.isoformat()
        data = await self._request("POST", "/memory/read", tenant_id=tenant_id, json=payload)
        
        def parse_item(item: Dict) -> MemoryItem:
            return MemoryItem(
                id=item["id"],
                text=item["text"],
                type=item["type"],
                confidence=item["confidence"],
                relevance=item["relevance"],
                timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
                metadata=item.get("metadata", {})
            )
        
        return MemoryReadResult(
            query=data["query"],
            memories=[parse_item(m) for m in data.get("memories", [])],
            facts=[parse_item(m) for m in data.get("facts", [])],
            preferences=[parse_item(m) for m in data.get("preferences", [])],
            episodes=[parse_item(m) for m in data.get("episodes", [])],
            llm_context=data.get("llm_context"),
            total_count=data.get("total_count", 0),
            elapsed_ms=data.get("elapsed_ms", 0)
        )

    async def stats(self, tenant_id: Optional[str] = None) -> MemoryStats:
        """Get memory statistics for the tenant."""
        data = await self._request("GET", "/memory/stats", tenant_id=tenant_id)
        return MemoryStats(
            total_memories=data["total_memories"],
            active_memories=data["active_memories"],
            by_type=data.get("by_type", {}),
            avg_confidence=data.get("avg_confidence", 0.0)
        )

    async def update(
        self,
        memory_id: str,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        feedback: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing memory or provide feedback."""
        payload = {"memory_id": memory_id}
        if text:
            payload["text"] = text
        if confidence is not None:
            payload["confidence"] = confidence
        if feedback:
            payload["feedback"] = feedback
        return await self._request("POST", "/memory/update", tenant_id=tenant_id, json=payload)

    async def forget(
        self,
        memory_ids: Optional[List[str]] = None,
        query: Optional[str] = None,
        before: Optional[datetime] = None,
        action: str = "delete",
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forget (delete/archive/silence) memories."""
        payload = {"action": action}
        if memory_ids:
            payload["memory_ids"] = memory_ids
        if query:
            payload["query"] = query
        if before:
            payload["before"] = before.isoformat()
        return await self._request("POST", "/memory/forget", tenant_id=tenant_id, json=payload)

    async def process_turn(
        self,
        user_message: str,
        assistant_response: Optional[str] = None,
        session_id: Optional[str] = None,
        max_context_tokens: int = 1500,
        tenant_id: Optional[str] = None,
    ) -> ProcessTurnResult:
        """Seamless memory: auto-retrieve context and optionally auto-store. Returns memory_context."""
        payload = {
            "user_message": user_message,
            "max_context_tokens": max_context_tokens,
        }
        if assistant_response is not None:
            payload["assistant_response"] = assistant_response
        if session_id:
            payload["session_id"] = session_id
        data = await self._request("POST", "/memory/turn", tenant_id=tenant_id, json=payload)
        return ProcessTurnResult(
            memory_context=data.get("memory_context", ""),
            memories_retrieved=data.get("memories_retrieved", 0),
            memories_stored=data.get("memories_stored", 0),
            reconsolidation_applied=data.get("reconsolidation_applied", False),
        )

    async def close(self):
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


if __name__ == "__main__":
    import os
    api_key = os.environ.get("AUTH__API_KEY", "")
    if not api_key:
        print("Set AUTH__API_KEY in your environment to run this test.")
        exit(1)
    client = CognitiveMemoryClient(api_key=api_key)
    
    try:
        health = client.health()
        print(f"API Status: {health['status']}")
        result = client.write(
            "This is a test memory from the Python client",
            session_id="test-session",
            context_tags=["conversation"],
        )
        print(f"Write result: {result}")
        memories = client.read("test memory", format="llm_context")
        print(f"Found {memories.total_count} memories")
        if memories.llm_context:
            print(f"LLM Context:\n{memories.llm_context}")
            
    except httpx.ConnectError:
        print("Could not connect to API. Make sure the server is running:")
        print("  docker compose -f docker/docker-compose.yml up api")
    finally:
        client.close()
