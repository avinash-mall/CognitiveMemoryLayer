"""
Cognitive Memory Layer - Python Client

A simple client for interacting with the Cognitive Memory Layer REST API.
This client can be used directly or integrated with LLM tool calling.

Usage:
    from memory_client import CognitiveMemoryClient
    
    client = CognitiveMemoryClient(api_key="demo-key-123")
    
    # Store a memory
    result = client.write("user-123", "User prefers vegetarian food")
    
    # Retrieve memories
    memories = client.read("user-123", "What food does the user like?")
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
class MemoryStats:
    """Memory statistics for a user."""
    user_id: str
    total_memories: int
    active_memories: int
    by_type: Dict[str, int]
    avg_confidence: float


class CognitiveMemoryClient:
    """
    Python client for the Cognitive Memory Layer API.
    
    Example:
        client = CognitiveMemoryClient(
            base_url="http://localhost:8000",
            api_key="demo-key-123"
        )
        
        # Write memory
        client.write("user-123", "The user lives in Paris")
        
        # Read memory with LLM-ready context
        result = client.read("user-123", "Where does the user live?", format="llm_context")
        print(result.llm_context)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "demo-key-123",
        timeout: float = 30.0
    ):
        """
        Initialize the memory client.
        
        Args:
            base_url: API server URL
            api_key: API authentication key
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an API request."""
        url = f"{self.base_url}/api/v1{endpoint}"
        response = self._client.request(
            method, 
            url, 
            headers=self._headers(),
            **kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._request("GET", "/health")
    
    def write(
        self,
        user_id: str,
        content: str,
        memory_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> MemoryWriteResult:
        """
        Store new information in memory.
        
        Args:
            user_id: Unique identifier for the user
            content: The information to store
            memory_type: Type of memory (episodic_event, semantic_fact, preference, etc.)
            metadata: Additional metadata to store
            turn_id: Conversation turn identifier
            agent_id: Agent that created this memory
            
        Returns:
            MemoryWriteResult with success status and memory_id
            
        Example:
            result = client.write(
                user_id="user-123",
                content="User mentioned they are allergic to peanuts",
                memory_type="constraint"
            )
        """
        payload = {
            "user_id": user_id,
            "content": content
        }
        if memory_type:
            payload["memory_type"] = memory_type
        if metadata:
            payload["metadata"] = metadata
        if turn_id:
            payload["turn_id"] = turn_id
        if agent_id:
            payload["agent_id"] = agent_id
            
        data = self._request("POST", "/memory/write", json=payload)
        return MemoryWriteResult(
            success=data.get("success", False),
            memory_id=data.get("memory_id"),
            chunks_created=data.get("chunks_created", 0),
            message=data.get("message", "")
        )
    
    def read(
        self,
        user_id: str,
        query: str,
        max_results: int = 10,
        memory_types: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        format: str = "packet"
    ) -> MemoryReadResult:
        """
        Retrieve relevant memories for a query.
        
        Args:
            user_id: Unique identifier for the user
            query: Natural language query
            max_results: Maximum number of memories to return
            memory_types: Filter by specific memory types
            since: Only return memories after this time
            until: Only return memories before this time
            format: Response format ("packet", "list", "llm_context")
            
        Returns:
            MemoryReadResult with retrieved memories
            
        Example:
            # Get memories as LLM-ready context
            result = client.read(
                user_id="user-123",
                query="What are the user's dietary restrictions?",
                memory_types=["preference", "constraint"],
                format="llm_context"
            )
            print(result.llm_context)
        """
        payload = {
            "user_id": user_id,
            "query": query,
            "max_results": max_results,
            "format": format
        }
        if memory_types:
            payload["memory_types"] = memory_types
        if since:
            payload["since"] = since.isoformat()
        if until:
            payload["until"] = until.isoformat()
            
        data = self._request("POST", "/memory/read", json=payload)
        
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
        user_id: str,
        memory_id: str,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an existing memory or provide feedback.
        
        Args:
            user_id: Unique identifier for the user
            memory_id: UUID of the memory to update
            text: New text content
            confidence: New confidence score (0-1)
            feedback: Feedback type ("correct", "incorrect", "outdated")
            
        Returns:
            Update result with new version number
            
        Example:
            # Mark a memory as incorrect
            client.update(
                user_id="user-123",
                memory_id="550e8400-e29b-41d4-a716-446655440000",
                feedback="incorrect"
            )
        """
        payload = {
            "user_id": user_id,
            "memory_id": memory_id
        }
        if text:
            payload["text"] = text
        if confidence is not None:
            payload["confidence"] = confidence
        if feedback:
            payload["feedback"] = feedback
            
        return self._request("POST", "/memory/update", json=payload)
    
    def forget(
        self,
        user_id: str,
        memory_ids: Optional[List[str]] = None,
        query: Optional[str] = None,
        before: Optional[datetime] = None,
        action: str = "delete"
    ) -> Dict[str, Any]:
        """
        Forget (delete/archive/silence) memories.
        
        Args:
            user_id: Unique identifier for the user
            memory_ids: Specific memory UUIDs to forget
            query: Natural language query to find memories to forget
            before: Forget memories older than this date
            action: Action type ("delete", "archive", "silence")
            
        Returns:
            Result with affected_count
            
        Example:
            # Forget old address information
            client.forget(
                user_id="user-123",
                query="old address",
                action="archive"
            )
        """
        payload = {
            "user_id": user_id,
            "action": action
        }
        if memory_ids:
            payload["memory_ids"] = memory_ids
        if query:
            payload["query"] = query
        if before:
            payload["before"] = before.isoformat()
            
        return self._request("POST", "/memory/forget", json=payload)
    
    def stats(self, user_id: str) -> MemoryStats:
        """
        Get memory statistics for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            MemoryStats with counts and averages
        """
        data = self._request("GET", f"/memory/stats/{user_id}")
        return MemoryStats(
            user_id=data["user_id"],
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
            result = await client.write("user-123", "User likes jazz music")
            memories = await client.read("user-123", "music preferences")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "demo-key-123",
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1{endpoint}"
        response = await self._client.request(
            method, 
            url, 
            headers=self._headers(),
            **kwargs
        )
        response.raise_for_status()
        return response.json()
    
    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")
    
    async def write(
        self,
        user_id: str,
        content: str,
        memory_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryWriteResult:
        payload = {"user_id": user_id, "content": content}
        if memory_type:
            payload["memory_type"] = memory_type
        if metadata:
            payload["metadata"] = metadata
            
        data = await self._request("POST", "/memory/write", json=payload)
        return MemoryWriteResult(
            success=data.get("success", False),
            memory_id=data.get("memory_id"),
            chunks_created=data.get("chunks_created", 0),
            message=data.get("message", "")
        )
    
    async def read(
        self,
        user_id: str,
        query: str,
        max_results: int = 10,
        format: str = "packet"
    ) -> MemoryReadResult:
        payload = {
            "user_id": user_id,
            "query": query,
            "max_results": max_results,
            "format": format
        }
        data = await self._request("POST", "/memory/read", json=payload)
        
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
    
    async def close(self):
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


if __name__ == "__main__":
    # Quick test
    client = CognitiveMemoryClient()
    
    try:
        health = client.health()
        print(f"API Status: {health['status']}")
        
        # Write a test memory
        result = client.write(
            user_id="test-user",
            content="This is a test memory from the Python client"
        )
        print(f"Write result: {result}")
        
        # Read it back
        memories = client.read(
            user_id="test-user",
            query="test memory",
            format="llm_context"
        )
        print(f"Found {memories.total_count} memories")
        if memories.llm_context:
            print(f"LLM Context:\n{memories.llm_context}")
            
    except httpx.ConnectError:
        print("Could not connect to API. Make sure the server is running:")
        print("  docker compose -f docker/docker-compose.yml up api")
    finally:
        client.close()
