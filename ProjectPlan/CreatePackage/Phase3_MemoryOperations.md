# Phase 3: Memory Operations API

## Objective

Implement all memory operations (write, read, update, forget, turn, stats, sessions) on both the sync and async clients, providing a complete Pythonic interface to every CML server endpoint.

---

## Task 3.1: Write Operation

### Sub-Task 3.1.1: Async Write

**Architecture**: Store new information into the CML memory system. Accepts plain text content with optional metadata, context tags, memory type, and session binding.

**Implementation** (`src/cml/async_client.py`):
```python
async def write(
    self,
    content: str,
    *,
    context_tags: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    memory_type: Optional[MemoryType] = None,
    namespace: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    turn_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> WriteResponse:
    """Store new information in memory.

    Args:
        content: The text content to store.
        context_tags: Optional tags for categorization (e.g., ["personal", "preference"]).
        session_id: Optional session identifier for grouping.
        memory_type: Optional explicit type (auto-detected if omitted).
        namespace: Optional namespace for isolation.
        metadata: Optional key-value metadata.
        turn_id: Optional conversation turn identifier.
        agent_id: Optional agent identifier.

    Returns:
        WriteResponse with memory_id, chunks_created, and message.

    Raises:
        AuthenticationError: If API key is invalid.
        ValidationError: If request payload is invalid.
        CMLError: For other server errors.
    """
    payload = WriteRequest(
        content=content,
        context_tags=context_tags,
        session_id=session_id,
        memory_type=memory_type,
        namespace=namespace,
        metadata=metadata or {},
        turn_id=turn_id,
        agent_id=agent_id,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/write", json=payload)
    return WriteResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD write(content, **options) -> WriteResponse:
    1. Build WriteRequest from parameters
    2. Serialize to dict, excluding None values
    3. POST /api/v1/memory/write with JSON body
    4. Parse response into WriteResponse
    5. RETURN WriteResponse(success, memory_id, chunks_created, message)
```

### Sub-Task 3.1.2: Sync Write

**Implementation** (`src/cml/client.py`):
```python
def write(
    self,
    content: str,
    *,
    context_tags: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    memory_type: Optional[MemoryType] = None,
    namespace: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    turn_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> WriteResponse:
    """Store new information in memory (synchronous)."""
    payload = WriteRequest(
        content=content,
        context_tags=context_tags,
        session_id=session_id,
        memory_type=memory_type,
        namespace=namespace,
        metadata=metadata or {},
        turn_id=turn_id,
        agent_id=agent_id,
    ).model_dump(exclude_none=True)

    data = self._transport.request("POST", "/api/v1/memory/write", json=payload)
    return WriteResponse(**data)
```

---

## Task 3.2: Read Operation

### Sub-Task 3.2.1: Async Read

**Architecture**: Retrieve relevant memories using hybrid search (vector + graph + lexical). Supports filtering by context tags, memory types, and time ranges. Returns structured results with an optional pre-formatted LLM context string.

**Implementation**:
```python
async def read(
    self,
    query: str,
    *,
    max_results: int = 10,
    context_filter: Optional[List[str]] = None,
    memory_types: Optional[List[MemoryType]] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    format: Literal["packet", "list", "llm_context"] = "packet",
) -> ReadResponse:
    """Retrieve relevant memories for a query.

    Args:
        query: The search query (natural language).
        max_results: Maximum number of results (1-50).
        context_filter: Filter by context tags.
        memory_types: Filter by memory types.
        since: Only return memories after this time.
        until: Only return memories before this time.
        format: Response format:
            - "packet": Categorized by type (facts, preferences, episodes).
            - "list": Flat list sorted by relevance.
            - "llm_context": Includes pre-formatted context string.

    Returns:
        ReadResponse with memories, optional LLM context, and elapsed time.
    """
    payload = ReadRequest(
        query=query,
        max_results=max_results,
        context_filter=context_filter,
        memory_types=memory_types,
        since=since,
        until=until,
        format=format,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/read", json=payload)
    return ReadResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD read(query, **filters) -> ReadResponse:
    1. Build ReadRequest with query and filters
    2. Serialize (exclude_none)
    3. POST /api/v1/memory/read
    4. Parse into ReadResponse
    5. ReadResponse contains:
       - memories: List[MemoryItem]  (all results)
       - facts: List[MemoryItem]      (semantic facts)
       - preferences: List[MemoryItem] (user preferences)
       - episodes: List[MemoryItem]    (episodic events)
       - llm_context: str | None       (formatted for prompt injection)
       - total_count: int
       - elapsed_ms: float
    6. RETURN ReadResponse

    Convenience: result.context → formatted LLM context string
```

---

## Task 3.3: Seamless Turn

### Sub-Task 3.3.1: Async Turn

**Architecture**: The "seamless" turn endpoint auto-retrieves relevant memories and optionally stores new information from the conversation. This is the primary integration point for chatbots.

**Implementation**:
```python
async def turn(
    self,
    user_message: str,
    *,
    assistant_response: Optional[str] = None,
    session_id: Optional[str] = None,
    max_context_tokens: int = 1500,
) -> TurnResponse:
    """Process a conversation turn with seamless memory.

    Automatically retrieves relevant memories for the user's message
    and optionally stores information from the assistant's response.

    Args:
        user_message: The user's message in this turn.
        assistant_response: Optional assistant response to store.
        session_id: Optional session identifier.
        max_context_tokens: Maximum tokens for memory context.

    Returns:
        TurnResponse with memory_context ready for prompt injection.

    Example:
        turn = await memory.turn(
            user_message="What should I eat tonight?",
            session_id="session-001"
        )
        # Inject turn.memory_context into your LLM prompt
    """
    payload = TurnRequest(
        user_message=user_message,
        assistant_response=assistant_response,
        session_id=session_id,
        max_context_tokens=max_context_tokens,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/turn", json=payload)
    return TurnResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD turn(user_message, assistant_response?, session_id?) -> TurnResponse:
    1. Build TurnRequest payload
    2. POST /api/v1/memory/turn
    3. Server internally:
       a. Retrieves relevant memories for user_message
       b. Formats memories into context string
       c. If assistant_response provided, extracts & stores new info
       d. Runs reconsolidation on retrieved memories
    4. Parse response → TurnResponse
    5. TurnResponse contains:
       - memory_context: str        (ready to inject into prompt)
       - memories_retrieved: int     (count of retrieved memories)
       - memories_stored: int        (count of newly stored memories)
       - reconsolidation_applied: bool
    6. RETURN TurnResponse
```

### Sub-Task 3.3.2: Sync Turn

Mirror of async with sync transport call.

---

## Task 3.4: Update Operation

### Sub-Task 3.4.1: Async Update

**Architecture**: Update an existing memory's text, confidence, importance, metadata, or provide feedback (correct/incorrect/outdated).

**Implementation**:
```python
async def update(
    self,
    memory_id: UUID,
    *,
    text: Optional[str] = None,
    confidence: Optional[float] = None,
    importance: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    feedback: Optional[Literal["correct", "incorrect", "outdated"]] = None,
) -> UpdateResponse:
    """Update an existing memory.

    Args:
        memory_id: UUID of the memory to update.
        text: New text content (triggers re-embedding).
        confidence: Updated confidence score (0.0-1.0).
        importance: Updated importance score (0.0-1.0).
        metadata: Updated metadata dict.
        feedback: Semantic feedback:
            - "correct": Boosts confidence by 0.2
            - "incorrect": Sets confidence to 0, marks as deleted
            - "outdated": Sets valid_to to now

    Returns:
        UpdateResponse with success, version number.
    """
    payload = UpdateRequest(
        memory_id=memory_id,
        text=text,
        confidence=confidence,
        importance=importance,
        metadata=metadata,
        feedback=feedback,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/update", json=payload)
    return UpdateResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD update(memory_id, **patches) -> UpdateResponse:
    1. Build UpdateRequest
    2. POST /api/v1/memory/update
    3. Server:
       a. Finds memory by ID
       b. Validates tenant ownership
       c. If text changed → re-embed, re-extract entities
       d. If feedback == "correct" → confidence += 0.2
       e. If feedback == "incorrect" → confidence = 0, status = DELETED
       f. If feedback == "outdated" → valid_to = now()
       g. Increment version
    4. Parse → UpdateResponse
    5. RETURN UpdateResponse(success, memory_id, version)
```

---

## Task 3.5: Forget Operation

### Sub-Task 3.5.1: Async Forget

**Architecture**: Remove memories by ID, query match, or time filter. Supports soft-delete (archive/silence) and hard-delete.

**Implementation**:
```python
async def forget(
    self,
    *,
    memory_ids: Optional[List[UUID]] = None,
    query: Optional[str] = None,
    before: Optional[datetime] = None,
    action: Literal["delete", "archive", "silence"] = "delete",
) -> ForgetResponse:
    """Forget (remove) memories.

    Args:
        memory_ids: Specific memory IDs to forget.
        query: Forget memories matching this query.
        before: Forget memories created before this time.
        action: Forget strategy:
            - "delete": Hard delete (permanent).
            - "archive": Soft delete, keep for audit.
            - "silence": Make hard to retrieve (needs strong cue).

    Returns:
        ForgetResponse with affected_count.

    Note:
        At least one of memory_ids, query, or before must be provided.
    """
    if not memory_ids and not query and not before:
        raise ValueError("At least one of memory_ids, query, or before must be provided")

    payload = ForgetRequest(
        memory_ids=memory_ids,
        query=query,
        before=before,
        action=action,
    ).model_dump(exclude_none=True)

    data = await self._transport.request("POST", "/api/v1/memory/forget", json=payload)
    return ForgetResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD forget(memory_ids?, query?, before?, action) -> ForgetResponse:
    1. Validate at least one selector provided
    2. Build ForgetRequest
    3. POST /api/v1/memory/forget
    4. Server:
       a. Collect target IDs from memory_ids, query results, time filter
       b. Deduplicate
       c. For each target:
          - IF action == "delete" → hard delete
          - IF action == "archive" → set status = ARCHIVED
          - IF action == "silence" → set status = SILENT
    5. RETURN ForgetResponse(success, affected_count)
```

---

## Task 3.6: Stats Operation

### Sub-Task 3.6.1: Async Stats

**Implementation**:
```python
async def stats(self) -> StatsResponse:
    """Get memory statistics for the current tenant.

    Returns:
        StatsResponse with counts, averages, and breakdowns.
    """
    data = await self._transport.request("GET", "/api/v1/memory/stats")
    return StatsResponse(**data)
```

**Pseudo-code**:
```
ASYNC METHOD stats() -> StatsResponse:
    1. GET /api/v1/memory/stats (tenant from header)
    2. Parse response:
       - total_memories: int
       - active_memories: int
       - silent_memories: int
       - archived_memories: int
       - by_type: Dict[str, int]  e.g. {"episodic_event": 42, "semantic_fact": 15}
       - avg_confidence: float
       - avg_importance: float
       - oldest_memory: datetime | None
       - newest_memory: datetime | None
       - estimated_size_mb: float
    3. RETURN StatsResponse
```

---

## Task 3.7: Session Management

### Sub-Task 3.7.1: Create Session

**Implementation**:
```python
async def create_session(
    self,
    *,
    name: Optional[str] = None,
    ttl_hours: int = 24,
    metadata: Optional[Dict[str, Any]] = None,
) -> SessionResponse:
    """Create a new memory session.

    Args:
        name: Optional human-readable session name.
        ttl_hours: Session time-to-live in hours (default: 24).
        metadata: Optional session metadata.

    Returns:
        SessionResponse with session_id, created_at, expires_at.
    """
    payload = {
        "name": name,
        "ttl_hours": ttl_hours,
        "metadata": metadata or {},
    }
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    data = await self._transport.request("POST", "/api/v1/session/create", json=payload)
    return SessionResponse(**data)
```

### Sub-Task 3.7.2: Get Session Context

**Implementation**:
```python
async def get_session_context(
    self,
    session_id: str,
) -> Dict[str, Any]:
    """Get full session context for LLM injection.

    Args:
        session_id: The session to retrieve context for.

    Returns:
        Dict with messages, tool_results, scratch_pad, and context_string.
    """
    data = await self._transport.request(
        "GET",
        f"/api/v1/session/{session_id}/context",
    )
    return data
```

---

## Task 3.8: Health Check

### Sub-Task 3.8.1: Health

**Implementation**:
```python
async def health(self) -> HealthResponse:
    """Check CML server health.

    Returns:
        HealthResponse with status and component details.
    """
    data = await self._transport.request("GET", "/api/v1/health")
    return HealthResponse(**data)
```

---

## Task 3.9: Delete All (GDPR)

### Sub-Task 3.9.1: Delete All Memories

**Implementation**:
```python
async def delete_all(self, *, confirm: bool = False) -> int:
    """Delete ALL memories for the current tenant (GDPR compliance).

    Args:
        confirm: Must be True to execute. Safety check.

    Returns:
        Number of memories deleted.

    Raises:
        ValueError: If confirm is not True.
    """
    if not confirm:
        raise ValueError(
            "delete_all() requires confirm=True. "
            "This permanently deletes ALL memories for the tenant."
        )
    data = await self._transport.request(
        "DELETE",
        "/api/v1/memory/all",
        use_admin_key=True,
    )
    return data.get("affected_count", 0)
```

**Pseudo-code**:
```
ASYNC METHOD delete_all(confirm=False) -> int:
    1. Safety check: confirm must be True
    2. DELETE /api/v1/memory/all (uses admin API key)
    3. Server paginates and deletes all records for tenant
    4. RETURN affected_count
```

---

## Task 3.10: Convenience Methods

### Sub-Task 3.10.1: Quick Memory Context

**Architecture**: Provide a one-line method to get memory context for LLM prompt injection.

**Implementation**:
```python
async def get_context(
    self,
    query: str,
    *,
    max_results: int = 10,
) -> str:
    """Get formatted memory context string for LLM prompt injection.

    Convenience method that calls read() with format="llm_context"
    and returns just the context string.

    Args:
        query: The search query.
        max_results: Maximum memories to include.

    Returns:
        Formatted context string ready for LLM prompt.
    """
    result = await self.read(query, max_results=max_results, format="llm_context")
    return result.context
```

### Sub-Task 3.10.2: Remember (Alias for Write)

**Implementation**:
```python
async def remember(self, content: str, **kwargs) -> WriteResponse:
    """Alias for write(). More intuitive for some use cases.

    Example:
        await memory.remember("User's birthday is March 15th")
    """
    return await self.write(content, **kwargs)
```

### Sub-Task 3.10.3: Search (Alias for Read)

**Implementation**:
```python
async def search(self, query: str, **kwargs) -> ReadResponse:
    """Alias for read(). More intuitive for search-oriented use cases.

    Example:
        results = await memory.search("birthday")
    """
    return await self.read(query, **kwargs)
```

---

## Task 3.11: Sync Client Implementation

### Sub-Task 3.11.1: Mirror All Methods

**Architecture**: The sync `CognitiveMemoryLayer` class mirrors every method of `AsyncCognitiveMemoryLayer`, but uses `HTTPTransport` (sync) instead of `AsyncHTTPTransport`.

**Pseudo-code**:
```
FOR EACH async method IN AsyncCognitiveMemoryLayer:
    CREATE sync method with same signature (minus async/await)
    REPLACE: await self._transport.request(...) → self._transport.request(...)
    RETURN same response type

Methods to mirror:
    write()            → sync write()
    read()             → sync read()
    turn()             → sync turn()
    update()           → sync update()
    forget()           → sync forget()
    stats()            → sync stats()
    health()           → sync health()
    create_session()   → sync create_session()
    get_session_context() → sync get_session_context()
    delete_all()       → sync delete_all()
    get_context()      → sync get_context()
    remember()         → sync remember()
    search()           → sync search()
```

---

## Task 3.12: Integration Patterns

### Sub-Task 3.12.1: OpenAI Integration Example

**Pseudo-code**:
```python
from openai import OpenAI
from cml import CognitiveMemoryLayer

openai_client = OpenAI()
memory = CognitiveMemoryLayer(api_key="sk-cml-...")

def chat_with_memory(user_message: str, session_id: str) -> str:
    # 1. Get memory context
    turn = memory.turn(user_message=user_message, session_id=session_id)

    # 2. Build prompt with memory context
    messages = [
        {"role": "system", "content": f"You have the following memories:\n{turn.memory_context}"},
        {"role": "user", "content": user_message},
    ]

    # 3. Call LLM
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    assistant_message = response.choices[0].message.content

    # 4. Store the response for future recall
    memory.turn(
        user_message=user_message,
        assistant_response=assistant_message,
        session_id=session_id,
    )

    return assistant_message
```

### Sub-Task 3.12.2: LangChain Integration Pattern

**Pseudo-code**:
```python
from langchain.memory import BaseMemory
from cml import CognitiveMemoryLayer


class CMLMemory(BaseMemory):
    """LangChain-compatible memory using CognitiveMemoryLayer."""

    client: CognitiveMemoryLayer
    session_id: str
    memory_key: str = "memory_context"

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict) -> dict:
        query = inputs.get("input", "")
        context = self.client.get_context(query)
        return {self.memory_key: context}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        self.client.turn(
            user_message=user_input,
            assistant_response=ai_output,
            session_id=self.session_id,
        )

    def clear(self) -> None:
        self.client.delete_all(confirm=True)
```

---

## Acceptance Criteria

- [ ] All 13 methods implemented on both async and sync clients
- [ ] All methods have comprehensive docstrings with Args, Returns, Raises
- [ ] Request payloads use Pydantic models with exclude_none serialization
- [ ] Response parsing handles all fields from the CML API
- [ ] `forget()` validates at least one selector provided
- [ ] `delete_all()` requires explicit `confirm=True`
- [ ] `get_context()` returns just the formatted string
- [ ] Convenience aliases (`remember`, `search`) delegate correctly
- [ ] Type annotations are complete and pass mypy
- [ ] Integration patterns documented for OpenAI and LangChain
