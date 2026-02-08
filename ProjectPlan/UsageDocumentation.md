# Cognitive Memory Layer - Usage Documentation

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [LLM Tool Calling Interface](#llm-tool-calling-interface)
4. [API Reference](#api-reference)
5. [Dashboard (Monitoring & Management)](#dashboard-monitoring--management)
6. [Holistic memory and context tags](#holistic-memory-and-context-tags)
7. [Memory Types](#memory-types)
8. [Authentication](#authentication)
9. [Response Formats](#response-formats)
10. [Best Practices for LLM Integration](#best-practices-for-llm-integration)
11. [Setup and Deployment](#setup-and-deployment)
12. [Configuration Reference](#configuration-reference)
13. [Advanced Features](#advanced-features)

---

## Overview

The Cognitive Memory Layer is a neuro-inspired memory system designed for LLMs and AI agents. It provides persistent, intelligent memory that goes beyond simple context windows by:

- **Storing** information with automatic importance filtering (Write Gate)
- **Retrieving** relevant memories using hybrid search (semantic + graph + lexical)
- **Updating** memories with belief revision and reconsolidation
- **Forgetting** irrelevant information through intelligent decay and compression
- **Consolidating** episodic memories into semantic facts during "sleep cycles"

### Architecture Summary

```mermaid
flowchart TB
    subgraph API["REST API (FastAPI)"]
        endpoints["/memory/write  /memory/read  /memory/update  /memory/forget"]
    end
    
    subgraph Orch["Memory Orchestrator"]
        flow["Write Gate → Hippocampal Store → Neocortical Store → Retrieval"]
    end
    
    subgraph Stores["Memory Stores"]
        working["Working Memory<br/>(Short-term)"]
        episodic["Episodic Store<br/>(PostgreSQL + pgvector)"]
        semantic["Semantic Store<br/>(Neo4j Knowledge Graph)"]
    end
    
    API --> Orch
    Orch --> working & episodic & semantic

    style API fill:#e3f2fd,color:#000
    style Orch fill:#fff3e0,color:#000
    style Stores fill:#e8f5e9,color:#000
```

---

## Quick Start

### 1. Start the Infrastructure

```bash
# Start all services (Postgres, Neo4j, Redis, API)
docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
docker compose -f docker/docker-compose.yml up api
```

### 2. Test the Health Endpoint

```bash
curl http://localhost:8000/api/v1/health
```

### 3. Store Your First Memory

Memory is **holistic** per tenant (no scopes). Use optional `context_tags` and `session_id` for categorization and origin tracking.

```bash
curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AUTH__API_KEY" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "content": "The user prefers vegetarian food and lives in Paris.",
    "context_tags": ["preference", "personal"],
    "session_id": "demo-session-001"
  }'
```
(Set `AUTH__API_KEY` in your environment or use a key you configured.)

### 4. Retrieve Memories

```bash
curl -X POST http://localhost:8000/api/v1/memory/read \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AUTH__API_KEY" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "query": "What are the user dietary preferences?",
    "context_filter": ["preference", "personal"]
  }'
```

### 5. Seamless Memory (per turn)

For chat integrations, use **Seamless Memory**: one call per turn to auto-retrieve context and optionally auto-store.

```bash
curl -X POST http://localhost:8000/api/v1/memory/turn \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AUTH__API_KEY" \
  -H "X-Tenant-ID: demo" \
  -d '{
    "user_message": "What do I like to eat?",
    "assistant_response": null,
    "session_id": "session-001",
    "max_context_tokens": 1500
  }'
```
Response includes `memory_context` (ready to inject into your LLM prompt), `memories_retrieved`, and `memories_stored`.

---

## LLM Tool Calling Interface

This section provides tool definitions that LLMs can use to interact with the Cognitive Memory Layer. These definitions follow the standard function calling format used by OpenAI, Anthropic, and other LLM providers.

### Tool Definitions for LLMs

#### 1. memory_write

Store important information in long-term memory. The system automatically manages context.

```json
{
  "name": "memory_write",
  "description": "Store important information in long-term memory. The system automatically manages context. Use when the user shares personal information, preferences, or significant facts.",
  "parameters": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "The information to store. Be specific and factual."
      },
      "memory_type": {
        "type": "string",
        "enum": ["episodic_event", "semantic_fact", "preference", "task_state", "procedure", "constraint", "hypothesis"],
        "description": "Type of memory. Use 'semantic_fact' for facts, 'preference' for preferences, 'constraint' for rules that must be followed."
      },
      "context_tags": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Optional tags for retrieval filtering (e.g. personal, conversation, task)"
      },
      "session_id": {
        "type": "string",
        "description": "Optional session ID for origin tracking"
      }
    },
    "required": ["content"]
  }
}
```

**Example Usage:**
```json
{
  "name": "memory_write",
  "arguments": {
    "content": "User is allergic to peanuts and requires all food recommendations to avoid peanut ingredients.",
    "memory_type": "constraint"
  }
}
```

#### 2. memory_read

Retrieve relevant memories. Usually automatic via `/memory/turn`; call explicitly for specific queries.

```json
{
  "name": "memory_read",
  "description": "Retrieve relevant memories. Usually automatic, but call explicitly for specific queries (e.g. user preferences, history, or personal information).",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language query describing what information you need"
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of memories to retrieve (default: 10, max: 50)",
        "default": 10
      },
      "context_filter": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Optional filter by context tags"
      },
      "format": {
        "type": "string",
        "enum": ["packet", "list", "llm_context"],
        "description": "Response format. Use 'llm_context' for a pre-formatted markdown string ready for LLM consumption.",
        "default": "packet"
      }
    },
    "required": ["query"]
  }
}
```

**Example Usage:**
```json
{
  "name": "memory_read",
  "arguments": {
    "query": "What dietary restrictions does the user have?",
    "format": "llm_context"
  }
}
```

#### 3. memory_update

Update or provide feedback on an existing memory. Holistic: tenant from auth.

```json
{
  "name": "memory_update",
  "description": "Update an existing memory or provide feedback. Use when the user corrects information, confirms a fact, or when information becomes outdated.",
  "parameters": {
    "type": "object",
    "properties": {
      "memory_id": {
        "type": "string",
        "description": "UUID of the memory to update"
      },
      "text": {
        "type": "string",
        "description": "New text content for the memory"
      },
      "confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": "Updated confidence score (0-1)"
      },
      "feedback": {
        "type": "string",
        "enum": ["correct", "incorrect", "outdated"],
        "description": "Feedback type: 'correct' reinforces the memory, 'incorrect' marks it invalid, 'outdated' adds a validity end date"
      }
    },
    "required": ["memory_id"]
  }
}
```

**Example Usage:**
```json
{
  "name": "memory_update",
  "arguments": {
    "memory_id": "550e8400-e29b-41d4-a716-446655440000",
    "feedback": "incorrect"
  }
}
```

#### 4. memory_forget

Remove or silence memories. Holistic: tenant from auth.

```json
{
  "name": "memory_forget",
  "description": "Forget specific memories. Use when the user explicitly requests deletion or when information should no longer be used.",
  "parameters": {
    "type": "object",
    "properties": {
      "memory_ids": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Specific memory UUIDs to forget"
      },
      "query": {
        "type": "string",
        "description": "Natural language query to find memories to forget"
      },
      "before": {
        "type": "string",
        "format": "date-time",
        "description": "Forget memories older than this date"
      },
      "action": {
        "type": "string",
        "enum": ["delete", "archive", "silence"],
        "description": "Action: 'delete' removes permanently, 'archive' keeps but hides, 'silence' makes harder to retrieve",
        "default": "delete"
      }
    }
  }
}
```

**Example Usage:**
```json
{
  "name": "memory_forget",
  "arguments": {
    "query": "old address information",
    "action": "archive"
  }
}
```

#### 5. memory_stats

Get statistics about memories for the tenant (no parameters; tenant from auth).

```json
{
  "name": "memory_stats",
  "description": "Get statistics about stored memories for the current tenant.",
  "parameters": {
    "type": "object",
    "properties": {}
  }
}
```

---

## API Reference

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

#### POST /memory/write

Store new information in memory.

**Request Headers:**
- `X-API-Key: <api_key>` (required)
- `Content-Type: application/json`

**Request Headers:**
- `X-Tenant-ID: <tenant_id>` (optional; default from API key)

**Request Body:**
```json
{
  "content": "string (required)",
  "context_tags": ["personal", "conversation"],
  "session_id": "string (optional - origin tracking)",
  "memory_type": "episodic_event|semantic_fact|preference|task_state|procedure|constraint|hypothesis (optional)",
  "metadata": { "key": "value" },
  "turn_id": "string (optional)",
  "agent_id": "string (optional)"
}
```

**Response:**
```json
{
  "success": true,
  "memory_id": "uuid",
  "chunks_created": 1,
  "message": "Memory stored successfully"
}
```

**Notes:**
- The Write Gate automatically filters low-importance content
- PII is automatically redacted before storage
- Content is chunked semantically if too long

---

#### POST /memory/turn (Seamless Memory)

Process a conversation turn: auto-retrieve relevant context and optionally auto-store. Returns `memory_context` ready to inject into your LLM prompt.

**Request Body:**
```json
{
  "user_message": "string (required)",
  "assistant_response": "string (optional)",
  "session_id": "string (optional)",
  "max_context_tokens": 1500
}
```

**Response:**
```json
{
  "memory_context": "# Retrieved Memory Context\n\n## Facts\n- ...",
  "memories_retrieved": 5,
  "memories_stored": 1,
  "reconsolidation_applied": false
}
```

---

#### POST /memory/read

Retrieve relevant memories. Holistic: tenant from auth.

**Request Headers:**
- `X-API-Key: <api_key>` (required)
- `Content-Type: application/json`
- `X-Tenant-ID: <tenant_id>` (optional)

**Request Body:**
```json
{
  "query": "string (required)",
  "max_results": 10,
  "context_filter": ["personal", "conversation"],
  "memory_types": ["semantic_fact", "preference"],
  "since": "2024-01-01T00:00:00Z",
  "until": "2024-12-31T23:59:59Z",
  "format": "packet|list|llm_context"
}
```

**Response (format: "packet"):**
```json
{
  "query": "user dietary preferences",
  "memories": [
    {
      "id": "uuid",
      "text": "User prefers vegetarian food",
      "type": "preference",
      "confidence": 0.95,
      "relevance": 0.87,
      "timestamp": "2024-01-15T10:30:00Z",
      "metadata": {}
    }
  ],
  "facts": [...],
  "preferences": [...],
  "episodes": [...],
  "llm_context": null,
  "total_count": 5,
  "elapsed_ms": 45.2
}
```

**Response (format: "llm_context"):**
```json
{
  "query": "user dietary preferences",
  "memories": [...],
  "llm_context": "# Retrieved Memory Context\n\n## Constraints (Must Follow)\n- **User is allergic to peanuts**\n\n## Known Facts\n- User is vegetarian [95%]\n\n## User Preferences\n- Prefers organic produce\n",
  "total_count": 3,
  "elapsed_ms": 52.1
}
```

---

#### POST /memory/update

Update an existing memory.

**Request Headers:**
- `X-API-Key: <api_key>` (required)
- `Content-Type: application/json`

**Request Body:**
```json
{
  "memory_id": "uuid (required)",
  "text": "Updated memory text (optional)",
  "confidence": 0.9,
  "importance": 0.8,
  "metadata": { "updated": true },
  "feedback": "correct|incorrect|outdated (optional)"
}
```

**Response:**
```json
{
  "success": true,
  "memory_id": "uuid",
  "version": 2,
  "message": "Memory updated successfully"
}
```

**Feedback Effects:**
- `correct`: Increases confidence by 0.2 (capped at 1.0)
- `incorrect`: Sets confidence to 0, marks as deleted
- `outdated`: Sets `valid_to` to current time

---

#### POST /memory/forget

Forget memories.

**Request Headers:**
- `X-API-Key: <api_key>` (required)
- `Content-Type: application/json`

**Request Body:**
```json
{
  "memory_ids": ["uuid1", "uuid2"],
  "query": "old address",
  "before": "2023-01-01T00:00:00Z",
  "action": "delete|archive|silence"
}
```

**Response:**
```json
{
  "success": true,
  "affected_count": 3,
  "message": "3 memories deleted"
}
```

---

#### GET /memory/stats

Get memory statistics for the authenticated tenant.

**Request Headers:**
- `X-API-Key: <api_key>` (required)

**Response:**
```json
{
  "total_memories": 150,
  "active_memories": 120,
  "silent_memories": 20,
  "archived_memories": 10,
  "by_type": {
    "semantic_fact": 45,
    "preference": 30,
    "episodic_event": 75
  },
  "avg_confidence": 0.78,
  "avg_importance": 0.65,
  "oldest_memory": "2024-01-01T10:00:00Z",
  "newest_memory": "2024-06-15T14:30:00Z",
  "estimated_size_mb": 0.15
}
```

---

#### Session Convenience Endpoints

For session-scoped operations, convenience endpoints are available:

**GET /session/{session_id}/context**

Get full session context for LLM injection.

**POST /session/{session_id}/write**

Write to memory with session_id for origin tracking.

**POST /session/{session_id}/read**

Read from memory (session_id for API compatibility).

---

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-06-15T14:30:00Z"
}
```

---

## Dashboard (Monitoring & Management)

A web-based dashboard provides comprehensive monitoring and management of the memory system. It is served from the same FastAPI application and requires **admin** API key authentication.

### Accessing the Dashboard

1. Start the API server (e.g. `docker compose -f docker/docker-compose.yml up api` or `uvicorn src.api.app:app --port 8000`).
2. Open a browser and go to **http://localhost:8000/dashboard**.
3. Enter your **admin API key** (the value of `AUTH__ADMIN_API_KEY`). The key is stored in the browser’s `localStorage` for the session.
4. Use the sidebar to switch between pages.

### Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI cards (total/active memories, avg confidence/importance, storage size, semantic facts), memory type/status charts, activity timeline, semantic facts by category, system health (PostgreSQL, Neo4j, Redis), recent events, events by type/operation, temporal range. |
| **Memory Explorer** | Paginated, filterable table of memory records. Filter by type, status, search text; sort by timestamp, confidence, importance, access count. Click a row to open the memory detail view. |
| **Memory Detail** | Full record view: content, key, namespace, context tags, entities/relations, metadata, confidence/importance gauges, access count, decay rate, provenance, version/supersedes, related events. |
| **Components** | Health status for PostgreSQL, Neo4j, and Redis (connection, latency, row/key counts). Short architecture description of sensory buffer, working memory, hippocampal/neocortical stores, consolidation, and forgetting. |
| **Events** | Paginated event log with filters (event type, operation). Expandable rows show full payload JSON. Optional auto-refresh every 5 seconds. |
| **Management** | Trigger **consolidation** (tenant/user) and **active forgetting** (tenant/user, dry-run, max memories). Results displayed in place. |

### Tenant Filtering

Use the **tenant selector** in the top bar to restrict overview, memory list, events, and management to a single tenant. "All Tenants" shows aggregated data.

### Dashboard API Endpoints

All dashboard endpoints live under `/api/v1/dashboard` and require **admin** permission (`X-API-Key` must be the admin key).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/dashboard/overview` | KPIs, type/status breakdowns, quality metrics, storage, semantic facts, event stats. Optional `tenant_id` query. |
| GET | `/api/v1/dashboard/memories` | Paginated memory list. Query params: `page`, `per_page`, `type`, `status`, `search`, `tenant_id`, `sort_by`, `order`. |
| GET | `/api/v1/dashboard/memories/{id}` | Full memory detail including related events. |
| GET | `/api/v1/dashboard/events` | Paginated event log. Query params: `page`, `per_page`, `event_type`, `operation`, `tenant_id`. |
| GET | `/api/v1/dashboard/timeline` | Memory counts per day for charts. Query params: `days`, `tenant_id`. |
| GET | `/api/v1/dashboard/components` | Health check for PostgreSQL, Neo4j, Redis (latency, counts). |
| GET | `/api/v1/dashboard/tenants` | List all tenants with memory/fact/event counts. |
| POST | `/api/v1/dashboard/consolidate` | Trigger consolidation. Body: `{ "tenant_id": "...", "user_id": "..." }`. |
| POST | `/api/v1/dashboard/forget` | Trigger forgetting. Body: `{ "tenant_id": "...", "user_id": "...", "dry_run": true, "max_memories": 5000 }`. |

### Implementation Notes

- **Frontend**: Vanilla HTML/CSS/JS SPA in `src/dashboard/static/` (no build step). Chart.js is loaded via CDN for overview charts.
- **Backend**: Routes in `src/api/dashboard_routes.py`; schemas in `src/api/schemas.py` (e.g. `DashboardOverview`, `DashboardMemoryDetail`).
- **Static files**: Served at `/dashboard/static`; SPA fallback at `/dashboard` and `/dashboard/*` serves `index.html`.

---

## Holistic memory and context tags

Memory access is **holistic per tenant**: there are no scopes or partitions. All memories for a tenant live in a single unified store. You identify the tenant via the `X-Tenant-ID` header (or the default tenant from your API key).

- **context_tags**: Optional list of tags (e.g. `personal`, `conversation`, `task`, `procedural`) for categorization. Use `context_filter` on read to optionally restrict retrieval to certain tags.
- **source_session_id**: Optional session ID when writing; used for origin tracking only, not for access control.
- **Seamless Memory**: Use the `/memory/turn` endpoint to auto-retrieve relevant context for each user message and optionally auto-store; the returned `memory_context` is ready to inject into your LLM prompt.

---

## Memory Types

| Type | Description | Use Case | Lifecycle |
|------|-------------|----------|-----------|
| `episodic_event` | What happened (full context) | Store conversation events, user actions | Fast decay unless reinforced |
| `semantic_fact` | Durable distilled facts | Store confirmed user information | Slow decay, high confidence |
| `preference` | User preferences | Store choices, likes/dislikes | Time-sliced on change |
| `task_state` | Current task progress | Track multi-step workflows | High churn, latest wins |
| `procedure` | How to do something | Store instructions, processes | Stable, reusable |
| `constraint` | Rules and policies | Store must-follow rules | Never auto-forget |
| `hypothesis` | Uncertain beliefs | Store inferences needing confirmation | Requires confirmation |
| `conversation` | Chat message/turn | Multi-turn dialogue tracking | Session-based |
| `message` | Single message | Individual message storage | Session-based |
| `tool_result` | Output from tool execution | Store tool/function outputs | Task-based |
| `reasoning_step` | Chain-of-thought step | Agent reasoning traces | Session-based |
| `scratch` | Temporary working memory | Short-lived working notes | Fast decay |
| `knowledge` | General world knowledge | Domain facts, procedures | Stable |
| `observation` | Agent observations | Environment/context observations | Session-based |
| `plan` | Agent plans/goals | Task planning, goals | Task-based |

### When to Use Each Type

**Use `semantic_fact` when:**
- User explicitly states a fact about themselves
- You've confirmed information multiple times
- Example: "User's name is John", "User works at Acme Corp"

**Use `preference` when:**
- User expresses a like, dislike, or choice
- Preferences may change over time
- Example: "User prefers dark mode", "User likes Italian food"

**Use `constraint` when:**
- Information that MUST be respected
- Safety-critical or compliance-related
- Example: "User is allergic to shellfish", "Never share user's email"

**Use `hypothesis` when:**
- You're inferring something not explicitly stated
- Needs user confirmation
- Example: "User might be interested in cooking based on questions"

**Use `episodic_event` when:**
- Storing raw conversation turns
- Recording what happened in a session
- Example: "On Jan 15, user asked about flight to Paris"

---

## Authentication

### API Keys (config-based)

The system uses API key authentication via the `X-API-Key` header. Keys are loaded from environment variables (no hardcoded keys).

**Environment variables:**
- `AUTH__API_KEY` – API key for read/write access (optional; if unset, no key is valid)
- `AUTH__ADMIN_API_KEY` – API key with admin permission (optional)
- `AUTH__DEFAULT_TENANT_ID` – Default tenant for authenticated requests (default: `default`)

**Headers:**
```
X-API-Key: <your-api-key>
X-Tenant-ID: optional-tenant-id
X-User-ID: optional-user-id override
```

**Example (development):** Set `AUTH__API_KEY=your-dev-key` and optionally `AUTH__ADMIN_API_KEY=your-admin-key` in `.env` or the shell before starting the API.

### Permissions

| Permission | Standard key (AUTH__API_KEY) | Admin key (AUTH__ADMIN_API_KEY) |
|------------|------------------------------|----------------------------------|
| Read | ✓ | ✓ |
| Write | ✓ | ✓ |
| Admin | ✗ | ✓ |

### Multi-Tenancy

The system supports multi-tenant isolation:
- Each API key is associated with a default `tenant_id` (from config)
- All operations are scoped to the tenant
- Tenants cannot access other tenants' data

---

## Response Formats

### Format: packet (default)

Returns categorized memories with full metadata:

```json
{
  "query": "...",
  "memories": [...],
  "facts": [...],
  "preferences": [...],
  "episodes": [...],
  "llm_context": null,
  "total_count": 10,
  "elapsed_ms": 45.0
}
```

### Format: llm_context

Returns a pre-formatted markdown string for direct LLM consumption:

```markdown
# Retrieved Memory Context

## Constraints (Must Follow)
- **Never share user's email address**

## Known Facts
- User lives in Paris [95%]
- User works at Acme Corp [88%]

## User Preferences
- Prefers vegetarian food
- Likes early morning meetings

## Recent Context
- [2024-06-01] User mentioned planning a trip to Japan

## Warnings
- ⚠️ Conflicting preferences detected for communication style
```

This format is ideal for injecting into system prompts or context windows.

---

## Best Practices for LLM Integration

### 1. When to Read Memory

**Always read before:**
- Answering questions about the user
- Making recommendations
- Starting a new conversation session

```python
# Example conversation flow (matches POST /api/v1/memory/read)
async def handle_message(message: str):
    # First, read relevant context
    context = await memory_read(
        query=message,
        format="llm_context"
    )
    
    # Include in system prompt
    system_prompt = f"""
    You are a helpful assistant. Here is what you know:
    
    {context.llm_context}
    
    Respond to the user's message.
    """
    
    response = await generate_response(system_prompt, message)
    return response
```

### 2. When to Write Memory

**Write when the user:**
- Shares personal information
- Expresses preferences
- Provides corrections
- Shares important context

**Don't write:**
- Casual conversation filler
- Repeated information (already stored)
- Temporary/session-specific data

```python
# After extracting important info from conversation
if contains_personal_info(message):
    await memory_write(
        content=extracted_fact,
        memory_type="semantic_fact"
    )
```

### 3. Handling Conflicts

When memory read returns warnings about conflicts:

```python
context = await memory_read(query=query, format="packet")

if context.warnings:
    # Ask user for clarification
    clarification_prompt = f"""
    I found some conflicting information:
    {context.warnings}
    
    Could you help me understand which is correct?
    """
```

### 4. Using Feedback for Learning

When users confirm or correct information:

```python
# User confirms: "Yes, that's correct!"
await memory_update(
    memory_id=retrieved_memory_id,
    feedback="correct"
)

# User corrects: "No, I actually live in London now"
await memory_update(
    memory_id=old_memory_id,
    feedback="outdated"
)
await memory_write(
    content="User lives in London",
    memory_type="semantic_fact"
)
```

### 5. Respecting Constraints

Always check for constraints before generating responses:

```python
context = await memory_read(
    query=current_topic,
    context_filter=["constraint"],  # Filter by context_tags
    format="llm_context"
)

# Constraints are included in llm_context output
# Include in system prompt as hard rules
```

---

## Setup and Deployment

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- OpenAI API key (for embeddings and LLM features)

### Docker Deployment (Recommended)

#### 1. Clone and Navigate

```bash
cd CognitiveMemoryLayer
```

#### 2. Create Environment File

Create a `.env` file in the project root:

```env
# Required for OpenAI embeddings
OPENAI_API_KEY=sk-your-key-here

# Or use nested format
EMBEDDING__API_KEY=sk-your-key-here
LLM__API_KEY=sk-your-key-here

# Optional: Use local embeddings instead
EMBEDDING__PROVIDER=local
EMBEDDING__LOCAL_MODEL=all-MiniLM-L6-v2

# Optional: Use alternative LLM endpoint (vLLM, Ollama, etc.)
LLM__BASE_URL=http://vllm:8000/v1
LLM__MODEL=meta-llama/Llama-3.2-1B-Instruct
```

#### 3. Build and Start Services

```bash
# Build the application image
docker compose -f docker/docker-compose.yml build app

# Start infrastructure
docker compose -f docker/docker-compose.yml up -d postgres neo4j redis

# Run database migrations
docker compose -f docker/docker-compose.yml run --rm app sh -c "alembic upgrade head"

# Start the API server
docker compose -f docker/docker-compose.yml up api
```

#### 4. Verify Installation

```bash
# Check health
curl http://localhost:8000/api/v1/health

# Test write (set AUTH__API_KEY in env first)
curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"content": "Test memory"}'
```

### Local Development Setup

#### 1. Install Dependencies

```bash
# Using Poetry
poetry install

# Or using pip
pip install -r requirements-docker.txt
```

#### 2. Start Infrastructure

```bash
# Start only database services
docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
```

#### 3. Set Environment Variables

```bash
export DATABASE__POSTGRES_URL="postgresql+asyncpg://memory:memory@localhost:5432/memory"
export DATABASE__NEO4J_URL="bolt://localhost:7687"
export DATABASE__NEO4J_USER="neo4j"
export DATABASE__NEO4J_PASSWORD="password"
export DATABASE__REDIS_URL="redis://localhost:6379"
export OPENAI_API_KEY="sk-your-key-here"
```

#### 4. Run Migrations

```bash
poetry run alembic upgrade head
# or
alembic upgrade head
```

#### 5. Start the API

```bash
poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# or
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Running Tests

```bash
# Run all tests with Docker
docker compose -f docker/docker-compose.yml run --rm app sh -c "alembic upgrade head && pytest tests -v --tb=short"

# Run specific test phases
pytest tests/unit -v
pytest tests/integration -v
pytest tests/e2e -v

# Run with coverage
pytest tests --cov=src --cov-report=html
```

### Optional: vLLM for Local LLM

For LLM-based compression without OpenAI:

```bash
# With GPU
docker compose -f docker/docker-compose.yml --profile vllm up -d vllm

# CPU only (slower)
docker compose -f docker/docker-compose.yml --profile vllm-cpu up -d vllm-cpu

# Set environment
export LLM__BASE_URL=http://localhost:8000/v1
```

---

## Configuration Reference

### Environment Variables

All configuration uses nested environment variables with `__` delimiter.

#### Database Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE__POSTGRES_URL` | `postgresql+asyncpg://localhost/memory` | PostgreSQL connection URL |
| `DATABASE__NEO4J_URL` | `bolt://localhost:7687` | Neo4j connection URL |
| `DATABASE__NEO4J_USER` | `neo4j` | Neo4j username |
| `DATABASE__NEO4J_PASSWORD` | `password` | Neo4j password |
| `DATABASE__REDIS_URL` | `redis://localhost:6379` | Redis connection URL |

#### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING__PROVIDER` | `openai` | Provider: `openai` or `local` |
| `EMBEDDING__MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING__DIMENSIONS` | `1536` | Embedding dimensions |
| `EMBEDDING__LOCAL_MODEL` | `all-MiniLM-L6-v2` | Local model (sentence-transformers) |
| `EMBEDDING__API_KEY` | None | OpenAI API key (or use `OPENAI_API_KEY`) |

#### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM__PROVIDER` | `openai` | Provider: `openai`, `vllm`, `ollama`, `gemini`, `claude` |
| `LLM__MODEL` | `gpt-4o-mini` | Model name |
| `LLM__API_KEY` | None | API key |
| `LLM__BASE_URL` | None | OpenAI-compatible endpoint (for vLLM, Ollama, or proxy) |

#### Auth Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH__API_KEY` | None | API key for read/write access |
| `AUTH__ADMIN_API_KEY` | None | API key with admin permission |
| `AUTH__DEFAULT_TENANT_ID` | `default` | Default tenant for authenticated requests |

---

## Advanced Features

### Consolidation (Sleep Cycle)

The consolidation engine runs periodically to:
1. Sample recent episodic memories
2. Cluster similar memories
3. Extract semantic facts (gists)
4. Migrate to neocortical store

**Trigger manually (admin):**
```bash
curl -X POST "http://localhost:8000/api/v1/admin/consolidate/USER_ID" \
  -H "X-API-Key: YOUR_ADMIN_API_KEY"
```
(Use the user_id to identify the user whose memories should be consolidated.)

### Active Forgetting

The forgetting system:
1. Scores memories by relevance (importance, recency, access frequency)
2. Plans actions: decay, silence, compress, archive, delete
3. Checks dependencies before deletion
4. Optionally uses LLM for compression

**Trigger manually (admin):**
```bash
curl -X POST "http://localhost:8000/api/v1/admin/forget/USER_ID?dry_run=true" \
  -H "X-API-Key: YOUR_ADMIN_API_KEY"
```

### Celery Background Tasks

For scheduled forgetting:

```bash
# Start Celery worker
celery -A src.celery_app worker -l info -Q forgetting

# Start Celery beat (scheduler)
celery -A src.celery_app beat -l info
```

### Prometheus Metrics

Available at `/metrics`:
- `memory_writes_total` - Counter by tenant and status
- `memory_reads_total` - Counter by tenant
- `retrieval_latency_seconds` - Histogram of retrieval times
- `memory_count` - Gauge of total memories

### GDPR Compliance

To delete all memories for a tenant/user, use the memory forget endpoint with the desired `memory_ids` or `query`. The admin API exposes consolidation and forgetting triggers only.

---

## Troubleshooting

### Common Issues

**1. "API key required" error**
- Ensure `X-API-Key` header is present
- Set `AUTH__API_KEY` in your environment (or `.env`) and use that value as the header

**2. "No significant information to store"**
- The Write Gate filtered the content as low importance
- Try more specific, factual content

**3. Empty retrieval results**
- Verify memories exist for the tenant
- Check the query is semantically related to stored content
- Ensure embeddings are being generated (check OpenAI key)

**4. Database connection errors**
- Verify infrastructure is running: `docker compose ps`
- Check connection strings in environment
- Run migrations: `alembic upgrade head`

### Logs

Enable debug logging:
```bash
export DEBUG=true
```

View structured logs in JSON format for parsing.

---

## API Documentation

Interactive API documentation is available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

**Web Dashboard (monitoring and management):** http://localhost:8000/dashboard — use your admin API key to sign in. See [Dashboard (Monitoring & Management)](#dashboard-monitoring--management) for details.

These provide full OpenAPI schema and allow testing endpoints directly in the browser.

---

## Summary

The Cognitive Memory Layer provides LLMs with a sophisticated memory system that mimics human memory architecture. Key points:

1. **Use `memory_read` before responding** to get relevant context
2. **Use `memory_write` for important information** - the system filters noise
3. **Use `memory_update` with feedback** to learn from corrections
4. **Use `memory_forget` for explicit deletion** requests
5. **Request `format: "llm_context"`** for easy system prompt injection
6. **Respect `constraints`** - they are safety-critical rules
7. **Use context tags** to categorize memories (e.g., "conversation", "preferences", "project_X")

For questions or issues, refer to the project documentation in the `ProjectPlan` folder.
