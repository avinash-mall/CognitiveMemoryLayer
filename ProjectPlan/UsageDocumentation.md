# Cognitive Memory Layer - Usage Documentation

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [LLM Tool Calling Interface](#llm-tool-calling-interface)
4. [API Reference](#api-reference)
5. [Memory Types](#memory-types)
6. [Authentication](#authentication)
7. [Response Formats](#response-formats)
8. [Best Practices for LLM Integration](#best-practices-for-llm-integration)
9. [Setup and Deployment](#setup-and-deployment)
10. [Configuration Reference](#configuration-reference)
11. [Advanced Features](#advanced-features)

---

## Overview

The Cognitive Memory Layer is a neuro-inspired memory system designed for LLMs and AI agents. It provides persistent, intelligent memory that goes beyond simple context windows by:

- **Storing** information with automatic importance filtering (Write Gate)
- **Retrieving** relevant memories using hybrid search (semantic + graph + lexical)
- **Updating** memories with belief revision and reconsolidation
- **Forgetting** irrelevant information through intelligent decay and compression
- **Consolidating** episodic memories into semantic facts during "sleep cycles"

### Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      REST API (FastAPI)                         │
│   /memory/write  /memory/read  /memory/update  /memory/forget   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    Memory Orchestrator                           │
│   Write Gate → Hippocampal Store → Neocortical Store → Retrieval │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────────┐
│ Working Memory│   │ Episodic Store  │   │ Semantic Store    │
│ (Short-term)  │   │ (PostgreSQL +   │   │ (Neo4j Knowledge  │
│               │   │  pgvector)      │   │  Graph)           │
└───────────────┘   └─────────────────┘   └───────────────────┘
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

```bash
curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key-123" \
  -d '{
    "user_id": "user-001",
    "content": "The user prefers vegetarian food and lives in Paris."
  }'
```

### 4. Retrieve Memories

```bash
curl -X POST http://localhost:8000/api/v1/memory/read \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key-123" \
  -d '{
    "user_id": "user-001",
    "query": "What are the user dietary preferences?"
  }'
```

---

## LLM Tool Calling Interface

This section provides tool definitions that LLMs can use to interact with the Cognitive Memory Layer. These definitions follow the standard function calling format used by OpenAI, Anthropic, and other LLM providers.

### Tool Definitions for LLMs

#### 1. memory_write

Store new information in long-term memory.

```json
{
  "name": "memory_write",
  "description": "Store new information in the user's long-term memory. Use this when the user shares important personal information, preferences, facts about themselves, or when you learn something significant about them. The system automatically filters trivial information.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "Unique identifier for the user"
      },
      "content": {
        "type": "string",
        "description": "The information to store. Be specific and factual."
      },
      "memory_type": {
        "type": "string",
        "enum": ["episodic_event", "semantic_fact", "preference", "task_state", "procedure", "constraint", "hypothesis"],
        "description": "Type of memory. Use 'semantic_fact' for facts, 'preference' for preferences, 'episodic_event' for specific events, 'constraint' for rules that must be followed."
      },
      "metadata": {
        "type": "object",
        "description": "Optional additional context (e.g., source, category)"
      }
    },
    "required": ["user_id", "content"]
  }
}
```

**Example Usage:**
```json
{
  "name": "memory_write",
  "arguments": {
    "user_id": "user-123",
    "content": "User is allergic to peanuts and requires all food recommendations to avoid peanut ingredients.",
    "memory_type": "constraint"
  }
}
```

#### 2. memory_read

Retrieve relevant memories for a query.

```json
{
  "name": "memory_read",
  "description": "Retrieve relevant memories about the user to inform your response. Use this before answering questions about the user's preferences, history, or personal information.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "Unique identifier for the user"
      },
      "query": {
        "type": "string",
        "description": "Natural language query describing what information you need"
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of memories to retrieve (default: 10, max: 50)",
        "default": 10
      },
      "memory_types": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["episodic_event", "semantic_fact", "preference", "task_state", "procedure", "constraint", "hypothesis"]
        },
        "description": "Filter by specific memory types"
      },
      "format": {
        "type": "string",
        "enum": ["packet", "list", "llm_context"],
        "description": "Response format. Use 'llm_context' for a pre-formatted markdown string ready for LLM consumption.",
        "default": "packet"
      }
    },
    "required": ["user_id", "query"]
  }
}
```

**Example Usage:**
```json
{
  "name": "memory_read",
  "arguments": {
    "user_id": "user-123",
    "query": "What dietary restrictions does the user have?",
    "memory_types": ["preference", "constraint"],
    "format": "llm_context"
  }
}
```

#### 3. memory_update

Update or provide feedback on an existing memory.

```json
{
  "name": "memory_update",
  "description": "Update an existing memory or provide feedback. Use when the user corrects information, confirms a fact, or when information becomes outdated.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "Unique identifier for the user"
      },
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
    "required": ["user_id", "memory_id"]
  }
}
```

**Example Usage:**
```json
{
  "name": "memory_update",
  "arguments": {
    "user_id": "user-123",
    "memory_id": "550e8400-e29b-41d4-a716-446655440000",
    "feedback": "incorrect"
  }
}
```

#### 4. memory_forget

Remove or silence memories.

```json
{
  "name": "memory_forget",
  "description": "Forget specific memories. Use when the user explicitly requests deletion or when information should no longer be used.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "Unique identifier for the user"
      },
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
    },
    "required": ["user_id"]
  }
}
```

**Example Usage:**
```json
{
  "name": "memory_forget",
  "arguments": {
    "user_id": "user-123",
    "query": "old address information",
    "action": "archive"
  }
}
```

#### 5. memory_stats

Get statistics about a user's memories.

```json
{
  "name": "memory_stats",
  "description": "Get statistics about a user's stored memories. Useful for understanding memory usage and health.",
  "parameters": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "Unique identifier for the user"
      }
    },
    "required": ["user_id"]
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

**Request Body:**
```json
{
  "user_id": "string (required)",
  "content": "string (required)",
  "memory_type": "episodic_event|semantic_fact|preference|task_state|procedure|constraint|hypothesis (optional)",
  "metadata": { "key": "value" },
  "turn_id": "string (optional - for conversation tracking)",
  "agent_id": "string (optional - which agent wrote this)"
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

#### POST /memory/read

Retrieve relevant memories.

**Request Headers:**
- `X-API-Key: <api_key>` (required)
- `Content-Type: application/json`

**Request Body:**
```json
{
  "user_id": "string (required)",
  "query": "string (required)",
  "max_results": 10,
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
  "user_id": "string (required)",
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
  "user_id": "string (required)",
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

#### GET /memory/stats/{user_id}

Get memory statistics.

**Request Headers:**
- `X-API-Key: <api_key>` (required)

**Response:**
```json
{
  "user_id": "user-123",
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

### API Keys

The system uses API key authentication via the `X-API-Key` header.

**Built-in Keys (for development):**
- `demo-key-123` - Standard read/write access
- `admin-key-456` - Full admin access

**Headers:**
```
X-API-Key: demo-key-123
X-Tenant-ID: optional-tenant-id
X-User-ID: optional-user-override
```

### Permissions

| Permission | demo-key-123 | admin-key-456 |
|------------|--------------|---------------|
| Read | ✓ | ✓ |
| Write | ✓ | ✓ |
| Admin | ✗ | ✓ |

### Multi-Tenancy

The system supports multi-tenant isolation:
- Each API key is associated with a `tenant_id`
- All operations are scoped to the tenant
- Users cannot access other tenants' data

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
# Pseudo-code for conversation flow
async def handle_message(user_id: str, message: str):
    # First, read relevant context
    context = await memory_read(
        user_id=user_id,
        query=message,
        format="llm_context"
    )
    
    # Include in system prompt
    system_prompt = f"""
    You are a helpful assistant. Here is what you know about the user:
    
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
        user_id=user_id,
        content=extracted_fact,
        memory_type="semantic_fact"
    )
```

### 3. Handling Conflicts

When memory read returns warnings about conflicts:

```python
context = await memory_read(user_id, query, format="packet")

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
    user_id=user_id,
    memory_id=retrieved_memory_id,
    feedback="correct"
)

# User corrects: "No, I actually live in London now"
await memory_update(
    user_id=user_id,
    memory_id=old_memory_id,
    feedback="outdated"
)
await memory_write(
    user_id=user_id,
    content="User lives in London",
    memory_type="semantic_fact"
)
```

### 5. Respecting Constraints

Always check for constraints before generating responses:

```python
context = await memory_read(
    user_id=user_id,
    query=current_topic,
    memory_types=["constraint"],
    format="packet"
)

constraints = [m.text for m in context.constraints]
# Include constraints in system prompt as hard rules
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

# Optional: vLLM for local LLM compression
LLM__VLLM_BASE_URL=http://vllm:8000/v1
LLM__VLLM_MODEL=meta-llama/Llama-3.2-1B-Instruct
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

# Test write
curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key-123" \
  -d '{"user_id": "test", "content": "Test memory"}'
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
export LLM__VLLM_BASE_URL=http://localhost:8000/v1
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
| `LLM__PROVIDER` | `openai` | Provider: `openai` or `vllm` |
| `LLM__MODEL` | `gpt-4o-mini` | Model name |
| `LLM__TEMPERATURE` | `0.0` | Generation temperature |
| `LLM__API_KEY` | None | API key |
| `LLM__VLLM_BASE_URL` | None | vLLM server URL |
| `LLM__VLLM_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` | vLLM model |

#### Memory Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY__SENSORY_BUFFER_MAX_TOKENS` | `500` | Max tokens in sensory buffer |
| `MEMORY__SENSORY_BUFFER_DECAY_SECONDS` | `30.0` | Buffer decay time |
| `MEMORY__WORKING_MEMORY_MAX_CHUNKS` | `10` | Max working memory chunks |
| `MEMORY__WRITE_GATE_THRESHOLD` | `0.3` | Importance threshold for storage |
| `MEMORY__CONSOLIDATION_INTERVAL_HOURS` | `6` | Hours between consolidation |
| `MEMORY__FORGETTING_INTERVAL_HOURS` | `24` | Hours between forgetting runs |

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
curl -X POST http://localhost:8000/api/v1/admin/consolidate/user-123 \
  -H "X-API-Key: admin-key-456"
```

### Active Forgetting

The forgetting system:
1. Scores memories by relevance (importance, recency, access frequency)
2. Plans actions: decay, silence, compress, archive, delete
3. Checks dependencies before deletion
4. Optionally uses LLM for compression

**Trigger manually (admin):**
```bash
curl -X POST "http://localhost:8000/api/v1/admin/forget/user-123?dry_run=true" \
  -H "X-API-Key: admin-key-456"
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

Delete all user data:
```bash
curl -X DELETE http://localhost:8000/api/v1/admin/user/user-123 \
  -H "X-API-Key: admin-key-456"
```

---

## Troubleshooting

### Common Issues

**1. "API key required" error**
- Ensure `X-API-Key` header is present
- Use `demo-key-123` for development

**2. "No significant information to store"**
- The Write Gate filtered the content as low importance
- Try more specific, factual content
- Lower `MEMORY__WRITE_GATE_THRESHOLD` (not recommended for production)

**3. Empty retrieval results**
- Verify memories exist for the user
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

For questions or issues, refer to the project documentation in the `ProjectPlan` folder.
