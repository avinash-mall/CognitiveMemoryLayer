# Cognitive Memory Layer - Usage Documentation

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Python SDK (cognitive-memory-layer)](#python-sdk-cognitive-memory-layer)
4. [LLM Tool Calling Interface](#llm-tool-calling-interface)
5. [API Reference](#api-reference)
6. [Dashboard (Monitoring & Management)](#dashboard-monitoring--management)
7. [Holistic memory and context tags](#holistic-memory-and-context-tags)
8. [Memory Types](#memory-types)
9. [Authentication](#authentication)
10. [Response Formats](#response-formats)
11. [Best Practices for LLM Integration](#best-practices-for-llm-integration)
12. [Setup and Deployment](#setup-and-deployment)
13. [Configuration Reference](#configuration-reference)
14. [Advanced Features](#advanced-features)

---

## Overview

The Cognitive Memory Layer is a neuro-inspired memory system designed for LLMs and AI agents. It provides persistent, intelligent memory that goes beyond simple context windows by:

- **Storing** information with automatic importance filtering (Write Gate)
- **Extracting** cognitive constraints (goals, values, policies, states, causal rules) at write time; entity and relation extraction uses batch processing in `encode_batch()` for lower latency on bulk writes
- **Retrieving** relevant memories using hybrid search (semantic + graph + constraints)
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

You can do the same flow from Python using the **cognitive-memory-layer** SDK; see [Python SDK (cognitive-memory-layer)](#python-sdk-cognitive-memory-layer) below.

---

## Python SDK (cognitive-memory-layer)

The **cognitive-memory-layer** package is the official Python SDK for the Cognitive Memory Layer (client mode and embedded mode). For installation, quick start, configuration, API reference, and examples, see:

- **[packages/py-cml/README.md](../packages/py-cml/README.md)** — overview, installation, quick start
- **[packages/py-cml/docs/](../packages/py-cml/docs/)** — [Getting Started](../packages/py-cml/docs/getting-started.md), [API Reference](../packages/py-cml/docs/api-reference.md), [Configuration](../packages/py-cml/docs/configuration.md), [Examples](../packages/py-cml/docs/examples.md), [Temporal Fidelity](../packages/py-cml/docs/temporal-fidelity.md)
- **[examples/](../examples/)** — runnable scripts (quickstart, chat, embedded, agent, etc.)

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
- `X-Tenant-ID: <tenant_id>` (optional; **admin key only** — standard keys use the tenant from config)
- `X-Eval-Mode: true` (optional) — When set, the response includes write-gate outcome and reason (see **Response** below). Use for benchmark/evaluation scripts to aggregate gating statistics (stored vs skipped and reasons).

**Request Body:**
```json
{
  "content": "string (required)",
  "context_tags": ["personal", "conversation"],
  "session_id": "string (optional - origin tracking)",
  "memory_type": "episodic_event|semantic_fact|preference|task_state|procedure|constraint|hypothesis (optional)",
  "metadata": { "key": "value" },
  "turn_id": "string (optional)",
  "agent_id": "string (optional)",
  "timestamp": "ISO 8601 datetime (optional - event time, defaults to now)"
}
```

**Response:**
```json
{
  "success": true,
  "memory_id": "uuid",
  "chunks_created": 1,
  "message": "Memory stored successfully",
  "eval_outcome": "stored",
  "eval_reason": "1 chunk(s) stored"
}
```

- `eval_outcome` and `eval_reason` are present only when the request included `X-Eval-Mode: true`. `eval_outcome` is `"stored"` or `"skipped"`; `eval_reason` is a short string (e.g. write-gate skip reason, or "N chunk(s) stored").

**Notes:**
- Request metadata is merged into the stored record; optional memory_type overrides automatic classification when provided.
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
  "max_context_tokens": 1500,
  "timestamp": "ISO 8601 datetime (optional - event time, defaults to now)",
  "user_timezone": "IANA timezone string (optional - e.g. America/New_York for retrieval today/yesterday)"
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

Retrieve relevant memories. Holistic: tenant from auth. The server applies memory_types, since, and until to retrieval; format controls response shape (packet vs flat list vs llm_context). Optional `user_timezone` (IANA, e.g. `America/New_York`) enables timezone-aware "today"/"yesterday" filters in the retrieval plan.

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
  "format": "packet|list|llm_context",
  "user_timezone": "IANA timezone string (optional - e.g. America/New_York for today/yesterday filters)"
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
  "constraints": [...],
  "llm_context": null,
  "total_count": 5,
  "elapsed_ms": 45.2
}
```

The `constraints` field contains cognitive constraints (goals, values, policies, states, causal rules) when `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` is true. Each item is a `MemoryItem` with `type: "constraint"`.

**Response (format: "llm_context"):**
```json
{
  "query": "user dietary preferences",
  "memories": [...],
  "llm_context": "# Retrieved Memory Context\n\n## Active Constraints (Must Follow)\n- **User is allergic to peanuts** (from turn_3)\n- **Trying to eat healthier** (from turn_7)\n\n## Known Facts\n- User is vegetarian [95%]\n\n## User Preferences\n- Prefers organic produce\n",
  "total_count": 3,
  "elapsed_ms": 52.1
}
```

**Response (format: "list"):** When format is `"list"`, the response contains a single flat list in `memories`; `facts`, `preferences`, `episodes`, and `constraints` are empty (no server-side categorization).

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

#### Session Endpoints

**POST /session/create**

Create a new memory session. Body: `{ "name": "optional", "ttl_hours": 24, "metadata": {} }`. Returns `session_id`, `created_at`, `expires_at`. Session is stored in Redis with TTL.

**POST /session/{session_id}/write**

Write to memory with `session_id` for origin tracking (body same as POST /memory/write).

**POST /session/{session_id}/read**

Read from memory (body same as POST /memory/read; session_id kept for API compatibility).

**GET /session/{session_id}/context**

Get full session context for LLM injection (messages, tool_results, scratch_pad, context_string). When session_id is provided, the server returns only memories scoped to that session (source_session_id).

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
| **Overview** | KPI cards (total/active memories, avg confidence/importance, storage size, semantic facts, labile memories, 24h requests), memory type/status charts, activity timeline, semantic facts by category, system health (PostgreSQL, Neo4j, Redis), recent events, reconsolidation queue status, request sparkline, events by type/operation, temporal range. |
| **Tenants** | Lists all tenants with memory counts (total and active), fact counts, event counts, last memory timestamp, and last event timestamp. Sortable by any column. Quick-link buttons navigate to Overview, Memories, or Events pre-filtered by that tenant. Summary KPI cards show total tenants, total memories, and the most active tenant. |
| **Memory Explorer** | Paginated, filterable table of memory records. Filter by type, status, search text; sort by timestamp, confidence, importance, access count. Click a row to open the memory detail view. **Bulk actions**: select memories via checkboxes and apply Archive, Silence, or Delete in bulk. **Export**: download memories as JSON. |
| **Sessions** | Active sessions from Redis with TTL, creation/expiry timestamps, and metadata. Displays TTL badges (green/yellow/red based on time remaining). Also shows memory counts per `source_session_id` from the database. Click a session ID to filter the Memory Explorer by that session. |
| **Memory Detail** | Full record view: content, key, namespace, context tags, entities/relations, metadata, confidence/importance gauges, access count, decay rate, provenance, version/supersedes, related events. |
| **Knowledge Graph** | Interactive graph visualization powered by vis-network. Shows graph stats (nodes, edges, entity types, tenants with graph data). Search for entities by name, select a tenant, adjust exploration depth (1-5 hops), and render an interactive network. Click nodes/edges to see details in a side panel. Entity type distribution table. |
| **API Usage** | Current rate-limit buckets from Redis with key type, identifier (masked for API keys), current count, limit, utilization progress bar, and TTL. KPI cards for active keys, avg utilization, configured RPM, and 24h request count. Chart.js line chart of hourly request volume. |
| **Components** | Health status for PostgreSQL, Neo4j, and Redis (connection, latency, row/key counts). Short architecture description of sensory buffer, working memory, hippocampal/neocortical stores, consolidation, and forgetting. |
| **Configuration** | Read-only config snapshot showing all settings grouped by section (Application, Database, Embedding, LLM, Auth). Secret values (API keys, passwords) are masked. Each setting shows its current value, default, source (env/override/default), and description. Editable settings (e.g. rate limit RPM, embedding model, debug mode) can be changed inline; overrides are stored in Redis. |
| **Retrieval Test** | Interactive query tool for debugging memory retrieval. Select a tenant, enter a query, set max results (1-50), optional context filter tags, and format (list/packet/llm_context). Returns scored memory results with relevance bars, type badges, confidence, timestamps, and metadata. If format is `llm_context`, shows the formatted context string in a code block. |
| **Events** | Paginated event log with filters (event type, operation). Expandable rows show full payload JSON. Optional auto-refresh every 5 seconds. |
| **Management** | Trigger **consolidation** (tenant/user), **active forgetting** (tenant/user, dry-run, max memories), and **reconsolidation** (release labile state per tenant). Results displayed in place. **Reconsolidation status** section shows labile memory counts from DB and Redis (scopes, sessions, memories) per tenant. **Job history** table shows the last N consolidation/forgetting/reconsolidation runs with type, tenant, status, dry-run flag, duration, and expandable result details. Jobs are persisted in the `dashboard_jobs` PostgreSQL table. |

### Tenant Filtering

Use the **tenant selector** in the top bar to restrict overview, memory list, events, sessions, and management to a single tenant. "All Tenants" shows aggregated data. On the Tenants page, quick-link buttons let you jump to a filtered view for any tenant.

### Dashboard API Endpoints

All dashboard endpoints live under `/api/v1/dashboard` and require **admin** permission (`X-API-Key` must be the admin key).

#### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/dashboard/overview` | KPIs, type/status breakdowns, quality metrics, storage, semantic facts, event stats. Optional `tenant_id` query. |
| GET | `/api/v1/dashboard/memories` | Paginated memory list. Query params: `page`, `per_page`, `type`, `status`, `search`, `tenant_id`, `source_session_id`, `sort_by`, `order`. |
| GET | `/api/v1/dashboard/memories/{id}` | Full memory detail including related events. |
| POST | `/api/v1/dashboard/memories/bulk-action` | Bulk memory actions. Body: `{ "memory_ids": ["..."], "action": "archive|silence|delete" }`. |
| GET | `/api/v1/dashboard/events` | Paginated event log. Query params: `page`, `per_page`, `event_type`, `operation`, `tenant_id`. |
| GET | `/api/v1/dashboard/timeline` | Memory counts per day for charts. Query params: `days`, `tenant_id`. |
| GET | `/api/v1/dashboard/components` | Health check for PostgreSQL, Neo4j, Redis (latency, counts). |
| GET | `/api/v1/dashboard/tenants` | List all tenants with memory/fact/event counts, active memory count, and last activity timestamps. |

#### Sessions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/dashboard/sessions` | Active sessions from Redis + memory counts per `source_session_id` from DB. Optional `tenant_id` query. |

#### Rate Limits & Request Stats

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/dashboard/ratelimits` | Current rate-limit usage per key from Redis (type, identifier, count, limit, utilization, TTL). |
| GET | `/api/v1/dashboard/request-stats` | Hourly request counts from Redis counters. Query param: `hours` (default 24, max 48). |

#### Knowledge Graph

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/dashboard/graph/stats` | Node/edge counts, entity type distribution, tenants with graph data. |
| GET | `/api/v1/dashboard/graph/explore` | Explore entity neighborhood. Query params: `tenant_id`, `entity`, `scope_id`, `depth`. |
| GET | `/api/v1/dashboard/graph/search` | Search entities by name pattern. Query params: `query`, `tenant_id`, `limit`. |

#### Configuration

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/dashboard/config` | Read-only config snapshot grouped by section with secrets masked. |
| PUT | `/api/v1/dashboard/config` | Update editable settings. Body: `{ "updates": { "key": "value" } }`. Stored in Redis. Only safe settings are writable. |

#### Labile / Reconsolidation

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/dashboard/labile` | DB labile counts + Redis scope/session/memory counts per tenant. Optional `tenant_id` query. |

#### Retrieval Test

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/dashboard/retrieval` | Test memory retrieval. Body: `{ "tenant_id", "query", "max_results", "context_filter", "memory_types", "format" }`. Returns scored results. |

#### Job History & Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/dashboard/jobs` | Paginated job history. Query params: `tenant_id`, `job_type`, `limit`. |
| POST | `/api/v1/dashboard/consolidate` | Trigger consolidation (with job tracking). Body: `{ "tenant_id": "...", "user_id": "..." }`. |
| POST | `/api/v1/dashboard/forget` | Trigger forgetting (with job tracking). Body: `{ "tenant_id": "...", "user_id": "...", "dry_run": true, "max_memories": 5000 }`. |
| POST | `/api/v1/dashboard/reconsolidate` | Release all labile state for a tenant (no belief revision). Body: `{ "tenant_id": "...", "user_id": "..." }`. |
| POST | `/api/v1/dashboard/database/reset` | Drop all tables and re-run migrations (use with caution). |

#### Export

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/dashboard/export/memories` | Export memories as JSON. Optional `tenant_id` query. |

### Implementation Notes

- **Frontend**: Vanilla HTML/CSS/JS SPA in `src/dashboard/static/` (no build step). Chart.js for charts; vis-network for knowledge graph visualization. Both loaded via CDN.
- **Backend**: Routes in `src/api/dashboard_routes.py`; schemas in `src/api/schemas.py` (e.g. `DashboardOverview`, `DashboardMemoryDetail`, `SessionInfo`, `GraphExploreResponse`, `ConfigSection`, `DashboardJobItem`, `DashboardRetrievalRequest`).
- **Job tracking**: Consolidation, forgetting, and reconsolidation runs are recorded in the `dashboard_jobs` PostgreSQL table (migration: `002_dashboard_jobs.py`). Job status, result, errors, and duration are tracked.
- **Request counting**: The `RequestLoggingMiddleware` increments hourly counters in Redis (`dashboard:reqcount:{YYYY-MM-DD-HH}`) with 48-hour TTL for the API Usage page.
- **Config overrides**: Runtime config changes are stored in Redis (`dashboard:config:overrides`) as JSON. Only safe settings (non-secrets, non-connection strings) are writable.
- **Static files**: Served at `/dashboard/static/*`; `/dashboard` and `/dashboard/*` (except `/dashboard/static/*`) serve `index.html` for client-side routing.

### Sidebar Navigation Order

1. Overview
2. Tenants
3. Memories
4. Sessions
5. Events
6. Knowledge Graph
7. API Usage
8. Components
9. Configuration
10. Retrieval Test
11. Management

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
- Cognitive constraints: goals, values, policies, states, causal rules
- Example: "User is allergic to shellfish", "Never share user's email", "We should save money for the trip", "I'm trying to eat healthier"
- When `FEATURES__CONSTRAINT_EXTRACTION_ENABLED=true`, constraints are also **automatically extracted** from text at write time by the `ConstraintExtractor` (goal/value/state/causal/policy patterns). They are stored both as episodic records with `MemoryType.CONSTRAINT` and as semantic facts with cognitive `FactCategory` values.

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
X-Tenant-ID: optional-tenant-id (admin key only; standard keys use tenant from API key config)
X-User-ID: optional-user-id override
```

**Example (development):** The project [.env.example](../.env.example) uses `AUTH__API_KEY=test-key` and `AUTH__ADMIN_API_KEY=test-key` for local development and for py-cml integration/e2e tests. Copy to `.env` or set these in your environment before starting the API. For production, use strong secrets.

### Permissions

| Permission | Standard key (AUTH__API_KEY) | Admin key (AUTH__ADMIN_API_KEY) |
|------------|------------------------------|----------------------------------|
| Read | ✓ | ✓ |
| Write | ✓ | ✓ |
| Admin | ✗ | ✓ |

### Multi-Tenancy

The system supports multi-tenant isolation:
- Each API key is associated with a default `tenant_id` (from `AUTH__DEFAULT_TENANT_ID` or config)
- All operations are scoped to the tenant
- **Tenant override**: The `X-Tenant-ID` header is honored only when using the **admin** API key; standard keys always use the tenant associated with that key
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
  "constraints": [...],
  "llm_context": null,
  "total_count": 10,
  "elapsed_ms": 45.0
}
```

The `constraints` field contains cognitive constraints (goals, values, policies, states, causal rules) extracted at write time when `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` is true. Each item is a `MemoryItem` with `type: "constraint"`. When no constraints are found, the list is empty.

### Format: list

When `format` is `"list"`, the response contains a single flat list in `memories`; `facts`, `preferences`, `episodes`, and `constraints` are empty (no server-side categorization).

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

**With the REST API:**
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

**With cognitive-memory-layer:** Use the **turn** endpoint for chat — it retrieves context and returns `memory_context` ready for your prompt. Or call **read** explicitly:

```python
from cml import CognitiveMemoryLayer

with CognitiveMemoryLayer(api_key="...", base_url="http://localhost:8000") as memory:
    # Option A: Seamless turn (retrieve + optional store in one call)
    turn = memory.turn(user_message=message, session_id="session-001")
    system_prompt = f"You are a helpful assistant. Relevant memories:\n\n{turn.memory_context}"

    # Option B: Explicit read with LLM-ready format
    result = memory.read(message, format="llm_context")
    system_prompt = f"You are a helpful assistant. Here is what you know:\n\n{result.context}"
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
# After extracting important info from conversation (REST or custom client)
if contains_personal_info(message):
    await memory_write(
        content=extracted_fact,
        memory_type="semantic_fact"
    )
```

**With cognitive-memory-layer:** Use `memory.write(...)` or store the exchange in one go with `memory.turn(..., assistant_response=assistant_msg)` so the server can extract and store new facts from the assistant reply.

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

**With cognitive-memory-layer:** Same flow using `memory.update(memory_id, feedback="correct"|"incorrect"|"outdated")` and `memory.write(content, ...)`.

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

- Docker and Docker Compose (for running the CML server)
- Python 3.11+ (for local development and for the cognitive-memory-layer SDK)
- OpenAI API key (for embeddings and LLM features)

### Using the Python SDK (cognitive-memory-layer)

To use CML from Python without writing HTTP calls yourself:

1. **Install the SDK:** `pip install cognitive-memory-layer` (or `pip install cognitive-memory-layer[embedded]` for embedded mode).
2. **Configure:** Set `CML_API_KEY` and `CML_BASE_URL` (e.g. `http://localhost:8000`), or pass them to `CognitiveMemoryLayer(...)`.
3. **Use the client:** See [Python SDK (cognitive-memory-layer)](#python-sdk-cognitive-memory-layer) and the [packages/py-cml](../packages/py-cml/) docs and examples (`getting-started.md`, `api-reference.md`, `examples/`).

### Docker Deployment (Recommended)

#### 1. Clone and Navigate

```bash
cd CognitiveMemoryLayer
```

#### 2. Create Environment File

Copy `.env.example` to `.env` in the project root and adjust as needed. The example sets `AUTH__API_KEY=test-key` and `AUTH__ADMIN_API_KEY=test-key` so the API and py-cml integration/e2e tests work without extra config. For production, set strong secrets.

```env
# Auth (required for API access; .env.example uses test-key for local/testing)
AUTH__API_KEY=test-key
AUTH__ADMIN_API_KEY=test-key

# Required for OpenAI embeddings (if using OpenAI)
OPENAI_API_KEY=sk-your-key-here
# Or use nested format: EMBEDDING__API_KEY=..., LLM__API_KEY=...

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
# Using Poetry (recommended; see pyproject.toml)
poetry install

# Or using pip (e.g. from requirements-docker.txt used in CI)
pip install -r requirements-docker.txt
```

#### 2. Start Infrastructure

```bash
# Start only database services (match URLs in .env or below)
docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
```

#### 3. Set Environment Variables

Copy `.env.example` to `.env` and edit, or set:

```bash
export DATABASE__POSTGRES_URL="postgresql+asyncpg://memory:memory@localhost:5432/memory"
export DATABASE__NEO4J_URL="bolt://localhost:7687"
export DATABASE__NEO4J_USER="neo4j"
export DATABASE__NEO4J_PASSWORD="password"
export DATABASE__REDIS_URL="redis://localhost:6379"
export AUTH__API_KEY="test-key"
export AUTH__ADMIN_API_KEY="test-key"
export OPENAI_API_KEY="sk-your-key-here"
```

#### 4. Run Migrations

```bash
poetry run alembic upgrade head
# or (with pip): alembic upgrade head
```

#### 5. Start the API

```bash
poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# or: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Running Tests

**Server tests** (from repository root):

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

**Python SDK (py-cml) tests** (from `packages/py-cml`):

1. **Unit tests** (no server): `pytest tests/unit/ -v`
2. **Integration and e2e tests** (require running CML API):
   - From repo root: `docker compose -f docker/docker-compose.yml up -d postgres neo4j redis api`
   - Ensure `.env` has `AUTH__API_KEY=test-key` and `AUTH__ADMIN_API_KEY=test-key` (see [.env.example](../.env.example)) so the API accepts the key used by tests.
   - From `packages/py-cml`: `pytest tests/integration/ tests/e2e/ -v -m "integration or e2e"`

If `CML_TEST_API_KEY` is not set, py-cml test conftests load the repo root `.env` and use `AUTH__API_KEY` / `AUTH__ADMIN_API_KEY`. Override with `CML_TEST_URL` and `CML_TEST_API_KEY` if needed.

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
| `DATABASE__POSTGRES_URL` | `postgresql+asyncpg://memory:memory@localhost/memory` | PostgreSQL connection URL (must use asyncpg driver) |
| `DATABASE__NEO4J_URL` | `bolt://localhost:7687` | Neo4j connection URL |
| `DATABASE__NEO4J_USER` | `neo4j` | Neo4j username |
| `DATABASE__NEO4J_PASSWORD` | *(empty)* | Neo4j password (set for remote/secured instances) |
| `DATABASE__REDIS_URL` | `redis://localhost:6379` | Redis connection URL |

#### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING__PROVIDER` | `openai` | Provider: `openai`, `local`, `openai_compatible`, `ollama` |
| `EMBEDDING__MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING__DIMENSIONS` | `1536` | Embedding dimensions (must match DB after migrations). Server and tests read from `.env`; Docker app/api use `.env` without override. |
| `EMBEDDING__LOCAL_MODEL` | `all-MiniLM-L6-v2` | Local model when provider=local (sentence-transformers) |
| `EMBEDDING__API_KEY` | None | API key (or use `OPENAI_API_KEY`) |
| `EMBEDDING__BASE_URL` | None | Base URL for openai_compatible/ollama embedding endpoint |

#### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM__PROVIDER` | `openai` | Provider: `openai`, `openai_compatible`, `ollama`, `gemini`, `claude` |
| `LLM__MODEL` | `gpt-4o-mini` | Model name |
| `LLM__API_KEY` | None | API key (or use `OPENAI_API_KEY`) |
| `LLM__BASE_URL` | None | OpenAI-compatible endpoint (for Ollama, vLLM, or proxy) |

#### LLM Internal (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_INTERNAL__PROVIDER` | None | Provider for internal tasks |
| `LLM_INTERNAL__MODEL` | None | Model for internal tasks |
| `LLM_INTERNAL__BASE_URL` | None | Base URL for internal LLM |
| `LLM_INTERNAL__API_KEY` | None | API key for internal LLM |

When any `LLM_INTERNAL__*` is set, used for SemanticChunker, Entity/Relation extractors, consolidation, reconsolidation, forgetting, QueryClassifier. If not set, default `LLM__*` is used.

#### Auth Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH__API_KEY` | None | API key for read/write access. Use `test-key` for local dev and py-cml integration/e2e tests (see [.env.example](../.env.example)). |
| `AUTH__ADMIN_API_KEY` | None | API key with admin permission (dashboard, consolidate, forget, list_tenants). Use `test-key` locally to match tests. |
| `AUTH__DEFAULT_TENANT_ID` | `default` | Default tenant for authenticated requests (overridable via `X-Tenant-ID` only when using admin key). |
| `AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Rate limit per tenant; `0` = disable. Use higher value for bulk eval (e.g. `600`). |

#### Feature Flags (optional performance & correctness)

All default to `true` unless noted. Set via `FEATURES__<NAME>=false` to disable.

| Variable | Default | Description |
|----------|---------|-------------|
| `FEATURES__STABLE_KEYS_ENABLED` | `true` | Use SHA256-based stable keys for consolidation facts and hippocampal records (avoids process-random collisions). |
| `FEATURES__WRITE_TIME_FACTS_ENABLED` | `true` | Populate semantic store at write time with rule-based preference/identity/location/occupation facts. |
| `FEATURES__BATCH_EMBEDDINGS_ENABLED` | `true` | Single `embed_batch()` per turn instead of per-chunk embedding calls. |
| `FEATURES__STORE_ASYNC` | `false` | Enqueue turn writes to Redis and process in background (reduces turn latency; requires Redis). |
| `FEATURES__CACHED_EMBEDDINGS_ENABLED` | `true` | Cache embeddings in Redis when Redis is configured. |
| `FEATURES__RETRIEVAL_TIMEOUTS_ENABLED` | `true` | Per-step and total retrieval timeouts (see Retrieval Settings). |
| `FEATURES__SKIP_IF_FOUND_CROSS_GROUP` | `true` | When a step has `skip_if_found` and returns results, skip remaining steps in other groups. |
| `FEATURES__DB_DEPENDENCY_COUNTS` | `true` | Forgetting uses one SQL query for dependency counts instead of O(n²) Python loop. |
| `FEATURES__BOUNDED_STATE_ENABLED` | `true` | Working/sensory in-memory state uses LRU+TTL (BoundedStateMap) to prevent unbounded growth. |
| `FEATURES__HNSW_EF_SEARCH_TUNING` | `true` | Set pgvector `hnsw.ef_search` at query time for recall/latency trade-off. |
| `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` | `true` | Extract cognitive constraints (goals, values, policies, states, causal rules) at write time. Constraints are stored as `MemoryType.CONSTRAINT` and as cognitive `FactCategory` facts. At read time, decision-style queries trigger constraint-first retrieval. See the [deep-research report](../evaluation/deep-research-report.md). |
| `FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR` | `true` | Use LLM (via unified extractor) for constraint extraction instead of rule-based. |
| `FEATURES__USE_LLM_WRITE_TIME_FACTS` | `true` | Use LLM (via unified extractor) for write-time fact extraction instead of rule-based. |
| `FEATURES__CHUNKER_REQUIRE_LLM` | `true` | Fail fast when `use_fast_chunker=false` and no LLM available; otherwise fallback to rule-based chunker. |
| `FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY` | `true` | Skip fast pattern path, always use LLM for query classification. |
| `FEATURES__USE_LLM_SALIENCE_REFINEMENT` | `true` | Use LLM salience from unified extractor instead of rule-based boosts. |
| `FEATURES__USE_LLM_PII_REDACTION` | `true` | Use LLM PII spans from unified extractor, merged with regex redaction. |
| `FEATURES__USE_LLM_WRITE_GATE_IMPORTANCE` | `true` | Use LLM importance from unified extractor instead of rule-based _compute_importance. |
| `FEATURES__USE_LLM_CONFLICT_DETECTION_ONLY` | `true` | Skip fast pattern path, always use LLM for conflict detection. |

When any write-path LLM flag is enabled, a single unified extraction call returns constraints, facts, salience, and importance in one LLM request. See [RuleBasedExtractorsAndLLMReplacement.md](../ProjectPlan/BaseCML/RuleBasedExtractorsAndLLMReplacement.md).

#### Retrieval Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS` | `5000` | Per-step timeout in milliseconds (when retrieval timeouts enabled). |
| `RETRIEVAL__TOTAL_TIMEOUT_MS` | `15000` | Total retrieval budget in milliseconds. |
| `RETRIEVAL__HNSW_EF_SEARCH` | `40` | Base `hnsw.ef_search` for pgvector; when tuning is enabled, max( this, top_k ) is used per query. |

Read requests can pass `user_timezone` (e.g. `America/New_York`) for timezone-aware "today"/"yesterday" filters in retrieval plans.

---

## Advanced Features

### Consolidation (Sleep Cycle)

The consolidation engine runs periodically to:
1. Sample recent episodic memories
2. Cluster similar memories
3. Extract semantic facts (gists)
4. Migrate to neocortical store

**Trigger manually (admin):**

- **REST (path param):** `POST /api/v1/admin/consolidate/{user_id}` with admin `X-API-Key`.
- **Dashboard API:** `POST /api/v1/dashboard/consolidate` with body `{ "tenant_id": "...", "user_id": "..." }` (admin key required).

```bash
curl -X POST "http://localhost:8000/api/v1/admin/consolidate/my-user-id" \
  -H "X-API-Key: YOUR_ADMIN_API_KEY"
```

### Active Forgetting

The forgetting system:
1. Scores memories by relevance (importance, recency, access frequency)
2. Plans actions: decay, silence, compress, archive, delete
3. Checks dependencies before deletion
4. Optionally uses LLM for compression

**Trigger manually (admin):**

- **REST (path param):** `POST /api/v1/admin/forget/{user_id}?dry_run=true` with admin `X-API-Key`.
- **Dashboard API:** `POST /api/v1/dashboard/forget` with body `{ "tenant_id", "user_id", "dry_run", "max_memories" }` (admin key required).

```bash
curl -X POST "http://localhost:8000/api/v1/admin/forget/my-user-id?dry_run=true" \
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
- `cml_retrieval_step_duration_seconds` - Per-step retrieval duration (when retrieval timeouts enabled)
- `cml_retrieval_step_result_count` - Results per step
- `cml_retrieval_timeout_total` - Count of retrieval steps that hit timeout
- `cml_fact_hit_rate_*` - Fact hit rate metrics (when applicable)

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

1. **Use `memory_read` before responding** to get relevant context (or use **cognitive-memory-layer** `turn()` for seamless retrieve + store).
2. **Use `memory_write` for important information** — the system filters noise (cognitive-memory-layer: `write()` or `remember()`).
3. **Use `memory_update` with feedback** to learn from corrections (cognitive-memory-layer: `update(..., feedback="correct"|"incorrect"|"outdated")`).
4. **Use `memory_forget` for explicit deletion** requests (cognitive-memory-layer: `forget()`).
5. **Request `format: "llm_context"`** for easy system prompt injection (cognitive-memory-layer: `read(..., format="llm_context")` or `get_context(query)`).
6. **Respect `constraints`** — they are safety-critical rules.
7. **Use context tags** to categorize memories (e.g., "conversation", "preferences", "project_X").

**Python users:** Use the **cognitive-memory-layer** package (`pip install cognitive-memory-layer`) for a typed, Pythonic API; see [Python SDK (cognitive-memory-layer)](#python-sdk-cognitive-memory-layer), [packages/py-cml](../packages/py-cml/), and [CreatePackage/CreatePackageStatus.md](CreatePackage/CreatePackageStatus.md) for full documentation and examples.

For questions or issues, refer to the project documentation in the `ProjectPlan` folder.
