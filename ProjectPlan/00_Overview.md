# Cognitive Memory Layer - Project Overview

## Executive Summary

A production-ready, neuro-inspired memory system for LLMs that replicates human memory architecture. This system enables AI agents to store, retrieve, consolidate, and forget information dynamically—moving beyond static context windows to true long-term memory.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REST API Layer (FastAPI)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ memory.write│  │ memory.read │  │memory.update│  │ memory.forget       │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────────┐
│                         Memory Orchestrator                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Write Gate   │  │ Read Planner │  │ Reconsolidate│  │ Policy Layer     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────────┐  ┌─────────────────────────┐  ┌───────────────────────┐
│  Working Memory   │  │   Hippocampal Store     │  │  Neocortical Store    │
│  (Short-term)     │  │   (Episodic/Fast)       │  │  (Semantic/Slow)      │
│                   │  │                         │  │                       │
│  - Sensory Buffer │  │  - Vector DB (pgvector) │  │  - Knowledge Graph    │
│  - Context Window │  │  - Dynamic KG edges     │  │    (Neo4j)            │
│  - Chunker        │  │  - Fast writes          │  │  - Structured schemas │
└───────────────────┘  └─────────────────────────┘  └───────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────────┐
│                        Background Workers                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐ │
│  │ Consolidation    │  │ Forgetting       │  │ Maintenance                │ │
│  │ ("Sleep" Cycle)  │  │ (Decay/Prune;    │  │ (Reindex, Cleanup)         │ │
│  │                  │  │  Celery task)    │  │                            │ │
│  └──────────────────┘  └──────────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────────┐
│                           Event Log (Append-Only)                            │
│                     Immutable audit trail for all operations                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Memory Types

| Type | Description | Lifecycle | Example |
|------|-------------|-----------|---------|
| `episodic_event` | What happened (full context) | Fast decay unless reinforced | "User mentioned moving to Paris on Jan 15" |
| `semantic_fact` | Durable distilled facts | Slow decay, high confidence | "User lives in Paris" |
| `preference` | User preferences (changeable) | Time-sliced on change | "User prefers vegetarian food" |
| `task_state` | Current task progress | High churn, latest wins | "Step 3 of 5 complete" |
| `procedure` | How to do something | Stable, reusable | "To book a flight: 1) search, 2) compare..." |
| `constraint` | Rules/policies | Never auto-forget | "Never share user's email" |
| `hypothesis` | Uncertain beliefs | Requires confirmation | "User might be interested in cooking" |

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| API Framework | FastAPI | Async, fast, OpenAPI docs |
| Primary DB | PostgreSQL + pgvector | ACID, vector search, production-ready |
| Graph DB | Neo4j | PPR algorithms, relationship queries |
| Cache | Redis | Hot memory cache, rate limiting |
| Queue | Redis/Celery | Background workers (e.g. forgetting task, beat) |
| Embeddings | OpenAI/Local (sentence-transformers) | Configurable |
| LLM | OpenAI/Anthropic/Local/vLLM | For extraction, summarization; vLLM (e.g. Llama 3.2 1B) for local compression |

## Phase Summary

| Phase | Name | Duration | Key Deliverables |
|-------|------|----------|------------------|
| 1 | Foundation & Core Data Models | Week 1-2 | Project structure, schemas, event log |
| 2 | Sensory Buffer & Working Memory | Week 2-3 | Short-term memory, chunking |
| 3 | Hippocampal Store | Week 3-4 | Vector store, write gate, episodic memory |
| 4 | Neocortical Store | Week 4-5 | Knowledge graph, semantic facts |
| 5 | Retrieval System | Week 5-6 | Hybrid retrieval, PPR, memory packets |
| 6 | Reconsolidation & Belief Revision | Week 6-7 | Conflict detection, updates |
| 7 | Consolidation Engine | Week 7-8 | Sleep cycle, gist extraction |
| 8 | Active Forgetting | Week 8-9 | Decay, silencing, pruning; LLM compression (vLLM), dependency check, Celery task |
| 9 | REST API & Integration | Week 9-10 | Full API, auth, multi-tenancy |
| 10 | Testing & Deployment | Week 10-12 | Tests, Docker, monitoring |

## Key Design Principles

1. **Append-Only Event Log**: All changes logged immutably for audit and replay
2. **Provenance Tracking**: Every memory has source, confidence, evidence pointers
3. **Write Gate**: Not everything gets stored—salience filtering prevents bloat
4. **Two-Lane Processing**: Sync path for low latency, async for heavy operations
5. **Belief Revision**: Memories update intelligently, not just overwrite
6. **Graceful Forgetting**: Decay → Silence → Compress → Delete (not just TTL)

## Directory Structure

```
CognitiveMemoryLayer/
├── src/
│   ├── api/                    # REST API endpoints
│   ├── core/                   # Core domain models
│   ├── memory/
│   │   ├── sensory/            # Sensory buffer
│   │   ├── working/            # Working memory
│   │   ├── hippocampal/        # Episodic store
│   │   ├── neocortical/        # Semantic store
│   │   └── orchestrator.py     # Main coordinator
│   ├── retrieval/              # Query and retrieval
│   ├── consolidation/          # Sleep cycle workers
│   ├── forgetting/             # Decay and pruning
│   ├── extraction/             # Entity/fact extraction
│   ├── storage/                # DB adapters
│   └── utils/                  # Helpers
├── tests/
├── config/
├── migrations/
├── docker/
└── docs/
```
