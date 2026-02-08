# Cognitive Memory Layer - Complete Project Plan

Merged from all ProjectPlan documents.



---

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
| `conversation` | Chat message/turn | Session-based | "Multi-turn dialogue" |
| `message` | Single message | Session-based | "Individual chat message" |
| `tool_result` | Tool execution output | Task-based | "API call returned..." |
| `reasoning_step` | Chain-of-thought step | Session-based | "Step 1: analyze..." |
| `scratch` | Temporary working memory | Fast decay | "Working notes" |
| `knowledge` | General world knowledge | Stable | "Domain facts" |
| `observation` | Agent observations | Session-based | "User seems frustrated" |
| `plan` | Agent plans/goals | Task-based | "Goal: complete booking" |

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
│   ├── dashboard/              # Web dashboard (static SPA + API routes)
│   │   └── static/             # HTML, CSS, JS (overview, memories, events, management)
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


---

# Phase 1: Foundation & Core Data Models

## Overview
**Duration**: Week 1-2  
**Goal**: Establish project structure, core data models, event logging, and storage abstractions.

## Architecture Decisions

- **Event Sourcing**: All state changes recorded in append-only log
- **CQRS Pattern**: Separate write models from read views
- **Repository Pattern**: Abstract storage backends
- **Dependency Injection**: Configurable components

---

## Task 1.1: Project Setup & Structure

### Description
Initialize the Python project with proper structure, dependencies, and configuration management.

### Subtask 1.1.1: Initialize Python Project

```pseudo
# Pseudo: Shell commands – Create project with poetry
```

### Subtask 1.1.2: Directory Structure Creation

```pseudo
# Pseudo: scripts/init_structure.py
```

### Subtask 1.1.3: Configuration Management

```pseudo
# Pseudo: src/core/config.py
```

---

## Task 1.2: Core Data Models

### Description
Define the fundamental data structures for memory records, events, and operations.

### Subtask 1.2.1: Memory Type Enums

```pseudo
# Pseudo: src/core/enums.py
```

### Subtask 1.2.2: Core Memory Record Schema

```pseudo
# Pseudo: src/core/schemas.py
```

### Subtask 1.2.3: Event Log Schema

```pseudo
# Pseudo: src/core/schemas.py (continued)
```

### Subtask 1.2.4: Memory Packet (Retrieval Response)

```pseudo
# Pseudo: src/core/schemas.py (continued)
```

---

## Task 1.3: Event Log Implementation

### Description
Implement the append-only event log for audit trail and replay capability.

### Subtask 1.3.1: SQLAlchemy Models

```pseudo
# Pseudo: src/storage/models.py
```

### Subtask 1.3.2: Event Log Repository

```pseudo
# Pseudo: src/storage/event_log.py
```

---

## Task 1.4: Storage Abstraction Layer

### Description
Create abstract interfaces for storage backends to enable swapping implementations.

### Subtask 1.4.1: Base Repository Interface

```pseudo
# Pseudo: src/storage/base.py
```

### Subtask 1.4.2: Database Connection Manager

```pseudo
# Pseudo: src/storage/connection.py
```

---

## Task 1.5: Database Migrations

### Description
Set up Alembic for database migrations.

### Subtask 1.5.1: Alembic Configuration

```pseudo
# Pseudo: migrations/env.py
```

### Subtask 1.5.2: Initial Migration Script

```pseudo
# Pseudo: migrations/versions/001_initial.py
```

---

## Deliverables Checklist

- [x] Poetry project initialized with all dependencies
- [x] Directory structure created
- [x] Configuration management with pydantic-settings
- [x] Core enums defined (MemoryType, MemoryStatus, etc.)
- [x] MemoryRecord schema with all fields
- [x] EventLog schema for audit trail
- [x] MemoryPacket schema for retrieval responses
- [x] SQLAlchemy models for PostgreSQL
- [x] EventLogRepository implementation
- [x] Abstract base classes for storage backends
- [x] Database connection manager
- [x] Alembic migrations configured
- [x] Initial migration with all tables and indexes


---

# Phase 2: Sensory Buffer & Working Memory

## Overview
**Duration**: Week 2-3  
**Goal**: Implement short-term memory systems that process raw input into semantically meaningful chunks before long-term encoding.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                      Incoming Turn                               │
│   {role: "user", content: "I just moved to Paris..."}           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Sensory Buffer                               │
│   - High fidelity storage of raw tokens                         │
│   - Decay after ~30 seconds                                      │
│   - Capacity: ~500 tokens                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Working Memory                               │
│   - Semantic chunking (sentences → facts)                        │
│   - Limited capacity (~10 chunks)                                │
│   - Actively manipulated for reasoning                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [To Hippocampal Store]
```

---

## Task 2.1: Sensory Buffer Implementation

### Description
A high-fidelity, short-lived buffer that temporarily holds raw input with timestamps.

### Subtask 2.1.1: Token-Level Buffer with Decay

```pseudo
# Pseudo: src/memory/sensory/buffer.py
```

### Subtask 2.1.2: Per-User Sensory Buffer Manager

```pseudo
# Pseudo: src/memory/sensory/manager.py
```

---

## Task 2.2: Working Memory with Semantic Chunking

### Description
Process sensory buffer contents into semantically meaningful chunks that can be encoded into long-term memory.

### Subtask 2.2.1: Chunk Data Structures

```pseudo
# Pseudo: src/memory/working/models.py
```

### Subtask 2.2.2: Semantic Chunker (LLM-based)

```pseudo
# Pseudo: src/memory/working/chunker.py
```

### Subtask 2.2.3: Working Memory Manager

```pseudo
# Pseudo: src/memory/working/manager.py
```

---

## Task 2.3: Integration with Memory Pipeline

### Description
Connect sensory buffer and working memory to the main memory orchestration flow.

### Subtask 2.3.1: Short-Term Memory Facade

```pseudo
# Pseudo: src/memory/short_term.py
```

---

## Task 2.4: LLM Utility Module

### Description
Implement the LLM client used by the chunker and other components.

### Subtask 2.4.1: LLM Client Abstraction

```pseudo
# Pseudo: src/utils/llm.py
```

---

## Deliverables Checklist

- [x] SensoryBuffer class with token-level storage and decay
- [x] SensoryBufferManager for per-user buffer management
- [x] SemanticChunk and WorkingMemoryState data models
- [x] SemanticChunker (LLM-based) for intelligent chunking
- [x] RuleBasedChunker for fast fallback
- [x] WorkingMemoryManager with capacity limits
- [x] ShortTermMemory facade unifying both components
- [x] LLMClient abstraction with OpenAI implementation
- [x] Unit tests for buffer operations
- [x] Unit tests for chunking logic
- [x] Integration test: full ingest flow


---

# Phase 3: Hippocampal Store (Episodic Memory)

## Overview
**Duration**: Week 3-4  
**Goal**: Implement the fast-write episodic memory store with vector search, write gate, and entity/relation extraction.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                  From Working Memory                             │
│           (SemanticChunks with salience > threshold)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Write Gate                                  │
│   - Salience check (importance, novelty, stability)             │
│   - Risk assessment (PII, secrets)                               │
│   - Deduplication check                                          │
│   Decision: STORE / SKIP / ASYNC_STORE                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌───────────────────┐  ┌───────────────────┐
        │  Sync Write Path  │  │ Async Write Path  │
        │  (Fast, minimal)  │  │ (Full extraction) │
        └───────────────────┘  └───────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Hippocampal Store                               │
│   ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│   │   Vector Index      │  │   Dynamic KG Edges              │  │
│   │   (pgvector/HNSW)   │  │   (Entity associations)         │  │
│   └─────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 3.1: Write Gate Implementation

### Description
The write gate decides whether and how to store incoming information, preventing memory bloat.

### Subtask 3.1.1: Write Gate Decision Model

```pseudo
# Pseudo: src/memory/hippocampal/write_gate.py
```

### Subtask 3.1.2: PII Redaction Module

```pseudo
# Pseudo: src/memory/hippocampal/redactor.py
```

---

## Task 3.2: Embedding Service

### Description
Generate embeddings for memory content using configurable models.

### Subtask 3.2.1: Embedding Client

```pseudo
# Pseudo: src/utils/embeddings.py
```

---

## Task 3.3: Entity and Relation Extraction

### Description
Extract structured entities and relations from text for knowledge graph building.

### Subtask 3.3.1: Entity Extractor

```pseudo
# Pseudo: src/extraction/entity_extractor.py
```

### Subtask 3.3.2: Relation Extractor (OpenIE-style)

```pseudo
# Pseudo: src/extraction/relation_extractor.py
```

---

## Task 3.4: Hippocampal Store Implementation

### Description
The main episodic memory store with vector search and fast writes.

### Subtask 3.4.1: PostgreSQL Vector Store

```pseudo
# Pseudo: src/storage/postgres.py
```

### Subtask 3.4.2: Hippocampal Store Facade

```pseudo
# Pseudo: src/memory/hippocampal/store.py
```

---

## Deliverables Checklist

- [x] WriteGate with salience, novelty, and risk evaluation
- [x] WriteGateResult and WriteDecision models
- [x] PIIRedactor for sensitive data handling
- [x] EmbeddingClient abstraction (OpenAI + Local)
- [x] CachedEmbeddings with Redis caching
- [x] EntityExtractor (LLM-based and spaCy fallback)
- [x] RelationExtractor for OpenIE-style triples
- [x] PostgresMemoryStore with pgvector integration
- [x] HippocampalStore facade coordinating all components
- [x] Unit tests for write gate logic
- [x] Unit tests for extraction
- [x] Integration tests for full encode flow


---

# Phase 4: Neocortical Store (Semantic Memory)

## Overview
**Duration**: Week 4-5  
**Goal**: Implement the structured semantic memory store with knowledge graph, schema management, and fact consolidation.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                  From Hippocampal Store                          │
│       (Episodic memories with entities and relations)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Consolidation Bridge                           │
│   - Pattern detection across episodes                            │
│   - Gist extraction                                              │
│   - Schema alignment                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Neocortical Store                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Knowledge Graph (Neo4j)                     │   │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐            │   │
│   │   │  User   │───▶│Location │◀───│  Event  │            │   │
│   │   │  Node   │    │  Node   │    │  Node   │            │   │
│   │   └─────────┘    └─────────┘    └─────────┘            │   │
│   │        │              │              │                  │   │
│   │        ▼              ▼              ▼                  │   │
│   │   [prefers]      [located_in]   [occurred_at]          │   │
│   └─────────────────────────────────────────────────────────┘   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Semantic Fact Store                         │   │
│   │   - Structured key-value facts                           │   │
│   │   - Schema-aligned properties                            │   │
│   │   - Version history                                      │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 4.1: Neo4j Graph Store Implementation

### Description
Implement the knowledge graph backend using Neo4j for entity relationships and graph queries.

### Subtask 4.1.1: Neo4j Connection and Base Operations

```pseudo
# Pseudo: src/storage/neo4j.py
```

### Subtask 4.1.2: Graph Store Initialization and Schema

```pseudo
# Pseudo: src/storage/neo4j_init.py
```

---

## Task 4.2: Semantic Fact Management

### Description
Implement structured fact storage with schema alignment and versioning.

### Subtask 4.2.1: Semantic Fact Schema

```pseudo
# Pseudo: src/memory/neocortical/schemas.py
```

### Subtask 4.2.2: Fact Store Implementation

```pseudo
# Pseudo: src/memory/neocortical/fact_store.py
```

---

## Task 4.3: Neocortical Store Facade

### Description
Create unified interface combining graph store and fact store.

### Subtask 4.3.1: Neocortical Store Implementation

```pseudo
# Pseudo: src/memory/neocortical/store.py
```

---

## Task 4.4: Database Migration for Semantic Facts

### Subtask 4.4.1: Semantic Facts Table Migration

```pseudo
# Pseudo: migrations/versions/002_semantic_facts.py
```

---

## Deliverables Checklist

- [x] Neo4jGraphStore with all CRUD operations
- [x] Personalized PageRank for multi-hop reasoning
- [x] Graph schema initialization script
- [x] SemanticFact and FactSchema models
- [x] SemanticFactStore with versioning
- [x] Temporal fact handling (valid_from/to)
- [x] NeocorticalStore facade
- [x] Graph-fact synchronization
- [x] User profile generation
- [x] Multi-hop query support
- [x] Database migration for semantic_facts table
- [x] Unit tests for graph operations
- [x] Unit tests for fact store
- [x] Integration tests for neocortical store


---

# Phase 5: Retrieval System

## Overview
**Duration**: Week 5-6  
**Goal**: Implement hybrid retrieval with query classification, multi-source search, and memory packet construction.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                      Incoming Query                              │
│              "What's my favorite cuisine?"                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Query Classifier                             │
│   Intent: preference_lookup                                      │
│   Entities: ["cuisine", "favorite"]                              │
│   Time scope: current                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Retrieval Planner                            │
│   Plan: [                                                        │
│     { source: "semantic_facts", key: "user:preference:cuisine" } │
│     { source: "vector", query: "favorite cuisine", top_k: 5 }   │
│   ]                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   KV Lookup   │ │ Vector Search │ │ Graph PPR     │
    │   (Fast)      │ │ (Medium)      │ │ (Multi-hop)   │
    └───────────────┘ └───────────────┘ └───────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Reranker                                   │
│   - Deduplicate                                                  │
│   - Score by relevance + recency + confidence                   │
│   - Apply diversity                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Packet Builder                         │
│   - Categorize results                                           │
│   - Format for LLM consumption                                   │
│   - Include provenance                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 5.1: Query Classification

### Description
Classify incoming queries to determine optimal retrieval strategy.

### Subtask 5.1.1: Query Intent Types

```pseudo
# Pseudo: src/retrieval/query_types.py
```

### Subtask 5.1.2: Query Classifier Implementation

```pseudo
# Pseudo: src/retrieval/classifier.py
```

---

## Task 5.2: Retrieval Planner

### Description
Generate retrieval plans based on query analysis.

### Subtask 5.2.1: Retrieval Plan Model

```pseudo
# Pseudo: src/retrieval/planner.py
```

---

## Task 5.3: Hybrid Retriever

### Description
Execute retrieval plans across multiple sources.

### Subtask 5.3.1: Retriever Implementation

```pseudo
# Pseudo: src/retrieval/retriever.py
```

---

## Task 5.4: Reranker

### Description
Rerank and diversify retrieval results.

### Subtask 5.4.1: Reranker Implementation

```pseudo
# Pseudo: src/retrieval/reranker.py
```

---

## Task 5.5: Memory Packet Builder

### Description
Build structured memory packets for LLM consumption.

### Subtask 5.5.1: Packet Builder Implementation

```pseudo
# Pseudo: src/retrieval/packet_builder.py
```

---

## Task 5.6: Main Retriever Facade

### Subtask 5.6.1: Unified Retrieval Interface

```pseudo
# Pseudo: src/retrieval/memory_retriever.py
```

---

## Deliverables Checklist

- [x] QueryIntent enum and QueryAnalysis model
- [x] QueryClassifier with fast patterns and LLM fallback
- [x] RetrievalPlan and RetrievalStep models
- [x] RetrievalPlanner generating plans from analysis
- [x] HybridRetriever executing plans
- [x] Individual source retrievers (facts, vector, graph)
- [x] MemoryReranker with MMR diversity
- [x] MemoryPacketBuilder with multiple formats
- [x] MemoryRetriever facade
- [x] Unit tests for classification
- [x] Unit tests for planning
- [x] Integration tests for full retrieval flow


---

# Phase 6: Reconsolidation & Belief Revision

## Overview
**Duration**: Week 6-7  
**Goal**: Implement memory updating after retrieval, conflict detection, and belief revision algorithms.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Retrieved                              │
│            (From retrieval system)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Labile State Manager                            │
│   - Mark retrieved memories as "labile" (unstable)               │
│   - Track which memories were used in this turn                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  New Information Extractor                       │
│   - Extract facts from user message + assistant response         │
│   - Compare against retrieved memories                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Conflict Detector                               │
│   - Identify contradictions                                      │
│   - Classify conflict type (temporal change, correction, etc.)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
    ┌───────────────────┐      ┌───────────────────┐
    │   Consistent      │      │   Contradictory   │
    │   → Reinforce     │      │   → Revise        │
    └───────────────────┘      └───────────────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Belief Revision Engine                          │
│   - Apply revision strategy (time-slice, update, invalidate)    │
│   - Update confidence scores                                     │
│   - Maintain audit trail                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 6.1: Labile State Management

### Description
Track memories that are in a "labile" (unstable) state after retrieval.

### Subtask 6.1.1: Labile State Tracker

```pseudo
# Pseudo: src/reconsolidation/labile_tracker.py
```

---

## Task 6.2: Conflict Detection

### Description
Detect contradictions between new information and existing memories.

### Subtask 6.2.1: Conflict Types and Detector

```pseudo
# Pseudo: src/reconsolidation/conflict_detector.py
```

---

## Task 6.3: Belief Revision Engine

### Description
Apply belief revision strategies based on conflict type.

### Subtask 6.3.1: Revision Strategies

```pseudo
# Pseudo: src/reconsolidation/belief_revision.py
```

---

## Task 6.4: Reconsolidation Orchestrator

### Description
Coordinate the full reconsolidation process.

### Subtask 6.4.1: Reconsolidation Service

```pseudo
# Pseudo: src/reconsolidation/service.py
```

---

## Deliverables Checklist

- [x] LabileMemory and LabileSession models
- [x] LabileStateTracker with session management
- [x] ConflictType enum and ConflictResult model
- [x] ConflictDetector with fast heuristics and LLM fallback
- [x] RevisionStrategy enum and RevisionPlan model
- [x] BeliefRevisionEngine with all strategies
- [x] ReconsolidationService orchestrating the flow
- [x] Unit tests for conflict detection
- [x] Unit tests for revision planning
- [x] Integration tests for full reconsolidation


---

# Phase 7: Consolidation Engine ("Sleep" Cycle)

## Overview

**Duration**: Week 7-8
**Goal**: Implement offline consolidation that transfers knowledge from episodic to semantic memory, extracts patterns, and compresses episodes.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    Consolidation Trigger                         │
│   - Scheduled (every N hours)                                    │
│   - Quota-based (episodic store > threshold)                    │
│   - Event-based (task completed, contradiction resolved)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Episode Sampler                               │
│   - Select recent high-importance episodes                       │
│   - Prioritize by access count, confidence, recency             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pattern Clusterer                             │
│   - Cluster episodes by semantic similarity                      │
│   - Identify recurring themes/topics                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gist Extractor                                │
│   - Summarize clusters into semantic facts                       │
│   - Extract generalizable knowledge                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Schema Aligner                                │
│   - Match gists to existing semantic schemas                     │
│   - Rapid integration if schema exists                           │
│   - Create new schema if novel pattern                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Migrator                                      │
│   - Write to neocortical (semantic) store                       │
│   - Mark episodes as consolidated                                │
│   - Optionally compress or archive episodes                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 7.1: Consolidation Triggers and Scheduler

### Description

Manage when consolidation runs based on various triggers.

### Subtask 7.1.1: Consolidation Trigger System

```pseudo
# Pseudo: src/consolidation/triggers.py
```

---

## Task 7.2: Episode Sampling and Clustering

### Description

Select and cluster episodes for consolidation.

### Subtask 7.2.1: Episode Sampler

```pseudo
# Pseudo: src/consolidation/sampler.py
```

### Subtask 7.2.2: Semantic Clusterer

```pseudo
# Pseudo: src/consolidation/clusterer.py
```

---

## Task 7.3: Gist Extraction and Summarization

### Description

Extract semantic gist from episode clusters.

### Subtask 7.3.1: Gist Extractor

```pseudo
# Pseudo: src/consolidation/summarizer.py
```

---

## Task 7.4: Schema Alignment and Migration

### Description

Align gists with existing schemas and migrate to semantic store.

### Subtask 7.4.1: Schema Aligner

```pseudo
# Pseudo: src/consolidation/schema_aligner.py
```

### Subtask 7.4.2: Consolidation Migrator

```pseudo
# Pseudo: src/consolidation/migrator.py
```

---

## Task 7.5: Main Consolidation Worker

### Subtask 7.5.1: Consolidation Worker Service

```pseudo
# Pseudo: src/consolidation/worker.py
```

---

## Deliverables Checklist

**Status:** ✅ Complete

| #  | Deliverable                                        | Status |
| -- | -------------------------------------------------- | ------ |
| 1  | TriggerCondition and ConsolidationTask models      | ✅     |
| 2  | ConsolidationScheduler with multiple trigger types | ✅     |
| 3  | EpisodeSampler with priority scoring               | ✅     |
| 4  | SemanticClusterer using embeddings                 | ✅     |
| 5  | GistExtractor with LLM summarization               | ✅     |
| 6  | SchemaAligner for rapid integration                | ✅     |
| 7  | ConsolidationMigrator for semantic store           | ✅     |
| 8  | ConsolidationWorker orchestrating full flow        | ✅     |
| 9  | ConsolidationReport for audit                      | ✅     |
| 10 | Background worker with task queue                  | ✅     |
| 11 | Unit tests for clustering and triggers             | ✅     |
| 12 | Integration tests for full consolidation           | ✅     |


---

# Phase 8: Active Forgetting

## Overview
**Duration**: Week 8-9  
**Goal**: Implement intelligent forgetting mechanisms including decay, silencing, compression, and pruning.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    Forgetting Triggers                           │
│   - Scheduled (daily/weekly)                                     │
│   - Quota exceeded (storage limit)                               │
│   - Performance degradation (retrieval latency)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Relevance Scorer                              │
│   - Importance score                                             │
│   - Recency score                                                │
│   - Usage frequency score                                        │
│   - Confidence score                                             │
│   - Combined weighted score                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Forgetting Policy Engine                      │
│   Score > 0.7  → Keep as-is                                     │
│   Score 0.4-0.7 → Decay confidence                              │
│   Score 0.2-0.4 → Silence (hard to retrieve)                    │
│   Score 0.1-0.2 → Compress (keep gist only)                     │
│   Score < 0.1  → Delete (if no dependencies)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌───────────────┐                  ┌───────────────┐
    │  Soft Actions │                  │  Hard Actions │
    │  - Decay      │                  │  - Compress   │
    │  - Silence    │                  │  - Archive    │
    └───────────────┘                  │  - Delete     │
                                       └───────────────┘
```

---

## Task 8.1: Relevance Scoring

### Description
Calculate relevance scores for memories to determine forgetting priority.

### Subtask 8.1.1: Relevance Score Calculator

```pseudo
# Pseudo: src/forgetting/scorer.py
```

---

## Task 8.2: Forgetting Policy Engine

### Description
Apply forgetting policies based on relevance scores.

### Subtask 8.2.1: Forgetting Actions

```pseudo
# Pseudo: src/forgetting/actions.py
```

### Subtask 8.2.2: Policy Executor

```pseudo
# Pseudo: src/forgetting/executor.py
```

---

## Task 8.3: Interference Management

### Description
Handle cases where new memories interfere with old ones.

### Subtask 8.3.1: Interference Detector

```pseudo
# Pseudo: src/forgetting/interference.py
```

---

## Task 8.4: Forgetting Worker

### Description
Main forgetting service that orchestrates the process.

### Subtask 8.4.1: Forgetting Worker Service

```pseudo
# Pseudo: src/forgetting/worker.py
```

---

## Task 8.5: LLM-Based Compression (Optional)

### Description
Use an LLM to summarize long memory text when compressing (instead of truncation). Supports vLLM with Llama 3.2-1B in Docker for local inference.

### Implementation

- **`src/forgetting/compression.py`**: `summarize_for_compression(text, max_chars, llm_client)` — when `llm_client` is provided and text exceeds `max_chars`, calls LLM for one-sentence gist; otherwise truncates.
- **`src/utils/llm.py`**: `VLLMClient` — OpenAI-compatible client for vLLM (e.g. `http://vllm:8000/v1`). Config: `LLM__PROVIDER=vllm`, `LLM__VLLM_BASE_URL`, `LLM__VLLM_MODEL`.
- **`ForgettingExecutor`**: Accepts `compression_llm_client` and `compression_max_chars`; in `_execute_compress` uses `summarize_for_compression` when client is set.
- **Docker**: Optional service `vllm` (profile `vllm`) in `docker/docker-compose.yml` — image `vllm/vllm-openai`, model e.g. `unslop/Llama-3.2-1B-Instruct`. Run with `docker compose --profile vllm up`.

---

## Task 8.6: Dependency Check Before Delete

### Description
Block soft-delete of a memory when other memories reference it (via `supersedes_id` or `metadata.evidence_refs`).

### Implementation

- **`PostgresMemoryStore.count_references_to(record_id)`**: Returns the number of other records in the same tenant/user that reference this ID (`supersedes_id` or `evidence_refs`). Used before allowing delete.
- **`ForgettingExecutor._execute_delete`**: Calls `count_references_to`; if count > 0, skips delete and appends a clear error to `ForgettingResult.errors` (e.g. "Skipped delete &lt;id&gt;: N dependency(ies)").

---

## Task 8.7: Celery / Background Task

### Description
Run active forgetting as a Celery task so it can be triggered from the API or on a schedule via Celery Beat.

### Implementation

- **`src/celery_app.py`**: Celery app with broker/backend from `DATABASE__REDIS_URL`. Task `run_forgetting_task(tenant_id, user_id, dry_run=False, max_memories=5000)` runs `ForgettingWorker.run_forgetting` via `asyncio.run` and returns a JSON-serializable report dict.
- **Beat schedule**: `forgetting-daily` — runs `run_forgetting_task` every 24 hours (86400 s). In production, call the task with specific tenant/user or iterate over registered users.
- **Queue**: Task routed to queue `forgetting`. Run worker with: `celery -A src.celery_app worker -Q forgetting` and beat with: `celery -A src.celery_app beat`.

---

## Deliverables Checklist

- [x] RelevanceWeights and ScorerConfig models
- [x] RelevanceScorer with multi-factor scoring
- [x] ForgettingAction enum and operation models
- [x] ForgettingPolicyEngine with action thresholds
- [x] ForgettingExecutor for all action types
- [x] InterferenceDetector for duplicates
- [x] ForgettingWorker orchestrating the flow
- [x] ForgettingScheduler for background runs
- [x] ForgettingReport for audit
- [x] Unit tests for scoring
- [x] Unit tests for policy decisions
- [x] Integration tests for full forgetting flow
- [x] LLM-based compression (summarize_for_compression, VLLMClient, vLLM in Docker)
- [x] Dependency check before delete (count_references_to, skip delete with error)
- [x] Celery task and beat schedule for forgetting


---

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

```pseudo
# Pseudo: src/api/app.py
```

### Subtask 9.1.2: Middleware Implementation

```pseudo
# Pseudo: src/api/middleware.py
```

---

## Task 9.2: Authentication and Authorization

### Description
Implement API key authentication and tenant isolation.

### Subtask 9.2.1: Auth Dependencies

```pseudo
# Pseudo: src/api/auth.py
```

---

## Task 9.3: API Routes

### Description
Implement the main API endpoints.

### Subtask 9.3.1: Request/Response Models

```pseudo
# Pseudo: src/api/schemas.py
```

### Subtask 9.3.2: Route Implementations

```pseudo
# Pseudo: src/api/routes.py
```

---

## Task 9.4: Memory Orchestrator

### Description
Main orchestrator coordinating all memory operations.

### Subtask 9.4.1: Orchestrator Implementation

```pseudo
# Pseudo: src/memory/orchestrator.py
```

---

## Task 9.5: Admin Routes

### Subtask 9.5.1: Admin Endpoints

```pseudo
# Pseudo: src/api/admin_routes.py
```

---

## Deliverables Checklist

- [x] FastAPI application factory with lifespan
- [x] RequestLoggingMiddleware with timing
- [x] RateLimitMiddleware with per-tenant limits
- [x] Config-based API key validation (AUTH__API_KEY, AUTH__ADMIN_API_KEY)
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


---

# Phase 10: Testing & Deployment

## Overview
**Duration**: Week 10-12  
**Goal**: Comprehensive testing, Docker containerization, CI/CD pipeline, and production deployment.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    Development                                   │
│   Local dev → Unit tests → Integration tests → E2E tests        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                                │
│   GitHub Actions / GitLab CI                                     │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│   │  Lint   │→│  Test   │→│  Build  │→│  Deploy Staging │    │
│   └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Production Environment                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Kubernetes Cluster                     │   │
│   │   ┌───────────┐  ┌───────────┐  ┌───────────────────┐  │   │
│   │   │ API Pods  │  │ Worker    │  │ Background Jobs   │  │   │
│   │   │ (x3)      │  │ Pods (x2) │  │ (Consolidation)   │  │   │
│   │   └───────────┘  └───────────┘  └───────────────────┘  │   │
│   └─────────────────────────────────────────────────────────┘   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Managed Services                       │   │
│   │   PostgreSQL    Neo4j AuraDB    Redis    Monitoring     │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 10.1: Unit Testing

### Description
Comprehensive unit tests for all components.

### Subtask 10.1.1: Test Configuration

```pseudo
# Pseudo: tests/conftest.py
```

### Subtask 10.1.2: Write Gate Tests

```pseudo
# Pseudo: tests/unit/test_write_gate.py
```

### Subtask 10.1.3: Relevance Scorer Tests

```pseudo
# Pseudo: tests/unit/test_relevance_scorer.py
```

### Subtask 10.1.4: Conflict Detection Tests

```pseudo
# Pseudo: tests/unit/test_conflict_detector.py
```

---

## Task 10.2: Integration Testing

### Description
Test component interactions and database operations.

### Subtask 10.2.1: Integration Test Setup

```pseudo
# Pseudo: tests/integration/conftest.py
```

### Subtask 10.2.2: Storage Integration Tests

```pseudo
# Pseudo: tests/integration/test_storage.py
```

---

## Task 10.3: End-to-End Testing

### Description
Test complete user flows through the API.

### Subtask 10.3.1: API E2E Tests

```pseudo
# Pseudo: tests/e2e/test_api_flows.py
```

---

## Task 10.4: Docker Configuration

### Description
Containerize the application and services.

### Subtask 10.4.1: Dockerfile

```pseudo
# Pseudo: Dockerfile – multi-stage build, deps, CMD
```

### Subtask 10.4.2: Docker Compose

```pseudo
# Pseudo: YAML config – docker/docker-compose.yml
```

---

## Task 10.5: CI/CD Pipeline

### Subtask 10.5.1: GitHub Actions Workflow

```pseudo
# Pseudo: YAML config – .github/workflows/ci.yml
```

---

## Task 10.6: Monitoring and Observability

### Subtask 10.6.1: Structured Logging

```pseudo
# Pseudo: src/utils/logging.py
```

### Subtask 10.6.2: Metrics Collection

```pseudo
# Pseudo: src/utils/metrics.py
```

---

## Deliverables Checklist

- [x] pytest configuration with fixtures
- [x] Unit tests for WriteGate
- [x] Unit tests for RelevanceScorer
- [x] Unit tests for ConflictDetector
- [x] Integration test setup with testcontainers
- [x] Storage integration tests
- [x] API E2E tests for full lifecycle
- [x] Dockerfile with multi-stage build
- [x] docker-compose.yml with all services
- [x] GitHub Actions CI workflow
- [x] Linting and type checking in CI
- [x] Test coverage reporting
- [x] Docker image build and push
- [x] Staging deployment job
- [x] Structured logging configuration
- [x] Prometheus metrics
- [x] Health check endpoints
- [x] Documentation (README, API docs)


---

# Phase 11: Holistic Memory Refactoring

**Status:** Completed

## Goal

Remove `MemoryScope` enum and scope-based partitioning. Replace with a unified memory store using contextual tags.

## Summary of Changes

- **Core**: Removed `MemoryScope`; added `MemoryContext` enum for tagging (non-exclusive). Updated `MemoryRecord` / `MemoryRecordCreate` to use `context_tags` and `source_session_id` instead of `scope` / `scope_id` / `user_id`.
- **Database**: Migration `004_remove_scopes_holistic` adds `context_tags` (TEXT[]) and `source_session_id`, drops scope columns. GIN index on `context_tags`.
- **Storage**: `PostgresMemoryStore` and neocortical/hippocampal stores use `tenant_id` only; optional `context_filter` for retrieval.
- **Orchestrator**: All methods use `tenant_id`, `context_tags`, `session_id`; no scope parameters.
- **Retrieval**: `MemoryRetriever` and `HybridRetriever` search across tenant memories; optional `context_filter`.
- **API**: Request/response schemas updated; stats endpoint is `GET /memory/stats` (tenant from auth).

## Architecture

Memory access is **holistic per tenant**: one logical store per tenant. Optional `context_tags` and `source_session_id` support categorization and origin tracking without partitioning.


---

# Phase 12: Seamless Memory Integration

**Status:** Completed

## Goal

Make memory retrieval automatic and unconscious, like human association.

## Summary of Changes

- **SeamlessMemoryProvider** (`src/memory/seamless_provider.py`): Processes a conversation turn by (1) auto-retrieving relevant context for the user message, (2) optionally storing user/assistant content, (3) running reconsolidation when an assistant response is provided. Returns `SeamlessTurnResult` with `memory_context` ready for LLM injection.
- **QueryClassifier**: Added `recent_context` to `classify()` for context-aware classification of vague queries.
- **MemoryRetriever**: Passes `recent_context` to the classifier when provided.
- **API**: New `POST /memory/turn` endpoint and `ProcessTurnRequest` / `ProcessTurnResponse` schemas.
- **Tools**: Simplified tool descriptions for `memory_write` and `memory_read` (no scope parameters).

## Usage

Use `POST /memory/turn` with `user_message` (and optionally `assistant_response`, `session_id`) to get `memory_context` for the current turn and optional auto-store.


---

# Phase 13: Code Improvements

**Status:** Completed

## Goal

Address identified issues: embeddings on update, working memory eviction, salience, reconsolidation behavior.

## Summary of Changes

- **Embedding updates (13.1)**: When `orchestrator.update()` is called with changed `text`, the orchestrator now re-embeds and re-extracts entities and passes the updated embedding/entities into the store.
- **Working memory eviction (13.2)**: `WorkingMemoryState.add_chunk()` uses recency-aware eviction: keep the most recent N chunks regardless of salience, then evict from older chunks by salience.
- **Sentiment-aware salience (13.3)**: Added `_compute_salience_boost_for_sentiment()` in the chunker; integrated into `RuleBasedChunker` and `SemanticChunker` so emotionally significant content gets a salience boost (capped at 0.3).
- **Reconsolidation archive (13.4)**: Belief revision now archives instead of deleting on contradiction/correction: `_plan_time_slice` and `_plan_correction` set `valid_to` and `status=ARCHIVED` on the old record instead of deleting.


---

# Phase 14: Documentation and Examples Update

**Status:** Completed

## Goal

Update all documentation and examples to reflect the holistic, seamless API.

## Summary of Changes

- **README.md**: Quick start uses holistic write/read and documents `POST /memory/turn`; API table updated; added "Seamless Memory" subsection.
- **UsageDocumentation.md**: Quick start and API reference rewritten without scopes; added "Holistic memory and context tags" and "Seamless Memory"; tool definitions and examples updated; `GET /memory/stats` documented.
- **Examples**: `memory_client.py` updated to holistic API and `process_turn()`; `basic_usage.py`, `chatbot_with_memory.py`, `openai_tool_calling.py`, `anthropic_tool_calling.py`, `langchain_integration.py` updated to remove scope/scope_id and use the new client; examples README updated.


---

# Cognitive Memory Layer – Project Status

**Last updated:** 2026-02-08

This document tracks what has been implemented against the plan in the `ProjectPlan` folder. It includes both the original RAG-based phases (1–14) and the new **Intrinsic Memory System** phases (I1–I10), which are planned but not yet implemented.

```pseudo
# Pseudo: Implementation as per task description.
```

---

## Phase 1: Foundation & Core Data Models ✅

**Status:** Implemented  
**Plan reference:** `Phase1_Foundation.md`

### Task 1.1: Project Setup & Structure ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Poetry project initialized | ✅ | `pyproject.toml` – Python ^3.11, FastAPI, SQLAlchemy, pgvector, Neo4j, Redis, Celery, OpenAI, sentence-transformers, tiktoken, pydantic-settings, structlog; dev: pytest, black, ruff, mypy, httpx, factory-boy, faker, alembic |
| Directory structure created | ✅ | `scripts/init_structure.py` run; `src/`, `tests/`, `config/`, `migrations/`, `docker/`, `docs/` |
| Configuration management | ✅ | `src/core/config.py` – `Settings`, `DatabaseSettings`, `EmbeddingSettings`, `LLMSettings`, `MemorySettings`, `get_settings()` with env file and nested delimiter |

### Task 1.2: Core Data Models ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Core enums | ✅ | `src/core/enums.py` – `MemoryType`, `MemoryStatus`, `MemorySource`, `OperationType` |
| MemoryRecord schema | ✅ | `src/core/schemas.py` – `MemoryRecord` with all fields (identity, type, text, embedding, entities, relations, metadata, temporal, scoring, usage, status, provenance, versioning, content_hash) |
| EventLog schema | ✅ | `src/core/schemas.py` – `EventLog`, `MemoryOperation` |
| MemoryPacket schema | ✅ | `src/core/schemas.py` – `RetrievedMemory`, `MemoryPacket` with `to_context_string()` |
| Supporting schemas | ✅ | `Provenance`, `EntityMention`, `Relation`, `MemoryRecordCreate` in `src/core/schemas.py` |

### Task 1.3: Event Log Implementation ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| SQLAlchemy models for PostgreSQL | ✅ | `src/storage/models.py` – `Base`, `EventLogModel`, `MemoryRecordModel` (pgvector `Vector(1536)`, indexes) |
| EventLogRepository | ✅ | `src/storage/event_log.py` – `append`, `get_by_id`, `get_user_events`, `replay_events` (async generator) |

### Task 1.4: Storage Abstraction Layer ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Abstract base classes | ✅ | `src/storage/base.py` – `MemoryStoreBase` (upsert, get_by_id, get_by_key, delete, update, vector_search, scan, count), `GraphStoreBase` (merge_node, merge_edge, get_neighbors, personalized_pagerank) |
| Database connection manager | ✅ | `src/storage/connection.py` – `DatabaseManager` singleton, `pg_session`, `neo4j_session`, `close()` |
| Redis client helper | ✅ | `src/storage/redis.py` – `get_redis_client()` from settings (async client for cache/embedding) |

### Task 1.5: Database Migrations ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Alembic configured | ✅ | `alembic.ini`, `migrations/env.py` (async migrations, settings-based URL) |
| Initial migration | ✅ | `migrations/versions/001_initial_schema.py` – event_log, memory_records, pgvector extension, HNSW index |

### Phase 1 Deliverables Checklist (from plan)

- [x] Poetry project initialized with all dependencies  
- [x] Directory structure created  
- [x] Configuration management with pydantic-settings  
- [x] Core enums defined (MemoryType, MemoryStatus, etc.)  
- [x] MemoryRecord schema with all fields  
- [x] EventLog schema for audit trail  
- [x] MemoryPacket schema for retrieval responses  
- [x] SQLAlchemy models for PostgreSQL  
- [x] EventLogRepository implementation  
- [x] Abstract base classes for storage backends  
- [x] Database connection manager  
- [x] Alembic migrations configured  
- [x] Initial migration with all tables and indexes  

---

## Phase 2: Sensory Buffer & Working Memory ✅

**Status:** Implemented  
**Plan reference:** `Phase2_SensoryWorkingMemory.md`

### Task 2.1: Sensory Buffer ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| SensoryBuffer with token-level storage and decay | ✅ | `src/memory/sensory/buffer.py` – BufferedToken, SensoryBufferConfig, ingest, get_recent, get_text, clear, capacity/decay, optional cleanup loop |
| SensoryBufferManager per-user | ✅ | `src/memory/sensory/manager.py` – get_buffer, ingest, get_recent_text, clear_user, cleanup_inactive |

### Task 2.2: Working Memory with Semantic Chunking ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| ChunkType, SemanticChunk, WorkingMemoryState | ✅ | `src/memory/working/models.py` |
| SemanticChunker (LLM-based) | ✅ | `src/memory/working/chunker.py` |
| RuleBasedChunker | ✅ | `src/memory/working/chunker.py` – sentence split, preference/fact/instruction/question markers |
| WorkingMemoryManager | ✅ | `src/memory/working/manager.py` – get_state, process_input, get_chunks_for_encoding, get_current_context, clear_user, get_stats |

### Task 2.3: Integration ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| ShortTermMemory facade | ✅ | `src/memory/short_term.py` – ingest_turn, get_immediate_context, get_encodable_chunks, clear |

### Task 2.4: LLM Utility ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| LLMClient abstraction + OpenAIClient | ✅ | `src/utils/llm.py` – complete, complete_json; MockLLMClient for tests |
| Optional api_key in config | ✅ | `src/core/config.py` – LLMSettings.api_key |

### Phase 2 Tests

- **Unit:** `tests/unit/test_phase2_sensory_working.py` – buffer, manager, RuleBasedChunker, WorkingMemoryState, ShortTermMemory
- **Integration:** `tests/integration/test_phase2_short_term_flow.py` – full ingest → encodable chunks → clear

**Phase 2 deliverables (from plan):** All checklist items completed.

---

## Phase 3: Hippocampal Store ✅

**Status:** Implemented  
**Plan reference:** `Phase3_HippocampalStore.md`

### Task 3.1: Write Gate Implementation ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| WriteDecision, WriteGateResult, WriteGateConfig | ✅ | `src/memory/hippocampal/write_gate.py` |
| WriteGate (evaluate: salience, novelty, risk, PII/secrets) | ✅ | `src/memory/hippocampal/write_gate.py` |
| PIIRedactor | ✅ | `src/memory/hippocampal/redactor.py` – patterns for SSN, email, phone, etc.; RedactionResult |

### Task 3.2: Embedding Service ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| EmbeddingClient ABC, EmbeddingResult | ✅ | `src/utils/embeddings.py` |
| OpenAIEmbeddings | ✅ | `src/utils/embeddings.py` |
| LocalEmbeddings (sentence-transformers) | ✅ | `src/utils/embeddings.py` |
| CachedEmbeddings (Redis) | ✅ | `src/utils/embeddings.py` |
| MockEmbeddingClient (tests) | ✅ | `src/utils/embeddings.py` |
| Optional api_key in config | ✅ | `src/core/config.py` – EmbeddingSettings.api_key |

### Task 3.3: Entity and Relation Extraction ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| EntityExtractor (LLM-based) | ✅ | `src/extraction/entity_extractor.py` – EntityType, EntityMention |
| RelationExtractor (LLM-based) | ✅ | `src/extraction/relation_extractor.py` – Relation extraction |

### Task 3.4: Hippocampal Store Implementation ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| PostgresMemoryStore (pgvector) | ✅ | `src/storage/postgres.py` – upsert, get_by_id, vector_search, scan, count; MemoryRecordCreate with embedding |
| HippocampalStore facade | ✅ | `src/memory/hippocampal/store.py` – encode_chunk (write gate → redact → embed → extract → upsert), search |

### Phase 3 Tests

- **Unit:** `tests/unit/test_phase3_write_gate.py` – WriteGate (skip/store/skip secrets/novelty), PIIRedactor (email, phone, clean)
- **Unit:** `tests/unit/test_phase3_embeddings.py` – MockEmbeddingClient (dimensions, deterministic embed, batch)
- **Integration:** `tests/integration/test_phase3_hippocampal_encode.py` – encode_chunk → record in DB, get_recent retrieval, search smoke test; skip low salience

**Phase 3 deliverables (from plan):** All checklist items completed.

---

## Phase 4: Neocortical Store ✅

**Status:** Implemented  
**Plan reference:** `Phase4_NeocorticalStore.md`

### Task 4.1: Neo4j Graph Store ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Neo4jGraphStore | ✅ | `src/storage/neo4j.py` – merge_node, merge_edge, get_neighbors, personalized_pagerank (GDS + fallback), get_entity_facts, search_by_pattern, delete_entity |
| Graph schema init | ✅ | `initialize_graph_schema()` in `src/storage/neo4j.py` – constraints and indexes |

### Task 4.2: Semantic Fact Management ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| FactCategory, FactSchema, SemanticFact, DEFAULT_FACT_SCHEMAS | ✅ | `src/memory/neocortical/schemas.py` |
| SemanticFactStore | ✅ | `src/memory/neocortical/fact_store.py` – upsert_fact, get_fact, get_facts_by_category, get_user_profile, search_facts, invalidate_fact; versioning and temporal handling |
| semantic_facts migration | ✅ | `migrations/versions/002_semantic_facts.py`; SemanticFactModel in `src/storage/models.py` |

### Task 4.3: Neocortical Store Facade ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| NeocorticalStore | ✅ | `src/memory/neocortical/store.py` – store_fact, store_relation(s), get_fact, get_user_profile, query_entity, multi_hop_query, find_schema_match, text_search, _sync_fact_to_graph |
| SchemaManager | ✅ | `src/memory/neocortical/schema_manager.py` – get_schema, get_schemas_for_category, register_schema, validate_key |

### Phase 4 Tests

- **Unit:** `tests/unit/test_phase4_schemas.py` – FactCategory, FactSchema, SemanticFact, DEFAULT_FACT_SCHEMAS, SchemaManager
- **Integration:** `tests/integration/test_phase4_fact_store.py` – upsert/get fact, get_facts_by_category, get_user_profile, search_facts
- **Integration:** `tests/integration/test_phase4_neocortical.py` – NeocorticalStore store_fact, get_fact, get_user_profile, text_search (with mock graph)

**Phase 4 deliverables (from plan):** All checklist items completed.

---

## Phase 5: Retrieval System ✅

**Status:** Implemented  
**Plan reference:** `Phase5_RetrievalSystem.md`

### Task 5.1: Query Classification ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| QueryIntent, QueryAnalysis | ✅ | `src/retrieval/query_types.py` |
| QueryClassifier | ✅ | `src/retrieval/classifier.py` – fast patterns (preference, identity, task, temporal, procedural), LLM fallback via complete_json |

### Task 5.2: Retrieval Planner ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| RetrievalSource, RetrievalStep, RetrievalPlan | ✅ | `src/retrieval/planner.py` |
| RetrievalPlanner | ✅ | `src/retrieval/planner.py` – plan from analysis (fast-path fact key, vector, graph, temporal, general hybrid) |

### Task 5.3: Hybrid Retriever ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| RetrievalResult, HybridRetriever | ✅ | `src/retrieval/retriever.py` – execute plan (parallel steps), _retrieve_facts, _retrieve_vector, _retrieve_graph, _retrieve_cache (logs decode errors), _to_retrieved_memories |

### Task 5.4: Reranker ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| RerankerConfig, MemoryReranker | ✅ | `src/retrieval/reranker.py` – relevance/recency/confidence/diversity, MMR-style diversity |

### Task 5.5 & 5.6: Packet Builder & MemoryRetriever ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| MemoryPacketBuilder | ✅ | `src/retrieval/packet_builder.py` – build (categorize by type), _detect_conflicts, to_llm_context (markdown/json) |
| MemoryRetriever | ✅ | `src/retrieval/memory_retriever.py` – retrieve (classify → plan → retrieve → rerank → packet), retrieve_for_llm |

### Phase 5 Tests

- **Unit:** `tests/unit/test_phase5_retrieval.py` – QueryClassifier (fast preference/identity, fallback), RetrievalPlanner (preference lookup, general), MemoryReranker, MemoryPacketBuilder
- **Integration:** `tests/integration/test_phase5_retrieval_flow.py` – full retrieve returns packet with facts, retrieve_for_llm returns string

**Phase 5 deliverables (from plan):** All checklist items completed.

---

## Phase 6: Reconsolidation & Belief Revision ✅

**Status:** Implemented  
**Plan reference:** `Phase6_Reconsolidation.md`

### Task 6.1: Labile State Management ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| LabileMemory, LabileSession models | ✅ | `src/reconsolidation/labile_tracker.py` |
| LabileStateTracker (mark_labile, get_labile_memories, release_labile, session cleanup) | ✅ | `src/reconsolidation/labile_tracker.py` |

### Task 6.2: Conflict Detection ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| ConflictType enum, ConflictResult model | ✅ | `src/reconsolidation/conflict_detector.py` |
| ConflictDetector (fast heuristics + LLM fallback, detect_batch) | ✅ | `src/reconsolidation/conflict_detector.py` |

### Task 6.3: Belief Revision Engine ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| RevisionStrategy, RevisionPlan, RevisionOperation | ✅ | `src/reconsolidation/belief_revision.py` |
| BeliefRevisionEngine (reinforce, time_slice, correction, contradiction, refinement, hypothesis) | ✅ | `src/reconsolidation/belief_revision.py` |

### Task 6.4: Reconsolidation Orchestrator ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| ReconsolidationService (process_turn, fact extraction fallback, apply operations) | ✅ | `src/reconsolidation/service.py` |
| FactExtractor / LLMFactExtractor | ✅ | `src/extraction/fact_extractor.py` – LLM-based extraction (same client as summarization); base no-op for tests |

### Phase 6 Tests

- **Unit:** `tests/unit/test_phase6_reconsolidation.py` – LabileStateTracker, ConflictDetector (fast path), BeliefRevisionEngine (reinforce, correction, time_slice)
- **Integration:** `tests/integration/test_phase6_reconsolidation_flow.py` – process_turn with no memories; correction flow (store → retrieve → correct → reconsolidate)

### Phase 6 Deliverables Checklist (from plan)

- [x] LabileMemory and LabileSession models
- [x] LabileStateTracker with session management
- [x] ConflictType enum and ConflictResult model
- [x] ConflictDetector with fast heuristics and LLM fallback
- [x] RevisionStrategy enum and RevisionPlan model
- [x] BeliefRevisionEngine with all strategies
- [x] ReconsolidationService orchestrating the flow
- [x] Unit tests for conflict detection
- [x] Unit tests for revision planning
- [x] Integration tests for full reconsolidation

---

## Phase 7: Consolidation Engine ✅

**Status:** Implemented  
**Plan reference:** `Phase7_Consolidation.md`

### Task 7.1: Consolidation Triggers and Scheduler ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| TriggerType, TriggerCondition, ConsolidationTask | ✅ | `src/consolidation/triggers.py` |
| ConsolidationScheduler (scheduled, quota, event, manual) | ✅ | `src/consolidation/triggers.py` – register_user, check_triggers, trigger_manual, get_next_task |

### Task 7.2: Episode Sampling and Clustering ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| SamplingConfig, EpisodeSampler | ✅ | `src/consolidation/sampler.py` – sample with importance/access/recency scoring; scan with since filter |
| EpisodeCluster, SemanticClusterer | ✅ | `src/consolidation/clusterer.py` – pure-Python cosine clustering (no sklearn) |

### Task 7.3: Gist Extraction ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| ExtractedGist, GistExtractor | ✅ | `src/consolidation/summarizer.py` – LLM extraction with JSON fallback to simple summary |

### Task 7.4: Schema Alignment and Migration ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| AlignmentResult, SchemaAligner | ✅ | `src/consolidation/schema_aligner.py` – key/preference/search match, suggest_schema |
| MigrationResult, ConsolidationMigrator | ✅ | `src/consolidation/migrator.py` – migrate to neocortical, mark_episodes_consolidated |

### Task 7.5: Consolidation Worker ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| ConsolidationReport, ConsolidationWorker | ✅ | `src/consolidation/worker.py` – sample → cluster → extract → align → migrate; background worker loop; structlog for completion/failure |

### Phase 7 Tests

- **Unit:** `tests/unit/test_phase7_consolidation.py` – ConsolidationScheduler (manual, quota, scheduled), SemanticClusterer (empty, single, similar embeddings)
- **Integration:** `tests/integration/test_phase7_consolidation_flow.py` – empty episodes report; full flow with fallback gist and migrate

### Phase 7 Deliverables Checklist (from plan)

- [x] TriggerCondition and ConsolidationTask models
- [x] ConsolidationScheduler with multiple trigger types
- [x] EpisodeSampler with priority scoring
- [x] SemanticClusterer using embeddings (pure Python)
- [x] GistExtractor with LLM summarization
- [x] SchemaAligner for rapid integration
- [x] ConsolidationMigrator for semantic store
- [x] ConsolidationWorker orchestrating full flow
- [x] ConsolidationReport for audit
- [x] Background worker with task queue
- [x] Unit tests for clustering and triggers
- [x] Integration tests for full consolidation

---

## Phase 8: Active Forgetting ✅

**Status:** Implemented  
**Plan reference:** `Phase8_ActiveForgetting.md`

### Task 8.1: Relevance Scoring ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| RelevanceWeights, RelevanceScore, ScorerConfig | ✅ | `src/forgetting/scorer.py` |
| RelevanceScorer (importance, recency, frequency, confidence, type bonus, dependency) | ✅ | `src/forgetting/scorer.py` – score, score_batch, _suggest_action |

### Task 8.2: Forgetting Policy Engine ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| ForgettingAction, ForgettingOperation, ForgettingResult | ✅ | `src/forgetting/actions.py` |
| ForgettingPolicyEngine (plan_operations, create_compression) | ✅ | `src/forgetting/actions.py` |
| ForgettingExecutor (decay, silence, compress, archive, delete) | ✅ | `src/forgetting/executor.py` – execute, _execute_* |

### Task 8.3: Interference Management ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| InterferenceResult, InterferenceDetector | ✅ | `src/forgetting/interference.py` – detect_duplicates (embeddings), detect_overlapping (text) |

### Task 8.4: Forgetting Worker ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| ForgettingReport, ForgettingWorker | ✅ | `src/forgetting/worker.py` – run_forgetting (scan → score → plan → duplicate resolution → execute) |
| ForgettingScheduler | ✅ | `src/forgetting/worker.py` – start, stop, schedule_user, _scheduler_loop |

### Task 8.5: LLM-Based Compression ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| summarize_for_compression | ✅ | `src/forgetting/compression.py` – LLM gist when client provided, else truncate |
| VLLMClient (OpenAI-compatible) | ✅ | `src/utils/llm.py` – base_url, vllm_model; config LLM__VLLM_BASE_URL, LLM__VLLM_MODEL |
| Executor/Worker compression_llm_client | ✅ | `src/forgetting/executor.py`, `worker.py` – optional LLM for compress |
| vLLM Docker service (Llama 3.2 1B) | ✅ | `docker/docker-compose.yml` – profile `vllm`, image vllm/vllm-openai |

### Task 8.6: Dependency Check Before Delete ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| PostgresMemoryStore.count_references_to | ✅ | `src/storage/postgres.py` – supersedes_id + evidence_refs in same tenant/user |
| Executor skip delete when refs > 0 | ✅ | `src/forgetting/executor.py` – _execute_delete checks refs, appends skip reason to errors |

### Task 8.7: Celery / Background Task ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Celery app (Redis broker) | ✅ | `src/celery_app.py` – broker/backend from settings |
| run_forgetting_task | ✅ | `src/celery_app.py` – tenant_id, user_id, dry_run, max_memories; returns report dict |
| Beat schedule (forgetting-daily) | ✅ | `src/celery_app.py` – 24h schedule, queue `forgetting` |

### Phase 8 Tests

- **Unit:** `tests/unit/test_phase8_forgetting.py` – RelevanceWeights, RelevanceScorer, ForgettingPolicyEngine, InterferenceDetector, **Compression (LLM/truncate), DependencyCheck (count_references_to, executor skip delete)**
- **Unit:** `tests/unit/test_phase8_celery.py` – task registration, beat schedule
- **Integration:** `tests/integration/test_phase8_forgetting_flow.py` – empty memories, dry run, decay reduces confidence
- **Integration:** `tests/integration/test_phase8_vllm_compression.py` – **real vLLM summarization** (skipped unless `LLM__VLLM_BASE_URL` or `VLLM_BASE_URL` is set). To run: start vLLM (`docker compose --profile vllm up -d vllm` or `--profile vllm-cpu up -d vllm-cpu`), then `docker compose run --rm -e LLM__VLLM_BASE_URL=http://vllm:8000/v1 app pytest tests/integration/test_phase8_vllm_compression.py -v`.

### Phase 8 Deliverables Checklist (from plan)

- [x] RelevanceWeights and ScorerConfig models
- [x] RelevanceScorer with multi-factor scoring
- [x] ForgettingAction enum and operation models
- [x] ForgettingPolicyEngine with action thresholds
- [x] ForgettingExecutor for all action types
- [x] InterferenceDetector for duplicates
- [x] ForgettingWorker orchestrating the flow
- [x] ForgettingScheduler for background runs
- [x] ForgettingReport for audit
- [x] Unit tests for scoring
- [x] Unit tests for policy decisions
- [x] Integration tests for full forgetting flow
- [x] LLM-based compression (vLLM/Llama 3.2 1B, summarize_for_compression)
- [x] Dependency check before delete (count_references_to, skip with error)
- [x] Celery task and beat schedule for forgetting

---

## Phase 9: REST API & Integration ✅

**Status:** Implemented  
**Plan reference:** `Phase9_RestAPI.md`

### Task 9.1: FastAPI Application Setup ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Application factory with lifespan | ✅ | `src/api/app.py` – `create_app()`, lifespan for DB + orchestrator |
| RequestLoggingMiddleware | ✅ | `src/api/middleware.py` – request_id, timing, X-Response-Time header |
| RateLimitMiddleware | ✅ | `src/api/middleware.py` – per-tenant (X-Tenant-ID), 60 req/min |
| CORS middleware | ✅ | `src/api/app.py` |

### Task 9.2: Authentication and Authorization ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| API key authentication (config-based) | ✅ | `src/api/auth.py` – keys from env (AUTH__API_KEY, AUTH__ADMIN_API_KEY) |
| AuthContext, get_auth_context | ✅ | `src/api/auth.py` |
| require_write_permission, require_admin_permission | ✅ | `src/api/auth.py` |

### Task 9.3: API Routes ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Request/Response Pydantic models | ✅ | `src/api/schemas.py` – Write, Read, Update, Forget, MemoryStats |
| POST /memory/write | ✅ | `src/api/routes.py` |
| POST /memory/read (format: packet, llm_context) | ✅ | `src/api/routes.py` |
| POST /memory/update (with feedback) | ✅ | `src/api/routes.py` |
| POST /memory/forget | ✅ | `src/api/routes.py` |
| GET /memory/stats | ✅ | `src/api/routes.py` |
| GET /health | ✅ | `src/api/routes.py` |

### Task 9.4: Memory Orchestrator ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| MemoryOrchestrator | ✅ | `src/memory/orchestrator.py` – write, read, update, forget, get_stats, delete_all_for_scope |
| Coordinates short-term, hippocampal, neocortical, retrieval, reconsolidation, consolidation, forgetting | ✅ | Factory `create(db_manager)` wires all deps |

### Task 9.5: Admin Routes ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| POST /admin/consolidate/{user_id} | ✅ | `src/api/admin_routes.py` |
| POST /admin/forget/{user_id} (dry_run) | ✅ | `src/api/admin_routes.py` |

### Task 9.6: Web Dashboard (Monitoring & Management) ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Dashboard API routes (admin-only) | ✅ | `src/api/dashboard_routes.py` – overview, memories list/detail, events, timeline, components health, tenants, consolidate, forget |
| Dashboard Pydantic schemas | ✅ | `src/api/schemas.py` – DashboardOverview, DashboardMemoryListItem/Detail, DashboardEventItem, TimelinePoint, ComponentStatus, TenantInfo, etc. |
| Static SPA (vanilla HTML/CSS/JS) | ✅ | `src/dashboard/static/` – index.html, css/styles.css, js/app.js, api.js, pages (overview, memories, detail, components, events, management), utils (formatters, charts) |
| FastAPI integration | ✅ | `src/api/app.py` – dashboard_router at `/api/v1`, static mount at `/dashboard/static`, SPA catch-all at `/dashboard` |
| Overview page | ✅ | KPI cards, type/status/timeline/facts charts (Chart.js), system health, recent events |
| Memory Explorer | ✅ | Filterable, sortable, paginated table; click-through to detail |
| Memory Detail | ✅ | Full record: content, metrics, provenance, entities/relations, metadata, related events |
| Components page | ✅ | PostgreSQL, Neo4j, Redis health and metrics; architecture legend |
| Events page | ✅ | Paginated event log, expandable payloads, auto-refresh toggle |
| Management page | ✅ | Trigger consolidation and forgetting with tenant selector, dry-run, result display |
| Auth | ✅ | Login overlay with admin API key; key stored in localStorage; all dashboard API calls use X-API-Key |

### Phase 9 Tests

- **Unit:** `tests/unit/test_phase9_api.py` – Auth config (_build_api_keys), schemas
- **Integration:** `tests/integration/test_phase9_api_flow.py` – health, auth required for write/read/stats

### Phase 9 Deliverables Checklist (from plan)

- [x] FastAPI application factory with lifespan
- [x] RequestLoggingMiddleware with timing
- [x] RateLimitMiddleware with per-tenant limits
- [x] Config-based API key validation (AUTH__API_KEY, AUTH__ADMIN_API_KEY)
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
- [x] OpenAPI documentation (via FastAPI)
- [x] Unit tests for routes/auth
- [x] Integration tests for API flow
- [x] Web dashboard (monitoring & management) at /dashboard with admin-only API

---

## Phase 10: Testing & Deployment ✅

**Status:** Implemented  
**Plan reference:** `Phase10_TestingDeployment.md`

### Task 10.1: Unit Testing ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| conftest fixtures (sample_memory_record, sample_chunk, mock_llm, mock_embeddings) | ✅ | `tests/conftest.py` |
| WriteGateConfig tests | ✅ | `tests/unit/test_phase3_write_gate.py` – TestWriteGateConfig |
| WriteGate PII redaction test | ✅ | `tests/unit/test_phase3_write_gate.py` – test_pii_triggers_redaction |
| RelevanceScorer tests | ✅ | Existing in `tests/unit/test_phase8_forgetting.py` |
| ConflictDetector tests | ✅ | Existing in `tests/unit/test_phase6_reconsolidation.py` |

### Task 10.2: Integration Testing ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Integration test setup with testcontainers | ✅ | `tests/integration/conftest.py` – PostgresContainer, Neo4jContainer, pg_engine, db_session when testcontainers installed |
| Integration tests (Postgres via docker-compose or testcontainers) | ✅ | Existing Phase 1–9 integration tests |

### Task 10.3: E2E Testing ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| API E2E tests | ✅ | `tests/e2e/test_api_flows.py` – full lifecycle (skip if no API key), unauthorized, health structure |

### Task 10.4: Docker Configuration ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Dockerfile with multi-stage build | ✅ | `docker/Dockerfile` – base → dependencies → production; non-root user, HEALTHCHECK |
| API service healthcheck | ✅ | `docker/docker-compose.yml` – curl health check for api service |

### Task 10.5: CI/CD Pipeline ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| GitHub Actions CI | ✅ | `.github/workflows/ci.yml` – lint (ruff, black), test (postgres, neo4j, redis) |
| Test coverage reporting | ✅ | pytest --cov=src --cov-report=xml; codecov-action (optional) |
| Docker image build and push | ✅ | build job on main: buildx, push to ghcr.io (latest + sha) |
| Docker image build/push on main | ✅ | ghcr.io (latest + sha); no deploy job (add when staging env exists) |

### Task 10.6: Logging & Observability ✅

| Deliverable | Status | Location / Notes |
|-------------|--------|------------------|
| Structured logging config | ✅ | `src/utils/logging_config.py` – configure_logging, get_logger |
| Prometheus metrics | ✅ | `src/utils/metrics.py` – MEMORY_WRITES, MEMORY_READS, RETRIEVAL_LATENCY, MEMORY_COUNT, track_retrieval_latency; /metrics endpoint; counters in API routes |

### Phase 10 Deliverables Checklist (from plan)

- [x] pytest configuration with fixtures
- [x] Unit tests for WriteGate (incl. config, PII)
- [x] Unit tests for RelevanceScorer
- [x] Unit tests for ConflictDetector
- [x] Integration test setup with testcontainers
- [x] Integration tests (docker-compose or testcontainers)
- [x] API E2E tests
- [x] Dockerfile with multi-stage build
- [x] Docker healthcheck for API
- [x] GitHub Actions CI workflow
- [x] Linting (ruff, black) in CI
- [x] Test coverage reporting (pytest-cov, codecov optional)
- [x] Docker image build and push (ghcr.io on main)
- [x] CI: lint, test, build and push image (deploy job omitted until staging configured)
- [x] Structured logging configuration
- [x] Prometheus metrics (/metrics, counters in routes)
- [x] Health check endpoints (existing)
- [x] Documentation (README, ProjectStatus)

---

## Phase 11: Holistic Memory Refactoring ✅

**Status:** Implemented  
**Plan reference:** `Phase11_HolisticMemory.md` (Seamless Holistic Memory plan)

- Removed `MemoryScope`; added `MemoryContext` for tagging. Updated `MemoryRecord` / `MemoryRecordCreate` with `context_tags`, `source_session_id`.
- Migration `004_remove_scopes_holistic`: drop scope columns, add `context_tags`, `source_session_id`; GIN index on `context_tags`.
- Storage, orchestrator, retrieval, API: all use tenant-only holistic access; optional `context_filter` on read.
- Stats endpoint: `GET /memory/stats` (tenant from auth).

---

## Phase 12: Seamless Memory Integration ✅

**Status:** Implemented  
**Plan reference:** `Phase12_SeamlessIntegration.md`

- `SeamlessMemoryProvider`: auto-retrieve per turn, optional auto-store, reconsolidation; returns `memory_context` for LLM injection.
- `POST /memory/turn` and `ProcessTurnRequest` / `ProcessTurnResponse`.
- QueryClassifier: `recent_context` for vague queries. Tool definitions simplified (no scope).

---

## Phase 13: Code Improvements ✅

**Status:** Implemented  
**Plan reference:** `Phase13_CodeImprovements.md`

- Orchestrator `update()`: re-embed and re-extract entities when text changes.
- Working memory: recency-aware eviction in `WorkingMemoryState.add_chunk()`.
- Chunker: sentiment-aware salience boost (`_compute_salience_boost_for_sentiment`).
- Belief revision: archive instead of delete (valid_to, status=ARCHIVED).

---

## Phase 14: Documentation & Examples Update ✅

**Status:** Implemented  
**Plan reference:** `Phase14_DocumentationUpdate.md`

- README and UsageDocumentation updated for holistic API and seamless memory; tool definitions and API reference without scopes.
- Examples: `memory_client` (holistic + `process_turn`), `basic_usage`, `chatbot_with_memory`, `openai_tool_calling`, `anthropic_tool_calling`, `langchain_integration` updated.

---

# Intrinsic Memory System — Project Plan Status

The following phases transform the system from external RAG to **intrinsic, active memory** integrated with the LLM's computation graph. All phases are **planned** (status: ⏳ Not Started).

**Plan reference:** `Phase1_Foundation_ModelAccessLayer.md` through `Phase10_ObservabilityBenchmarking.md`

---

## Intrinsic Phase 1: Foundation & Model Access Layer ⏳

**Status:** Not Started  
**Plan reference:** `Phase1_Foundation_ModelAccessLayer.md`

| Task | Description | Status |
|------|-------------|--------|
| 1.1 | Model Backend Abstraction — `ModelBackend` ABC, `ModelSpec`, `HookHandle`, `InterfaceCapability` | ⏳ |
| 1.2 | Hook Manager & Lifecycle — centralized hook registration, ordering, safety guards | ⏳ |
| 1.3 | Model Inspector & Vocabulary Mapper — layer shapes, attention heads, vocab mapping | ⏳ |
| 1.4 | Intrinsic Memory Bus — `InjectionChannel`, `MemoryVector`, `InjectionRequest` | ⏳ |
| 1.5 | Configuration & Feature Flags — `IntrinsicMemoryConfig` | ⏳ |
| 1.6 | Directory Structure & Module Scaffolding — `src/intrinsic/` layout | ⏳ |

---

## Intrinsic Phase 2: Logit Interface ⏳

**Status:** Not Started  
**Plan reference:** `Phase2_LogitInterface.md`

| Task | Description | Status |
|------|-------------|--------|
| 2.1 | Token-Memory Mapper — memory text → key tokens → token_ids → bias values | ⏳ |
| 2.2 | Simple Logit Bias Engine — aggregate biases, apply logit_bias | ⏳ |
| 2.3 | kNN-LM Interpolator — blend parametric and non-parametric distributions | ⏳ |
| 2.4 | Logit Hook for Local Models — lm_head hook | ⏳ |
| 2.5 | Memory Bus Integration — LogitInjectionHook, channel dispatch | ⏳ |
| 2.6 | API Integration for Logit Bias — extend OpenAI client, SeamlessMemoryProvider | ⏳ |

---

## Intrinsic Phase 3: Activation Interface ⏳

**Status:** Not Started  
**Plan reference:** `Phase3_ActivationInterface.md`

| Task | Description | Status |
|------|-------------|--------|
| 3.1 | Steering Vector Derivation Engine — CDD, IdentityV, MeanCenteringPCA | ⏳ |
| 3.2 | Activation Injection Engine — hidden state steering, norm preservation | ⏳ |
| 3.3 | Multi-Layer Injection Strategy — layer selection, SteeringVectorField | ⏳ |
| 3.4 | Memory Bus Integration — ActivationInjectionHook | ⏳ |
| 3.5 | Steering Vector Cache & Pre-computation | ⏳ |

---

## Intrinsic Phase 4: Synaptic Interface ⏳

**Status:** Not Started  
**Plan reference:** `Phase4_SynapticInterface.md`

| Task | Description | Status |
|------|-------------|--------|
| 4.1 | KV Encoder Pipeline — memory text → precomputed KV pairs | ⏳ |
| 4.2 | KV-Cache Injector — append/prepend virtual KV pairs | ⏳ |
| 4.3 | Temporal Decay (SynapticRAG) — decay manager for injected memories | ⏳ |
| 4.4 | Synaptic Interface — Complete Pipeline | ⏳ |
| 4.5 | Position Encoding Remapping | ⏳ |

---

## Intrinsic Phase 5: Controller & Gating Unit ⏳

**Status:** Not Started  
**Plan reference:** `Phase5_ControllerGatingUnit.md`

| Task | Description | Status |
|------|-------------|--------|
| 5.1 | Relevance Gate — multi-factor memory filtering | ⏳ |
| 5.2 | Interface Router — channel selection by memory type and backend capabilities | ⏳ |
| 5.3 | Complete Controller Pipeline — gate → route → calibrate → dispatch | ⏳ |
| 5.4 | Fallback Chain Manager — graceful degradation | ⏳ |
| 5.5 | Integration with Existing Memory Pipeline | ⏳ |

---

## Intrinsic Phase 6: Memory Encoding Pipeline ⏳

**Status:** Not Started  
**Plan reference:** `Phase6_MemoryEncodingPipeline.md`

| Task | Description | Status |
|------|-------------|--------|
| 6.1 | Unified Encoded Memory Data Structure — `MemoryAnalysis`, `EncodedMemory` | ⏳ |
| 6.2 | Hippocampal Encoder — Main Pipeline — logit, activation, KV in single pass | ⏳ |
| 6.3 | Encoding Cache & Store — `EncodedMemoryStore` | ⏳ |
| 6.4 | Projection Head — Learned Steering Vector generation | ⏳ |

---

## Intrinsic Phase 7: Memory Hierarchy & Cache Management ⏳

**Status:** Not Started  
**Plan reference:** `Phase7_MemoryHierarchyCacheManagement.md`

| Task | Description | Status |
|------|-------------|--------|
| 7.1 | Tiered Memory Store — L1 (GPU HBM), L2 (CPU DRAM), L3 (NVMe SSD) | ⏳ |
| 7.2 | Predictive Pre-fetcher — anticipate memory needs | ⏳ |
| 7.3 | Eviction Policies — EvicPress-inspired, importance-aware eviction | ⏳ |

---

## Intrinsic Phase 8: Weight Adaptation Interface ⏳

**Status:** Not Started  
**Plan reference:** `Phase8_WeightAdaptationInterface.md`

| Task | Description | Status |
|------|-------------|--------|
| 8.1 | Adapter Registry & Manager — LoRA adapter catalog, load/unload | ⏳ |
| 8.2 | Adapter Router / Classifier — query classification for adapter selection | ⏳ |
| 8.3 | Hypernetwork (Advanced) — on-the-fly LoRA weight generation | ⏳ |
| 8.4 | Memory Bus Integration | ⏳ |

---

## Intrinsic Phase 9: Integration & Migration ⏳

**Status:** Not Started  
**Plan reference:** `Phase9_IntegrationMigration.md`

| Task | Description | Status |
|------|-------------|--------|
| 9.1 | Application Startup Integration — wire intrinsic system into FastAPI lifespan | ⏳ |
| 9.2 | New API Endpoints — `/intrinsic/status`, `/intrinsic/diagnostics`, `/intrinsic/encode`, etc. | ⏳ |
| 9.3 | Enhanced `/memory/turn` Endpoint — `IntrinsicInjectionInfo` in response | ⏳ |
| 9.4 | Migration Strategy — `MigrationMode`, `MigrationManager` (RAG → hybrid → intrinsic) | ⏳ |
| 9.5 | End-to-End Integration Tests | ⏳ |

---

## Intrinsic Phase 10: Observability & Benchmarking ⏳

**Status:** Not Started  
**Plan reference:** `Phase10_ObservabilityBenchmarking.md`

| Task | Description | Status |
|------|-------------|--------|
| 10.1 | Metrics & Monitoring — Prometheus metrics (injections, latency, cache hits, safety reverts) | ⏳ |
| 10.2 | Benchmarking Framework — A/B testing (RAG vs. Intrinsic) | ⏳ |
| 10.3 | Safety Guardrails — OutputQualityMonitor, EmergencyKillSwitch | ⏳ |
| 10.4 | Interpretability Tools — InjectionAttributor | ⏳ |
| 10.5 | Production Hardening Checklist | ⏳ |

---

## Intrinsic Memory Phases Summary

| Phase | Name | Status |
|-------|------|--------|
| I1 | Foundation & Model Access Layer | ⏳ Not Started |
| I2 | Logit Interface | ⏳ Not Started |
| I3 | Activation Interface | ⏳ Not Started |
| I4 | Synaptic Interface | ⏳ Not Started |
| I5 | Controller & Gating Unit | ⏳ Not Started |
| I6 | Memory Encoding Pipeline | ⏳ Not Started |
| I7 | Memory Hierarchy & Cache Management | ⏳ Not Started |
| I8 | Weight Adaptation Interface | ⏳ Not Started |
| I9 | Integration & Migration | ⏳ Not Started |
| I10 | Observability & Benchmarking | ⏳ Not Started |

---

## Phases 3–10

| Phase | Name | Status |
|-------|------|--------|
| 3 | Hippocampal Store | ✅ Implemented |
| 4 | Neocortical Store | ✅ Implemented |
| 5 | Retrieval System | ✅ Implemented |
| 6 | Reconsolidation & Belief Revision | ✅ Implemented |
| 7 | Consolidation Engine | ✅ Implemented |
| 8 | Active Forgetting | ✅ Implemented |
| 9 | REST API & Integration | ✅ Implemented |
| 10 | Testing & Deployment | ✅ Implemented |

---

## How to Run / Use

### Docker (recommended for Phase 1)

All builds and tests run via Docker:

```pseudo
# Pseudo: Shell commands – Build app image; run migrations and tests via docker compose.
```

Or run only tests (Postgres must be up):

```pseudo
# Pseudo: Shell commands – Implementation as per task description.
```

**Phase 1:** 19 tests (4 integration, 15 unit) — all passing.  
**Phase 2:** 14 tests (1 integration, 13 unit) — all passing.  
**Phase 3:** 12 tests (2 integration, 10 unit) — all passing.  
**Phase 4:** 16 tests (6 integration, 10 unit) — all passing.  
**Phase 5:** 11 tests (2 integration, 9 unit) — all passing.  
**Phase 6:** 9 tests (2 integration, 7 unit) — all passing.  
**Phase 7:** 10 tests (2 integration, 8 unit) — all passing.  
**Phase 8:** 28 tests (5 integration, 23 unit) — all passing (vLLM integration tests skip when vLLM not configured).  
**Phase 9:** 11 tests (5 integration, 6 unit) — API auth, schemas, health, auth-required.  
**Phase 10:** 6 tests (3 E2E, 3 unit additions) — conftest fixtures, WriteGateConfig, PII, E2E flows.  
**Total:** 138 tests (135 passed, 3 skipped: 2 vLLM + 1 full lifecycle when no API key).

### Local

- **Install:** `poetry install`  
- **Create DB and run migrations:** Set `DATABASE__POSTGRES_URL` (or default) then `poetry run alembic upgrade head`  
- **Import from code:** Use `from src.core.config import get_settings`, `from src.storage.event_log import EventLogRepository`, etc., with project root on `PYTHONPATH` or after `poetry install`.

---

## Notes

- `MemoryRecordModel` uses column name `"metadata"` in the DB; the ORM attribute is `meta` to avoid clashing with SQLAlchemy `Base.metadata`.  
- Event log is append-only; `EventLogRepository.replay_events()` is an async generator for state rebuild.  
- `MemoryStoreBase` is implemented by `PostgresMemoryStore`; `GraphStoreBase` by `Neo4jGraphStore`.  
- Phase 4 adds `semantic_facts` table (migration 002) and `NeocorticalStore` (graph + fact store).  
- Phase 5 adds hybrid retrieval: `QueryClassifier`, `RetrievalPlanner`, `HybridRetriever`, `MemoryReranker`, `MemoryPacketBuilder`, `MemoryRetriever`.
- Phase 6 adds reconsolidation: `LabileStateTracker`, `ConflictDetector`, `BeliefRevisionEngine`, `ReconsolidationService`; `PostgresMemoryStore.update()` supports `valid_to` and `metadata` for revision patches.
- Phase 7 adds consolidation: `ConsolidationScheduler`, `EpisodeSampler`, `SemanticClusterer`, `GistExtractor`, `SchemaAligner`, `ConsolidationMigrator`, `ConsolidationWorker`; `PostgresMemoryStore.scan()` supports `since` filter for time-window sampling.
- Phase 8 adds active forgetting: `RelevanceScorer`, `ForgettingPolicyEngine`, `ForgettingExecutor`, `InterferenceDetector`, `ForgettingWorker`, `ForgettingScheduler`; `PostgresMemoryStore.update()` supports `entities` and `relations` for compress. Optional: LLM-based compression via `summarize_for_compression` and `VLLMClient` (vLLM + Llama 3.2 1B in Docker); dependency check before delete via `count_references_to`; Celery task `run_forgetting_task` and beat schedule.
- Phase 9 adds REST API: `src/api/app.py`, `auth.py`, `middleware.py`, `schemas.py`, `routes.py`, `admin_routes.py`; `MemoryOrchestrator` in `src/memory/orchestrator.py`; endpoints `/api/v1/memory/write`, `/read`, `/update`, `/forget`, `/stats`, `/health`; admin endpoints `/api/v1/admin/consolidate`, `/forget`; API key auth, multi-tenancy (X-Tenant-ID). Phase 11+ refactor: holistic (tenant-only) memory; no scopes. Phase 12 adds `/memory/turn` (seamless memory) and `SeamlessMemoryProvider`. **Dashboard (Task 9.6):** Web app at `/dashboard` (vanilla HTML/CSS/JS SPA in `src/dashboard/static/`), dashboard API at `/api/v1/dashboard/*` (overview, memories, events, timeline, components, tenants, consolidate, forget); admin key required.
- Phase 10 adds: conftest fixtures (sample_memory_record, sample_chunk, mock_llm, mock_embeddings); E2E tests in `tests/e2e/test_api_flows.py`; integration test setup with testcontainers (`tests/integration/conftest.py`); multi-stage Dockerfile (`docker/Dockerfile`); GitHub Actions CI with lint, test (coverage + codecov), build/push to ghcr.io; API healthcheck in docker-compose; `src/utils/logging_config.py` for structured logging; `src/utils/metrics.py` for Prometheus (MEMORY_WRITES, MEMORY_READS, RETRIEVAL_LATENCY, MEMORY_COUNT) and `/metrics` endpoint.
- **Intrinsic Memory System (I1–I10):** Plan documents exist (`Phase1_Foundation_ModelAccessLayer.md` through `Phase10_ObservabilityBenchmarking.md`). These phases define the architecture for moving from external RAG to intrinsic LLM memory integration (Model Access Layer, Logit/Activation/Synaptic/Weight interfaces, Controller, Hippocampal Encoder, Tiered Cache, Integration, Observability). Implementation has not yet started.


---

