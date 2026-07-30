<div align="center">

# Cognitive Memory Layer

### A Neuro-Inspired Memory System for AI

*Store. Retrieve. Consolidate. Forget.*
*Just like the human brain.*

<br/>

[![Quick Start](https://img.shields.io/badge/Quick%20Start-5%20min-success?style=for-the-badge&logo=rocket)](#-quick-start)
[![Docs](https://img.shields.io/badge/Docs-Full%20API-blue?style=for-the-badge&logo=gitbook)](./docs/usage.md)
[![Tests](https://img.shields.io/badge/Tests-1233-brightgreen?style=for-the-badge&logo=pytest)](./tests/README.md)
[![Version](https://img.shields.io/badge/version-1.5.0-blue?style=for-the-badge)](#)

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20DB-008CC1?style=flat-square&logo=neo4j&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Cache+Queue-DC382D?style=flat-square&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-GPL--3.0-A42E2B?style=flat-square&logo=gnu&logoColor=white)

</div>

---

<div align="center">

**[The Problem](#-the-problem)** &#8226;
**[How Your Brain Solves It](#-how-your-brain-solves-it)** &#8226;
**[How CML Implements It](#-how-cml-implements-it)** &#8226;
**[Architecture](#-architecture)** &#8226;
**[Quick Start](#-quick-start)** &#8226;
**[Evaluation](#-evaluation-highlights)** &#8226;
**[Docs](#-documentation)**

</div>

---

## The Problem

> *"The brain does not simply store memories; it actively reconstructs them."*
> &mdash; **Bartlett, 1932** <sup>[7]</sup>

Current Large Language Models have a fundamental cognitive deficit. They operate with **fixed context windows** and **static weights** &mdash; the computational equivalent of a brilliant mind with amnesia:

| What LLMs Lack | What Happens | What Humans Do |
| :--- | :--- | :--- |
| Dynamic knowledge updates | Stale information persists | Continuously revise beliefs |
| Graceful forgetting | Context bloat, noise accumulation | Actively prune irrelevant traces |
| Episodic &rarr; semantic consolidation | All memories treated equally | Distill experiences into knowledge |
| Latent constraint tracking | Safety-critical context is lost | "I can't eat that &mdash; I'm allergic" |

The **Cognitive Memory Layer** bridges this gap by implementing how human memory *actually* works &mdash; not as a simple database, but as a living, reconstructive system grounded in decades of neuroscience research.

---

## How Your Brain Solves It

CML's architecture is a direct computational translation of three foundational theories from cognitive neuroscience. Each maps to a specific subsystem in the codebase.

### Theory 1: The Multi-Store Model <sup>[Atkinson & Shiffrin, 1968]</sup>

Human memory is not a single store &mdash; it is a **pipeline**. Information flows through distinct stages, each with different capacity, encoding, and duration:

```
                        Attention                    Rehearsal
  Sensory Register  ──────────────►  Short-Term  ──────────────►  Long-Term
  (250ms, huge)                      (20s, ~7 items)              (lifetime, unlimited)
       │                                  │                            │
       └─── Decay                         └─── Displacement            └─── Interference
```

**In CML:** Raw input enters the `SensoryBuffer` (token-level encoding via tiktoken). A `WorkingMemoryManager` enforces the ~7-item capacity limit. `SemchunkChunker` handles semantic segmentation before long-term encoding.

> **Reference**: Miller, G.A. (1956). ["The magical number seven, plus or minus two."](https://doi.org/10.1037/h0043158) *Psychological Review*, 63(2), 81-97.

### Theory 2: Complementary Learning Systems <sup>[McClelland et al., 1995]</sup>

The brain uses **two complementary systems** that learn at fundamentally different speeds &mdash; and this tension is not a bug, it's the architecture:

| System | Brain Region | Speed | What It Stores | Decay |
| :--- | :--- | :--- | :--- | :--- |
| **Hippocampal** | Medial Temporal Lobe | Fast (one-shot) | Rich episodic traces | Rapid |
| **Neocortical** | Distributed Cortex | Slow (gradual) | Distilled semantic knowledge | Stable |

The hippocampus captures today's lunch conversation in full context. Over time &mdash; during sleep &mdash; **sharp-wave ripples** replay these episodes, training the neocortex to extract the durable semantic pattern: *"this person is vegetarian."*

**In CML:** `HippocampalStore` performs one-shot episodic encoding with dense vector embeddings. `NeocorticalStore` maintains structured semantic facts with schema alignment. The `ConsolidationEngine` performs the equivalent of sleep-replay: sampling, clustering, gist extraction, and migration from hippocampal to neocortical stores.

> **Reference**: McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). ["Why there are complementary learning systems in the hippocampus and neocortex."](https://doi.org/10.1037/0033-295X.102.3.419) *Psychological Review*, 102(3), 419-457.

### Theory 3: Reconsolidation <sup>[Nader et al., 2000]</sup>

A discovery that upended memory science: retrieved memories become **temporarily unstable** ("labile") and can be modified before restabilizing. Memory is not read-only &mdash; every retrieval is a potential rewrite.

**In CML:** The `LabileStateTracker` marks retrieved memories as labile for 5 minutes. The `ConflictDetector` identifies contradictions, refinements, and supersessions. The `BeliefRevisionEngine` applies one of 6 strategies (reinforce, correct, time-slice, merge, demote, supersede) before restabilizing.

> **Reference**: Nader, K., Schafe, G.E., & Le Doux, J.E. (2000). ["Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval."](https://doi.org/10.1038/35021052) *Nature*, 406(6797), 722-726.

### Theory 4: Active Forgetting <sup>[Shuai et al., 2010]</sup>

Forgetting is not failure &mdash; it is an **active, protein-mediated process**. The proteins Rac1 and Cofilin actively prune synaptic connections, removing traces that are no longer relevant. This keeps memory efficient and prevents catastrophic interference.

**In CML:** The `ForgettingWorker` runs a relevance scorer every 24h. Memories are triaged into five actions &mdash; Keep, Decay, Silence, Compress, or Delete &mdash; based on importance, recency, frequency, confidence, type, and dependency count.

> **Reference**: Shuai, Y., et al. (2010). ["Forgetting is regulated through Rac activity in Drosophila."](https://doi.org/10.1016/j.cell.2009.12.044) *Cell*, 140(4), 579-589.

---

## How CML Implements It

CML integrates five AI memory research frameworks, mapping each onto a specific capability:

| Framework | Paper | What CML Takes From It |
| :--- | :--- | :--- |
| **CLS Theory** | McClelland et al. (1995) | Dual-store: fast hippocampal + slow neocortical |
| **HippoRAG** | Gutierrez et al. (2024) <sup>[8]</sup> | Knowledge graph + Personalized PageRank for multi-hop retrieval |
| **HawkinsDB** | Based on Thousand Brains Theory | 15 cognitive memory types with biological decay profiles |
| **Mem0** | mem0.ai (2024) <sup>[9]</sup> | Add/Update/Delete/No-change ops + belief revision |
| **LoCoMo-Plus** | Li et al. (2024) <sup>[14]</sup> | Level-2 cognitive constraint evaluation (goal/value/state/causal/policy) |

### The Cognitive Constraint Layer

This is what makes CML different from a vector database with extra steps. Standard RAG retrieves text that *looks similar*. CML retrieves **latent constraints** &mdash; goals, values, policies, states, and causal rules &mdash; even when the query is semantically distant from the constraint:

```
User stored:    "I'm deathly allergic to shellfish"      (cue)
User asks:      "Recommend a restaurant for tonight"      (trigger)
                 ↑                                         ↑
           These are semantically distant &mdash; cosine similarity is low.
           But the constraint is CRITICAL to the response.
```

CML solves this through **multi-path retrieval**: vector search + knowledge graph traversal + structured constraint lookup + domain-aware rescoring. Constraints are extracted at write time, stored with type metadata (goal/value/policy/state/causal), and surfaced at retrieval time regardless of surface-level semantic similarity.

---

## Architecture

### System Overview

```mermaid
flowchart LR
    C[Client / SDK] --> API[FastAPI API]
    API --> ORCH[MemoryOrchestrator]

    ORCH --> WRITE[Write Path]
    ORCH --> READ[Read Path]
    ORCH --> BG[Background Workers]

    WRITE --> STM[Short-term chunking]
    WRITE --> GATE[Write gate + PII redaction]
    WRITE --> EXT[Extraction]
    EXT -->|LLM enabled| LLM[LLM_INTERNAL unified extractor]
    EXT -->|LLM disabled| HEUR[Regex heuristics]

    READ --> CLS2[Query classifier]
    READ --> PLAN[Retrieval planner]
    READ --> RET[Hybrid retriever]
    READ --> RER[Memory reranker]

    RET --> PG[(Postgres + pgvector)]
    RET --> NEO[(Neo4j)]
    RET --> FACT[(Semantic facts)]

    BG --> CONS[Consolidation]
    BG --> RECON[Reconsolidation]
    BG --> FORGET[Forgetting]
```

### Data Flow: Write &rarr; Store &rarr; Read

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F8FAFC', 'primaryTextColor': '#0F172A', 'primaryBorderColor': '#475569', 'lineColor': '#64748B', 'fontSize': '13px'}}}%%
flowchart LR
    classDef write fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e40af
    classDef process fill:#fff7ed,stroke:#ea580c,stroke-width:2px,color:#9a3412
    classDef store fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87
    classDef read fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#065f46

    subgraph Write ["WRITE PATH"]
        direction TB
        W1["Input Text"]:::write
        W2["Sensory Buffer"]:::write
        W3["Semantic Chunker"]:::write
        W4["Write Gate"]:::process
        W5["PII Redactor"]:::process
        W6["Batch Embedder"]:::process
        W7["Unified Extractor"]:::process
        W1 --> W2 --> W3 --> W4 --> W5
        W5 --> W6
        W5 --> W7
    end

    subgraph Store ["STORAGE"]
        direction TB
        S1[("Postgres\npgvector")]:::store
        S2[("Neo4j\nGraph")]:::store
        S3[("Redis\nCache")]:::store
    end

    subgraph Read ["READ PATH"]
        direction TB
        R1["Query"]:::read
        R2["Classifier"]:::read
        R3["Planner"]:::read
        R4["Hybrid Retriever"]:::read
        R5["Reranker"]:::read
        R6["Memory Packet"]:::read
        R1 --> R2 --> R3 --> R4 --> R5 --> R6
    end

    W6 --> S1
    W7 --> S1
    W7 --> S2
    W7 --> S3
    S1 --> R4
    S2 --> R4
    S3 --> R4
```

### Neuroscience &rarr; Code Mapping

Every module in CML corresponds to a specific biological mechanism. Click to explore each:

<details>
<summary><strong>Sensory & Working Memory</strong> &mdash; Miller's 7&plusmn;2 capacity limit</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Sensory register | `SensoryBuffer` (tiktoken token-ID storage) | `src/memory/sensory/buffer.py` |
| Working memory (7&plusmn;2) | `WorkingMemoryManager` (max=10) + `BoundedStateMap` | `src/memory/working/manager.py` |
| Semantic chunking | `SemchunkChunker` (Hugging Face tokenizer) | `src/memory/working/chunker.py` |

</details>

<details>
<summary><strong>Encoding Gate</strong> &mdash; CREB/Npas4 neuronal selection</summary>

Not all experiences become memories. CML's `WriteGate` mirrors the CREB protein's role in selecting which neurons participate in engram formation (Han et al., 2007 <sup>[5]</sup>).

| Biology | Code | Location |
| :--- | :--- | :--- |
| CREB allocation | `WriteGate.evaluate()` &mdash; salience, novelty, risk | `src/memory/hippocampal/write_gate.py` |
| PII redaction | `PIIRedactor` strips sensitive data before storage | `src/memory/hippocampal/redactor.py` |
| Constraint boost | Constraint chunks get `importance += 0.2` | `WriteGateConfig` |

</details>

<details>
<summary><strong>Hippocampal Store</strong> &mdash; One-shot episodic encoding with pattern separation</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| One-shot encoding | `HippocampalStore.encode_batch()` | `src/memory/hippocampal/store.py` |
| Pattern separation | SHA256 stable keys + unique embeddings | `PostgresMemoryStore` |
| Unified extraction | Entities, relations, constraints, facts in one call | `src/extraction/unified_write_extractor.py` |

</details>

<details>
<summary><strong>Neocortical Store</strong> &mdash; Slow semantic learning with schema alignment</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Schema-based storage | `FactSchema` + `FactCategory` | `src/memory/neocortical/schemas.py` |
| Cognitive categories | GOAL, STATE, VALUE, CAUSAL, POLICY | `src/memory/neocortical/schemas.py` |
| Graph traversal | Personalized PageRank on Neo4j | `src/storage/neo4j.py` |

</details>

<details>
<summary><strong>Retrieval</strong> &mdash; Tulving's ecphory (cue-engram interaction)</summary>

Memory retrieval is not lookup &mdash; it is **ecphory**: the interaction between a retrieval cue and a stored engram that *reconstructs* the memory (Tulving, 1983 <sup>[2]</sup>).

| Biology | Code | Location |
| :--- | :--- | :--- |
| Query classification | `QueryClassifier` (10 intents) | `src/retrieval/classifier.py` |
| Retrieval planning | `RetrievalPlanner` (parallel step groups) | `src/retrieval/planner.py` |
| Hybrid search | Vector + Graph + Constraints + Facts | `src/retrieval/retriever.py` |
| Constraint-aware reranking | Type-stability weighting, domain rescoring | `src/retrieval/reranker.py` |

</details>

<details>
<summary><strong>Consolidation</strong> &mdash; Sleep-cycle replay (sharp-wave ripples)</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Episode sampling | `EpisodeSampler` (7d episodes, 90d constraints) | `src/consolidation/sampler.py` |
| Semantic clustering | `SemanticClusterer` | `src/consolidation/clusterer.py` |
| Gist extraction | `GistSummarizer` (preserves constraint types) | `src/consolidation/summarizer.py` |
| Migration | `ConsolidationMigrator` (hippo &rarr; neocortex) | `src/consolidation/migrator.py` |

</details>

<details>
<summary><strong>Active Forgetting</strong> &mdash; Rac1/Cofilin synaptic pruning</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Relevance scoring | `ForgettingScorer` (6-factor composite) | `src/forgetting/scorer.py` |
| Interference | `InterferenceDetector` | `src/forgetting/interference.py` |
| Five actions | Keep / Decay / Silence / Compress / Delete | `src/forgetting/actions.py` |

</details>

### Memory Types

CML supports 15 memory types, each with a biological analog and distinct decay profile:

| Type | Description | Decay |
| :--- | :--- | :--- |
| `episodic_event` | A specific personal event anchored in time/place. | Fast |
| `semantic_fact` | A factual statement about the world or domain knowledge. | Slow |
| `procedure` | Step-by-step instructions or process knowledge. | Stable |
| `constraint` | A hard/soft rule, policy, must/never condition. | Stable |
| `hypothesis` | A tentative explanation or guess. | Confirm |
| `preference` | A stable like/dislike or personal choice. | Medium |
| `task_state` | Current progress/status of a task. | Very Fast |
| `conversation` | General conversational turn with little durable content. | Session |
| `message` | Message-like communication content. | Session |
| `tool_result` | Output or observation from a tool/API/query. | Task |
| `reasoning_step` | Intermediate reasoning or derivation step. | Session |
| `scratch` | Temporary short-term note. | Fast |
| `knowledge` | Domain information suitable for long-term memory. | Stable |
| `observation` | Observed condition or signal. | Session |
| `plan` | Future actions or strategy. | Task |

Types are defined in `src/core/enums.py` as `MemoryType`. **Decay**: Fast/Very Fast = short-lived; Slow/Stable = long-lived (higher retention); Session = per conversation; Task = per task; Confirm = until confirmed or rejected. **Implementation**: type assignment at write in `src/memory/hippocampal/write_gate.py` (`_determine_memory_types()`); retention by type in `src/forgetting/scorer.py` (`ScorerConfig.type_bonuses`).

### Extraction & Enrichment

`FEATURES__USE_LLM_ENABLED` selects between two write paths. There is one intelligence
path (the LLM) and one deterministic fallback; there is no third model tier.

<details>
<summary><strong>LLM enabled</strong> &mdash; the unified write-path extractor (recommended)</summary>

A single call to `LLM_INTERNAL` per chunk returns everything the write path needs:
entities, relations, constraints, write-time facts, PII spans, memory type, importance,
salience, confidence, context tags and decay rate. See
`src/extraction/unified_write_extractor.py`.

Read-path enrichment (query classification, conflict detection, constraint supersession,
consolidation gists, forgetting compression) uses the same client.

</details>

<details>
<summary><strong>LLM disabled</strong> &mdash; deterministic heuristics</summary>

| Signal | Heuristic | Location |
| :--- | :--- | :--- |
| Write-time facts | Regex families (preference / identity / location / occupation) | `src/extraction/write_time_facts.py` |
| Constraints | Regex patterns + chunk-type mapping, 5 cognitive types | `src/extraction/constraint_extractor.py` |
| PII redaction | Regex pattern table | `src/memory/hippocampal/redactor.py`, `src/utils/ner.py` |
| Novelty | Jaccard word overlap | `src/memory/hippocampal/write_gate.py` |
| Importance | Upstream chunk salience | `src/memory/hippocampal/write_gate.py` |
| Duplicate detection | Embedding cosine + Jaccard | `src/forgetting/interference.py` |
| Forgetting policy | Threshold chain + safety guard (never discards depended-on, important, or hot-recent memories) | `src/forgetting/scorer.py` |
| Entities / relations | Not extracted — the graph stays empty in this mode | — |

Safety-critical paths (secret detection, the PII regex baseline, deterministic key
generation) are never delegated to a model in either mode.

</details>

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/avinash-mall/CognitiveMemoryLayer.git
cd CognitiveMemoryLayer
cp .env.minimal .env          # minimal config — works out of the box

# 2. Start everything (GPU auto-detected)
./docker/up.sh up -d api

# 3. Verify
curl http://localhost:8000/api/v1/health
# {"status":"healthy", ...}
```

`./docker/up.sh` wraps `docker compose` and automatically applies the GPU override when `nvidia-smi` is available. The `api` service starts Postgres, Neo4j, Redis, runs migrations, and serves the API — all in one command. Models are downloaded from HuggingFace Hub automatically on first start.

<details>
<summary><strong>Python SDK (no Docker)</strong></summary>

```bash
pip install -e ".[server,dev]"
cp .env.minimal .env
./docker/up.sh up -d postgres neo4j redis
alembic upgrade head
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

</details>

<details>
<summary><strong>Python SDK</strong></summary>

```bash
pip install cognitive-memory-layer
```

```python
from cml import CognitiveMemoryLayer

memory = CognitiveMemoryLayer(base_url="<your-cml-url>", api_key="your-key")
memory.write(content="I never eat shellfish - severe allergy.", tenant_id="demo")
response = memory.read(query="Recommend a restaurant", tenant_id="demo")
# Constraints section will include the shellfish allergy ^
```

Sync, async, and embedded (SQLite) modes. See [SDK docs](packages/py-cml/docs/).

</details>

---

## Getting a Runnable Project

This section covers the **full setup from scratch** — cloning the code from GitHub, downloading the 25 GB of trained model weights from Hugging Face Hub, configuring infrastructure, and verifying everything works.

### Prerequisites

| Requirement | Minimum version | Notes |
| :--- | :--- | :--- |
| Python | 3.11+ | 3.14 recommended |
| Docker + Compose | 24+ | For Postgres, Neo4j, Redis |
| `uv` or `pip` | any | `uv` strongly recommended for speed |
| `huggingface-cli` | 0.19+ | `pip install huggingface_hub[cli]` |
| Disk space | ~30 GB free | ~25 GB models + repo + venv |
| RAM | 8 GB+ | 16 GB recommended for full model pack |

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/avinash-mall/CognitiveMemoryLayer.git
cd CognitiveMemoryLayer
```

> The repository contains only **code, configs, and tokenizer metadata** (~100 MB). Model weights are stored separately on Hugging Face Hub (see Step 3).

---

### Step 2 — Set Up the Python Environment

**With uv (recommended):**

```bash
pip install uv
uv venv .venv --python 3.11
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[server,dev]"
```

**With pip:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[server,dev]"
```

This installs the CML server, the `py-cml` client SDK, all runtime extras (FastAPI, pgvector, sentence-transformers, etc.), and dev tools (pytest, ruff, mypy).

---

### Step 3 — Model Weights

CML no longer ships trained task models — the internal LLM handles extraction and
classification (see `FEATURES__USE_LLM_ENABLED`). Three model artifacts are still
downloaded on demand from public registries the first time the write path runs:

| Artifact | Source | Configured by |
|---|---|---|
| Embedding model (`nomic-ai/nomic-embed-text-v2-moe`, revision-pinned) | HuggingFace | `EMBEDDING_INTERNAL__LOCAL_MODEL` |
| Chunker tokenizer (`google/flan-t5-base`) | HuggingFace | `CHUNKER__TOKENIZER` |
| tiktoken BPE ranks (`cl100k_base`) | openaipublic.blob.core.windows.net | — |

In Docker these land on the `hf-cache` and `tiktoken-cache` volumes, so they are
downloaded once and survive `docker compose up --force-recreate`.

> **Note:** the embedding model is loaded with `trust_remote_code=True`, so the first
> download also fetches and **executes** the model's own Python from HuggingFace. It is
> pinned to a specific revision. Set `EMBEDDING_INTERNAL__PROVIDER` to `mock`, `openai`,
> `ollama`, or `vllm` if you would rather not run remote code.

**Air-gapped / fully offline runs.** Warm the caches once on a networked machine, then
pin them shut:

```bash
docker compose -f docker/docker-compose.yml up -d api        # first run populates the volumes
# ...then for subsequent runs, in .env:
HF_HUB_OFFLINE=1
```

With the volumes warm and `HF_HUB_OFFLINE=1` set, the server makes no outbound requests
other than to your configured LLM endpoint.

---

### Step 4 — Configure the Environment

**Minimal (recommended to get started):**

```bash
cp .env.minimal .env
```

`.env.minimal` works out of the box — it uses local GPU embeddings, connects to the Docker-managed Postgres/Neo4j/Redis, and requires no external API keys. Edit it only if you want to enable LLM-assisted extraction or change the default models.

**Full config** (for production or custom setups):

```bash
cp .env.example .env
```

Key variables to fill in:

```dotenv
# ── Required ──────────────────────────────────────────────────
DATABASE__POSTGRES_URL=postgresql+asyncpg://cml:cml@localhost:5432/cml
AUTH__API_KEY=your-api-key-here          # used by clients
AUTH__ADMIN_API_KEY=your-admin-key-here  # dashboard + admin routes

# ── Optional (enables LLM-assisted extraction) ─────────────────
LLM_INTERNAL__PROVIDER=openai_compatible
LLM_INTERNAL__BASE_URL=http://localhost:8001/v1
LLM_INTERNAL__MODEL=your-model-name

# ── Optional (Hugging Face token for private model downloads) ──
HF_TOKEN=hf_...
```

---

### Step 5 — Start All Services

```bash
# Starts Postgres, Neo4j, Redis, runs migrations, and serves the API.
# GPU is auto-detected via nvidia-smi — no flags needed on GPU hosts.
./docker/up.sh up -d api
```

That single command is all you need. `docker/up.sh` wraps `docker compose` and automatically adds `-f docker/docker-compose.gpu.yml` when `nvidia-smi` is available on the host, giving the API container access to the GPU for local embeddings. You can override detection with `GPU=1 ./docker/up.sh ...` (force GPU) or `GPU=0 ./docker/up.sh ...` (force CPU-only).

Verify all services are running:

```bash
./docker/up.sh ps
```

<details>
<summary><strong>Infrastructure only (run the server outside Docker)</strong></summary>

```bash
# Start only Postgres, Neo4j, Redis
./docker/up.sh up -d postgres neo4j redis

# Run migrations and start the server locally
alembic upgrade head
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

</details>

---

### Step 6 — Verify It Works

**Health check:**

```bash
curl http://localhost:8000/api/v1/health
# {"status":"healthy", ...}
```

**Write and read a memory:**

```bash
# Write
curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{"content": "I never eat shellfish — severe allergy.", "tenant_id": "demo"}'

# Read
curl -X POST http://localhost:8000/api/v1/memory/read \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Recommend a restaurant for tonight", "tenant_id": "demo"}'
```

**Admin dashboard:** open [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/) and authenticate with `AUTH__ADMIN_API_KEY` (default `test-key` from `.env.minimal`).


---

### Step 7 — Run the Test Suite

```bash
pytest tests/unit -v --tb=short          # unit tests, hermetic (no DB/LLM needed)
pytest tests/integration -v --tb=short  # 88 integration tests (requires running stack)
pytest tests/e2e -v                     # 5 end-to-end API tests
```

---

### Artifact Locations Summary

| Artifact | Source | Local Path |
| :--- | :--- | :--- |
| Code & configs | [GitHub](https://github.com/avinash-mall/CognitiveMemoryLayer) | `./` (repo root) |
| Python client SDK | [PyPI](https://pypi.org/project/cognitive-memory-layer/) | `pip install cognitive-memory-layer` |

---

## Evaluation Highlights

Evaluated on **LoCoMo-Plus** (2,387 samples, LLM-as-judge) &mdash; the first benchmark that tests *cognitive* memory (constraints, beliefs, causal reasoning), not just factual recall. CML uses a **fully local** `google/gemma-4-31b-it` model via vLLM &mdash; **zero API dependency**, zero per-query cost.

| Category | CML (local 31B) | GPT-4o (full ctx) | Mem0 (GPT-4o) | A-Mem (GPT-4o) |
| :--- | :--- | :--- | :--- | :--- |
| **Adversarial** | **64.80%** | 48.99% | 30.50% | 35.20% |
| **Temporal** | **48.60%** | 45.79% | 39.40% | 49.30% |
| **Single-hop** | 56.96% | 78.13% | 80.20% | 76.90% |
| **Overall** | **48.58%** | 62.99% | 57.24% | 59.64% |

> **Adversarial robustness:** CML more than doubles Mem0's adversarial score (64.80% vs 30.50%) and beats GPT-4o full-context by +15.81% &mdash; using a local 31B model vs closed-source APIs. This demonstrates that CML's constraint-aware retrieval architecture provides real robustness, independent of model size.

> **Temporal reasoning:** CML outperforms GPT-4o full-context (48.60% vs 45.79%) through explicit timestamp handling and temporal context in retrieval.

Full results &amp; competitor analysis: [evaluation/EVALUATION_REPORT.md](evaluation/EVALUATION_REPORT.md) &#8226; Run: `cml-eval run-full --repo-root .` (or legacy: `python evaluation/scripts/run_full_eval.py`)

---

## Testing

```bash
pytest tests/unit -v --tb=short        # 812 unit tests
pytest tests/integration -v --tb=short  # 88 integration tests
pytest tests/e2e -v                     # 5 end-to-end API tests
```

Full-stack quality test with LLM-as-judge: `python scripts/test_memory_quality.py`

Details: [tests/README.md](tests/README.md)

---

## Admin Dashboard

CML ships a built-in admin dashboard at **[http://localhost:8000/dashboard/](http://localhost:8000/dashboard/)** (or your configured host/port). It provides real-time observability and management across all subsystems with no external dependencies &mdash; a vanilla JS single-page application served by the API itself.

### Pages

| Page | What It Shows |
| :--- | :--- |
| **Overview** | KPI cards (memories, facts, events, tenants), request sparkline, reconsolidation queue status |
| **Tenants** | Per-tenant memory/fact/event counts, last activity, quick-link filters |
| **Sessions** | Active Redis sessions with TTL, memory counts per session |
| **Memory Explorer** | Searchable memory list with bulk actions (archive, silence, delete), JSON export, inline write panel, detail view with lineage chain |
| **Facts Explorer** | Semantic facts with category/tenant filters, current-only toggle, inline invalidation, JSON export |
| **Knowledge Graph** | Interactive neovis.js graph with entity search, depth control, edge labels |
| **Events** | Event log with type/operation/tenant filters, auto-refresh, JSON export |
| **API Usage** | Rate-limit buckets, hourly request volume chart |
| **Components** | PostgreSQL, Neo4j, Redis health with latency; Embedding model info (provider, model, dimensions, batch size); Server info (version, Python, workers) |
| **Retrieval Test** | Interactive query tool with scored results, supersession badges, lineage links |
| **Configuration** | Live config viewer with inline editing for safe settings (embedding batch size, rate limits, retrieval tuning, feature flags) |
| **Management** | Consolidation, forgetting, reconsolidation triggers with job history |

### Key Features

- **Memory write from UI** &mdash; collapsible panel on Memory Explorer to write memories with content, session ID, namespace, memory type, and metadata fields
- **Export** &mdash; one-click JSON export for memories, events, and semantic facts (with optional tenant filter)
- **Infrastructure visibility** &mdash; Components page shows embedding model config (provider, model, dimensions, batch size, device) and server info (CML version, Python version, uvicorn workers) alongside database health
- **Live config editing** &mdash; edit embedding batch size, rate limits, retrieval parameters, and feature flags directly from the dashboard (changes persist to `.env`)

### Authentication

The dashboard requires admin authentication via `AUTH__ADMIN_API_KEY`. Enter the key on first visit; it is stored in browser localStorage. State-changing requests require a CSRF header (`X-Requested-With: XMLHttpRequest`), which the dashboard sends automatically.

See [Usage Documentation](docs/usage.md) for full API details.

---

## Documentation

| | Document | Description |
|---|----------|-------------|
| **API** | [Usage & API Reference](docs/usage.md) | Full API, config, dashboard, runtime modes |
| **SDK** | [Python SDK](packages/py-cml/docs/README.md) | Getting started, API reference, examples |
| **Eval** | [Evaluation](evaluation/README.md) | LoCoMo-Plus harness, scripts, comparison |
| **SDK Eval** | [Eval Module](packages/py-cml/docs/evaluation.md) | `cml-eval` CLI, Python API, typed configs |
| **Dev** | [Contributing](CONTRIBUTING.md) | Setup, code standards, PR process |
| **Changelog** | [Release History](CHANGELOG.md) | Version history |
| **Roadmap** | [Future Plans](ProjectPlan/ActiveCML/) | Intrinsic memory integration phases (designed, not started) |

---

## Future Roadmap

The next frontier: **intrinsic memory** &mdash; injecting memories directly into the LLM's computational graph (steering vectors, KV-cache manipulation, logit biases) instead of context-window stuffing.

See [ProjectPlan/ActiveCML/](ProjectPlan/ActiveCML/) for the 10-phase roadmap index. None of it is implemented — the detailed specs were compressed into that index and remain in git history.

---

## References

<details open>
<summary><strong>Neuroscience Foundations</strong></summary>

1. McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). ["Why there are complementary learning systems in the hippocampus and neocortex."](https://doi.org/10.1037/0033-295X.102.3.419) *Psychological Review*, 102(3), 419-457.
2. Tulving, E. (1983). ["Elements of Episodic Memory."](https://books.google.com/books/about/Elements_of_episodic_memory.html?id=3nQ6AAAAMAAJ) Oxford University Press.
3. Nader, K., Schafe, G.E., & Le Doux, J.E. (2000). ["Fear memories require protein synthesis in the amygdala for reconsolidation."](https://doi.org/10.1038/35021052) *Nature*, 406(6797), 722-726.
4. Shuai, Y., et al. (2010). ["Forgetting is regulated through Rac activity in Drosophila."](https://doi.org/10.1016/j.cell.2009.12.044) *Cell*, 140(4), 579-589.
5. Han, J.H., et al. (2007). ["Neuronal competition and selection during memory formation."](https://doi.org/10.1126/science.1139438) *Science*, 316(5823), 457-460.
6. Miller, G.A. (1956). ["The magical number seven, plus or minus two."](https://doi.org/10.1037/h0043158) *Psychological Review*, 63(2), 81-97.
7. Bartlett, F.C. (1932). ["Remembering: A Study in Experimental and Social Psychology."](https://archive.org/details/rememberingstudy00bart) Cambridge University Press.
12. Rasch, B., & Born, J. (2013). ["About sleep's role in memory."](https://doi.org/10.1152/physrev.00032.2012) *Physiological Reviews*, 93(2), 681-766.
13. Atkinson, R.C. & Shiffrin, R.M. (1968). "Human memory: A proposed system and its control processes." *Psychology of Learning and Motivation*, 2, 89-195.

</details>

<details open>
<summary><strong>AI Memory Frameworks</strong></summary>

8. HippoRAG (2024). ["Neurobiologically Inspired Long-Term Memory for Large Language Models."](https://arxiv.org/abs/2405.14831) *arXiv:2405.14831*.
9. Mem0 (2024). ["Mem0: The Memory Layer for AI Applications."](https://github.com/mem0ai/mem0) GitHub.
10. HawkinsDB (2024). ["HawkinsDB: A Thousand Brains Theory inspired Database."](https://github.com/harishsg993010/HawkinsDB) GitHub.
11. Wu, T., et al. (2024). ["From Human Memory to AI Memory: A Survey."](https://arxiv.org/abs/2404.15965) *arXiv:2404.15965*.
14. Li, Y., et al. (2024). ["Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation."](https://arxiv.org/abs/2602.10715) *arXiv:2602.10715*.

</details>

---

<div align="center">

*"Memory is the diary that we all carry about with us."* &mdash; Oscar Wilde

**CML transforms that diary into a computational system that learns, consolidates, and gracefully forgets &mdash; just like we do.**

<br/>

![GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-A42E2B?style=for-the-badge&logo=gnu&logoColor=white)

</div>
