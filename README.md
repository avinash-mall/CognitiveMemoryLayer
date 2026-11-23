

<div align="center">

# Cognitive Memory Layer

### A Neuro-Inspired Memory System for AI

*Store. Retrieve. Consolidate. Forget.*
*Just like the human brain.*

<br/>

[![Quick Start](https://img.shields.io/badge/Quick%20Start-5%20min-success?style=for-the-badge&logo=rocket)](#-quick-start)
[![Docs](https://img.shields.io/badge/Docs-Full%20API-blue?style=for-the-badge&logo=gitbook)](./docs/usage.md)
[![Tests](https://img.shields.io/badge/Tests-1227-brightgreen?style=for-the-badge&logo=pytest)](./tests/README.md)
[![Version](https://img.shields.io/badge/version-2.0.0-blue?style=for-the-badge)](#)

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

**In CML:** Raw input enters the `SensoryBuffer` (token-level encoding via tiktoken, with time- and capacity-based decay). A `WorkingMemoryManager` enforces a bounded capacity (10 chunks, protecting recent items and evicting by salience) and operates as a **Baddeley Episodic Buffer** (`FEATURES__WORKING_MEMORY_CONTEXT_BINDING_ENABLED=true`), binding cross-turn coreference/topic anchors and exposing a zero-latency working-memory overlay on reads. `SemchunkChunker` handles semantic segmentation before long-term encoding.

> **Reference**: Miller, G.A. (1956). ["The magical number seven, plus or minus two."](https://doi.org/10.1037/h0043158) *Psychological Review*, 63(2), 81-97.

### Theory 2: Complementary Learning Systems <sup>[McClelland et al., 1995]</sup>

The brain uses **two complementary systems** that learn at fundamentally different speeds &mdash; and this tension is not a bug, it's the architecture:

| System | Brain Region | Speed | What It Stores | Decay |
| :--- | :--- | :--- | :--- | :--- |
| **Hippocampal** | Medial Temporal Lobe | Fast (one-shot) | Rich episodic traces | Rapid |
| **Neocortical** | Distributed Cortex | Slow (gradual) | Distilled semantic knowledge | Stable |

The hippocampus captures today's lunch conversation in full context. Over time &mdash; during sleep &mdash; **sharp-wave ripples** replay these episodes, training the neocortex to extract the durable semantic pattern: *"this person is vegetarian."*

**In CML:** `HippocampalStore` performs one-shot episodic encoding with dense vector embeddings. `NeocorticalStore` maintains structured semantic facts, routed to a schema key on write. During consolidation (`ConsolidationWorker`), salience-weighted Sharp-Wave Ripple (SWR) replay samples episodes, induces cross-episode associative links (`swr_links_induced`), clusters them, extracts gists, and migrates durable knowledge to the neocortical store. The background scheduler (`FEATURES__CONSOLIDATION_SCHEDULER_ENABLED`) is off by default for predictable latency and zero surprise background load, and can be enabled via config or triggered on demand via the admin API and dashboard.

> **Reference**: McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). ["Why there are complementary learning systems in the hippocampus and neocortex."](https://doi.org/10.1037/0033-295X.102.3.419) *Psychological Review*, 102(3), 419-457.

### Theory 3: Reconsolidation <sup>[Nader et al., 2000]</sup>

A discovery that upended memory science: retrieved memories become **temporarily unstable** ("labile") and can be modified before restabilizing. Memory is not read-only &mdash; every retrieval is a potential rewrite.

**In CML:** The `LabileStateTracker` and `ReconsolidationService` mark retrieved memories labile in both Redis and PostgreSQL (`labile_state`), detect conflicts via `ConflictDetector`, and apply `BeliefRevisionEngine` updates (`REINFORCE`, `TIME_SLICE`, `MERGE`, `ADD_HYPOTHESIS`) while recording belief-evolution lineage before restabilizing.

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
| Bounded working memory &amp; Baddeley Episodic Buffer | `WorkingMemoryManager` + `BoundedStateMap` (cross-turn coreference/topic binding) | `src/memory/working/manager.py` |
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
| Deduplication (*not* pattern separation) | SHA256 stable keys — see note | `PostgresMemoryStore` |
| Unified extraction | Entities, relations, constraints, facts in one call | `src/extraction/unified_write_extractor.py` |

The SHA256 key is a deduplication device, not a dentate-gyrus analog. Any one-character
difference yields an unrelated key, so it gives maximal separation with no similarity
structure at all. Near-duplicate suppression is done separately, by the write gate's
lexical novelty check.

</details>

<details>
<summary><strong>Neocortical Store</strong> &mdash; Slow semantic learning with schema alignment</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Schema-based storage | `FactSchema` + `FactCategory` | `src/memory/neocortical/schemas.py` |
| Cognitive categories | GOAL, STATE, VALUE, CAUSAL, POLICY | `src/memory/neocortical/schemas.py` |
| Graph traversal | Personalized PageRank on Neo4j (requires the GDS plugin; otherwise a variable-length path-count fallback) | `src/storage/neo4j.py` |

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
| Salience-weighted SWR replay &amp; sampling | `EpisodeSampler` + `ConsolidationWorker` (`swr_links_induced`) | `src/consolidation/sampler.py`, `src/consolidation/worker.py` |
| Semantic clustering | `SemanticClusterer` | `src/consolidation/clusterer.py` |
| Gist extraction | `GistExtractor` (preserves constraint types + detail recovery) | `src/consolidation/summarizer.py` |
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

<details>
<summary><strong>Neuro-Cognitive Subsystems &amp; Laya System 1</strong> &mdash; Amygdala, BA10, CA1, DMN, Subcortical Fast-Path</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Amygdala somatic markers | `SomaticMarkerTagger` (valence/arousal, flashbulb decay modulation, state-congruent resonance) | `src/memory/somatic.py` |
| Rostral PFC (BA10) prospective memory | `ProspectiveMemoryEngine` (causal intention triggers + read-time parent ecphory) | `src/memory/prospective.py` |
| CA1 comparator &amp; OFC reality monitoring | `CA1RealityMonitor` (presupposition checks, speaker-attribution binding, concise refusal alerts) | `src/retrieval/reality_monitor.py` |
| Default Mode Network (mPFC / TPJ) | `DMNTheoryOfMind` (mentalizing &amp; motivational synthesis for cognitive/common-sense queries) | `src/memory/dmn.py` |
| Subcortical System 1 fast decision layer | `LayaSystem1Engine` (Convai Laya ModernBERT-large ~20ms GPU routing + calibrated CA1 grounding) | `src/core/laya_system1.py` |
| ActiveCML intrinsic memory | `IntrinsicMemoryController` (logit bias, `contextvars`-isolated activation steering, KV-prefix, LoRA routing) | `src/intrinsic/` |

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

`FEATURES__USE_LLM_ENABLED` (default: **`true`**) selects between two write paths. There is one intelligence
path (the LLM) and one deterministic fallback (`FEATURES__USE_LLM_ENABLED=false`); there is no third model tier. If a cloud provider (`openai`, `gemini`, `claude`) is configured without an API key, CML logs a warning and falls back gracefully to deterministic heuristics.

<details>
<summary><strong>LLM enabled (default)</strong> &mdash; the unified write-path extractor</summary>

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
<summary><strong>Local Server (run API outside Docker)</strong></summary>

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

This section covers the **full setup from scratch** — cloning the repository, installing dependencies, configuring infrastructure, and verifying everything works.

### Prerequisites

| Requirement | Minimum version | Notes |
| :--- | :--- | :--- |
| Python | 3.11+ | 3.13 recommended |
| Docker + Compose | 24+ | For Postgres, Neo4j, Redis |
| `uv` or `pip` | any | `uv` strongly recommended for speed |
| Disk space | ~6 GB free | Repo + venv + on-demand HF embeddings/tokenizer (~2–3 GB) |
| RAM | 8 GB+ | 16 GB recommended (local embeddings use ~3 GB) |

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/avinash-mall/CognitiveMemoryLayer.git
cd CognitiveMemoryLayer
```

> The repository contains **code, configs, and evaluation datasets** (~65 MB). Small runtime artifacts (embedding model, tokenizer, and optional Laya System 1 model) are downloaded on demand on first start (see Step 3).

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
| System 1 decision model (`convaiinnovations/laya`, optional) | HuggingFace | `FEATURES__LAYA_SYSTEM1_ENABLED`, `FEATURES__LAYA_MODEL_ID` |
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
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"content": "I never eat shellfish — severe allergy."}'

# Read
curl -X POST http://localhost:8000/api/v1/memory/read \
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Recommend a restaurant for tonight"}'
```

Authentication is the `X-API-Key` header, not `Authorization: Bearer`. The tenant is
resolved from the API key — a `tenant_id` in the request body is **ignored**, so a
call that passes one there silently reads the default tenant. Admin keys can select a
tenant with the `X-Tenant-Id` header.

**Admin dashboard:** open [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/) and authenticate with `AUTH__ADMIN_API_KEY` (default `test-key` from `.env.minimal`).


---

### Step 7 — Run the Test Suite

```bash
pytest tests/unit -q                                        # hermetic (no DB/LLM needed)
pytest tests/integration tests/e2e packages/py-cml/tests -q # requires a running stack
```

---

### Artifact Locations Summary

| Artifact | Source | Local Path |
| :--- | :--- | :--- |
| Code & configs | [GitHub](https://github.com/avinash-mall/CognitiveMemoryLayer) | `./` (repo root) |
| Python client SDK | [PyPI](https://pypi.org/project/cognitive-memory-layer/) | `pip install cognitive-memory-layer` |

---

## Evaluation Highlights

Evaluated **from scratch on the complete LoCoMo-Plus benchmark (2,387 samples, 0 errors, `$0` API cost)** &mdash; 1,986 original LoCoMo factual questions across 5 categories plus 401 **Cognitive** cue&ndash;trigger constraint instances (Li et al., 2026, [arXiv:2602.10715](https://arxiv.org/abs/2602.10715)).

**CML v2.0 setup (2026-11-23, from scratch):** Fresh Postgres/Neo4j/Redis wipe, all 411 conversations (242,658 turns) ingested with local `nomic-embed-text-v2-moe` embeddings and heuristic write path (`USE_LLM_ENABLED=false` during ingestion, `0` write-time LLM calls); read-time Neuro-Cognitive stack + Cognitive HyDE + LLM Cognitive Cue Selector + 2-hop bridge recollection + speaker persona augmentation + chronological temporal ordering + calibrated CA1 speaker-swap monitoring, powered 100% locally by **`qwen3.8-27b`** (`llama-server`, thinking disabled, `temperature=0`, `$0` API cost) for internal read-time LLM, QA, and LLM-as-Judge. Artifact: [`evaluation/results/locomo_plus_2026-11-23_qwen38_full_2387_summary.json`](evaluation/results/locomo_plus_2026-11-23_qwen38_full_2387_summary.json).

### Full LoCoMo-Plus Comparison (2,387 Samples)

| System / Method | Category | Backbone / QA Model | Single-hop (841) | Multi-hop (282) | Temporal (321) | Common-sense (96) | Adversarial (446) | **LoCoMo Factual Avg (1,986)** | **Cognitive (401)** | **Gap** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CML v2.0 (Neuro + HyDE + CA1)** <sup>[0]</sup> | Memory System | `qwen3.8-27b` (100% local, `$0` API) | 60.52% | 37.77% | **50.16%** | 40.10% | **80.27%** | **59.06%** | 37.41% | 21.65 |
| **Kumiho Cognitive Memory** <sup>[1]</sup> | Memory System | `gpt-4o` (graph-native) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | **93.30%** | &mdash; |
| **Madeleine + HyperMem** <sup>[2]</sup> | Memory System | Involuntary recall + HyperMem | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | 66.60% | &mdash; |
| **HyperMem** <sup>[2]</sup> | Memory System | HyperMem baseline | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | 52.90% | &mdash; |
| **Madeleine (standalone)** <sup>[2]</sup> | Memory System | Zero online LLM calls | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | 52.40% | &mdash; |
| **A-Mem** <sup>[3]</sup> | Memory System | `gpt-4o` | 76.90% | **55.60%** | 49.30% | 68.10% | 35.20% | 59.64% | 42.44% | 17.20 |
| **SeCom** <sup>[3]</sup> | Memory System | `gpt-4o` | 77.60% | 50.90% | 42.30% | **71.40%** | 31.80% | 57.53% | 42.63% | 14.90 |
| **Mem0** <sup>[3]</sup> | Memory System | `gpt-4o` | **80.20%** | 48.10% | 39.40% | 66.20% | 30.50% | 57.24% | 41.44% | 15.80 |
| RAG (`text-embedding-large`) <sup>[3]</sup> | RAG Baseline | `gpt-4o` | 49.76% | 22.78% | 40.00% | 21.35% | 59.73% | 45.32% | 29.77% | 15.55 |
| RAG (`text-embedding-small`) <sup>[3]</sup> | RAG Baseline | `gpt-4o` | 39.17% | 17.79% | 34.69% | 14.61% | 51.90% | 37.23% | 24.94% | 12.29 |
| RAG (`text-ada-002`) <sup>[3]</sup> | RAG Baseline | `gpt-4o` | 40.00% | 16.73% | 37.81% | 15.73% | 49.44% | 37.38% | 23.47% | 13.91 |
| Gemini-2.5-Pro <sup>[3]</sup> | Full Context | Full context (1M) | 77.83% | 52.48% | **73.83%** | 63.54% | 73.03% | **71.78%** | 45.72% | 26.06 |
| Gemini-2.5-Flash <sup>[3]</sup> | Full Context | Full context (1M) | 77.71% | 54.26% | 66.04% | 66.67% | 65.84% | 69.25% | 44.58% | 24.67 |
| Qwen2.5-14B-Instruct <sup>[3]</sup> | Full Context | Full context | 76.33% | 48.23% | 38.94% | 57.29% | 68.09% | 63.45% | 44.21% | 19.24 |
| GPT-4o <sup>[3]</sup> | Full Context | Full context | 78.13% | 52.30% | 45.79% | 69.79% | 48.99% | 62.99% | 41.94% | 21.05 |
| GPT-4.1 <sup>[3]</sup> | Full Context | Full context | 80.30% | 53.90% | 58.88% | 72.92% | 37.30% | 62.21% | 43.58% | 18.63 |
| Qwen3-14B <sup>[3]</sup> | Full Context | Full context | 65.96% | 46.45% | 53.89% | 59.38% | 60.45% | 59.65% | 40.56% | 19.09 |
| GPT-5-nano <sup>[3]</sup> | Full Context | Full context | 75.00% | 54.08% | 50.16% | 73.96% | 17.53% | 54.96% | 40.12% | 14.84 |

<sup>[0]</sup> Full 2,387-sample run from scratch (`2,387/2,387` valid, `0` errors, `0` empty contexts, overall 6-category avg **55.43%**, `1,323.0 / 2,387`); judge: `qwen3.8-27b` (`temperature=0`, thinking disabled).  
<sup>[1]</sup> Self-reported by [Kumiho](https://kumiho.io) on the 401-sample Cognitive split using `gpt-4o`; factual category breakdown not published.  
<sup>[2]</sup> Reported in the *Madeleine* paper (arXiv 2026) under the official LoCoMo-Plus protocol on the 401-sample Cognitive split; factual category breakdown not published.  
<sup>[3]</sup> Table 1 of the LoCoMo-Plus paper ([arXiv:2602.10715](https://arxiv.org/abs/2602.10715), 2,387 samples, judge: `gemini-2.5-flash`). **LoCoMo Factual Avg** is the sample-weighted mean across the 5 factual categories (1,986 items); **Gap** = LoCoMo Factual Avg &minus; Cognitive.

### Unbiased Analysis &amp; Trade-Offs

- **Surpasses Mem0 (`57.24%`) and SeCom (`57.53%`) on LoCoMo Factual Avg &mdash; 100% locally at `$0` API cost:** CML achieves **59.06%** LoCoMo Factual Avg (`+1.82pp` over Mem0, `+1.53pp` over SeCom, `+13.74pp` over `RAG text-embedding-large`, within `0.58pp` of A-Mem) while running on a local quantized 27B model (`qwen3.8-27b`) with zero write-time LLM calls.
- **#1 Adversarial (`80.27%`) &amp; #1 Memory-System Temporal (`50.16%`):** Selective CA1 speaker-swap monitoring achieves `80.5%` refusal on adversarial traps while keeping factual refusal rates low (`19.4&ndash;24.0%`), and chronological timeline assembly lifts Temporal (`50.16%`) above A-Mem (`49.30%`), GPT-4o (`45.79%`), SeCom (`42.30%`), and Mem0 (`39.40%`).
- **Strong Cognitive leap (`37.41%`, `150/401`) &amp; remaining gaps:** Cognitive HyDE + LLM Cue Selection beat all RAG baselines (`23.5&ndash;29.8%`) and closed to within `4.0pp` of Mem0 (`41.44%`). Remaining deficits in `single-hop` (`60.52%`), `multi-hop` (`37.77%`), and `common-sense` (`40.10%`) reflect the local 27B IQ3_S model vs. GPT-4o plus heuristic ingestion (`USE_LLM_ENABLED=false` during bulk ingestion left the Neo4j entity/relation graph empty; `FEATURES__USE_LLM_ENABLED=true` is enabled by default for production deployments).
- **Judge parity note:** Paper Table 1 baselines are judged by `gemini-2.5-flash`; this 100%-local run uses `qwen3.8-27b` (`temperature=0`) for self-consistent local judging.

### Reproducing the Full 2,387-Sample Local Run

```bash
# 1. Build the official 2,387-sample unified dataset
cd evaluation/locomo_plus/data && python unified_input.py && cd ../../..

# 2. Run against a local OpenAI-compatible LLM (e.g. llama-server qwen3.8-27b on :8000)
python evaluation/scripts/run_locomo_openrouter_batch.py \
  --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json \
  --out-dir evaluation/outputs/qwen38_full_2387 \
  --cml-url http://localhost:8001 \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.8-27b \
  --judge-model qwen3.8-27b \
  --sync-workers 2
```

Full diagnostics &amp; methodology: [evaluation/EVALUATION_REPORT.md](evaluation/EVALUATION_REPORT.md) &#8226; CLI harness: `cml-eval run-full --repo-root .`

---

## Testing

```bash
# 896 hermetic unit tests — no DB, no LLM, mock embeddings
pytest tests/unit -q

# 331 tests against a live server on :8000
pytest tests/integration tests/e2e packages/py-cml/tests -q
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
| **Roadmap** | [ActiveCML Plans](ProjectPlan/ActiveCML/) | Intrinsic memory integration roadmap (Tiers I-2, I-3, I-4, I-5, I-8 implemented in `src/intrinsic/`) |

---

## ActiveCML Intrinsic Memory

CML now includes the tiered **ActiveCML** intrinsic memory subsystem (`src/intrinsic/`, enabled via `FEATURES__INTRINSIC_MEMORY_ENABLED=true`), moving beyond context-window stuffing with four depth tiers:

- **I-2 Logit Bias (`src/intrinsic/logit_interface.py`)**: Synthesizes token-level logit bias maps from high-priority constraints and entities.
- **I-3 Activation Steering (`src/intrinsic/activation_hooks.py`)**: `contextvars`-isolated residual-stream steering hooks safe under concurrent requests.
- **I-4 Synaptic KV-Prefix (`src/intrinsic/prefix_synaptic.py`)**: Canonical prefix block synthesis and tiered KV-prefix caching.
- **I-5 / I-8 Controller &amp; Dynamic LoRA Router (`src/intrinsic/controller.py`, `src/intrinsic/lora_router.py`)**: Unified intrinsic payload synthesis and tenant/domain LoRA adapter routing.

See [ProjectPlan/ActiveCML/](ProjectPlan/ActiveCML/) for the broader 10-phase roadmap index.

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
