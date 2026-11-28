<div align="center">

# Cognitive Memory Layer (CML v3.0)

### A Whole-Brain Neuro-Inspired Memory System for AI

*Store. Retrieve. Consolidate. Simulate. Adapt. Forget.*
*From fast hippocampal episodes to cortical hypergraphs, forward preplay, and dopaminergic skill learning.*

<br/>

[![Quick Start](https://img.shields.io/badge/Quick%20Start-5%20min-success?style=for-the-badge&logo=rocket)](#-quick-start)
[![Docs](https://img.shields.io/badge/Docs-Full%20API-blue?style=for-the-badge&logo=gitbook)](./docs/usage.md)
[![Tests](https://img.shields.io/badge/Tests-1237-brightgreen?style=for-the-badge&logo=pytest)](./tests/README.md)
[![Version](https://img.shields.io/badge/version-3.0.1-blue?style=for-the-badge)](#)

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector%20%2B%20tsvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-%2B%20Cortical%20Hypergraph-008CC1?style=flat-square&logo=neo4j&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-FTS5%20Embedded%20Parity-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Cache%2BQueue-DC382D?style=flat-square&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-GPL--3.0-A42E2B?style=flat-square&logo=gnu&logoColor=white)

</div>

---

<div align="center">

**[The Problem](#-the-problem)** &#8226;
**[What's New in v3.0](#-whats-new-in-v30--whole-brain-architecture)** &#8226;
**[How Your Brain Solves It](#-how-your-brain-solves-it)** &#8226;
**[Architecture](#-architecture)** &#8226;
**[Whole-Brain APIs](#-whole-brain-apis-v30)** &#8226;
**[Quick Start](#-quick-start)** &#8226;
**[Evaluation](#-evaluation-highlights)** &#8226;
**[Docs](#-documentation)**

</div>

---

## The Problem

> *"The brain does not simply store memories; it actively reconstructs them."*
> &mdash; **Bartlett, 1932** <sup>[7]</sup>

Current Large Language Models and flat vector-RAG pipelines have a fundamental cognitive deficit. They operate with **fixed context windows**, **static weights**, and **nearest-neighbor chunk lookup** &mdash; the computational equivalent of a brilliant mind with anterograde amnesia:

| What Standard LLMs &amp; RAG Lack | What Happens in Practice | How the Human Brain Solves It |
| :--- | :--- | :--- |
| **Dynamic belief revision** | Stale facts &amp; superseded states persist | Reconsolidation marks retrieved traces *labile* &amp; updates lineage |
| **Multi-turn consolidation** | Pronouns, implicit links &amp; chapter arcs fragment | Sleep replay distills dialogue windows into coref facts, gists &amp; self-models |
| **Multi-hop relational associative recall** | Pairwise cosine misses 2&ndash;3 hop entity chains | Cortical N-ary hypergraphs &amp; CA3 spreading activation (`&lt;5ms` PPR) |
| **Exact vs. approximate time grounding** | `"last month"` turns into fabricated exact dates | Hippocampal Time Cells preserve true calendar granularity (`April 2023`, `2022`) |
| **Semantically distant latent constraints** | `"Recommend a restaurant"` misses `"shellfish allergy"` | Grounded Proustian Cue Pointers + CA1 reality monitoring |
| **Procedural learning from outcomes** | Agents repeat the exact same mistakes across sessions | Basal Ganglia skill compilation + VTA Dopaminergic Reward Prediction Error |
| **Pre-action counterfactual verification** | Unsafe or misaligned responses fire before checking | Hippocampal forward preplay simulates outcomes against constraints &amp; values |
| **Graceful active forgetting** | Context bloat &amp; catastrophic retrieval interference | Rac1/Cofilin synaptic pruning decays, silences, or compresses obsolete traces |

The **Cognitive Memory Layer (CML v3.0)** bridges these gaps by translating **eight interacting systems from cognitive and computational neuroscience** into a production-ready server (FastAPI + PostgreSQL + Neo4j + Redis) and a **100%-feature-equivalent single-file embedded Python SDK** (`SQLite + FTS5`, zero server required).

---

## What's New in v3.0 &mdash; Whole-Brain Architecture

CML v3.0 upgrades CML from a dual-store hippocampal/neocortical pipeline into a **Whole-Brain Cognitive Architecture** with **`LLM_INTERNAL` enabled by default** across any OpenAI-compatible API (OpenAI, OpenRouter, vLLM, SGLang, Ollama, or local servers) &mdash; achieving **#1 among all memory systems on BOTH LoCoMo Factual Avg (`69.99%`) and LoCoMo-Plus Cognitive (`50.12%`)** under the **official Table 1 `gpt-4o` + `gemini-2.5-flash` Judge protocol (`91.70%` Adversarial, `66.65%` Overall across all 2,387 samples, `~800`-token median packets)** as well as **100%-local `qwen3.8-27b` (`68.40%` Factual / `70.25%` Temporal at `$0` API cost)**, surpassing A-Mem (`59.64%` Factual / `42.44%` Cognitive), SeCom (`57.53%` / `42.63%`), Mem0 (`57.24%` / `41.44%`), full-context GPT-4o (`62.99%` / `41.94%`), full-context GPT-4.1 (`62.21%` / `43.58%`), and even 1M-context Gemini-2.5-Flash (`69.25%` / `44.58%`):

1. **Hybrid CLS Encoding + Windowed Micro-Sleep Consolidation (`POST /memory/consolidate_windowed`):** Fast System-1 Hippocampal batch writes (`FEATURES__BATCH_WRITE_SYSTEM1_FASTPATH=true`, `>300 turns/s`) paired with `LLM_INTERNAL` CRC32 content-defined dialogue chunking + SQLite extraction cache distill multi-turn episodes into coreference-resolved semantic facts (with multi-valued predicate hashing), N-ary Cortical Hyperedges, Grounded Proustian Cue Pointers (`cue -> verbatim episode`), Conway Tier-2 Chapter Gists, and Tier-3 Persistent DMN Self-Profiles.
2. **2-Stage `LLM_INTERNAL` Ecphory &amp; Dentate Gyrus IDF Pattern Separation:** Stage-1 multi-angle query reformulation runs across Dense `pgvector` + conjunction-boosted IDF PostgreSQL `tsvector` GIN (`FTS5` `bm25()` in embedded mode) + Cortical Hypergraph PPR + post-merge `&plusmn;2` conversational contiguity, followed by Stage-2 Kumiho-style batched `LLM_INTERNAL` candidate selection, 2nd-hop bridge resolution, and CA1 evidence-grounding verification (`supported` | `inferable` | `speaker_swap` | `unsupported_premise`).
3. **Bipartite Cortical Hypergraph &amp; `&lt;5ms` Personalized PageRank:** `CorticalHypergraphEngine` binds N-ary event tuples (`{participants, action, object, location, time_bucket}`) into a sparse bipartite incidence graph with sub-5ms CA3 spreading activation and synaptic plasticity.
4. **Granularity-Aware Hippocampal Time Cells:** Preserves exact calendar precision (`day`, `week`, `month`, `season`, `year`, skipping `vague` phrases like `"recently"` instead of fabricating fake `YYYY-MM-DD` offsets) and computes deterministic calendar deltas via `HippocampalTimeCellEngine`.
5. **4-Pathway Ecphory, True-Logit CUDA Cross-Encoder &amp; ACC 2-Hop Loop:** Fuses Dense Vector + Dentate Gyrus BM25 + Cortical Hypergraph PPR + Proustian Cues with Synaptic Co-Activation Resonance, true-logit GPU Cross-Encoder discrimination, CLS stratified slot allocation, and Anterior Cingulate Cortex (ACC) iterative 2-hop prefrontal search.
6. **Basal Ganglia Procedural Memory &amp; VTA Dopaminergic RPE (`POST /memory/feedback`):** Applies positive/negative reward signals to strengthen winning hyperedges, mark refuted memories `labile`, and compile reusable Procedural Skill playbooks.
7. **Hippocampal Forward Preplay / Counterfactual Simulation (`POST /memory/simulate`):** Evaluates candidate agent actions or draft replies against stored constraints, Conway Tier-3 DMN self-profiles, and procedural pitfalls before execution.
8. **LC-NE / ACh Neuromodulation, Perirhinal Multimodal Engrams (`media=`) &amp; Full SQLite Embedded Parity:** `NeuromodulatoryController` regulates encoding/retrieval gain and exploration temperature; `PerirhinalSensoryEncoder` binds visual/multimodal descriptors on `write()`; and `py-cml` `3.0.0` ships `SQLiteFactStore` + `SQLiteGraphStore` + `FTS5` + Cortical Hypergraph in a single SQLite file (retiring `NoOpGraphStore` and `NoOpFactStore` with zero server dependencies).

---

## How Your Brain Solves It

CML v3.0 is a direct computational translation of **eight foundational mechanisms** from cognitive neuroscience. Every mechanism maps 1-to-1 onto concrete modules in `src/`.

### Theory 1: Multi-Store &amp; Perirhinal Multimodal Sensory Pipeline <sup>[Atkinson &amp; Shiffrin, 1968; Baddeley, 2000; Murray &amp; Bussey, 1999]</sup>

```
  Text + Media (Perirhinal)         Attention / Binding                CREB Write Gate
  Sensory Register  ──────────────►  Baddeley Working Memory  ───────►  Long-Term Stores
  (250ms, token + visual)            (10 chunks, coref overlay)         (Hippocampus + Cortex)
        │                                      │                                │
        └─── Decay                             └─── Salience eviction           └─── Pruning
```

**In CML:** Raw text enters `SensoryBuffer` (tiktoken token-ID buffer) while optional `media` attachments pass through `PerirhinalSensoryEncoder` (`src/memory/sensory/perirhinal.py`), binding dense visual/sensory descriptors into unified 768-d representations. `WorkingMemoryManager` enforces a bounded capacity and acts as a **Baddeley Episodic Buffer** (`FEATURES__WORKING_MEMORY_CONTEXT_BINDING_ENABLED=true`), tracking cross-turn coreference anchors and providing a zero-latency working-memory overlay on reads before `SemchunkChunker` segments input for long-term encoding.

### Theory 2: Complementary Learning Systems, Dentate Gyrus &amp; Micro-Sleep <sup>[McClelland et al., 1995; Rasch &amp; Born, 2013]</sup>

| System | Brain Region | Speed | What It Stores in CML v3.0 | Persistence |
| :--- | :--- | :--- | :--- | :--- |
| **Hippocampal + Dentate Gyrus** | Medial Temporal Lobe | Fast (one-shot) | Verbatim episodes + dense 768-d vectors + orthogonal `tsvector` / `FTS5` sparse lexical traces | Postgres / SQLite |
| **Cortical + Hypergraph** | Distributed Neocortex | Consolidated | Structured `FactSchema` facts + N-ary `HyperedgeEngram` incidence graph (`&lt;5ms` PPR) | Postgres + Neo4j / SQLite |

**In CML:** `HippocampalStore` encodes episodes in one shot alongside **Dentate Gyrus sparse pattern separation** (`sparse_lexical_search()` via Postgres `tsvector` GIN or SQLite `FTS5` `bm25()`). During **Windowed Micro-Sleep Consolidation** (`WindowedMicroSleepConsolidator`), CRC32 content-defined dialogue windows (~14&ndash;30 turns) are consolidated in a single cached LLM pass into coreference-resolved atomic facts and N-ary `HyperedgeEngram`s inside `CorticalHypergraphEngine`, complemented by `ConsolidationWorker` Sharp-Wave Ripple (SWR) replay.

### Theory 3: Conway's 3-Tier Self-Memory System &amp; Proustian Recall <sup>[Conway &amp; Pleydell-Pearce, 2000; Berntsen, 2009]</sup>

Human autobiographical memory is hierarchical:
- **Tier 1 &mdash; Event-Specific Knowledge:** Verbatim hippocampal dialogue episodes (`HippocampalStore`).
- **Tier 2 &mdash; Lifetime Periods / Chapter Gists:** Window-level narrative chapter summaries (`conway_tier="chapter_gist"`).
- **Tier 3 &mdash; The Conceptual Self (DMN):** Persistent per-speaker values, recurring stressors, identity traits, and communication style (`ConceptualSelfProfile` in `PersistentDMNStore`, `src/memory/dmn.py`).

To bridge **semantic disconnect** (where `"Recommend a restaurant"` shares zero keywords with `"deathly allergic to shellfish"`), consolidation also indexes **Grounded Proustian Cue Pointers** &mdash; future trigger scenarios that resolve directly back to their **verbatim `parent_episode_id`** at read time so exact episodic detail is never lost to paraphrase.

### Theory 4: Hippocampal Time Cells &amp; Calendar Granularity <sup>[Eichenbaum, 2014]</sup>

Converting `"last month"` into `-30 days` (`2023-04-25`) fabricates false precision when a speaker meant `April 2023`. CML's upgraded `temporal_resolver.py` models **multi-scale Hippocampal Time Cells**, tagging every temporal anchor with an explicit `TemporalGranularity` (`day`, `week`, `month`, `season`, `year`, and skipping `vague` phrases like `"recently"`). At read time, `HippocampalTimeCellEngine` (`src/retrieval/temporal_engine.py`) renders `[said 2023-05-25, refers to April 2023]` and synthesizes a deterministic `## Computed Event Timeline & Calendar Deltas` block.

### Theory 5: Reconsolidation, CA1 Reality Monitoring &amp; Neuromodulation <sup>[Nader et al., 2000; Hasselmo, 2006; Aston-Jones &amp; Cohen, 2005]</sup>

- **Reconsolidation:** `LabileStateTracker` and `ReconsolidationService` mark retrieved memories labile, detect contradictions via `ConflictDetector`, and apply `BeliefRevisionEngine` operations (`REINFORCE`, `TIME_SLICE`, `MERGE`, `ADD_HYPOTHESIS`).
- **CA1 Comparator &amp; Orbitofrontal Reality Monitoring:** `CA1RealityMonitor` (`src/retrieval/reality_monitor.py`) paired with Stage-2 `LLM_INTERNAL` grounding verifies query presuppositions and **speaker-attribution bindings** (catching adversarial speaker-swap and unmentioned-premise traps at **`91.70%`** accuracy while keeping single-hop false refusals down to `9.0%`).
- **Acetylcholine (ACh) &amp; Locus Coeruleus Norepinephrine (NE):** `NeuromodulatoryController` (`src/memory/neuromodulation.py`) computes an `ACh` encoding/retrieval mode gate from novelty and an `NE` exploration gain from retrieval entropy, dynamically scaling Cross-Encoder temperature and triggering **ACC 2-hop iterative recollection** when initial evidence is insufficient.

### Theory 6: Basal Ganglia Procedural Skills &amp; VTA Dopaminergic RPE <sup>[Schultz et al., 1997]</sup>

Memory is not purely declarative. Via `POST /api/v1/memory/feedback` (`memory.feedback()`), `DopamineRPEEngine` (`src/reconsolidation/dopamine_rpe.py`) computes a **Reward Prediction Error** $\delta = r - V$:
- **Positive RPE ($\delta > 0$):** Strengthens synaptic weights on retrieved memories and co-activated cortical hyperedges, and compiles reusable **Procedural Skill** playbooks (`ProceduralMemoryEngine`, `src/memory/procedural.py`).
- **Negative RPE ($\delta < 0$):** Depresses misleading hyperedge weights, marks cited memories `labile` for reconsolidation, and records explicit pitfall warnings in the procedural skill store.

### Theory 7: Hippocampal Forward Preplay &amp; Counterfactual Simulation <sup>[Pfeiffer &amp; Foster, 2013]</sup>

Before rats choose a path at a maze junction, hippocampal place cells sweep forward along candidate trajectories (*forward preplay*). Via `POST /api/v1/memory/simulate` (`memory.simulate()`), `HippocampalPreplaySimulator` (`src/retrieval/preplay.py`) evaluates a proposed action or draft response against stored constraints, Conway Tier-3 DMN profiles, and procedural pitfalls, returning `alignment_score`, `constraint_violations`, `predicted_outcomes`, and `safer_alternative`.

### Theory 8: Active Forgetting <sup>[Shuai et al., 2010]</sup>

Forgetting is an active Rac1/Cofilin-mediated pruning process. `ForgettingWorker` scores traces across importance, recency, access frequency, confidence, memory type, and dependency count, triaging memories into five actions: **Keep**, **Decay**, **Silence** (dormant until reactivated by a high-specificity Bjork ecphory cue), **Compress**, or **Delete**.

---

## How CML Implements It

CML synthesizes eight complementary neuro-AI frameworks into one cohesive engine:

| Framework | Paper / Origin | What CML v3.0 Takes From It |
| :--- | :--- | :--- |
| **CLS Theory** | McClelland et al. (1995) <sup>[1]</sup> | Fast hippocampal episodes + consolidated cortical stores + stratified packet slots |
| **Conway SMS** | Conway &amp; Pleydell-Pearce (2000) <sup>[15]</sup> | 3-tier autobiographical hierarchy (Verbatim Episodes &rarr; Chapter Gists &rarr; DMN Self-Profile) |
| **HippoRAG + Hypergraph** | Gutierrez et al. (2024) <sup>[8]</sup> | Knowledge graph + N-ary Bipartite Cortical Hypergraph with `&lt;5ms` Personalized PageRank |
| **Madeleine / Proustian** | Berntsen (2009); Madeleine (2026) <sup>[19]</sup> | Grounded involuntary cue pointers resolving directly back to verbatim parent episodes |
| **HawkinsDB** | Thousand Brains Theory <sup>[10]</sup> | 15 cognitive memory types with distinct biological decay profiles |
| **Mem0** | mem0.ai (2024) <sup>[9]</sup> | Add/Update/Delete/No-change belief revision &amp; supersession lineage |
| **LoCoMo-Plus** | Li et al. (2026) <sup>[14]</sup> | Level-2 implicit cognitive constraint memory (`goal`/`value`/`state`/`causal`/`policy`) |
| **ActiveCML &amp; Laya** | CML Intrinsic + Convai Laya | Subcortical ~20ms System 1 routing + logit bias, activation steering, KV-prefix &amp; LoRA routing |

### 4-Pathway Ecphory &amp; The Cognitive Constraint Layer

Standard RAG retrieves chunks with high surface cosine similarity. CML v3.0 reconstructs context through **4-Pathway Ecphory + Prefrontal Reranking**:

```
Stored (Session 1):  "I went into anaphylaxis from shrimp last spring."   (episodic cue)
Micro-Sleep Pointer: "restaurant / seafood / dinner recommendation" ────► [verbatim parent_episode_id]
User Query (Later):  "Where should we book dinner tonight?"               (semantically distant trigger)
```

1. **4 Parallel Candidate Pathways:** (a) **Dense Vector** (`pgvector` / SQLite cosine + 3-cue Cognitive HyDE), (b) **Dentate Gyrus Sparse BM25** (`tsvector` GIN / SQLite `FTS5`), (c) **Bipartite Cortical Hypergraph PPR** (`&lt;5ms` multi-hop entity-engram spreading activation), and (d) **Grounded Proustian Cue Pointers** (resolving synthetic cue hits directly to their verbatim hippocampal episodes).
2. **Synaptic Co-Activation Resonance &amp; True-Logit CUDA Cross-Encoder:** Candidates matched by multiple pathways receive a non-linear coincidence bonus before `CrossEncoderScorer` (`BAAI/bge-reranker-base`, raw logits with date/speaker headers) scores query&ndash;evidence entailment.
3. **ACC Iterative 2-Hop Loop &amp; CLS Stratified Slot Allocation:** If LC-NE/ACh neuromodulation detects low top-3 support, the Anterior Cingulate Cortex loop extracts missing entity/temporal bridges and fires a targeted second hop. Finally, CLS stratified allocation guarantees verbatim episodes are never crowded out by synthetic facts or off-topic constraints.

---

## Architecture

### Whole-Brain System Overview (CML v3.0)

```mermaid
flowchart TB
    CLIENT["Client / py-cml SDK\n(Sync, Async, or Zero-Server Embedded)"] --> API["FastAPI REST API / Embedded Engine"]
    API --> ORCH["MemoryOrchestrator"]

    ORCH --> WRITE["Write & Perirhinal Multimodal Path"]
    ORCH --> SLEEP["Windowed Micro-Sleep Consolidator"]
    ORCH --> READ["4-Pathway Ecphory & ACC Read Path"]
    ORCH --> SIM["Hippocampal Forward Preplay (/simulate)"]
    ORCH --> FB["Dopaminergic RPE & Procedural (/feedback)"]
    ORCH --> FORGET["Rac1/Cofilin Active Forgetting"]

    WRITE --> GATE["Sensory + Baddeley WM + CREB WriteGate + Time Cells"]
    SLEEP --> CONWAY["Coref Facts + N-ary Hyperedges + Proustian Cues\n+ Conway Chapter Gists + Tier-3 DMN Self-Profiles"]

    READ --> FUSE["Dense + Dentate Gyrus BM25 + <5ms Hypergraph PPR + Proustian Cues"]
    FUSE --> RERANK["CUDA Cross-Encoder + LC-NE/ACh Gain + CA1 Reality Monitor"]
    RERANK --> PACKET["MemoryPacket + Timeline Deltas + ActiveCML Intrinsic"]

    GATE --> STORE[("Server: Postgres (pgvector + tsvector) + Neo4j + Redis\nEmbedded: Single-File SQLite + FTS5 + SQLiteGraph/FactStore")]
    CONWAY --> STORE
    STORE --> FUSE
    FB --> STORE
    SIM --> FUSE
```

### Data Flow: Write &rarr; Micro-Sleep &rarr; 4-Pathway Read &rarr; Preplay &amp; Feedback

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F8FAFC', 'primaryTextColor': '#0F172A', 'primaryBorderColor': '#475569', 'lineColor': '#64748B', 'fontSize': '13px'}}}%%
flowchart LR
    classDef write fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e40af
    classDef sleep fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e
    classDef store fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87
    classDef read fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#065f46
    classDef loop fill:#ffe4e6,stroke:#e11d48,stroke-width:2px,color:#9f1239

    subgraph Encode ["1. WRITE & CONSOLIDATE"]
        direction TB
        W1["Text + Media\n(Perirhinal)"]:::write
        W2["Baddeley WM +\nSemchunk + WriteGate"]:::write
        W3["Time Cells +\nSomatic + Prospective"]:::write
        C1["Windowed Micro-Sleep\n(CRC32 Chunks + Cache)"]:::sleep
        W1 --> W2 --> W3 --> C1
    end

    subgraph BrainStore ["2. WHOLE-BRAIN STORES"]
        direction TB
        S1[("Hippocampal Episodes\npgvector + tsvector / FTS5")]:::store
        S2[("Cortical Hypergraph (<5ms PPR)\n+ Neo4j / SQLiteGraphStore")]:::store
        S3[("Conway Tier-2 Gists +\nTier-3 DMN Self + Skills")]:::store
    end

    subgraph Ecphory ["3. 4-PATHWAY READ"]
        direction TB
        R1["Query + Laya System 1\n+ LC-NE / ACh Gain"]:::read
        R2["Dense + BM25 + Hypergraph PPR\n+ Grounded Proustian Cues"]:::read
        R3["CUDA Cross-Encoder +\nACC 2-Hop + CLS Slots"]:::read
        R4["Time Cell Deltas +\nCA1 Monitor + Packet"]:::read
        R1 --> R2 --> R3 --> R4
    end

    subgraph ActionLoop ["4. SIMULATE & ADAPT"]
        direction TB
        A1["Forward Preplay\n(/memory/simulate)"]:::loop
        A2["Dopaminergic RPE\n(/memory/feedback)"]:::loop
    end

    W3 --> S1
    C1 --> S1
    C1 --> S2
    C1 --> S3
    S1 --> R2
    S2 --> R2
    S3 --> R2
    R4 --> A1
    A2 --> S2
    A2 --> S3
```

### Neuroscience &rarr; Code Mapping

Every module in CML corresponds to a specific biological mechanism. Click to explore each:

<details>
<summary><strong>Sensory, Perirhinal &amp; Working Memory</strong> &mdash; Multimodal binding &amp; Baddeley's Episodic Buffer</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Sensory register | `SensoryBuffer` (tiktoken token-ID storage) | `src/memory/sensory/buffer.py` |
| Perirhinal multimodal cortex | `PerirhinalSensoryEncoder` (`media=` visual/sensory descriptor extraction &amp; binding) | `src/memory/sensory/perirhinal.py` |
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
<summary><strong>Hippocampal Store</strong> &mdash; One-shot episodic encoding, Dentate Gyrus BM25 &amp; Time Cells</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| One-shot encoding | `HippocampalStore.encode_batch()` | `src/memory/hippocampal/store.py` |
| Dentate Gyrus sparse pattern separation | `sparse_lexical_search()` (Postgres `tsvector` GIN + SQLite `FTS5` `bm25()`) | `src/storage/postgres.py`, `packages/py-cml/src/cml/storage/sqlite_store.py` |
| Hippocampal Time Cells | Granularity-aware relative date binding (`day`/`week`/`month`/`season`/`year`, skipping `vague`) + `HippocampalTimeCellEngine` | `src/extraction/temporal_resolver.py`, `src/retrieval/temporal_engine.py` |
| Unified extraction | Entities, relations, constraints, facts in one call | `src/extraction/unified_write_extractor.py` |

</details>

<details>
<summary><strong>Neocortical Store &amp; Hypergraph</strong> &mdash; Schema alignment &amp; &lt;5ms Cortical Spreading Activation</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Schema-based storage | `FactSchema` + `FactCategory` (`Postgres` + `SQLiteFactStore`) | `src/memory/neocortical/schemas.py`, `packages/py-cml/src/cml/storage/sqlite_neocortical.py` |
| Cognitive categories | GOAL, STATE, VALUE, CAUSAL, POLICY | `src/memory/neocortical/schemas.py` |
| Bipartite Cortical Hypergraph | `CorticalHypergraphEngine` (N-ary `HyperedgeEngram` + `&lt;5ms` Personalized PageRank) | `src/memory/neocortical/hypergraph.py` |
| Graph traversal | Neo4j (`Server`) &amp; `SQLiteGraphStore` (`py-cml` embedded parity) | `src/storage/neo4j.py`, `packages/py-cml/src/cml/storage/sqlite_neocortical.py` |

</details>

<details>
<summary><strong>Retrieval</strong> &mdash; 4-Pathway Ecphory, CUDA Cross-Encoder &amp; ACC Iterative Control</summary>

Memory retrieval is not lookup &mdash; it is **ecphory**: the interaction between a retrieval cue and a stored engram that *reconstructs* the memory (Tulving, 1983 <sup>[2]</sup>).

| Biology | Code | Location |
| :--- | :--- | :--- |
| Query classification &amp; planning | `QueryClassifier` + `RetrievalPlanner` | `src/retrieval/classifier.py`, `src/retrieval/planner.py` |
| 4-Pathway hybrid ecphory | Dense Vector + Dentate Gyrus BM25 + Cortical Hypergraph PPR + Grounded Proustian Cues (`cue -> verbatim episode`) | `src/retrieval/memory_retriever.py` |
| Prefrontal Cross-Encoder &amp; CLS allocation | `CrossEncoderScorer` (true-logit CUDA reranking) + `MemoryReranker` + ACC up-to-2-hop loop | `src/retrieval/cross_encoder.py`, `src/retrieval/reranker.py` |

</details>

<details>
<summary><strong>Consolidation</strong> &mdash; Windowed Micro-Sleep &amp; Sharp-Wave Ripple Replay</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Windowed Micro-Sleep Consolidation | `WindowedMicroSleepConsolidator` (CRC32 content-defined windows: coref atomic facts, N-ary hyperedges, Proustian cues, Conway Tier-2 Chapter Gists &amp; Tier-3 DMN Self-Profiles) | `src/consolidation/windowed_consolidator.py` |
| Salience-weighted SWR replay &amp; sampling | `EpisodeSampler` + `ConsolidationWorker` (`swr_links_induced`) | `src/consolidation/sampler.py`, `src/consolidation/worker.py` |
| Semantic clustering &amp; migration | `SemanticClusterer` + `GistExtractor` + `ConsolidationMigrator` | `src/consolidation/` |

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
<summary><strong>Whole-Brain Subsystems &amp; Laya System 1</strong> &mdash; Amygdala, BA10, CA1, DMN, LC-NE/ACh, Basal Ganglia, Preplay</summary>

| Biology | Code | Location |
| :--- | :--- | :--- |
| Amygdala somatic markers | `SomaticMarkerTagger` (valence/arousal, flashbulb decay modulation, state-congruent resonance) | `src/memory/somatic.py` |
| Rostral PFC (BA10) prospective memory | `ProspectiveMemoryEngine` (causal intention triggers + read-time parent ecphory) | `src/memory/prospective.py` |
| CA1 comparator &amp; OFC reality monitoring | `CA1RealityMonitor` (presupposition checks, speaker-attribution binding, concise refusal alerts) | `src/retrieval/reality_monitor.py` |
| Default Mode Network (Conway Tier-3 Self) | `DMNTheoryOfMind` + `PersistentDMNStore` (persistent per-speaker `ConceptualSelfProfile` + mentalizing) | `src/memory/dmn.py` |
| LC-NE &amp; ACh neuromodulatory gain | `NeuromodulatoryController` (encoding vs. retrieval mode balance + surprise-driven exploration temperature) | `src/memory/neuromodulation.py` |
| Basal Ganglia / VTA Dopaminergic RPE | `ProceduralMemoryEngine` + `DopamineRPEEngine` (`POST /memory/feedback` &mdash; skill compilation &amp; synaptic weight updates) | `src/memory/procedural.py`, `src/reconsolidation/dopamine_rpe.py` |
| Hippocampal forward preplay | `HippocampalPreplaySimulator` (`POST /memory/simulate` &mdash; counterfactual rollout against constraints &amp; DMN profiles) | `src/retrieval/preplay.py` |
| Subcortical System 1 &amp; ActiveCML | `LayaSystem1Engine` + `IntrinsicMemoryController` | `src/core/laya_system1.py`, `src/intrinsic/` |

</details>

### Memory Types

CML supports 15 memory types, each with a biological analog and distinct decay profile:

| Type | Description | Decay |
| :--- | :--- | :--- |
| `episodic_event` | A specific personal event anchored in time/place. | Fast |
| `semantic_fact` | A factual statement about the world or domain knowledge. | Slow |
| `procedure` | Step-by-step instructions or compiled procedural skill playbook. | Stable |
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

Types are defined in `src/core/enums.py` as `MemoryType`. **Implementation**: type assignment at write in `src/memory/hippocampal/write_gate.py` (`_determine_memory_types()`); retention by type in `src/forgetting/scorer.py` (`ScorerConfig.type_bonuses`).

### Extraction, Consolidation &amp; Enrichment Modes

CML v3.0 supports three complementary operating regimes:

1. **Fast Hippocampal Write + Windowed Micro-Sleep Consolidation (Recommended for High-Throughput + SOTA Accuracy):** Ingest turns at ~190 turns/s (`FEATURES__USE_LLM_ENABLED=false` during bulk ingest or rapid chat bursts) and run `consolidate_windowed` periodically (~16&times; fewer LLM calls than per-turn extraction via CRC32 content-defined windows + SQLite extraction cache).
2. **Online Per-Turn Unified LLM Extraction (`FEATURES__USE_LLM_ENABLED=true`, default):** `UnifiedWritePathExtractor` extracts entities, relations, constraints, facts, PII spans, and importance on every `write()`. If a cloud provider (`openai`, `gemini`, `claude`) is configured without an API key, CML logs a warning and falls back gracefully to deterministic heuristics.
3. **Zero-LLM Deterministic Mode (`FEATURES__USE_LLM_ENABLED=false`):** Pure regex/Jaccard write-time facts, 5-category constraint extraction, granularity-aware Time Cells, Dentate Gyrus BM25, dense vectors, Somatic/Prospective tagging, and CUDA Cross-Encoder reranking with zero LLM calls.

---

## Whole-Brain APIs (v3.0)

Every Whole-Brain capability is available identically across the **FastAPI REST server**, **`CognitiveMemoryLayer` (sync HTTP)**, **`AsyncCognitiveMemoryLayer` (async HTTP)**, and **`EmbeddedCognitiveMemoryLayer` (zero-server single-file SQLite + FTS5)**.

### Python SDK (`Server` or `Embedded` Zero-Server Mode)

```python
import asyncio
from cml import EmbeddedCognitiveMemoryLayer

async def main():
    # Single-file SQLite + FTS5 + SQLiteFactStore + SQLiteGraphStore + Cortical Hypergraph!
    # Or connect to a server: CognitiveMemoryLayer(base_url="http://localhost:8000", api_key="test-key")
    async with EmbeddedCognitiveMemoryLayer(db_path="brain.db") as memory:
        # 1. Write episodic & multimodal Perirhinal memories
        await memory.write(
            "I went into anaphylaxis from shrimp last April — strict shellfish ban.",
            session_id="s1",
            metadata={"speaker": "Alice"},
            media=[{"caption": "Medical allergy bracelet: NO SHELLFISH"}],
        )

        # 2. Windowed Micro-Sleep Consolidation (extracts coref facts, N-ary hyperedges,
        #    Proustian cue pointers, Conway Chapter Gists & Tier-3 DMN Self-Profiles)
        await memory.consolidate_windowed(session_id="s1")

        # 3. 4-Pathway Read (Dense + BM25 + <5ms Hypergraph PPR + Proustian Cues + Cross-Encoder)
        read_res = await memory.read("Where should we book dinner for Alice tonight?", format="llm_context")
        print(read_res.context)

        # 4. Hippocampal Forward Preplay / Counterfactual Simulation before acting
        sim = await memory.simulate(
            scenario="Book a table at Harbor Crabhouse (famous for shrimp boils) for Alice",
            domain="dining",
        )
        print(sim.alignment_score, sim.constraint_violations, sim.safer_alternative)

        # 5. Dopaminergic RPE Feedback & Procedural Skill Compilation after acting
        fb = await memory.feedback(
            query="Where should we book dinner for Alice tonight?",
            outcome="Booked vegetarian-friendly Mediterranean bistro; Alice loved it.",
            reward=1.0,
            domain="dining",
            memory_ids=[m.id for m in read_res.memories],
        )
        print(fb.rpe_delta, fb.compiled_skill)

asyncio.run(main())
```

### REST Endpoints Summary

| Endpoint | Brain Subsystem | Purpose |
| :--- | :--- | :--- |
| `POST /api/v1/memory/write` | Hippocampus + Dentate Gyrus + Perirhinal + Time Cells | Store text &amp; optional `media` descriptors with dense + `tsvector` indexing |
| `POST /api/v1/memory/consolidate_windowed` | Micro-Sleep + Cortical Hypergraph + Conway SMS | Distill dialogue windows into facts, N-ary hyperedges, Proustian cues, gists &amp; DMN profiles |
| `POST /api/v1/memory/read` | 4-Pathway Ecphory + Cross-Encoder + ACC + CA1 | Reconstruct `MemoryPacket` with timeline deltas, reality-monitor alerts &amp; ActiveCML payload |
| `POST /api/v1/memory/turn` | Working Memory + Read + Write + Reconsolidation | Seamless conversational turn handler (`user_message` + `assistant_response`) |
| `POST /api/v1/memory/simulate` | Hippocampal Forward Preplay | Counterfactual verification against constraints, DMN profiles &amp; procedural pitfalls |
| `POST /api/v1/memory/feedback` | Basal Ganglia + VTA Dopaminergic RPE | Apply reward `[-1.0, +1.0]` to update synaptic weights, labile state &amp; procedural skills |
| `POST /api/v1/memory/forget` | Rac1/Cofilin Active Pruning | Trigger targeted or policy-based forgetting / archiving |

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

`./docker/up.sh` wraps `docker compose` and automatically applies the GPU override when `nvidia-smi` is available. The `api` service starts Postgres (`pgvector` + `tsvector`), Neo4j, Redis, runs Alembic migrations (including `003_whole_brain_v3`), and serves the API — all in one command.

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
<summary><strong>Python SDK (HTTP or Zero-Server Embedded SQLite)</strong></summary>

```bash
pip install cognitive-memory-layer              # HTTP sync + async clients
pip install "cognitive-memory-layer[embedded]"  # Full Whole-Brain engine over SQLite + FTS5
```

See [SDK docs](packages/py-cml/README.md) for sync, async, and embedded usage.

</details>

---

## Getting a Runnable Project

### Prerequisites

| Requirement | Minimum version | Notes |
| :--- | :--- | :--- |
| Python | 3.11+ | 3.12 / 3.13 recommended |
| Docker + Compose | 24+ | For Server mode (Postgres, Neo4j, Redis); not needed for `[embedded]` mode |
| `uv` or `pip` | any | `uv` strongly recommended for speed |
| Disk space | ~6 GB free | Repo + venv + on-demand HF embeddings/reranker/tokenizer (~2.5&ndash;3.5 GB) |
| RAM | 8 GB+ | 16 GB recommended (local embeddings + cross-encoder) |

---

### Step 1 &mdash; Clone the Repository

```bash
git clone https://github.com/avinash-mall/CognitiveMemoryLayer.git
cd CognitiveMemoryLayer
```

---

### Step 2 &mdash; Set Up the Python Environment

**With uv (recommended):**

```bash
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[server,dev]"
```

**With pip:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[server,dev]"
```

---

### Step 3 &mdash; Model Weights

Runtime model artifacts are downloaded automatically on demand and cached across restarts (`hf-cache` and `tiktoken-cache` volumes in Docker):

| Artifact | Source | Configured by |
| :--- | :--- | :--- |
| Embedding model (`nomic-ai/nomic-embed-text-v2-moe`, 768-d, revision-pinned) | HuggingFace | `EMBEDDING_INTERNAL__LOCAL_MODEL` |
| Cross-Encoder reranker (`BAAI/bge-reranker-base`, CUDA FP16 / CPU fallback) | HuggingFace | `FEATURES__CROSS_ENCODER_ENABLED`, `RETRIEVAL__CROSS_ENCODER_MODEL` |
| Chunker tokenizer (`google/flan-t5-base`) | HuggingFace | `CHUNKER__TOKENIZER` |
| System 1 decision model (`convaiinnovations/laya`, optional) | HuggingFace | `FEATURES__LAYA_SYSTEM1_ENABLED`, `FEATURES__LAYA_MODEL_ID` |
| tiktoken BPE ranks (`cl100k_base`) | openaipublic.blob.core.windows.net | &mdash; |

> **Note:** `nomic-embed-text-v2-moe` is loaded with `trust_remote_code=True` (pinned to a verified revision). Set `EMBEDDING_INTERNAL__PROVIDER` to `mock`, `openai`, `ollama`, or `vllm` if you prefer not to execute model-repo code. For **air-gapped / offline runs**, warm the Docker volumes once and set `HF_HUB_OFFLINE=1` in `.env`.

---

### Step 4 &mdash; Configure the Environment

```bash
cp .env.minimal .env   # Works out of the box with local GPU/CPU embeddings & Docker DBs
# Or copy full reference template:
cp .env.example .env
```

Key settings in `.env`:

```dotenv
# ── Required (Server Mode) ────────────────────────────────────
DATABASE__POSTGRES_URL=postgresql+asyncpg://cml:cml@localhost:5432/cml
AUTH__API_KEY=your-api-key-here          # used by clients (X-API-Key header)
AUTH__ADMIN_API_KEY=your-admin-key-here  # dashboard + admin routes

# ── Internal LLM (enabled by default; accepts any OpenAI-compatible external or local API) ──
LLM_INTERNAL__PROVIDER=openai_compatible
LLM_INTERNAL__BASE_URL=https://api.openai.com/v1     # or https://openrouter.ai/api/v1, http://127.0.0.1:8000/v1 (vLLM/Ollama)
LLM_INTERNAL__MODEL=gpt-4o-mini                      # or google/gemini-2.5-flash, qwen3.8-27b, etc.
LLM_INTERNAL__API_KEY=your-llm-api-key               # optional ("dummy" auto-used) for local endpoints

# ── Whole-Brain & Hybrid CLS Feature Flags (all enabled by default) ───
FEATURES__USE_LLM_ENABLED=true
FEATURES__BATCH_WRITE_SYSTEM1_FASTPATH=true
FEATURES__HYDE_RETRIEVAL_ENABLED=true
FEATURES__LLM_MULTI_ANGLE_EXPANSION_ENABLED=true
FEATURES__LLM_JUDGED_DELIVERY_ENABLED=true
FEATURES__SPARSE_BM25_ENABLED=true
FEATURES__HYPERGRAPH_PPR_ENABLED=true
FEATURES__CONWAY_HIERARCHY_ENABLED=true
FEATURES__CROSS_ENCODER_ENABLED=true
FEATURES__ACC_ITERATIVE_LOOP_ENABLED=true
FEATURES__NEUROMODULATION_ENABLED=true
FEATURES__PREPLAY_SIMULATION_ENABLED=true
FEATURES__PROCEDURAL_RPE_ENABLED=true
FEATURES__PERIRHINAL_MULTIMODAL_ENABLED=true
FEATURES__WINDOWED_CONSOLIDATION_ENABLED=true
```

---

### Step 5 &mdash; Start All Services &amp; Verify

```bash
# Starts Postgres, Neo4j, Redis, runs migrations, and serves the API (GPU auto-detected)
./docker/up.sh up -d api
./docker/up.sh ps

# Health check
curl http://localhost:8000/api/v1/health

# Write -> Consolidate -> Read -> Simulate -> Feedback
curl -X POST http://localhost:8000/api/v1/memory/write \
  -H "X-API-Key: test-key" -H "Content-Type: application/json" \
  -d '{"content": "I never eat shellfish — severe allergy.", "session_id": "demo"}'

curl -X POST http://localhost:8000/api/v1/memory/read \
  -H "X-API-Key: test-key" -H "Content-Type: application/json" \
  -d '{"query": "Recommend a restaurant for tonight"}'

curl -X POST http://localhost:8000/api/v1/memory/simulate \
  -H "X-API-Key: test-key" -H "Content-Type: application/json" \
  -d '{"scenario": "Order the shrimp platter for the table", "domain": "dining"}'
```

Authentication uses the `X-API-Key` header (`tenant_id` is resolved from the API key; admin keys can select any tenant via `X-Tenant-Id`). Open the **Admin Dashboard** at [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/).

---

## Evaluation Highlights

CML is evaluated end-to-end across **two complementary long-horizon agent memory benchmarks** (`0` errors across all runs):

| Benchmark | Scale &amp; Domain | Runtime / Backbone | Headline Accuracy | Standout Category Scores | API Cost | Summary Artifact |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| **LoCoMo-Plus** *(Table 1 Cloud)* | `2,387` samples (multi-session chat, `6` categories) | `openai/gpt-4o` QA + `gemini-2.5-flash` Judge | **66.65%** Overall (`69.99%` Factual) | **91.70%** Adversarial (**#1**), **50.12%** Cognitive (**#1**), **71.70%** Single-hop | Cloud | [`locomo_plus_..._v3_1_...json`](evaluation/results/locomo_plus/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json) |
| **LoCoMo-Plus** *(100% Local)* | `2,387` samples (multi-session chat, `6` categories) | Local `qwen3.8-27b` (`IQ3_S` + `dflash2`) QA + Judge | **63.53%** Overall (`68.40%` Factual) | **82.06%** Adversarial, **70.25%** Temporal, **70.21%** Single-hop | **\$0.00** | [`locomo_plus_..._qwen38_...json`](evaluation/results/locomo_plus/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json) |
| **LongMemEval-V2** *(100% Local `small`)* | `451` Qs (multimodal Web + ServiceNow trajectories) | Local `qwen3.8-27b` (`IQ3_S` + `dflash2`) Reader + Judge | **47.67%** Full-Set (**56.97%** Non-Abst) | **61.32%** Workflow (`77.03%` non-abs, `90.48%` Web), **51.72%** Gotchas (`66.67%` Web) | **\$0.00** | [`longmemeval_v2_..._451_summary.json`](evaluation/results/longmemeval_v2/longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json) |

Expand either collapsible benchmark section below for category definitions, full head-to-head comparison tables, trade-off analysis, and reproduction commands (or see [evaluation/EVALUATION_REPORT.md](evaluation/EVALUATION_REPORT.md) for the complete diagnostics report).

<details>
<summary><strong>LoCoMo-Plus Benchmark &mdash; Conversational Factual &amp; Implicit Cognitive Memory (2,387 Samples)</strong></summary>

Evaluated **on the complete LoCoMo-Plus benchmark (2,387 samples, 0 errors)** &mdash; 1,986 original LoCoMo factual questions across 5 categories plus 401 **Cognitive** cue&ndash;trigger constraint instances (Li et al., 2026, [arXiv:2602.10715](https://arxiv.org/abs/2602.10715)) &mdash; under **both** the **official Table 1 cloud protocol (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge)** and a **100%-local `\$0`-API configuration (`qwen3.8-27b`)**:

### What Each Benchmark Category Measures &amp; How CML Solves It

| Category (Samples) | What It Evaluates | CML Score (`gpt-4o`) | Key CML Mechanism Driving Performance |
| :--- | :--- | :---: | :--- |
| **Single-hop (`841`)** | Direct recall of a specific fact, object, event, or shared-photo detail mentioned in one past session (`89.2%` packet recall). | **71.70%** | Perirhinal `[Shared photo: ...]` preservation + IDF `tsvector` BM25 + `&plusmn;2` adjacent Q&amp;A turn contiguity. |
| **Multi-hop (`282`)** | Synthesizing scattered clues or aggregating lists/counts across multiple weeks of conversation (`83.3%` packet recall). | **47.87%** | Stage-1 multi-angle expansion + non-hub Entity-Bridge boosts + Stage-2 2nd-hop bridge fetch + multi-valued fact hashing. |
| **Temporal (`321`)** | Resolving relative time expressions (`"yesterday"`, `"last week"`, `"last year"`), event ordering, and multi-month durations (`85.4%` packet recall). | **60.59%** *(70.25% local)* | Granularity-aware Hippocampal Time Cells (`[said YYYY-MM-DD, refers to ...]`) + deterministic calendar deltas. |
| **Common-sense (`96`)** | Combining stored persona traits, goals, values, and habits with world knowledge to answer hypothetical `"Would / Likely..."` questions. | **50.52%** *(58.85% open)* | Conway Tier-3 Persistent DMN Self-Profiles + Persona-Profile Augmentation + Stage-2 `"inferable"` grounding (eliminating false abstentions). |
| **Adversarial (`446`)** | Rejecting misleading presuppositions and **speaker-swap traps** (asking what Person A did when actually Person B did it). | **91.70%** **(#1 Overall)** | CA1 Speaker-Attribution Comparator + Stage-2 `speaker_swap` / `unsupported_premise` reality monitoring. |
| **Cognitive / LoCoMo-Plus (`401`)** | **Level-2 Implicit Memory Awareness:** Connecting a speaker's casual remark weeks later (*trigger*) to an earlier unstated cause, mishap, allergy, scare, or habit (*cue*) across **zero lexical overlap**. | **50.12%** **(#1 Table 1)** | Uncapped Windowed Micro-Sleep **Grounded Proustian Cue Pointers** (`cue -> verbatim episode`) + 4-angle Cognitive HyDE + Stage-2 cue selection. |
| **Gap (`Avg - Cognitive`)** &darr; | Drop-off from factual recall (`LoCoMo Avg`) to implicit cognitive memory (`Cognitive`); **lower is better**. | **19.87** *(14.35 macro)* | Simultaneous gains on both Factual (`69.99%`) and Cognitive (`50.12%`) narrow the factual&ndash;cognitive divide by nearly `10pp`. |

### Headline Achievements

- **#1 Memory System on BOTH LoCoMo Factual Avg (`69.99%`) AND LoCoMo-Plus Cognitive (`50.12%`):** Under the exact Table 1 `openai/gpt-4o` + `google/gemini-2.5-flash` Judge protocol, CML outperforms **A-Mem (`59.64%` Factual / `42.44%` Cognitive)** by **`+10.35pp` / `+7.68pp`**, **SeCom (`57.53%` / `42.63%`)** by **`+12.46pp` / `+7.49pp`**, and **Mem0 (`57.24%` / `41.44%`)** by **`+12.75pp` / `+8.68pp`**.
- **Surpasses Full-Context Frontier LLMs Using `~800`-Token Memory Packets:** Beats **Full-Context `GPT-4o` (`62.99%` Factual / `41.94%` Cognitive)** by **`+7.00pp` / `+8.18pp`**, **Full-Context `GPT-4.1` (`62.21%` / `43.58%`)** by **`+7.78pp` / `+6.54pp`**, **1M-Context `Gemini-2.5-Flash` (`69.25%` / `44.58%`)** on both Factual and Cognitive, and **1M-Context `Gemini-2.5-Pro` (`45.72%` Cognitive)** by **`+4.40pp`** on Cognitive memory.
- **#1 Adversarial Robustness (`91.70%`, `409/446`) &amp; #1 `gpt-4o` Memory-System Temporal (`60.59%`, `70.25%` local):** Outperforms A-Mem (`35.20%`), SeCom (`31.80%`), and Mem0 (`30.50%`) by **`+56.5pp` to `+61.2pp`** on adversarial traps and by **`+11.3pp` to `+21.2pp`** on temporal reasoning.
- **Committed Summary Artifacts:**
  - Official Hybrid CLS + 2-Stage `LLM_INTERNAL` (`2026-11-27`, `66.65%` Overall, `1,591.0 / 2,387`): [`evaluation/results/locomo_plus/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json`](evaluation/results/locomo_plus/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json)
  - 100%-Local Zero-API-Cost (`2026-11-25`, `qwen3.8-27b`, `\$0.00` API cost, `63.53%` Overall): [`evaluation/results/locomo_plus/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json`](evaluation/results/locomo_plus/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json)

### Full LoCoMo-Plus Comparison (2,387 Samples)

| System / Method | Category | Backbone / QA Model | Single-hop (841) | Multi-hop (282) | Temporal (321) | Common-sense (96) | Adversarial (446) | **LoCoMo Factual Avg (1,986)** | **Cognitive (401)** | **Gap** &darr; |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CML v3.1 (Hybrid CLS + 2-Stage `LLM_INTERNAL`)** <sup>[0a]</sup> | Memory System | `gpt-4o` (judge: `gemini-2.5-flash`) | 71.70% | 47.87% | **60.59%** | 50.52% *(58.85%&dagger;)* | **91.70%** | **69.99%** | **50.12%** | 19.87 *(14.35&Dagger;)* |
| **CML v3.0 (Whole-Brain, Local `\$0` API)** <sup>[0b]</sup> | Memory System | `qwen3.8-27b` (100% local, `\$0` API) | 70.21% | 47.52% | **70.25%** | 44.27% | 82.06% | **68.40%** | 39.40% | 29.00 |
| **CML v3.0 (`\$0` Local Store + `gpt-4o` QA)** <sup>[0c]</sup> | Memory System | `gpt-4o` (judge: `gemini-2.5-flash`) | 63.32% | 42.20% | 57.63% | 39.06% | 87.67% | 63.70% | 33.92% | 29.78 |
| **Kumiho Cognitive Memory** <sup>[1]</sup> | Memory System | `gpt-4o` (graph-native, 401 split only) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | 93.30%* | &mdash; |
| **Madeleine + HyperMem** <sup>[2]</sup> | Memory System | Involuntary recall + HyperMem (401 only) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | 66.60%* | &mdash; |
| **HyperMem** <sup>[2]</sup> | Memory System | HyperMem baseline (401 split only) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | 52.90%* | &mdash; |
| **Madeleine (standalone)** <sup>[2]</sup> | Memory System | Zero online LLM calls (401 split only) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | 52.40%* | &mdash; |
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

<sup>[0a]</sup> Official Table 1 protocol (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge via OpenRouter, `temperature=0`, `2,387/2,387` valid, `0` errors, overall 6-category avg **66.65%**, `1,591.0 / 2,387`; `LLM_INTERNAL`: `gemini-2.5-flash` Windowed Micro-Sleep Consolidation + `gemini-2.5-flash-lite` 2-stage read; &dagger;`58.85%` Common-sense and `9.96` macro Gap under the open-inferential QA prompt; &Dagger;unweighted 5-category macro mean **64.48%**, macro Gap **14.35pp**).  
<sup>[0b]</sup> Full 2,387-sample 100%-local run from scratch (`2,387/2,387` valid, `0` errors, overall 6-category avg **63.53%**, `1,516.5 / 2,387`); QA &amp; judge: `qwen3.8-27b` (`temperature=0`, thinking disabled, `\$0` API cost).  
<sup>[0c]</sup> Prior `\$0`-consolidation baseline reusing local `qwen3.8-27b` (`IQ3_S`) consolidated store with `FEATURES__USE_LLM_ENABLED=false` at read time (`58.69%` overall, `1,401.0 / 2,387`).  
<sup>[1]</sup> *Self-reported by [Kumiho](https://kumiho.io) on the 401-sample Cognitive split only; 1,986-sample factual categories not evaluated/published.  
<sup>[2]</sup> *Reported in *Madeleine* (arXiv 2026) on the 401-sample Cognitive split only; 1,986-sample factual categories not evaluated/published.  
<sup>[3]</sup> Table 1 of the LoCoMo-Plus paper ([arXiv:2602.10715](https://arxiv.org/abs/2602.10715), all 2,387 samples, judge: `gemini-2.5-flash`). **LoCoMo Factual Avg** is the sample-weighted mean across the 5 factual categories (1,986 items); **Gap** = LoCoMo Factual Avg &minus; Cognitive.

### Unbiased Analysis &amp; Trade-Offs

- **Clean 6-for-6 Category Sweep with Hybrid CLS + 2-Stage `LLM_INTERNAL`:** Enabling `LLM_INTERNAL` by default (`WindowedMicroSleepConsolidator` + Stage-1 multi-angle expansion + Stage-2 judged delivery + Perirhinal `[Shared photo]` preservation + Cognitive self-contamination stripping) lifted **every single category** over the heuristic-read v3.0 baseline: `Cognitive` **`33.92%` &rarr; `50.12%` (`+16.20pp`)**, `Common-sense` **`39.06%` &rarr; `50.52%` (`58.85%` open, `+11.46–19.79pp`)**, `Single-hop` **`63.32%` &rarr; `71.70%` (`+8.38pp`)**, `Multi-hop` **`42.20%` &rarr; `47.87%` (`+5.67pp`)**, `Adversarial` **`87.67%` &rarr; `91.70%` (`+4.03pp`)**, and `Temporal` **`57.63%` &rarr; `60.59%` (`+2.96pp`)**, raising overall 2,387-sample accuracy from `58.69%` to **`66.65%` (`+190.0` points)**.
- **Why Competing Systems Show Artificial ~15pp Gaps vs. CML's High-Floor Balance:** A-Mem (`35.20%`), SeCom (`31.80%`), and Mem0 (`30.50%`) collapse on the 446 `Adversarial` traps, artificially depressing their `LoCoMo Factual Avg` (`57.24–59.64%`) so `Factual - Cognitive` looks narrow (`14.9–17.2`). CML simultaneously achieves **`91.70%` Adversarial**, **`69.99%` Factual Avg**, AND **`50.12%` Cognitive** (`14.35pp` macro Gap / `19.87pp` weighted Gap).
- **Remaining Headroom on `Single-hop` / `Multi-hop`:** Packet evidence recall is already **`89.2%`** on `single-hop` and **`83.3%`** on `multi-hop`; remaining deductions are primarily `partial` (`0.5`) grades (`148` in `single-hop`, `134` in `multi-hop`) where `gemini-2.5-flash` penalizes minor phrasing differences against short reference strings.

### Reproducing LoCoMo-Plus (2,387 Samples)

```bash
# 1. Build the official 2,387-sample unified dataset
cd evaluation/locomo_plus/data && python unified_input.py && cd ../../..

# 2a. Official Table 1 Cloud QA + Judge (OpenRouter: openai/gpt-4o + google/gemini-2.5-flash)
python evaluation/scripts/run_locomo_openrouter_batch.py \
  --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json \
  --out-dir evaluation/outputs/locomo_plus/cml_v3_gpt4o_gemini25flash_full_2387 \
  --cml-url http://localhost:8000 \
  --skip-ingestion \
  --base-url https://openrouter.ai/api/v1 \
  --model openai/gpt-4o \
  --judge-model google/gemini-2.5-flash \
  --sync-workers 10 \
  --qa-max-tokens 200 \
  --judge-max-tokens 160

# 2b. 100%-Local From-Scratch Run ($0 API cost: qwen3.8-27b)
python evaluation/scripts/run_locomo_openrouter_batch.py \
  --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json \
  --out-dir evaluation/outputs/locomo_plus/cml_v3_qwen38_full_2387 \
  --cml-url http://localhost:8001 \
  --consolidate-windowed \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.8-27b \
  --judge-model qwen3.8-27b \
  --sync-workers 2
```

CLI harness: `cml-eval run-full --repo-root .`

</details>

<details>
<summary><strong>LongMemEval-V2 Benchmark &mdash; Multimodal Web &amp; Enterprise Agent Memory (451 Questions)</strong></summary>

Evaluated across **all 451 questions (`0` errors)** of **LongMemEval-V2** (`LME-V2-Small`: `240` **Web** across customized Magento storefront, Magento Admin CMS, and Reddit/Postmill + `211` **Enterprise** ServiceNow/WorkArena, including `29` multimodal `question_image` screenshots, `100` trajectories / `~25M` tokens per haystack; Wu et al., 2026, [arXiv:2605.12493](https://arxiv.org/pdf/2605.12493.pdf)) using native `CMLMemory` (`evaluation/longmemeval_v2/memory_modules/cml_memory.py`) and a **100%-local `\$0`-API configuration (`qwen3.8-27b` IQ3_S + DFlash 2 on a single 16 GB RTX 5070 Ti)**:

### What Each Ability Measures &amp; How CML Solves It

| Ability Pillar (`451`) | What It Evaluates | CML Score (`qwen3.8-27b` Local) | Key CML Mechanism Driving Performance |
| :--- | :--- | :---: | :--- |
| **Workflow / Procedure (`106` = `74` + `32` `-abs`)** | Recalling exact multi-step UI navigation paths, filter recipes, and shortcuts &mdash; or flagging unsupported workflows (`procedure-abs`). | **61.32%** *(**77.03%** non-abs, **90.48%** Web)* | Synthesized trajectory workflow playbooks + annotated agent thought-action sequences. |
| **Gotchas (`29`)** | Identifying recurring local traps, broken UI affordances, and deployment-specific failure modes (`outcome=FAILURE`). | **51.72%** *(**66.67%** Web)* | Explicit `[FAILED TRAJECTORY / ENVIRONMENT GOTCHA]` callouts + multimodal `question_image` landmark extraction. |
| **Static (`189` = `134` + `55` `-abs`)** | Remembering UI landmarks, mandatory form fields, catalog options, and ServiceNow layouts (`56.76%` Enterprise non-abs). | **44.97%** *(**52.99%** non-abs)* | `2,200`-char sliding-window AXTree indexing (`250`-char overlap) + lossless `StaticText`-echo deduplication (`10.5K` chars). |
| **Dynamic (`127` = `86` + `41` `-abs`)** | Tracking how actions mutate UI state, ticket statuses, banners, and filtered views over time (`51.43%` Enterprise non-abs). | **39.37%** *(**47.67%** non-abs)* | Cross-trajectory state mutation retrieval over Dense `pgvector` + PostgreSQL `tsvector` BM25 + Hypergraph PPR. |
| **Premise Awareness (`128` `-abs`)** | Detecting assumptions that are valid in standard software but false in the customized deployment (`static-abs`, `dynamic-abs`, `procedure-abs`). | **24.22%** *(32.8% `UNKNOWN`)* | Strict haystack query-privacy scoping (`7/7` `test_query_privacy.py` checks) + UI-surface query decomposition. |

### Headline Achievements

- **#1 Non-Coding-Agent Memory System on BOTH `Workflow / Procedure` (`61.32%` full / `77.03%` non-abs / `90.48%` Web) AND `Gotchas` (`51.72%` full / `66.67%` Web):** Running 100% locally at **`\$0.00` API cost**, CML outperforms **RAG + notes (`52.80%` Workflow / `41.40%` Gotchas)** by **`+8.52pp` / `+10.32pp`**, **AgentRunbook-R (`52.80%` / `31.00%`)** by **`+8.52pp` / `+20.72pp`** (at faster `19.65s` vs. `26.90s` query latency), and **Perpetual Recall V3 (`56.60%` / `37.93%`)** by **`+4.72pp` / `+13.79pp`**.
- **Beats Cloud `GPT-5.4-mini` (`xhigh`) Coding Agents on Key Experience Pillars at `5.5–9x` Lower Latency:** Surpasses **Cloud `Codex` (`57.50%` Workflow, `177.2s`)** by **`+3.82pp`** on `Workflow / Procedure` and **Cloud `AgentRunbook-C` (`48.30%` Gotchas, `108.3s`)** by **`+3.42pp`** on `Gotchas`.
- **Single-16GB-GPU Zero-VRAM Co-Location:** Zero-VRAM hybrid retrieval (`CUDA_VISIBLE_DEVICES=""` on CML query server, `~1.1s` `pgvector` + `tsvector` BM25 + PPR) runs alongside exclusive-GPU `qwen3.8-27b` (`51.72%` accuracy on the `29` visual `question_image` items).
- **Committed Summary Artifact:**
  - 100%-Local Zero-API-Cost (`2026-11-27`, `451 / 451` valid, `0` errors, `56.97%` Non-Abstention, `47.67%` Full-Set): [`evaluation/results/longmemeval_v2/longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json`](evaluation/results/longmemeval_v2/longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json)

### Full LongMemEval-V2 Comparison (`LME-V2-Small`, 451 Questions)

| System / Method | Family | Controller / Reader / Judge | Static (189) | Dynamic (127) | **Workflow / Proc. (106)** | **Gotchas (29)** | **Overall (451)** | Latency &darr; | API Cost |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CML (`CMLMemory`, Local `\$0` API)** <sup>[0]</sup> | **Neuro Memory** | **Local `qwen3.8-27b` (`IQ3_S`) Reader + Judge** | 44.97% *(52.99%&dagger;)* | 39.37% *(47.67%&dagger;)* | **61.32%** *(**77.03%&dagger;**)* | **51.72%** *(**66.67%&Dagger;**)* | **47.67%** *(**56.97%&dagger;**)* | **19.65s** *(13.17s&Dagger;)* | **\$0.00** |
| **Perpetual Recall V3** *(Equinox Labs)* <sup>[L]</sup> | Hybrid Retrieval | `Qwen3.5-9B` + Cloud Judge (`gpt-5.2`) | 58.20% | 46.46% | 56.60% | 37.93% | 53.22% | 1.96s | Cloud |
| **AgentRunbook-R** *(Reference)* <sup>[P,L]</sup> | RAG (3-Pool Active) | `Qwen3.5-9B` + `Qwen3-Embed-8B` + `gpt-5.2` | **66.10%** | **58.30%** | 52.80% | 31.00% | **58.60%** | 26.90s | Cloud |
| **RAG: `query -> slice + notes`** *(Ref)* <sup>[P,L]</sup> | RAG Baseline | `Qwen3.5-9B` + `Qwen3-Embed-8B` + `gpt-5.2` | 52.40% | 49.60% | 52.80% | 41.40% | 51.00% | **0.20s** | Cloud |
| **RAG: `query -> slice`** <sup>[P,L]</sup> | RAG Baseline | `Qwen3.5-9B` + `Qwen3-Embed-8B` + `gpt-5.2` | 47.10% | 42.50% | 41.50% | 20.70% | 42.80% | **0.10s** | Cloud |
| **OPS-Agentic-Memory-v1** *(Alibaba)* <sup>[L]</sup> | Coding Agent | Frontier Coding Agent + `gpt-5.2` Judge | **87.83%** | **81.10%** | **76.42%** | **58.62%** | **81.37%** | 153.64s | Cloud |
| **AgentRunbook-C v2 Online (`xhigh`)** <sup>[L]</sup> | Coding Agent | `gpt-5.4-mini` (`xhigh`) + `Qwen3.5-9B` + `gpt-5.2` | &mdash; | &mdash; | &mdash; | &mdash; | 75.61% | 130.54s | Cloud |
| **AgentRunbook-C v2 Online (`low`)** <sup>[L]</sup> | Coding Agent | `gpt-5.4-mini` (`low`) + `Qwen3.5-9B` + `gpt-5.2` | &mdash; | &mdash; | &mdash; | &mdash; | 65.41% | 26.11s | Cloud |
| **AgentRunbook-C** *(Reference)* <sup>[P,L]</sup> | Coding Agent | `gpt-5.4-mini` (`xhigh`) + `Qwen3.5-9B` + `gpt-5.2` | 82.00% | 72.40% | 72.60% | 48.30% | 74.90% | 108.30s | Cloud |
| **Codex** *(Reference)* <sup>[P,L]</sup> | Coding Agent | `gpt-5.4-mini` (`xhigh`) + `Qwen3.5-9B` + `gpt-5.2` | 80.40% | 67.00% | 57.50% | 58.60% | 69.90% | 177.20s | Cloud |
| Kimi-K2.5 *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `Kimi-K2.5` parametric weights only | 18.30% | 19.70% | 11.50% | 17.10% | 14.10% | 0s | Cloud |
| Claude Opus 4.6 *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `Claude Opus 4.6` parametric weights only | 9.60% | 12.10% | 13.40% | 37.90% | 11.80% | 0s | Cloud |
| Gemini-3.1-Pro-Preview *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `Gemini-3.1-Pro` parametric weights only | 10.40% | 9.60% | 14.70% | 24.10% | 11.00% | 0s | Cloud |
| GPT-5.2 *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `GPT-5.2` parametric weights only | 0.00% | 0.00% | 3.20% | 21.00% | 4.70% | 0s | Cloud |
| No Retrieval *(Baseline)* <sup>[P,L]</sup> | Reader Only | `Qwen3.5-9B` (thinking enabled) | 0.00% | 0.80% | 9.40% | 13.80% | 1.30% | 0s | &mdash; |

<sup>[0]</sup> Full 451-question 100%-local run (`qwen3.8-27b` IQ3_S + DFlash 2 Reader + Judge, `0` errors, `\$0.00` API cost). &dagger;Non-abstention accuracy (`323` Qs): **`56.97%` Overall**, **`77.03%` Procedure** (`90.48%` Web), **`52.99%` Static** (`56.76%` Enterprise), **`47.67%` Dynamic** (`51.43%` Enterprise). &Dagger;Web domain (`240` Qs): **`49.17%` Full-Set**, **`59.52%` Non-Abstention**, **`66.67%` Gotchas**, **`13.17s` Query Latency**.  
<sup>[P]</sup> Tables 3 &amp; 9 of Wu et al. (2026), *LongMemEval-V2* ([arXiv:2605.12493](https://arxiv.org/pdf/2605.12493.pdf)). <sup>[L]</sup> Official [LongMemEval-V2 Public Leaderboard](https://xiaowu0162.github.io/longmemeval-v2/#leaderboard) (`Static`, `Dynamic`, and `Workflow/Proc.` fold each category's `-abs` premise items into its pillar: `189 + 127 + 106 + 29 = 451`).

### Unbiased Analysis &amp; Trade-Offs

- **Procedural &amp; Failure-Gotcha Strength (`77.03%` non-abs `procedure`, `90.48%` Web; `51.72%` `gotchas`, `66.67%` Web):** Indexing trajectory-level workflow playbooks with explicit `outcome=FAILURE` gotcha callouts preserves end-to-end action causality that flat state-slice RAG (`20.70%` Gotchas) and compressed note pools (`31.00%` Gotchas in `AgentRunbook-R`) lose.
- **Full-Set (`47.67%`) vs. Non-Abstention (`56.97%`) on 3-Bit Quantized Local Models:** LongMemEval-V2's strict `-abs` judge grades generic `\boxed{UNKNOWN}` as **`0`** (`32.8%` of CML's `-abs` answers) unless the response pinpoints the exact flawed sub-premise (`24.22%` pass rate). Under 3-bit `IQ3_S` quantization, `qwen3.8-27b` conservatively outputs `UNKNOWN` when a queried control is absent from retrieved AXTrees rather than asserting global absence across the deployment.
- **Bounded Indexed Retrieval (`13–27s`) vs. Coding-Agent File Sandboxes (`108–177s`):** Coding-agent methods (`AgentRunbook-C`, `OPS-Agentic-Memory-v1`) reach higher `Static`/`Dynamic` accuracy by running cloud `gpt-5.4-mini` (`xhigh`) bash/grep loops over raw JSON files for `2–3 minutes` per query, whereas CML retrieves in a single bounded pass at `\$0.00` API cost.

### Reproducing LongMemEval-V2 (451 Questions)

```bash
# 1. Download, symlink screenshots, and validate dataset (xiaowu0162/longmemeval-v2)
cd evaluation/longmemeval_v2
python data/download_data.py --data-root data/longmemeval-v2
python data/prepare_data.py --data-root data/longmemeval-v2 --mode symlink
python data/validate_data.py --data-root data/longmemeval-v2 --tier small
cd ../..

# 2. Pre-ingest trajectories on GPU, then run all 451 questions (Web + Enterprise) locally
python evaluation/scripts/run_longmemeval_v2_local.py --tier small --preingest-only
python evaluation/scripts/run_longmemeval_v2_local.py \
  --tier small \
  --methods cml \
  --local-base-url http://127.0.0.1:8023/v1 \
  --local-model qwen3.8-27b \
  --cml-url http://127.0.0.1:8000 \
  --memory-context-max-tokens 16500 \
  --skip-existing \
  --resume
```

</details>

---

## Testing

```bash
# 911 hermetic unit tests — no DB, no LLM, mock embeddings
pytest tests/unit -q

# 348 tests across integration, e2e, and py-cml (unit + embedded + live)
pytest tests/integration tests/e2e packages/py-cml/tests -q
```

Full-stack quality test with LLM-as-judge: `python scripts/test_memory_quality.py` &#8226; Details: [tests/README.md](tests/README.md)

---

## Admin Dashboard

CML ships a built-in zero-dependency single-page admin dashboard at **[http://localhost:8000/dashboard/](http://localhost:8000/dashboard/)**:

| Page | What It Shows |
| :--- | :--- |
| **Overview** | KPI cards (memories, semantic facts, tenants), request sparkline, reconsolidation queue status |
| **Tenants &amp; Sessions** | Per-tenant memory/fact counts, active Redis sessions with TTL |
| **Memory &amp; Facts Explorer** | Searchable memory/fact tables, bulk archive/silence/delete, inline write panel, supersession lineage, JSON export |
| **Knowledge Graph** | Interactive `neovis.js` entity-relation graph explorer with depth control |
| **Retrieval Test** | Interactive query workbench showing scored results, retrieval sources, supersession badges, and lineage |
| **Components &amp; Config** | Live PostgreSQL/Neo4j/Redis health, embedding/server metadata, and inline feature-flag / tuning editor |
| **Management** | One-click Consolidation, Forgetting, and Reconsolidation job triggers with run history |

Authenticate with `AUTH__ADMIN_API_KEY` (default `test-key` in `.env.minimal`). See [Usage Documentation](docs/usage.md) for full API details.

---

## ActiveCML Intrinsic Memory

Alongside `MemoryPacket` context reconstruction, CML includes the tiered **ActiveCML** intrinsic memory subsystem (`src/intrinsic/`, enabled via `FEATURES__INTRINSIC_MEMORY_ENABLED=true`, accessible via `response.intrinsic_injection` in `py-cml`):

- **I-2 Logit Bias (`src/intrinsic/logit_interface.py`):** Synthesizes token-level logit bias maps from high-priority constraints and entities.
- **I-3 Activation Steering (`src/intrinsic/activation_hooks.py`):** `contextvars`-isolated residual-stream steering hooks safe under concurrent async requests.
- **I-4 Synaptic KV-Prefix (`src/intrinsic/prefix_synaptic.py`):** Canonical prefix block synthesis and tiered KV-prefix caching.
- **I-5 / I-8 Controller &amp; Dynamic LoRA Router (`src/intrinsic/controller.py`, `src/intrinsic/lora_router.py`):** Unified intrinsic payload synthesis and tenant/domain LoRA adapter routing.

See [ProjectPlan/ActiveCML/](ProjectPlan/ActiveCML/) for the 10-phase roadmap index.

---

## Documentation

| | Document | Description |
|---|----------|-------------|
| **API** | [Usage &amp; API Reference](docs/usage.md) | Full REST API, configuration, dashboard, runtime modes |
| **SDK** | [Python SDK (`py-cml` 3.0.0)](packages/py-cml/README.md) | Sync, async, and zero-server SQLite embedded Whole-Brain SDK |
| **Eval** | [Evaluation Report](evaluation/EVALUATION_REPORT.md) | Full LoCoMo-Plus (2,387) &amp; LongMemEval-V2 (451) diagnostics, comparisons &amp; methodology |
| **SDK Eval** | [Eval Module](packages/py-cml/docs/evaluation.md) | `cml-eval` CLI, Python API, typed configs |
| **Dev** | [Contributing](CONTRIBUTING.md) | Setup, code standards, PR process |
| **Changelog** | [Release History](CHANGELOG.md) | Full version history (`3.0.0` Whole-Brain release) |

---

## References

<details open>
<summary><strong>Neuroscience Foundations</strong></summary>

1. McClelland, J.L., McNaughton, B.L., &amp; O'Reilly, R.C. (1995). ["Why there are complementary learning systems in the hippocampus and neocortex."](https://doi.org/10.1037/0033-295X.102.3.419) *Psychological Review*, 102(3), 419&ndash;457.
2. Tulving, E. (1983). *Elements of Episodic Memory*. Oxford University Press.
3. Nader, K., Schafe, G.E., &amp; Le Doux, J.E. (2000). ["Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval."](https://doi.org/10.1038/35021052) *Nature*, 406(6797), 722&ndash;726.
4. Shuai, Y., et al. (2010). ["Forgetting is regulated through Rac activity in Drosophila."](https://doi.org/10.1016/j.cell.2009.12.044) *Cell*, 140(4), 579&ndash;589.
5. Han, J.H., et al. (2007). ["Neuronal competition and selection during memory formation."](https://doi.org/10.1126/science.1139438) *Science*, 316(5823), 457&ndash;460.
6. Miller, G.A. (1956). ["The magical number seven, plus or minus two."](https://doi.org/10.1037/h0043158) *Psychological Review*, 63(2), 81&ndash;97.
7. Bartlett, F.C. (1932). *Remembering: A Study in Experimental and Social Psychology*. Cambridge University Press.
12. Rasch, B., &amp; Born, J. (2013). ["About sleep's role in memory."](https://doi.org/10.1152/physrev.00032.2012) *Physiological Reviews*, 93(2), 681&ndash;766.
13. Atkinson, R.C. &amp; Shiffrin, R.M. (1968). "Human memory: A proposed system and its control processes." *Psychology of Learning and Motivation*, 2, 89&ndash;195.
15. Conway, M.A., &amp; Pleydell-Pearce, C.W. (2000). ["The construction of autobiographical memories in the self-memory system."](https://doi.org/10.1037/0033-295X.107.2.261) *Psychological Review*, 107(2), 261&ndash;288.
16. Eichenbaum, H. (2014). ["Time cells in the hippocampus: a new dimension for mapping memories."](https://doi.org/10.1038/nrn3827) *Nature Reviews Neuroscience*, 15(11), 732&ndash;744.
17. Schultz, W., Dayan, P., &amp; Montague, P.R. (1997). ["A neural substrate of prediction and reward."](https://doi.org/10.1126/science.275.5306.1593) *Science*, 275(5306), 1593&ndash;1599.
18. Pfeiffer, B.E., &amp; Foster, D.J. (2013). ["Hippocampal place-cell sequences depict future paths to remembered goals."](https://doi.org/10.1038/nature12112) *Nature*, 497(7447), 74&ndash;79.
19. Berntsen, D. (2009). *Involuntary Autobiographical Memories: An Introduction to the Unbidden Past*. Cambridge University Press.
20. Hasselmo, M.E. (2006). ["The role of acetylcholine in learning and memory."](https://doi.org/10.1016/j.conb.2006.09.002) *Current Opinion in Neurobiology*, 16(6), 710&ndash;715.
21. Aston-Jones, G., &amp; Cohen, J.D. (2005). ["An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance."](https://doi.org/10.1146/annurev.neuro.28.061604.135709) *Annual Review of Neuroscience*, 28, 403&ndash;450.
22. Murray, E.A., &amp; Bussey, T.J. (1999). ["Perceptual-mnemonic functions of the perirhinal cortex."](https://doi.org/10.1016/S1364-6613(99)01303-0) *Trends in Cognitive Sciences*, 3(4), 142&ndash;151.

</details>

<details open>
<summary><strong>AI Memory Frameworks</strong></summary>

8. HippoRAG (2024). ["Neurobiologically Inspired Long-Term Memory for Large Language Models."](https://arxiv.org/abs/2405.14831) *arXiv:2405.14831*.
9. Mem0 (2024). ["Mem0: The Memory Layer for AI Applications."](https://github.com/mem0ai/mem0) GitHub.
10. HawkinsDB (2024). ["HawkinsDB: A Thousand Brains Theory inspired Database."](https://github.com/harishsg993010/HawkinsDB) GitHub.
11. Wu, T., et al. (2024). ["From Human Memory to AI Memory: A Survey."](https://arxiv.org/abs/2404.15965) *arXiv:2404.15965*.
14. Li, Y., et al. (2026). ["Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents."](https://arxiv.org/abs/2602.10715) *arXiv:2602.10715*.
23. Wu, D., et al. (2026). ["LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues."](https://arxiv.org/pdf/2605.12493.pdf) *arXiv:2605.12493*.

</details>

---

<div align="center">

*"Memory is the diary that we all carry about with us."* &mdash; Oscar Wilde

**CML v3.0 transforms that diary into a living whole-brain system that encodes, consolidates, simulates, adapts, and gracefully forgets &mdash; just like we do.**

<br/>

![GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-A42E2B?style=for-the-badge&logo=gnu&logoColor=white)

</div>
