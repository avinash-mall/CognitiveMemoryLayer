# Beating LoCoMo: a comprehensive engineering playbook for Locomo-Plus

**The Locomo-Plus evaluation results are disappointing because the system's memory pipeline fails on three fundamental axes: retrieval recall for semantically disconnected cues, temporal reasoning without explicit date handling, and multi-hop synthesis across distant sessions.** These weaknesses align precisely with published benchmark patterns — the best baseline system (Gemini-2.5-Pro) achieves only **71.78%** on factual LoCoMo and a dismal **26.06%** on the new cognitive-memory LoCoMo-Plus tasks, while RAG methods score **37–45%** and **12–16%** respectively. However, a recently published system (Kumiho) demonstrates that **93.3% on LoCoMo-Plus** and **0.565 F1 on LoCoMo** are achievable through four architectural innovations: prospective indexing at write time, event-level extraction preserving causal chains, graph-augmented hybrid retrieval, and model-decoupled recall. This report dissects every failure mode, maps each to a concrete fix, and provides the pseudo-code, packages, and prioritized roadmap to close the gap.

---

## 1. Diagnosis: where the current pipeline breaks down

### 1.1 Per-category score breakdown from the LoCoMo-Plus paper

The paper (Li et al. 2026, arXiv 2602.10715) reports LLM-judge accuracy across all tested systems. The table below shows every system evaluated on the original LoCoMo factual benchmark, with per-category breakdowns:

| System | Single-hop | Multi-hop | Temporal | Commonsense | Adversarial | LoCoMo Avg | LoCoMo-Plus |
|--------|-----------|-----------|----------|-------------|-------------|------------|-------------|
| Qwen2.5-3B | 68.3 | 38.7 | 18.4 | 48.4 | 11.7 | 42.2 | 10.8 |
| Qwen2.5-14B | 76.3 | 48.2 | 38.9 | 57.3 | 68.1 | 63.5 | 19.2 |
| gpt-4.1 | 80.3 | 53.9 | 58.9 | 72.9 | 37.3 | 62.2 | 18.6 |
| gpt-4o | 78.1 | 52.3 | 45.8 | 69.8 | 49.0 | 63.0 | 21.1 |
| gemini-2.5-pro | 77.8 | 52.5 | 73.8 | 63.5 | 73.0 | **71.8** | **26.1** |
| RAG (ada-002) | 40.0 | 16.7 | 37.8 | 15.7 | 49.4 | 37.4 | 13.9 |
| RAG (emb-large) | 49.8 | 22.8 | 40.0 | 21.4 | 59.7 | 45.3 | 15.6 |
| Mem0 (GPT-4o) | 80.2 | 48.1 | 39.4 | 66.2 | 30.5 | 57.2 | 15.8 |
| SeCom (GPT-4o) | 77.6 | 50.9 | 42.3 | 71.4 | 31.8 | 57.5 | 14.9 |
| A-Mem (GPT-4o) | 76.9 | 55.6 | 49.3 | 68.1 | 35.2 | 59.6 | 17.2 |
| **Kumiho (GPT-4o)** | — | — | — | — | — | **56.5 F1** | **93.3** |

The Kumiho benchmarks repository reports LoCoMo F1 separately: single-hop **0.462**, multi-hop **0.355**, temporal **0.533**, open-domain **0.290**, adversarial **0.975**, overall **0.565**. On LoCoMo-Plus by constraint type: causal **96.0%**, state **96.0%**, value **96.0%**, goal **85.0%**.

### 1.2 Six failure modes diagnosed from these numbers

**Failure Mode 1 — Multi-hop retrieval collapse.** Multi-hop scores are consistently the worst non-adversarial category. RAG with text-embedding-large achieves only **22.8%** (vs **49.8%** single-hop). The problem is fundamental: standard dense retrieval fetches passages similar to the query, but multi-hop questions require connecting facts from two or more sessions that may share no vocabulary with each other.

**Failure Mode 2 — Temporal reasoning without explicit dates.** Temporal scores vary wildly: **18.4%** for Qwen2.5-3B to **73.8%** for Gemini-2.5-Pro. RAG methods score **34–40%** on temporal, indicating that retrieval finds some relevant passages but the LLM cannot reason about dates, orderings, and intervals embedded in session metadata. Relative time expressions ("last week," "a few months ago") are never resolved to absolute timestamps.

**Failure Mode 3 — Adversarial hallucination.** Small models and memory systems catastrophically hallucinate on adversarial questions (Qwen2.5-3B: **11.7%**, Mem0: **30.5%**). The system retrieves superficially similar passages and the LLM fabricates an answer rather than recognizing the question is unanswerable.

**Failure Mode 4 — Cognitive cue-trigger semantic disconnect.** All systems score **10–26%** on LoCoMo-Plus, a **31–46 point gap** from factual LoCoMo. The benchmark specifically filters out cue-trigger pairs with high BM25 or MPNet similarity, meaning standard embedding retrieval *cannot* bridge the gap between cue and trigger. This is the defining challenge.

**Failure Mode 5 — RAG destroys commonsense/open-domain.** RAG methods score only **14.6–21.4%** on commonsense, far below full-context models (63–74%). Retrieved fragments crowd out the LLM's parametric world knowledge.

**Failure Mode 6 — Answer format mismatch.** The original LoCoMo uses token-level F1 with Porter stemming against ground-truth answers averaging **5.18 tokens**. Verbose LLM outputs (50+ tokens) are systematically penalized. The LoCoMo-Plus paper demonstrates that all string-matching metrics (BLEU, ROUGE, F1) have severe length bias.

---

## 2. What the SOTA does differently: lessons from systems that win

### 2.1 Kumiho's four architectural innovations

The Kumiho system achieves **93.3%** on LoCoMo-Plus (vs 26.1% for Gemini-2.5-Pro) through a dual-store architecture (Redis working memory + Neo4j long-term graph) with four key innovations:

**Prospective indexing** is the single most impactful technique. At write time — when a new conversation turn is ingested — the system generates *future-facing implications* of the current statement. For example, "I cut sugary drinks after my cousin's diabetes diagnosis" would generate implications like "User avoids sugary drinks," "User is health-conscious due to family medical history," "Dietary recommendations should exclude sugary beverages." These prospective indexes bridge the cue-trigger semantic gap because they are semantically aligned with likely future queries even when the original cue is not.

**Event extraction** preserves causal chains that narrative summarization drops. Instead of summarizing "the user discussed health topics," the system extracts atomic events: "User's cousin diagnosed with Type 2 diabetes → User cut sugary drinks from diet." This preserves the *why* behind facts.

**Sibling relevance filtering** applies embedding-based quality control over retrieved context, ensuring only contextually relevant memories reach the answer generator.

**Model-decoupled architecture** separates recall from generation. Recall accuracy is **98.5%** regardless of answer model; end-to-end accuracy scales with model reasoning capacity (88% with GPT-4o-mini, 93.3% with GPT-4o).

### 2.2 Mem0 and Mem0-graph: the update pipeline matters

Mem0 achieves **66.9% LLM-judge accuracy** on LoCoMo (the highest reported before Kumiho on that metric) through a two-phase pipeline. The Extraction Phase ingests three context sources simultaneously — the latest exchange, a rolling conversation summary, and the m most recent messages — then uses an LLM to extract candidate facts. The Update Phase is where the real gains come: new facts are compared against existing memories via semantic similarity, and an LLM-based resolver decides to ADD, UPDATE, DELETE, or SKIP each candidate. This deduplication and conflict resolution step prevents memory noise accumulation.

Mem0-graph adds **~2–3 points on temporal queries** by storing entity-relation triples in Neo4j with explicit temporal edges. The graph structure enables subgraph retrieval and triplet matching, which helps connect multi-hop facts that embedding search alone cannot bridge.

### 2.3 Zep/Graphiti: bi-temporal modeling for temporal reasoning

Graphiti (the engine behind Zep) uses a three-layer hierarchical subgraph — episodic (raw messages with timestamps), semantic (entity nodes + relationship edges), and community (clustered entity groups with summaries). The critical innovation is **bi-temporal modeling**: every edge tracks both when an event occurred (`valid_at`) and when it was ingested (`ingested_at`). Edge invalidation (not deletion) when facts change preserves historical accuracy. Retrieval combines semantic embedding search, BM25, and graph traversal with temporal filtering — "What was true at time T?" becomes a native query.

### 2.4 HippoRAG: Personalized PageRank for multi-hop

HippoRAG achieves **up to 20% improvement** over standard RAG on multi-hop benchmarks (MuSiQue, 2WikiMultiHopQA) through a neurobiologically-inspired architecture. An LLM extracts OpenIE triples from each passage into a knowledge graph. At query time, named entities are extracted from the query, linked to KG nodes, and **Personalized PageRank** distributes probability through the graph to find connected relevant subgraphs. This achieves multi-hop retrieval in a single step — no iterative LLM calls — at 10–20× lower cost and 6–13× faster than IRCoT.

---

## 3. Improvement lever A: better memory extraction

The foundation of any memory system is what gets extracted from conversations. Current approaches that store session summaries or raw dialogue chunks lose critical information.

### 3.1 Atomic event extraction with structured output

Replace session-level summaries with atomic, self-contained event units. Each extracted memory should contain: the fact itself, the speaker, all entities mentioned, temporal metadata (when it happened, when it was stated), and the session/turn source.

```python
# Pseudo-code: Structured memory extraction pipeline
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import instructor
from openai import OpenAI

class ExtractedMemory(BaseModel):
    """A single atomic memory unit extracted from conversation."""
    content: str = Field(description="Self-contained factual statement")
    speaker: str = Field(description="Who stated or implied this")
    subject_entity: str = Field(description="Primary entity this fact is about")
    related_entities: list[str] = Field(default_factory=list)
    event_date: Optional[datetime] = Field(
        None, description="When the event occurred (absolute date)")
    stated_date: datetime = Field(
        description="Session timestamp when this was stated")
    validity_start: Optional[datetime] = Field(
        None, description="When this fact became true")
    validity_end: Optional[datetime] = Field(
        None, description="When this fact stopped being true, if known")
    memory_type: str = Field(
        description="One of: fact, preference, goal, state, value, event")
    causal_chain: Optional[str] = Field(
        None, description="If caused by another event, describe the link")
    confidence: float = Field(ge=0.0, le=1.0)

class MemoryExtractionResult(BaseModel):
    memories: list[ExtractedMemory]

EXTRACTION_PROMPT = """You are a memory extraction system. Given a conversation
session with a timestamp, extract ALL atomic memories. Each memory must be:
1. Self-contained (understandable without the original conversation)
2. Atomic (one fact per memory, not compound statements)
3. Speaker-attributed (who said or implied this)
4. Temporally grounded (resolve "yesterday" to actual dates using session date)
5. Entity-normalized (use consistent entity names across extractions)

For implicit information (goals, preferences, emotional states, values),
extract these as explicit memories with memory_type="goal"/"state"/"value".

Session timestamp: {session_date}
Speakers: {speaker_names}
Conversation:
{conversation_text}
"""

client = instructor.from_openai(OpenAI())

def extract_memories(session_text: str, session_date: datetime,
                     speakers: list[str]) -> list[ExtractedMemory]:
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=MemoryExtractionResult,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(
            session_date=session_date.isoformat(),
            speaker_names=", ".join(speakers),
            conversation_text=session_text
        )}],
        max_retries=3,
    )
    return result.memories
```

### 3.2 Prospective indexing: the key to LoCoMo-Plus

Generate forward-looking implications at write time. This is the technique that drives Kumiho's **93.3%** on LoCoMo-Plus. For each extracted memory, ask the LLM: "What future questions, decisions, or behaviors should this information influence?"

```python
PROSPECTIVE_PROMPT = """Given this memory extracted from a conversation:
"{memory_content}"

Generate 3-5 future-facing implications. Each should describe a scenario
where this memory would be relevant to a future query or decision.
Focus on behavioral constraints, not just factual recall.

Examples of good implications:
- Memory: "User is on a strict diet to lose weight"
  → "When recommending food/restaurants, avoid high-calorie options"
  → "If user asks about indulgent activities, consider diet constraint"
  → "Health and fitness discussions should reference weight loss goal"
"""

class ProspectiveIndex(BaseModel):
    implications: list[str] = Field(
        description="Future scenarios where this memory is relevant")

def generate_prospective_indexes(memory: ExtractedMemory) -> list[str]:
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ProspectiveIndex,
        messages=[{"role": "user", "content": PROSPECTIVE_PROMPT.format(
            memory_content=memory.content
        )}],
    )
    return result.implications
```

Each prospective index is embedded and stored alongside the original memory. At retrieval time, queries match against *both* the original memory embedding *and* all prospective index embeddings. This bridges the cue-trigger semantic disconnect that makes LoCoMo-Plus hard.

### 3.3 Entity-relation extraction for graph storage

For graph-based memory, extract explicit entity-relation triples with temporal metadata:

```python
class Relation(BaseModel):
    subject: str
    predicate: str
    object: str
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None
    source_session: int
    source_turn: int

RELATION_PROMPT = """Extract all entity-relation triples from this conversation.
Format: (Subject, Predicate, Object, ValidFrom, ValidUntil)
Resolve all relative time references to absolute dates.
Mark temporal validity: if a fact replaces a previous one, set ValidUntil
on the old fact.

Session date: {session_date}
Conversation: {text}
"""
```

---

## 4. Improvement lever B: hybrid retrieval with reranking

### 4.1 Three-stage retrieval architecture

The optimal retrieval pipeline combines dense embedding search, sparse keyword search, and graph traversal, followed by cross-encoder reranking:

```python
# Pseudo-code: Hybrid retrieval pipeline
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
import numpy as np

class HybridRetriever:
    def __init__(self):
        self.embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        self.vector_db = QdrantClient(host="localhost", port=6333)
        self.bm25_index = None  # Built during ingestion
        self.graph = None       # Neo4j or NetworkX graph

    def retrieve(self, query: str, top_k: int = 20,
                 final_k: int = 7) -> list[dict]:
        # Stage 1: Parallel retrieval from multiple sources
        dense_results = self._dense_search(query, top_k=top_k)
        sparse_results = self._bm25_search(query, top_k=top_k)
        graph_results = self._graph_search(query, top_k=top_k)
        # Also search prospective indexes
        prospective_results = self._prospective_search(query, top_k=top_k)

        # Stage 2: Reciprocal Rank Fusion
        all_results = self._rrf_merge(
            [dense_results, sparse_results, graph_results,
             prospective_results],
            k=60  # RRF constant
        )

        # Stage 3: Cross-encoder reranking
        candidates = all_results[:top_k * 2]
        pairs = [[query, doc["content"]] for doc in candidates]
        scores = self.reranker.compute_score(pairs)
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:final_k]

    def _rrf_merge(self, result_lists: list[list[dict]],
                   k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion across multiple retriever outputs."""
        scores = {}
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                doc_id = doc["id"]
                if doc_id not in scores:
                    scores[doc_id] = {"doc": doc, "score": 0.0}
                scores[doc_id]["score"] += 1.0 / (k + rank + 1)
        merged = sorted(scores.values(),
                       key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in merged]

    def _dense_search(self, query, top_k):
        embedding = self.embedder.encode([query],
                        return_dense=True)["dense_vecs"][0]
        return self.vector_db.search(
            collection_name="memories",
            query_vector=embedding, limit=top_k)

    def _bm25_search(self, query, top_k):
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.memories[i] for i in top_indices if scores[i] > 0]
```

### 4.2 Query rewriting and HyDE for cognitive memory

For LoCoMo-Plus queries (which have low semantic overlap with their cues), generate a hypothetical memory that *would* answer the question before searching:

```python
HYDE_PROMPT = """Given this query from a user in a conversation:
"{query}"

Imagine you have perfect memory of all past conversations. Write what a
relevant past memory entry would look like that would help answer this
query. Focus on implicit constraints, goals, preferences, or states
the user may have expressed earlier.

Hypothetical memory:"""

def hyde_retrieval(query: str, retriever: HybridRetriever) -> list[dict]:
    # Generate hypothetical document
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user",
                   "content": HYDE_PROMPT.format(query=query)}],
        temperature=0.7,
    )
    hypothetical_memory = response.choices[0].message.content

    # Search with both original query AND hypothetical memory
    results_original = retriever.retrieve(query, top_k=15)
    results_hyde = retriever.retrieve(hypothetical_memory, top_k=15)

    # Merge via RRF
    return retriever._rrf_merge([results_original, results_hyde])[:7]
```

### 4.3 Multi-query expansion

Generate multiple reformulations of the query to capture different semantic aspects:

```python
MULTI_QUERY_PROMPT = """Generate 3 different reformulations of this query
that capture different aspects of what information might be needed:
Query: "{query}"

Reformulation 1 (direct factual): ...
Reformulation 2 (implicit/behavioral): ...
Reformulation 3 (temporal/contextual): ...
"""
```

---

## 5. Improvement lever C: graph-based memory with temporal edges

### 5.1 Temporal-aware memory store schema

```python
# Neo4j schema for temporal knowledge graph
# Using Graphiti-inspired bi-temporal model

SCHEMA_CYPHER = """
// Entity nodes
CREATE CONSTRAINT entity_name IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.name IS UNIQUE;

// Entity properties: name, type, summary, embedding (vector index)
// Relation edges with temporal validity
// (:Entity)-[:RELATES_TO {
//   predicate: "works_at",
//   valid_at: datetime,    -- when this became true
//   invalid_at: datetime,  -- when this stopped being true (null = current)
//   ingested_at: datetime, -- when we learned this
//   source_session: int,
//   source_turn: int,
//   embedding: [float],    -- for semantic search over edges
//   confidence: float
// }]->(:Entity)

// Episode nodes for raw message storage
// (:Episode {
//   session_id: int,
//   turn_id: int,
//   timestamp: datetime,
//   speaker: string,
//   content: string,
//   embedding: [float]
// })

// Link episodes to entities they mention
// (:Episode)-[:MENTIONS]->(:Entity)
"""

# Python class for temporal memory operations
class TemporalMemoryGraph:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def add_fact(self, subject: str, predicate: str, obj: str,
                 valid_at: datetime, source_session: int,
                 embedding: list[float]):
        """Add a new fact, invalidating contradictory existing facts."""
        with self.driver.session() as session:
            # Check for existing contradictory facts
            existing = session.run("""
                MATCH (s:Entity {name: $subject})-[r:RELATES_TO]->(o:Entity)
                WHERE r.predicate = $predicate AND r.invalid_at IS NULL
                RETURN r, o.name as object_name
            """, subject=subject, predicate=predicate).data()

            # Invalidate contradictory facts (don't delete!)
            for fact in existing:
                if fact["object_name"] != obj:
                    session.run("""
                        MATCH (s:Entity {name: $subject})
                              -[r:RELATES_TO {predicate: $pred}]->
                              (o:Entity {name: $old_obj})
                        WHERE r.invalid_at IS NULL
                        SET r.invalid_at = $now
                    """, subject=subject, pred=predicate,
                       old_obj=fact["object_name"], now=valid_at)

            # Add new fact
            session.run("""
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                CREATE (s)-[:RELATES_TO {
                    predicate: $predicate,
                    valid_at: $valid_at,
                    invalid_at: null,
                    ingested_at: datetime(),
                    source_session: $session_id,
                    embedding: $embedding
                }]->(o)
            """, subject=subject, object=obj, predicate=predicate,
               valid_at=valid_at, session_id=source_session,
               embedding=embedding)

    def query_at_time(self, entity: str, reference_time: datetime,
                      hop_depth: int = 2) -> list[dict]:
        """Retrieve all facts about an entity valid at a specific time."""
        with self.driver.session() as session:
            return session.run("""
                MATCH (s:Entity {name: $entity})-[r:RELATES_TO*1.."""
                + str(hop_depth) + """]->(o:Entity)
                WHERE r.valid_at <= $time
                  AND (r.invalid_at IS NULL OR r.invalid_at > $time)
                RETURN s.name, r.predicate, o.name, r.valid_at
            """, entity=entity, time=reference_time).data()
```

### 5.2 Personalized PageRank for multi-hop retrieval

Adapted from HippoRAG's approach, using NetworkX for prototyping:

```python
import networkx as nx
import numpy as np

class PPRRetriever:
    """Personalized PageRank over memory graph for multi-hop retrieval."""

    def __init__(self, graph: nx.DiGraph, embedder):
        self.graph = graph
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        # Step 1: Extract entities from query
        query_entities = self._extract_entities(query)

        # Step 2: Link query entities to graph nodes
        seed_nodes = {}
        for entity in query_entities:
            best_node, score = self._link_to_graph(entity)
            if score > 0.7:  # similarity threshold
                seed_nodes[best_node] = score

        if not seed_nodes:
            return []  # Fall back to dense retrieval

        # Step 3: Normalize seed weights to create personalization vector
        total = sum(seed_nodes.values())
        personalization = {n: 0.0 for n in self.graph.nodes()}
        for node, weight in seed_nodes.items():
            personalization[node] = weight / total

        # Step 4: Run Personalized PageRank
        ppr_scores = nx.pagerank(
            self.graph,
            alpha=0.5,  # teleport probability (lower = more exploration)
            personalization=personalization,
            max_iter=100,
            tol=1e-6
        )

        # Step 5: Rank passages by aggregated node scores
        # Each passage is linked to the nodes it mentions
        passage_scores = {}
        for node, score in ppr_scores.items():
            if "source_passages" in self.graph.nodes[node]:
                for passage_id in self.graph.nodes[node]["source_passages"]:
                    passage_scores[passage_id] = (
                        passage_scores.get(passage_id, 0) + score
                    )

        ranked = sorted(passage_scores.items(),
                       key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in ranked[:top_k]]

    def _link_to_graph(self, entity_text: str):
        """Link entity text to nearest graph node via embedding."""
        entity_emb = self.embedder.encode([entity_text])[0]
        best_node, best_score = None, -1
        for node in self.graph.nodes():
            if "embedding" in self.graph.nodes[node]:
                sim = np.dot(entity_emb, self.graph.nodes[node]["embedding"])
                if sim > best_score:
                    best_score = sim
                    best_node = node
        return best_node, best_score
```

---

## 6. Improvement lever D: temporal reasoning fixes

Temporal reasoning is the category with the highest variance across systems — from **18.4%** (Qwen2.5-3B) to **73.8%** (Gemini-2.5-Pro). The key insight from systems that score well on temporal: **every memory must carry absolute timestamps, and the retrieval system must support temporal filtering natively.**

### 6.1 Resolve all relative time references at extraction time

During memory extraction, convert every relative time expression to an absolute date using the session timestamp as anchor:

```python
TEMPORAL_RESOLUTION_PROMPT = """Given this conversation excerpt from
session dated {session_date}:
"{text}"

Identify ALL time references (explicit and implicit) and resolve them:
- "yesterday" → {session_date - 1 day}
- "last week" → week of {computed date range}
- "recently" → within 7-14 days before {session_date}
- "a few months ago" → approximately {session_date - 3 months}
- No time reference → default to session date

Return each memory with its resolved absolute timestamp."""
```

### 6.2 Timeline retrieval for temporal questions

When a temporal question is detected, retrieve a chronological timeline rather than a similarity-ranked list:

```python
def temporal_retrieval(query: str, entity: str,
                       memory_graph: TemporalMemoryGraph) -> str:
    """Retrieve a chronological timeline for temporal reasoning."""
    # Get all facts about the entity, ordered by time
    all_facts = memory_graph.query_all_facts(entity)
    all_facts.sort(key=lambda f: f["valid_at"])

    # Format as explicit timeline for the LLM
    timeline = "CHRONOLOGICAL TIMELINE:\n"
    for fact in all_facts:
        status = "CURRENT" if fact["invalid_at"] is None else "SUPERSEDED"
        timeline += (f"[{fact['valid_at'].strftime('%Y-%m-%d')}] "
                    f"({status}) {fact['subject']} {fact['predicate']} "
                    f"{fact['object']}\n")
    return timeline
```

### 6.3 Temporal metadata filtering in retrieval

Add temporal filters to the vector search so that queries like "What was X doing in March?" only retrieve memories from that time period:

```python
# Qdrant temporal filtering example
from qdrant_client.models import Filter, FieldCondition, Range

def temporal_filtered_search(query_embedding, start_date, end_date):
    return vector_db.search(
        collection_name="memories",
        query_vector=query_embedding,
        query_filter=Filter(must=[
            FieldCondition(key="valid_at",
                          range=Range(gte=start_date.isoformat())),
            FieldCondition(key="valid_at",
                          range=Range(lte=end_date.isoformat())),
        ]),
        limit=20
    )
```

---

## 7. Improvement lever E: multi-hop iterative retrieval

### 7.1 IRCoT-style iterative retrieval loop

For multi-hop questions, alternate between reasoning and retrieval:

```python
def iterative_retrieval_cot(question: str, retriever: HybridRetriever,
                            max_steps: int = 4) -> tuple[str, list[dict]]:
    """IRCoT: Interleaving Retrieval with Chain-of-Thought."""
    collected_passages = []
    cot_so_far = ""

    # Initial retrieval with original question
    initial_results = retriever.retrieve(question, top_k=5)
    collected_passages.extend(initial_results)

    for step in range(max_steps):
        # Generate next CoT step
        cot_prompt = f"""Question: {question}
Retrieved context: {format_passages(collected_passages)}
Chain of thought so far: {cot_so_far}

Generate the next reasoning step. If you can answer, write "Answer: ..."
Otherwise, write what information is still needed."""

        response = llm_call(cot_prompt)

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip(), collected_passages

        cot_so_far += f"\nStep {step+1}: {response}"

        # Use the new CoT step as a retrieval query
        new_results = retriever.retrieve(response, top_k=3)
        collected_passages.extend(new_results)

        # Deduplicate
        seen_ids = set()
        unique = []
        for p in collected_passages:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                unique.append(p)
        collected_passages = unique

    # Final answer after max steps
    final_prompt = f"""Question: {question}
All retrieved context: {format_passages(collected_passages)}
Reasoning: {cot_so_far}
Provide your final answer:"""
    return llm_call(final_prompt), collected_passages
```

### 7.2 Question decomposition for explicit multi-hop

```python
DECOMPOSE_PROMPT = """Break this question into simpler sub-questions that
can each be answered independently. Return 2-4 sub-questions.

Question: {question}

Sub-questions:
1. ...
2. ...
"""

def decomposed_retrieval(question: str, retriever: HybridRetriever):
    # Decompose into sub-questions
    sub_questions = llm_decompose(question)
    sub_answers = []

    for sq in sub_questions:
        results = retriever.retrieve(sq, top_k=5)
        answer = llm_answer(sq, results)
        sub_answers.append({"question": sq, "answer": answer,
                           "evidence": results})

    # Synthesize final answer
    synthesis_prompt = f"""Original question: {question}
Sub-question answers:
{format_sub_answers(sub_answers)}

Synthesize these into a final answer:"""
    return llm_call(synthesis_prompt)
```

---

## 8. Improvement lever F: prompt engineering for answer generation

### 8.1 Category-aware answering prompt

```python
ANSWERING_PROMPT = """You are a conversational memory assistant. Answer the
question using ONLY the provided context from past conversations.

RULES:
1. Be CONCISE. Answer in 1-2 sentences maximum. Aim for under 10 words
   when a short factual answer suffices.
2. If the context contains timestamps, use them for temporal reasoning.
   Think step-by-step about dates and orderings.
3. If the context does NOT contain information to answer the question,
   say exactly: "I don't have information about that from our previous
   conversations."
4. Attribute information to the correct speaker.
5. If the question involves comparing or connecting multiple pieces of
   information, explicitly reason through each piece.

CONTEXT (chronologically ordered with timestamps):
{formatted_context}

QUESTION: {question}

ANSWER:"""
```

### 8.2 Answer format alignment with F1 scoring

For the original LoCoMo F1 metric, brevity is critical. The ground-truth answers average **5.18 tokens**. A post-processing step that extracts the core answer helps enormously:

```python
EXTRACT_SHORT_ANSWER = """Given this full answer to a question, extract
ONLY the core factual answer in as few words as possible.

Question: {question}
Full answer: {verbose_answer}

Core answer (1-10 words):"""

def post_process_for_f1(question: str, answer: str) -> str:
    """Compress verbose LLM answer for F1-friendly format."""
    short = llm_call(EXTRACT_SHORT_ANSWER.format(
        question=question, verbose_answer=answer))
    return short.strip().rstrip(".")
```

### 8.3 Adversarial handling

For adversarial questions, the system must recognize when retrieved context does *not* actually answer the question. Add a verification step:

```python
VERIFY_PROMPT = """Given this question and retrieved context, determine:
Does the context ACTUALLY contain information that answers this
specific question? Or does it merely contain related but non-answering
information?

Context: {context}
Question: {question}

Answer ONLY "ANSWERABLE" or "UNANSWERABLE" with a brief reason."""
```

---

## 9. Improvement lever G: evaluation harness sanity checks

### 9.1 Common evaluation bugs

**Off-by-one session indexing.** Verify that session indices in the evaluation data match your conversation ingestion pipeline. A mismatch means retrieved context comes from wrong sessions.

**Judge prompt mismatch.** The LoCoMo-Plus paper uses **gemini-2.5-flash** as primary judge with a constraint-consistency protocol, not a string-matching protocol. If your harness uses a different judge prompt, scores may not be comparable. The original LoCoMo uses token-level F1 with Porter stemming (implemented in `evaluation.py` in the snap-research/locomo repository).

**Task disclosure inflation.** The LoCoMo-Plus paper shows that explicitly telling models the task type ("this is a temporal reasoning question") inflates scores, especially for temporal and adversarial categories. Ensure your evaluation uses unified dialogue input without task disclosure.

### 9.2 Debugging checklist

```
□ Per-category score breakdown computed (not just overall)
□ Retrieval recall@k measured independently of generation quality
    → For each question, check if the correct evidence passage is
      in the top-k retrieved results
□ Oracle-context upper bound: feed ground-truth evidence to the LLM
    → If oracle score is low, the problem is generation, not retrieval
    → If oracle score is high but real score is low, focus on retrieval
□ Judge-consistency check: run judge on 50 samples twice, measure agreement
□ Length distribution of generated answers vs ground-truth answers
    → If your answers average 50+ tokens and GT averages 5, that's your
      F1 problem
□ Speaker attribution accuracy: spot-check 20 answers for correct speaker
□ Temporal answer accuracy: spot-check 20 temporal answers for correct dates
□ Adversarial detection rate: what fraction of adversarial questions does
    the system correctly refuse to answer?
□ Session index alignment: verify session IDs in eval data match ingestion
□ Memory count per conversation: are all sessions being ingested?
□ Embedding model consistency: same model for ingestion and retrieval?
```

### 9.3 Oracle-context experiment

This is the single most diagnostic experiment you can run:

```python
def oracle_experiment(dataset, llm):
    """Measure answer quality with perfect retrieval."""
    results = {}
    for question in dataset:
        # Use the ground-truth evidence passages, not retrieved ones
        oracle_context = question["evidence_dialogues"]
        answer = llm.answer(question["text"], oracle_context)
        score = compute_f1(answer, question["gold_answer"])
        results[question["category"]].append(score)

    # If oracle F1 > 0.8: your LLM can answer; fix retrieval
    # If oracle F1 < 0.5: your LLM struggles; fix prompts/model
    return {cat: np.mean(scores) for cat, scores in results.items()}
```

---

## 10. LoCoMo question categories mapped to improvement techniques

| Category | Current Weakness | Primary Fix | Secondary Fix | Expected Gain |
|----------|-----------------|-------------|---------------|---------------|
| **Single-hop** | RAG: 40–50%, Memory: 76–80% | Better embeddings (BGE-M3), cross-encoder reranking | Atomic memory extraction | +5–10 F1 pts |
| **Multi-hop** | RAG: 17–23%, Memory: 48–56% | PPR graph traversal, IRCoT iterative retrieval | Question decomposition, memory linking (A-MEM style) | +10–20 F1 pts |
| **Temporal** | RAG: 35–40%, Memory: 39–49% | Bi-temporal modeling, timestamp metadata on every memory, timeline retrieval | Relative→absolute time resolution at extraction, temporal filtering | +15–25 F1 pts |
| **Open-domain** | RAG: 15–21%, Memory: 66–71% | Preserve parametric knowledge by not over-stuffing context | Minimal retrieved context + explicit "use your world knowledge" instruction | +10–15 F1 pts |
| **Adversarial** | RAG: 49–60%, Memory: 31–35% | Answer verification step, confidence thresholding | Train system to output "no information" when retrieval confidence is low | +20–40 F1 pts |
| **Cognitive (LoCoMo-Plus)** | All systems: 10–26% | **Prospective indexing at write time** | Event extraction, HyDE retrieval, causal chain preservation | +30–60 accuracy pts |

---

## 11. Recommended Python package stack

| Layer | Package | Purpose | Install |
|-------|---------|---------|---------|
| **Embeddings** | `FlagEmbedding` (BGE-M3) | Dense + sparse + ColBERT in one model, 8192 tokens, multilingual | `pip install FlagEmbedding` |
| **Reranking** | `FlagEmbedding` (bge-reranker-v2-m3) | Cross-encoder reranking | Same package |
| **Sparse retrieval** | `rank_bm25` | BM25 keyword search | `pip install rank_bm25` |
| **Vector DB** | `qdrant-client` | Production vector store with payload filtering, hybrid search | `pip install qdrant-client` |
| **Graph DB** | `neo4j` | Temporal KG with Cypher queries, vector indexes | `pip install neo4j` |
| **Graph (prototyping)** | `networkx` | PPR, graph algorithms | `pip install networkx` |
| **Structured extraction** | `instructor` | Pydantic-validated LLM outputs with retries | `pip install instructor` |
| **LLM gateway** | `litellm` | Unified API for 100+ providers | `pip install litellm` |
| **NER/NLP** | `spacy` + `en_core_web_trf` | Entity extraction, dependency parsing | `pip install spacy` |
| **Prompt compression** | `llmlingua` | 5–20× compression with minimal quality loss | `pip install llmlingua` |
| **RAG evaluation** | `ragas` | Faithfulness, answer relevancy, context precision/recall | `pip install ragas` |
| **Framework (optional)** | `llama-index` | RAG orchestration, HyDE, multi-query, agent memory | `pip install llama-index` |
| **Prompt optimization** | `dspy` | Programmatic prompt compilation and optimization | `pip install dspy` |

---

## 12. Prioritized roadmap with expected impact

### Quick wins (1–2 days each)

**1. Answer format compression.** Add a post-processing step that extracts short factual answers from verbose LLM outputs. Ground-truth LoCoMo answers average 5.18 tokens. This alone can boost F1 by **+5–10 points** across all categories with zero retrieval changes.

**2. Adversarial verification gate.** Add a binary verification step that checks whether retrieved context actually answers the question. Systems like Kumiho achieve **0.975 F1** on adversarial by correctly refusing to answer. Expected impact: **+20–40 F1 points** on adversarial category.

**3. Switch embeddings to BGE-M3.** Replace text-ada-embedding-002 or text-embedding-small with BGE-M3, which provides dense, sparse, and ColBERT representations in a single model. The LoCoMo-Plus paper shows RAG with text-embedding-large (45.3% avg) vs text-ada-002 (37.4%), a **+8 point** gain from better embeddings alone.

**4. Add BM25 + RRF fusion.** Add BM25 as a second retrieval pathway with Reciprocal Rank Fusion. BM25 captures exact keyword matches that dense retrieval misses, particularly for names and specific terms. Expected: **+3–5 F1 points** overall.

### Medium effort (1 week each)

**5. Atomic memory extraction with structured output.** Replace session summaries with atomic fact extraction using Instructor + Pydantic schemas. Each memory gets speaker attribution, temporal metadata, and entity tags. Expected: **+5–10 F1 points** across all categories, with largest gains on temporal.

**6. Temporal metadata and timeline retrieval.** Timestamp every extracted memory with absolute dates (resolving relative references). Add temporal filtering to retrieval. For temporal questions, generate a chronological timeline as context. The Mem0-graph paper shows **+3 points** on temporal from graph structure alone; full temporal modeling should yield **+10–15 F1 points** on temporal.

**7. Cross-encoder reranking.** Add bge-reranker-v2-m3 as a reranking stage after initial retrieval. Cross-encoders attend jointly to query and document, catching relevance signals that bi-encoders miss. Expected: **+3–7 F1 points** overall.

**8. Prospective indexing for LoCoMo-Plus.** Implement write-time generation of future-facing implications. This is the single highest-impact technique for cognitive memory. Kumiho demonstrates **+67 points** over Gemini-2.5-Pro on LoCoMo-Plus. Even a basic implementation should yield **+20–40 accuracy points** on LoCoMo-Plus.

### Larger rewrites (2–4 weeks)

**9. Full knowledge graph with Neo4j.** Build a Graphiti-style temporal knowledge graph with entity nodes, relation edges, bi-temporal validity, and edge invalidation. Enable graph traversal at retrieval time. Expected: **+5–10 F1 points** overall, with largest gains on multi-hop and temporal.

**10. Personalized PageRank multi-hop retrieval.** Implement HippoRAG-style PPR over the memory graph. Extract query entities, seed PageRank, and retrieve passages via graph connectivity rather than embedding similarity alone. Expected: **+10–15 F1 points** on multi-hop.

**11. IRCoT iterative retrieval.** Implement the interleaving retrieval + chain-of-thought loop for complex multi-hop questions. Detect multi-hop questions and route them through the iterative pipeline. Expected: **+5–10 F1 points** on multi-hop (cumulative with PPR).

**12. End-to-end evaluation pipeline overhaul.** Implement per-category scoring, oracle-context experiments, retrieval recall@k measurement, judge consistency checks, and automated regression testing. This doesn't improve scores directly but prevents false improvements and identifies the true bottleneck at each stage.

---

## 13. The most important insight: prospective indexing changes everything

Across all the research analyzed, one technique stands out as transformative: **generating forward-looking implications at memory write time.** Standard memory systems store what was said; prospective indexing stores what it *means for the future*. This single change is what enables Kumiho to achieve **98.5% recall accuracy** on LoCoMo-Plus, where the cue and trigger have deliberately low semantic similarity.

The mechanism is straightforward. When the system ingests "Since my cousin got diagnosed with Type 2 diabetes, I cut sugary drinks completely out of my diet," it generates and indexes implications like "User avoids sugary beverages for health reasons," "Dietary recommendations should exclude high-sugar options," and "Family health history influences user's dietary choices." When a future trigger query arrives — perhaps "Should I try that new bubble tea place?" — the prospective index "Dietary recommendations should exclude high-sugar options" has high semantic similarity to the query even though the original cue about the cousin's diabetes does not.

This is not merely clever prompt engineering. It represents a fundamental shift from *reactive retrieval* (find what matches the query) to *proactive indexing* (anticipate what queries will need this information). Combined with event extraction that preserves causal chains, this approach closes the 67-point gap between baseline systems and human-level performance on cognitive memory tasks. The estimated cost is modest: **~$14 for 401 LoCoMo-Plus entries** using GPT-4o-mini for the indexing pipeline and GPT-4o only for final answer generation.

For teams running Locomo-Plus evaluations today, the recommended starting point is implementing prospective indexing and atomic event extraction, then adding hybrid retrieval with temporal filtering. These changes address the three dominant failure modes simultaneously: semantic disconnect (prospective indexing), temporal reasoning (timestamped memories + timeline retrieval), and multi-hop chaining (graph-augmented retrieval). The quick wins of answer compression and adversarial gating can be implemented in parallel for immediate F1 gains on the original LoCoMo metric.