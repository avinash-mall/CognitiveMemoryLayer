# Human-memory redesign plan

Research pass of 2026-08-02. 26 sources, 121 extracted claims, 25 adversarially
verified (3 independent refutation votes each, 2/3 kills a claim). Baseline is the
2026-08-02 LoCoMo-Plus run: **overall 0.4860**, Cognitive 0.254, common-sense 0.276,
multi-hop 0.346, temporal 0.355, single-hop 0.576, adversarial 0.753.

## How to read the evidence tiers

The synthesis stage of the research run died on a session limit, so claims are tiered
here by hand. **Do not treat the tiers as interchangeable** — the biggest headline
number in the whole set (MRAgent, +23% on LoCoMo) is tier 3.

| Tier | Meaning |
| :--- | :--- |
| **1 — confirmed** | Survived 3 independent adversarial refutation votes. |
| **2 — contested** | Verifiers split, or the paper's own data contradicts its prose. |
| **3 — unverified** | Extracted from primary source; its verifier votes never ran. |
| **4 — cognitive science only** | Human-memory evidence, no LLM benchmark behind it. |

Per lever G, our absolute scores use a local Qwen judge and are **not** comparable to
any published number below. Cross-system comparisons (Mem0 92.5, RecMem 81.10) are
context for mechanism choice, never targets to hit.

---

## The finding that reorders everything

**We already have HippoRAG's retrieval mechanism, and we render it as HippoRAG's
documented failure mode.**

`NeocorticalStore.multi_hop_query` (`src/memory/neocortical/store.py:237`) runs
**Personalized PageRank** over the entity graph — the exact mechanism HippoRAG reports
up to 20% multi-hop gains from (tier 1). It takes PPR's top 20 entities, keeps 10, and
then returns `{entity, relations, facts}` — *structured entity records with no source
text*. `packet_builder` renders that as `Entity: user\n - LOCATION: Seattle`.

EcphoryRAG ablates precisely this (tier 1, 3-0):

> The "Entity-Only" method performs poorly [...] the LLM requires the original,
> grounded text from the source chunks to understand nuance and synthesize a
> high-quality answer. This validates that our engrams act as a **precise index**,
> but the text chunks provide the **essential content** for final reasoning.

That is our own measurement, arrived at independently: every arm with a populated graph
scored *below* the empty-graph baseline on multi-hop (0.33 → 0.23/0.27/0.26), and
excluding the prong entirely was worth +0.033 overall. We concluded the graph prong was
useless and switched it off. The literature says the prong was fine and the **rendering**
was wrong — we were feeding the index to the generator instead of using it to select
content.

This also revises STATE.md's standing claim that lever E (an iterative hop loop) is the
highest-value unshipped item. The evidence for that is genuinely split (see item 5), and
the index-vs-content fix is both better-evidenced and far cheaper.

**The back-pointer we need already exists.** Relation edges are written with
`evidence_ids=[str(record.id)]` (`orchestrator.py:862` →
`neocortical/store.py:201`), and the read path returns `properties(r)` including that
field (`neo4j.py:526`). So PPR → entities → edges → episodic record IDs → grounded text
is reachable **with no write-path change and no re-ingestion.** It is a retrieval-only
change, A/B-able on the frozen corpus.

---

## Ordered plan

Cost is quoted in the only unit that matters here — how the change gets measured.
Frozen-corpus A/B is ~55 min and needs the ingestion checkpoint copied into the new
`--out-dir`; a fresh-ingest arm is ~2 h and needs its own `--tenant-prefix`.

### 1. Graph as index, not content — resolve PPR hits to episodic text

*Incremental. Retrieval-only. Frozen-corpus A/B (~55 min). ~40 lines.*

**Human analogue.** Hippocampal index theory: the index is not the memory. A cue
pattern-completes to an index, and the index *reinstates* the cortical trace. You never
recall the pointer.

**Evidence.** Tier 1: EcphoryRAG Entity-Only ablation (above); HippoRAG PPR up to 20%
on multi-hop. Tier 1 from HippoRAG 2: graph methods otherwise cost 5-10 F1 on simple QA,
and the fix is passage-node integration — *not* dropping the graph, which is what we did.
Plus our own three arms.

**Change.** In `_retrieve_graph` (`retriever.py:591`), after `multi_hop_query` returns
PPR-ranked entities, collect `relation_properties["evidence_ids"]` across their edges,
`get_by_ids_batch` the episodic records, and return **those** as the prong's results
(keeping the PPR score, already rank-normalised into [0.55, 0.85], as relevance). The
entity profile becomes a selection key that never reaches the packet.

**Companion fix.** `merge_edges_batch` uses `r += edge.properties` on MATCH
(`neo4j.py:300`), so `evidence_ids` is **overwritten** — each edge points only at the
most recent episode that asserted it. Union it instead. This is a write-path change, so
it needs a fresh-ingest arm; but the frozen corpus already carries one ID per edge, so
item 1 is fully A/B-able without it. Ship the union with the next write-path arm.

**Expected.** multi-hop (0.346), single-hop (0.576). Flip
`FEATURES__GRAPH_RESULTS_IN_PACKET=true` for the arm — the flag exists and its default
is documented as evidence-driven, so flipping it back is a one-line arm.

### 2. Sufficiency gate — a metacognitive signal distinct from retrieval score

*Incremental. Retrieval-only. Frozen-corpus A/B. **Prerequisite for items 1, 4, 5.***

**Human analogue.** Feeling-of-knowing, and the dual-process split between fast coarse
*familiarity* and deliberate *recollection*.

**Why this is not a nice-to-have.** Adversarial is n=446 — 19% of the set — it is our
strongest category (0.753), and **every packet-enriching change has cost us there**. The
−0.029 last run burned ~0.005 overall, roughly a fifth of the +0.023 we netted. Items 1,
4 and 5 all put *more* into the packet. Without this gate they partly cancel themselves.

**Evidence.** Tier 3, but unusually consistent across independent sources: a
sufficient-context classifier improves correct-among-answered by 2-10% (Gemini/GPT/Gemma);
AbstentionBench finds abstention unsolved across 20 frontier LLMs, barely improving with
scale, and reasoning fine-tuning *degrades* it by 24% — so we cannot delegate refusal to
the answering model. RF-Mem shows the cheap version: **mean similarity and entropy of the
first-pass retrieval scores** are enough to gate. Tier 1 caveat from the same body:
partial context still helps even when it doesn't contain the answer, so a hard abstain on
every "insufficient" verdict discards real wins — threshold it, don't binarise it.

**Change.** Compute familiarity from the first-pass score distribution in
`memory_retriever.retrieve` (no LLM), attach it to `MemoryPacket`, surface it in the API
response, and have the eval answer prompt consume it. There is currently **no** abstention
or confidence signal anywhere in the packet or API.

**Expected.** Protects adversarial (0.753) while items 1/4/5 land; secondarily lifts it.

### 3. Temporal contiguity — expand a hit into its conversational neighbourhood

*Incremental. Retrieval-only. Frozen-corpus A/B. ~15 lines.*

**Human analogue.** The temporal contiguity effect: recalling one item preferentially
cues items encoded at nearby positions, and encoding-context reinstatement recovers items
otherwise scored as forgotten (tier 4, Yonelinas 2019).

**Change.** For each surviving top-k vector hit, pull its ±k turn neighbours from the
same session and add them as context. Conversation logs make "adjacent in encoding" exact
and free — it is an ordered `timestamp`/`source_session_id` lookup, no new structure.

**Why early.** Cheapest item in the plan and it attacks two weak categories at once. It
is also the mechanism behind a tier-1 result we should not ignore: agentic-traversal
accuracy *collapses* when agents are restricted to their cited evidence (76%→68%,
80%→28%), i.e. correct multi-hop answers depend on visited-but-uncited neighbourhood
context. Grounded neighbourhood text is doing the work.

**Expected.** temporal (0.355), multi-hop (0.346).

### 4. Bi-temporal fact validity

*Redesign (schema migration). Write-path. Fresh-ingest arm (~2 h).*

**Human analogue.** Event time is encoded separately from encoding time; humans date
events by landmarks and relative order, not by a single stamp.

**Evidence.** Tier 3, strongest per-category numbers in the whole set. Zep's bi-temporal
model (four timestamps per edge: created/expired for ingestion, valid/invalid for when
the fact *held*) plus contradiction-driven edge invalidation lifted temporal 45.1 → 62.4
(+38% relative) and multi-session 44.3 → 57.9. TSM reports up to +12.2pp absolute on
LongMemEval/LoCoMo from organising by *semantic* time rather than dialogue time and
consolidating point-wise memories into durative units. MemoTime: retrieval strategy
selected per temporal operator (before/after, duration, ordering), up to +24%.

**Where we stand.** Episode-level source monitoring already distinguishes said-date from
event-date — that half is shipped and verified at scale (18,490 `event_date` records).
But `semantic_facts` carries a single timestamp and no validity interval, so supersession
is destructive and "what did I believe in March" is unanswerable. STATE.md lever C.

**Change.** Add `valid_from`/`valid_to` to `semantic_facts`; on supersession, close the
old interval instead of deactivating the row. **Do not add the columns before the reader
exists** — that is the write-only bug class we already removed once. Reader first: a
point-in-time fact lookup in the fact prong.

**Expected.** temporal (0.355), and the knowledge-update failure mode generally.

### 5. Adequacy-gated second retrieval pass

*Incremental, but LLM-costly. Retrieval-only. Frozen-corpus A/B, slower arm.*

**Human analogue.** Recollection as a controlled, iterative search that runs only when
familiarity fails to settle the question.

**Evidence is genuinely split — this is why it is item 5, not item 1.**

*For:* DualRAG's reason-then-query loop scores 70.1 vs IRCoT 58.3 on MuSiQue and 84.8 vs
77.2 on 2Wiki (tier 1, 2-0), and survives distillation to a 7B model (58.6 vs 34.0) —
which matters because we run local models. HippoRAG inside an IRCoT loop gives further
substantial gains, so graph and iteration are complementary, not substitutes (tier 1).

*Against:* EcphoryRAG's own depth ablation is **null** — depth 0 scores 0.714 EM, depth 2
scores 0.722, and depths 1 and 3 are *worse* than no expansion at all, a spread inside its
own reported std dev; its prose claims the opposite of its table (tier 1). Worse, that
paper never traverses its graph at query time at all — the "traversal" is ANN search over
an entity-embedding centroid (tier 1). The claim that single-step PPR *matches* IRCoT was
**killed 0-3**, so PPR is not a full substitute. Agentic-GraphRAG's pro-iteration numbers
are tier 2 (contested, 30-question sample). And Zep won its temporal/multi-session gains
with three parallel prongs and rerankers — **no agentic loop**.

**Read.** Iteration is real but is not the cheap win, and the null ablations cluster
around systems whose index quality was already doing the work. Fix the index (item 1),
add neighbourhood grounding (item 3), *then* see whether multi-hop is still short.

**Change when it comes.** Gate the second pass on item 2's familiarity signal —
fast-then-slow, the shape ComoRAG and PRIME use — so latency is paid only where the first
pass was uncertain. Aggregate across passes into an entity-structured outline rather than
concatenating packets (ablating that structure costs 1.7 F1, tier 1).

**Expected.** multi-hop (0.346), Cognitive (0.254).

### 6. Recurrence-gated consolidation

*Incremental. Write-path. Fresh-ingest arm. Primarily a **performance** item.*

**Human analogue.** Consolidation is selective and priority-ordered; replay tags what is
worth keeping rather than transferring everything.

**Evidence.** Tier 3: RecMem defers LLM consolidation until an interaction shows semantic
recurrence (~k=4-5 neighbours at cosine 0.6-0.7), scoring 81.10 on LoCoMo against Mem0's
62.92 while cutting memory-construction tokens **87%** (193.2K vs 1520.8K). Its
co-referent clustering across sessions gives its most consistent category gain on
temporal. Tier 4 support: consolidating unpredictable experience *degrades*
generalisation, so consolidation is justified only where it aids generalisation.

**Why we care.** The write path is our measured bottleneck and the LLM is ~95.8% of write
latency. This is the one item that improves quality and cost together.

**Caveat, stated by the source.** The trigger is two hand-set thresholds needing
per-domain recalibration, and recurrence-as-salience structurally misses
single-occurrence critical items. The raw-retention layer is what covers those — see
below.

### 7. Gist-conditioned detail recovery

*Incremental. Consolidation path. Bundle with item 6's arm.*

Tier 3: a second extraction pass that uses the episodic summary as a reference to find
facts the summary *omitted* beats extracting semantic facts directly from raw dialogue by
5.72 points on LoCoMo. Gist abstraction and detail recovery are complementary, and detail
recovery must be **conditioned on** the gist rather than run independently. Our
consolidation is single-pass gist extraction, so this is a second pass over material we
already cluster.

---

## Considered and rejected

The ask was "redesign if needed." On the evidence, the biggest available redesigns are
the ones to **not** do.

**Do not redesign toward heavier consolidation, and never retire raw episodes.** The
strongest single number recovered in this pass: ablating the raw un-consolidated
interaction layer drops LoCoMo 81.10 → **51.88**, against 70.58 without semantic memory
and 79.94 without episodic memory. Consolidated gists do not cover the evidence needed at
query time — by a factor no other ablation in the set approaches. Independent tier-1
agreement: preserving uncompressed episodic context and re-reading it at recall beats
structure-first designs by +7.8 F1 on LoCoMo multi-hop, with the explicit argument that
embedding/graph compression destroys the contextual dependencies deep reasoning needs.
Tier 4 agreement: detail-rich episodic memory stays permanently hippocampal, and
unpredictable experience should *never* be consolidated. Our retention curve and gist
demotion must keep raw episodes retrievable indefinitely. **Any decay that makes an
episode unretrievable is now a known regression risk, not a feature.**

**Do not build agentic LLM-steered graph traversal as the first move.** See item 5 —
null depth ablations, one paper contradicting its own table, a killed claim, and Zep
winning without a loop.

**Do not replace time-based decay with an interference model yet.** Tier 4 says
forgetting is driven by contextual interference from temporally adjacent events rather
than by consolidation failure — theoretically better than our decay curve, but it is
cognitive-science-only with no benchmark behind it, and it touches the retention path we
just shipped. Item 3 extracts the useful, cheap half of the same finding.

**Do not chase full episodic re-reading.** The +7.8 F1 above costs **~11.25 s per query**
against Mem0's P50 ≤1.1 s. Item 3 is the bounded approximation.

**Do not trust benchmark-driven prioritisation of the write path.** Public memory
benchmarks grade almost exclusively the retrieval step; write/consolidation quality and
per-user isolation under load are barely measured. Our own write-path work being a wash
on LoCoMo-Plus is weak evidence that it does not matter.

---

## Sequencing and measurement

Frozen-corpus arms first, all four independently A/B-able against the same corpus, one
change per arm:

1. **Item 1** (graph as index) — watch multi-hop, single-hop.
2. **Item 2** (sufficiency gate) — watch adversarial; this is the arm that decides
   whether the others can ship without self-cancelling.
3. **Item 3** (contiguity expansion) — watch temporal, multi-hop.
4. **Item 5** (gated second pass), only if multi-hop is still short after 1 and 3.

Then one fresh-ingest arm bundling the write-path items (evidence_ids union, item 4
bi-temporal, items 6-7 consolidation), since they share the ~2 h ingestion cost.

Two standing rules from this session's measurements, both learned the hard way:

- **Subset deltas are a directional screen, not an estimate.** Last run they
  over-predicted by ~1.6× overall and ~3× on Cognitive. Sign held on all six categories;
  magnitude did not.
- **Single-sample diagnosis has failed four times here; distribution measurements have
  held every time.** Sample ≥40 queries before believing any claim about score
  distributions.

Also outstanding and independent of this plan: **arm C**, the prospective-indexing A/B
(`FEATURES__PROSPECTIVE_INDEXING_ENABLED`, currently off). Its stated precondition —
A+B verified non-negative — is now met. It needs two fresh-ingest arms and its own
`--tenant-prefix`; adversarial regression decides the default.
