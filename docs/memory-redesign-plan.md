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

**We have an entity index, and we render it as the documented failure mode.**

`NeocorticalStore.multi_hop_query` (`src/memory/neocortical/store.py:237`) ranks entities
from the query's seeds, keeps the top 10, and returns `{entity, relations, facts}` —
*structured entity records with no source text*. `packet_builder` renders that as
`Entity: user\n - LOCATION: Seattle`.

**Correction, found while implementing this.** An earlier draft of this document said
the prong "runs Personalized PageRank", the mechanism HippoRAG reports up to 20%
multi-hop gains from (tier 1). It does not. `personalized_pagerank` only reaches GDS
when the plugin is installed, and it is not installed here — `CALL gds.list()` errors.
Every query takes the fallback: an unbounded `(seed)-[*1..3]-(related)` path count,
which is also where the unbounded scores (315, 744) came from. So the *ranking* is
proximity-by-path-count, not PageRank. Real PPR is still an available upgrade; installing
GDS would make the docstring true and is the cheapest way to test whether better cue
ranking is worth anything on top of the fix below.

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

**The back-pointer we need already exists, and we throw it away.** Relation edges are
written with `evidence_ids=[str(record.id)]` (`orchestrator.py:862` →
`neocortical/store.py:201`), the read path returns `properties(r)` including that field
(`neo4j.py:526`), and `multi_hop_query` passes it through untouched — then
`_format_entity_info` (`retriever.py:799-804`) reads only `predicate` and
`related_entity` and **discards `relation_properties`**. The join key reaches the
retriever and is dropped one function short of being useful.

Measured on the `full2` corpus (2026-08-02): **583,346 edges, of which 464,511 (79.6%)
carry a non-empty `evidence_ids`**, and zero carry an empty list. The 118,835 NULLs are
the fact-sync path (`neocortical/store.py:330-340`), which passes no properties at all.
So PPR → entities → edges → episodic record IDs → grounded text is reachable **with no
write-path change and no re-ingestion**, on four fifths of the graph.

---

## Ordered plan

Cost is quoted in the only unit that matters here — how the change gets measured.
Frozen-corpus A/B is ~55 min and needs the ingestion checkpoint copied into the new
`--out-dir`; a fresh-ingest arm is ~2 h and needs its own `--tenant-prefix`.

### 1. Graph as index, not content — resolve entity hits to episodic text

*Incremental. Retrieval-only. **Measured: the first version lost 0.057.** Flag off.*

> **Result, 2026-08-02.** Full 2,387-sample frozen-corpus arm, only this change:
> **0.4860 → 0.4292**. Every factual category fell — single-hop −0.097, temporal −0.098,
> multi-hop −0.090, common-sense −0.063, Cognitive −0.060 — and adversarial *rose*
> +0.074, which is the standing signature of a packet so degraded the model refuses more.
> Median context grew 1583 → 2741 characters.
>
> **The resolution was not the problem; the ranking was.** Graph hits carried the
> traversal score, which the retriever normalises into a constant 0.55–0.85 band. The
> median vector cosine is ~0.62, so roughly half of every graph batch outranked the
> median genuine match *regardless of the question*, and 10–15 of them were injected into
> a packet with 5–8 episode slots. The prong stopped emitting neighbourhood summaries and
> started emitting well-ranked irrelevant episodes instead — a better failure, still a
> failure. This is the "post-rerank relevance is a constant" finding from item 2 doing
> real damage.
>
> **Fix measured, and it works.** Graph-resolved records are now scored by cosine against
> the query, so the graph decides *candidacy* and similarity decides *rank* — which is
> what "entity structures are an index, text is the content" should have meant all along.
> Arm `item1b`: **0.4860 → 0.5046 (+0.019)**, the best overall score recorded on this
> corpus, and +0.075 against the band-scored arm. Cognitive +0.040, single-hop +0.038,
> multi-hop +0.007, temporal +0.002; common-sense −0.021 and adversarial only −0.009,
> a far smaller refusal trade than any previous packet-enriching change.
>
> **Attributed.** `item1b` bundled two changes — its server ran `7ac29fa`, which also
> carries temporal contiguity (item 3, default on). Arm `item1c` (graph on, contiguity
> off) splits them: **graph +0.0119, contiguity +0.0067**, roughly additive. The graph
> number to quote is **+0.012**, not +0.019.
>
> Graph-only *raises* adversarial (+0.011). The refusal trade shows up only with
> contiguity — which fits: contiguity adds more context, while cosine-ranked graph
> results mostly reorder context that was already competing for the same slots.
>
> Both flags now default on. `FEATURES__GRAPH_RESULTS_IN_PACKET` was flipped three times
> across this investigation, each time on a measured arm and never on an argument.
>
> **Real PageRank is not the missing piece — arm `item1d`.** GDS 2.13.4 was deployed and
> verified genuinely running (bounded PPR scores through the real code path, zero leaked
> projections), then measured: **0.5031 against the traversal fallback's 0.5046**, a
> −0.0015 gap that is under 4 samples out of 2,387. Meanwhile read latency went
> ~174 ms → ~700 ms, and the first call per worker took 2554 ms, over the 2 s step budget.
> Four to five times the cost for nothing measurable.
>
> This is the clearest confirmation yet of the tier-1 EcphoryRAG result that opened this
> document: **cue quality and text grounding carry the multi-hop gain, not traversal
> sophistication.** Swapping a 2-hop path count for real Personalized PageRank — the exact
> mechanism HippoRAG reports up to 20% from — moved nothing, while changing how the
> resolved text was *ranked* moved +0.075. `FEATURES__GRAPH_PAGERANK_ENABLED` defaults
> **off**; the plugin stays installed and the code path stays correct, so this is one env
> var to re-measure if entity extraction or graph density changes.
>
> **Do not read this as "the literature was wrong."** The Entity-Only ablation is about
> what reaches the generator, and grounded text still beats entity profiles there. What
> this arm adds is that index-driven *recall* must not become index-driven *ranking*.

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

**Companion fix — shipped 2026-08-03.** `merge_edges_batch` used `r += edge.properties`
on MATCH, so `evidence_ids` was **overwritten** — each edge pointed only at the most
recent episode that asserted it. `merge_edge` had the identical bug, which this note
missed by naming only the batch path. Both now union, dedupe, and keep the 50 most recent
IDs; an edge that never carries evidence (the fact-sync path, 118,835 edges) still reads
NULL rather than `[]`. Verified against live Neo4j 5.26. It changes nothing on the frozen
corpus, which already carries one ID per edge — the gain needs re-ingestion to appear.

**Watch the packet slots when flipping the flag.** The graph floor (0.55) sits *above*
`episode_relevance_threshold` (0.4) and graph hits become `EPISODIC_EVENT`
(`_dict_to_record` falls back to that type), so **every** surviving graph hit clears the
episode filter and competes for the 5-or-8 episode slots. That is the mechanism by which
the prong emptied Recent Events before. Resolving to real episodic records makes those
slots worth spending, but the arm should check episode-slot occupancy, not just the score.

**Expected.** multi-hop (0.346), single-hop (0.576). Flip
`FEATURES__GRAPH_RESULTS_IN_PACKET=true` for the arm — the flag exists and its default
is documented as evidence-driven, so flipping it back is a one-line arm.

**Adjacent dead knob, free to fix here.** `step.top_k` is never passed to
`multi_hop_query`, whose 20→10 is hardcoded — so the planner's carefully chosen 15
(multi-hop) and 10 (constraint) are dead for this prong.

### 2. Sufficiency gate — **the cheap version is dead, measured**

*Signal shipped as metadata; the abstention mechanism is NOT shipped. Retrieval-only.*

> **Result, 2026-08-02.** The cheap familiarity signal does not work. Across 25 queries
> per category against the frozen full2 corpus, **no retrieval-score statistic separates
> unanswerable questions from answerable ones**:
>
> | signal | adversarial | answerable | delta |
> | :--- | ---: | ---: | ---: |
> | median top cosine | 0.624 | 0.638 | −0.014 |
> | median mean-of-top-5 | 0.596 | 0.586 | **+0.010** (wrong way) |
> | median top1−top5 margin | 0.063 | 0.065 | −0.002 |
>
> All three deltas are noise, and the mean points the wrong way. A threshold on any of
> them fires on correct answers as often as on absent ones, trading five categories for
> nothing on the sixth. The refusal nudge was written, measured, and removed; a test
> pins it out so it is not reintroduced without new evidence.
>
> A second measurement worth keeping: **post-rerank relevance is not a signal at all.**
> Sampling 60 queries through the live read path, the top score was *exactly 0.850 on
> every single one* — `GRAPH_RELEVANCE_CEILING`. Prong scores are per-source constants
> (facts 0.8, constraints 0.75, graph banded 0.55–0.85), so anything thresholding the
> reranked set is thresholding a constant. Only the vector prong's cosine varies with
> the query. Any future confidence work must read that, not the packet.
>
> **What survives.** The signal is still computed and returned (`packet.sufficiency`,
> and on `ReadMemoryResponse`) because "retrieval found nothing at all" is real
> information and distinct from ranking. It just does not steer the prompt.
>
> **Where the evidence actually pointed.** Re-reading the sources after this result: the
> +2–10% figure came from an **LLM autorater** classifying sufficiency, and RF-Mem's
> entropy gate routes to a *deeper retrieval path*, not to abstention. Neither claims a
> score threshold can decide refusal. The remaining honest version of this item is a
> cheap LLM sufficiency call on the assembled packet — a different, more expensive
> mechanism, and it should be scoped as one.

The original reasoning, kept because the motivation still holds:

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
response, and have the answer prompt consume it.

**Half of this is already computed and thrown away.** `packet_builder` builds
`open_questions` — one entry per memory with `confidence < 0.5` — and it reaches
**nothing**: not the markdown renderer, not `_format_json`, not `ReadMemoryResponse`,
which has no field for it. `warnings` is nearly as bad (markdown only, and only if ≥50
chars of budget survive). The seamless turn path is worse still: it rebuilds a filtered
packet that structurally drops both, and returns `""` for an empty result set with no
"nothing relevant" sentinel. Every existing threshold in the system silently *shrinks*
the packet rather than signalling anything. So this item is partly a plumbing job on
signals we already compute — the same write-only bug class we have now removed twice.

**Measurement confound — do not repeat the earlier mistake.** This arm only pays off if
the answer prompt consumes the signal and abstains, so it is *not* a pure retrieval
change: it moves the QA prompt, and adversarial is the category most sensitive to prompt
wording. Hold the prompt fixed across the comparison and vary only the signal, or the arm
measures two things at once. Three arms were spent untangling exactly this shape earlier.

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

*Incremental, not a migration — the columns already exist. Write-path. Fresh-ingest arm
(~2 h).* **Shipped 2026-08-03, unmeasured.**

> **Shipped.** Facts: the extractor emits `valid_from`/`valid_to`, and a fact that
> arrives already over is created `is_current=False` so the flag and the `valid_to`
> reader agree. Episodes: `event_date` is a column with an index
> (`002_event_date_column.py`), backfilled 23,463/23,463 with zero mismatches.
>
> **The coalesce is the load-bearing part.** Only 4.3% of records carry an `event_date`,
> so the planner's filter resolves against `COALESCE(event_date, timestamp)` under an
> explicit `time_basis="event"` key. A bare-column compare would have hidden 95.7% of the
> corpus and looked like a scoring regression. The key is scoped to the planner because
> temporal contiguity anchors its window on a seed's own turn timestamp — re-dating that
> window would drop precisely the neighbours whose turn recalls a distant event.
>
> No arm has been run. See the sequencing note at the foot of this document.

**Human analogue.** Event time is encoded separately from encoding time; humans date
events by landmarks and relative order, not by a single stamp.

**Evidence.** Tier 3, strongest per-category numbers in the whole set. Zep's bi-temporal
model (four timestamps per edge: created/expired for ingestion, valid/invalid for when
the fact *held*) plus contradiction-driven edge invalidation lifted temporal 45.1 → 62.4
(+38% relative) and multi-session 44.3 → 57.9. TSM reports up to +12.2pp absolute on
LongMemEval/LoCoMo from organising by *semantic* time rather than dialogue time and
consolidating point-wise memories into durative units. MemoTime: retrieval strategy
selected per temporal operator (before/after, duration, ordering), up to +24%.

**Where we stand — better than STATE.md lever C claims, and worse.** `semantic_facts`
*already has* `valid_from`, `valid_to` and `is_current`, and reads already filter on
them. What is missing is that **no path ever sets an interval from content**:
`valid_from` is stamped with the write time on every create, and `valid_to` is set only
on supersession. "I was vegetarian until June 2024" produces a fact valid from the moment
we heard it, with an open end. So the storage layer is ready and the extraction layer
is not — this is an extractor change, not a migration.

The episodic side has the mirror-image problem. Source monitoring is shipped and verified
at scale (18,490 `event_date` records), but `event_date` lives **only in the metadata
JSON** — never a column, never indexed, and read in exactly one place
(`packet_builder.py:290`, for rendering). Nothing filters, sorts or ranges on it; the
planner's time filter and `vector_search` both hit `timestamp`, which is turn time. We
extract event time correctly and then cannot query by it. `MemoryRecordCreate` also
exposes neither `valid_from` nor `valid_to`, so an API writer cannot set an interval at
all.

**Change.** Extract validity intervals from content into the existing fact columns; on
supersession close the old interval rather than only flipping `is_current`. Promote
`event_date` to a queryable column with an index, and route the planner's time filter
through it. **Reader first, in both cases** — that is the write-only bug class we have
now removed twice, and `event_date` is currently a live example of it.

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

### 6. Make consolidation actually run, then gate it on recurrence

*Incremental. Write-path. Fresh-ingest arm. Primarily a **performance** item.*
**Shipped 2026-08-03, unmeasured, default off.**

> **Shipped, and the diagnosis below was incomplete.** It is not only that
> `start_background_worker` had no caller — the registry its loop polls was never
> populated either, because nothing ever called `register_user`. So wiring the consumer
> alone would have produced a loop polling an empty queue forever, a fifth instance of
> the same class. The registry was deleted and replaced with a sweep that enumerates
> tenants over the quota in one GROUP BY and enqueues them directly, started from
> `lifespan` behind `FEATURES__CONSOLIDATION_SCHEDULER_ENABLED` (default off).
>
> Trigger counts are exposed at `GET /api/v1/admin/consolidation/status`, per this
> item's own instruction to verify by count rather than by diff.
>
> The absorbing 7-day window is gone: eligibility is "un-consolidated", pushed into SQL,
> with no age bound. The recurrence gate ships as `consolidation_recurrence_min`,
> **defaulting to 1, which gates nothing** — at 2 it stops a tenant whose episodes never
> cluster from producing gists at all, which two integration tests caught, and that trade
> is unmeasured.
>
> **Hazard for whoever runs the arm.** 664 tenants in the dev database are over the
> quota and 859 are the frozen eval corpus. Enabling the sweep against that store would
> consolidate the corpus every frozen-corpus arm depends on. Use a fresh
> `--tenant-prefix`.

**Before any of the below: consolidation never fires on its own.** `ConsolidationWorker.
start_background_worker()` has **no caller anywhere in the repo**, and
`ConsolidationScheduler.check_triggers` is called only from
`tests/unit/test_consolidation_triggers_clusterer_sampler.py`. The documented 6-hour
interval and 500-episode quota have never fired in production. The only live entry points
are two HTTP routes (`admin_routes.py:24`, `dashboard/jobs_routes.py:308`) — consolidation
is **manual-only**.

This is the third instance of the class that has now cost this project two wrong
conclusions: `encode_chunk` (0 callers, flagged on, documented as shipped), eval-mode
graph blindness, and now the consolidation scheduler. Anything measured about
consolidation quality to date was measured about a pipeline that only ran when someone
clicked. **Verify by trigger count, not by reading the diff.**

Two more limits worth knowing before tuning anything here:

- **The 7-day window is absorbing, not sliding.** `EpisodeSampler` scans
  `time_window_days=7` (90 for constraints), so an episode not sampled within 7 days of
  its `timestamp` is never eligible again. With a manual-only trigger, that means most
  episodes were never consolidation candidates at all.
- **Nothing re-runs over consolidated material.** `exclude_consolidated=True` is the
  default and the only call site uses it; consolidation reads only `memory_records` and
  `semantic_facts` is write-only from here. There is no gist-of-gists pass, no
  re-clustering, no re-derivation when new evidence lands. This is exactly the
  "single-pass, no repeated reinstatement" deficit — and the fix is a scheduler plus a
  second-order pass, not a rewrite.

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
**Shipped 2026-08-03, unmeasured, default off.**

> **Shipped** behind `FEATURES__CONSOLIDATION_DETAIL_RECOVERY_ENABLED` — its own flag
> rather than the scheduler's, so the arm can attribute the two separately. Output that
> merely restates the gist is dropped: a restatement would migrate as a second semantic
> fact competing with the first, and gist-vs-source demotion only knows how to demote
> *episodes*, so both would sit in the packet.

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
demotion must keep raw episodes retrievable indefinitely.

**This is a live risk, not a hypothetical one.** `forgetting/executor.py` transitions
records to `SILENT` (`:106`), `COMPRESSED` (`:132`) and `ARCHIVED` (`:179`, `:188`), and
`vector_search` filters `status='active'`. So a decayed episode becomes genuinely
**unretrievable** by the vector prong — not merely down-ranked. Against the 81.10 → 51.88
ablation, every such transition is spending the single most valuable thing in the store.
Audit what fraction of episodes has left `ACTIVE`, and treat any non-trivial number as a
regression to investigate before adding consolidation pressure on top.

**Do not build agentic LLM-steered graph traversal as the first move.** See item 5 —
null depth ablations, one paper contradicting its own table, a killed claim, and Zep
winning without a loop.

**Do not replace time-based decay with a contextual-interference model yet.** Tier 4 says
forgetting is driven by contextual interference from temporally adjacent events rather
than by consolidation failure — theoretically better than our decay curve, but it is
cognitive-science-only with no benchmark behind it, and it touches the retention path we
just shipped. Note the naming trap: `forgetting/interference.py` already exists, but it
detects near-duplicates and text overlap — *semantic redundancy*, a different mechanism
from temporal-context interference. Do not read the existing module as this box already
being ticked. Item 3 extracts the useful, cheap half of the finding.

**Do not chase full episodic re-reading.** The +7.8 F1 above costs **~11.25 s per query**
against Mem0's P50 ≤1.1 s. Item 3 is the bounded approximation.

**Do not trust benchmark-driven prioritisation of the write path.** Public memory
benchmarks grade almost exclusively the retrieval step; write/consolidation quality and
per-user isolation under load are barely measured. Our own write-path work being a wash
on LoCoMo-Plus is weak evidence that it does not matter.

---

## Incidental findings

Surfaced while mapping the code for this plan. Not part of any item, individually cheap,
and each one distorts measurements of the items above — so they are worth clearing first
or at least knowing about when reading an arm's numbers.

- **The constraint boost swamps every other term.** `rerank_with_breakdown` adds
  `min(1, relevance) * 2.0` to constraints, against a base score whose maximum is ~0.9.
  Any constraint therefore outranks every episode, fact and preference unconditionally.
  If constraints are in the packet, the reranker's weights are decoration.
- **Gist demotion misses the constraint prong.** `gist_keys` is built only from
  `retrieval_source == "facts"`, but the constraints prong stamps `"constraints"` — so a
  gist promoted into a constraint category never demotes the episodes it summarised, and
  both the gist and its sources occupy the packet.
- **The fact prong has no tokenisation.** `search_facts` is a whole-query
  `ILIKE %query%` over key/subject/value with no trigram index and no embeddings, so any
  multi-word query matches almost nothing. Its results are also scored at a flat 0.8
  regardless of match quality, as constraint-category facts are at a flat 0.75.
- **`_apply_diversity` is dead code** — no callers; `_apply_diversity_with_indices` is
  the live one.

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

**All four of those are now implemented and tested, and none is measured** (2026-08-03).
That arm is the outstanding work. Everything they add is default-off or default-no-op
except the `evidence_ids` union and the `event_date` column, both of which are
behaviour-preserving on existing data by construction. Flags to move on the arm:

| flag | default | what it turns on |
| :--- | :--- | :--- |
| `FEATURES__CONSOLIDATION_SCHEDULER_ENABLED` | off | the sweep that makes consolidation fire at all |
| `FEATURES__CONSOLIDATION_RECURRENCE_MIN` | 1 (no-op) | skip clusters below k; RecMem reports 87% token saving at 4-5 |
| `FEATURES__CONSOLIDATION_DETAIL_RECOVERY_ENABLED` | off | item 7's second pass |

Run it against its own `--tenant-prefix`: with the shipped 500-episode quota, 664 tenants
in the dev store qualify and 859 are the frozen eval corpus, so a sweep there would
consolidate the corpus the read-path arms depend on.

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
