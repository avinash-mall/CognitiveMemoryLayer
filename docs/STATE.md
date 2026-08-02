# Working state

Current snapshot of active work. Update this file as work progresses; add
sibling notes (`docs/<topic>.md`) for anything too big to inline here, and
link them below. Delete notes when they stop being true.

## Where things stand

- **CI is green** (lint + test + build, plus py-cml and CodeQL) as of `9929413`, the tip
  of the 2.0.0 cleanup. It was first made green at `215da0f` — the first passing
  `CI/CD Pipeline` run in the visible history, since every run back through 2026-06-20
  failed at the Docker build and the integration suite had not executed in six weeks. If
  CI goes red, it was genuinely green here. The `build` job takes ~35 min and is the slow
  leg; `lint` and `test` land in the first few minutes, and `test` is the meaningful
  signal because it runs the integration suite in containers against fresh
  Postgres/Neo4j/Redis, independent of any local `.venv`, running services, or `.env`.
- Suite sizes, so drift is visible: **799** unit (hermetic), **341** integration + e2e +
  py-cml against a live server. Grew from 716/327 across three passes: source monitoring,
  gist demotion, the retention curve; then temporal resolution, prospective indexing,
  the scan_texts_for_gate SQL guard and multi-valued facts; then the graph-as-index fix,
  temporal contiguity and the sufficiency signal. Not regressions.
- **The Docker `api` container bakes its source in — there is no volume mount.** Editing
  `src/` does not change what the running container serves, so a live suite against it
  tests the last image build, not your working tree. For verifying local changes, run
  `uvicorn src.api.app:app --port 8000` from source against the same Postgres/Neo4j/Redis
  (the container publishes on `:6000`, so the two coexist). Full live suite: ~2.5 min.
- **The cleanup pass is complete and released as 2.0.0** (−19,600 lines across 27
  commits, ending at `9929413`). Removed: all obsolete docs, five unwired modules, the
  event-log surface, the async storage pipeline, answer verification/compression, the
  BM25 index, the retrieval hot-cache path, nine unread config keys, and the SDK surface
  that depended on them. See `CHANGELOG.md` [2.0.0] for the breaking inventory.
- **Version lives in three places and they must agree**: the tracked `VERSION` file,
  `.env.minimal`, and `packages/py-cml/src/cml/_version.py` (which `hatch_build.py` does
  *not* feed — it is a separate hardcoded string). `hatch_build.get_version()` resolves
  `VERSION` env → `.env` → `VERSION` file, so an untracked local `.env` shadows the file;
  CI has no `.env` and therefore reads the file. They had drifted to 1.4.2 while
  `CHANGELOG.md` already documented a released 1.5.0. All three now say 2.0.0.
- The modelpack removal (`51afd15`) and its follow-up cleanup are complete: dead code,
  stale docs, dead env keys, vendored assets, cache volumes.

## Active work

### Human-memory research pass (2026-08-02)

Plan and evidence: [memory-redesign-plan.md](memory-redesign-plan.md). Three retrieval
items implemented; one was killed by its own measurement.

- **Item 1 — graph prong is an index, not content** (`412a3e3`). Resolves ranked entities
  to the episodic records their edges cite (`evidence_ids`) instead of rendering
  `Entity: x\n - REL: y`. Also dropped the fallback traversal from 3 hops to 2 and capped
  resolved records at `step.top_k`. **The prong used to time out and contribute nothing**
  (1.5–3.5 s against a 2 s budget); it now completes in ~174 ms.
  **Measured and it lost.** Full 2,387-sample frozen-corpus arm
  (`evaluation/outputs/item1/`, provenance in `ARM_PROVENANCE.txt`, server pid verified
  unchanged across the run): **0.4860 → 0.4292**, every factual category down (single-hop
  −0.097, temporal −0.098, multi-hop −0.090) and adversarial *up* +0.074. Median context
  1583 → 2741 chars.

  Cause was **ranking, not resolution**. Graph hits carried the traversal score, which
  normalises into a constant 0.55–0.85 band; the median vector cosine is ~0.62, so about
  half of every graph batch outranked the median genuine match *regardless of the
  question*, and 10–15 were injected into a packet with 5–8 episode slots. The prong went
  from emitting neighbourhood summaries to emitting well-ranked irrelevant episodes.

  Graph-resolved records are now scored by **cosine against the query** — graph decides
  candidacy, similarity decides rank. Unmeasured, so
  `FEATURES__GRAPH_RESULTS_IN_PACKET` is back to default **false** and `main` sits at the
  known-best 0.4860 config. Turning it on is the next arm.
- **Item 3 — temporal contiguity** (`48e4d71`, fixed in `f4b40fa`). Expands the top 3
  vector hits into the ±2 turns around them. Ordered by `metadata.turn_idx`, **not**
  timestamp: every turn of a LoCoMo session shares one identical timestamp (28 turns,
  span 0.000000 s), and `written_at` is scrambled by concurrent ingestion workers.
  Not yet measured.
- **Item 2 — sufficiency gate. Signal shipped, abstention deliberately not.** Measured at
  25 queries/category: no retrieval-score statistic separates unanswerable from
  answerable (top cosine 0.624 vs 0.638; mean-of-top-5 0.596 vs 0.586, *the wrong way*;
  margin 0.063 vs 0.065). A test pins the refusal nudge out. `packet.sufficiency`,
  `open_questions` and `warnings` now reach `ReadMemoryResponse` — all three were
  computed and reached no caller before.

**GDS is now part of the build** (`docker/neo4j.Dockerfile`, jar pinned to 2.13.4 with a
sha256 check). Two things this uncovered:

- **The GDS code path could never have worked.** It called
  `gds.pageRank.stream({nodeQuery: ..., relationshipQuery: ...})` — GDS 1.x anonymous
  projection, removed in GDS 2.0. Verified against a real GDS 2.13.4 server: *"Type
  mismatch: expected String but was Map"*. The `except Neo4jClientError` branch then
  silently substituted the path-count fallback, so **installing the plugin alone would
  have changed nothing**. Rewritten to the 2.x contract: project a uniquely-named graph,
  stream, drop in a `finally` (a leaked projection pins its nodes in heap for the life of
  the database).
- **`NEO4J_PLUGINS` was not an option** — it downloads the plugin on every container
  start, breaking rule 1. The jar is fetched at build time, like pip wheels.

CI builds the same image rather than pinning the stock one, because without the plugin
the fallback makes the tests pass either way — which is how the 1.x call survived.

⚠️ **Not yet deployed.** The live `docker-neo4j-1` still runs the stock image; switching
it needs `./docker/up.sh build neo4j && ./docker/up.sh up -d neo4j`, which was deferred
because recreating the container mid-arm would corrupt the running measurement. **Verify
`MATCH ()-[r]->() RETURN count(r)` is still ~583k afterwards**: Neo4j's data lives on an
**anonymous** volume (no named volume in `docker-compose.yml`), so a recreate preserves
it but any `down -v` / `up -V` destroys the frozen corpus and costs ~9.5 h to rebuild.
Giving it a named volume is worth doing, but it is a data migration, not a plugin install,
so it was deliberately not bundled here.

**Two facts worth more than the items themselves:**

- **Post-rerank relevance is a constant, not a signal.** 60 sampled queries through the
  live read path returned a top score of *exactly 0.850* every time —
  `GRAPH_RELEVANCE_CEILING`. Prong scores are per-source constants (facts 0.8,
  constraints 0.75, graph banded 0.55–0.85). Anything thresholding the reranked set is
  thresholding a constant; only the vector prong's cosine varies with the query.
- **Consolidation never fires on its own.** `start_background_worker` has no caller
  anywhere and `check_triggers` is called only from tests, so the documented 6-hour
  interval and 500-episode quota have never run — the two HTTP routes are the only live
  entry points. Third instance of the class that produced two wrong conclusions already
  (`encode_chunk`, eval-mode graph blindness).

### Earlier

- **LoCoMo-Plus subset A/B is running.** Everything below is implemented and committed.
  Arms, each against a server run from source on `:8000` using
  `evaluation/locomo_plus/data/unified_input_subset_v2.json` (43 conversations,
  16,484 turns, 677 samples, ~1.5-2 h per arm) and **its own `--tenant-prefix`**:
  - **0** — free, already computed: `make_locomo_subset.py --baseline` restricts the
    committed full-run artifact to the subset. Overall 0.4993 there vs 0.4631 full.
  - **A+B** — run three times; see the results table below. Arm 3 carries all write-path
    fixes and is a wash overall (0.480 vs 0.499).
  - **4** — arm 3's frozen corpus re-scored with `FEATURES__GRAPH_RESULTS_IN_PACKET=false`.
    **0.513, the only arm above baseline.** Flag flipped to false on that evidence.
  - **C** — `FEATURES__PROSPECTIVE_INDEXING_ENABLED=true`. **Not run.** The flag stays
    off. Watch Cognitive and common-sense for gain and **adversarial for regression** —
    that decides the default. Note Cognitive swings 0.200-0.325 across identical-code
    runs at n=40, so arm C needs either a bigger Cognitive quota or repeated runs to say
    anything about the category it targets.

  Compare per-category against arm 0, not against the full run's 0.4631.

  **Arms A and B are bundled**, deliberately: both are committed and both are keepers, so
  separating them would cost another ~2 h to attribute between two changes neither of
  which is going to be reverted. The decision-bearing split — prospective on vs off — is
  preserved. Do not read per-category movement inside A+B as attributable to one of them.

  **Always pass `--tenant-prefix`.** Tenants are `{prefix}-{canonical_index}` and the
  index is relative to whichever `--unified-file` was passed, so a subset run reuses the
  full run's tenant IDs for entirely different conversations. This bit once: a subset run
  collided on `lp-199`, which held 369 records from the full run's conversation 199, and
  that single tenant carries 242 of the subset's 677 samples (36%). The run was killed
  and the flag added (`d26b683`) rather than deleting the older data.

## Full LoCoMo-Plus re-run complete (2026-08-02) — 0.4631 -> 0.4860

All fixes at shipped defaults: prospective off, graph excluded from the packet,
`episode_relevance_threshold` 0.4. 411 conversations / 242,658 turns, identical scope to
the 2026-07-31 run, **2,387/2,387 valid, 0 errors**. Artifacts:
`evaluation/results/locomo_plus_2026-08-02_{summary,judged}.json`.

| category | n | 2026-07-31 | **2026-08-02** | delta | subset predicted |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Cognitive | 401 | 0.2120 | **0.2544** | +0.042 | +0.125 |
| adversarial | 446 | 0.7825 | **0.7534** | −0.029 | −0.053 |
| common-sense | 96 | 0.2396 | **0.2760** | +0.036 | +0.025 |
| multi-hop | 282 | 0.3387 | **0.3457** | +0.007 | +0.040 |
| single-hop | 841 | 0.5380 | **0.5755** | +0.038 | +0.068 |
| temporal | 321 | 0.3131 | **0.3551** | +0.042 | +0.062 |
| **overall** | 2387 | **0.4631** | **0.4860** | **+0.023** | +0.037 |

Five of six categories improved; adversarial is the standing trade (more usable context
means the model refuses less, and refusing is what adversarial rewards).

**The subset over-predicted, consistently.** Direction was right on every category and
the sign never flipped, but magnitude was roughly 1.6x too large overall and much worse
on the small categories — Cognitive predicted +0.125 against an actual +0.042 at n=40 vs
n=401. Treat subset deltas as a directional screen, not an estimate, and discount
anything from Cognitive or common-sense hardest. Multi-hop is the weakest agreement
(+0.040 predicted, +0.007 actual): the graph exclusion helped far less at full scale.

Conditions: 1 uvicorn worker, 20 ingestion workers, ~9.5 h ingestion at 6.7 rec/s, ~4 h
QA, ~0.5 h judge. Per lever G still not comparable to published gemini-judged baselines.

Ingestion checkpoints per conversation, so a crash resumes — but any re-run **must** keep
`--out-dir evaluation/outputs/full2`, or QA silently switches to per-sample tenants.

## Subset A/B results, and the three bugs the measurement found (2026-07-31)

Two full subset runs, 677 samples each, against arm 0 = the committed full run restricted
to the same samples (overall **0.4993**).

| category | n | arm 0 | v1 | v2 | arm 3 | arm 4 | **arm 5** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cognitive | 40 | 0.200 | 0.325 | 0.275 | 0.200 | 0.250 | **0.325** |
| adversarial | 151 | 0.808 | 0.735 | 0.768 | 0.782 | 0.748 | **0.755** |
| common-sense | 40 | 0.350 | 0.375 | 0.313 | 0.400 | 0.400 | **0.375** |
| multi-hop | 100 | 0.330 | 0.225 | 0.270 | 0.260 | 0.310 | **0.370** |
| single-hop | 250 | 0.520 | 0.496 | 0.510 | 0.528 | 0.576 | **0.588** |
| temporal | 96 | 0.323 | 0.125 | 0.167 | 0.260 | 0.344 | **0.385** |
| **overall** | 677 | **0.499** | **0.439** | **0.458** | **0.480** | **0.513** | **0.536** |

Arms 4 and 5 are the trustworthy ones: both re-scored arm 3's *frozen* corpus with a
single setting changed, so neither carries re-ingestion variance. Arm 4 excluded graph
results (+0.033); arm 5 additionally lowered `episode_relevance_threshold` 0.5 -> 0.4
(+0.024). **Final: 0.536 vs the 0.499 baseline, +0.037, with every category above
baseline except adversarial.** Both defaults are now flipped in config.

Adversarial is the one consistent cost, 0.808 -> 0.755. It is a genuine trade rather
than a bug: that category rewards *refusing* when the answer is absent, so a packet
carrying more usable context makes the model refuse less and occasionally answer
something it should have declined. Everything that improves recall will show up here as
a small loss.

Do not push the threshold below 0.4 — 0.3 admits 100% of retrieved episodes, so it stops
being a filter at all, and the remaining headroom between 0.4 (97.7%) and 0.3 is 2.3%.

**Arm 4 is the trustworthy one, and it is the only result above baseline.** It re-scored
arm 3's *frozen* corpus with a single flag changed — no re-ingestion, so none of the
run-to-run variance that makes the other columns hard to read. Excluding graph results
from the packet: **overall 0.480 -> 0.513**, temporal +0.084, multi-hop +0.050,
single-hop +0.048. `FEATURES__GRAPH_RESULTS_IN_PACKET` now defaults to **false** on that
evidence.

The one cost is adversarial, 0.782 -> 0.748. That is coherent rather than surprising:
adversarial rewards refusing when the answer is not present, so a packet carrying more
usable context makes the model refuse less, and it sometimes answers an adversarial
question it should have declined. Within the 0.073 spread that category shows across
runs, and a good trade for +0.033 overall.

Turn the flag back on when lever E (iterative reason/retrieve) makes the graph prong
return answers instead of neighbourhood summaries.

**Verdict: the write-path work is a wash overall, and multi-hop is a real regression.**
Arm 3 lands at 0.480 against arm 0's 0.499 — a gap of ~13 of 677 samples, smaller than
the spread single-hop alone shows across identical-code runs, so overall is not
distinguishable from baseline. Per category, four of six sit inside the run-to-run
spread. Two do not:

- **common-sense +0.050** — the one clear gain.
- **multi-hop −0.070** — and this is the important one, because it is the category
  commit `6d8138e` was meant to fix. *Every* arm with a populated graph (0.225, 0.270,
  0.260) scores below arm 0's empty-graph 0.330. Populating the knowledge graph makes
  multi-hop **worse**, reproducibly.

  The mechanism is already documented above: `multi_hop_query` has no hop loop. It runs
  one PPR pass and returns *entity profiles* — "Entity: user / LOCATION: Seattle / …" —
  which summarise a neighbourhood rather than answering anything. Those blobs then take
  packet slots from episodes that would. Normalising their relevance (`18c947b`) stopped
  them dominating but did not make them useful. Lever E (iterative reason/retrieve) is
  what would; until it exists, the graph prong is not earning its place in the packet,
  and excluding graph results from the packet is worth measuring as a cheap alternative.

  Note this does **not** argue for reverting `6d8138e`. Eval mode hiding the graph meant
  the benchmark was not measuring the shipped system. It is now, and what it measures is
  that this graph prong does not help.

Three bugs were found by the measurement, all the same shape — a branch that could not be
wrong while the feature feeding it never ran:

1. **The LLM's `event_date` overrode the regex's** (`a909f0c`). 178 of 1,328 records
   (13.4%) carried a date inconsistent with their own timestamp, and every one carried
   the *same* hallucinated day. Re-running `extract_event_date` on the stored text and
   timestamp reproduces the correct date every time, so the resolver was fine and the
   precedence was not. v1 → v2 is this fix alone: +0.019 overall, +0.042 temporal.
2. **`event_date` *replaced* the turn date in the packet** (`342a32e`). `packet_builder`
   rendered `event_date if event_date else timestamp`, so 1,328 episodes lost the "when
   was this said" anchor and the bracket silently meant different things on different
   lines. Now `[said X, refers to Y]`, collapsing to one date when they agree.
3. **Graph relevance is unbounded and outranked everything** (`18c947b`). One real query
   returned graph blobs at 315.67 / 265.33 / **744.50** against episodes at 0.27-0.40.
   Ranking happens on the raw value before the reranker clamps, so the blobs took every
   top slot and pushed every conversation turn below `episode_relevance_threshold`,
   emptying Recent Events. Now rank-normalised into [0.55, 0.85].

All three fixes were verified live on arm 3's own data before that run was scored:
0 anchor mismatches (was 178/1328), graph blobs at 0.69 (were 315-744), and a Recent
Events section present in the packet where it had been absent.

**`--skip-ingestion` needs `--out-dir` pointed at the run that ingested the corpus.**
`phase_b_qa` reads `locomo_ingestion_checkpoint.json` *from out_dir* to decide whether
tenants are per-conversation or per-sample. A fresh out-dir has no checkpoint, so it
silently falls back to `{i: i}` and every query targets `{prefix}-{sample_index}` instead
of `{prefix}-{canonical_index}` — tenants that mostly do not exist. Median context length
26 characters instead of 1737, and a score of adversarial 1.0 with single-hop, temporal
and common-sense exactly 0.0, because refusing is correct for adversarial and wrong for
everything else.

That pattern was produced twice and misdiagnosed twice — first as a broken
`--skip-ingestion`, then as unbounded graph relevance. Both wrong; it was the out-dir.
`b0027e4` now raises instead of scoring. Either point `--out-dir` at the original run or
copy its checkpoint across; a read-path A/B on a frozen corpus is then ~55 min instead of
~2 h, and is by far the most trustworthy comparison available here.

## Open, with evidence, not yet acted on

- **`episode_relevance_threshold` (0.5) is binding but not fatal — an earlier note here
  said episodes "routinely score below" it and that was wrong.** That claim came from one
  hand-picked query. Sampled properly across 40 real queries from the subset (n=383
  episodes): p10 0.462, median **0.555**, p90 0.680, max 0.838. **74.9% already clear
  0.5**; 97.7% clear 0.4. So Recent Events is usually populated, not usually empty.
  It is still binding at the margin: ~7.2 episodes survive per query against a
  `max_episodes_default` of 8, so the threshold rather than the cap is what limits the
  section. Lowering to 0.4 lets the cap bind instead — measured as arm 5.

## What was wrong with the write path (fixed 2026-07-31)

`HippocampalStore.encode_chunk` had **zero production callers** — only `encode_batch` is
live. Everything that existed solely inside it therefore never ran, despite being
feature-flagged on. The database was the proof, across 245,386 records: 0 prospective
records, 117 with `event_date`, 0 with `causal_chain`, but 218,418 *with* extracted
entities (so unified extraction, which lives in `encode_batch`, always ran).

Fixed across `6d8138e`, `bb3a4a5`, `f662975`, `e3635ad`. See the eval-mode correction
under Known issues for what this means for the 0.4631 score.

Deliberately still not carried over: `metadata["temporal_references"]` and
`metadata["causal_chain"]`. Nothing reads either, and adding them would recreate the
write-only bug class the previous pass removed. ~8 lines each when a renderer wants them.

## The write-path/read-path disconnect (fixed 2026-07-31)

An audit of the memory subsystems against the README found the gap was not missing
biology — it was that the biology already implemented was *disconnected*. Fields were
written on the write path and read by nothing that ranks or renders. Three commits
(`9fa06dd`, `3d04dfb`, `5f04478`) closed the load-bearing ones:

- `provenance.source` was surfaced only for constraints, so LLM-generated prospective
  implications (on by default) rendered identically to user statements. Now rendered by
  all three renderers from `core.schemas.source_label`. The fact prong is exempt on
  purpose — `semantic_facts` has no provenance column, so `_fact_to_record` stamps every
  row `AGENT_INFERRED` as a placeholder; labelling from it would have downgraded
  "Earlier you said" to "Inferred" on constraints.
- `decay_rate` was written per record and read by no scorer. The forgetting scorer and
  the reranker used two *different* curves (exponential 30-day half-life vs. hyperbolic
  `1/(1+age*0.1)`). Both now call `src/utils/retention.py`. Near-identical at the default
  rate — which is why the scorer's threshold tests did not move — but a memory marked
  ephemeral now retains 3% after a week instead of 85%.
- `access_count` reaches the forgetting scorer but still not the reranker, and that is
  now a deliberate, tested decision rather than an oversight. A frequency term was built
  and removed: only the vector prong increments the counter and `semantic_facts` has no
  such column, so any weight biases against fact- and graph-sourced hits, and the counter
  grows with every query — ranking would stop being a function of the query alone. It
  also could not be shown to help, because the quality harness is too noisy to resolve
  it (see the reproducibility note under Known issues).
- `consolidated_into_fact_key` was written by the migrator and read by nothing in
  retrieval, so gist and verbatim competed in one result set. The reranker now demotes a
  source episode when its gist is present. Conditional on the key, so an unrelated fact
  in the results penalises nothing.

Both halves were verified against real data, not just unit tests:

- **The forgetting curve change is a no-op on this database.** Scored 3,804 real records
  under the old `0.5**(age/30)` and the new `exp(-decay_rate*age)`: every row with
  `decay_rate > 0.05` (321), every row at exactly `0.05` (483 — the largest *relative*
  shift of any rate present: at 90 days 0.125 → 0.011), plus a random 3,000 of the rest.
  **Zero** suggested-action changes in all three cohorts; COMPRESS steady at 42. The
  distribution is 187,731 rows at 0.01, 477-483 at 0.05, 321 at 0.1, and **none at 0.5**
  — the profile that would move (`decay_rate=0.5`, ~7d) simply does not occur here.
  Synthetically it shifts `decay` → `silence`, never toward COMPRESS/DELETE, so the
  destructive band is not in play. Worth re-checking if the extractor ever starts
  emitting 0.5 in volume, since `forgetting-daily` is the one job that *is* scheduled.
- **The gist demotion fires on real consolidated data.** `consolidated_into_fact_key`
  values all resolve to live `semantic_facts.key` rows (checked across 10 keys), and
  reranking a real consolidated episode ("User said they like pizza", relevance 0.92)
  against its real gist (`user:preference:food_preference`, relevance 0.70) puts the gist
  first with `demoted_superseded_by_gist` on the episode. The join is sound.

Still disconnected, deliberately (each would recreate the same bug class or needs a
schema change nobody reads yet): `MemoryRecordModel.labile` is never assigned;
`FactSchema.multi_valued`/`validators` are ignored by `_update_fact`, so the one
multi-valued schema (`user:preference:cuisine`) supersedes instead of appending;
`ShortTermMemory.get_immediate_context`/`get_encodable_chunks` have no callers, making
the sensory buffer's decay and working memory's capacity policy correct implementations
of behaviour nothing observes; `WriteDecision.STORE_SYNC` is a `StrEnum` alias colliding
with `STORE`; `SensoryBuffer.start_cleanup_loop` is never called.

## Architecture facts (load-bearing)

- **Auth is the `X-API-Key` header** (`src/api/auth.py:16`), not `Authorization: Bearer`.
  The README's curl examples said Bearer and returned 401 for everyone who copied them;
  fixed 2026-07-31. Admin keys additionally accept `X-Tenant-Id`.

- Single LLM path: `FEATURES__USE_LLM_ENABLED` is the only LLM switch.
  On: the unified extractor (`src/extraction/unified_write_extractor.py`)
  drives extraction/classification/enrichment via `LLM_INTERNAL__*`.
  Off: heuristic-only mode (regex PII, Jaccard novelty, regex facts/
  constraints, no entity graph) — this is what the hermetic unit suite runs.
- The old custom-model path (modelpack/sklearn/DeBERTa/spaCy, HF summarizer,
  `packages/models/` training pipeline) was removed in commit 51afd15.
  Hot pairwise scoring uses embedding-cosine/Jaccard, never per-pair model calls.
- Local dev LLMs: qwen35-4b (vLLM, `http://localhost:8012/v1`, thinking
  disabled via `LLM_INTERNAL__EXTRA_BODY`) internal; Qwen3.6-27B-FP8
  (`http://localhost:8002/v1`) eval. Embeddings: local nomic-embed-text-v2-moe.
- Similarity primitives live in one place: `src/utils/similarity.py`
  (`word_set`, `jaccard`, `cosine_similarity`), used by the reranker, interference
  detection, schema alignment, and clustering. `consolidation/worker.py:_token_set` is
  the one deliberate exception — regex-tokenised, feeding an intersection check rather
  than a ratio; its `# ponytail:` comment says why. The SDK keeps its own cosine copy in
  `packages/py-cml/src/cml/storage/sqlite_store.py` — `py-cml` is a separate distribution
  and imports `src.*` only lazily inside `cml/embedded.py`, never at module scope, so it
  must not depend on `src/utils/similarity.py`.

## Invariants — do not "simplify" these

Each of these was a real bug. They look like tidy-up targets and are not.

- **`vector_search(min_similarity=-1.0)`** must stay at −1.0. Cosine similarity is valid
  on [−1, 1], so a `0.0` default is a *filter*, not "unset": it silently discards every
  negatively-correlated row. Invisible with nomic-embed (effectively non-negative vectors);
  with the hashed mock embeddings CI uses, real matches vanish. Guarded by
  `tests/unit/test_vector_search_similarity_floor.py`.
- **Retrieval sources do not share a relevance scale.** Vector/fact prongs emit cosine-like
  0..1; the graph prong passes through a raw, unbounded Neo4j co-occurrence score (245.5
  observed). `MemoryReranker._score_components` clamps relevance to [0,1] and records the
  clamp in `notes` — keep that guard.
- **Reranker breakdown keys** must match `RetrievalExplainRerankItem` (`id`,
  `source_type`, …). The dashboard renders them directly and the response model validates
  them, so renaming one returns HTTP 500 from the explain endpoint.
- **`encode_batch` returns a 4-tuple on every path**, including the early return taken when
  the write gate skips every chunk. Callers unpack a fixed arity, so a stale early return
  only breaks writes where nothing survives — which the main-path tests never reach.
  Guarded by an arity-stability test that compares the two paths rather than hard-coding 4.
- **`get_internal_llm_client()` never returns `None`** — it raises (OpenAIError without
  credentials, ValueError for a provider needing a key). Guarding on `is None` leaves a
  mock fallback unreachable in exactly the credential-less environment it exists for.
- **`rrf_merge` is live scoring arithmetic on the HyDE path** and had zero tests until
  `tests/unit/test_retrieval_rrf.py`. It survived the BM25 removal for that reason; the
  fusion constant `k` and the `id(doc)` fallback for id-less docs both change ranking, so
  keep the test if the file moves again.
- **`hippocampal/store.py` reads settings via a *function-local* import, on purpose.**
  `from ...core.config import get_settings` at module scope binds the function object
  at import time, so `monkeypatch.setattr("src.core.config.get_settings", ...)` stops
  reaching those reads — the store silently falls back to the real config and the LLM
  write path goes dark while the tests still "pass" their earlier assertions. Hoisting
  those imports broke `tests/integration/test_unified_write_path.py` exactly that way.
  The one module-level import (`_settings_for_pool_size`) is fine: it is read once at
  import to size `_GATE_EXECUTOR` and never used for per-call reads. Note also that a
  local import anywhere in a function makes the name local to the *whole* function, so
  each settings-reading function needs its import at the top, not next to first use.
- **Judge/JSON calls to a reasoning model** must disable thinking or budget 2000+ tokens.
  `LLM_EVAL` (Qwen3.6-27B) otherwise spends the whole budget on `reasoning_content` and
  returns empty `content` with `finish_reason=length`, which reads downstream as a score of
  0. `src/utils/llm.py` logs `llm_empty_content` whenever a completion comes back empty.

## Measured baselines

Throughput at `485ad77`, quality at `8304f8f`, both on this host (4× A100 80GB shared with
resident vLLM servers).

- Write path: **0.33 turns/s** with the LLM on (mean 3.03 s/turn, qwen35-4b);
  **17.31 turns/s** heuristic-only. Method + hardware in `evaluation/EVALUATION_REPORT.md` §5.
- Retrieval quality: **9.8/10** judge, **100%** recall, 6/6 constraint consistency,
  p50 1055 ms (`scripts/test_memory_quality.py`; artifact committed under
  `evaluation/results/`).
- The LoCoMo-Plus scores in `evaluation/` are pre-51afd15 history and are NOT reproducible
  — their source artifacts were never committed. Don't cite them as current.
- **The modelpack gate numbers in `CHANGELOG.md` [1.4.2] have no surviving evidence.**
  `packages/models/trained_models/` (25 GB: 16 safetensors, 17 joblib, and 60 metrics /
  epoch-stats JSONs with the per-model accuracy, macro-F1 and confusion matrices) was
  deleted on 2026-07-30 by explicit decision. It was never in git and the training
  pipeline that produced it went in 51afd15, so none of it is regenerable. Treat those
  changelog figures as historical claims that cannot be re-derived.

## Known issues / open decisions

- **The running `docker-api-1` container (published on `:6000`) is slow enough to fail
  the live suite, and that is unrelated to any code in this tree.** Measured 2026-07-31
  with nothing else running: three sequential writes took **8.4 s, 16.1 s, 18.8 s** —
  degrading, against the 3.03 s/turn baseline below. A full live suite against it took
  **54:45 and failed 15 of 337**, every failure a 120 s client timeout on a write path
  (`test_write_read`, `test_turn`, `test_batch`, `test_api_ingestion`). The identical
  suite against a server started from source on `:8000`, same Postgres/Neo4j/Redis,
  passes **337/337 in 2:24**. So it is the container, not the code — the image predates
  these commits and does not contain them.
  Not diagnosed further, only measured. The one datum worth having: GPU3, which the
  container pins via `CUDA_VISIBLE_DEVICES=3`, sat at **89% utilisation and 79.4/81.9 GB**
  while idle from CML's perspective, and the container reports `device: "auto"`,
  `batch_size: 0`. Rebuilding or restarting it is a deployment decision, not a code fix.
- **`scripts/test_memory_quality.py` is not reproducible enough for small A/Bs.** Three
  identical runs of identical code against a frozen tenant (`--skip-ingestion --tenant`)
  gave MISS/PASS/PASS on the same `semantic_disconnect` probe — 97%/100%/100% recall,
  9.6-9.9 judge. HyDE (`FEATURES__HYDE_RETRIEVAL_ENABLED`, on by default) generates a
  hypothetical document per query through the LLM, so the query embedding itself differs
  run to run, and probes sitting near a ranking boundary flip. A single before/after pair
  will therefore "prove" whatever it happened to draw — that mistake was made during the
  reranker work and caught only by re-running. Anything that moves ranking by a few
  percent needs repeated runs per side, or `FEATURES__HYDE_RETRIEVAL_ENABLED=false` to
  make retrieval deterministic first.
- **LoCoMo-Plus re-run complete (2026-07-31)** — first reproducible run since the
  modelpack removal. **Overall 0.4631** (1105.5/2387, all valid, 0 errors); by category:
  adversarial 0.78, single-hop 0.54, multi-hop 0.34, temporal 0.31, common-sense 0.24,
  Cognitive 0.21. Artifacts: `evaluation/results/locomo_plus_2026-07-31_{summary,judged}.json` (summary + per-sample judge records). Conditions:
  server at 983a9f9 (4 uvicorn workers, CPU embedder), QA+judge on local Qwen3.6-27B-FP8.
  Per lever G these numbers are NOT comparable to published gemini-judged baselines — only
  relative movement against this artifact is meaningful. Full pipeline cost on this host:
  ~11.5h ingestion (218k turns, shared GPUs) + ~2h QA + ~0.5h judge.

  **What it actually measured, corrected 2026-07-31.** This entry used to say
  "X-Eval-Mode skips unified extraction — no LLM enrichment on stored memories". That was
  false, and the same claim was in README.md and evaluation/README.md. `encode_batch`
  re-runs unified extraction per chunk whenever `unified_results is None`
  (`store.py:444-453`), so eval mode substitutes N per-chunk `extract()` calls for one
  `extract_batch()` and can be *more* LLM work, not less. 218,418 of 245,386 records carry
  extracted entities, which is the proof.

  What eval mode did skip was `_sync_to_graph` — the only writer of Neo4j entities — plus
  write-time facts and constraints. So **multi-hop 0.34 was scored against an empty graph
  and single-hop/common-sense against a dead fact prong**; Neo4j held ~2,291 nodes against
  218,418 entity-carrying records. Fixed in `6d8138e`, verified by node count. Separately,
  temporal resolution never ran on any write path at all (`encode_chunk` had no caller),
  so **temporal 0.31 was measured with `event_date` absent** from all but 117 records —
  fixed in `bb3a4a5`. Those three categories are the ones to watch on a re-run.
- **Write throughput is LLM-token-bound, measured not guessed.** One LLM call per write
  (~884 prompt + ~481 output tokens) is 95.8% of write latency; under sustained conc=40
  load vLLM holds 40 running / 0 waiting while GPU2 (qwen35-4b) pins at 93-99% and
  postgres sits flat at 41 connections. Remaining levers: shrink extraction output
  tokens, or serve the model with more capacity. More API workers past 4 will not help.
- **The eval harness's QA and judge phases are serial `for` loops** (~3.3 s/sample and
  ~1.4 s/sample) against an LLM with demonstrated 40-way headroom — ~3h that could be
  ~10min with a worker pool like Phase A already has. Not the long pole while ingestion
  dominates.
- **`EMBEDDING_INTERNAL__DEVICE` cannot select a GPU** — it only knows auto|cpu|cuda, and
  `auto` puts every uvicorn worker's ~2.2GB model copy on GPU0. On this host (GPU0 full
  of a resident vLLM) multi-worker startup OOMed until the container pinned
  CUDA_VISIBLE_DEVICES=3. If multi-worker becomes the norm, either share the embedder or
  teach the knob cuda:N.
- **"Multi-hop" retrieval has no depth control, and does not run PageRank.**
  `NeocorticalStore.multi_hop_query` calls `personalized_pagerank`, which reaches GDS
  only when that plugin is installed — **it is not installed here** (`CALL gds.list()`
  errors), so every query takes the fallback: a `(seed)-[*1..2]-(related)` path count.
  That is where the unbounded scores came from. Depth was 3 until `412a3e3`; measured on
  a real tenant, depth 3 reached 504 entities against depth 2's 502 while counting ~63x
  as many paths and taking 2.5x as long — which pushed the prong past its 2s step budget,
  so it timed out and contributed nothing at all. At depth 2 it completes in ~174ms.
  There is still no hop loop. Installing GDS would make the docstring true and is the
  cheapest way to test whether better cue ranking helps.
- **Graph relevance is rank-normalized** into `[GRAPH_RELEVANCE_FLOOR, CEILING]` =
  [0.55, 0.85] (`retriever.py`), which both stops unbounded Neo4j scores (315/265/744
  observed) from taking every top slot and keeps graph hits ordered among themselves.
  This line previously claimed hits were clamped to exactly 1.0 — stale since `18c947b`.
- **`event_log` is now an orphan table.** Nothing ever wrote a row, so its whole read
  surface (routes, dashboard panels, SDK `get_events`, `EventLogModel`) was removed.
  The table and `migrations/versions/001_initial_schema.py` were deliberately left alone
  — a destructive migration doesn't belong in a cleanup. Drop it in a migration whenever
  someone is willing to own the data loss, or resurrect the writer instead.
- **Unshipped retrieval improvements**, salvaged from the deleted `Improvement_Report.md`.
  This list used to claim levers A, B and D had *shipped*. Two of those three had not:
  - **A — prospective indexing.** Written, then wired only into `encode_chunk`, which had
    no caller. Zero prospective records existed across 245,386. Now genuinely live
    (`f662975`), batched to one `embed_batch` per write, and **default off** pending the
    LoCoMo subset A/B. `extraction/prospective_indexer.py` was deleted — its per-record
    LLM fallback was the wrong shape for a batch path.
  - **B — BM25+RRF hybrid.** Removed as unwired; no plan step ever produced a sparse
    retrieval step. `rrf_merge` survives in `src/retrieval/rrf.py` for the HyDE merge.
  - **D — temporal resolution.** Same story as A: only 117 of 245,386 records carried
    `event_date`. Live on `encode_batch` since `bb3a4a5`.

  Genuinely unshipped:
  - **C — bi-temporal graph edges** (Graphiti-style `valid_from`/`valid_to` on relations),
    so the graph can answer "what did I believe then" rather than only "now". Related and
    larger: there is no sequence structure at all, so "what happened before X" is
    unanswerable — it needs X resolved to a timestamp first, and nothing does that.
    `planner.py` handles three English literals ("today"/"yesterday"/"week").
  - **E — multi-hop iterative retrieval** (IRCoT-style reason/retrieve loop). This used to
    be called the highest-value item. The 2026-08-02 research pass demotes it: the
    evidence for hop loops is split (one paper's depth ablation is null and contradicts
    its own prose; "PPR matches IRCoT" was refuted 0-3; Zep won its temporal gains with
    no loop at all), and a cheaper, better-evidenced fix outranks it — the graph prong
    already runs PPR but returns entity profiles instead of the episodic text they index.
    See [memory-redesign-plan.md](memory-redesign-plan.md) items 1 and 5.
  - **H — `semantic_facts` usage tracking.** The table has no `access_count`,
    `last_accessed_at` or `importance`, so consolidation migrates knowledge *out of* both
    the strengthening and the decay loops. This is why the retention and frequency terms
    only half-work. Do not add the columns until a reader exists — that is exactly the
    write-only bug class fixed above.
  - **F — category-aware answer prompt** for the QA path.
  - **G — judge comparability caveat:** LoCoMo-Plus scores in the paper use
    gemini-2.5-flash with a constraint-consistency protocol. Our harness uses a local
    Qwen judge, so absolute scores are NOT comparable to published baselines — only
    relative movement is meaningful.
- Heuristic query classification is keyword-sensitive: `\bcareer\b` sits in the goal
  pattern, so "profession job career" classifies as `constraint_check`. Correct-ish but
  worth knowing when a retrieval result looks oddly constraint-shaped.

## Dashboard notes

- All third-party assets are vendored in `src/dashboard/static/vendor/` (chart.js,
  vis-network, Inter + JetBrains Mono woff2) with hashes and licenses recorded in that
  directory's README. There is **no build step** — `src/api/app.py` serves the static
  tree verbatim, so any CDN URL added to it ships straight to users. Keep
  `grep -rn "https://" src/dashboard/static` (excluding `vendor/`) empty.
- **No test anywhere touches `src/dashboard/static/js/**`.** Neither suite imports it and
  CI cannot see it, so a removed field, a dead import, or a stale `data-page` target ships
  silently. Deleting a Python response field without deleting its JS reader renders
  `undefined`/`NaN` rather than erroring; a missing module import kills the *entire*
  dashboard, not one page. When changing either side, verify by loading the dashboard in a
  real browser and walking every nav target with the console open. Playwright's chromium
  is already cached under `~/.cache/ms-playwright`, and `uv run --with playwright python`
  drives it without touching the project env. Two cheap static checks catch the fatal
  classes without a browser: every `import {a, b} from './x.js'` resolves to a file that
  exports those names, and every `data-page="x"` has both a `pages.x` entry in `app.js`
  and a `#page-x` div in `index.html` (currently 15 pages, 13 of them in the nav —
  `overview` and `detail` are reached without a nav item).
- Dashboard POST routes require `X-Requested-With: XMLHttpRequest` (CSRF middleware in
  `src/api/app.py`) — without it you get 403, which is easy to misread as a real failure.
- The Config page's editable fields are created **on click** of a `.config-edit-btn`
  pencil, not rendered up front — so "0 inputs on the page" is expected, and a config item
  that renders but does nothing is only visible by opening its editor.
- The API resolves the tenant from the API key, plus `X-Tenant-Id` for admin keys. A
  `tenant_id` in a request body is ignored, so a curl that passes it there silently reads
  the default tenant.

## Model artifacts (offline posture)

- Three artifacts download on demand: the nomic embedding model + flan-t5-base tokenizer
  (HuggingFace) and tiktoken's `cl100k_base` ranks. In Docker they persist on the
  `hf-cache` / `tiktoken-cache` volumes. Warm them once, then `HF_HUB_OFFLINE=1` makes
  the server fully offline apart from the configured LLM endpoint.
- The embedding model uses `trust_remote_code=True` — first load executes remote Python
  from HF (revision-pinned). `EMBEDDING_INTERNAL__PROVIDER=mock` avoids it entirely.
- Cache dirs are created and chowned to `appuser` **before** the `USER` switch in the
  Dockerfile; a fresh named volume inherits its mount point's ownership, so doing it
  later makes every download fail EACCES.

## Docker notes

- Images build from `python:3.12-slim`. There is deliberately no CUDA toolkit: torch's
  wheels vendor their own CUDA runtime and the GPU comes from nvidia-container-toolkit.
- `torch` must stay an exact `+cu128` pin in `requirements-runtime.txt` — with
  `--extra-index-url` pip takes the highest version across indexes, and PyPI's
  default-CUDA build outranks every cu128 wheel. That is what broke CI for six weeks.
- `ci.yml` runs the suite with `compose run --no-deps`; without it, `app`'s `depends_on`
  drags in `api-test`, which has no CI `image:` override and triggers a fresh build.

## Notes index

- [usage.md](usage.md) — durable reference: server API, endpoints, configuration.
  Linked into by README, CONTRIBUTING, and four py-cml docs (some by anchor).
- [memory-redesign-plan.md](memory-redesign-plan.md) — 2026-08-02 research pass against
  the human-memory literature and 2024-26 LLM-memory benchmarks. Ordered improvement
  plan with evidence tiers, plus a "considered and rejected" section that argues against
  the two biggest available redesigns. Headline: the graph prong returned entity
  profiles instead of the episodic text those entities index — the documented
  Entity-Only failure mode — and edges already carried `evidence_ids`, so fixing it was
  retrieval-only. Shipped in `412a3e3`. Note the prong does **not** run Personalized
  PageRank despite its docstring: GDS is not installed, so every query takes the
  path-count fallback.
