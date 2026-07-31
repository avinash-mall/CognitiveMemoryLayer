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
- Suite sizes, so drift is visible: **740** unit (hermetic), **337** integration + e2e +
  py-cml against a live server. Unit was 716, then 721 (uncommitted drift), then +19 for
  source monitoring, gist demotion and the retention curve. Not regressions.
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

(nothing in flight)

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
  Cognitive 0.21. Artifact: `evaluation/results/locomo_plus_2026-07-31.json`. Conditions:
  server at 983a9f9 (4 uvicorn workers, CPU embedder), ingestion via eval-mode writes
  (X-Eval-Mode skips unified extraction — no LLM enrichment on stored memories), QA+judge
  on local Qwen3.6-27B-FP8. Per lever G these numbers are NOT comparable to published
  gemini-judged baselines — only relative movement against this artifact is meaningful.
  Multi-hop and temporal remain the weak categories, consistent with lever E being unshipped.
  Full pipeline cost on this host: ~11.5h ingestion (218k turns, shared GPUs) + ~2h QA +
  ~0.5h judge.
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
- **"Multi-hop" retrieval has no depth control.** `NeocorticalStore.multi_hop_query`
  runs Personalized PageRank from the seeds, takes the top 20, keeps 10, and attaches
  each entity's relations and facts. There is no hop loop. It used to accept
  `max_hops` and never read it (the retriever passed `3`), which is worth knowing
  because multi-hop is the weakest measured retrieval category — lever E in the
  unshipped list below is the thing that would actually add iterative depth.
- **Graph relevance is clamped, not normalized.** Every graph hit now lands at exactly 1.0,
  so graph results no longer dominate but are also no longer ordered among themselves. A
  proper per-source normalization is an open improvement.
- **`event_log` is now an orphan table.** Nothing ever wrote a row, so its whole read
  surface (routes, dashboard panels, SDK `get_events`, `EventLogModel`) was removed.
  The table and `migrations/versions/001_initial_schema.py` were deliberately left alone
  — a destructive migration doesn't belong in a cleanup. Drop it in a migration whenever
  someone is willing to own the data loss, or resurrect the writer instead.
- **Unshipped retrieval improvements**, salvaged from the deleted `Improvement_Report.md`
  (levers A, B and D shipped as `extraction/prospective_indexer.py`, the BM25+RRF hybrid,
  and `extraction/temporal_resolver.py`. The BM25 half has since been removed as unwired —
  no plan step ever produced a sparse retrieval step — leaving `rrf_merge` in
  `src/retrieval/rrf.py` for the HyDE merge):
  - **C — bi-temporal graph edges** (Graphiti-style `valid_from`/`valid_to` on relations),
    so the graph can answer "what did I believe then" rather than only "now". Related and
    larger: there is no sequence structure at all, so "what happened before X" is
    unanswerable — it needs X resolved to a timestamp first, and nothing does that.
    `planner.py` handles three English literals ("today"/"yesterday"/"week").
  - **E — multi-hop iterative retrieval** (IRCoT-style reason/retrieve loop). Multi-hop is
    the weakest measured category, so this is the highest-value one.
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
