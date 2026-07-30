# Working state

Current snapshot of active work. Update this file as work progresses; add
sibling notes (`docs/<topic>.md`) for anything too big to inline here, and
link them below. Delete notes when they stop being true.

## Where things stand

- **CI is green** (lint + test + build, plus py-cml and CodeQL) as of `215da0f`. That is
  the first passing `CI/CD Pipeline` run in the visible history — every run back through
  2026-06-20 failed at the Docker build, so the integration suite had not executed in six
  weeks. If CI goes red, it was genuinely green here.
- Suite sizes, so drift is visible: **759** unit (hermetic), **338** integration + e2e +
  py-cml against a live server, of which py-cml alone is 241.
- The modelpack removal (`51afd15`) and its follow-up cleanup are complete: dead code,
  stale docs, dead env keys, vendored assets, cache volumes.

## Active work

(nothing in flight)

## Architecture facts (load-bearing)

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

## Known issues / open decisions

- **LoCoMo-Plus has not been re-run** against the current write path. ~10.7k LLM calls
  (5,882 turn ingests + 2,387 QA + 2,387 judge) on shared GPUs; datasets are committed so
  it needs no downloads, only time. Opt-in, not scheduled.
- **Graph relevance is clamped, not normalized.** Every graph hit now lands at exactly 1.0,
  so graph results no longer dominate but are also no longer ordered among themselves. A
  proper per-source normalization is an open improvement.
- **`packages/models/trained_models/` still exists locally** — 241 untracked files
  (15 `.joblib`) with absolute paths from another host. Nothing can load them. Left in
  place deliberately: not in git, so deleting is unrecoverable. Needs an explicit call.
- **Unshipped retrieval improvements**, salvaged from the deleted `Improvement_Report.md`
  (levers A, B and D shipped as `extraction/prospective_indexer.py`, the BM25+RRF hybrid,
  and `extraction/temporal_resolver.py`; the BM25 half is being removed as unwired, leaving
  `rrf_merge` for HyDE):
  - **C — bi-temporal graph edges** (Graphiti-style `valid_from`/`valid_to` on relations),
    so the graph can answer "what did I believe then" rather than only "now".
  - **E — multi-hop iterative retrieval** (IRCoT-style reason/retrieve loop). Multi-hop is
    the weakest measured category, so this is the highest-value one.
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
- Dashboard POST routes require `X-Requested-With: XMLHttpRequest` (CSRF middleware in
  `src/api/app.py`) — without it you get 403, which is easy to misread as a real failure.
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
