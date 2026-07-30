# Working state

Current snapshot of active work. Update this file as work progresses; add
sibling notes (`docs/<topic>.md`) for anything too big to inline here, and
link them below. Delete notes when they stop being true.

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

## Active work

(nothing in flight)

## Known issues

- `CI/CD Pipeline` test job fails at Docker image build (megablocks: CUDA 12.8
  builder image vs CUDA 13.0 torch wheel from PyPI). Pre-existing since
  2026-06-20, unrelated to the modelpack removal.
- Offline rule (CLAUDE.md #1) violations in the dashboard: chart.js loaded
  from jsdelivr in `src/dashboard/static/index.html:8`; vis-network CDN
  fallback in `src/dashboard/static/js/pages/graph.js:55-58` (a local
  `bundle.js` is built for the graph, but the fallback still points at a CDN).
  Both need bundling to comply.

## Notes index

(none yet)
