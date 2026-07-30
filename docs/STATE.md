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

## Dashboard notes

- All third-party assets are vendored in `src/dashboard/static/vendor/` (chart.js,
  vis-network, Inter + JetBrains Mono woff2) with hashes and licenses recorded in that
  directory's README. There is **no build step** — `src/api/app.py` serves the static
  tree verbatim, so any CDN URL added to it ships straight to users. Keep
  `grep -rn "https://" src/dashboard/static` (excluding `vendor/`) empty.

## Docker notes

- Images build from `python:3.12-slim`. There is deliberately no CUDA toolkit: torch's
  wheels vendor their own CUDA runtime and the GPU comes from nvidia-container-toolkit.
- `torch` must stay an exact `+cu128` pin in `requirements-runtime.txt` — with
  `--extra-index-url` pip takes the highest version across indexes, and PyPI's
  default-CUDA build outranks every cu128 wheel. That is what broke CI for six weeks.

## Notes index

(none yet)
