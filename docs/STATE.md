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

(none open)

## Retrieval notes

- **Sources do not share a relevance scale.** Vector/fact prongs emit cosine-like 0..1;
  the graph prong passes through a raw Neo4j co-occurrence score that is unbounded.
  `MemoryReranker._score_components` clamps relevance to [0,1] and notes the clamp — do
  not remove that guard. A proper per-source normalization is still an open improvement:
  today every graph hit lands at exactly 1.0, so graph hits no longer dominate but are
  also no longer ordered among themselves.
- Reranker breakdown rows must keep the key names in `RetrievalExplainRerankItem`
  (`id`, `source_type`, ...) — the dashboard renders them directly and the response model
  validates them, so renaming a key 500s the explain endpoint.

## Dashboard notes

- All third-party assets are vendored in `src/dashboard/static/vendor/` (chart.js,
  vis-network, Inter + JetBrains Mono woff2) with hashes and licenses recorded in that
  directory's README. There is **no build step** — `src/api/app.py` serves the static
  tree verbatim, so any CDN URL added to it ships straight to users. Keep
  `grep -rn "https://" src/dashboard/static` (excluding `vendor/`) empty.

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

## Notes index

(none yet)
