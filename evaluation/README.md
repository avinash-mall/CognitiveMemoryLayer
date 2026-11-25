# Evaluation with CML

This folder contains the [Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) benchmark setup and CML-backed evaluation scripts. Locomo-Plus unifies LoCoMo (five factual categories) with a sixth **Cognitive** category for long-context memory evaluation.

## Layout

| Path | Description |
|------|-------------|
| `locomo_plus/` | Data pipeline (`locomo10.json`, `locomo_plus.json`, `unified_input_samples_v2.json`), `task_eval`, baseline scripts. See [locomo_plus/README.md](locomo_plus/README.md). |
| `cml-eval` CLI (`packages/py-cml/src/cml/eval/`) | Canonical CML-backed evaluation suite (`run-full`, `run-locomo`, `validate`, `report`, `compare`). Install via `pip install -e ".[eval]"`. |
| `scripts/run_locomo_openrouter_batch.py` | End-to-end LoCoMo-Plus driver using OpenRouter's `/api/v1/batches` (`:batch`) API or local OpenAI-compatible server. |
| `results/` | Committed benchmark summary JSON artifacts. |
| `outputs/` | Created at run time; holds predictions, judged records, judge summary, and `run_full_eval_state.json` (failure state for resume). |

## Prerequisites

1. **CML API** running (e.g. via Docker; see main project README).
2. **Embedding** (for CML): project root `.env` — `EMBEDDING_INTERNAL__MODEL`, `EMBEDDING_INTERNAL__DIMENSIONS`. If using Ollama embeddings, pull the model and set dimensions; then drop DBs and re-run migrations if changed.
3. **QA model**: Phase B uses the **LLM** from project root `.env` (`LLM_EVAL__PROVIDER`, `LLM_EVAL__MODEL`, `LLM_EVAL__BASE_URL`, `LLM_EVAL__API_KEY`). Same as [.env.example lines 45–49](../.env.example) — e.g. `openai`, `[REDACTED]`, or `ollama` (with `LLM_EVAL__BASE_URL=[REDACTED]`).
4. **Python deps**: `requests`, `tqdm`, `openai`; for LLM-as-judge: set OPENAI_API_KEY, or provide an OpenAI-compatible endpoint via OPENAI_BASE_URL / LLM_EVAL__BASE_URL.

## Configuration

CML server config (embedding model, rate limit, optional `LLM_INTERNAL__*`) is read from the project root `.env`.

### Performance tuning for bulk eval

Rate limiting is disabled by default (`AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=0`), but the default server worker count and embedding batch size are still conservative for bulk evaluation. For runs with many `--ingestion-workers`, add the following to your project root `.env`:

```bash
# Keep rate limiting disabled during bulk ingestion
AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=0

# Larger embedding batches for GPU (default 8)
EMBEDDING_INTERNAL__LOCAL_BATCH_SIZE=64

# Multiple uvicorn workers (default 1; each loads its own embedding model)
UVICORN_WORKERS=4
```

When running outside Docker, start the server with multiple workers:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

With Docker, the `UVICORN_WORKERS` env var is used automatically by `docker-compose.yml`. For GPU support, include the GPU override:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up api
```

### Chunker (semchunk)

CML uses [semchunk](https://github.com/isaacus-dev/semchunk) with a Hugging Face tokenizer for semantic chunking. Configure in `.env`:

- **CHUNKER__TOKENIZER** — Hugging Face tokenizer model ID (default: google/flan-t5-base)
- **CHUNKER__CHUNK_SIZE** — Max tokens per chunk (default: 500; align with embedding model max input)
- **CHUNKER__OVERLAP_PERCENT** — Overlap ratio 0-1 (default: 0.15 = 15%)

### Environment variables (eval script)

| Variable | Default | Description |
|----------|---------|-------------|
| `CML_BASE_URL` | `http://localhost:8000` | CML API base URL |
| `CML_API_KEY` | `test-key` | API key (must match `AUTH__API_KEY`; for Phase A–B consolidation/reconsolidation must have dashboard/admin permission, e.g. `AUTH__ADMIN_API_KEY`) |
| `LLM_EVAL__PROVIDER` | `openai` | LLM provider for QA: `openai` \| `[REDACTED]` \| `ollama` \| `gemini` \| `claude` (see project root [.env.example](../.env.example) lines 45–49) |
| `LLM_EVAL__MODEL` | `gpt-4o-mini` | Model for QA (e.g. `gpt-4o-mini`, `[REDACTED]` for Ollama) |
| `LLM_EVAL__BASE_URL` | — | OpenAI-compatible endpoint (e.g. `[REDACTED]` for Ollama) |
| `LLM_EVAL__API_KEY` | — | API key (optional for Ollama; server may use `OPENAI_API_KEY`) |
| `OPENAI_API_KEY` | — | Required for LLM-as-judge; also used for `LLM_EVAL__API_KEY` when not set |

## Run full evaluation (`cml-eval`)

Install evaluation extras (`pip install -e ".[eval]"`) and run from the **project root**:

```bash
cml-eval run-full --repo-root .
```

This runs: (1) Docker down -v, (2) Docker up (postgres, neo4j, redis, api), (3) API health wait, (4) `run-locomo` (ingest, consolidation + reconsolidation, QA, judge), (5) performance table. After steps 3, 4, and 5 the pipeline validates outputs; if validation fails, the run stops and writes `evaluation/outputs/run_full_eval_state.json` with the failed step and message. Use **`--resume`** to continue from the failed step (`--resume` implies `--skip-docker`).

### Options

| Option | Description |
|--------|-------------|
| `--skip-docker` | Skip steps 1–3 (use when API is already running) |
| `--resume` | Resume from last failure; implies `--skip-docker`. |
| `--limit-samples N` | Run only first N samples (for quick testing) |

Examples:

```bash
# API already running, quick 50-sample test
cml-eval run-full --skip-docker --limit-samples 50

# Full run without Docker steps
cml-eval run-full --skip-docker

# Resume after a failure (skips Docker, continues from failed step/sample)
cml-eval run-full --resume
```

### Output table

The pipeline prints a table matching the paper format:

| Method | single-hop | multi-hop | temporal | commonsense | adversarial | average | LoCoMo-Plus | Gap |
|--------|------------|-----------|----------|-------------|-------------|---------|-------------|-----|
| CML+&lt;LLM_EVAL__MODEL&gt; | ... | ... | ... | ... | ... | ... | ... | ... |

**Gap** = LoCoMo average − LoCoMo-Plus (performance drop from factual to cognitive memory).

## Run evaluation manually

Between **Phase A** (ingestion) and **Phase B** (QA), the script runs **consolidation** and **reconsolidation** (release labile) for each eval tenant via the dashboard API, unless `--skip-consolidation` is set. The API key must have dashboard/admin permission for this step.

When you need finer control (e.g. skip ingestion, skip consolidation, score-only, verbose):

```bash
cml-eval run-locomo \
  --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json \
  --out-dir evaluation/outputs
```

### `cml-eval run-locomo` options

| Argument | Description |
|----------|-------------|
| `--limit-samples N` | Run only first N samples |
| `--ingestion-workers N` | Concurrent workers for Phase A ingestion (default 10) |
| `--skip-ingestion` | Skip Phase A (reuse existing CML state) |
| `--skip-consolidation` | Skip consolidation and reconsolidation between Phase A and Phase B |
| `--score-only` | Run only Phase C (judge) on existing predictions |
| `--max-results N` | CML read top-k (default 25) |
| `--verbose` | Per-sample retrieval diagnostics |
| `--cml-url`, `--cml-api-key` | Override CML connection |
| `--judge-model` | Model for LLM-as-judge (default: `LLM_EVAL__MODEL` (fallback `LLM_INTERNAL__MODEL`) or gpt-4o-mini) |

### Outputs

| File | Description |
|------|-------------|
| `locomo_plus_qa_cml_predictions.json` | Per-sample predictions (before judge) |
| `locomo_plus_qa_cml_judged.json` | Judged records (judge_label, judge_reason, judge_score) |
| `locomo_plus_qa_cml_judge_summary.json` | Aggregate by category (for report table) |

### Generate report & comparison tables

```bash
cml-eval report --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json --method "CML+qwen3.8-27b"
cml-eval compare --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json --method "CML+qwen3.8-27b"
```

Use `--method` to match your QA model (same as `LLM_EVAL__MODEL` from .env (or fallback `LLM_INTERNAL__MODEL`), e.g. `CML+gpt-4o-mini` or `CML+[REDACTED]`).

## Level-2 Cognitive Memory

The harness supports **LoCoMo-Plus Level-2** (cognitive constraints):

- **Timestamp fidelity** — `DATE:` lines are parsed to UTC and passed to CML write via `timestamp`. Metadata includes `speaker`, `date_str`, `session_idx`.
- **Neutral prompting** — QA prompt avoids memory-aware task disclosure.
- **Constraint-aware retrieval** — With `FEATURES__CONSTRAINT_EXTRACTION_ENABLED=true`, goals, values, policies, states, and causal rules are extracted; decision-style questions trigger constraint-first retrieval.
- **Verbose diagnostics** — Use `--verbose` for per-sample memory type counts and context length.

## Build unified input (optional)

To rebuild `unified_input_samples_v2.json` from source:

```bash
cd evaluation/locomo_plus/data && python unified_input.py
```

Requires `locomo10.json` and `locomo_plus.json` in `evaluation/locomo_plus/data/`.

## Run without CML (OpenAI-style API)

Direct LLM evaluation (no CML ingestion or retrieval):

```bash
# From project root
$env:PYTHONPATH = "evaluation\locomo_plus"   # Windows
export PYTHONPATH=evaluation/locomo_plus     # Unix

python evaluation/locomo_plus/scripts/run_evaluate.py --backend call_llm --model gpt-4o-mini
python evaluation/locomo_plus/scripts/run_judge.py --model gpt-4o-mini
```

Outputs: `evaluation/outputs/locomo_plus_predictions.json`, `evaluation/outputs/locomo_plus_judged.json`. Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) in `evaluation/locomo_plus/scripts/env.local.sh` or the environment.

---

## CML vs Other Methods (Comparison)

See the full, unified 2,387-sample LoCoMo-Plus comparison table, refusal diagnostics, and methodology in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

Committed summary artifact: [`results/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json`](results/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json).

---

## References

- **Locomo-Plus paper:** [arXiv:2602.10715](https://arxiv.org/abs/2602.10715)
- **Locomo-Plus repo:** [github.com/xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus)
- [LoCoMo paper / site](https://snap-research.github.io/locomo/)
- [LoCoMo repo](https://github.com/snap-research/locomo)
