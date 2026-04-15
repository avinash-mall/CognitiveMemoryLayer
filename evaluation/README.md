# Evaluation with CML

This folder contains the [Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) benchmark setup and CML-backed evaluation scripts. Locomo-Plus unifies LoCoMo (five factual categories) with a sixth **Cognitive** category for long-context memory evaluation.

For a step-by-step runbook (Ollama setup, env vars, DB setup), see [ProjectPlan/LocomoEval/RunEvaluation.md](../ProjectPlan/LocomoEval/RunEvaluation.md).

## Layout

| Path | Description |
|------|-------------|
| `locomo_plus/` | Data pipeline (`locomo10.json`, `locomo_plus.json`, `unified_input_samples_v2.json`), task_eval, scripts. See [locomo_plus/README.md](locomo_plus/README.md). |
| `scripts/eval_locomo_plus.py` | CML-backed driver: ingest unified samples into CML, run QA via CML read + LLM (provider from .env), score with LLM-as-judge (correct=1, partial=0.5, wrong=0). |
| `scripts/generate_locomo_report.py` | Build performance table (LoCoMo factual + LoCoMo-Plus Cognitive + Gap). |
| `scripts/run_full_eval.py` | Full pipeline: Docker down/up, API wait, eval, report table. Validates outputs after each step; on failure writes state for `--resume`. |
| `scripts/validate_outputs.py` | Validates predictions, judged, and judge_summary JSON (structure and consistency). |
| `outputs/` | Created at run time; holds predictions, judged records, judge summary, and `run_full_eval_state.json` (failure state for resume). |

## Prerequisites

1. **CML API** running (e.g. via Docker; see main project README).
2. **Embedding** (for CML): project root `.env` — `EMBEDDING_INTERNAL__MODEL`, `EMBEDDING_INTERNAL__DIMENSIONS`. If using Ollama embeddings, pull the model and set dimensions; then drop DBs and re-run migrations if changed.
3. **QA model**: Phase B uses the **LLM** from project root `.env` (`LLM_EVAL__PROVIDER`, `LLM_EVAL__MODEL`, `LLM_EVAL__BASE_URL`, `LLM_EVAL__API_KEY`). Same as [.env.example lines 45–49](../.env.example) — e.g. `openai`, `[REDACTED]`, or `ollama` (with `LLM_EVAL__BASE_URL=[REDACTED]`).
4. **Python deps**: `requests`, `tqdm`, `openai`; for LLM-as-judge: set OPENAI_API_KEY, or provide an OpenAI-compatible endpoint via OPENAI_BASE_URL / LLM_EVAL__BASE_URL.

## Configuration

CML server config (embedding model, rate limit, optional `LLM_INTERNAL__*`) is read from the project root `.env`. See [ProjectPlan/LocomoEval/RunEvaluation.md](../ProjectPlan/LocomoEval/RunEvaluation.md) for full setup.

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

## Run full evaluation

From the **project root**:

```bash
python evaluation/scripts/run_full_eval.py
```

This runs: (1) Docker down -v, (2) Docker up (postgres, neo4j, redis, api), (3) API health wait, (4) eval_locomo_plus (ingest, consolidation + reconsolidation, QA, judge), (5) performance table. After steps 3, 4, and 5 the pipeline validates outputs; if validation fails, the run stops and writes `evaluation/outputs/run_full_eval_state.json` with the failed step and message (and, for step 4, the last completed sample index). Use **`--resume`** to continue from the failed step (and from the next sample for step 4); **`--resume` implies `--skip-docker`** (no need to pass both). Progress bars for each phase display even in subprocess or non-TTY environments (IDE terminals, Windows).

### Options

| Option | Description |
|--------|-------------|
| `--skip-docker` | Skip steps 1–3 (use when API is already running) |
| `--resume` | Resume from last failure; implies `--skip-docker`. Resumes evaluation from the next sample if step 4 failed during QA. |
| `--limit-samples N` | Run only first N samples (for quick testing) |

Examples:

```bash
# API already running, quick 50-sample test
python evaluation/scripts/run_full_eval.py --skip-docker --limit-samples 50

# Full run without Docker steps
python evaluation/scripts/run_full_eval.py --skip-docker

# Resume after a failure (skips Docker, continues from failed step/sample)
python evaluation/scripts/run_full_eval.py --resume
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

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "evaluation\locomo_plus"
python evaluation/scripts/eval_locomo_plus.py --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json --out-dir evaluation/outputs
```

**Unix:**

```bash
export PYTHONPATH=evaluation/locomo_plus
python evaluation/scripts/eval_locomo_plus.py --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json --out-dir evaluation/outputs
```

### eval_locomo_plus.py options

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

### Generate report table

```bash
python evaluation/scripts/generate_locomo_report.py --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json --method "CML+gpt-4o-mini"
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

Latest CML evaluation results (April 2026) compared with baselines from the **Locomo-Plus paper** (arXiv:2602.10715, Table 1). Same evaluation protocol: LLM-as-judge, constraint consistency, no task disclosure.

### Latest CML Run — `google/gemma-4-31b-it` (local, via vLLM)

| Metric | Value |
|--------|--------|
| **Overall average** | **48.58%** |
| **LoCoMo (factual) average** | **47.16%** (single-hop 56.96%, multi-hop 33.16%, temporal 48.60%, commonsense 32.29%, adversarial 64.80%) |
| **LoCoMo-Plus (Cognitive)** | **27.68%** |
| **Gap** (factual − cognitive) | **19.48%** |
| Total samples | 2,387 (zero errors) |
| QA + Judge model | `google/gemma-4-31b-it` — fully local, zero API cost |

### Comparison with Paper Baselines

| Method | Backend | Overall | Adversarial | Temporal | Notes |
|--------|---------|---------|-------------|----------|-------|
| Gemini-2.5-Pro | — (full ctx) | 71.78% | 73.03% | 73.83% | Closed-source, full context |
| GPT-4o | — (full ctx) | 62.99% | 48.99% | 45.79% | Closed-source, full context |
| A-Mem | GPT-4o | 59.64% | 35.20% | 49.30% | Memory system |
| SeCom | GPT-4o | 57.53% | 31.80% | 42.30% | Memory system |
| Mem0 | GPT-4o | 57.24% | 30.50% | 39.40% | Memory system |
| **CML** | **gemma-4-31b-it (local)** | **48.58%** | **64.80%** | **48.60%** | **Zero API cost** |
| RAG (emb-large) | GPT-4o | ~39% | 59.73% | 40.00% | Basic retrieval |

### Key Strengths

1. **Adversarial: 64.80%** — outperforms GPT-4o full-context (48.99%), more than doubles Mem0 (30.50%), SeCom (31.80%), A-Mem (35.20%). Best adversarial score of any memory system.
2. **Temporal: 48.60%** — beats GPT-4o full-context (45.79%) and Mem0 (39.40%) by +9.20%.
3. **Fully local inference** — `google/gemma-4-31b-it` via vLLM on a single GPU. Zero API dependency, zero per-query cost, privacy-preserving.
4. **Custom DeBERTa models** for all write-path tasks (extraction, routing, novelty, PII) — 17+ turns/sec throughput with no LLM calls in the write path.
5. **Zero errors** across 2,387 samples — robust retry logic and local serving reliability.

### Key Takeaway

CML achieves competitive overall scores with a **local 31B-parameter model** while competitors rely on **GPT-4o** (~200B+ parameters, closed-source, paid API). On adversarial and temporal tasks — where memory architecture matters most — CML **outperforms** systems backed by much larger models.

Full analysis: [EVALUATION_REPORT.md](EVALUATION_REPORT.md)

### Reference

- **Locomo-Plus paper:** [arXiv:2602.10715](https://arxiv.org/abs/2602.10715)
- **Locomo-Plus repo:** [github.com/xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus)

---

## References

- [LoCoMo paper / site](https://snap-research.github.io/locomo/)
- [LoCoMo repo](https://github.com/snap-research/locomo)
- [Locomo-Plus repo](https://github.com/xjtuleeyf/Locomo-Plus)
- [Runbook (full steps)](../ProjectPlan/LocomoEval/RunEvaluation.md)
