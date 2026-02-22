# LoCoMo Evaluation — Complete Runbook

This document describes every step to run the Locomo-Plus evaluation with CML as the RAG backend. The evaluation uses **eval_locomo_plus.py** (unified LoCoMo + Locomo-Plus) and produces a performance table. QA uses the LLM from project `.env` (`LLM__PROVIDER`, `LLM__MODEL`, `LLM__BASE_URL`) — same as the rest of the codebase; no Ollama-specific wiring.

**References:** [evaluation/README.md](../../evaluation/README.md), [Locomo-Plus repo](https://github.com/xjtuleeyf/Locomo-Plus).

---

## 1. Prerequisites (one-time)

### 1.1. Ollama

- Install [Ollama](https://ollama.com) and ensure it is running (`ollama serve` or the desktop app).

### 1.2. Pull models

- **Embedding model** (CML ingestion and retrieval):
  ```bash
  ollama pull <your model>
  ```
  Use a model that matches `EMBEDDING__MODEL` in `.env` (e.g. `mxbai-embed-large` or `embeddinggemma`).
- **QA model** (evaluation script): Uses `LLM__MODEL` from project root `.env` (see Section 2.4). If using Ollama, pull the model (e.g. `ollama pull gpt-oss:20b`) and set `LLM__PROVIDER=ollama`, `LLM__MODEL=gpt-oss:20b`, `LLM__BASE_URL=http://localhost:11434/v1`.

### 1.3. Embedding dimension

CML's vector size must match the embedding model output. Set `EMBEDDING__DIMENSIONS` in `.env` to match your model (e.g. 1024 for mxbai-embed-large, 768 for embeddinggemma). If you change model or dimension, drop databases and re-run migrations (Section 3).

### 1.4. Evaluation data

From project root, confirm:

- `evaluation/locomo_plus/` — task_eval, data pipeline
- `evaluation/locomo_plus/data/locomo10.json` — LoCoMo factual data
- `evaluation/locomo_plus/data/unified_input_samples_v2.json` — unified samples (LoCoMo 5 categories + Cognitive)
- `evaluation/scripts/eval_locomo_plus.py`

### 1.5. Python dependencies

- **Ingestion + QA:** `requests`, `tqdm`
  ```bash
  pip install requests tqdm
  ```
- **LLM-as-judge:** set `OPENAI_API_KEY` in the environment. The judge uses `gpt-4o-mini` by default; override with `--judge-model`.

---

## 2. Project root `.env` (for CML API)

Use the project root `.env` (copy from `.env.example` if needed).

### 2.1. Database

- `DATABASE__POSTGRES_URL=postgresql+asyncpg://memory:memory@localhost/memory`
- `DATABASE__NEO4J_URL=bolt://localhost:7687`
- `DATABASE__NEO4J_USER=neo4j`
- `DATABASE__NEO4J_PASSWORD=password`
- `DATABASE__REDIS_URL=redis://localhost:6379`

### 2.2. Authentication

- `AUTH__API_KEY=test-key` (match `CML_API_KEY` when running the eval script)
- **`AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE`**: Default 60 can cause 429 during bulk ingestion. Set to 600 or higher (e.g. 10000 for heavy bulk). Value is read from `.env`. Rebuild/restart the API after changing.

### 2.3. Embeddings (Ollama)

- `EMBEDDING__PROVIDER=ollama`
- `EMBEDDING__MODEL` — set to your model (e.g. `mxbai-embed-large:latest`, `embeddinggemma` or `embeddinggemma:latest`)
- **`EMBEDDING__DIMENSIONS`** — must match model output (e.g. 1024 for mxbai-embed-large, 768 for embeddinggemma)
- `EMBEDDING__BASE_URL=http://localhost:11434/v1`  
  If the API runs in Docker and Ollama is on the host: `EMBEDDING__BASE_URL=http://host.docker.internal:11434/v1`

### 2.4. LLM (Ollama, for CML internals)

- `LLM__PROVIDER=ollama`
- `LLM__MODEL=gpt-oss:20b` (or your chosen model)
- `LLM__BASE_URL=http://localhost:11434/v1`  
  If API in Docker: `LLM__BASE_URL=http://host.docker.internal:11434/v1`

### 2.5. LLM Internal (optional)

Optional `LLM_INTERNAL__PROVIDER`, `LLM_INTERNAL__MODEL`, `LLM_INTERNAL__BASE_URL`, `LLM_INTERNAL__API_KEY`. When set, the server uses a separate model for chunking and entity/relation extraction; can speed up Phase A ingestion. If not set, `LLM__*` is used.

Example:
```
LLM_INTERNAL__PROVIDER=ollama
LLM_INTERNAL__MODEL=llama3.2:3b
LLM_INTERNAL__BASE_URL=http://host.docker.internal:11434/v1
LLM_INTERNAL__API_KEY=
```

---

## 3. Drop databases and recreate (clean schema)

Do this when you change `EMBEDDING__DIMENSIONS` or need a clean DB.

From **project root**:

1. **Tear down and remove volumes:**
   ```bash
   docker compose -f docker/docker-compose.yml down -v
   ```

2. **Start services** (migrations run on API startup):
   ```bash
   docker compose -f docker/docker-compose.yml up -d --build postgres neo4j redis api
   ```

3. **Wait 30–60 s, then check health:**
   ```bash
   curl -s http://localhost:8000/api/v1/health
   ```

4. **Optional smoke test:**
   ```bash
   curl -s -X POST http://localhost:8000/api/v1/memory/write \
     -H "X-API-Key: test-key" -H "Content-Type: application/json" \
     -d "{\"content\":\"Test memory.\",\"session_id\":\"test\"}"
   curl -s -X POST http://localhost:8000/api/v1/memory/read \
     -H "X-API-Key: test-key" -H "Content-Type: application/json" \
     -d "{\"query\":\"test\",\"format\":\"llm_context\",\"max_results\":5}"
   ```

---

## 4. Run evaluation

### Option A: Full pipeline (recommended)

From **project root**:

```bash
python evaluation/scripts/run_full_eval.py
```

This runs: (1) Docker down -v, (2) Docker up, (3) API health wait, (4) eval_locomo_plus (ingest, QA, judge), (5) performance table. After steps 3, 4, and 5 the pipeline validates outputs; if validation fails, the run stops and writes `evaluation/outputs/run_full_eval_state.json`. Use **`--resume`** to continue from the failed step (and from the next sample if step 4 failed during QA). **`--resume` implies `--skip-docker`** (no need to pass both). Progress bars for each phase display even in IDE terminals and Windows (non-TTY).

**If API is already running:**

```bash
python evaluation/scripts/run_full_eval.py --skip-docker
```

**Quick test (50 samples):**

```bash
python evaluation/scripts/run_full_eval.py --skip-docker --limit-samples 50
```

**Resume after a failure:**

```bash
python evaluation/scripts/run_full_eval.py --resume
```

### Option B: Manual run (eval_locomo_plus only)

Set `OPENAI_API_KEY` and run:

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

### Phases

1. **Phase A:** Ingest each sample into CML (one tenant per sample). `DATE:` lines parsed to UTC; metadata includes `speaker`, `date_str`, `session_idx`.
2. **Phase B:** For each QA item: CML read → LLM generates answer. Predictions are written incrementally; if the run fails, resume continues from the next sample.
3. **Phase C:** LLM-as-judge (correct=1, partial=0.5, wrong=0) writes judged records and summary.

### eval_locomo_plus.py options

| Argument | Description |
|----------|-------------|
| `--limit-samples N` | Run only first N samples |
| `--ingestion-workers N` | Concurrent workers for Phase A ingestion (default 10) |
| `--skip-ingestion` | Skip Phase A (reuse existing CML state) |
| `--score-only` | Run only Phase C on existing predictions |
| `--max-results 25` | CML read top-k |
| `--verbose` | Per-sample retrieval diagnostics |
| `--cml-url`, `--cml-api-key` | CML connection |
| (QA uses `LLM__*` from .env; no CLI override) | — |
| `--judge-model` | Judge model (default gpt-4o-mini) |

---

## 5. Outputs

| File | Description |
|------|-------------|
| `evaluation/outputs/locomo_plus_qa_cml_predictions.json` | Predictions (before judge); written incrementally for resume |
| `evaluation/outputs/locomo_plus_qa_cml_judged.json` | Judged records |
| `evaluation/outputs/locomo_plus_qa_cml_judge_summary.json` | Aggregate by category |
| `evaluation/outputs/run_full_eval_state.json` | Failure state for `--resume` (step, message, last completed sample) |

---

## 6. Generate report table

If you ran manually (not via run_full_eval), generate the table:

```bash
python evaluation/scripts/generate_locomo_report.py --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json --method "CML+gpt-oss:20b"
```

The table shows: Method | single-hop | multi-hop | temporal | commonsense | adversarial | average | LoCoMo-Plus | Gap.  
**Gap** = LoCoMo average − LoCoMo-Plus.

---

## 7. Troubleshooting

| Issue | Action |
|-------|--------|
| **429 Too Many Requests** | Set `AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE` in `.env` to 600 or higher, restart API. |
| **Embedding dimension mismatch** | Set `EMBEDDING__DIMENSIONS` correctly, then run Section 3 again. |
| **Judge fails (no OPENAI_API_KEY)** | Set `OPENAI_API_KEY` or `OPENAI_BASE_URL`. |
| **CML API not reachable** | Use `CML_BASE_URL=http://localhost:8000` (or correct host). |
| **Ollama not reachable from container** | Set `EMBEDDING__BASE_URL` and `LLM__BASE_URL` to `http://host.docker.internal:11434/v1`. |
| **Ollama 404 from script** | Ensure Ollama is running; `OLLAMA_BASE_URL` has no `/v1`. |

---

## 8. Checklist

- [ ] Ollama installed and running
- [ ] Embedding and QA models configured (match `EMBEDDING__MODEL` and `LLM__MODEL` in .env)
- [ ] `EMBEDDING__DIMENSIONS` set in `.env` to match your embedding model (e.g. 1024, 768)
- [ ] `evaluation/locomo_plus/data/unified_input_samples_v2.json` exists
- [ ] Python deps: `requests`, `tqdm`; `OPENAI_API_KEY` set
- [ ] `.env`: DB, `AUTH__API_KEY`, `AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE` set for bulk ingestion, `EMBEDDING__*`, `LLM__*` (optionally `LLM_INTERNAL__*` for faster ingestion)
- [ ] `docker compose -f docker/docker-compose.yml down -v`
- [ ] `docker compose -f docker/docker-compose.yml up -d --build postgres neo4j redis api`
- [ ] `curl -s http://localhost:8000/api/v1/health` OK
- [ ] `python evaluation/scripts/run_full_eval.py` or manual eval + report

---

## 9. Interpretation

The table shows judge scores (0–100%) per category. **Average** is the mean of the five factual categories (LoCoMo). **LoCoMo-Plus** is the Cognitive score. **Gap** = Average − LoCoMo-Plus (positive = factual memory outperforms cognitive memory on this run).
