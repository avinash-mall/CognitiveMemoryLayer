# LoCoMo Evaluation — Complete Runbook

This document lists **every step** required to run the Locomo-Plus evaluation with CML as the RAG backend and local Ollama (e.g. `gpt-oss-20b` for QA, `embeddinggemma` for embeddings). The evaluation uses **eval_locomo_plus.py** (unified LoCoMo + Locomo-Plus) and produces a performance table. Do not skip steps; order matters.

**References:** [evaluation/README.md](../../evaluation/README.md) (layout and overview), [Locomo-Plus repo](https://github.com/xjtuleeyf/Locomo-Plus). Strategy and steps are described in this runbook.

---

## 1. Prerequisites (one-time)

### 1.1. Ollama installed and running

- Install [Ollama](https://ollama.com) on the machine that will run the evaluation (or where the CML API will reach it).
- Ensure the Ollama service is running (e.g. `ollama serve` or the desktop app).

### 1.2. Pull required models in Ollama

- **Embedding model** (used by CML for ingestion and retrieval):
  ```bash
  ollama pull embeddinggemma
  ```
- **QA model** (used by the evaluation script to generate answers):
  ```bash
  ollama pull gpt-oss-20b
  ```
  (Or use another model and set `OLLAMA_QA_MODEL` / `--ollama-model` when running the script.)

### 1.3. Confirm embedding dimension

- CML's database vector size **must** match the embedding model's output dimension. The embedding model can be **EmbeddingGemma** (typically 768) or another (e.g. **mxbai-embed-large**, 1024). Set `EMBEDDING__MODEL` and **`EMBEDDING__DIMENSIONS`** in `.env` to match the model in use; if you change either, you must drop databases and re-run migrations (Section 3).
- Confirm dimension on your setup, e.g.:
  ```bash
  ollama run embeddinggemma "test"
  ```
  Or call the [Ollama embed API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings) and check `len(embeddings[0])`. Use this number as `EMBEDDING__DIMENSIONS` in Section 2.3.

### 1.4. Evaluation folder and data

- From the **project root** (`CognitiveMemoryLayer/`):
  - Confirm the evaluation folder exists and contains Locomo-Plus:
    - `evaluation/locomo_plus/` — task_eval, data pipeline
    - `evaluation/locomo_plus/data/locomo10.json` — LoCoMo factual data
    - `evaluation/locomo_plus/data/unified_input_samples_v2.json` — unified samples (LoCoMo 5 categories + Cognitive)
  - Confirm the script exists: `evaluation/scripts/eval_locomo_plus.py`.

### 1.5. Python dependencies for the evaluation script

- **Minimum** (for ingestion + QA): `requests`, `tqdm`.
  ```bash
  pip install requests tqdm
  ```
- **For LLM-as-judge** (Phase C): set **`OPENAI_API_KEY`** in the environment (or point `OPENAI_BASE_URL` at an Ollama-compatible endpoint). The judge uses `gpt-4o-mini` by default; override with `--judge-model`.

---

## 2. Project root `.env` (required for CML API)

All of the following are read by the CML API (and migrations). Use the **project root** `.env` (copy from `.env.example` if needed).

### 2.1. Database (match `docker/docker-compose.yml` for local run)

- `DATABASE__POSTGRES_URL=postgresql+asyncpg://memory:memory@localhost/memory`  
  (If CML runs in Docker, the API container uses `postgres:5432`; these values are set in compose.)
- `DATABASE__NEO4J_URL=bolt://localhost:7687`
- `DATABASE__NEO4J_USER=neo4j`
- `DATABASE__NEO4J_PASSWORD=password`
- `DATABASE__REDIS_URL=redis://localhost:6379`

### 2.2. Authentication

- `AUTH__API_KEY=test-key`  
  (Use the same value as `CML_API_KEY` when running the evaluation script.)
- Optionally: `AUTH__ADMIN_API_KEY=test-key`, `AUTH__DEFAULT_TENANT_ID=default`.
- **`AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE`** (default 60): Rate limit per tenant. The default 60/min causes **429 Too Many Requests** during bulk ingestion. For full evaluation set to **600** (or higher) in project root `.env`, then rebuild/restart the API.

### 2.3. Embeddings (Ollama) — critical

- `EMBEDDING__PROVIDER=ollama`
- `EMBEDDING__MODEL=embeddinggemma` (or `embeddinggemma:latest`), or e.g. `mxbai-embed-large:latest` (1024 dimensions).
- **`EMBEDDING__DIMENSIONS`** must match the model output (e.g. **768** for EmbeddingGemma, **1024** for mxbai-embed-large). Use the value you confirmed in Section 1.3.
- `EMBEDDING__BASE_URL=http://localhost:11434/v1`  
  If the **API runs inside Docker** and Ollama is on the host, use:  
  `EMBEDDING__BASE_URL=http://host.docker.internal:11434/v1`

**Important:** The migration creates the pgvector column using `EMBEDDING__DIMENSIONS` at import time. If you change model or dimension later, you must drop databases and re-run migrations (Section 3).

### 2.4. LLM (Ollama, for CML internals if used)

- `LLM__PROVIDER=ollama`
- `LLM__MODEL=gpt-oss-20b`
- `LLM__BASE_URL=http://localhost:11434/v1`  
  If the API runs in Docker and Ollama is on the host:  
  `LLM__BASE_URL=http://host.docker.internal:11434/v1`

---

## 3. Drop databases and recreate Docker (clean schema)

Do this **whenever** you change `EMBEDDING__DIMENSIONS` or need a clean DB (e.g. first run or after switching embedding model).

From the **project root**:

1. **Tear down containers and remove volumes** (this deletes Postgres/Neo4j/Redis data):
   ```bash
   docker compose -f docker/docker-compose.yml down -v
   ```

2. **Start dependencies and the API** (migrations run on API startup):
   ```bash
   docker compose -f docker/docker-compose.yml up -d --build postgres neo4j redis api
   ```

3. **Wait for services** (e.g. 30–60 s), then **verify CML health**:
   ```bash
   curl -s http://localhost:8000/api/v1/health
   ```
   You should get a successful JSON response.

4. (Optional) **Smoke test** the API (write + read) to confirm embedding and DB:
   ```bash
   curl -s -X POST http://localhost:8000/api/v1/memory/write \
     -H "X-API-Key: test-key" -H "Content-Type: application/json" \
     -d "{\"content\":\"Test memory for LoCoMo.\",\"session_id\":\"test\"}"
   curl -s -X POST http://localhost:8000/api/v1/memory/read \
     -H "X-API-Key: test-key" -H "Content-Type: application/json" \
     -d "{\"query\":\"test\",\"format\":\"llm_context\",\"max_results\":5}"
   ```

---

## 4. Run the evaluation script

From the **project root**, with **CML API** and **Ollama** running. Set **`OPENAI_API_KEY`** for the LLM-as-judge phase.

### 4.1. Set PYTHONPATH and run (full evaluation)

- **Windows (PowerShell):**
  ```powershell
  $env:PYTHONPATH = "evaluation\locomo_plus"
  python evaluation/scripts/eval_locomo_plus.py --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json --out-dir evaluation/outputs
  ```

- **Unix / WSL:**
  ```bash
  export PYTHONPATH=evaluation/locomo_plus
  python evaluation/scripts/eval_locomo_plus.py --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json --out-dir evaluation/outputs
  ```

The script will:
1. **Phase A:** Ingest each sample's conversation into CML (one tenant per sample).
2. **Phase B:** For each QA item, call CML read (`llm_context`), then Ollama to generate an answer.
3. **Phase C:** Run LLM-as-judge (correct=1, partial=0.5, wrong=0) and write judged records and summary.

**Ingestion throttling and retries:** The eval script uses a short delay between CML writes and retries with backoff on 429. With `AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=600` set (Section 2.2), ingestion proceeds without hitting the limit; a full run can still take many hours due to embedding and LLM latency.

### 4.2. Optional script arguments

| Argument | Description |
|----------|-------------|
| `--limit-samples N` | Run only the first N samples (e.g. `5` for a quick test). |
| `--skip-ingestion` | Skip Phase A (reuse existing CML state; use only if data is already ingested). |
| `--score-only` | Run only Phase C (LLM-as-judge) on existing predictions. |
| `--max-results 25` | CML read top-k (default 25). |
| `--cml-url URL` | CML API base URL (default from `CML_BASE_URL` or `http://localhost:8000`). |
| `--cml-api-key KEY` | API key (default from `CML_API_KEY` or `test-key`). |
| `--ollama-url URL` | Ollama base URL without `/v1` (default from `OLLAMA_BASE_URL` or `http://localhost:11434`). |
| `--ollama-model NAME` | Ollama model for QA (default from `OLLAMA_QA_MODEL` or `gpt-oss-20b`). |
| `--judge-model NAME` | Model for LLM-as-judge (default `gpt-4o-mini`). |

Example (five samples, quick test):
```bash
export PYTHONPATH=evaluation/locomo_plus
python evaluation/scripts/eval_locomo_plus.py --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json --out-dir evaluation/outputs --limit-samples 5
```

### 4.3. Environment variables for the script (optional overrides)

- `CML_BASE_URL` — CML API base URL (default `http://localhost:8000`).
- `CML_API_KEY` — Must match `AUTH__API_KEY` (default `test-key`).
- `OLLAMA_BASE_URL` — Ollama URL, no `/v1` (default `http://localhost:11434`).
- `OLLAMA_QA_MODEL` — Model for QA (default `gpt-oss-20b`).
- `OPENAI_API_KEY` — Required for LLM-as-judge (or set `OPENAI_BASE_URL` for Ollama-compatible endpoint).

---

## 5. Outputs

- **`evaluation/outputs/locomo_plus_qa_cml_predictions.json`** — Per-sample predictions (before judge).
- **`evaluation/outputs/locomo_plus_qa_cml_judged.json`** — Judged records (judge_label, judge_score).
- **`evaluation/outputs/locomo_plus_qa_cml_judge_summary.json`** — Aggregate stats by category (single-hop, multi-hop, temporal, common-sense, adversarial, Cognitive).

Scoring is only run when there are predictions.

---

## 6. Generate report table

From the **project root**, after the evaluation completes:

```bash
python evaluation/scripts/generate_locomo_report.py --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json --method "CML+gpt-oss:20b"
```

This prints a table with columns: Method | single-hop | multi-hop | temporal | commonsense | adversarial | average | LoCoMo-Plus | Gap. **Gap** = average (of five factual categories) − Cognitive.

---

## 7. Troubleshooting

| Issue | What to do |
|-------|------------|
| **429 Too Many Requests** on write | Set `AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=600` in project root `.env`, then rebuild/restart the API: `docker compose -f docker/docker-compose.yml up -d --build api`. |
| **422 Unprocessable Entity** on write | Check session timestamps and content format; the script parses input_prompt into turns for CML ingestion. |
| **Embedding dimension mismatch** (DB vs model) | Set `EMBEDDING__DIMENSIONS` to the actual model output (e.g. 768 for EmbeddingGemma, 1024 for mxbai-embed-large). Then run **Section 3** again (`down -v` and `up -d --build ...`). |
| **Judge fails (no OPENAI_API_KEY)** | Set `OPENAI_API_KEY` or point `OPENAI_BASE_URL` at an Ollama-compatible endpoint. |
| **CML API not reachable** | If the script runs on the host and CML is in Docker, use `CML_BASE_URL=http://localhost:8000`. If both are in Docker, use the API service name or `host.docker.internal`. |
| **Ollama not reachable from CML container** | Set `EMBEDDING__BASE_URL` and `LLM__BASE_URL` to `http://host.docker.internal:11434/v1` in `.env`. |
| **404 or connection errors to Ollama from the script** | Ensure Ollama is running and `OLLAMA_BASE_URL` (or `--ollama-url`) is correct; no `/v1` in the base URL (script adds it for chat/completions). |
| **Wrong or missing predictions** | Do not use `--skip-ingestion` unless you have already ingested the same data. |

---

## 8. Checklist (quick copy)

- [ ] Ollama installed and running
- [ ] `ollama pull embeddinggemma` and `ollama pull gpt-oss-20b` (or chosen QA model)
- [ ] Embedding dimension confirmed (e.g. 768 or 1024) and set in `.env` as `EMBEDDING__DIMENSIONS`
- [ ] `evaluation/locomo_plus/` present; `evaluation/locomo_plus/data/unified_input_samples_v2.json` exists; `evaluation/scripts/eval_locomo_plus.py` exists
- [ ] Python deps: `requests`, `tqdm`; set `OPENAI_API_KEY` for LLM-as-judge
- [ ] Project root `.env` set: DB, `AUTH__API_KEY`, **`AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=600`** for full evaluation, `EMBEDDING__*` (including `EMBEDDING__DIMENSIONS`), `LLM__*`; use `host.docker.internal` if API in Docker and Ollama on host
- [ ] `docker compose -f docker/docker-compose.yml down -v`
- [ ] `docker compose -f docker/docker-compose.yml up -d --build postgres neo4j redis api`
- [ ] `curl -s http://localhost:8000/api/v1/health` OK
- [ ] `PYTHONPATH=evaluation/locomo_plus` (or `evaluation\locomo_plus` on Windows)
- [ ] `python evaluation/scripts/eval_locomo_plus.py --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json --out-dir evaluation/outputs`
- [ ] Check `evaluation/outputs/locomo_plus_qa_cml_judge_summary.json`
- [ ] `python evaluation/scripts/generate_locomo_report.py --method "CML+gpt-oss:20b"`

---

## 9. Run all phases (one script)

The script **`evaluation/scripts/run_full_eval.py`** runs all phases in order and prints the performance table. It does not start a separate monitor; it runs the four steps sequentially. To run without attaching to the terminal (e.g. so it continues after you close the shell), use the background commands below.

**Phases:** (1) Tear down containers and volumes (`docker compose down -v`), (2) Build and start postgres, neo4j, redis, api, (3) Wait for CML API health, (4) Run Locomo-Plus evaluation (ingestion, QA, LLM-as-judge) via `eval_locomo_plus.py`, then generate and print the report table.

### 9.1. Command to run (foreground)

From the **project root**:

```bash
python evaluation/scripts/run_full_eval.py
```

Progress and the current step are printed to stdout. The script exits when all phases complete or when a step fails. The final output is the performance table (Method | single-hop | multi-hop | temporal | commonsense | adversarial | average | LoCoMo-Plus | Gap).

### 9.2. Command to run without monitoring (background)

Run the same script in the background so it continues after you close the terminal. Ensure **`evaluation/outputs`** exists (the script or eval creates it).

**Windows (PowerShell)** — from project root:

```powershell
New-Item -ItemType Directory -Force -Path evaluation/outputs | Out-Null; Start-Process -FilePath python -ArgumentList "evaluation/scripts/run_full_eval.py" -NoNewWindow -RedirectStandardOutput evaluation/outputs/run_full_eval.log -RedirectStandardError evaluation/outputs/run_full_eval_err.log
```

**Unix / WSL** — from project root:

```bash
mkdir -p evaluation/outputs && nohup python evaluation/scripts/run_full_eval.py > evaluation/outputs/run_full_eval.log 2>&1 &
```

Check progress with `tail -f evaluation/outputs/run_full_eval.log` (Unix) or by opening `evaluation/outputs/run_full_eval.log` (Windows).

---

## 10. Interpretation

The report table shows judge scores (0–100%) for each category. **Average** is the mean of the five factual categories (LoCoMo). **LoCoMo-Plus** is the Cognitive category score. **Gap** = Average − LoCoMo-Plus (positive means factual memory outperforms cognitive memory on this run).

Same **annotation file** and same **judge** (correct=1, partial=0.5, wrong=0). Compare CML (memory-backed RAG) across categories to understand strengths and gaps.
