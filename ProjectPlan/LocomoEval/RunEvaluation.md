# LoCoMo Evaluation — Complete Runbook

This document lists **every step** required to run the LoCoMo evaluation with CML as the RAG backend and local Ollama (e.g. `gpt-oss-20b` for QA, `embeddinggemma` for embeddings). Do not skip steps; order matters.

**References:** [EvaluationPlan.md](EvaluationPlan.md) (strategy), [evaluation/README.md](../../evaluation/README.md) (layout), [LoCoMo repo](https://github.com/snap-research/locomo).

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

- CML’s database vector size **must** match the embedding model’s output dimension. The embedding model can be **EmbeddingGemma** (typically 768) or another (e.g. **mxbai-embed-large**, 1024). Set `EMBEDDING__MODEL` and **`EMBEDDING__DIMENSIONS`** in `.env` to match the model in use; if you change either, you must drop databases and re-run migrations (Section 3).
- Confirm dimension on your setup, e.g.:
  ```bash
  ollama run embeddinggemma "test"
  ```
  Or call the [Ollama embed API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings) and check `len(embeddings[0])`. Use this number as `EMBEDDING__DIMENSIONS` in Section 2.3.

### 1.4. Evaluation folder and LoCoMo repo

- From the **project root** (`CognitiveMemoryLayer/`):
  - Create the evaluation folder and clone LoCoMo if not already done:
    ```bash
    mkdir -p evaluation
    git clone https://github.com/snap-research/locomo.git evaluation/locomo
    ```
  - Confirm the dataset exists: `evaluation/locomo/data/locomo10.json`.
  - Confirm the script exists: `evaluation/scripts/eval_locomo.py`.

### 1.5. Python dependencies for the evaluation script

- **Minimum** (for ingestion + QA): `requests`, `tqdm`.
  ```bash
  pip install requests tqdm
  ```
- **For scoring** (Phase C: F1, recall, stats): LoCoMo’s `task_eval` needs `bert-score`, `nltk`, `regex`, `numpy`. Either:
  - Install from LoCoMo: `pip install -r evaluation/locomo/requirements.txt`, or  
  - Install only scoring deps: `pip install bert-score nltk regex numpy`
- If using `nltk`, download data once:
  ```bash
  python -m nltk.downloader punkt
  ```

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
- **`AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE`** (default 60): Rate limit per tenant. The default 60/min causes **429 Too Many Requests** during bulk LoCoMo ingestion. For full evaluation set to **600** (or higher) in project root `.env`, then rebuild/restart the API.

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

From the **project root**, with **CML API** and **Ollama** running.

### 4.1. Set PYTHONPATH and run (full evaluation)

- **Windows (PowerShell):**
  ```powershell
  $env:PYTHONPATH = "evaluation\locomo"
  python evaluation/scripts/eval_locomo.py --data-file evaluation/locomo/data/locomo10.json --out-dir evaluation/outputs
  ```

- **Unix / WSL:**
  ```bash
  export PYTHONPATH=evaluation/locomo
  python evaluation/scripts/eval_locomo.py --data-file evaluation/locomo/data/locomo10.json --out-dir evaluation/outputs
  ```

The script will:
1. **Phase A:** Ingest each sample’s conversation into CML (one tenant per `sample_id`).
2. **Phase B:** For each QA item, call CML read (`llm_context`), then Ollama to generate an answer.
3. **Phase C:** Run LoCoMo’s `eval_question_answering` and `analyze_aggr_acc`, and write the output and stats files.

**Ingestion throttling and retries:** The eval script uses a short delay between CML writes (`INGESTION_DELAY_SEC = 0.2` in [evaluation/scripts/eval_locomo.py](../../evaluation/scripts/eval_locomo.py)) and retries with backoff on 429. With `AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=600` set (Section 2.2), ingestion proceeds without hitting the limit; a full run can still take many hours due to embedding and LLM latency.

**Session timestamps:** LoCoMo data uses human-readable session times (e.g. `"1:56 pm on 8 May, 2023"`). The evaluation script converts these to ISO for CML’s `timestamp` field using **python-dateutil** when available; otherwise it omits `timestamp` to avoid 422. Optional: `pip install python-dateutil`.

### 4.2. Optional script arguments

| Argument | Description |
|----------|-------------|
| `--limit-samples N` | Run only the first N samples (e.g. `1` for a quick test). |
| `--skip-ingestion` | Skip Phase A (reuse existing CML state; use only if data is already ingested). |
| `--overwrite` | Overwrite existing predictions in the output file. |
| `--max-results 25` | CML read top-k (default 25). |
| `--cml-url URL` | CML API base URL (default from `CML_BASE_URL` or `http://localhost:8000`). |
| `--cml-api-key KEY` | API key (default from `CML_API_KEY` or `test-key`). |
| `--ollama-url URL` | Ollama base URL without `/v1` (default from `OLLAMA_BASE_URL` or `http://localhost:11434`). |
| `--ollama-model NAME` | Ollama model for QA (default from `OLLAMA_QA_MODEL` or `gpt-oss-20b`). |

Example (one sample, quick test):
```bash
export PYTHONPATH=evaluation/locomo
python evaluation/scripts/eval_locomo.py --data-file evaluation/locomo/data/locomo10.json --out-dir evaluation/outputs --limit-samples 1
```

### 4.3. Environment variables for the script (optional overrides)

- `CML_BASE_URL` — CML API base URL (default `http://localhost:8000`).
- `CML_API_KEY` — Must match `AUTH__API_KEY` (default `test-key`).
- `OLLAMA_BASE_URL` — Ollama URL, no `/v1` (default `http://localhost:11434`).
- `OLLAMA_QA_MODEL` — Model for QA (default `gpt-oss-20b`).

---

## 5. Outputs

- **`evaluation/outputs/locomo10_qa_cml.json`** — Per-sample QA with predictions, F1, and recall (and `_context` for recall).
- **`evaluation/outputs/locomo10_qa_cml_stats.json`** — Aggregate stats by category (counts, accuracy, recall).

Scoring is only run when there are predictions (e.g. not when `--limit-samples 0` with no QA phase).

---

## 6. Troubleshooting

| Issue | What to do |
|-------|------------|
| **429 Too Many Requests** on write | Set `AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=600` in project root `.env`, then rebuild/restart the API: `docker compose -f docker/docker-compose.yml up -d --build api`. |
| **422 Unprocessable Entity** on write | LoCoMo session dates (e.g. `"1:56 pm on 8 May, 2023"`) must be parseable or omitted. Install `python-dateutil` so the script can convert them to ISO; otherwise the script omits `timestamp` and the write should succeed. |
| **Embedding dimension mismatch** (DB vs model) | Set `EMBEDDING__DIMENSIONS` to the actual model output (e.g. 768 for EmbeddingGemma, 1024 for mxbai-embed-large). Then run **Section 3** again (`down -v` and `up -d --build ...`). |
| **`ModuleNotFoundError: bert_score`** (or nltk, regex, numpy) | Install scoring deps (Step 1.5). Or run with `--limit-samples 0 --skip-ingestion` only (no scoring). |
| **CML API not reachable** | If the script runs on the host and CML is in Docker, use `CML_BASE_URL=http://localhost:8000`. If both are in Docker, use the API service name or `host.docker.internal`. |
| **Ollama not reachable from CML container** | Set `EMBEDDING__BASE_URL` and `LLM__BASE_URL` to `http://host.docker.internal:11434/v1` in `.env`. |
| **404 or connection errors to Ollama from the script** | Ensure Ollama is running and `OLLAMA_BASE_URL` (or `--ollama-url`) is correct; no `/v1` in the base URL (script adds it for chat/completions). |
| **Wrong or missing predictions** | Use `--overwrite` to recompute; do not use `--skip-ingestion` unless you have already ingested the same data. |

---

## 7. Checklist (quick copy)

- [ ] Ollama installed and running
- [ ] `ollama pull embeddinggemma` and `ollama pull gpt-oss-20b` (or chosen QA model)
- [ ] Embedding dimension confirmed (e.g. 768 or 1024) and set in `.env` as `EMBEDDING__DIMENSIONS`
- [ ] `evaluation/` folder present; `evaluation/locomo` cloned; `evaluation/scripts/eval_locomo.py` and `evaluation/locomo/data/locomo10.json` exist
- [ ] Python deps: `requests`, `tqdm`; for scoring: `bert-score`, `nltk`, `regex`, `numpy` (+ `nltk.download('punkt')`); optional: `python-dateutil` for session timestamps
- [ ] Project root `.env` set: DB, `AUTH__API_KEY`, **`AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=600`** for full evaluation, `EMBEDDING__*` (including `EMBEDDING__DIMENSIONS`), `LLM__*`; use `host.docker.internal` if API in Docker and Ollama on host
- [ ] `docker compose -f docker/docker-compose.yml down -v`
- [ ] `docker compose -f docker/docker-compose.yml up -d --build postgres neo4j redis api`
- [ ] `curl -s http://localhost:8000/api/v1/health` OK
- [ ] `PYTHONPATH=evaluation/locomo` (or `evaluation\locomo` on Windows)
- [ ] `python evaluation/scripts/eval_locomo.py --data-file evaluation/locomo/data/locomo10.json --out-dir evaluation/outputs`
- [ ] Check `evaluation/outputs/locomo10_qa_cml.json` and `evaluation/outputs/locomo10_qa_cml_stats.json`

---

## 8. Run all phases (one script)

The script **`evaluation/scripts/run_full_eval.py`** runs all phases in order and prints the current step and progress. It does not start a separate monitor; it runs the four steps sequentially. To run without attaching to the terminal (e.g. so it continues after you close the shell), use the background commands below.

**Phases:** (1) Tear down containers and volumes (`docker compose down -v`), (2) Build and start postgres, neo4j, redis, api, (3) Wait for CML API health, (4) Run LoCoMo evaluation (ingestion, QA, scoring) via `eval_locomo.py` with the correct `PYTHONPATH`.

### 8.1. Command to run (foreground)

From the **project root**:

```bash
python evaluation/scripts/run_full_eval.py
```

Progress and the current step are printed to stdout. The script exits when all phases complete or when a step fails.

### 8.2. Command to run without monitoring (background)

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

## 9. Comparing to other systems

CML’s LoCoMo run produces **overall accuracy** and **recall** (and per-category stats) in `evaluation/outputs/locomo10_qa_cml_stats.json`. To compare with the **LoCoMo paper baselines**, run their scripts for the same data and then compare metrics.

### 9.1. What to compare

- **Base / long-context LLMs** (no RAG): GPT-4, GPT-3.5 with different context lengths, Claude, Gemini, or HuggingFace models. They use truncated conversation as context.
- **RAG baselines**: GPT-3.5-turbo with retrieval over (a) **dialogs**, (b) **observations**, or (c) **session summaries** (LoCoMo uses Dragon + embeddings; we use CML as the retriever).

Same metrics: **overall accuracy** (F1), **overall recall** (when evidence is available), and per-category accuracy/recall.

### 9.2. Running LoCoMo baselines

From the **LoCoMo repo** (not project root):

```bash
cd evaluation/locomo
```

1. **Set environment:** Edit `scripts/env.sh`: set `OUT_DIR`, `DATA_FILE_PATH` (e.g. `./data/locomo10.json`), and the API keys you need (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`). Source it: `source scripts/env.sh`.

2. **Optional (for RAG):** Generate observations and session summaries (needed for RAG baselines):
   ```bash
   bash scripts/generate_observations.sh
   bash scripts/generate_session_summaries.sh
   ```
   This writes under `$EMB_DIR` / `$OUT_DIR`. RAG scripts also need embeddings (Dragon/Contriever); see LoCoMo README.

3. **Run one or more baselines** (examples; require the corresponding API keys):
   - OpenAI (base or long-context): `bash scripts/evaluate_gpts.sh`
   - RAG (dialog / observation / summary): `bash scripts/evaluate_rag_gpts.sh`
   - HuggingFace: `bash scripts/evaluate_hf_llm.sh`
   - Claude: `bash scripts/evaluate_claude.sh`
   - Gemini: `bash scripts/evaluate_gemini.sh`

Outputs go to **`evaluation/locomo/outputs/`**: `locomo10_qa.json` (predictions) and **`locomo10_qa_stats.json`** (aggregate stats). Each run **appends** a new model entry to the same stats file, so one stats file can hold many baselines.

### 9.3. Comparison table (same metrics)

CML results are in **`evaluation/outputs/locomo10_qa_cml_stats.json`** (one model key, e.g. `gpt-oss-20b_cml_top_25`). LoCoMo baseline results are in **`evaluation/locomo/outputs/locomo10_qa_stats.json`** (multiple model keys).

From the **project root**, run the comparison script on one or more stats files; it prints overall accuracy and recall for every model:

```bash
python evaluation/scripts/compare_results.py evaluation/outputs/locomo10_qa_cml_stats.json evaluation/locomo/outputs/locomo10_qa_stats.json
```

You can pass only the CML stats file, or only the LoCoMo one, or both. Output is a text table: one row per model, columns e.g. **Model**, **Overall accuracy**, **Overall recall**.

Interpretation: same **annotation file** (`locomo10.json`) and same **metrics** (F1 for accuracy, evidence-based recall). Compare CML (memory-backed RAG) vs. base LLMs (truncated context) vs. LoCoMo RAG (dialog/observation/summary + Dragon).
