# LoCoMo Evaluation with CML

This folder contains the [LoCoMo](https://github.com/snap-research/locomo) benchmark setup and a CML-backed evaluation script that uses the Cognitive Memory Layer as the retriever and local Ollama for the QA LLM.

## Layout

- **`locomo/`** – Cloned [snap-research/locomo](https://github.com/snap-research/locomo) repo (data, `task_eval`, scripts).
- **`scripts/eval_locomo.py`** – Driver: ingest conversations into CML, run QA via CML read + Ollama, score with LoCoMo’s `eval_question_answering` and `analyze_aggr_acc`.
- **`outputs/`** – Created at run time; holds `locomo10_qa_cml.json` and `locomo10_qa_cml_stats.json`.

## Prerequisites

1. **CML API** running (e.g. Docker with embedding and LLM set for Ollama; see main project README and “Build” below).
2. **Ollama** on the host with:
   - **Embedding model**: `embeddinggemma` (used by CML for ingestion/retrieval).
   - **QA model**: `gpt-oss-20b` (used by the eval script to generate answers).
3. **Embedding dimension**: CML’s DB vector size must match the embedding model output (see “Embedding dimension” below).
4. **Python deps**: `requests`, `tqdm`; for LoCoMo scoring, install from `locomo/requirements.txt` or at least: `bert-score`, `nltk`, `regex`, `numpy` (and run `python -m nltk.downloader punkt` if needed).

## Embedding dimension (EmbeddingGemma)

- EmbeddingGemma in Ollama typically outputs **768** dimensions (confirm with one call: `ollama run embeddinggemma "test"` and check vector length, or use the [Ollama embed API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings)).
- Set in the **project root `.env`** (used by CML API):
  - `EMBEDDING__PROVIDER=ollama`
  - `EMBEDDING__MODEL=embeddinggemma` (or `embeddinggemma:latest`)
  - **`EMBEDDING__DIMENSIONS=768`** (or the actual dimension from your run)
  - `EMBEDDING__BASE_URL=http://localhost:11434/v1`
- The migration creates the pgvector column using `EMBEDDING__DIMENSIONS` at import time. If you change the embedding model or dimension, you must **drop databases and re-run migrations** (see “Build” below).

## Build (drop DBs and recreate Docker)

From the **project root**:

1. Set `.env` (see main `.env.example`), including:
   - Embedding: `EMBEDDING__PROVIDER=ollama`, `EMBEDDING__MODEL=embeddinggemma`, `EMBEDDING__DIMENSIONS=768`, `EMBEDDING__BASE_URL=http://localhost:11434/v1`
   - LLM (for CML internals if needed): `LLM__PROVIDER=ollama`, `LLM__MODEL=gpt-oss-20b`, `LLM__BASE_URL=http://localhost:11434/v1`
   - If the API runs **inside Docker** and Ollama is on the host: use `http://host.docker.internal:11434/v1` for `EMBEDDING__BASE_URL` and `LLM__BASE_URL`.

2. Tear down containers and **remove volumes** (drops DB data):
   ```bash
   docker compose -f docker/docker-compose.yml down -v
   ```

3. Rebuild and start the API (and its DBs):
   ```bash
   docker compose -f docker/docker-compose.yml up -d --build postgres neo4j redis api
   ```

4. Check health:
   ```bash
   curl -s http://localhost:8000/api/v1/health
   ```

## Configuration (eval script)

Environment variables (or CLI flags):

| Variable         | Default                 | Description                    |
|------------------|-------------------------|--------------------------------|
| `CML_BASE_URL`   | `http://localhost:8000` | CML API base URL               |
| `CML_API_KEY`    | `test-key`              | API key (match `AUTH__API_KEY`)|
| `OLLAMA_BASE_URL`| `http://localhost:11434`| Ollama URL (no `/v1`)          |
| `OLLAMA_QA_MODEL` | `gpt-oss-20b`           | Ollama model for QA            |

## Run evaluation

From the **project root**, with CML API and Ollama running:

1. Put the LoCoMo repo on `PYTHONPATH` and run the script:

   **Windows (PowerShell):**
   ```powershell
   $env:PYTHONPATH = "evaluation\locomo"
   python evaluation/scripts/eval_locomo.py --data-file evaluation/locomo/data/locomo10.json --out-dir evaluation/outputs
   ```

   **Unix:**
   ```bash
   export PYTHONPATH=evaluation/locomo
   python evaluation/scripts/eval_locomo.py --data-file evaluation/locomo/data/locomo10.json --out-dir evaluation/outputs
   ```

2. Optional args:
   - `--limit-samples N` – run only the first N samples (e.g. `1` for a quick test).
   - `--skip-ingestion` – skip Phase A (reuse existing CML state).
   - `--overwrite` – overwrite existing predictions in the output file.
   - `--max-results 25` – CML read top-k (default 25).
   - `--cml-url`, `--cml-api-key`, `--ollama-url`, `--ollama-model` – override env.

3. Outputs:
   - `evaluation/outputs/locomo10_qa_cml.json` – per-sample QA with predictions and F1/recall.
   - `evaluation/outputs/locomo10_qa_cml_stats.json` – aggregate stats by category.

## Comparing to other systems

To compare CML with LoCoMo baselines (base/long-context LLMs, RAG over dialog/observation/summary):

1. **Run LoCoMo baselines** from `evaluation/locomo`: set `scripts/env.sh`, then run e.g. `scripts/evaluate_gpts.sh`, `scripts/evaluate_rag_gpts.sh`, etc. Their stats go to `evaluation/locomo/outputs/locomo10_qa_stats.json`.
2. **Print a comparison table** (overall accuracy and recall for every model) from project root:
   ```bash
   python evaluation/scripts/compare_results.py evaluation/outputs/locomo10_qa_cml_stats.json evaluation/locomo/outputs/locomo10_qa_stats.json
   ```

Full steps (env, API keys, RAG prep) are in **`ProjectPlan/LocomoEval/RunEvaluation.md`** §9.

## Reference

- [LoCoMo paper / site](https://snap-research.github.io/locomo/)
- [LoCoMo repo](https://github.com/snap-research/locomo)
- [Project evaluation plan](../ProjectPlan/LocomoEval/EvaluationPlan.md)
