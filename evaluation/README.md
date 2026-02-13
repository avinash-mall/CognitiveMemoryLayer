# LoCoMo Evaluation with CML

This folder contains the [LoCoMo](https://github.com/snap-research/locomo) benchmark setup and a CML-backed evaluation script that uses the Cognitive Memory Layer as the retriever and local Ollama for the QA LLM.

## Layout

- **`locomo/`** – Cloned [snap-research/locomo](https://github.com/snap-research/locomo) repo (data, `task_eval`, scripts).
- **`scripts/eval_locomo.py`** – Driver: ingest conversations into CML, run QA via CML read + Ollama, score with LoCoMo's `eval_question_answering` and `analyze_aggr_acc`.
- **`outputs/`** – Created at run time; holds `locomo10_qa_cml.json`, `locomo10_qa_cml_stats.json`, `locomo10_gating_stats.json` (default; use `--no-eval-mode` to skip), and optionally `locomo10_qa_cml_timing.json` (with `--log-timing`).

## Prerequisites

1. **CML API** running (e.g. Docker with embedding and LLM set for Ollama; see main project README).
2. **Ollama** on the host: **embedding model** `embeddinggemma` (for CML ingestion/retrieval), **QA model** `gpt-oss-20b` (for the eval script). Set **`EMBEDDING__DIMENSIONS`** in project root `.env` to match your embedding model (e.g. 768 for EmbeddingGemma); if you change it, drop DBs and re-run migrations.
3. **Python deps**: `requests`, `tqdm`; for LoCoMo scoring, install from `locomo/requirements.txt` or at least: `bert-score`, `nltk`, `regex`, `numpy` (and run `python -m nltk.downloader punkt` if needed).

For the full step-by-step runbook (env, API keys, RAG prep, troubleshooting), see [ProjectPlan/LocomoEval/RunEvaluation.md](../ProjectPlan/LocomoEval/RunEvaluation.md).

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
   - `--no-eval-mode` – disable X-Eval-Mode (eval mode is on by default); when on, API returns stored/skipped and reason; script writes **`locomo10_gating_stats.json`** (stored/skipped counts, skip reason counts).
   - `--log-timing` – record per-question CML read and Ollama latency and token usage; script writes **`locomo10_qa_cml_timing.json`** (per_question and aggregate mean/p95 latency, total tokens).
   - `--cml-url`, `--cml-api-key`, `--ollama-url`, `--ollama-model` – override env.

3. Outputs:
   - `evaluation/outputs/locomo10_qa_cml.json` – per-sample QA with predictions and F1/recall.
   - `evaluation/outputs/locomo10_qa_cml_stats.json` – aggregate stats by category.
   - `evaluation/outputs/locomo10_gating_stats.json` – by default (unless `--no-eval-mode`): write-gate outcomes (total_writes, stored_count, skipped_count, skip_reason_counts).
   - `evaluation/outputs/locomo10_qa_cml_timing.json` – when `--log-timing` is used: per-question and aggregate latency (CML read, Ollama) and token usage.

## Comparing to other systems

To compare CML with LoCoMo baselines (base/long-context LLMs, RAG over dialog/observation/summary):

1. **Run LoCoMo baselines** from `evaluation/locomo`: set `scripts/env.sh`, then run e.g. `scripts/evaluate_gpts.sh`, `scripts/evaluate_rag_gpts.sh`, etc. Their stats go to `evaluation/locomo/outputs/locomo10_qa_stats.json`.
2. **Print a comparison table** (overall accuracy and recall for every model) from project root:
   ```bash
   python evaluation/scripts/compare_results.py evaluation/outputs/locomo10_qa_cml_stats.json evaluation/locomo/outputs/locomo10_qa_stats.json
   ```

## Reference

- [LoCoMo paper / site](https://snap-research.github.io/locomo/)
- [LoCoMo repo](https://github.com/snap-research/locomo)
- [Evaluation runbook (full steps)](../ProjectPlan/LocomoEval/RunEvaluation.md)
