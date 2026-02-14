# Evaluation with CML

This folder contains [Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) benchmark setup and CML-backed evaluation scripts. Locomo-Plus unifies LoCoMo (five factual categories) with a sixth **Cognitive** category for long-context memory evaluation.

## Layout

- **`locomo_plus/`** – Locomo-Plus: data pipeline (`locomo10.json`, `locomo_plus.json`, `unified_input_samples_v2.json`), task_eval, scripts. See [locomo_plus/README.md](locomo_plus/README.md).
- **`scripts/eval_locomo_plus.py`** – Driver: ingest unified samples into CML, run QA via CML read + Ollama, score with LLM-as-judge (correct=1, partial=0.5, wrong=0).
- **`scripts/generate_locomo_report.py`** – Build performance table (LoCoMo factual categories + Locomo-Plus Cognitive + Gap).
- **`scripts/run_full_eval.py`** – Full pipeline: Docker up, CML API wait, eval_locomo_plus, report table.
- **`outputs/`** – Created at run time; holds Locomo-Plus output JSONs and judge summary.

## Prerequisites

1. **CML API** running (e.g. Docker with embedding and LLM set for Ollama; see main project README).
2. **Ollama** on the host: **embedding model** `embeddinggemma` (for CML ingestion/retrieval), **QA model** `gpt-oss-20b` (for the eval script). Set **`EMBEDDING__DIMENSIONS`** in project root `.env` to match your embedding model (e.g. 768 for EmbeddingGemma); if you change it, drop DBs and re-run migrations.
3. **Python deps**: `requests`, `tqdm`; for LLM-as-judge, set `OPENAI_API_KEY` (or point at an Ollama-compatible `OPENAI_BASE_URL`).

For the full step-by-step runbook (env, API keys, RAG prep, troubleshooting), see [ProjectPlan/LocomoEval/RunEvaluation.md](../ProjectPlan/LocomoEval/RunEvaluation.md).

## Configuration (eval script)

Environment variables (or CLI flags):

| Variable         | Default                 | Description                    |
|------------------|-------------------------|--------------------------------|
| `CML_BASE_URL`   | `http://localhost:8000` | CML API base URL               |
| `CML_API_KEY`    | `test-key`              | API key (match `AUTH__API_KEY`)|
| `OLLAMA_BASE_URL`| `http://localhost:11434`| Ollama URL (no `/v1`)          |
| `OLLAMA_QA_MODEL` | `gpt-oss-20b`           | Ollama model for QA            |
| `OPENAI_API_KEY` | —                       | Required for LLM-as-judge      |

## Run full evaluation

From the **project root**, run the full pipeline (Docker down/up, API wait, eval, report table):

```bash
python evaluation/scripts/run_full_eval.py
```

## Run evaluation manually

1. Set `PYTHONPATH` and run the Locomo-Plus eval:

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

2. Optional args:
   - `--limit-samples N` – run only the first N samples (e.g. `5` for a quick test).
   - `--skip-ingestion` – skip Phase A (reuse existing CML state).
   - `--score-only` – run only the LLM-as-judge on existing predictions.
   - `--max-results 25` – CML read top-k (default 25).
   - `--ollama-model`, `--judge-model`, `--cml-url`, `--cml-api-key`, `--ollama-url` – override env.

3. Outputs:
   - `evaluation/outputs/locomo_plus_qa_cml_predictions.json` – per-sample predictions.
   - `evaluation/outputs/locomo_plus_qa_cml_judged.json` – judged records.
   - `evaluation/outputs/locomo_plus_qa_cml_judge_summary.json` – aggregate by category.

4. Generate report table:
   ```bash
   python evaluation/scripts/generate_locomo_report.py --method "CML+gpt-oss:20b"
   ```

## Build unified input (optional)

To rebuild `unified_input_samples_v2.json` from source data:

```bash
cd evaluation/locomo_plus/data && python unified_input.py
```

Requires `locomo10.json` and `locomo_plus.json` in `evaluation/locomo_plus/data/`.

## Run without CML (OpenAI-style API)

```bash
# From project root
$env:PYTHONPATH = "evaluation/locomo_plus"
python evaluation/locomo_plus/scripts/run_evaluate.py --backend call_llm --model gpt-4o-mini
python evaluation/locomo_plus/scripts/run_judge.py --model gpt-4o-mini
```

Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) in `evaluation/locomo_plus/scripts/env.local.sh` or the environment.

---

## Reference

- [LoCoMo paper / site](https://snap-research.github.io/locomo/)
- [LoCoMo repo](https://github.com/snap-research/locomo)
- [Locomo-Plus repo](https://github.com/xjtuleeyf/Locomo-Plus)
- [Evaluation runbook (full steps)](../ProjectPlan/LocomoEval/RunEvaluation.md)
