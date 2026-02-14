# Evaluation with CML

This folder contains the [Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) benchmark setup and CML-backed evaluation scripts. Locomo-Plus unifies LoCoMo (five factual categories) with a sixth **Cognitive** category for long-context memory evaluation.

## Layout

| Path | Description |
|------|-------------|
| `locomo_plus/` | Data pipeline (`locomo10.json`, `locomo_plus.json`, `unified_input_samples_v2.json`), task_eval, scripts. See [locomo_plus/README.md](locomo_plus/README.md). |
| `scripts/eval_locomo_plus.py` | CML-backed driver: ingest unified samples into CML, run QA via CML read + Ollama, score with LLM-as-judge (correct=1, partial=0.5, wrong=0). |
| `scripts/generate_locomo_report.py` | Build performance table (LoCoMo factual + LoCoMo-Plus Cognitive + Gap). |
| `scripts/run_full_eval.py` | Full pipeline: Docker down/up, API wait, eval, report table. |
| `outputs/` | Created at run time; holds predictions, judged records, and judge summary. |

## Prerequisites

1. **CML API** running (e.g. via Docker; see main project README).
2. **Ollama** on the host:
   - **Embedding model** `embeddinggemma` (for CML ingestion/retrieval)
   - **QA model** `gpt-oss:20b` or `gpt-oss-20b` (set via `OLLAMA_QA_MODEL` or `--ollama-model`)
   - Set `EMBEDDING__DIMENSIONS` in project root `.env` to match your embedding model (e.g. 768 for EmbeddingGemma). If you change it, drop DBs and re-run migrations.
3. **Python deps**: `requests`, `tqdm`; for LLM-as-judge, set `OPENAI_API_KEY` (or point at an Ollama-compatible `OPENAI_BASE_URL`).

For a step-by-step runbook, see [ProjectPlan/LocomoEval/RunEvaluation.md](../ProjectPlan/LocomoEval/RunEvaluation.md).

## Configuration

### Environment variables (eval script)

| Variable | Default | Description |
|----------|---------|-------------|
| `CML_BASE_URL` | `http://localhost:8000` | CML API base URL |
| `CML_API_KEY` | `test-key` | API key (must match `AUTH__API_KEY`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama URL (no `/v1`) |
| `OLLAMA_QA_MODEL` | `gpt-oss:20b` | Ollama model for QA |
| `OPENAI_API_KEY` | — | Required for LLM-as-judge |

## Run full evaluation

From the **project root**:

```bash
python evaluation/scripts/run_full_eval.py
```

This runs: (1) Docker down -v, (2) Docker up (postgres, neo4j, redis, api), (3) API health wait, (4) eval_locomo_plus (ingest, QA, judge), (5) performance table.

### Options

| Option | Description |
|--------|-------------|
| `--skip-docker` | Skip steps 1–3 (use when API is already running) |
| `--limit-samples N` | Run only first N samples (for quick testing) |

Examples:

```bash
# API already running, quick 50-sample test
python evaluation/scripts/run_full_eval.py --skip-docker --limit-samples 50

# Full run without Docker steps
python evaluation/scripts/run_full_eval.py --skip-docker
```

### Output table

The pipeline prints a table matching the paper format:

| Method | single-hop | multi-hop | temporal | commonsense | adversarial | average | LoCoMo-Plus | Gap |
|--------|------------|-----------|----------|-------------|-------------|---------|-------------|-----|
| CML+gpt-oss:20b | ... | ... | ... | ... | ... | ... | ... | ... |

**Gap** = LoCoMo average − LoCoMo-Plus (performance drop from factual to cognitive memory).

## Run evaluation manually

When you need finer control (e.g. skip ingestion, score-only, verbose):

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
| `--skip-ingestion` | Skip Phase A (reuse existing CML state) |
| `--score-only` | Run only Phase C (judge) on existing predictions |
| `--max-results N` | CML read top-k (default 25) |
| `--verbose` | Per-sample retrieval diagnostics |
| `--cml-url`, `--cml-api-key` | Override CML connection |
| `--ollama-url`, `--ollama-model` | Override Ollama connection |
| `--judge-model` | Model for LLM-as-judge (default gpt-4o-mini) |

### Outputs

| File | Description |
|------|-------------|
| `locomo_plus_qa_cml_predictions.json` | Per-sample predictions (before judge) |
| `locomo_plus_qa_cml_judged.json` | Judged records (judge_label, judge_reason, judge_score) |
| `locomo_plus_qa_cml_judge_summary.json` | Aggregate by category (for report table) |

### Generate report table

```bash
python evaluation/scripts/generate_locomo_report.py --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json --method "CML+gpt-oss:20b"
```

Use `--method` to match your QA model (e.g. `CML+gpt-oss:20b`).

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

## References

- [LoCoMo paper / site](https://snap-research.github.io/locomo/)
- [LoCoMo repo](https://github.com/snap-research/locomo)
- [Locomo-Plus repo](https://github.com/xjtuleeyf/Locomo-Plus)
- [Runbook (full steps)](../ProjectPlan/LocomoEval/RunEvaluation.md)
