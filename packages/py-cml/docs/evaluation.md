# Evaluation Module (`cml.eval`)

`cml.eval` packages LoCoMo/Locomo-Plus evaluation workflows into reusable Python APIs and a CLI.

## Install

```bash
pip install "cognitive-memory-layer[eval]"
```

The eval extras install `requests`, `tqdm`, `python-dotenv`, and `openai`. If you run `cml-eval` without the extras, you will see a clear message telling you which dependency is missing.

## CLI

All subcommands are available via the `cml-eval` entry point.

### `run-full` — Full pipeline (Docker + ingest + QA + judge + report)

```bash
cml-eval run-full --repo-root .
```

| Flag | Default | Description |
|------|---------|-------------|
| `--repo-root` | auto-detected | Path to the CML repository root |
| `--skip-docker` | off | Skip Docker tear-down/rebuild (API must already be running) |
| `--limit-samples N` | all | Run only the first N samples |
| `--ingestion-workers N` | 5 | Concurrent workers for Phase A ingestion |
| `--resume` | off | Resume from last failure (implies `--skip-docker`) |
| `--score-only` | off | Run only Phase C (judge) + performance table on existing predictions |

State is saved to `evaluation/outputs/run_full_eval_state.json` for resume support.

### `run-locomo` — Locomo-Plus CML-backed evaluation

```bash
cml-eval run-locomo \
  --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json \
  --out-dir evaluation/outputs
```

| Flag | Default | Description |
|------|---------|-------------|
| `--unified-file` | auto-detected | Path to `unified_input_samples_v2.json` |
| `--out-dir` | `evaluation/outputs` | Output directory |
| `--cml-url` | `CML_BASE_URL` env or `http://localhost:8000` | CML server URL |
| `--cml-api-key` | `CML_API_KEY` env or `test-key` | API key for CML |
| `--max-results` | 25 | Max memory results per read query |
| `--limit-samples N` | all | Run only the first N samples |
| `--skip-ingestion` | off | Skip Phase A (assume data already ingested) |
| `--skip-consolidation` | off | Skip consolidation between Phase A and B |
| `--score-only` | off | Run only Phase C (judge) on existing predictions |
| `--judge-model` | `LLM_EVAL__MODEL` or `gpt-4o-mini` | LLM model for judge scoring |
| `--verbose` | off | Emit per-sample retrieval diagnostics |
| `--ingestion-workers N` | 10 | Concurrent workers for Phase A |

### `validate` — Validate evaluation output artifacts

```bash
cml-eval validate --outputs-dir evaluation/outputs
```

Checks predictions, judged results, and summary JSON for structural correctness, cross-file consistency (lengths, scores, question matching), and valid judge labels/scores.

### `report` — Generate performance table

```bash
cml-eval report \
  --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json \
  --method "CML+gpt-4o-mini"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--summary` | auto-detected | Path to judge summary JSON |
| `--method` | `CML` | Method name shown in the table |
| `--no-title` | off | Omit the table title/description |

### `compare` — Compare scores with paper baselines

```bash
cml-eval compare \
  --summary evaluation/outputs/locomo_plus_qa_cml_judge_summary.json \
  --method "CML+gpt-4o-mini"
```

Prints a comparison table against the LoCoMo-Plus paper baselines (Table 1, arXiv:2602.10715) including Qwen, GPT-4o, Gemini, RAG, Mem0, SeCom, and A-Mem.

## Python API

All public functions and types are importable from `cml.eval`:

```python
from pathlib import Path
from cml.eval import (
    LocomoEvalConfig,
    FullEvalConfig,
    run_locomo_plus,
    run_full_eval,
    validate_outputs,
    generate_locomo_report,
    compare_locomo_scores,
)
```

### `run_locomo_plus(config: LocomoEvalConfig) -> list[dict]`

Run the full Locomo-Plus evaluation pipeline (ingest, QA, judge) and return prediction records.

```python
records = run_locomo_plus(
    LocomoEvalConfig(
        unified_file=Path("evaluation/locomo_plus/data/unified_input_samples_v2.json"),
        out_dir=Path("evaluation/outputs"),
        cml_url="http://localhost:8000",
        cml_api_key="test-key",
        limit_samples=10,         # quick test with 10 samples
        ingestion_workers=5,
    )
)
```

### `run_full_eval(config: FullEvalConfig) -> int`

Run the full pipeline including optional Docker management, health checks, ingestion, QA, judging, validation, and report generation. Returns 0 on success, non-zero on failure.

```python
rc = run_full_eval(FullEvalConfig(repo_root=Path("."), skip_docker=True))
```

### `validate_outputs(outputs_dir: Path) -> list[str]`

Validate evaluation output artifacts. Returns a list of error strings (empty on success).

```python
errors = validate_outputs(Path("evaluation/outputs"))
if errors:
    for e in errors:
        print(f"  - {e}")
```

### `generate_locomo_report(summary_path: Path, method: str, no_title: bool = False) -> str`

Generate a formatted performance table string from a judge summary file.

### `compare_locomo_scores(summary_path: Path, method: str) -> str`

Generate a comparison table against paper baselines.

## Standalone pip install notes

When installed via `pip install "cognitive-memory-layer[eval]"` outside the repository:

- **Sample loading** works anywhere — the unified JSON file loader is built into the package.
- **LLM-as-judge** (Phase C) requires the `evaluation/locomo_plus/` directory from the CML repository, as it uses prompt templates and judge logic from `task_eval/`. Point `--repo-root` to a CML checkout, or run from within the repo.
- The `validate`, `report`, and `compare` commands work fully standalone with any output JSON files.

## Legacy script compatibility

Legacy commands still work and are now thin wrappers:

- `python evaluation/scripts/eval_locomo_plus.py ...`
- `python evaluation/scripts/run_full_eval.py ...`
- `python evaluation/scripts/validate_outputs.py ...`
- `python evaluation/scripts/generate_locomo_report.py ...`
- `python evaluation/scripts/compare_locomo_scores.py ...`
