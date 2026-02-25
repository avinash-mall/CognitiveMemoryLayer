# Dataset Preparation for 15-Way Memory-Type Classifier

This directory contains the data preparation pipeline for the Cognitive Memory Layer (CML) 15-way memory-type classifier.

- Raw datasets: `packages/models/datasets/`
- Prepared outputs: `packages/models/prepared/`
- Entry script: `packages/models/scripts/prepare.py`
- Training script: `packages/models/scripts/train.py`

## 1. Context

CML routes incoming text into 15 memory categories. The prepared dataset is used to train a lightweight classifier for offline and low-latency routing when LLM-heavy paths are disabled.

## 2. Memory Types

| Type | Description |
|------|-------------|
| `episodic_event` | Past events and narratives |
| `semantic_fact` | Stable factual statements |
| `preference` | Likes, dislikes, personal preferences |
| `constraint` | Rules, policies, obligations |
| `procedure` | How-to or step-by-step instructions |
| `hypothesis` | Uncertain or speculative statements |
| `task_state` | Current progress / status |
| `conversation` | Multi-turn dialogue context |
| `message` | Single utterance or turn |
| `tool_result` | Logs, outputs, errors, traces |
| `reasoning_step` | Rationale and reasoning statements |
| `scratch` | Temporary drafts and working notes |
| `knowledge` | General world/domain knowledge |
| `observation` | Direct observations of state/screen |
| `plan` | Goals and forward-looking steps |

## 3. What `prepare.py` Does

The script runs the full pipeline in one command:

1. Audit dataset presence and file metadata
2. Extract real examples using schema-aware handlers per dataset
3. Generate synthetic examples using an LLM (default) to backfill label deficits
4. Enforce per-label target counts
5. Write stratified train/test/eval splits and reports

Key behavior:

- Real samples are used first.
- Synthetic samples are generated only for missing counts.
- Synthetic generation is diverse and seeded from random real conversation examples.
- Deduplication is applied to avoid repeated rows and conflicting cross-label duplicates.

## 4. Datasets

All dataset files are expected directly under `packages/models/datasets` (flat layout):

- `cooperbench_train.parquet`
- `locomo_plus_locomo10.json`
- `locomo_plus_locomo_plus.json`
- `wikipedia_cleaned_20231101.en`
- `coqa_train.parquet`
- `coqa_validation.parquet`
- `personachat_train_both_revised.txt`
- `wikihow_sep.csv`
- `wikihow_cleaned.csv`
- `reddit_writing_prompts_train_00000.parquet`
- `reddit_writing_prompts_train_00001.parquet`
- `chatgpt4o_writing_prompts_sharegpt.jsonl`
- `structured_wikipedia_enwiki_namespace_0.zip`

Reference links:

- CooperBench: <https://huggingface.co/datasets/CodeConflict/cooperbench-dataset>
- LoCoMo-Plus: <https://github.com/xjtuleeyf/Locomo-Plus>
- LoCoMo: <https://github.com/snap-research/locomo>
- Wikipedia: <https://huggingface.co/datasets/wikimedia/wikipedia>
- CoQA: <https://huggingface.co/datasets/stanfordnlp/coqa>
- CoQA docs: <https://stanfordnlp.github.io/coqa/>
- PersonaChat: <https://huggingface.co/datasets/awsaf49/persona-chat>
- wikiHow cleaned: <https://huggingface.co/datasets/gursi26/wikihow-cleaned>
- WikiHow dataset: <https://github.com/mahnazkoupaee/WikiHow-Dataset>
- WritingPrompts: <https://huggingface.co/datasets/euclaise/WritingPrompts>
- ChatGPT-4o Writing Prompts: <https://huggingface.co/datasets/Gryphe/ChatGPT-4o-Writing-Prompts>
- Structured Wikipedia: <https://huggingface.co/datasets/wikimedia/structured-wikipedia>

## 5. Usage

From `packages/models`:

```bash
cd packages/models

# Full pipeline (real extraction + synthetic backfill + split + reports)
python -m scripts.prepare

# Skip synthetic backfill
python -m scripts.prepare --no-synthetic

# LLM synthetic mode with explicit provider settings
python -m scripts.prepare --llm-model gpt-oss:20b --llm-base-url http://localhost:11434/v1

# Only print stats from existing prepared outputs
python -m scripts.prepare --stats-only

# Set custom targets
python -m scripts.prepare --target-per-label 10000 --max-real-per-label 10000 --seed 42

# Train classifier from prepared train/test/eval splits
python -m scripts.train

# Train with custom hyperparameters/output path
python -m scripts.train --max-features 200000 --max-iter 20 --output-dir artifacts/classifier
```

From repo root:

```bash
python -c "import sys; sys.path.insert(0, 'packages/models'); from scripts.prepare import main; raise SystemExit(main())"
```

For training from repo root:

```bash
python -c "import sys; sys.path.insert(0, 'packages/models'); from scripts.train import main; raise SystemExit(main())"
```

## 6. Outputs

| Path | Description |
|------|-------------|
| `packages/models/prepared/dataset_audit.json` | Dataset audit report (exists/missing and file sizes) |
| `packages/models/prepared/validation_report.json` | Real/synthetic/final counts, source contribution summary |
| `packages/models/prepared/validated_sample.csv` | Sample rows for manual inspection |
| `packages/models/prepared/train.parquet` | Train split (`text`, `label`, `source`) |
| `packages/models/prepared/test.parquet` | Test split (`text`, `label`, `source`) |
| `packages/models/prepared/eval.parquet` | Eval split (`text`, `label`, `source`) |
| `packages/models/prepared/synthetic.parquet` | Synthetic rows generated in the current run |
| `packages/models/prepared/label_counts.json` | Final counts per label |
| `packages/models/prepared/preparation_report.txt` | Text summary of counts and split sizes |

Training outputs:

| Path | Description |
|------|-------------|
| `packages/models/artifacts/classifier/model.joblib` | Trained scikit-learn pipeline + metadata + label maps |
| `packages/models/artifacts/classifier/label_map.json` | Label-to-id and id-to-label map |
| `packages/models/artifacts/classifier/metrics_test.json` | Aggregate test metrics and confusion matrix |
| `packages/models/artifacts/classifier/metrics_eval.json` | Aggregate eval metrics and confusion matrix |
| `packages/models/artifacts/classifier/classification_report_test.json` | Per-label precision/recall/F1 (test) |
| `packages/models/artifacts/classifier/classification_report_eval.json` | Per-label precision/recall/F1 (eval) |
| `packages/models/artifacts/classifier/training_metadata.json` | Training run config and dataset metadata |

## 7. Main Config (in code)

Config is defined directly in `scripts/prepare.py`:

- `DEFAULT_TARGET_PER_LABEL` (default: `10000`)
- `DEFAULT_MAX_REAL_PER_LABEL` (default: `20000`)
- `SPLIT_RATIOS` (default: `0.8 / 0.1 / 0.1`)
- `DEFAULT_SEED` (default: `42`)
- `DATASET_FILES` (expected filenames under `datasets/`)

CLI overrides:

- `--target-per-label`
- `--max-real-per-label`
- `--seed`
- `--no-synthetic`
- `--stats-only`
- `--llm-model`
- `--llm-base-url`
- `--llm-api-key`
- `--llm-temperature`
- `--llm-timeout-seconds`
- `--llm-batch-size`

Training CLI overrides (`scripts/train.py`):

- `--train-path`
- `--test-path`
- `--eval-path`
- `--output-dir`
- `--max-features`
- `--min-df`
- `--max-iter`
- `--alpha`
- `--seed`
- `--predict-batch-size`

## 8. Notes

- Synthetic generation uses LLM mode; provide either OpenAI credentials or an OpenAI-compatible local endpoint (for example Ollama).
- The default sampling temperature is intentionally high to increase diversity.
- Both `prepare.py` and `train.py` show progress bars for long-running stages.
- If `--no-synthetic` is used and some labels are below target, the script prints deficits.
- The script is intentionally self-contained (single file) and does not require separate config JSON files.
