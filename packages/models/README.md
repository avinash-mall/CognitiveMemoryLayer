# Unified Custom Model Pipeline

This folder now uses:

- One preparation script: `packages/models/scripts/prepare.py`
- One training script: `packages/models/scripts/train.py`
- One shared config: `packages/models/model_pipeline.toml`

All trained models are written to:

- `packages/models/trained_models`

## Models Created and What They Solve

The pipeline trains 3 consolidated models to replace most LLM-gated paths.

| Model name | Family | Main tasks |
|---|---|---|
| `router_model.joblib` | `router` | `memory_type`, `query_intent`, `constraint_dimension`, `context_tag`, `salience_bin`, `importance_bin`, `confidence_bin`, `decay_profile` |
| `extractor_model.joblib` | `extractor` | `constraint_type`, `constraint_scope`, `fact_type`, `pii_presence` |
| `pair_model.joblib` | `pair` | `conflict_detection`, `constraint_rerank` |

### Issue-to-model mapping (from `REPORT.md`)

| Issue | Best model(s) | Why |
|---|---|---|
| `C-01` Semantic disconnect for constraints | `extractor_model` + `pair_model` + `router_model` | Scope typing + query context + pairwise relevance improves semantic linkage. |
| `C-02` Constraint dilution | `pair_model` + `extractor_model` | Relevance scoring + scope-aware filtering improves top-k constraints. |
| `C-03` Missing fast constraint dimensions | `router_model` | Predicts `constraint_dimension` without LLM classifier. |
| `L-04` Empty constraint scope | `extractor_model` | Predicts `constraint_scope` directly. |
| `L-06` Write gate misses important constraints | `router_model` | Learned salience/importance bins reduce heuristic-only drops. |
| `C-04`, `C-05`, `C-06` | Partial by all 3 | Models improve signals; runtime policy/supersession/consolidation logic still required. |

### Issues better solved by NER (spaCy-style) than classifier-only models

Classifier models help with labels, but these are better handled with explicit NER/rules:

- Fine-grained PII span extraction/redaction (`USE_LLM_PII_REDACTION` replacement quality)
- Entity-level constraint scope extraction (multi-entity scope, aliases, cross-sentence coreference)
- Structured relation extraction for graph updates

Recommended hybrid: model classification + NER post-processing.

## Datasets and Which Model They Train

Configured in `packages/models/model_pipeline.toml` (`[[datasets]]` entries).

| Dataset name (config) | Source | Trains |
|---|---|---|
| `banking77` | `banking77` | `router_model` |
| `trec` | `ag_news` | `router_model` |
| `massive` | `dbpedia_14` | `router_model` |
| `moral_stories` | `go_emotions` (`simplified`) | `extractor_model` |
| `pii_masking` | `ai4privacy/pii-masking-200k` | `extractor_model` |
| `snli` | `stanfordnlp/snli` | `pair_model` |
| `multi_nli` | `nyu-mll/multi_nli` | `pair_model` |
| `ms_marco` | `microsoft/ms_marco` (`v1.1`) | `pair_model` |
| `quora_duplicates` | `sentence-transformers/quora-duplicates` (`pair-class`) | `pair_model` |

Local bootstrap dataset:

- `packages/models/prepared_data/{train,test,eval}.parquet` (if present) is used as weak supervision.

If required internet datasets are missing locally, `prepare.py` auto-downloads them using links/IDs in config.
Downloaded artifacts are cached under `packages/models/datasets` (existing flat datasets folder).

## LLM Feature Replacement Map

| Previous LLM-gated feature | Replacement |
|---|---|
| `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` | `extractor_model` (`constraint_type`, `constraint_scope`) |
| `FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR` | `extractor_model` |
| `FEATURES__USE_LLM_WRITE_TIME_FACTS` | `extractor_model` (`fact_type`) |
| `FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY` | `router_model` (`query_intent`, `constraint_dimension`) |
| `FEATURES__USE_LLM_SALIENCE_REFINEMENT` | `router_model` (`salience_bin`) |
| `FEATURES__USE_LLM_PII_REDACTION` | `extractor_model` (`pii_presence`) + recommended NER for span-level masking |
| `FEATURES__USE_LLM_WRITE_GATE_IMPORTANCE` | `router_model` (`importance_bin`) |
| `FEATURES__USE_LLM_MEMORY_TYPE` | `router_model` (`memory_type`) |
| `FEATURES__USE_LLM_CONFIDENCE` | `router_model` (`confidence_bin`) |
| `FEATURES__USE_LLM_CONTEXT_TAGS` | `router_model` (`context_tag`) |
| `FEATURES__USE_LLM_DECAY_RATE` | `router_model` (`decay_profile`) |
| `FEATURES__USE_LLM_CONFLICT_DETECTION_ONLY` | `pair_model` (`conflict_detection`) |
| `FEATURES__USE_LLM_CONSTRAINT_RERANKER` | `pair_model` (`constraint_rerank`) |

### What is left (not fully replaced by classifiers)

- Precise PII span extraction/redaction quality (best with NER/rule layer)
- Consolidation-time semantic preservation and supersession policy logic
- Retrieval/store algorithmic efficiency fixes (`E-*` issues in report)

## Config File

Use:

- `packages/models/model_pipeline.toml`

It contains:

- Paths (`prepared_dir`, `bootstrap_prepared_dir`, `trained_models_dir`, dataset storage path)
- Prepare settings (seed, split ratios, sample caps, auto-download behavior)
- Train settings (`max_iter` as epoch-like SGD iterations, `max_features`, `min_df`, etc.)
- Optional/reserved LLM settings (kept in config for completeness, currently disabled by default)
- Dataset definitions (`name`, `link`, `dataset_id`, `split`, `max_rows`, `required`)

## Usage Guide

From repo root:

```bash
# Optional deps
pip install datasets pyarrow pandas scikit-learn joblib tqdm

# 1) Prepare all model-family datasets (auto-download missing required datasets)
python -m packages.models.scripts.prepare

# 2) Train/test/eval all model families
python -m packages.models.scripts.train
```

From `packages/models`:

```bash
cd packages/models
python -m scripts.prepare
python -m scripts.train
```

Common overrides:

```bash
# Use a custom config file
python -m scripts.prepare --config model_pipeline.toml
python -m scripts.train --config model_pipeline.toml

# Preparation overrides
python -m scripts.prepare --max-rows-per-source 20000 --max-per-task-label 15000 --seed 123

# Training overrides
python -m scripts.train --max-iter 35 --max-features 300000 --predict-batch-size 4096
```

Outputs:

- Prepared datasets: `packages/models/prepared_data/modelpack/*`
- Trained models and metrics: `packages/models/trained_models/*`
