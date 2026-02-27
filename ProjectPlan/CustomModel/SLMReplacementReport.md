# SLM Replacement Report

Date: 2026-02-27

## Goal

Replace LLM-dependent classification/extraction fallback logic with a small custom model pack + NER-based runtime path while preserving optional LLM mode.

## Implemented

1. Unified modelpack runtime loader
- File: `src/utils/modelpack.py`
- Loads:
  - `router_model.joblib`
  - `extractor_model.joblib`
  - `pair_model.joblib`

2. Runtime task wiring
- `query_domain`: classifier/planner/retriever
- `scope_match`: retriever/reranker
- `constraint_stability`: reranker recency weighting
- `supersession`: constraint supersession checks

3. NER runtime fallback replacement
- entity extraction: `src/extraction/entity_extractor.py`
- relation extraction: `src/extraction/relation_extractor.py`
- write-time facts fallback: `src/extraction/write_time_facts.py`
- PII span helper: `src/utils/ner.py`

4. Pipeline consolidation in `packages/models`
- prepare: `packages/models/scripts/prepare.py`
- train: `packages/models/scripts/train.py`
- config: `packages/models/model_pipeline.toml`
- model outputs: `packages/models/trained_models`

## Current runtime decision matrix

| Mode | Write/read behavior |
|---|---|
| `FEATURES__USE_LLM_ENABLED=false` | modelpack + NER/non-LLM paths |
| `FEATURES__USE_LLM_ENABLED=true` | fine-grained `FEATURES__USE_LLM_*` controls LLM paths |

## Remaining scope not replaced by custom classifiers

1. Unified generative write extraction quality (`unified_write_extractor`)
2. Consolidation gist generation
3. Forgetting compression summaries
4. LLM-forced paths when corresponding flags explicitly require LLM

## References

- Canonical usage/runtime docs: [../UsageDocumentation.md](../UsageDocumentation.md)
- Open pending items: [../BaseCML/PENDING_ISSUES.md](../BaseCML/PENDING_ISSUES.md)
- Model pipeline details: [../../packages/models/README.md](../../packages/models/README.md)
