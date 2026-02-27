# PENDING_ISSUES (Post-Custom-Models)

Date: 2026-02-27

This is the canonical list of unresolved gaps after modelpack + NER integration.

## Already completed in runtime

1. Modelpack runtime adapter and task wiring for:
- `query_domain`
- `scope_match`
- `constraint_stability`
- `supersession`

2. NER-based fallback wiring for:
- entity extraction
- relation extraction
- write-time fact extraction
- PII span detection support

3. Retrieval/ops fixes:
- `retrieval_meta` diagnostics returned by read APIs
- batched semantic fact retrieval by categories
- batched associative expansion by `source_session_id`
- explicit async `DatabaseManager.create()` lifecycle

## Remaining technical gaps

1. Consolidation semantic lineage completeness
- Gap: supersession lineage is stored for episodic deactivation, but cross-key semantic-fact lineage is still limited.
- Need: explicit lineage links across consolidated semantic facts and episodic constraints.

2. Conflict detector model integration depth
- Gap: pair-model integration exists for supersession/relevance tasks, but conflict reasoning remains partially heuristic/LLM.
- Need: complete pair-model conflict path coverage and evaluation against reconsolidation outcomes.

3. Consolidation quality guardrails
- Gap: consolidation preserves more constraint semantics now, but quality can still drift on mixed-topic clusters.
- Need: stronger validation and fallback policies before writing gists to semantic facts.

## `LLM_INTERNAL__*` tasks still required

These are not replaced by current custom classifiers:

1. Unified write extraction generation (`src/extraction/unified_write_extractor.py`)
2. Consolidation gist generation (`src/consolidation/summarizer.py`)
3. Forgetting compression summaries (`src/forgetting/compression.py`)
4. Optional LLM-only classifier/reranker modes when forced by flags:
- `FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY=true`
- `FEATURES__USE_LLM_CONSTRAINT_RERANKER=true`

## NER-priority follow-ups

1. Alias/coreference normalization for entity scope matching across turns.
2. Expanded deterministic PII recognizers (regional phone/address formats).
3. Entity canonicalization for consolidation-time supersession decisions.
