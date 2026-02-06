# Phase 13: Code Improvements

**Status:** Completed

## Goal

Address identified issues: embeddings on update, working memory eviction, salience, reconsolidation behavior.

## Summary of Changes

- **Embedding updates (13.1)**: When `orchestrator.update()` is called with changed `text`, the orchestrator now re-embeds and re-extracts entities and passes the updated embedding/entities into the store.
- **Working memory eviction (13.2)**: `WorkingMemoryState.add_chunk()` uses recency-aware eviction: keep the most recent N chunks regardless of salience, then evict from older chunks by salience.
- **Sentiment-aware salience (13.3)**: Added `_compute_salience_boost_for_sentiment()` in the chunker; integrated into `RuleBasedChunker` and `SemanticChunker` so emotionally significant content gets a salience boost (capped at 0.3).
- **Reconsolidation archive (13.4)**: Belief revision now archives instead of deleting on contradiction/correction: `_plan_time_slice` and `_plan_correction` set `valid_to` and `status=ARCHIVED` on the old record instead of deleting.
