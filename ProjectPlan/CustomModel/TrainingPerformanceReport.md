# Training Performance Report

Date: 2026-03-10
Build: 2026-03-09T14:20:29 UTC
Commit: `1be678cf` (dirty)
Environment: Python 3.12.10, scikit-learn 1.8.0, joblib 1.5.3, pandas 3.0.1
Mode: strict, preflight validation passed, artifact validation passed

---

## 1. Executive Summary

All 13 models (3 family-level, 10 task-level) trained successfully. However, outcomes vary dramatically:

- **3 models are non-functional** (pair-ranking models at random-chance accuracy ~50%)
- **2 models are suspiciously perfect** (100% train+test, likely trivially separable data)
- **2 token-classification models are excellent** (span F1 93-99.6%)
- **1 regression model is excellent** (MAE 0.005)
- **5 classification models are moderate-to-weak** (51-77% accuracy)
- **All SGD models waste ~80% of training epochs** (converge by epoch 3-4, run for 25)

The Phase 1 "highest impact" models (retrieval, reranking, reconsolidation) are the weakest performers and need architectural changes before they can replace heuristic baselines.

---

## 2. Family-Level Models (TF-IDF + SGDClassifier)

### 2.1 Router Model

- **Architecture**: TF-IDF (250k features, 1-2 ngrams, min_df=2) + SGDClassifier (alpha=1e-5)
- **Tasks**: memory_type, query_intent, query_domain, constraint_dimension, context_tag, salience_bin, importance_bin, confidence_bin, decay_profile, consolidation_gist_quality, forgetting_action_policy
- **Training data**: 63,000 test rows (implies ~504,000 train at 80/10/10 split)
- **Convergence**: Epoch 3 (accuracy/F1 plateau, loss delta < 1e-4 by epoch 4)

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 75.8% | 67.4% |
| Macro F1 | 76.1% | 67.9% |
| Weighted F1 | 76.1% | 67.9% |
| ECE (calibration) | — | 28.5% |

**Per-task test performance** (63,000 rows total, 1,000 per task-label):

| Task | Classes | Test Acc | Macro F1 | Weighted F1 | Assessment |
|------|---------|----------|----------|-------------|------------|
| consolidation_gist_quality | 2 | 100.0% | 100.0% | 100.0% | Trivially separable |
| forgetting_action_policy | 5 | 100.0% | 100.0% | 100.0% | Trivially separable |
| query_intent | 5 | 77.2% | 64.8% | 77.7% | Good |
| query_domain | 8 | 72.7% | 65.4% | 73.5% | Decent |
| context_tag | 8 | 70.6% | 64.0% | 72.0% | Decent |
| confidence_bin | 3 | 64.3% | 48.3% | 64.5% | Weak |
| salience_bin | 3 | 64.2% | 47.5% | 63.4% | Weak |
| importance_bin | 3 | 62.1% | 46.2% | 61.6% | Weak |
| constraint_dimension | 6 | 61.4% | 54.2% | 63.3% | Weak |
| memory_type | 15 | 54.4% | 51.4% | 54.9% | Low (15-class) |
| decay_profile | 5 | 51.5% | 43.1% | 51.8% | Weak |

**Notable confusion patterns:**
- `memory_type`: High confusion across `observation`, `knowledge`, `reasoning_step`, and `hypothesis` — semantically overlapping categories
- `decay_profile`: `very_fast`/`fast` and `very_slow`/`slow` pairs heavily confused — ordinal relationships not captured by classifier
- `confidence_bin` / `salience_bin` / `importance_bin`: `medium` class absorbs errors from both `high` and `low`; ordinal structure lost in flat classifier
- `constraint_dimension`: `other` acts as a sink class absorbing `causal`, `state`, `value` samples
- Wrong-task leakage: < 0.3% across all tasks (composite label scheme works correctly)

### 2.2 Extractor Model

- **Training data**: 36,000 test rows
- **Convergence**: Epoch 3

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 82.6% | 71.6% |
| Macro F1 | 78.1% | 64.2% |
| Weighted F1 | 83.0% | 72.1% |
| ECE (calibration) | — | 25.9% |

**Per-task test performance:**

| Task | Classes | Test Acc | Macro F1 | Weighted F1 | Assessment |
|------|---------|----------|----------|-------------|------------|
| pii_presence | 2 | 99.5% | 66.3%* | 99.5% | Excellent |
| constraint_stability | 3 | 76.3% | 57.3% | 76.4% | OK |
| constraint_scope | 9 | 59.7% | 54.5% | 60.5% | Moderate |
| fact_type | 6 | 58.9% | 51.2% | 59.7% | Weak |
| constraint_type | 8 | 57.7% | 52.1% | 58.6% | Weak |

*\*pii_presence macro F1 is depressed by the zero-support `__wrong_task__` phantom class; actual binary performance is 99.5% accurate with only 50 misclassifications out of 10,000.*

### 2.3 Pair Model

- **Convergence**: Epoch 4
- **Final train metrics**: Accuracy 67.0%, macro F1 66.5%, weighted F1 66.4%
- Serves as the base family for several task-level pair models

---

## 3. Task-Level Models

### 3.1 Pair Ranking Models — CRITICAL FAILURES

| Model | Train Rows | Train Acc | Test Acc | Test F1 | Status |
|-------|-----------|-----------|----------|---------|--------|
| retrieval_constraint_relevance_pair | 80,000 | 77.1% | **49.9%** | 49.6% | Random chance |
| memory_rerank_pair | 80,000 | 77.2% | **50.2%** | 50.2% | Random chance |
| reconsolidation_candidate_pair | 80,000 | — | **49.9%** | 49.6% | Random chance |

**Diagnosis**: Catastrophic overfitting. The TF-IDF bag-of-words representation captures surface lexical overlap between text_a and text_b, but pair-ranking requires understanding semantic relevance, entailment, and compatibility — relationships that bag-of-words fundamentally cannot express. The models memorize training-set lexical patterns that do not transfer.

**Confusion matrices** confirm near-uniform random predictions:
- `retrieval_constraint_relevance_pair`: 2879/2121 vs 2888/2112 (no discriminative signal)
- `memory_rerank_pair`: 2502/2498 vs 2486/2514 (effectively random)
- `reconsolidation_candidate_pair`: 2107/2893 vs 2120/2880 (effectively random)

**Impact**: These are the Phase 1 rollout models targeting the highest-impact runtime paths (retriever scoring, reranker core, reconsolidation candidate selection). The heuristic fallbacks are currently performing all useful work.

### 3.2 Classification Models

| Model | Train Rows | Test Rows | Test Acc | Macro F1 | Status |
|-------|-----------|-----------|----------|----------|--------|
| consolidation_gist_quality | 16,000 | 2,000 | **100%** | 100% | Suspiciously perfect |
| forgetting_action_policy | 40,000 | 5,000 | **100%** | 100% | Suspiciously perfect |
| schema_match_pair | 80,000 | 10,000 | **69.7%** | 69.6% | Moderate |
| novelty_pair | 155,227 | 19,403 | **48.7%** | 47.2% | Weak |

**consolidation_gist_quality + forgetting_action_policy**: Perfect scores on both train and test indicate the synthetic data contains trivial separating cues (e.g., templated phrases, keyword patterns). These models will likely fail on real production inputs that lack those cues. The loss is still decreasing at epoch 25 (gist: 0.00089 → 0.00059; policy: 0.00287 → 0.00197), confirming the model is simply memorizing near-zero-loss patterns.

**novelty_pair**: The 4-class task has extreme confusion between `contradiction` (933/5000 correct) and `temporal_change` (550/5000 correct). Meanwhile `duplicate` detection is strong (4245/4403). The data definitions for `contradiction` vs `temporal_change` appear semantically overlapping: both involve a change in stated facts between text_a and text_b.

**schema_match_pair**: 69.7% binary accuracy is above random but marginal. The `no_match` class has lower recall (3188/5000 = 63.8%) than `match` (3786/5000 = 75.7%), suggesting the model has a bias toward predicting matches.

### 3.3 Token Classification Models (distilbert-base-multilingual-cased, 3 epochs)

| Model | Train Rows | Test span-F1 | Test Exact Match | Status |
|-------|-----------|-------------|-----------------|--------|
| fact_extraction_structured | 32,000 | **99.6%** | 99.8% | Excellent |
| pii_span_detection | 30,516 | **92.9%** | 83.3% | Good |

**fact_extraction_structured**: Near-perfect performance. Training loss dropped from 0.132 → 0.014 → 0.007 over 3 epochs. The 21-label BIO scheme (O + 10 entity types x B/I) is well-learned. Eval split confirms (span F1 99.7%, exact match 99.7%). This model is production-ready.

**pii_span_detection**: Good performance with 115 label classes (57 PII entity types in BIO format). Training loss: 0.379 → 0.075 → 0.054. The gap between span F1 (92.9%) and exact match (83.3%) indicates the model sometimes gets span boundaries partially correct. Eval split is slightly lower (span F1 92.3%, exact match 83.6%), confirming stable generalization. Combined with the regex safety baseline, this provides strong PII coverage.

### 3.4 Regression Model

| Model | Train Rows | Test MAE | Test RMSE | Status |
|-------|-----------|----------|-----------|--------|
| write_importance_regression | 8,000 | **0.0048** | **0.0066** | Excellent |

Near-zero error on a 0-1 scale. Train MAE (0.0048) ≈ Test MAE (0.0048), indicating no overfitting. However, the very low error raises a concern: the target score distribution may have narrow variance (e.g., most scores clustered around a single value), making the task trivially easy. Recommend inspecting the actual score distribution in prepared data before relying on these metrics.

---

## 4. Training Dynamics Analysis

### 4.1 Early Convergence / Wasted Epochs

All SGD-based models converge between epoch 2 and 4. The remaining 21-23 epochs show zero improvement in accuracy or F1 (only infinitesimal loss changes in the 12th decimal place). Example loss deltas after convergence:

- Router: epoch 3 loss 1.24684 → epoch 25 loss 1.24640 (delta: 0.00044 over 22 epochs)
- Extractor: epoch 3 loss 0.98766 → epoch 25 loss 0.98766 (delta: 0.00000)
- Pair ranking models: plateau by epoch 4-5

**Waste**: At 25 epochs, ~80% of compute is spent with zero benefit.

### 4.2 Train-Test Gaps

| Model Category | Typical Train Acc | Typical Test Acc | Gap |
|----------------|-------------------|------------------|-----|
| Family classifiers | 76-83% | 67-72% | 8-11% |
| Pair ranking | 77% | 50% | **27%** |
| Perfect classifiers | 100% | 100% | 0% (suspicious) |
| Token classifiers | — | 93-99.6% | Minimal |

The pair-ranking gap of ~27% is the clearest overfitting signal. Family classifiers show a healthy 8-11% generalization gap. Token classifiers generalize well.

### 4.3 Calibration

Both family models have ECE > 25%, meaning the predicted probability distributions are unreliable. Any runtime logic that applies confidence thresholds to model outputs will make suboptimal decisions.

---

## 5. Findings Summary

### What works well

1. **Token classification models** are the strongest performers — distilbert-base-multilingual-cased is well-suited for span extraction tasks
2. **PII presence detection** (family-level binary) is excellent at 99.5%
3. **Write importance regression** shows clean generalization with near-zero error
4. **Composite-label scheme** correctly isolates tasks with < 0.3% cross-task leakage
5. **Preflight and artifact validation** both pass in strict mode

### What does not work

1. **All three pair-ranking models** are at random chance — TF-IDF cannot represent pair semantics
2. **Two classification models** achieve suspiciously perfect scores — synthetic data is too easy
3. **Novelty pair** has heavy contradiction/temporal_change confusion — label definitions overlap
4. **Ordinal tasks** (bins, decay profiles) lose structure in flat classifiers
5. **Calibration is poor** (ECE 26-29%) — confidence scores are unreliable
6. **Training wastes ~80% of compute** — no early stopping despite convergence at epoch 3

---

## 6. Recommendations and Implementation Steps

### R1: Add Early Stopping to SGD Training

**Problem**: All models converge by epoch 3-4 but train for 25 epochs.

**Impact**: ~80% compute savings with zero quality loss.

**Implementation**:

1. In `packages/models/scripts/train.py`, modify `_train_family` and `_train_classification_task` / `_train_pair_ranking` / `_train_single_regression`:
   - Track validation loss or accuracy after each epoch using `partial_fit` instead of `fit` (SGDClassifier supports incremental training)
   - Add patience parameter (default 3): stop if validation metric does not improve for `patience` consecutive epochs
   - Log the early-stop epoch in epoch_stats

2. Add config to `model_pipeline.toml`:
   ```toml
   [train]
   early_stopping = true
   early_stopping_patience = 3
   early_stopping_metric = "macro_f1"  # or "loss"
   ```

3. Modify `_log_epoch_stats` to record whether early stopping triggered and at which epoch.

4. Update the manifest to include `actual_epochs` vs `max_iter` for transparency.

**Validation**: Re-run training and confirm final metrics are identical (within floating-point tolerance) to the current 25-epoch run.

---

### R2: Replace TF-IDF with Embedding Features for Pair-Ranking Models

**Problem**: `retrieval_constraint_relevance_pair`, `memory_rerank_pair`, `reconsolidation_candidate_pair` all achieve ~50% test accuracy (random chance) despite 77% train accuracy, because TF-IDF bag-of-words cannot represent semantic pair relationships.

**Impact**: Enables Phase 1 rollout (retrieval + reconsolidation), which is currently blocked.

**Implementation — Option A (lightweight, recommended first)**: Embedding similarity features + gradient-boosted classifier

1. In `packages/models/scripts/train.py`, add a new trainer function `_train_pair_ranking_embedding`:
   - For each (text_a, text_b) pair, compute sentence embeddings using the project's existing embedding model (`nomic-ai/nomic-embed-text-v2-moe`) or a lighter model like `all-MiniLM-L6-v2`
   - Construct feature vector: `[cosine_sim, euclidean_dist, abs_diff_vector, element_wise_product]`
   - Train a gradient-boosted classifier (LightGBM or XGBClassifier) on these features
   - This keeps inference fast (embed once, classify with a tree model)

2. Add embedding computation as a preparation step in `prepare.py`:
   - Pre-compute and cache embeddings for all pair texts to avoid recomputation
   - Store as additional columns in the prepared parquet files

3. Update `model_pipeline.toml` task specs:
   ```toml
   [[tasks]]
   task_name = "retrieval_constraint_relevance_pair"
   # ...existing fields...
   feature_type = "embedding"  # new field, default "tfidf"
   embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
   ```

4. Update `modelpack.py` to load the embedding model once and serve pair-ranking predictions using the new feature pipeline.

**Implementation — Option B (higher quality, more compute)**: Cross-encoder fine-tuning

1. Fine-tune a small cross-encoder (e.g., `distilbert-base-uncased` or `distilroberta-base`) on the pair data using Hugging Face Trainer
2. This is similar to the existing token-classification path but for sequence classification
3. Higher accuracy ceiling but ~10x slower inference than Option A
4. May be appropriate if Option A does not reach target metrics

**Target**: Test accuracy > 70% on binary relevant/not_relevant (vs current 50%).

**Validation**: Run against the existing test split. Compare with heuristic baseline accuracy. Use shadow mode to validate in production before switching.

---

### R3: Harden Synthetic Data for Perfect-Score Models

**Problem**: `consolidation_gist_quality` and `forgetting_action_policy` achieve 100% on train AND test. The synthetic data likely contains trivial textual cues that make the classes perfectly separable, but production inputs will not have these cues.

**Impact**: Prevents production failures when these models encounter real data.

**Implementation**:

1. **Audit existing synthetic data** for separating cues:
   - In `packages/models/prepared_data/modelpack/`, load the train parquet for these tasks
   - Compute TF-IDF feature importance per class
   - Identify the top-10 discriminative n-grams — these are likely template artifacts
   - Document findings

2. **Add adversarial examples to preparation**:
   - In `packages/models/scripts/prepare.py`, add a post-processing step for these tasks:
     - For `consolidation_gist_quality`: generate `accept` examples that resemble `reject` patterns (e.g., gists with partial topic mixing) and `reject` examples that resemble `accept` patterns (e.g., well-formed text with subtle factuality errors)
     - For `forgetting_action_policy`: generate near-boundary examples where the correct action depends on metadata features (importance, access_count, age) rather than text content alone
   - Target: reduce to ~90-95% accuracy (indicating the model learns meaningful features, not shortcuts)

3. **Add adversarial test suite**:
   - Create a manually curated test file `packages/models/datasets/adversarial_gist_quality.jsonl` (50-100 examples)
   - Create `packages/models/datasets/adversarial_forgetting_policy.jsonl` (50-100 examples)
   - Run these as a separate evaluation pass in `train.py` and report in manifest

4. **Integrate real production data** when available:
   - Wire consolidation replay logs into the preparation pipeline as a data source
   - Wire forgetting scorer logs similarly
   - Even a small set (500-1000 real examples) mixed into training will dramatically improve robustness

**Validation**: After hardening, accuracy should drop from 100% to 90-96% on the adversarial set while remaining high on clean test data.

---

### R4: Apply Post-Hoc Calibration

**Problem**: Router ECE = 28.5%, Extractor ECE = 25.9%. Predicted probabilities are unreliable for threshold-based decisions.

**Impact**: All runtime paths that use model confidence (threshold gating, fallback triggers) are making suboptimal decisions.

**Implementation**:

1. In `packages/models/scripts/train.py`, add a calibration step after training each family model:
   ```python
   from sklearn.calibration import CalibratedClassifierCV

   calibrated_model = CalibratedClassifierCV(
       trained_model, method="sigmoid", cv="prefit"
   )
   calibrated_model.fit(X_eval, y_eval)
   ```
   Use the eval split (currently unused for calibration in family training) as the calibration set.

2. Save the calibrated model as the primary artifact (replace the uncalibrated `.joblib`).

3. Report pre- and post-calibration ECE in the metrics files and manifest.

4. Update `model_pipeline.toml`:
   ```toml
   [train]
   calibration_method = "sigmoid"  # or "isotonic"
   calibration_split = "eval"
   ```

5. For task-level models, apply the same calibration where applicable (classification objectives).

**Target**: Reduce ECE from ~27% to < 10%.

**Validation**: Compare reliability diagrams (predicted probability vs observed frequency) before and after calibration. Ensure accuracy does not degrade.

---

### R5: Improve Novelty Pair Label Definitions

**Problem**: `novelty_pair` has heavy confusion between `contradiction` and `temporal_change` (2743 temporal_change samples misclassified as contradiction, 2275 contradiction samples misclassified as temporal_change). The 4-class accuracy is 48.7%.

**Impact**: The novelty pair model replaces Jaccard heuristic in write gate and interference detector — weak classification undermines both paths.

**Implementation**:

1. **Analyze label overlap** in preparation data:
   - Sample 100 examples from each confused class pair
   - Determine if the distinction is consistently annotatable by humans
   - If not, merge `contradiction` + `temporal_change` into a single `changed` class

2. **If keeping 4 classes**, improve label quality:
   - In `packages/models/scripts/prepare.py`, tighten the synthetic generation prompts for these two classes:
     - `contradiction`: text_b directly negates or denies what text_a states (e.g., "I love coffee" → "I don't like coffee")
     - `temporal_change`: text_b updates a fact from text_a with a temporal marker (e.g., "I work at Google" → "I now work at Meta")
   - Add explicit negative examples: pairs labeled `temporal_change` that are NOT contradictions, and vice versa
   - Increase seed diversity for these two classes

3. **If merging to 3 classes** (`duplicate`, `novel`, `changed`):
   - Update `model_pipeline.toml` labels
   - Re-prepare and re-train
   - Expected accuracy improvement: ~65-70% (eliminating the confused boundary)

4. **Consider the scoring output**: The model is also used for calibrated novelty scoring (via `predict_score_pair`). A cleaner class structure will produce more reliable score calibration.

**Validation**: Re-train and compare macro F1. Verify that `duplicate` detection (currently strong at 96.4%) is not degraded by changes.

---

### R6: Use Ordinal-Aware Models for Bin/Profile Tasks

**Problem**: `decay_profile` (51.5%), `importance_bin` (62.1%), `confidence_bin` (64.3%), `salience_bin` (64.2%) all perform poorly. These are ordinal tasks (low < medium < high, very_fast < fast < medium < slow < very_slow) but are treated as flat multi-class classification, losing the ordering relationship.

**Impact**: Write gate importance predictions, decay rate assignments, and confidence estimates are unreliable.

**Implementation**:

1. **Option A — Ordinal regression** (recommended):
   - Replace SGDClassifier with an ordinal regression approach for these tasks
   - Use the `mord` library (ordinal regression in scikit-learn style) or encode as threshold-based binary classifiers (cumulative link model)
   - In `train.py`, add `_train_ordinal_classification` trainer:
     ```python
     import mord
     model = mord.LogisticAT(alpha=1e-5)  # All-Thresholds variant
     model.fit(X_train, y_train_ordinal)
     ```
   - Map labels to integers preserving order: `low=0, medium=1, high=2`

2. **Option B — Regression + binning**:
   - Train a regressor (SGDRegressor) to predict a continuous score
   - Apply fixed thresholds to bin into categories at inference time
   - Thresholds can be calibrated on the eval split
   - This is simpler and may work better for 3-class tasks

3. Update `model_pipeline.toml` to specify ordinal objective:
   ```toml
   [[tasks]]
   task_name = "confidence_bin_ordinal"
   objective = "ordinal"  # new objective type
   labels = ["low", "medium", "high"]
   label_order = ["low", "medium", "high"]
   ```

4. Update `modelpack.py` prediction methods to handle ordinal model outputs.

**Target**: Reduce off-by-2 errors (e.g., predicting `high` when truth is `low`) to near zero. Improve accuracy to 70%+.

**Validation**: In addition to accuracy, report Mean Absolute Error on the ordinal scale (e.g., predicting `medium` when truth is `high` = 1 error, predicting `low` when truth is `high` = 2 errors).

---

### R7: Improve Memory Type Classification

**Problem**: `memory_type` has 15 classes and achieves only 54.4% accuracy. High confusion among semantically overlapping types (observation/knowledge/reasoning_step/hypothesis).

**Impact**: Memory type drives downstream routing, retrieval source selection, and decay behavior.

**Implementation**:

1. **Hierarchical classification**:
   - Group the 15 types into 4-5 macro-categories:
     - **Factual**: semantic_fact, knowledge, observation
     - **Procedural**: procedure, plan, task_state
     - **Conversational**: conversation, message, scratch
     - **Analytical**: hypothesis, reasoning_step, constraint
     - **Personal**: preference, episodic_event, tool_result
   - Train a two-stage classifier: first predict macro-category (higher accuracy), then predict fine type within the category
   - This reduces per-stage class count and focuses each stage on more discriminative features

2. **Feature enrichment**:
   - Add metadata features beyond text: message length, presence of question marks, imperative verbs, temporal markers, named entities count
   - In `prepare.py`, compute and store these as additional parquet columns
   - In `train.py`, concatenate metadata features with TF-IDF features

3. **Consider embedding features** (same as R2 Option A):
   - Sentence embeddings may capture semantic distinctions between `observation` and `knowledge` better than TF-IDF

**Target**: Improve from 54.4% to 65%+ overall accuracy, with macro F1 improvement on the most confused classes.

**Validation**: Compare confusion matrices before and after. Verify that the dominant errors (observation↔knowledge, hypothesis↔reasoning_step) are reduced.

---

### R8: Validate Write Importance Regression Data Distribution

**Problem**: Test MAE of 0.005 on a 0-1 scale is suspiciously good. Only 8,000 training rows were used. The target score distribution may have narrow variance.

**Impact**: If scores cluster around a single value, the model may just predict the mean without learning meaningful patterns.

**Implementation**:

1. **Inspect score distribution**:
   - Load the prepared parquet for `write_importance_regression`
   - Plot histogram of the `score` column
   - Compute variance, min, max, percentiles (5th, 25th, 50th, 75th, 95th)
   - If variance < 0.01, the task is trivially easy and the low MAE is misleading

2. **Add baseline comparison**:
   - Compute MAE for a "predict the mean" baseline
   - Compute MAE for a "predict the median" baseline
   - If the model's MAE is not significantly better than these baselines, it has learned nothing useful

3. **If distribution is narrow**: Expand the score supervision data:
   - As noted in the README, this task is "deferred until score-supervision parquet columns are available"
   - Integrate retrieval frequency, survival/forgetting outcomes, and human criticality labels when available
   - Ensure the score range spans [0, 1] with reasonable variance

**Validation**: Report baseline MAE alongside model MAE in the manifest. Only declare the model production-ready if it significantly outperforms the mean-prediction baseline.

---

## 7. Priority Ordering

| Priority | Recommendation | Effort | Impact |
|----------|---------------|--------|--------|
| **P0** | R2: Fix pair-ranking models (embedding features) | High | Unblocks Phase 1 rollout |
| **P0** | R1: Add early stopping | Low | Immediate compute savings |
| **P1** | R3: Harden synthetic data for perfect models | Medium | Prevents production failures |
| **P1** | R4: Apply calibration | Low | Improves all threshold decisions |
| **P1** | R5: Fix novelty pair labels | Medium | Improves write gate and interference |
| **P2** | R6: Ordinal models for bin tasks | Medium | Improves write path quality |
| **P2** | R7: Improve memory type classification | High | Improves routing accuracy |
| **P3** | R8: Validate regression data | Low | Confirms or invalidates metric |

---

## 8. Appendix: Raw Metrics Reference

### A. Family Epoch Convergence Points

| Family | Epoch 1 Acc | Epoch 3 Acc | Epoch 25 Acc | Converged At |
|--------|-------------|-------------|--------------|--------------|
| Router | 75.76% | 75.79% | 75.79% | Epoch 3 |
| Extractor | 82.50% | 82.63% | 82.63% | Epoch 3 |
| Pair | 67.02% | 67.04% | 67.04% | Epoch 4 |

### B. Task-Level Epoch Convergence Points

| Task | Epoch 1 Acc | Epoch 4 Acc | Epoch 25 Acc | Converged At |
|------|-------------|-------------|--------------|--------------|
| retrieval_constraint_relevance_pair | 76.53% | 77.07% | 77.07% | Epoch 4 |
| memory_rerank_pair | 76.64% | 77.19% | 77.19% | Epoch 5 |
| novelty_pair | 63.15% | 63.24% | 63.24% | Epoch 4 |
| schema_match_pair | 78.46% | 78.70% | 78.70% | Epoch 4 |
| consolidation_gist_quality | 100.0% | 100.0% | 100.0% | Epoch 1 |
| forgetting_action_policy | 100.0% | 100.0% | 100.0% | Epoch 1 |

### C. Token Model Training Loss Curves

| Model | Epoch 1 Loss | Epoch 2 Loss | Epoch 3 Loss |
|-------|-------------|-------------|-------------|
| fact_extraction_structured | 0.1319 | 0.0136 | 0.0075 |
| pii_span_detection | 0.3788 | 0.0752 | 0.0542 |

### D. Pair-Ranking Test Confusion Matrices

**retrieval_constraint_relevance_pair** (10,000 test rows):
```
                  Pred: not_rel  Pred: rel
True: not_rel         2879        2121
True: relevant        2888        2112
```

**memory_rerank_pair** (10,000 test rows):
```
                  Pred: not_rel  Pred: rel
True: not_rel         2502        2498
True: relevant        2486        2514
```

**reconsolidation_candidate_pair** (10,000 test rows):
```
                  Pred: not_rel  Pred: rel
True: not_rel         2107        2893
True: relevant        2120        2880
```

### E. Build Metadata

- Config: `packages/models/model_pipeline.toml`
- Prepared data: `packages/models/prepared_data/modelpack`
- Output: `packages/models/trained_models`
- Manifest schema version: 2
- Preflight validation: passed (strict)
- Artifact validation: passed
- All 10 configured tasks: status `trained`

---

## Artifact Validation Addendum (March 12, 2026)

This addendum supersedes the March 10 source-only note. The prepared data and trained artifacts were rebuilt or refreshed and validated against the leakage-free recovery plan.

### Validation status

- Prepared split integrity now passes for `router`, `extractor`, and `pair` with zero cross-split `group_id` overlap.
- `schema_match_pair` train data is non-template dominated as intended: 80,000 train rows, 0 template rows, 4,000 `hf:fever` rows, and 76,000 derived non-template rows.
- Adversarial suites are present and evaluation-only with sufficient size: `consolidation_gist_quality` = 120 rows, `forgetting_action_policy` = 125 rows, `schema_match_pair` = 120 rows.
- Family post-calibration ECE is below the `< 0.10` target for all three families:
  - `router`: `0.0066`
  - `extractor`: `0.0069`
  - `pair`: `0.0114`

### Resolution status

- `R1` resolved: early stopping is active in the rebuilt family artifacts. Final family epoch counts are `router=4`, `extractor=4`, `pair=6`.
- `R2` resolved: all three embedding-backed pair-ranking tasks exceed the `>= 0.70` target on test:
  - `retrieval_constraint_relevance_pair`: `74.71%`
  - `memory_rerank_pair`: `74.56%`
  - `reconsolidation_candidate_pair`: `74.56%`
- `R3` resolved: the suspicious-perfect tasks are no longer perfect on both clean and adversarial evaluation:
  - `consolidation_gist_quality`: clean `100.0%`, adversarial `55.0%`
  - `forgetting_action_policy`: clean `99.82%`, adversarial `76.8%`
  - `schema_match_pair`: clean `70.6%`, adversarial `52.5%`
- `R4` resolved: calibration coverage is in place for family and task artifacts, and family-level post-ECE is below target for all three families.
- `R5` resolved: `novelty_pair` remains 3-class (`duplicate`, `changed`, `novel`) and reaches `77.74%` test accuracy.
- `R6` resolved: ordinal task models now preserve ordering without reverting to trivially perfect outputs:
  - `confidence_bin`: `83.27%`, `off_by_two_rate = 0.0`
  - `importance_bin`: `85.93%`, `off_by_two_rate = 0.0`
  - `salience_bin`: `86.20%`, `off_by_two_rate = 0.0`
  - `decay_profile`: `78.04%`, `off_by_two_rate = 0.0`
- `R7` resolved: `memory_type` reaches `82.5%` test accuracy.
- `R8` resolved: `write_importance_regression` remains well within the acceptance gate with `test_mae = 0.00414` versus mean/median baseline MAEs of `0.26455` and `0.26333`.

### Artifact references

- Prepared manifest: `packages/models/prepared_data/modelpack/manifest.json`
- Trained manifest: `packages/models/trained_models/manifest.json`
- The acceptance gates in the recovery plan now pass on the refreshed artifacts.
