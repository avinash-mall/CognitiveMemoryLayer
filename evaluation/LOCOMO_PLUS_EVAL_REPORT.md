# LoCoMo-Plus Evaluation Comparison Report

Date: 2026-04-07

This report compares the local CML evaluation artifacts in this repository with the current online LoCoMo-Plus paper and upstream GitHub repository. For local numbers, the raw saved outputs are treated as authoritative. Existing prose docs in this repo are treated as secondary because they diverge from the saved artifacts.

## 1. Local Evaluation Snapshot

Primary local sources:

- [Judge summary](outputs/locomo_plus_qa_cml_judge_summary.json)
- [Judged records](outputs/locomo_plus_qa_cml_judged.json)
- [Predictions](outputs/locomo_plus_qa_cml_predictions.json)

Methodology used for the paper-style local summary:

- Factual average = unweighted mean of `single-hop`, `multi-hop`, `temporal`, `common-sense`, and `adversarial`
- Cognitive score = `Cognitive`
- Gap = factual average - cognitive score

| Metric | Local artifact value |
| --- | ---: |
| Total samples | 2,387 |
| Overall average across all six categories | 24.32% |
| Factual average (paper-style) | 28.26% |
| Cognitive | 21.32% |
| Gap | 6.94% |
| single-hop | 18.43% |
| multi-hop | 20.04% |
| temporal | 29.28% |
| common-sense | 39.58% |
| adversarial | 33.97% |
| Model id recorded in saved predictions | `Qwen/Qwen3.5-27B_cml` |

### Local docs are stale relative to the saved outputs

The existing local comparison prose does not match the raw summary JSON. `COMPARISON.md` still reports `31.49 / 21.45 / 10.04` and category values that differ materially from the saved outputs, while also describing the run as `CML + gpt-oss:20b` rather than the model id recorded in the predictions file. See [COMPARISON.md](COMPARISON.md) and [evaluation/README.md](README.md).

| Metric | Raw saved outputs | Existing local prose |
| --- | ---: | ---: |
| Factual average | 28.26% | 31.49% |
| Cognitive | 21.32% | 21.45% |
| Gap | 6.94% | 10.04% |
| single-hop | 18.43% | 56.06% |
| multi-hop | 20.04% | 46.10% |
| temporal | 29.28% | 5.92% |
| common-sense | 39.58% | 40.62% |
| adversarial | 33.97% | 8.74% |

## 2. Comparison To Paper Table 1

Online source:

- [LoCoMo-Plus paper, arXiv HTML v1](https://arxiv.org/html/2602.10715v1)

Current paper Table 1 reports a broader and newer baseline set than the local hard-coded comparison table. A few reference rows are reproduced below for context.

| Method | Factual average | Cognitive | Gap |
| --- | ---: | ---: | ---: |
| Local CML artifact run | 28.26% | 21.32% | 6.94% |
| gpt-4o | 62.99% | 21.05% | 41.94% |
| gemini-2.5-flash | 69.25% | 24.67% | 44.58% |
| gemini-2.5-pro | 71.78% | 26.06% | 45.72% |
| Text-embedding-large | 45.32% | 15.55% | 29.77% |
| Mem0 | 57.24% | 15.80% | 41.44% |
| SeCom | 57.53% | 14.90% | 42.63% |
| A-Mem | 59.64% | 17.20% | 42.44% |

Key observations:

- The local factual average, `28.26%`, is below every current Table 1 baseline in the paper.
- The local cognitive score, `21.32%`, is close to `gpt-4o` (`21.05%`) and above many paper baselines if taken literally.
- The local gap, `6.94%`, is far smaller than the paper range (`23.47%` to `45.72%`), but this should not be interpreted as evidence of superior robustness without the caveats in Section 4.
- Among the current paper rows, only `gemini-2.5-flash` and `gemini-2.5-pro` are higher than the local run on the raw cognitive number alone.

### Why this comparison is only partially apples-to-apples

The local run preserves some of the paper's high-level evaluation ideas, including unified-input evaluation and LLM-as-judge, but it is not a byte-for-byte reproduction of the upstream evaluation path.

- The paper evaluates full-context models, upstream RAG baselines, and memory systems under its own benchmark harness; the local repo routes evaluation through a CML-backed ingest and retrieval pipeline.
- The paper states that its RAG baselines retrieve the top-5 relevant dialogue segments.
- The local CML harness defaults to `--max-results 25`, which changes the retrieval regime before any model comparison is made.
- The local artifact model id is `Qwen/Qwen3.5-27B_cml`, while the local prose comparison still frames the run as `gpt-oss:20b`.

## 3. Comparison To The Upstream GitHub Evaluation Framework

Online source:

- [xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus)

The upstream GitHub repo presents LoCoMo-Plus as a benchmark and evaluation framework built around unified input generation plus an evaluate-then-judge flow. The local repo adapts that benchmark into a CML-specific pipeline.

| Aspect | Upstream repo / paper | Local repo |
| --- | --- | --- |
| Core workflow | Build unified input, run evaluation, then run judge | Ingest into CML, optionally consolidate/reconsolidate, query CML during QA, then run judge |
| Retrieval setup | Paper RAG baselines use top-5 segments | Local CML harness defaults to `max_results=25` |
| Reported local model | Not applicable | Existing prose says `gpt-oss:20b`, but saved predictions record `Qwen/Qwen3.5-27B_cml` |
| Comparison table freshness | Current paper Table 1 includes newer rows | Local hard-coded baseline list is stale |

Additional local-vs-upstream mismatches:

- The local baseline table in [compare.py](../packages/py-cml/src/cml/eval/compare.py) omits current paper rows such as `Qwen3-4B`, `Qwen3-8B`, `gpt-5-nano`, `gpt-4.1`, and `Text-embedding-small`.
- The same file still labels one row as `gpt-4o (full context)`, while the current paper table labels that row simply as `gpt-4o`.
- The local manual-eval docs in [evaluation/README.md](README.md) correctly describe the CML path as a specialized adaptation, but [COMPARISON.md](COMPARISON.md) overstates protocol equivalence when it says the same evaluation protocol is used.

## 4. Audit Findings And Confidence Caveats

The raw artifacts suggest that the local run is not just a low-scoring run; it is also a degraded evaluation run with substantial pipeline failures. That matters because the judge summary converts those failures into zero-score rows.

### Audit findings

| Finding | Evidence from raw artifacts | Likely impact |
| --- | --- | --- |
| All saved predictions contain chain-of-thought-style output | `2387 / 2387` predictions include `Thinking Process:` rather than only the requested short answer | The QA model did not follow the intended output format |
| Many predictions explicitly report missing context | `828 / 2387` predictions mention missing or absent context | Suggests retrieval, prompt construction, or context-delivery issues in a large subset of samples |
| Most judged rows are unlabeled | `1647 / 2387` judged rows have blank `judge_label`; another `65 / 2387` use `...` | Official averages are strongly affected by evaluation failure, not only model failure |
| Large share of judge failures are API errors | `974 / 2387` judged rows preserve `[API Error: ...]` in `judge_reason` | External judge calls failed and were counted as zero |
| Remaining judge failures are unparseable reasoning dumps | `673 / 2387` blank-label rows contain judge-side reasoning text instead of parseable JSON | Judge output formatting failed and was counted as zero |

Per-category missing-or-ellipsis judge rates:

| Category | Missing or ellipsis labels |
| --- | ---: |
| single-hop | 80.86% |
| multi-hop | 78.37% |
| temporal | 68.85% |
| common-sense | 60.42% |
| adversarial | 65.47% |
| Cognitive | 59.85% |

### Why the summary is depressed by run-health issues

The judge implementation treats unknown or blank labels as zero-score and still includes them in the final averages. In other words, the published local summary conflates:

- true model errors
- upstream judge API failures
- judge formatting failures

This behavior follows directly from the local judge code:

- [task_eval/llm_as_judge.py](locomo_plus/task_eval/llm_as_judge.py) maps only `correct`, `partial`, and `wrong` to scores
- unknown or blank labels score as `0.0`
- category and overall averages are computed over all rows, including failed rows
- [task_eval/utils.py](locomo_plus/task_eval/utils.py) serializes API exceptions as literal `[API Error: ...]` strings, which then propagate into the judged output

### Bottom line

The saved local artifacts do support a factual comparison report, but only with strong caveats:

- The raw local paper-style numbers are `28.26 / 21.32 / 6.94`.
- Those numbers do not match the existing local comparison prose.
- The current local comparison code uses a stale paper baseline table.
- The saved run appears evaluation-degraded enough that its summary is best read as a mixed signal of model quality plus pipeline instability, not a clean benchmark result.

## Sources

Local repository sources:

- [outputs/locomo_plus_qa_cml_judge_summary.json](outputs/locomo_plus_qa_cml_judge_summary.json)
- [outputs/locomo_plus_qa_cml_judged.json](outputs/locomo_plus_qa_cml_judged.json)
- [outputs/locomo_plus_qa_cml_predictions.json](outputs/locomo_plus_qa_cml_predictions.json)
- [evaluation/README.md](README.md)
- [evaluation/COMPARISON.md](COMPARISON.md)
- [packages/py-cml/src/cml/eval/compare.py](../packages/py-cml/src/cml/eval/compare.py)
- [packages/py-cml/src/cml/eval/locomo.py](../packages/py-cml/src/cml/eval/locomo.py)
- [evaluation/locomo_plus/task_eval/llm_as_judge.py](locomo_plus/task_eval/llm_as_judge.py)
- [evaluation/locomo_plus/task_eval/utils.py](locomo_plus/task_eval/utils.py)

Online sources:

- [LoCoMo-Plus paper, arXiv HTML v1](https://arxiv.org/html/2602.10715v1)
- [Upstream LoCoMo-Plus GitHub repo](https://github.com/xjtuleeyf/Locomo-Plus)
