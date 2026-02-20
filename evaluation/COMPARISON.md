# LoCoMo-Plus: CML vs Other Methods

This document compares **this project's** (CML + gpt-oss:20b) evaluation scores with baselines from the **Locomo-Plus paper** (arXiv:2602.10715, Table 1). The same evaluation protocol is used: LLM-as-judge, constraint consistency, no task disclosure.

## Latest CML Run (from `outputs/locomo_plus_qa_cml_judge_summary.json`)

| Metric | Value |
|--------|--------|
| **LoCoMo (factual) average** | **31.49%** (single-hop 56.06%, multi-hop 46.10%, temporal 5.92%, commonsense 40.62%, adversarial 8.74%) |
| **LoCoMo-Plus (Cognitive)** | **21.45%** |
| **Gap** (factual − cognitive) | **10.04%** |
| Total samples | 2,387 |

## Comparison with Paper Baselines

Run the comparison script to print the full table:

```bash
python evaluation/scripts/compare_locomo_scores.py
```

### Takeaways

1. **Model and setup**
   - **CML** uses a **local QA model** (`gpt-oss:20b`) with CML as the retrieval/memory backend.
   - Paper baselines use **GPT-4o**, **Gemini**, or **Qwen** (full context or RAG/memory systems with GPT-4o).

2. **Factual memory (LoCoMo)**
   - CML’s factual average (**31.49%**) is:
     - **Below** RAG baselines in the paper (e.g. text-embedding-large **45.32%**, Mem0 **57.24%**, gpt-4o full context **62.99%**).
     - Consistent with a smaller QA model and retrieval-dependent pipeline; stronger backbones in the paper get 37–71% on factual.

3. **Cognitive memory (LoCoMo-Plus)**
   - CML’s Cognitive score (**21.45%**) is:
     - **Above** several paper baselines (e.g. Mem0 15.80%, SeCom 14.90%, RAG 12–15%, smaller Qwen 9–19%).
     - **Below** the best in the paper (e.g. gemini-2.5-pro 26.06%, gemini-2.5-flash 24.67%, gpt-4o 21.05%).
   - Cognitive is hard for everyone; the paper reports a large gap for all methods.

4. **Gap (factual − cognitive)**
   - CML’s gap (**10.04%**) is **smaller** than most paper baselines (roughly 18–45% in the paper).
   - So CML’s relative drop from factual to cognitive is smaller, even though absolute factual performance is lower due to the smaller QA model.

### Reference

- **Locomo-Plus paper:** [arXiv:2602.10715](https://arxiv.org/abs/2602.10715) — *Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents* (Li et al., 2026).
- **Repo:** [github.com/xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus).
