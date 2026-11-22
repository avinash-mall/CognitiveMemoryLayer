# CML Evaluation Report: Full LoCoMo-Plus Benchmark (2,387 Samples)

**Date:** 2026-11-22  
**CML Version:** 2.0.0 (Neuro-Cognitive Suite + ActiveCML + Laya System 1)  
**QA & Judge Model:** `openai/gpt-4o-mini` via OpenRouter (`temperature=0`, `0` errors)  
**Committed Artifact:** [`evaluation/results/locomo_plus_2026-11-22_gpt4omini_full_2387_summary.json`](results/locomo_plus_2026-11-22_gpt4omini_full_2387_summary.json)  
**References:**
- LoCoMo-Plus paper: [arXiv:2602.10715](https://arxiv.org/abs/2602.10715) — *Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents* (Li et al., 2026)
- Official LoCoMo-Plus repo: [github.com/xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus)

---

## 1. Benchmark Overview

**LoCoMo-Plus** evaluates long-horizon conversational memory across **2,387 unified samples** built from `locomo10.json` and `locomo_plus.json` via [`evaluation/locomo_plus/data/unified_input.py`](locomo_plus/data/unified_input.py):

- **Level-1 — LoCoMo Factual (1,986 samples across 5 categories):**
  - `single-hop` (841), `adversarial` (446), `temporal` (321), `multi-hop` (282), `common-sense` (96)
- **Level-2 — LoCoMo-Plus Cognitive (401 samples):**
  - Implicit cue–trigger constraint memory (causal, state, goal, value) under semantic disconnect, scored with LLM-as-judge (`correct=1`, `partial=0.5`, `wrong=0`).

---

## 2. CML v2.0 Full 2,387-Sample Results (`openai/gpt-4o-mini`)

### Runtime Configuration

| Parameter | Value |
| :--- | :--- |
| Dataset | `unified_input_samples_v2.json` — **2,387 samples** (411 deduplicated conversations, 242,658 ingested turns) |
| Write Path | `FEATURES__USE_LLM_ENABLED=false` (deterministic heuristics + Neuro-Cognitive tagging, **0 write-time LLM calls**) |
| Embeddings | Local GPU `nomic-ai/nomic-embed-text-v2-moe` (768-dim, `batch_size=64`) |
| Neuro-Cognitive Flags | Somatic Markers, Baddeley Working Memory, BA10 Prospective Memory, CA1 Reality Monitoring, DMN Theory-of-Mind, Ecphory Reactivation, ActiveCML, Adaptive Episode Budget, **Laya System 1** (`convaiinnovations/laya`) — all `true` |
| QA & Judge Model | `openai/gpt-4o-mini` (OpenRouter, `temperature=0`, 24 concurrent workers) |
| Token Usage & Cost | QA: `2,451,927` in / `51,030` out · Judge: `601,712` in / `69,969` out · **Total API cost: `$0.5248`** |
| Valid / Errors | **2,387 / 2,387 valid (`0` errors)** |

### Per-Category Scores & Diagnostics

| Category | Samples | Correct | Partial | Wrong | Score | **Average** | Refusal Rate | Empty Ctx | Median Ctx (chars) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single-hop` | 841 | 296 | 288 | 257 | 440.0 | **52.32%** | 21.6% | 0 | 2,986 |
| `multi-hop` | 282 | 35 | 142 | 105 | 106.0 | **37.59%** | 24.5% | 0 | 2,920 |
| `temporal` | 321 | 104 | 50 | 167 | 129.0 | **40.19%** | 30.8% | 0 | 3,379 |
| `common-sense` | 96 | 30 | 23 | 43 | 41.5 | **43.23%** | 42.7% | 0 | 3,005 |
| `adversarial` | 446 | 408 | 0 | 38 | 408.0 | **91.48%** | 56.3% | 0 | 3,093 |
| **LoCoMo Factual Avg** | **1,986** | — | — | — | **1,124.5** | **56.62%** | — | **0** | — |
| **`Cognitive` (LoCoMo-Plus)** | **401** | 98 | 0 | 303 | 98.0 | **24.44%** | 1.5% | 0 | 2,565 |
| **Overall (All 6 Categories)** | **2,387** | 971 | 503 | 913 | **1,222.5** | **51.21%** | — | **0** | — |

- **Gap (LoCoMo Factual Avg − Cognitive):** `56.62% − 24.44% = 32.18pp`

---

## 3. Head-to-Head Comparison Across Published LoCoMo-Plus Systems

All paper baselines are from **Table 1** of [arXiv:2602.10715](https://arxiv.org/abs/2602.10715) (judge: `gemini-2.5-flash`). **LoCoMo Factual Avg** is the sample-weighted average of the 5 factual categories (1,986 items); **Cognitive** is the 401-item LoCoMo-Plus split.

### Memory Systems & RAG Baselines

| System | QA Model | Single-hop | Multi-hop | Temporal | Common-sense | Adversarial | **LoCoMo Factual Avg** | **Cognitive** | **Gap** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CML v2.0 (Neuro + Laya)** | `gpt-4o-mini` (heuristic write) | 52.32% | 37.59% | 40.19% | 43.23% | **91.48%** | **56.62%** | 24.44% | 32.18 |
| **Kumiho Cognitive Memory** <sup>[1]</sup> | `gpt-4o` (graph-native) | — | — | — | — | — | — | **93.30%** | — |
| **A-Mem** | `gpt-4o` | 76.90% | **55.60%** | **49.30%** | 68.10% | 35.20% | **59.64%** | 42.44% | 17.20 |
| **SeCom** | `gpt-4o` | 77.60% | 50.90% | 42.30% | **71.40%** | 31.80% | 57.53% | **42.63%** | 14.90 |
| **Mem0** | `gpt-4o` | **80.20%** | 48.10% | 39.40% | 66.20% | 30.50% | 57.24% | 41.44% | 15.80 |
| RAG (`text-embedding-large`) | `gpt-4o` | 49.76% | 22.78% | 40.00% | 21.35% | 59.73% | 45.32% | 29.77% | 15.55 |
| RAG (`text-embedding-small`) | `gpt-4o` | 39.17% | 17.79% | 34.69% | 14.61% | 51.90% | 37.23% | 24.94% | 12.29 |
| RAG (`text-ada-002`) | `gpt-4o` | 40.00% | 16.73% | 37.81% | 15.73% | 49.44% | 37.38% | 23.47% | 13.91 |

<sup>[1]</sup> Self-reported by [Kumiho](https://kumiho.io) on the 401-sample Cognitive split using GPT-4o; per-category factual scores are not published.

### Full-Context LLMs (Paper Table 1 — No Retrieval)

| Model | Single-hop | Multi-hop | Temporal | Common-sense | Adversarial | **LoCoMo Factual Avg** | **Cognitive** | **Gap** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Gemini-2.5-Pro | 77.83% | 52.48% | **73.83%** | 63.54% | **73.03%** | **71.78%** | **45.72%** | 26.06 |
| Gemini-2.5-Flash | 77.71% | 54.26% | 66.04% | 66.67% | 65.84% | 69.25% | 44.58% | 24.67 |
| Qwen2.5-14B-Instruct | 76.33% | 48.23% | 38.94% | 57.29% | 68.09% | 63.45% | 44.21% | 19.24 |
| GPT-4o | 78.13% | 52.30% | 45.79% | 69.79% | 48.99% | 62.99% | 41.94% | 21.05 |
| GPT-4.1 | **80.30%** | 53.90% | 58.88% | 72.92% | 37.30% | 62.21% | 43.58% | 18.63 |
| Qwen3-14B | 65.96% | 46.45% | 53.89% | 59.38% | 60.45% | 59.65% | 40.56% | 19.09 |
| Qwen3-8B | 69.34% | 43.79% | 39.88% | 59.90% | 53.48% | 56.86% | 39.18% | 17.68 |
| GPT-5-nano | 75.00% | 54.08% | 50.16% | **73.96%** | 17.53% | 54.96% | 40.12% | 14.84 |
| Qwen3-4B | 69.52% | 46.10% | 33.33% | 55.21% | 48.76% | 54.91% | 39.21% | 15.70 |
| Qwen2.5-7B-Instruct | 70.72% | 39.54% | 21.81% | 37.50% | 20.22% | 45.31% | 35.74% | 9.57 |
| Qwen2.5-3B-Instruct | 68.25% | 38.65% | 18.38% | 48.44% | 11.69% | 42.20% | 31.38% | 10.82 |

---

## 4. Unbiased Analysis

### Where CML Excels
1. **Competitive Factual Recall at Ultra-Low Cost:** CML with `gpt-4o-mini` and zero write-time LLM calls (`$0.52` total) achieves **56.62%** LoCoMo Factual Avg — outperforming `RAG (text-embedding-large, GPT-4o)` by **+11.30pp** (`45.32%`) and full-context `GPT-5-nano` by **+1.66pp** (`54.96%`), and landing within **0.62pp of Mem0 (GPT-4o)** (`57.24%`).
2. **Adversarial False-Premise Resistance (`91.48%`):** CA1 Reality Monitoring + strict grounding instructions cut hallucinations dramatically (`408/446`), ahead of Gemini-2.5-Pro (`73.03%`), GPT-4o (`48.99%`), A-Mem (`35.20%`), and Mem0 (`30.50%`).
3. **Zero Empty Retrievals & Zero Errors:** `0/2,387` empty contexts (median 2,565–3,379 chars) and `0/2,387` pipeline errors across 242,658 ingested turns.

### Where CML Has Clear Gaps (Honest Assessment)
1. **Over-Cautious Refusal Trade-Off on Factual & Common-Sense Queries:** Refusal diagnostics show `gpt-4o-mini` refused `56.3%` of adversarial questions (helpful), but also refused `42.7%` of `common-sense`, `30.8%` of `temporal`, `24.5%` of `multi-hop`, and `21.6%` of `single-hop` questions when answers required inference beyond verbatim text. Relaxing refusal wording for non-adversarial intents is an immediate lever to recover factual recall.
2. **Cognitive (`24.44%` on all 401 samples) under Heuristic Write Path:** Because `FEATURES__USE_LLM_ENABLED=false` performs zero write-time LLM extraction, the Neo4j entity-relation graph is not populated and implicit constraints without explicit lexical markers rely solely on dense vector similarity. That puts Cognitive in line with `RAG (text-embedding-small)` (`24.94%`) and behind GPT-4o memory systems (`41.4–42.6%`) and Kumiho (`93.3%`).
3. **Controlled Paired A/B & Reasoning-Model Evidence:** The Neuro-Cognitive + Laya System 1 subsystems themselves provide verified lift when holding the answerer constant:
   - **60-sample paired A/B (`z-ai/glm-5.3-flash`, same corpus & judge):** CML Baseline `59.17%` → CML + Neuro + Laya **`61.67%` (`+2.50pp`)**, with **`Cognitive` `+20.0pp`** (`40.0% → 60.0%`).
   - **677-sample stratified subset (`z-ai/glm-5.3-flash:batch`, `$0.0246`):** **`54.80%` overall** (`single-hop 62.80%`, `adversarial 61.59%`, `temporal 50.52%`, `Cognitive 47.50%`, `common-sense 46.25%`, `multi-hop 35.00%`).

---

## 5. Write-Path Throughput & Retrieval Quality Suite

### Write-Path Throughput (measured on `locomo10.json` turns)

| Mode | Mean | p50 | p95 | Sequential Throughput |
| :--- | :---: | :---: | :---: | :---: |
| `FEATURES__USE_LLM_ENABLED=true` (unified LLM extractor) | 3.027 s | 2.723 s | 4.427 s | **0.33 turns/s** |
| `FEATURES__USE_LLM_ENABLED=false` (regex/Jaccard + Neuro tagging) | 0.058 s | 0.024 s | 0.038 s | **17.31 turns/s** (~200–350 turns/s batched w/ 3 workers) |

### Memory Quality Retrieval Suite (`scripts/test_memory_quality.py`)
- **31 probes across 14 categories:** **100% retrieval recall (`31/31`)**, **100% constraint consistency (`6/6`)**, **77 ms P50 read latency** on local GPU.

---

## 6. Historical Run Archive

| Date | Samples | QA / Judge Model | LoCoMo Factual Avg | Cognitive | Overall | Artifact |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **2026-11-22 (Current)** | **2,387** | `openai/gpt-4o-mini` / `openai/gpt-4o-mini` | **56.62%** | **24.44%** | **51.21%** | [`locomo_plus_2026-11-22_gpt4omini_full_2387_summary.json`](results/locomo_plus_2026-11-22_gpt4omini_full_2387_summary.json) |
| 2026-11-22 (Subset) | 677 | `z-ai/glm-5.3-flash:batch` / same | 55.26% | 47.50% | 54.80% | `evaluation/outputs/glm53_batch_full_laya_neuro/` (local) |
| 2026-08-04 (Historical) | 2,387 | `gpt-5.6-luna-pro` / `deepseek-v4-flash` | 54.83% | 38.16% | 52.03% | [`locomo_plus_2026-08-04_external-models-luna-deepseek_summary.json`](results/locomo_plus_2026-08-04_external-models-luna-deepseek_summary.json) |
| 2026-08-02 (Historical) | 2,387 | Local Qwen / Local Qwen | 53.27% | 25.44% | 48.60% | [`locomo_plus_2026-08-02_summary.json`](results/locomo_plus_2026-08-02_summary.json) |
