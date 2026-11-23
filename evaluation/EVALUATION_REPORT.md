# CML Evaluation Report: Full LoCoMo-Plus Benchmark (2,387 Samples)

**Date:** 2026-11-23  
**CML Version:** 2.0.0 (Neuro-Cognitive Suite + ActiveCML + Cognitive HyDE/Cue Selector + 2-Hop Bridge + Persona Profile + Chronological Temporal + CA1 Speaker-Swap Dominance)  
**Model:** `qwen3.8-27b` (100% local via `llama-server`, IQ3_S, `temperature=0`, thinking disabled, **`$0.00` API cost**, `0` errors)  
**Committed Artifact:** [`evaluation/results/locomo_plus_2026-11-23_qwen38_full_2387_summary.json`](results/locomo_plus_2026-11-23_qwen38_full_2387_summary.json)  
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

## 2. CML v2.0 Full 2,387-Sample From-Scratch Results (`qwen3.8-27b`, 100% Local)

### Runtime Configuration

| Parameter | Value |
| :--- | :--- |
| Dataset | `unified_input_samples_v2.json` — **2,387 samples** (fresh DB wipe; 411 deduplicated conversations, 242,658 ingested turns in 21m 18s at 189.8 turns/s) |
| Write Path | `FEATURES__USE_LLM_ENABLED=false` during bulk ingestion (deterministic heuristics + Neuro-Cognitive tagging, **0 write-time LLM calls**) |
| Read Path | `FEATURES__USE_LLM_ENABLED=true`, `FEATURES__HYDE_RETRIEVAL_ENABLED=true` (`qwen3.8-27b`), 3-cue Cognitive HyDE + LLM Cognitive Cue Selector, 2-hop Bridge Recollection (`first_pass[:12]`), Speaker Persona Augmentation, Chronological Temporal Ordering (`temporal_contiguity_neighbours=3`), CA1 Top-6 Speaker-Swap Dominance |
| Embeddings | Local `nomic-ai/nomic-embed-text-v2-moe` (768-dim) |
| QA & Judge Model | Local `qwen3.8-27b` (`http://127.0.0.1:8000/v1`, `temperature=0`, `enable_thinking=false`) |
| Token Usage & Cost | QA: `3,441,363` in / `80,632` out · Judge: `667,863` in / `109,489` out · **Total API cost: `$0.00`** |
| Valid / Errors | **2,387 / 2,387 valid (`0` errors, `0` empty contexts)** |

### Per-Category Scores & Diagnostics

| Category | Samples | Correct | Partial | Wrong | Score | **Average** | Refusal Rate | EvRecall@Packet | Empty Ctx | Median Ctx (chars) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single-hop` | 841 | 472 | 74 | 295 | 509.0 | **60.52%** | 19.4% | 71.5% | 0 | 3,762 |
| `multi-hop` | 282 | 53 | 107 | 122 | 106.5 | **37.77%** | 19.9% | 53.9% | 0 | 3,616 |
| `temporal` | 321 | 144 | 34 | 143 | 161.0 | **50.16%** | 24.0% | 75.1% | 0 | 3,959 |
| `common-sense` | 96 | 35 | 7 | 54 | 38.5 | **40.10%** | 24.0% | 46.9% | 0 | 3,890 |
| `adversarial` | 446 | 358 | 0 | 88 | 358.0 | **80.27%** | 80.5% | 63.0% | 0 | 3,670 |
| **LoCoMo Factual Avg** | **1,986** | — | — | — | **1,173.0** | **59.06%** | — | — | **0** | — |
| **`Cognitive` (LoCoMo-Plus)** | **401** | 150 | 0 | 251 | 150.0 | **37.41%** | 1.2% | 56.1% | 0 | 3,268 |
| **Overall (All 6 Categories)** | **2,387** | 1,212 | 222 | 953 | **1,323.0** | **55.43%** | — | — | **0** | — |

- **Gap (LoCoMo Factual Avg − Cognitive):** `59.06% − 37.41% = 21.65pp`

---

## 3. Unified Head-to-Head Comparison on Full LoCoMo-Plus (2,387 Samples)

| System / Method | Category | Backbone / QA Model | Single-hop (841) | Multi-hop (282) | Temporal (321) | Common-sense (96) | Adversarial (446) | **LoCoMo Factual Avg (1,986)** | **Cognitive (401)** | **Gap** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CML v2.0 (Neuro + HyDE + CA1)** <sup>[0]</sup> | Memory System | `qwen3.8-27b` (100% local, `$0` API) | 60.52% | 37.77% | **50.16%** | 40.10% | **80.27%** | **59.06%** | 37.41% | 21.65 |
| **Kumiho Cognitive Memory** <sup>[1]</sup> | Memory System | `gpt-4o` (graph-native) | — | — | — | — | — | — | **93.30%** | — |
| **Madeleine + HyperMem** <sup>[2]</sup> | Memory System | Involuntary recall + HyperMem | — | — | — | — | — | — | 66.60% | — |
| **HyperMem** <sup>[2]</sup> | Memory System | HyperMem baseline | — | — | — | — | — | — | 52.90% | — |
| **Madeleine (standalone)** <sup>[2]</sup> | Memory System | Zero online LLM calls | — | — | — | — | — | — | 52.40% | — |
| **A-Mem** <sup>[3]</sup> | Memory System | `gpt-4o` | 76.90% | **55.60%** | 49.30% | 68.10% | 35.20% | 59.64% | 42.44% | 17.20 |
| **SeCom** <sup>[3]</sup> | Memory System | `gpt-4o` | 77.60% | 50.90% | 42.30% | **71.40%** | 31.80% | 57.53% | 42.63% | 14.90 |
| **Mem0** <sup>[3]</sup> | Memory System | `gpt-4o` | **80.20%** | 48.10% | 39.40% | 66.20% | 30.50% | 57.24% | 41.44% | 15.80 |
| RAG (`text-embedding-large`) <sup>[3]</sup> | RAG Baseline | `gpt-4o` | 49.76% | 22.78% | 40.00% | 21.35% | 59.73% | 45.32% | 29.77% | 15.55 |
| RAG (`text-embedding-small`) <sup>[3]</sup> | RAG Baseline | `gpt-4o` | 39.17% | 17.79% | 34.69% | 14.61% | 51.90% | 37.23% | 24.94% | 12.29 |
| RAG (`text-ada-002`) <sup>[3]</sup> | RAG Baseline | `gpt-4o` | 40.00% | 16.73% | 37.81% | 15.73% | 49.44% | 37.38% | 23.47% | 13.91 |
| Gemini-2.5-Pro <sup>[3]</sup> | Full Context | Full context (1M) | 77.83% | 52.48% | **73.83%** | 63.54% | 73.03% | **71.78%** | 45.72% | 26.06 |
| Gemini-2.5-Flash <sup>[3]</sup> | Full Context | Full context (1M) | 77.71% | 54.26% | 66.04% | 66.67% | 65.84% | 69.25% | 44.58% | 24.67 |
| Qwen2.5-14B-Instruct <sup>[3]</sup> | Full Context | Full context | 76.33% | 48.23% | 38.94% | 57.29% | 68.09% | 63.45% | 44.21% | 19.24 |
| GPT-4o <sup>[3]</sup> | Full Context | Full context | 78.13% | 52.30% | 45.79% | 69.79% | 48.99% | 62.99% | 41.94% | 21.05 |
| GPT-4.1 <sup>[3]</sup> | Full Context | Full context | 80.30% | 53.90% | 58.88% | 72.92% | 37.30% | 62.21% | 43.58% | 18.63 |
| Qwen3-14B <sup>[3]</sup> | Full Context | Full context | 65.96% | 46.45% | 53.89% | 59.38% | 60.45% | 59.65% | 40.56% | 19.09 |
| Qwen3-8B <sup>[3]</sup> | Full Context | Full context | 69.34% | 43.79% | 39.88% | 59.90% | 53.48% | 56.86% | 39.18% | 17.68 |
| GPT-5-nano <sup>[3]</sup> | Full Context | Full context | 75.00% | 54.08% | 50.16% | **73.96%** | 17.53% | 54.96% | 40.12% | 14.84 |
| Qwen3-4B <sup>[3]</sup> | Full Context | Full context | 69.52% | 46.10% | 33.33% | 55.21% | 48.76% | 54.91% | 39.21% | 15.70 |
| Qwen2.5-7B-Instruct <sup>[3]</sup> | Full Context | Full context | 70.72% | 39.54% | 21.81% | 37.50% | 20.22% | 45.31% | 35.74% | 9.57 |
| Qwen2.5-3B-Instruct <sup>[3]</sup> | Full Context | Full context | 68.25% | 38.65% | 18.38% | 48.44% | 11.69% | 42.20% | 31.38% | 10.82 |

<sup>[0]</sup> Full 2,387-sample run from scratch (`qwen3.8-27b` local QA + Judge, `temperature=0`, `$0` API cost).  
<sup>[1]</sup> Self-reported by [Kumiho](https://kumiho.io) on the 401-sample Cognitive split using `gpt-4o`; per-category factual scores are not published.  
<sup>[2]</sup> Reported in the *Madeleine* paper (arXiv 2026) under the official LoCoMo-Plus protocol on the 401-sample Cognitive split; per-category factual scores are not published.  
<sup>[3]</sup> Table 1 of [arXiv:2602.10715](https://arxiv.org/abs/2602.10715) (judge: `gemini-2.5-flash`). **LoCoMo Factual Avg** is the sample-weighted average of the 5 factual categories (1,986 items); **Gap** = LoCoMo Factual Avg − Cognitive.

---

## 4. Unbiased Analysis

### Where CML Excels
1. **Beats Mem0 (`57.24%`) & SeCom (`57.53%`) on LoCoMo Factual Avg at `$0` API Cost:** CML reaches **59.06%** LoCoMo Factual Avg across all 1,986 factual samples (`+1.82pp` over Mem0, `+1.53pp` over SeCom, `+4.10pp` over full-context GPT-5-nano, `+13.74pp` over `RAG text-embedding-large`), within `0.58pp` of A-Mem (`59.64%`), while running 100% locally on a quantized 27B model with zero write-time LLM calls.
2. **#1 Adversarial Robustness (`80.27%`) with Selective Calibration:** Refusal diagnostics prove true discrimination rather than blind over-refusal — **`80.5%`** refusal on adversarial traps vs. only **`19.4–24.0%`** across answerable factual categories.
3. **#1 Memory-System Temporal Reasoning (`50.16%`):** Chronological timeline rendering (`## Recent Events (Chronological Order)`) + `temporal_contiguity_neighbours=3` beats A-Mem (`49.30%`), GPT-4o (`45.79%`), SeCom (`42.30%`), and Mem0 (`39.40%`).
4. **Major Cognitive Recovery (`37.41%`, `150/401`):** 3-cue Cognitive HyDE + LLM Cue Selector beat every RAG baseline (`23.5–29.8%`) by `+7.6pp` to `+13.9pp` and reached within `4.0pp` of Mem0 (`41.44%`).

### Remaining Gaps
1. **Single-hop (`60.52%`), Multi-hop (`37.77%`), and Common-sense (`40.10%`):** Trail GPT-4o-backed memory systems (`76.9–80.2%`, `48.1–55.6%`, `66.2–71.4%`) due to two concrete factors: (a) local quantized 27B IQ3_S vs. GPT-4o answer capacity, and (b) bulk ingestion ran with `FEATURES__USE_LLM_ENABLED=false` (leaving Neo4j entity/relation links empty; `FEATURES__USE_LLM_ENABLED=true` is enabled by default for production deployments).
2. **Judge Parity Note:** Paper Table 1 baselines are judged by `gemini-2.5-flash`; this 100%-local run uses `qwen3.8-27b` (`temperature=0`) for both QA and self-consistent local judging.

---

## 5. Write-Path Throughput & Retrieval Quality Suite

### Write-Path Throughput (measured on `locomo10.json` turns)

| Mode | Mean | p50 | p95 | Sequential Throughput |
| :--- | :---: | :---: | :---: | :---: |
| `FEATURES__USE_LLM_ENABLED=true` (unified LLM extractor — default) | 3.027 s | 2.723 s | 4.427 s | **0.33 turns/s** |
| `FEATURES__USE_LLM_ENABLED=false` (regex/Jaccard + Neuro tagging) | 0.058 s | 0.024 s | 0.038 s | **17.31 turns/s** (~190 turns/s batched on CPU w/ 3 workers) |

### Memory Quality Retrieval Suite (`scripts/test_memory_quality.py`)
- **31 probes across 14 categories:** **100% retrieval recall (`31/31`)**, **100% constraint consistency (`6/6`)**, **77 ms P50 read latency** on local GPU.
