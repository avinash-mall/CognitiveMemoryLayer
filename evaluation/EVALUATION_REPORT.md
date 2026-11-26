# CML Evaluation Report: Full LoCoMo-Plus Benchmark (2,387 Samples)

**Date:** 2026-11-26  
**CML Version:** 3.0.0 (Whole-Brain Architecture: Windowed Micro-Sleep Consolidation + Dentate Gyrus BM25 + Bipartite Cortical Hypergraph `<5ms` PPR + Conway 3-Tier SMS + Granularity-Aware Hippocampal Time Cells + GPU Cross-Encoder + ACC 2-Hop Recollection + LC-NE/ACh Neuromodulation + CA1 Speaker-Swap Dominance)  
**Evaluated Configurations:**
1. **Official Table 1 Protocol (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge via OpenRouter):** `2,387 / 2,387` valid (`0` errors, **`$11.04`** total API cost, `$0.00` local memory engine cost) — [`evaluation/results/locomo_plus_2026-11-26_cml_v3_gpt4o_gemini25flash_full_2387_summary.json`](results/locomo_plus_2026-11-26_cml_v3_gpt4o_gemini25flash_full_2387_summary.json)
2. **100% Local Zero-API-Cost (`qwen3.8-27b` QA + Judge via `llama-server` w/ `dflash2`, IQ3_S):** `2,387 / 2,387` valid (`0` errors, **`$0.00`** API cost) — [`evaluation/results/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json`](results/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json)

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

## 2. Official Paper Protocol Results (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge)

### Runtime Configuration

| Parameter | Value |
| :--- | :--- |
| Dataset | `unified_input_samples_v2.json` — **2,387 samples** (411 deduplicated conversations, 242,658 ingested turns + CRC32 Content-Defined Windowed Micro-Sleep Consolidation across all 411 tenants) |
| Write & Consolidation Path | 100% local (`$0.00`): `nomic-embed-text-v2-moe` + Postgres `tsvector` GIN + `WindowedMicroSleepConsolidator` (`qwen3.8-27b`) producing `56,567` coref-resolved semantic facts, `26,300` N-ary Cortical Hyperedges, Grounded Proustian Cue Pointers (`cue -> verbatim parent_episode_id`), Conway Tier-2 Chapter Gists, and `924` Tier-3 Persistent DMN Self-Profiles |
| Read Path | 100% local (`$0.00`): 4-Pathway Ecphory (Dense pgvector + Dentate Gyrus `tsvector` BM25 + `<5ms` Bipartite Cortical Hypergraph PPR + Grounded Proustian Cues) + Synaptic Co-Activation Resonance + CUDA True-Logit Cross-Encoder (`BAAI/bge-reranker-base`) + Granularity-Aware Hippocampal Time Cell Engine + ACC Iterative 2-Hop Recollection + CLS Stratified Slot Allocation + CA1 Top-6 Speaker-Swap Dominance |
| QA Model | **`openai/gpt-4o`** via OpenRouter (`temperature=0`, `max_tokens=200`) |
| Judge Model | **`google/gemini-2.5-flash`** via OpenRouter (`temperature=0`, `reasoning: {"enabled": false}`, `max_tokens=160`) — exact official Table 1 judge |
| Token Usage & Cost | QA: `4,040,311` in / `48,254` out (`$10.5833`) · Judge: `630,125` in / `107,049` out (`$0.4567`) · **Total OpenRouter cost: `$11.0400`** |
| Valid / Errors | **2,387 / 2,387 valid (`0` errors, `0` empty contexts)** |

### Per-Category Scores & Diagnostics (`openai/gpt-4o` + `google/gemini-2.5-flash` Judge)

| Category | Samples | Correct | Partial | Wrong | Score | **Average** | Refusal Rate | EvRecall@Packet | Empty Ctx | Median Ctx (chars) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single-hop` | 841 | 470 | 125 | 246 | 532.5 | **63.32%** | 11.3% | 71.2% | 0 | 4,767 |
| `multi-hop` | 282 | 59 | 120 | 103 | 119.0 | **42.20%** | 14.5% | 62.4% | 0 | 4,758 |
| `temporal` | 321 | 182 | 6 | 133 | 185.0 | **57.63%** | 18.4% | 76.3% | 0 | 5,639 |
| `common-sense` | 96 | 34 | 7 | 55 | 37.5 | **39.06%** | 36.5% | 53.3% | 0 | 4,772 |
| `adversarial` | 446 | 391 | 0 | 55 | 391.0 | **87.67%** | 84.8% | 59.0% | 0 | 4,660 |
| **LoCoMo Factual Avg** | **1,986** | — | — | — | **1,265.0** | **63.70%** *(57.98% macro)* | — | — | **0** | — |
| **`Cognitive` (LoCoMo-Plus)** | **401** | 136 | 0 | 265 | 136.0 | **33.92%** | 0.0% | 54.4% | 0 | 6,294 |
| **Overall (All 6 Categories)** | **2,387** | 1,272 | 258 | 857 | **1,401.0** | **58.69%** | — | — | **0** | — |

- **Gap (LoCoMo Factual Weighted Avg − Cognitive):** `63.70% − 33.92% = 29.78pp` (`24.06pp` using unweighted 5-category macro mean `57.98%`)

---

## 3. 100%-Local Zero-API-Cost Results (`qwen3.8-27b` QA + Judge)

Evaluated on the identical 2,387 retrieved CML v3.0 memory packets using local `qwen3.8-27b` (`http://127.0.0.1:8000/v1`, `temperature=0`, `enable_thinking=false`, **`$0.00` API cost**, QA: `4,477,084` in / `76,537` out · Judge: `665,975` in / `111,371` out):

| Category | Samples | Correct | Partial | Wrong | Score | **Average** | Refusal Rate | EvRecall@Packet | Empty Ctx | Median Ctx (chars) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single-hop` | 841 | 546 | 89 | 206 | 590.5 | **70.21%** | 10.1% | 71.2% | 0 | 4,767 |
| `multi-hop` | 282 | 73 | 122 | 87 | 134.0 | **47.52%** | 10.6% | 62.4% | 0 | 4,758 |
| `temporal` | 321 | 215 | 21 | 85 | 225.5 | **70.25%** | 11.2% | 76.3% | 0 | 5,639 |
| `common-sense` | 96 | 37 | 11 | 48 | 42.5 | **44.27%** | 21.9% | 53.3% | 0 | 4,772 |
| `adversarial` | 446 | 366 | 0 | 80 | 366.0 | **82.06%** | 83.4% | 59.0% | 0 | 4,660 |
| **LoCoMo Factual Avg** | **1,986** | — | — | — | **1,358.5** | **68.40%** | — | — | **0** | — |
| **`Cognitive` (LoCoMo-Plus)** | **401** | 158 | 0 | 243 | 158.0 | **39.40%** | 0.0% | 54.4% | 0 | 6,294 |
| **Overall (All 6 Categories)** | **2,387** | 1,395 | 243 | 749 | **1,516.5** | **63.53%** | — | — | **0** | — |

---

## 4. Unified Head-to-Head Comparison on Full LoCoMo-Plus (2,387 Samples)

| System / Method | Category | Backbone / QA Model | Single-hop (841) | Multi-hop (282) | Temporal (321) | Common-sense (96) | Adversarial (446) | **LoCoMo Factual Avg (1,986)** | **Cognitive (401)** | **Gap** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CML v3.0 (Whole-Brain, Official Judge)** <sup>[0a]</sup> | Memory System | `gpt-4o` (judge: `gemini-2.5-flash`) | 63.32% | 42.20% | **57.63%** | 39.06% | **87.67%** | **63.70%** | 33.92% | 29.78 |
| **CML v3.0 (Whole-Brain, Local)** <sup>[0b]</sup> | Memory System | `qwen3.8-27b` (100% local, `$0` API) | 70.21% | 47.52% | **70.25%** | 44.27% | 82.06% | **68.40%** | 39.40% | 29.00 |
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

<sup>[0a]</sup> Official Table 1 protocol (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge via OpenRouter, `temperature=0`, `$11.04` API cost, overall 6-category avg **58.69%**, `1,401.0 / 2,387`; unweighted 5-category factual macro mean **57.98%** vs. A-Mem `57.02%`, SeCom `54.80%`, Mem0 `52.88%`). Write/consolidation and retrieval remained 100% local (`$0`).  
<sup>[0b]</sup> Full 2,387-sample 100%-local run from scratch (`qwen3.8-27b` local QA + Judge, `temperature=0`, `$0` API cost, overall 6-category avg **63.53%**, `1,516.5 / 2,387`).  
<sup>[1]</sup> Self-reported by [Kumiho](https://kumiho.io) on the 401-sample Cognitive split using `gpt-4o`; per-category factual scores are not published.  
<sup>[2]</sup> Reported in the *Madeleine* paper (arXiv 2026) under the official LoCoMo-Plus protocol on the 401-sample Cognitive split; per-category factual scores are not published.  
<sup>[3]</sup> Table 1 of [arXiv:2602.10715](https://arxiv.org/abs/2602.10715) (judge: `gemini-2.5-flash`). **LoCoMo Factual Avg** is the sample-weighted average of the 5 factual categories (1,986 items); **Gap** = LoCoMo Factual Avg − Cognitive.

---

## 5. Unbiased Analysis

### Where CML v3.0 Excels Under Official `gpt-4o` + `gemini-2.5-flash` Evaluation
1. **#1 Among All `gpt-4o` Memory Systems on LoCoMo Factual Avg (`63.70%` weighted / `57.98%` macro) — Beats Full-Context GPT-4o (`62.99%`) & GPT-4.1 (`62.21%`):** Under the exact Table 1 `gpt-4o` + `gemini-2.5-flash` protocol, CML v3.0 surpasses **A-Mem (`59.64%`, `+4.06pp`)**, **SeCom (`57.53%`, `+6.17pp`)**, **Mem0 (`57.24%`, `+6.46pp`)**, **RAG `text-embedding-large` (`45.32%`, `+18.38pp`)**, **full-context GPT-4o (`62.99%`, `+0.71pp`)**, and **full-context GPT-4.1 (`62.21%`, `+1.49pp`)** — while performing all write-time ingestion, consolidation, and retrieval 100% locally (`$0` internal memory cost vs. `gpt-4o` indexing in A-Mem/SeCom/Mem0).
2. **#1 Overall Adversarial Robustness (`87.67%`):** CA1 Orbitofrontal Reality Monitoring + true-logit CUDA Cross-Encoder discrimination paired with `gpt-4o` reach **`87.67%`** (`391 / 446`) on adversarial traps — outperforming **A-Mem (`35.20%`) by `+52.47pp`**, **Mem0 (`30.50%`) by `+57.17pp`**, **full-context GPT-4o (`48.99%`) by `+38.68pp`**, and **1M-context Gemini-2.5-Pro (`73.03%`) by `+14.64pp`**.
3. **#1 `gpt-4o` Memory-System Temporal Reasoning (`57.63%`, and `70.25%` local):** Granularity-Aware Hippocampal Time Cells + `HippocampalTimeCellEngine` deterministic calendar deltas achieve **`57.63%`** with `gpt-4o` + `gemini-2.5-flash` (`+8.33pp` over A-Mem `49.30%`, `+15.33pp` over SeCom `42.30%`, `+18.23pp` over Mem0 `39.40%`, and `+11.84pp` over full-context GPT-4o `45.79%`).

### Controlled Diagnosis: Why `gpt-4o` + `gemini-2.5-flash` Differs from `qwen3.8-27b` on Identical Packets
Holding all 2,387 retrieved CML v3.0 memory packets strictly identical isolates three concrete effects:
1. **`gpt-4o` RLHF Epistemic Abstention on Inferential Prompts:** `gpt-4o` abstains substantially more often when a conclusion requires inductive inference rather than explicit statement: refusal rises from `21.9%` → **`36.5%`** (`+14.6pp`) on `common-sense`, `11.2%` → **`18.4%`** (`+7.2pp`) on `temporal`, and `10.6%` → **`14.5%`** (`+3.9pp`) on `multi-hop` (while lifting `adversarial` from `82.06%` → **`87.67%`**). Tuning CA1 refusal instructions specifically for frontier RLHF caution would recover several points on `common-sense`, `single-hop`, and `temporal`.
2. **Stricter Independent `gemini-2.5-flash` Judge Calibration:** On `single-hop` (where refusal barely moved: `11.3%` vs `10.1%`), `gemini-2.5-flash` graded `125` answers `partial` (`0.5`) vs `89` under local `qwen3.8-27b`, and on `temporal` awarded only `6` `partial` credits vs `21`, penalizing approximate date phrasing more strictly.
3. **Local Quantized Consolidation (`$0`) vs. Cloud `gpt-4o` Consolidation:** The underlying consolidated artifacts (`semantic_facts`, `hyperedge_engrams`, `proustian_pointers`, `dmn_self_profiles`) were extracted once by quantized local `qwen3.8-27b` (`IQ3_S`, `$0`) rather than `gpt-4o` (`~$22.70` consolidation cost). Mem0, SeCom, and A-Mem use `gpt-4o` during both memory construction and QA, explaining most of the remaining gap on `Cognitive` (`33.92%` vs `41.44–42.63%`) and `single-hop` (`63.32%` vs `76.90–80.20%`).

---

## 6. Write-Path Throughput & Retrieval Quality Suite

### Ingestion & Consolidation Throughput (measured on `unified_input_samples_v2.json`)

| Mode | Throughput / Latency | Notes |
| :--- | :---: | :--- |
| Fast Hippocampal Write (`FEATURES__USE_LLM_ENABLED=false`) | **~190 turns/s** (`0.024 s` p50) | One-shot pgvector + `tsvector` + deterministic Neuro tagging |
| Windowed Micro-Sleep Consolidation (`consolidate_windowed`) | **24-turn CRC32 windows** (~16× fewer LLM calls than per-turn) | Content-defined chunking + content-addressable SQLite extraction cache |
| Per-Turn Unified LLM Write (`FEATURES__USE_LLM_ENABLED=true`) | **0.33 turns/s** (`2.723 s` p50) | Online single-turn LLM extraction |

### Memory Quality Retrieval Suite (`scripts/test_memory_quality.py`)
- **31 probes across 14 categories:** **100% retrieval recall (`31/31`)**, **100% constraint consistency (`6/6`)**, **`<5ms`** cortical hypergraph PPR traversal.
