# CML Evaluation Report: Full LoCoMo-Plus Benchmark (2,387 Samples)

**Date:** 2026-11-27  
**CML Version:** 3.0.1+ (`v3.1` Hybrid CLS + Default `LLM_INTERNAL` Whole-Brain Architecture: System-1 Batch Fastpath + Windowed Micro-Sleep Consolidation + Dentate Gyrus Conjunction/IDF BM25 + Bipartite Cortical Hypergraph `<5ms` PPR + Conway 3-Tier SMS + 2-Stage `LLM_INTERNAL` Multi-Angle Expansion & Judged Delivery + `±2` Conversational Contiguity + CA1 Speaker-Attribution & Evidence-Grounding Monitor)  
**Evaluated Configurations:**
1. **Official Table 1 Protocol (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge via OpenRouter, `LLM_INTERNAL` enabled):** `2,387 / 2,387` valid (`0` errors, **`66.65%` Overall**, **`69.99%` LoCoMo Factual Avg**, **`50.12%` Cognitive**, **`91.70%` Adversarial**) — [`evaluation/results/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json`](results/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json)
2. **100% Local Zero-API-Cost (`qwen3.8-27b` QA + Judge via `llama-server` w/ `dflash2`, IQ3_S):** `2,387 / 2,387` valid (`0` errors, **`$0.00`** API cost, **`63.53%` Overall**, **`68.40%` LoCoMo Factual Avg**, **`70.25%` Temporal**) — [`evaluation/results/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json`](results/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json)
3. **Prior `$0`-Local-Store Baseline (`FEATURES__USE_LLM_ENABLED=false` at read time + `gpt-4o` QA):** `2,387 / 2,387` valid (`58.69%` Overall, `63.70%` Factual Avg, `33.92%` Cognitive) — [`evaluation/results/locomo_plus_2026-11-26_cml_v3_gpt4o_gemini25flash_full_2387_summary.json`](results/locomo_plus_2026-11-26_cml_v3_gpt4o_gemini25flash_full_2387_summary.json)

**References:**
- LoCoMo-Plus paper: [arXiv:2602.10715](https://arxiv.org/abs/2602.10715) — *Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents* (Li et al., 2026)
- Official LoCoMo-Plus repo: [github.com/xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus)

---

## 1. Benchmark Overview & Category Definitions

**LoCoMo-Plus** evaluates long-horizon conversational memory across **2,387 unified samples** built from `locomo10.json` and `locomo_plus.json` via [`evaluation/locomo_plus/data/unified_input.py`](locomo_plus/data/unified_input.py), scored with an independent LLM-as-judge (`correct = 1.0`, `partial = 0.5`, `wrong = 0.0`):

| Category | Samples | Level | What the Category Evaluates |
| :--- | :---: | :---: | :--- |
| **`single-hop`** | 841 | Level-1 (LoCoMo) | Direct factual recall of a single event, attribute, quote, or shared photo caption from one session in a ~400–700 turn multi-month dialogue. |
| **`multi-hop`** | 282 | Level-1 (LoCoMo) | Multi-session synthesis requiring linking scattered clues (e.g., `"moved from my home country"` -> `"Sweden"`) or aggregating complete lists/counts across weeks. |
| **`temporal`** | 321 | Level-1 (LoCoMo) | Resolving relative time expressions (`"yesterday"`, `"last week"`, `"last Sunday"`, `"last year"`) against session timestamps, ordering events, and computing durations. |
| **`common-sense`** | 96 | Level-1 (LoCoMo) | Inductive persona & world-knowledge reasoning: combining a speaker's values, career goals, habits, and past experiences to answer hypothetical `"Would / Likely..."` questions. |
| **`adversarial`** | 446 | Level-1 (LoCoMo) | Robustness against false presuppositions and **speaker-swap traps** (asking what Person A did when actually Person B did it, requiring explicit abstention). |
| **`Cognitive` (LoCoMo-Plus)** | 401 | Level-2 (Cognitive) | **Implicit cue–trigger memory awareness:** connecting a casual remark weeks later (*trigger*) to an earlier mishap, scare, injury, allergy, or habit (*cue*) across **zero lexical overlap**. |
| **`Gap`** (`Factual Avg − Cognitive`) | — | Cross-Level | Performance drop from explicit factual recall to implicit cognitive recall (**lower is better**). |

---

## 2. Official Paper Protocol Results (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge)

### Runtime Configuration

| Parameter | Value |
| :--- | :--- |
| Dataset | `unified_input_samples_v2.json` — **2,387 samples** (`411` deduplicated conversations, `242,740` ingested turns with `[Shared photo: ...]` captions preserved and `401/401` Cognitive self-contaminating future trigger turns stripped) |
| Write & Consolidation Path | Hybrid CLS: System-1 Hippocampal batch encoding (`>300 turns/s`, `nomic-embed-text-v2-moe` + Postgres `tsvector` GIN) + `WindowedMicroSleepConsolidator` (`LLM_INTERNAL`: `google/gemini-2.5-flash` w/ CRC32 Content-Defined Chunking + SQLite cache) producing `294,062` active memory records, multi-valued hashed semantic facts, N-ary Cortical Hyperedges, Grounded Proustian Cue Pointers (`cue -> verbatim parent_episode_id`), Conway Tier-2 Chapter Gists, and Tier-3 Persistent DMN Self-Profiles |
| Read Path | 2-Stage `LLM_INTERNAL` (`google/gemini-2.5-flash-lite`) + 4-Pathway Ecphory: Stage-1 multi-angle query expansion over Dense `pgvector` + Conjunction-Boosted IDF `tsvector` BM25 + `<5ms` Bipartite Cortical Hypergraph PPR + Grounded Proustian Cues + `±2` Post-Merge Conversational Contiguity + CUDA Cross-Encoder (`BAAI/bge-reranker-base`) -> Stage-2 batched candidate selection, 2nd-hop bridge fetch, and CA1 Speaker-Swap / Evidence-Grounding monitor (`max_results = 14`) |
| QA Model | **`openai/gpt-4o`** via OpenRouter (`temperature=0`, structured brief CoT + `Final Answer:`) |
| Judge Model | **`google/gemini-2.5-flash`** via OpenRouter (`temperature=0`, `reasoning: {"enabled": false}`) — exact official Table 1 judge |
| Valid / Errors | **2,387 / 2,387 valid (`0` errors, `0` empty contexts)** |

### Per-Category Scores & Diagnostics (`CML v3.1` — `openai/gpt-4o` + `google/gemini-2.5-flash` Judge)

| Category | Samples | Correct | Partial | Wrong | Score | **Average** | Delta vs. v3.0 | Refusal Rate | EvRecall@Packet | Empty Ctx | Median Ctx (chars) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single-hop` | 841 | 529 | 148 | 164 | 603.0 | **71.70%** | **`+8.38pp`** | 9.0% | **89.2%** | 0 | 2,874 |
| `multi-hop` | 282 | 68 | 134 | 80 | 135.0 | **47.87%** | **`+5.67pp`** | 14.9% | **83.3%** | 0 | 2,941 |
| `temporal` | 321 | 191 | 7 | 123 | 194.5 | **60.59%** | **`+2.96pp`** | 14.0% | **85.4%** | 0 | 3,713 |
| `common-sense` | 96 | 44 | 9 | 43 | 48.5 | **50.52%** *(58.85%†)* | **`+11.46pp`** *(+19.79pp†)* | 26.0% | 64.6% | 0 | 3,044 |
| `adversarial` | 446 | 409 | 0 | 37 | 409.0 | **91.70%** | **`+4.03pp`** | 91.3% | 74.0% | 0 | 2,854 |
| **LoCoMo Factual Avg** | **1,986** | — | — | — | **1,390.0** | **69.99%** *(64.48% macro)* | **`+6.29pp`** *(+6.50pp macro)* | — | — | **0** | — |
| **`Cognitive` (LoCoMo-Plus)** | **401** | 201 | 0 | 200 | 201.0 | **50.12%** | **`+16.20pp`** | 0.5% | 62.6% | 0 | 3,843 |
| **Overall (All 6 Categories)** | **2,387** | 1,442 | 298 | 647 | **1,591.0** | **66.65%** | **`+7.96pp` (`+190.0` pts)** | — | — | **0** | — |

- **Gap (`LoCoMo Factual Weighted Avg − Cognitive`):** `69.99% − 50.12% = 19.87pp` (down from `29.78pp`, **`−9.91pp`**); **Macro Gap:** `64.48% − 50.12% = 14.35pp` (down from `24.06pp`, **`−9.71pp`**).
- *†Under the open-inferential QA prompt (`cml_v31_gpt4o_gemini25flash_full_2387`), `common-sense` reached **`58.85%`** (`52` correct, `9` partial, `15.6%` refusal) with a `9.96pp` macro Gap.*

---

## 3. 100%-Local Zero-API-Cost Results (`qwen3.8-27b` QA + Judge)

Evaluated from scratch using local `qwen3.8-27b` (`http://127.0.0.1:8000/v1`, `temperature=0`, `enable_thinking=false`, **`$0.00` API cost**):

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

| System / Method | Category | Backbone / QA Model | Single-hop (841) | Multi-hop (282) | Temporal (321) | Common-sense (96) | Adversarial (446) | **LoCoMo Factual Avg (1,986)** | **Cognitive (401)** | **Gap** ↓ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CML v3.1 (Hybrid CLS + 2-Stage `LLM_INTERNAL`)** <sup>[0a]</sup> | Memory System | `gpt-4o` (judge: `gemini-2.5-flash`) | 71.70% | 47.87% | **60.59%** | 50.52% *(58.85%†)* | **91.70%** | **69.99%** | **50.12%** | 19.87 *(14.35‡)* |
| **CML v3.0 (Whole-Brain, Local `$0` API)** <sup>[0b]</sup> | Memory System | `qwen3.8-27b` (100% local, `$0` API) | 70.21% | 47.52% | **70.25%** | 44.27% | 82.06% | **68.40%** | 39.40% | 29.00 |
| **CML v3.0 (`$0` Local Store + `gpt-4o` QA)** <sup>[0c]</sup> | Memory System | `gpt-4o` (judge: `gemini-2.5-flash`) | 63.32% | 42.20% | 57.63% | 39.06% | 87.67% | 63.70% | 33.92% | 29.78 |
| **Kumiho Cognitive Memory** <sup>[1]</sup> | Memory System | `gpt-4o` (graph-native, 401 only) | — | — | — | — | — | — | 93.30%* | — |
| **Madeleine + HyperMem** <sup>[2]</sup> | Memory System | Involuntary recall + HyperMem (401 only) | — | — | — | — | — | — | 66.60%* | — |
| **HyperMem** <sup>[2]</sup> | Memory System | HyperMem baseline (401 only) | — | — | — | — | — | — | 52.90%* | — |
| **Madeleine (standalone)** <sup>[2]</sup> | Memory System | Zero online LLM calls (401 only) | — | — | — | — | — | — | 52.40%* | — |
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

<sup>[0a]</sup> Official Table 1 protocol (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge via OpenRouter, `temperature=0`, overall 6-category avg **66.65%**, `1,591.0 / 2,387`; †`58.85%` Common-sense under open-inferential QA prompt; ‡unweighted 5-category factual macro mean **64.48%** vs. A-Mem `57.02%`, SeCom `54.80%`, Mem0 `52.88%`, macro Gap **14.35pp**).  
<sup>[0b]</sup> Full 2,387-sample 100%-local run from scratch (`qwen3.8-27b` local QA + Judge, `$0` API cost, overall 6-category avg **63.53%**, `1,516.5 / 2,387`).  
<sup>[0c]</sup> Prior `$0`-local-store baseline (`FEATURES__USE_LLM_ENABLED=false` at read time, overall 6-category avg `58.69%`, `1,401.0 / 2,387`).  
<sup>[1]</sup> *Self-reported by [Kumiho](https://kumiho.io) on the 401-sample Cognitive split only; 1,986-sample factual scores not published.  
<sup>[2]</sup> *Reported in *Madeleine* (arXiv 2026) on the 401-sample Cognitive split only; 1,986-sample factual scores not published.  
<sup>[3]</sup> Table 1 of [arXiv:2602.10715](https://arxiv.org/abs/2602.10715) (judge: `gemini-2.5-flash`). **LoCoMo Factual Avg** is the sample-weighted average of the 5 factual categories (1,986 items); **Gap** = LoCoMo Factual Avg − Cognitive.

---

## 5. Key Achievements & Root-Cause Attribution

1. **#1 Among All Full-Benchmark Memory Systems on BOTH Factual (`69.99%`) AND Cognitive (`50.12%`):**
   - Surpasses **A-Mem** by **`+10.35pp` Factual / `+7.68pp` Cognitive**, **SeCom** by **`+12.46pp` / `+7.49pp`**, and **Mem0** by **`+12.75pp` / `+8.68pp`**.
   - Surpasses **Full-Context GPT-4o (`62.99%` / `41.94%`)**, **Full-Context GPT-4.1 (`62.21%` / `43.58%`)**, and **1M-Context Gemini-2.5-Flash (`69.25%` / `44.58%`)**, and beats **1M-Context Gemini-2.5-Pro (`45.72%`)** by **`+4.40pp`** on `Cognitive` — while injecting median packets of only **`~2,900–3,800` chars (`~800` tokens)**.
2. **#1 Overall Adversarial Robustness (`91.70%`, `409 / 446`):**
   - CA1 Speaker-Attribution monitoring + Stage-2 `speaker_swap` / `unsupported_premise` grounding + explicit speaker-swap refusal alignment lift Adversarial accuracy to **`91.70%`** (`+56.5pp` to `+61.2pp` above A-Mem/SeCom/Mem0) while cutting `single-hop` false refusals from `11.3%` down to **`9.0%`**.
3. **Architectural Drivers of the `+190.0`-Point Gain (`1,401.0 -> 1,591.0`):**
   - **Cognitive (`33.92% -> 50.12%`, `+16.20pp`):** Stripping self-contaminating future `trigger_query` turns during ingestion, restoring dynamic `session_k` boundaries, mapping `A:`/`B:` to speaker names, uncapping Windowed Micro-Sleep **Grounded Proustian Cue Pointers**, and Stage-1/2 `LLM_INTERNAL` cue selection.
   - **Single-hop (`63.32% -> 71.70%`, `+8.38pp`) & Multi-hop (`42.20% -> 47.87%`, `+5.67pp`):** Preserving `1,306` `[Shared photo: <blip_caption>]` descriptions + `164` multi-line turns, removing `RetrievalPlanner`'s `skip_if_found=True` vector-skip trap, conjunction-boosted IDF `tsvector` ranking, `±2` post-merge conversational contiguity, and multi-valued `FactStore` key hashing (`89.2%` `single-hop` and `83.3%` `multi-hop` packet recall).
   - **Common-sense (`39.06% -> 50.52%` / `58.85%` open, `+11.46–19.79pp`):** Replacing noisy regex presupposition warnings with Stage-2 `"inferable"` grounding + Conway Tier-3 DMN persona profiles.
   - **Temporal (`57.63% -> 60.59%`, `+2.96pp`):** Hippocampal Time Cell granularity preservation + canonical LoCoMo relative-date resolution (`The week before D Month YYYY`).

---

## 6. Write-Path Throughput & Retrieval Quality Suite

| Mode | Throughput / Latency | Notes |
| :--- | :---: | :--- |
| Hybrid CLS Batch Write (`FEATURES__BATCH_WRITE_SYSTEM1_FASTPATH=true`) | **`>300 turns/s`** (`0.024 s` p50) | Default for `/memory/write_batch`: GPU batch embedding + `tsvector` + neuro tagging |
| Windowed Micro-Sleep Consolidation (`consolidate_windowed`) | **18–30 turn CRC32 windows** (~18× fewer LLM calls than per-turn) | Content-defined chunking + persistent SQLite extraction cache (`>90%` cache hit across variants) |
| Live Single-Turn Write (`FEATURES__USE_LLM_ENABLED=true`) | **~0.35 turns/s** | Online System-2 `LLM_INTERNAL` extraction on `/memory/write` |
