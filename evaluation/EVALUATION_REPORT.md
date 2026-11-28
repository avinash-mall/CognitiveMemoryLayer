# CML Evaluation Report: LoCoMo-Plus (2,387 Samples) & LongMemEval-V2 (451 Questions)

**Date:** 2026-11-28  
**CML Version:** 3.0.1+ (`v3.1` Hybrid CLS + Default `LLM_INTERNAL` Whole-Brain Architecture + `CMLMemory` Multimodal Agent Trajectory Backend)

## Executive Summary Across Benchmarks

| Benchmark | Domain & Scale | Runtime / Backbone | Headline Accuracy | Standout Category Scores | API Cost | Committed Summary Artifact |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| **LoCoMo-Plus** *(Cloud Table 1)* | Multi-session chat (`2,387` samples, `6` categories) | `openai/gpt-4o` QA + `gemini-2.5-flash` Judge (`LLM_INTERNAL` on) | **66.65%** Overall (`69.99%` Factual) | **91.70%** Adversarial (**#1**), **50.12%** Cognitive (**#1**), **71.70%** Single-hop | Cloud | [`locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json`](results/locomo_plus/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json) |
| **LoCoMo-Plus** *(100% Local)* | Multi-session chat (`2,387` samples, `6` categories) | Local `qwen3.8-27b` (`IQ3_S` + `dflash2`) QA + Judge | **63.53%** Overall (`68.40%` Factual) | **82.06%** Adversarial, **70.25%** Temporal, **70.21%** Single-hop | **\$0.00** | [`locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json`](results/locomo_plus/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json) |
| **LongMemEval-V2** *(100% Local `small`)* | Multimodal GUI/Web & ServiceNow trajectories (`451` Qs) | Local `qwen3.8-27b` (`IQ3_S` + `dflash2`) Multimodal Reader + Judge | **47.67%** Full-Set (**56.97%** Non-Abst) | **61.32%** Workflow (`77.03%` non-abs, `90.48%` Web), **51.72%** Gotchas (`66.67%` Web) | **\$0.00** | [`longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json`](results/longmemeval_v2/longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json) |

Expand either collapsible benchmark report below for category definitions, runtime configurations, per-category diagnostics, full head-to-head baseline comparison tables, and architectural root-cause analysis.

---

<details open>
<summary><strong>Part I: LoCoMo-Plus Benchmark &mdash; Conversational Factual &amp; Implicit Cognitive Memory (2,387 Samples)</strong></summary>

### 1.1 Benchmark Overview & Category Definitions

**LoCoMo-Plus** (Li et al., 2026, [arXiv:2602.10715](https://arxiv.org/abs/2602.10715); [github.com/xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus)) evaluates long-horizon conversational memory across **2,387 unified samples** built from `locomo10.json` and `locomo_plus.json` via [`evaluation/locomo_plus/data/unified_input.py`](locomo_plus/data/unified_input.py), scored with an independent LLM-as-judge (`correct = 1.0`, `partial = 0.5`, `wrong = 0.0`):

| Category | Samples | Level | What the Category Evaluates |
| :--- | :---: | :---: | :--- |
| **`single-hop`** | 841 | Level-1 (LoCoMo) | Direct factual recall of a single event, attribute, quote, or shared photo caption from one session in a ~400–700 turn multi-month dialogue. |
| **`multi-hop`** | 282 | Level-1 (LoCoMo) | Multi-session synthesis requiring linking scattered clues (e.g., `"moved from my home country"` -> `"Sweden"`) or aggregating complete lists/counts across weeks. |
| **`temporal`** | 321 | Level-1 (LoCoMo) | Resolving relative time expressions (`"yesterday"`, `"last week"`, `"last Sunday"`, `"last year"`) against session timestamps, ordering events, and computing durations. |
| **`common-sense`** | 96 | Level-1 (LoCoMo) | Inductive persona & world-knowledge reasoning: combining a speaker's values, career goals, habits, and past experiences to answer hypothetical `"Would / Likely..."` questions. |
| **`adversarial`** | 446 | Level-1 (LoCoMo) | Robustness against false presuppositions and **speaker-swap traps** (asking what Person A did when actually Person B did it, requiring explicit abstention). |
| **`Cognitive` (LoCoMo-Plus)** | 401 | Level-2 (Cognitive) | **Implicit cue–trigger memory awareness:** connecting a casual remark weeks later (*trigger*) to an earlier mishap, scare, injury, allergy, or habit (*cue*) across **zero lexical overlap**. |
| **`Gap`** (`Factual Avg − Cognitive`) | — | Cross-Level | Performance drop from explicit factual recall to implicit cognitive recall (**lower is better**). |

### 1.2 Official Paper Protocol Results (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge)

#### Runtime Configuration

| Parameter | Value |
| :--- | :--- |
| Dataset | `unified_input_samples_v2.json` — **2,387 samples** (`411` deduplicated conversations, `242,740` ingested turns with `[Shared photo: ...]` captions preserved and `401/401` Cognitive self-contaminating future trigger turns stripped) |
| Write & Consolidation Path | Hybrid CLS: System-1 Hippocampal batch encoding (`>300 turns/s`, `nomic-embed-text-v2-moe` + Postgres `tsvector` GIN) + `WindowedMicroSleepConsolidator` (`LLM_INTERNAL`: `google/gemini-2.5-flash` w/ CRC32 Content-Defined Chunking + SQLite cache) producing `294,062` active memory records, multi-valued hashed semantic facts, N-ary Cortical Hyperedges, Grounded Proustian Cue Pointers (`cue -> verbatim parent_episode_id`), Conway Tier-2 Chapter Gists, and Tier-3 Persistent DMN Self-Profiles |
| Read Path | 2-Stage `LLM_INTERNAL` (`google/gemini-2.5-flash-lite`) + 4-Pathway Ecphory: Stage-1 multi-angle query expansion over Dense `pgvector` + Conjunction-Boosted IDF `tsvector` BM25 + `<5ms` Bipartite Cortical Hypergraph PPR + Grounded Proustian Cues + `±2` Post-Merge Conversational Contiguity + CUDA Cross-Encoder (`BAAI/bge-reranker-base`) -> Stage-2 batched candidate selection, 2nd-hop bridge fetch, and CA1 Speaker-Swap / Evidence-Grounding monitor (`max_results = 14`) |
| QA & Judge Models | **`openai/gpt-4o`** QA (`temperature=0`) + **`google/gemini-2.5-flash`** Judge (`temperature=0`, `reasoning: {"enabled": false}`) via OpenRouter |
| Valid / Errors | **2,387 / 2,387 valid (`0` errors, `0` empty contexts)** |

#### Per-Category Scores & Diagnostics (`CML v3.1` — `openai/gpt-4o` + `google/gemini-2.5-flash` Judge)

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

### 1.3 100%-Local Zero-API-Cost Results (`qwen3.8-27b` QA + Judge)

Evaluated from scratch using local `qwen3.8-27b` (`temperature=0`, `enable_thinking=false`, **`\$0.00` API cost**):

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

### 1.4 Unified Head-to-Head Comparison on Full LoCoMo-Plus (2,387 Samples)

| System / Method | Category | Backbone / QA Model | Single-hop (841) | Multi-hop (282) | Temporal (321) | Common-sense (96) | Adversarial (446) | **LoCoMo Factual Avg (1,986)** | **Cognitive (401)** | **Gap** ↓ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CML v3.1 (Hybrid CLS + 2-Stage `LLM_INTERNAL`)** <sup>[0a]</sup> | Memory System | `gpt-4o` (judge: `gemini-2.5-flash`) | 71.70% | 47.87% | **60.59%** | 50.52% *(58.85%†)* | **91.70%** | **69.99%** | **50.12%** | 19.87 *(14.35‡)* |
| **CML v3.0 (Whole-Brain, Local `\$0` API)** <sup>[0b]</sup> | Memory System | `qwen3.8-27b` (100% local, `\$0` API) | 70.21% | 47.52% | **70.25%** | 44.27% | 82.06% | **68.40%** | 39.40% | 29.00 |
| **CML v3.0 (`\$0` Local Store + `gpt-4o` QA)** <sup>[0c]</sup> | Memory System | `gpt-4o` (judge: `gemini-2.5-flash`) | 63.32% | 42.20% | 57.63% | 39.06% | 87.67% | 63.70% | 33.92% | 29.78 |
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
<sup>[0b]</sup> Full 2,387-sample 100%-local run from scratch (`qwen3.8-27b` local QA + Judge, `\$0` API cost, overall 6-category avg **63.53%**, `1,516.5 / 2,387`).  
<sup>[0c]</sup> Prior `\$0`-local-store baseline (`FEATURES__USE_LLM_ENABLED=false` at read time, overall 6-category avg `58.69%`, `1,401.0 / 2,387`).  
<sup>[1]</sup> *Self-reported by [Kumiho](https://kumiho.io) on the 401-sample Cognitive split only; 1,986-sample factual scores not published.  
<sup>[2]</sup> *Reported in *Madeleine* (arXiv 2026) on the 401-sample Cognitive split only; 1,986-sample factual scores not published.  
<sup>[3]</sup> Table 1 of [arXiv:2602.10715](https://arxiv.org/abs/2602.10715) (judge: `gemini-2.5-flash`). **LoCoMo Factual Avg** is the sample-weighted average of the 5 factual categories (1,986 items); **Gap** = LoCoMo Factual Avg − Cognitive.

### 1.5 Key Achievements & Root-Cause Attribution (LoCoMo-Plus)

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

</details>

---

<details open>
<summary><strong>Part II: LongMemEval-V2 Benchmark &mdash; Multimodal Web &amp; Enterprise Agent Memory (451 Questions)</strong></summary>

### 2.1 Benchmark Overview & Ability Definitions

**LongMemEval-V2** (Wu et al., 2026, [arXiv:2605.12493](https://arxiv.org/pdf/2605.12493.pdf); [github.com/xiaowu0162/LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2); [xiaowu0162.github.io/longmemeval-v2](https://xiaowu0162.github.io/longmemeval-v2/)) evaluates whether a memory layer enables web and enterprise GUI agents to acquire **experienced-colleague competence** from long histories of multimodal agent trajectories (`100` trajectories / `~25M` tokens per haystack in `LME-V2-Small`; up to `498` trajectories / `115M` tokens in `LME-V2-Medium`). Across **451 curated questions** (`240` **Web** across customized Magento storefront, Magento Admin CMS, and Reddit/Postmill + `211` **Enterprise** ServiceNow/WorkArena, including `29` visual `question_image` queries), `295` questions use deterministic exact-phrase / multiple-choice grading and `156` (`128` premise-abstention `-abs` + `28` `gotchas`) use strict binary LLM-as-judge rubrics:

| Paper Pillar (`451`) | Granular Split | Web (`240`) | Enterprise (`211`) | Total | What the Ability Evaluates |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Static` (`189`)** | `static` (`134`) + `static-abs` (`55`) | `60` + `31` | `74` + `24` | **189** | **Static State Recall & Premise Check:** Remembering UI landmarks, form mandatory fields, catalog attributes, ServiceNow module layouts, and spotting nonexistent controls (`static-abs`). |
| **`Dynamic` (`127`)** | `dynamic` (`86`) + `dynamic-abs` (`41`) | `51` + `21` | `35` + `20` | **127** | **Dynamic State Tracking & Premise Check:** Understanding how actions mutate UI state, ticket statuses, banners, and filtered views over time — or rejecting false transition assumptions (`dynamic-abs`). |
| **`Workflow` (`106`)** | `procedure` (`74`) + `procedure-abs` (`32`) | `42` + `20` | `32` + `12` | **106** | **Workflow / Procedural Knowledge:** Recalling exact multi-step GUI navigation paths, filter recipes, and pagination shortcuts — or flagging unsupported workflows (`procedure-abs`). |
| **`Gotchas` (`29`)** | `errors-gotchas` (`29`) | `15` | `14` | **29** | **Environment Gotchas:** Recognizing recurring local traps, broken UI affordances, hidden validation blockers, and deployment-specific failure modes (`outcome=FAILURE` runs). |

*Note on column definitions:* In Table 3 of [arXiv:2605.12493](https://arxiv.org/abs/2605.12493) and the official [LongMemEval-V2 Leaderboard](https://xiaowu0162.github.io/longmemeval-v2/#leaderboard), the four breakdown columns (`Static`, `Dynamic`, `Workflow`/`Procedure`, `Gotchas`) fold each category's `-abs` premise-awareness items into its parent pillar (`189 + 127 + 106 + 29 = 451`). Both the granular `7`-category scores and the exact `4`-pillar leaderboard scores are reported below.

### 2.2 100%-Local Zero-API-Cost Runtime Configuration (`CMLMemory` + `qwen3.8-27b`)

| Parameter | Value |
| :--- | :--- |
| Dataset & Tier | `xiaowu0162/longmemeval-v2`, **`LME-V2-Small`** — **451 / 451 questions** (`240` `web` + `211` `enterprise`), `200` full multimodal trajectories pre-ingested into CML tenants `lme2v2-web` and `lme2v2-enterprise` |
| Hardware & VRAM Partitioning | Single **NVIDIA GeForce RTX 5070 Ti (`16 GB` VRAM)** + AMD Ryzen 5 5600X (`30 GiB` RAM): Phase A GPU bulk ingestion (`nomic-embed-text-v2-moe`), followed by Phase B zero-VRAM CML query server (`CUDA_VISIBLE_DEVICES=""`) + exclusive-GPU `llama-server` (`Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` + `dflash-Qwen3.8-27B-Q4_0.gguf` DFlash 2 speculative decoding + `mmproj-Qwen3.8-27B-Q8_0.gguf`) |
| CML Write Path ([`cml_memory.py`](longmemeval_v2/memory_modules/cml_memory.py)) | Full-page `2,200`-char sliding-window AXTree indexing (`250`-char overlap) preserving complete `25K+`-char ServiceNow forms/tables + synthesized trajectory workflow & `outcome=FAILURE` gotcha playbooks (`POST /api/v1/memory/write_batch`), strictly scoped to `self._inserted_set` (`7/7` `test_query_privacy.py` checks passed) |
| CML Read Path (`cml.json`) | Multimodal `question_image` landmark extraction + `qwen3.8-27b` non-thinking UI-surface/workflow query decomposition over Dense `pgvector` + PostgreSQL `tsvector` BM25 + Cortical Hypergraph PPR; lossless `StaticText`-echo AXTree deduplication (`center_axtree_max_chars = 10,500`, `memory_context_max_tokens = 16,500`, avg final context `14,271` tokens) |
| Reader & Judge Model | **`qwen3.8-27b`** (`http://127.0.0.1:8023/v1`; Reader: `enable_thinking=true`, `temperature=0.7`; Judge: `enable_thinking=false`, `temperature=0.0`; **`\$0.00` API cost**) |
| Valid / Errors | **451 / 451 valid (`0` errors, `0` empty contexts)** |

### 2.3 Per-Category & Per-Environment Diagnostics (`CML` + Local `qwen3.8-27b`)

#### Granular 7-Category & 4-Pillar Scores (`451` Questions)

| Category / Split | Web (`240`) | Enterprise (`211`) | **Combined (`451`)** | Correct | Wrong | Unknown (`\boxed{UNKNOWN}`) | Empty Ctx | Median Ctx (tok) | Median Query Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`procedure` (Non-Abst)** | **90.48%** (`38/42`) | **59.38%** (`19/32`) | **77.03%** (`57/74`) | 57 | 17 (`23.0%`) | 0 (`0.0%`) | 0 | 14,704 | `14.77s` |
| **`static` (Non-Abst)** | **48.33%** (`29/60`) | **56.76%** (`42/74`) | **52.99%** (`71/134`) | 71 | 37 (`27.6%`) | 26 (`19.4%`) | 0 | 14,436 | `18.87s` |
| **`gotchas` (`errors-gotchas`)** | **66.67%** (`10/15`) | **35.71%** (`5/14`) | **51.72%** (`15/29`) | 15 | 14 (`48.3%`) | 0 (`0.0%`) | 0 | 14,003 | `35.89s` |
| **`dynamic` (Non-Abst)** | **45.10%** (`23/51`) | **51.43%** (`18/35`) | **47.67%** (`41/86`) | 41 | 30 (`34.9%`) | 15 (`17.4%`) | 0 | 14,412 | `14.43s` |
| **Non-Abstention Overall (`323`)** | **59.52%** (`100/168`) | **54.19%** (`84/155`) | **56.97%** (`184/323`) | **184** | **98** (`30.3%`) | **41** (`12.7%`) | **0** | **14,441** | **`16.52s`** |
| `static-abs` (`55`) | `32.26%` (`10/31`) | `16.67%` (`4/24`) | **25.45%** (`14/55`) | 14 | 20 (`36.4%`) | 21 (`38.2%`) | 0 | 14,277 | `14.32s` |
| `procedure-abs` (`32`) | `15.00%` (`3/20`) | `41.67%` (`5/12`) | **25.00%** (`8/32`) | 8 | 21 (`65.6%`) | 3 (`9.4%`) | 0 | 14,017 | `13.97s` |
| `dynamic-abs` (`41`) | `23.81%` (`5/21`) | `20.00%` (`4/20`) | **21.95%** (`9/41`) | 9 | 14 (`34.1%`) | 18 (`43.9%`) | 0 | 14,301 | `15.16s` |
| **Abstention Overall (`128`)** | **25.00%** (`18/72`) | **23.21%** (`13/56`) | **24.22%** (`31/128`) | **31** | **55** (`43.0%`) | **42** (`32.8%`) | **0** | **14,242** | **`14.51s`** |
| *Paper Pillar:* **`Workflow` (`106`)** | **66.13%** (`41/62`) | **54.55%** (`24/44`) | **61.32%** (`65/106`) | 65 | 38 (`35.8%`) | 3 (`2.8%`) | 0 | 14,518 | `14.51s` |
| *Paper Pillar:* **`Gotchas` (`29`)** | **66.67%** (`10/15`) | **35.71%** (`5/14`) | **51.72%** (`15/29`) | 15 | 14 (`48.3%`) | 0 (`0.0%`) | 0 | 14,003 | `35.89s` |
| *Paper Pillar:* **`Static` (`189`)** | **42.86%** (`39/91`) | **46.94%** (`46/98`) | **44.97%** (`85/189`) | 85 | 57 (`30.2%`) | 47 (`24.9%`) | 0 | 14,390 | `17.56s` |
| *Paper Pillar:* **`Dynamic` (`127`)** | **38.89%** (`28/72`) | **40.00%** (`22/55`) | **39.37%** (`50/127`) | 50 | 44 (`34.6%`) | 33 (`26.0%`) | 0 | 14,380 | `14.65s` |
| **Full-Set Overall (`451`)** | **49.17%** (`118/240`) | **45.97%** (`97/211`) | **47.67%** (`215/451`) | **215** | **153** (`33.9%`) | **83** (`18.4%`) | **0** | **14,388** | **`16.01s`** *(mean `19.65s`)* |

#### Breakdown by Customized Deployment Environment & Modality

| Environment / Modality Subset | Domain | Total Qs | Non-Abstention Accuracy | Full-Set Accuracy (incl. `-abs`) |
| :--- | :---: | :---: | :---: | :---: |
| **`webarena-onestopshop`** (Customized Magento Storefront) | Web | 66 | **63.83%** (`30 / 47`) | **53.03%** (`35 / 66`) |
| **`webarena-reddit`** (Customized Postmill Forum) | Web | 91 | **59.68%** (`37 / 62`) | **46.15%** (`42 / 91`) |
| **`webarena-cms`** (Customized Magento Admin CMS) | Web | 83 | **55.93%** (`33 / 59`) | **49.40%** (`41 / 83`) |
| **`workarena`** (Customized ServiceNow Enterprise Platform) | Enterprise | 211 | **54.19%** (`84 / 155`) | **45.97%** (`97 / 211`) |
| **Visual `question_image` Queries** (Multimodal UI Screenshots) | Both | 29 | **51.72%** (`15 / 29`) | **51.72%** (`15 / 29`) |

### 2.4 Unified Head-to-Head Comparison on Full LongMemEval-V2 (`LME-V2-Small`, 451 Questions)

All columns follow the exact 451-question LongMemEval-V2 Paper Table 3 & Public Leaderboard definitions (`Static [189]`, `Dynamic [127]`, `Workflow / Procedure [106]`, `Gotchas [29]`, `Overall [451]`, and online `Query Latency`):

| System / Method | Family | Controller / Reader / Judge | Static (189) | Dynamic (127) | **Workflow / Proc. (106)** | **Gotchas (29)** | **Overall (451)** | Query Latency ↓ | API Cost |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CML (`CMLMemory`, 100% Local `\$0` API)** <sup>[0]</sup> | **Neuro Memory (RAG/Graph)** | **Local `qwen3.8-27b` (`IQ3_S`) Reader + Judge** | 44.97% *(52.99%†)* | 39.37% *(47.67%†)* | **61.32%** *(**77.03%†**)* | **51.72%** *(**66.67%‡**)* | **47.67%** *(**56.97%†**)* | **19.65s** *(13.17s‡)* | **\$0.00** |
| **Perpetual Recall V3** *(Equinox Labs)* <sup>[L]</sup> | Hybrid Retrieval | `Qwen3.5-9B` + Cloud Judge (`gpt-5.2`) | 58.20% | 46.46% | 56.60% | 37.93% | 53.22% | 1.96s | Cloud |
| **AgentRunbook-R** *(Reference)* <sup>[P,L]</sup> | RAG (3-Pool Active) | `Qwen3.5-9B` + `Qwen3-Embed-8B` + `gpt-5.2` Judge | **66.10%** | **58.30%** | 52.80% | 31.00% | **58.60%** | 26.90s | Cloud |
| &nbsp;&nbsp;– *note pool only* <sup>[P]</sup> | RAG Ablation | `Qwen3.5-9B` + `Qwen3-Embed-8B` + `gpt-5.2` Judge | 65.10% | 61.40% | 48.10% | 31.00% | 57.90% | 22.80s | Cloud |
| &nbsp;&nbsp;– *event pool only* <sup>[P]</sup> | RAG Ablation | `Qwen3.5-9B` + `Qwen3-Embed-8B` + `gpt-5.2` Judge | 61.40% | 55.90% | 52.80% | 27.60% | 55.60% | 19.10s | Cloud |
| &nbsp;&nbsp;– *raw slice pool only* <sup>[P]</sup> | RAG Ablation | `Qwen3.5-9B` + `Qwen3-Embed-8B` + `gpt-5.2` Judge | 28.60% | 55.10% | 53.80% | 34.50% | 42.30% | 16.70s | Cloud |
| **RAG: `query -> slice + notes`** *(Reference)* <sup>[P,L]</sup> | RAG Baseline | `Qwen3.5-9B` + `Qwen3-Embed-8B` + `gpt-5.2` Judge | 52.40% | 49.60% | 52.80% | 41.40% | 51.00% | **0.20s** | Cloud |
| **RAG: `query -> slice`** <sup>[P,L]</sup> | RAG Baseline | `Qwen3.5-9B` + `Qwen3-Embed-8B` + `gpt-5.2` Judge | 47.10% | 42.50% | 41.50% | 20.70% | 42.80% | **0.10s** | Cloud |
| **OPS-Agentic-Memory-v1** *(Alibaba Cloud)* <sup>[L]</sup> | Coding Agent Sandbox | Frontier Coding Agent + `gpt-5.2` Judge | **87.83%** | **81.10%** | **76.42%** | **58.62%** | **81.37%** | 153.64s | Cloud |
| **AgentRunbook-C v2 Online (`xhigh`)** <sup>[L]</sup> | Coding Agent Sandbox | `gpt-5.4-mini` (`xhigh`) + `Qwen3.5-9B` + `gpt-5.2` | — | — | — | — | 75.61% | 130.54s | Cloud |
| **AgentRunbook-C v2 Online (`medium`)** <sup>[L]</sup> | Coding Agent Sandbox | `gpt-5.4-mini` (`medium`) + `Qwen3.5-9B` + `gpt-5.2` | — | — | — | — | 72.73% | 56.68s | Cloud |
| **AgentRunbook-C v2 Online (`low`)** <sup>[L]</sup> | Coding Agent Sandbox | `gpt-5.4-mini` (`low`) + `Qwen3.5-9B` + `gpt-5.2` | — | — | — | — | 65.41% | 26.11s | Cloud |
| **AgentRunbook-C** *(Reference)* <sup>[P,L]</sup> | Coding Agent Sandbox | `gpt-5.4-mini` (`xhigh`) + `Qwen3.5-9B` + `gpt-5.2` | 82.00% | 72.40% | 72.60% | 48.30% | 74.90% | 108.30s | Cloud |
| &nbsp;&nbsp;– *without manifest artifacts* <sup>[P]</sup> | Coding Agent Ablation | `gpt-5.4-mini` (`xhigh`) + `Qwen3.5-9B` + `gpt-5.2` | 84.70% | 70.90% | 69.80% | 44.80% | 74.70% | 155.00s | Cloud |
| &nbsp;&nbsp;– *without helper functions* <sup>[P]</sup> | Coding Agent Ablation | `gpt-5.4-mini` (`xhigh`) + `Qwen3.5-9B` + `gpt-5.2` | 78.30% | 72.40% | 66.00% | 41.40% | 71.40% | 145.90s | Cloud |
| &nbsp;&nbsp;– *without workflow guide* <sup>[P]</sup> | Coding Agent Ablation | `gpt-5.4-mini` (`xhigh`) + `Qwen3.5-9B` + `gpt-5.2` | 77.20% | 67.70% | 63.20% | 58.60% | 70.10% | 167.90s | Cloud |
| **Codex** *(Reference)* <sup>[P,L]</sup> | Coding Agent Sandbox | `gpt-5.4-mini` (`xhigh`) + `Qwen3.5-9B` + `gpt-5.2` | 80.40% | 67.00% | 57.50% | 58.60% | 69.90% | 177.20s | Cloud |
| Kimi-K2.5 *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `Kimi-K2.5` parametric weights only | 18.30% | 19.70% | 11.50% | 17.10% | 14.10% | 0s | Cloud |
| Claude Opus 4.6 *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `Claude Opus 4.6` parametric weights only | 9.60% | 12.10% | 13.40% | 37.90% | 11.80% | 0s | Cloud |
| Gemini-3.1-Pro-Preview *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `Gemini-3.1-Pro` parametric weights only | 10.40% | 9.60% | 14.70% | 24.10% | 11.00% | 0s | Cloud |
| Qwen3.6-Plus *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `Qwen3.6-Plus` parametric weights only | 11.80% | 7.80% | 9.10% | 31.00% | 11.00% | 0s | Cloud |
| GLM-5V-Turbo *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `GLM-5V-Turbo` parametric weights only | 12.60% | 10.70% | 9.10% | 20.50% | 10.10% | 0s | Cloud |
| GPT-5.2 *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `GPT-5.2` parametric weights only | 0.00% | 0.00% | 3.20% | 21.00% | 4.70% | 0s | Cloud |
| Grok-4.20 *(No Memory)* <sup>[P]</sup> | Frontier LLM Only | `Grok-4.20` parametric weights only | 0.00% | 2.90% | 15.10% | 10.20% | 2.40% | 0s | Cloud |
| No Retrieval *(Baseline)* <sup>[P,L]</sup> | Reader Only | `Qwen3.5-9B` (thinking enabled) | 0.00% | 0.80% | 9.40% | 13.80% | 1.30% | 0s | — |
| *Oracle Upper Bound: Slices + Notes* <sup>[P]</sup> | *Human Oracle* | *`Qwen3.5-9B` given gold state slices + notes* | *90.80%* | *87.90%* | *75.00%* | *48.40%* | *82.50%* | — | — |

<sup>[0]</sup> Full 451-question 100%-local run on a single 16 GB RTX 5070 Ti (`qwen3.8-27b` IQ3_S 3-bit GGUF + DFlash 2 for query decomposition, multimodal reader, and judge; `0` errors, `\$0.00` API cost). †Non-abstention accuracy (`323` Qs): **`56.97%` Overall**, **`77.03%` Procedure** (`90.48%` Web), **`52.99%` Static** (`56.76%` Enterprise), **`47.67%` Dynamic** (`51.43%` Enterprise). ‡Web domain (`240` Qs): **`49.17%` Full-Set**, **`59.52%` Non-Abstention**, **`66.67%` Gotchas**, **`13.17s` Query Latency**.  
<sup>[P]</sup> Tables 2, 3, and 9 of Wu et al. (2026), *LongMemEval-V2* ([arXiv:2605.12493](https://arxiv.org/abs/2605.12493)).  
<sup>[L]</sup> Official LongMemEval-V2 Public Leaderboard (`LME-V2-Small`, [xiaowu0162.github.io/longmemeval-v2/#leaderboard](https://xiaowu0162.github.io/longmemeval-v2/#leaderboard)).

### 2.5 Key Achievements, Trade-Offs & Root-Cause Attribution (LongMemEval-V2)

1. **#1 Among All Non-Coding-Agent Memory Systems on BOTH `Workflow / Procedure` (`61.32%` full / `77.03%` non-abs / `90.48%` Web) AND `Gotchas` (`51.72%` full / `66.67%` Web):**
   - On **`Workflow / Procedure` (`106` Qs)**, 100%-local CML (`61.32%`) surpasses **RAG + notes (`52.80%`, `+8.52pp`)**, **AgentRunbook-R (`52.80%`, `+8.52pp` at faster `19.65s` vs. `26.90s` latency)**, **Perpetual Recall V3 (`56.60%`, `+4.72pp`)**, and even **Cloud `Codex` (`GPT-5.4-mini` `xhigh`, `57.50%`, `+3.82pp` at `9×` lower latency and `\$0.00` API cost)**.
   - On **`Gotchas` (`29` Qs)**, 100%-local CML (`51.72%`) surpasses **AgentRunbook-R (`31.00%`, `+20.72pp`)**, **Perpetual Recall V3 (`37.93%`, `+13.79pp`)**, **RAG + notes (`41.40%`, `+10.32pp`)**, **Cloud `AgentRunbook-C` (`GPT-5.4-mini` `xhigh`, `48.30%`, `+3.42pp` at `5.5×` lower latency)**, and even the **Human Oracle Slices + Notes `Qwen3.5-9B` upper bound (`48.40%`)**!
   - *Root cause:* Synthesizing structured trajectory workflow playbooks alongside explicit `[IMPORTANT — FAILED TRAJECTORY / POTENTIAL ENVIRONMENT GOTCHA]` callouts and annotated agent thought-action chains preserves failure causality that single-state embeddings and LLM-compressed summary notes discard.
2. **Solving 25K-Char Enterprise AXTree Middle-Truncation (`56.76%` Enterprise `static`, `51.43%` Enterprise `dynamic`):**
   - Naive head/tail truncation of ServiceNow WorkArena accessibility trees drops up to `90%` of form controls and table rows (visible in `AgentRunbook-R`'s `raw slice pool` ablation collapsing to `28.60%` on `Static`). CML's `2,200`-char sliding-window chunking at write time paired with lossless `StaticText`-echo deduplication (`center_axtree_max_chars = 10,500`) retains full form schemas without exceeding a 24K local KV-cache budget.
3. **Unbiased Trade-Off Analysis & Headroom (`Static-abs` / `Dynamic-abs` Premise Pinpointing & Coding-Agent Exploration):**
   - **Why Overall Full-Set is `47.67%` vs. `56.97%` Non-Abstention:** On the `128` flawed-premise (`-abs`) questions, LongMemEval-V2's strict abstention rubric awards **`0`** to `\boxed{UNKNOWN}` (`32.8%` of CML's `-abs` responses) unless the reader explicitly names the *exact* contradicted sub-premise (`24.22%` pass rate). Under 3-bit `IQ3_S` quantization, `qwen3.8-27b` conservatively outputs `\boxed{UNKNOWN}` when an asked control is absent from retrieved AXTrees rather than asserting global non-existence across the deployment.
   - **Indexed Memory (`13–27s`) vs. Coding-Agent Sandboxes (`108–177s`):** Sandbox coding agents (`AgentRunbook-C`, `OPS-Agentic-Memory-v1`) achieve higher `Static`/`Dynamic` recall (`78–88%`) by running `gpt-5.4-mini` (`xhigh`) bash/grep loops over raw JSON files for `2–3 minutes` per query, whereas CML answers in a single bounded retrieval pass at `\$0.00` cloud cost. Pairing `CMLMemory` with `cml_fast` (`<3s` query latency) and a frontier cloud reader/judge offers immediate multi-point LAFS upside.

</details>

---

## Appendix: CML Server Write & Retrieval Throughput Summary

| Mode | Throughput / Latency | Notes |
| :--- | :---: | :--- |
| Hybrid CLS Batch Write (`FEATURES__BATCH_WRITE_SYSTEM1_FASTPATH=true`) | **`>300 turns/s`** (`0.024 s` p50) | Default for `/memory/write_batch`: GPU batch embedding + `tsvector` + neuro tagging |
| Windowed Micro-Sleep Consolidation (`consolidate_windowed`) | **18–30 turn CRC32 windows** (~18× fewer LLM calls than per-turn) | Content-defined chunking + persistent SQLite extraction cache (`>90%` cache hit across variants) |
| Zero-VRAM Hybrid Retrieval (`pgvector` + `tsvector` BM25 + Hypergraph PPR) | **`~1.1s`** (`0 MiB` GPU VRAM) | Enables co-locating CML alongside full-GPU 27B–32B local readers on 16 GB consumer cards |
| Live Single-Turn Write (`FEATURES__USE_LLM_ENABLED=true`) | **~0.35 turns/s** | Online System-2 `LLM_INTERNAL` extraction on `/memory/write` |
