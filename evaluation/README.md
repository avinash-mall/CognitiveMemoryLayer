# Evaluation with CML

This directory organizes multi-benchmark evaluations for **Cognitive Memory Layer (CML)** across two complementary long-horizon agent memory benchmarks:

1. **[LoCoMo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) (`2,387` samples — Conversational Factual & Implicit Cognitive Memory):**
   - Unifies the five LoCoMo factual categories (`single-hop`, `multi-hop`, `temporal`, `common-sense`, `adversarial`) with a sixth **Cognitive** cue–trigger category:
     - **`single-hop` (841):** Direct recall of a single fact, event, or `[Shared photo: ...]` detail (**71.70%** `gpt-4o` / **70.21%** local).
     - **`multi-hop` (282):** Synthesizing scattered clues or multi-session lists/counts (**47.87%** `gpt-4o` / **47.52%** local).
     - **`temporal` (321):** Resolving relative dates (`"yesterday"`, `"last week"`), ordering, and durations (**60.59%** `gpt-4o` / **70.25%** local).
     - **`common-sense` (96):** Combining stored persona values, goals, and habits with world knowledge (**50.52%** / **58.85%** open).
     - **`adversarial` (446):** Rejecting false presuppositions and speaker-swap traps (**91.70%**, **#1 overall**).
     - **`Cognitive` / LoCoMo-Plus (401):** Implicit cue–trigger memory awareness across zero lexical overlap (**50.12%**, **#1 Table 1**).
     - **`Gap` (`LoCoMo Avg − Cognitive`):** Drop-off from factual (`69.99%`) to cognitive (`50.12%`) recall (**19.87** weighted / **14.35** macro; lower is better).
2. **[LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) (`451` questions — Multimodal Web & Enterprise Agent Experience Memory):**
   - Evaluates whether a memory system turns long histories of multimodal GUI/web-agent trajectories (`100`–`500` trajectories per haystack across `web` and `enterprise` ServiceNow/WorkArena domains) into actionable colleague-level experience across **5 abilities**:
     - **`procedure` (`74`):** Recurring UI workflows and multi-step execution sequences (**77.03%** overall; **90.48%** on `web`, 100% local `qwen3.8-27b`).
     - **`static` (`134`):** Persistent UI landmarks, layouts, module affordances, and configuration facts (**52.99%** overall; **56.76%** on `enterprise`).
     - **`gotchas` (`29`):** Environment-specific pitfalls, broken affordances, and recurring failure modes (**51.72%** overall; **66.67%** on `web`).
     - **`dynamic` (`86`):** Tracking how state and agent actions mutate records over time (**47.67%** overall; **51.43%** on `enterprise`).
     - **`*-abs` Premise Awareness (`128`):** Detecting false premises (`static-abs`, `dynamic-abs`, `procedure-abs`) in customized deployments (**24.22%** abstention; **56.97%** non-abstention avg, **47.67%** full-set across all `451` questions at **\$0.00** API cost).

See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for the complete head-to-head breakdown across both benchmarks.

## Organized Directory Layout

| Path | Description |
|------|-------------|
| `locomo_plus/` | LoCoMo-Plus data pipeline (`locomo10.json`, `locomo_plus.json`, `unified_input_samples_v2.json`), `task_eval`, and baseline scripts. See [locomo_plus/README.md](locomo_plus/README.md). |
| `longmemeval_v2/` | Official LongMemEval-V2 harness, dataset preparation/validation (`data/`), leaderboard LAFS tools (`leaderboard/`), and native CML backend (`memory_modules/cml_memory.py`). See [longmemeval_v2/README.md](longmemeval_v2/README.md). |
| `cml-eval` CLI (`packages/py-cml/src/cml/eval/`) | Canonical CML-backed LoCoMo-Plus CLI (`run-full`, `run-locomo`, `validate`, `report`, `compare`). Install via `pip install -e ".[eval]"`. |
| `scripts/run_locomo_openrouter_batch.py` | End-to-end LoCoMo-Plus driver supporting OpenRouter sync/batch or local OpenAI-compatible servers with checkpointing and cost tracking. |
| `scripts/run_longmemeval_v2_local.py` | End-to-end LongMemEval-V2 driver for local LLM/VLM servers (`--preingest-only`, `--resume`, `--skip-existing`, domain metric merging, and LAFS calculation). |
| `results/locomo_plus/` | Committed LoCoMo-Plus summary JSON artifacts. |
| `results/longmemeval_v2/` | Committed LongMemEval-V2 summary & LAFS JSON artifacts. |
| `outputs/locomo_plus/` | Runtime LoCoMo-Plus predictions, judged records, checkpoints, and `run_full_eval_state.json` (git-ignored). |
| `outputs/longmemeval_v2/` | Runtime LongMemEval-V2 per-domain outputs (`per_question.jsonl`, `reader_outputs.jsonl`, `aggregated_metrics.json`, combined metrics; git-ignored). |

---

<details open>
<summary><strong>Benchmark 1: LongMemEval-V2 (451 Questions &mdash; Multimodal Web &amp; Enterprise Agent Memory)</strong></summary>

### 1. Download & Validate Dataset (`xiaowu0162/longmemeval-v2`)

```bash
cd evaluation/longmemeval_v2
python data/download_data.py --data-root data/longmemeval-v2
python data/prepare_data.py --data-root data/longmemeval-v2 --mode symlink
python data/validate_data.py --data-root data/longmemeval-v2 --tier small
python data/validate_data.py --data-root data/longmemeval-v2 --tier medium
cd ../..
```

### 2. Two-Phase Local Single-GPU Execution (16 GB VRAM Optimal Split)

When running a 27B multimodal reader/judge locally (e.g., `qwen3.8-27b` with DFlash 2 speculative decoding on a 16 GB GPU), separate GPU bulk embedding ingestion from GPU reader inference so neither contends for VRAM:

```bash
# Phase A: Pre-ingest full-page sliding-window AXTree chunks + workflow/gotcha playbooks on GPU
# (Run while local LLM server is stopped so CML's nomic-embed-text-v2-moe uses CUDA at ~50+ chunks/s)
python evaluation/scripts/run_longmemeval_v2_local.py \
  --tier small \
  --cml-url http://127.0.0.1:8000 \
  --preingest-only

# Phase B: Start CML in zero-VRAM hybrid retrieval mode (CPU query embed + Postgres pgvector/BM25 + PPR)
# alongside the exclusive-GPU local Qwen3.8-27B server (port 8023), then run all 451 questions:
python evaluation/scripts/run_longmemeval_v2_local.py \
  --tier small \
  --methods cml \
  --local-base-url http://127.0.0.1:8023/v1 \
  --local-model qwen3.8-27b \
  --cml-url http://127.0.0.1:8000 \
  --memory-context-max-tokens 16500 \
  --prompt-build-max-workers 1 \
  --reader-max-concurrent-requests 2 \
  --skip-existing \
  --resume
```

You can also invoke the shell wrapper directly inside `evaluation/longmemeval_v2/`:

```bash
cd evaluation/longmemeval_v2
DATA_ROOT="$PWD/data/longmemeval-v2" \
OUTPUT_ROOT="../outputs/longmemeval_v2" \
TIER=small \
METHOD=cml \
READER_BASE_URL=http://127.0.0.1:8023/v1 \
READER_MODEL=qwen3.8-27b \
EVALUATOR_BASE_URL=http://127.0.0.1:8023/v1 \
EVALUATOR_MODEL=qwen3.8-27b \
bash evaluation/scripts/run_cml.sh --resume
```

- **`cml` (`evaluation/memory_configs/cml.json`):** Full hybrid retrieval (`pgvector` + `tsvector` BM25 + Hypergraph PPR) with multimodal `question_image` landmark extraction, `qwen3.8-27b` UI-surface/workflow query decomposition, lossless `StaticText`-echo AXTree deduplication, and explicit `FAILURE`-outcome gotcha callouts.
- **`cml_fast` (`evaluation/memory_configs/cml_fast.json`):** Low-latency operating point (`enable_query_expansion=false`, `<3s` query latency) for multi-point Pareto LAFS curves (`--methods cml cml_fast`).
- **Query Privacy:** Verified by `python -m unittest discover -s evaluation/longmemeval_v2/tests -v` (`7/7` query-privacy isolation tests passing).

</details>

<details open>
<summary><strong>Benchmark 2: LoCoMo-Plus (2,387 Samples &mdash; Conversational Factual &amp; Implicit Cognitive Memory)</strong></summary>

### Prerequisites

1. **CML API** running (e.g. via Docker; see main project README).
2. **Embedding** (for CML): project root `.env` — `EMBEDDING_INTERNAL__MODEL`, `EMBEDDING_INTERNAL__DIMENSIONS`.
3. **QA model**: Phase B uses the **LLM** from project root `.env` (`LLM_EVAL__PROVIDER`, `LLM_EVAL__MODEL`, `LLM_EVAL__BASE_URL`, `LLM_EVAL__API_KEY`).
4. **Python deps**: `requests`, `tqdm`, `openai`; for LLM-as-judge: set `OPENAI_API_KEY`, or provide an OpenAI-compatible endpoint via `OPENAI_BASE_URL` / `LLM_EVAL__BASE_URL`.

### Performance tuning for bulk eval

Rate limiting is disabled by default (`AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=0`), but the default server worker count and embedding batch size are conservative for bulk evaluation. For runs with many `--ingestion-workers`, add to `.env`:

```bash
AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=0
EMBEDDING_INTERNAL__LOCAL_BATCH_SIZE=64
UVICORN_WORKERS=4
```

When running outside Docker:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Run full LoCoMo-Plus evaluation (`cml-eval`)

Install evaluation extras (`pip install -e ".[eval]"`) and run from the **project root**:

```bash
cml-eval run-full --repo-root .
```

This runs: (1) Docker down -v, (2) Docker up (postgres, neo4j, redis, api), (3) API health wait, (4) `run-locomo` (ingest, consolidation + reconsolidation, QA, judge), (5) performance table. If validation fails, state is saved to `evaluation/outputs/locomo_plus/run_full_eval_state.json` for `--resume`.

### Run LoCoMo-Plus manually

```bash
cml-eval run-locomo \
  --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json \
  --out-dir evaluation/outputs/locomo_plus
```

### Generate report & comparison tables

```bash
cml-eval report --summary evaluation/results/locomo_plus/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json --method "CML v3.1 (GPT-4o)"
cml-eval compare --summary evaluation/results/locomo_plus/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json --method "CML v3.1 (GPT-4o)"
```

### Run Official Table 1 Cloud Benchmark via OpenRouter (`gpt-4o` + `gemini-2.5-flash`)

```bash
python evaluation/scripts/run_locomo_openrouter_batch.py \
  --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json \
  --out-dir evaluation/outputs/locomo_plus/cml_v3_gpt4o_gemini25flash_full_2387 \
  --cml-url http://localhost:8000 \
  --skip-ingestion \
  --base-url https://openrouter.ai/api/v1 \
  --model openai/gpt-4o \
  --judge-model google/gemini-2.5-flash \
  --sync-workers 10 \
  --qa-max-tokens 200 \
  --judge-max-tokens 160
```

### Build unified input (optional)

```bash
cd evaluation/locomo_plus/data && python unified_input.py
```

### Run without CML (OpenAI-style API)

```bash
export PYTHONPATH=evaluation/locomo_plus
python evaluation/locomo_plus/scripts/run_evaluate.py --backend call_llm --model gpt-4o-mini
python evaluation/locomo_plus/scripts/run_judge.py --model gpt-4o-mini
```

Outputs are written to `evaluation/outputs/locomo_plus/locomo_plus_predictions.json` and `evaluation/outputs/locomo_plus/locomo_plus_judged.json`.

</details>

---

## Committed Summary Artifacts

- **LongMemEval-V2 (`451` questions, `small` tier, 100% local `qwen3.8-27b`, \$0.00 API cost):**
  - [`results/longmemeval_v2/longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json`](results/longmemeval_v2/longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json)
- **LoCoMo-Plus (`2,387` samples):**
  - Official Table 1 protocol (`openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge): [`results/locomo_plus/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json`](results/locomo_plus/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json)
  - 100%-local zero-API-cost run (`qwen3.8-27b` QA + Judge): [`results/locomo_plus/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json`](results/locomo_plus/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json)
  - Prior \$0-local-store baseline (`FEATURES__USE_LLM_ENABLED=false` at read time + `gpt-4o` QA): [`results/locomo_plus/locomo_plus_2026-11-26_cml_v3_gpt4o_gemini25flash_full_2387_summary.json`](results/locomo_plus/locomo_plus_2026-11-26_cml_v3_gpt4o_gemini25flash_full_2387_summary.json)

---

## References

- **LongMemEval-V2 paper:** [arXiv:2605.12493](https://arxiv.org/pdf/2605.12493.pdf) — *LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues* (Wu et al., 2026)
- **LongMemEval-V2 repo & dataset:** [github.com/xiaowu0162/LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) • [huggingface.co/datasets/xiaowu0162/longmemeval-v2](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)
- **Locomo-Plus paper:** [arXiv:2602.10715](https://arxiv.org/abs/2602.10715)
- **Locomo-Plus repo:** [github.com/xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus)
- **LoCoMo paper / repo:** [snap-research.github.io/locomo](https://snap-research.github.io/locomo/) • [github.com/snap-research/locomo](https://github.com/snap-research/locomo)
