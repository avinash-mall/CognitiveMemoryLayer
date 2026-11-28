# LongMemEval-V2


<p align="center">
  <a href="https://xiaowu0162.github.io/longmemeval-v2/"><img src="https://img.shields.io/badge/🌐-Website-2a75d0?style=flat-square" height="23"></a>
  <a href="https://arxiv.org/pdf/2605.12493.pdf"><img src="https://img.shields.io/badge/📝-Paper-d03c36?style=flat-square" height="23"></a>
  <a href="https://huggingface.co/datasets/xiaowu0162/longmemeval-v2" ><img src="https://img.shields.io/badge/🤗-Data-167f5f?style=flat-square" height="23"></a>
  <a href="https://xiaowu0162.github.io/longmemeval-v2/#leaderboard" ><img src="https://img.shields.io/badge/🏆-Leaderboard-d89216?style=flat-square" height="23"></a>
</p>

**LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues**

[Di Wu](https://xiaowu0162.github.io/),
[Zixiang Ji](https://www.linkedin.com/in/zixiang-ji-56902624b/),
[Asmi Kawatkar](https://www.linkedin.com/in/asmi-kawatkar),
[Bryan Kwan](https://www.linkedin.com/in/kwan-bryan),
[Jia-Chen Gu](https://jasonforjoy.github.io/index.html),
[Nanyun Peng](https://vnpeng.net/), and
[Kai-Wei Chang](https://kwchang.net/)



This is the official LongMemEval-V2 repository. It contains the public
evaluation harness, data preparation tools, leaderboard packaging utilities,
and the memory baselines reported with the benchmark.

## News 
- [2026/08] Update: [AgentRunbook-C V2](https://xiaowu0162.github.io/longmemeval-v2/agentrunbook-c-v2/).

## Overview

LongMemEval-V2 evaluates whether memory systems can help agents acquire the
experience needed to become knowledgeable colleagues in customized
environments. The benchmark pairs manually curated questions with long
histories of multimodal web-agent trajectories. A memory system consumes the
trajectory history and returns compact evidence for downstream question
answering; evaluation targets both answer accuracy and query latency.

LongMemEval-V2 contains:

- 451 manually curated questions.
- 5 memory abilities.
- Up to 500 trajectories per haystack.
- Up to 115M tokens in the largest haystacks.
- Two domains: web and enterprise.
- Two public leaderboard tiers: small and medium.

The benchmark tests five core memory abilities:

- **Static state recall**: remembers important landmarks, page layouts, module
  affordances, and subtle state differences.
- **Dynamic state tracking**: understands how states and actions change the
  environment over time.
- **Workflow knowledge**: knows the steps needed to complete recurring tasks in
  customized environments.
- **Environment gotchas**: recognizes recurring local failure modes and avoids
  environment-specific traps.
- **Premise awareness**: detects assumptions that are valid elsewhere but wrong
  in the current deployment.

## Repository Layout

```text
data/                 download, preparation, and validation scripts
evaluation/           evaluation runner, scoring code, configs, and shell wrappers
leaderboard/          metric merging, LAFS scoring, and submission packaging
memory_modules/       memory backend implementations
```

The repository implements the following memory modules:

- `no_retrieval`: no memory context.
- `rag_query_to_slice`: RAG query to raw state slices.
- `rag_query_to_slice_notes`: RAG query to raw state slices plus trajectory
  notes.
- `agentrunbook_r`: AgentRunbook-R.
- `codex`: vanilla Codex coding-agent memory baseline.
- `agentrunbook_c`: AgentRunbook-C.
- `cml` (`cml_fast`): Native **CognitiveMemoryLayer** (`CMLMemory`, `memory_modules/cml_memory.py`) with full-page sliding-window AXTree indexing, procedural workflow & gotcha playbooks, hybrid `pgvector` + `tsvector` BM25 + Cortical Hypergraph PPR retrieval, and 100%-local LLM/VLM support.

## Setup: Environment

LongMemEval-V2 uses Python 3.11. The default conda environment installs
PyTorch through `requirements-torch.txt`. For CUDA 12.4 machines, the torch
install command is:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Create the environment and install the package:

```bash
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda activate lme-v2-release
pip install -e .
```

Researchers using a different CUDA or CPU setup should install the appropriate
PyTorch build first, either with a direct `pip install` command or by editing
`requirements-torch.txt` before creating the environment.

The environment does not include vLLM. Start or forward your own
OpenAI-compatible model servers, then point the scripts to them. The paper runs
use Qwen3.5-9B as the fixed reader and Qwen3-Embedding-8B for embedding-based
methods. For `codex` and `agentrunbook_c`, download Codex v0.117.0 separately
and set `CODEX_BINARY`.

## Setup: Data

Download and prepare:

```bash
python data/download_data.py --data-root data/longmemeval-v2
export DATA_ROOT="$(pwd)/data/longmemeval-v2"
python data/prepare_data.py --data-root "$DATA_ROOT" --mode symlink
python data/validate_data.py --data-root "$DATA_ROOT" --tier small
```

The default dataset repository is
`xiaowu0162/longmemeval-v2`. Screenshot bundles are stored as `.tar.gz`
archives under `trajectory_screenshots/`; `prepare_data.py` extracts them when
needed and links the resulting directories into:

```text
screenshots/<trajectory_id>/<step>.png
```

## Setup: Model Endpoints and Software

Example endpoint settings:

```bash
# for all experiments
export READER_BASE_URL=http://localhost:8023/v1
export READER_MODEL=Qwen/Qwen3.5-9B

# additionally for RAG and AgentRunbook-R
export LME_CONTROLLER_BASE_URL=http://localhost:8023/v1
export LME_CONTROLLER_MODEL=Qwen/Qwen3.5-9B
export LME_EMBEDDING_BASE_URL=http://localhost:8114/v1
export LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

Set for LLM judge (default `gpt-5.2` with `medium` reasoning):

```bash
export OPENAI_API_KEY=...
```

For Codex and AgentRunbook-C:

```bash
export CODEX_BINARY=/path/to/codex-binary
export CODEX_MODEL=gpt-5.4-mini
export CODEX_REASONING_EFFORT=xhigh
```

Codex also expects common command-line tools such as `rg` and `find`.

## Reproducing Baselines

Each shell script accepts extra argparse flags after the environment variables:

```bash
export DATA_ROOT=/path/to/longmemeval-v2
export OUTPUT_ROOT=runs
export TIER=small

evaluation/scripts/run_no_retrieval.sh
evaluation/scripts/run_rag_query_to_slice.sh
evaluation/scripts/run_rag_query_to_slice_notes.sh
evaluation/scripts/run_agentrunbook_r.sh
evaluation/scripts/run_codex.sh
evaluation/scripts/run_agentrunbook_c.sh
```

Each script runs both the web and enterprise domains for the selected tier, writing
outputs such as `runs/no_retrieval_web_small` and
`runs/no_retrieval_enterprise_small`. Set `TIER=medium` to run LME-V2-Medium.

Each run writes `aggregated_metrics.json`. To combine matching enterprise and web runs for the same method and tier:

```bash
python leaderboard/combine_aggregated_metrics.py \
  runs/agentrunbook_r_enterprise_small/aggregated_metrics.json \
  runs/agentrunbook_r_web_small/aggregated_metrics.json \
  -o runs/agentrunbook_r_small_combined_metrics.json
```

## Implementing Your Method

Memory backends inherit from `memory_modules.memory.Memory`. For a minimal
example, see `memory_modules/no_retrieval.py`; for indexed retrieval examples,
see `memory_modules/rag.py` and `memory_modules/agentrunbook_r.py`.

A backend should:

- decorate the class with `@register_memory`;
- set a unique `memory_type`;
- implement `insert(self, trajectory)`, which receives each full trajectory
  object selected for the current haystack;
- implement `query(self, query, query_image=None)`, which receives the question
  text and optional question screenshot path.

`query` must return a list of memory context items:

```python
[
    {"type": "text", "value": "retrieved notes or evidence"},
    {"type": "image", "value": "/absolute/or/relative/path/to/image.png"},
]
```

Text values must be non-empty strings. Image values must point to existing
files. The harness appends these items to the reader prompt and enforces
`--memory-context-max-tokens` before calling the answer model.

Backends receive no benchmark metadata during `query`: retrieval is based only
on the question text and optional question image. `self.get_query_context()`
contains only a random, run-local `query_invocation_id` for trace and lifecycle
bookkeeping. The evaluator keeps the dataset question ID, question type, raw
question record, gold answer, and evaluator configuration private. Optional
hooks include `post_query_hook(...)` for per-query metadata and
`_save_backend(...)` / `_load_backend(...)` for persisted memory state.

To run a new backend directly, create a memory config JSON:

```json
{
  "memory_type": "your_memory_type",
  "memory_params": {}
}
```

Then pass it to `evaluation/harness.py` with `--memory-config-path`. To expose
the method through `evaluation/run_eval.py` and the shell wrappers, add the
method name and config construction there as well.

## CognitiveMemoryLayer (CML) Integration & Local Evaluation

This checkout includes a native **CognitiveMemoryLayer** backend (`memory_modules/cml_memory.py`, `@register_memory class CMLMemory`, `memory_type = "cml"`) and two configuration presets:

- **`evaluation/memory_configs/cml.json` (`--method cml`):** Full hybrid retrieval (`pgvector` + PostgreSQL `tsvector` BM25 + Cortical Hypergraph PPR) with multimodal `question_image` landmark extraction, non-thinking UI-surface/workflow query decomposition, lossless `StaticText`-echo AXTree deduplication (`center_axtree_max_chars = 10500`), and explicit `outcome=FAILURE` gotcha callouts.
- **`evaluation/memory_configs/cml_fast.json` (`--method cml_fast`):** Fast operating point (`enable_query_expansion = false`) for multi-point Pareto LAFS curves.

### Reproducing the Full 451-Question Local Benchmark (`qwen3.8-27b`, \$0.00 API Cost)

From the repository root (`/nvme/CognitiveMemoryLayer`):

```bash
# 1. Pre-ingest trajectories on GPU while the local LLM server is stopped
.venv/bin/python evaluation/scripts/run_longmemeval_v2_local.py \
  --tier small \
  --cml-url http://127.0.0.1:8000 \
  --preingest-only

# 2. Run CML in zero-VRAM hybrid retrieval mode + local Qwen3.8-27B (port 8023)
.venv/bin/python evaluation/scripts/run_longmemeval_v2_local.py \
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

Or via the standalone shell wrapper (`evaluation/scripts/run_cml.sh`) from inside `evaluation/longmemeval_v2/`.

### Full 451-Question `small`-Tier Results (`qwen3.8-27b` IQ3_S, 100% Local)

| Split / Metric | Web (`240`) | Enterprise (`211`) | Combined (`451`) |
| :--- | :---: | :---: | :---: |
| **Non-Abstention Overall (`323`)** | **59.52%** (`100/168`) | **54.19%** (`84/155`) | **56.97%** (`184/323`) |
| `procedure` (`74`) | **90.48%** (`38/42`) | **59.38%** (`19/32`) | **77.03%** (`57/74`) |
| `static` (`134`) | **48.33%** (`29/60`) | **56.76%** (`42/74`) | **52.99%** (`71/134`) |
| `gotchas` (`29`) | **66.67%** (`10/15`) | **35.71%** (`5/14`) | **51.72%** (`15/29`) |
| `dynamic` (`86`) | **45.10%** (`23/51`) | **51.43%** (`18/35`) | **47.67%** (`41/86`) |
| **Abstention Overall (`128`)** | **25.00%** (`18/72`) | **23.21%** (`13/56`) | **24.22%** (`31/128`) |
| **Full-Set Accuracy (`451`)** | **49.17%** (`118/240`) | **45.97%** (`97/211`) | **47.67%** (`215/451`) |
| **Avg Memory Query Latency** | `13.17s` | `27.03s` | `19.65s` |

Committed summary artifact: [`../results/longmemeval_v2/longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json`](../results/longmemeval_v2/longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json).

### Query-Privacy Verification

```bash
python -m unittest discover -s tests -v
```

## Submitting to Leaderboard

Leaderboard entries measure how much a memory system improves the released
baseline + AgentRunbook accuracy-latency frontier. The score is LAFS gain over
the fixed reference frontier, and a submission may include multiple latency
operating points for the same method and tier.

See [leaderboard/README.md](leaderboard/README.md) for the full packaging
instructions.

Submit leaderboard packages through the
[submission form](https://forms.gle/rxUpiuRKDERqpqSi9). Please do not submit
leaderboard entries as GitHub issues. Informal submission issues will be closed
or deleted.

## Citation


```bibtex
@article{wu2026longmemevalv2,
      title={LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues}, 
      author={Di Wu and Zixiang Ji and Asmi Kawatkar and Bryan Kwan and Jia-Chen Gu and Nanyun Peng and Kai-Wei Chang},
      year={2026},
      eprint={2605.12493},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.12493}, 
}
