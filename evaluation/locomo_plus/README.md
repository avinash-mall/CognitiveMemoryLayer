# Locomo-Plus Evaluation

[Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus) extends LoCoMo with a sixth task category **Cognitive**, evaluating whether models can connect a later *trigger query* to an earlier *cue dialogue* in multi-session conversations. Scoring uses LLM-as-judge (correct=1, partial=0.5, wrong=0).

## Layout

| Directory   | Contents                                                                 |
|------------|---------------------------------------------------------------------------|
| `data/`    | `build_conv.py`, `unified_input.py`, `locomo_plus.json`, `unified_input_samples_v2.json` |
| `task_eval/` | `evaluate_qa.py`, `llm_as_judge.py`, `prompt.py`, `utils.py`            |
| `scripts/` | `env.sh`, `env.local.sh.example`, `evaluate.sh`, `judge.sh`, `run_evaluate.py`, `run_judge.py` |

`locomo10.json` lives in `evaluation/locomo_plus/data/` (LoCoMo factual data for unified sample building).

## Environment

- **Generation / Evaluate / Judge**: `OPENAI_API_KEY` (required for `call_llm`). Optional: `OPENAI_BASE_URL` (e.g. for Ollama-compatible endpoint).
- Copy `scripts/env.local.sh.example` to `scripts/env.local.sh` and set your API keys (do not commit).

## Quick start

1. **Build unified input** (LoCoMo 5 categories + Cognitive):

   ```bash
   cd evaluation/locomo_plus/data && python unified_input.py
   ```

2. **Run evaluation** (without CML, using OpenAI-style API):

   **Unix:**
   ```bash
   export PYTHONPATH=evaluation/locomo_plus
   ./evaluation/locomo_plus/scripts/evaluate.sh gpt-4o-mini call_llm 0.3 4
   ./evaluation/locomo_plus/scripts/judge.sh
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:PYTHONPATH = "evaluation/locomo_plus"
   python evaluation/locomo_plus/scripts/run_evaluate.py --backend call_llm --model gpt-4o-mini
   python evaluation/locomo_plus/scripts/run_judge.py --model gpt-4o-mini
   ```

3. **Run CML-backed evaluation** (ingest into CML, QA via CML read + LLM from .env, judge):

   From project root:

   ```powershell
   $env:PYTHONPATH = "evaluation/locomo_plus"
   python evaluation/scripts/eval_locomo_plus.py --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json --out-dir evaluation/outputs --limit-samples 5
   ```

   CML server config (embedding model, rate limit, optional `LLM_INTERNAL__*`) is in the project root `.env`. See [evaluation/README](../../evaluation/README.md) and [RunEvaluation](../../ProjectPlan/LocomoEval/RunEvaluation.md) for setup.

## Outputs

- `evaluation/outputs/locomo_plus_predictions.json` – predictions (from run_evaluate or eval_locomo_plus)
- `evaluation/outputs/locomo_plus_judged.json` – judged records (judge_label, judge_reason, judge_score)
- `evaluation/outputs/locomo_plus_qa_cml_predictions.json` – CML-backed predictions
- `evaluation/outputs/locomo_plus_qa_cml_judged.json` – CML-backed judged output
