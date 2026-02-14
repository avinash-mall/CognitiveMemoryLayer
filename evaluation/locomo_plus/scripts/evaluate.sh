#!/bin/bash
# Run model on unified input (six categories), write predictions to JSON.
# Usage: from project root: ./evaluation/locomo_plus/scripts/evaluate.sh [model] [backend] [temp] [concurrency]
# Example: ./evaluation/locomo_plus/scripts/evaluate.sh gpt-4o-mini call_llm 0.3 4
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_PLUS_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LOCOMO_PLUS_DIR"
source scripts/env.sh

MODEL="${1:-mock}"
BACKEND="${2:-call_test}"
TEMP="${3:-0.3}"
CONCURRENCY="${4:-1}"
OUT_FILE="${OUT_DIR}/${QA_OUTPUT_FILE}"

if [[ "$BACKEND" == "call_test" && "$MODEL" != "mock" ]]; then
  BACKEND="call_llm"
fi

echo "Data file: $DATA_FILE_PATH"
echo "Output file: $OUT_FILE"
echo "Model: $MODEL | Backend: $BACKEND | Temperature: $TEMP | Concurrency: $CONCURRENCY"

python3 task_eval/evaluate_qa.py \
  --data-file "$DATA_FILE_PATH" \
  --out-file "$OUT_FILE" \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --temperature "$TEMP" \
  --concurrency "$CONCURRENCY"

echo "Done. Predictions written to $OUT_FILE"
