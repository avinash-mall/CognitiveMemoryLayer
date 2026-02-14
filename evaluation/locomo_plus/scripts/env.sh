# Locomo-Plus evaluation env
# Path to unified input JSON (from data/unified_input.py output)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCOMO_PLUS_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_ROOT="$(dirname "$LOCOMO_PLUS_DIR")"

DATA_FILE_PATH="${LOCOMO_PLUS_DIR}/data/unified_input_samples_v2.json"
OUT_DIR="${EVAL_ROOT}/outputs"
QA_OUTPUT_FILE=locomo_plus_predictions.json

# Load env.local.sh for API keys (do not commit)
if [ -f "${SCRIPT_DIR}/env.local.sh" ]; then
  source "${SCRIPT_DIR}/env.local.sh"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
