#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
METHOD="${CML_METHOD:-cml}"
DATA_ROOT_VALUE="${DATA_ROOT:-$REPO_ROOT/data/longmemeval-v2}"
OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-$REPO_ROOT/../../outputs/longmemeval_v2}"
TIER_VALUE="${TIER:-small}"

for arg in "$@"; do
  case "$arg" in
    --method|--method=*|--data-root|--data-root=*|--domain|--domain=*|--tier|--tier=*|--output-dir|--output-dir=*)
      echo "This wrapper owns --method, --data-root, --domain, --tier, and --output-dir. Set CML_METHOD, DATA_ROOT, TIER, or OUTPUT_ROOT instead." >&2
      exit 2
      ;;
  esac
done

for domain in web enterprise; do
  python3 "$REPO_ROOT/evaluation/run_eval.py" \
    --method "$METHOD" \
    --data-root "$DATA_ROOT_VALUE" \
    --domain "$domain" \
    --tier "$TIER_VALUE" \
    --output-dir "$OUTPUT_ROOT_VALUE/${METHOD}_${domain}_${TIER_VALUE}" \
    "$@"
done

python3 "$REPO_ROOT/leaderboard/combine_aggregated_metrics.py" \
  "$OUTPUT_ROOT_VALUE/${METHOD}_enterprise_${TIER_VALUE}/aggregated_metrics.json" \
  "$OUTPUT_ROOT_VALUE/${METHOD}_web_${TIER_VALUE}/aggregated_metrics.json" \
  -o "$OUTPUT_ROOT_VALUE/${METHOD}_${TIER_VALUE}_combined_metrics.json"
