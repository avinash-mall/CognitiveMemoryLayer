#!/bin/bash
# Wait for GPU memory to free up (FAP and memory_type trainers to finish),
# then retrain schema_match_pair.
# Run from repo root: bash packages/models/scripts/wait_and_retrain_smp.sh

set -e

echo "[smp-retry] Waiting for GPU memory to be available..."
while true; do
    # Check if any cml.modeling.train processes are running (other than this script's python)
    PIDS=$(pgrep -f "cml.modeling.train" 2>/dev/null || true)
    if [ -z "$PIDS" ]; then
        echo "[smp-retry] No training processes detected. Starting schema_match_pair retraining..."
        break
    fi
    echo "[smp-retry] Training in progress (PIDs: $PIDS). Waiting 60s..."
    sleep 60
done

cd "$(dirname "$0")/../../.."
echo "[smp-retry] Running: python -m cml.modeling.train --tasks schema_match_pair"
python -m cml.modeling.train --tasks schema_match_pair
echo "[smp-retry] Done."
