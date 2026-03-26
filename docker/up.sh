#!/usr/bin/env bash
# docker/up.sh — auto-detects NVIDIA GPU and applies the GPU override when available.
#
# Usage (same as `docker compose`):
#   ./docker/up.sh up api
#   ./docker/up.sh up -d postgres neo4j redis
#   ./docker/up.sh down
#   ./docker/up.sh logs -f api
#
# GPU detection: nvidia-smi must be on PATH and succeed.
# Override: set GPU=1 to force GPU, GPU=0 to force CPU-only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="-f ${SCRIPT_DIR}/docker-compose.yml"
GPU_OVERRIDE="${SCRIPT_DIR}/docker-compose.gpu.yml"

# Respect explicit override first
if [[ "${GPU:-}" == "1" ]]; then
    echo "[docker/up.sh] GPU=1 — forcing GPU mode"
    GPU_FLAG="-f ${GPU_OVERRIDE}"
elif [[ "${GPU:-}" == "0" ]]; then
    echo "[docker/up.sh] GPU=0 — forcing CPU-only mode"
    GPU_FLAG=""
elif command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "[docker/up.sh] GPU detected: ${GPU_NAME} — enabling GPU mode"
    GPU_FLAG="-f ${GPU_OVERRIDE}"
else
    echo "[docker/up.sh] No GPU detected — running CPU-only"
    GPU_FLAG=""
fi

exec docker compose ${BASE} ${GPU_FLAG} "$@"
