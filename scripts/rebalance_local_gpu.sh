#!/usr/bin/env bash
# Rebalance RTX 5070 Ti (16.3 GB VRAM) for CML v3.0 Whole-Brain Architecture:
# Frees ~2.6-3.2 GB VRAM for GPU Cross-Encoder (bge-reranker-v2-m3 FP16) + GPU Embeddings
# while enabling 3-way parallel continuous batching (--parallel 3) and keeping
# speculative decoding (dflash) + multimodal vision projector (--mmproj).
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-/nvme/qwen3.8/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/nvme/qwen3.8/models/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf}"
DRAFT_PATH="${DRAFT_PATH:-/nvme/qwen3.8/models/dflash-Qwen3.8-27B-Q4_0.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-/nvme/qwen3.8/models/mmproj-Qwen3.8-27B-Q8_0.gguf}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8000}"
NGL="${LLAMA_NGL:-99}"
CTX_SIZE="${LLAMA_CTX_SIZE:-8192}"
PARALLEL="${LLAMA_PARALLEL:-1}"
LOG_FILE="${LLAMA_LOG:-/tmp/llama_server_rebalanced.log}"

echo "[rebalance_local_gpu] Stopping existing llama-server on port ${PORT} (if running)..."
pkill -f "llama-server.*--port ${PORT}" || true
sleep 2

echo "[rebalance_local_gpu] Launching llama-server (-ngl ${NGL}, --ctx-size ${CTX_SIZE}, --parallel ${PARALLEL})..."
nohup "${LLAMA_BIN}" \
  -m "${MODEL_PATH}" \
  --alias qwen3.8-27b \
  --host "${HOST}" \
  --port "${PORT}" \
  -ngl "${NGL}" \
  -fa on \
  --ctx-size "${CTX_SIZE}" \
  --parallel "${PARALLEL}" \
  --batch-size 512 \
  --ubatch-size 256 \
  --cache-type-k f8_e4m3 \
  --cache-type-v f8_e4m3 \
  --cache-ram 4096 \
  --jinja \
  --reasoning-format deepseek \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --metrics \
  --fit off \
  -md "${DRAFT_PATH}" \
  -ngld 99 \
  -ctkd q8_0 \
  -ctvd q8_0 \
  --spec-type draft-dflash \
  --spec-draft-n-max 5 \
  --spec-draft-n-ubatch 32 \
  --spec-draft-sampling probabilistic \
  --mmproj "${MMPROJ_PATH}" \
  > "${LOG_FILE}" 2>&1 &

echo "[rebalance_local_gpu] Waiting for http://${HOST}:${PORT}/health ..."
for _ in $(seq 1 90); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "[rebalance_local_gpu] llama-server is healthy!"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
    exit 0
  fi
  sleep 2
done

echo "[rebalance_local_gpu] ERROR: llama-server did not become healthy within 180s. Tail of ${LOG_FILE}:" >&2
tail -n 40 "${LOG_FILE}" >&2
exit 1
