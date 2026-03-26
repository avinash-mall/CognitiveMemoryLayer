#!/usr/bin/env bash
# CML Docker entrypoint — downloads model weights from HuggingFace Hub
# if they are not already present, then executes the container command.
set -e

MODELS_DIR="${CML_MODELS_DIR:-/app/packages/models/trained_models}"
HF_REPO="${CML_MODELS_HF_REPO:-avinashm/CognitiveMemoryLayer-models}"
AUTO_DOWNLOAD="${CML_MODELS_AUTO_DOWNLOAD:-true}"

# Check if models are already available (manifest + 3 family models).
models_present() {
    [ -f "$MODELS_DIR/manifest.json" ] && \
    [ -f "$MODELS_DIR/router_model.joblib" ] && \
    [ -f "$MODELS_DIR/extractor_model.joblib" ] && \
    [ -f "$MODELS_DIR/pair_model.joblib" ]
}

if [ "$AUTO_DOWNLOAD" = "true" ] || [ "$AUTO_DOWNLOAD" = "1" ]; then
    if ! models_present; then
        echo "[entrypoint] Models not found in $MODELS_DIR — downloading from $HF_REPO ..."
        mkdir -p "$MODELS_DIR"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    local_dir='${MODELS_DIR}',
    token=None,  # public repo; set HF_TOKEN env if private
    allow_patterns=['*.joblib', '*.json', '*.safetensors', '*.txt', '*.bin'],
    ignore_patterns=['*.parquet', '*.csv', '*.arrow', '*.md'],
)
print('[entrypoint] Model download complete.')
" || echo "[entrypoint] WARNING: Model download failed; server will start with heuristic fallbacks."
    else
        echo "[entrypoint] Models already present in $MODELS_DIR — skipping download."
    fi
else
    echo "[entrypoint] Auto-download disabled (CML_MODELS_AUTO_DOWNLOAD=$AUTO_DOWNLOAD)."
fi

# Hand off to the container CMD (e.g. uvicorn, pytest).
exec "$@"
