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
        MODELS_DIR="$MODELS_DIR" HF_REPO="$HF_REPO" HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" python3 - <<'PY' || echo "[entrypoint] WARNING: Model download failed; server will start with heuristic fallbacks."
import os
import sys
from huggingface_hub import snapshot_download

models_dir = os.environ["MODELS_DIR"]
repo_id = os.environ["HF_REPO"]
token = os.environ.get("HF_TOKEN") or None

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=models_dir,
        token=token,
        allow_patterns=["*.joblib", "*.json", "*.safetensors", "*.txt", "*.bin"],
        ignore_patterns=["*.parquet", "*.csv", "*.arrow", "*.md"],
    )
except PermissionError as exc:
    target = exc.filename or models_dir
    print(
        f"[entrypoint] ERROR: Model artifacts directory is not writable at {target}. "
        "If this path is bind-mounted from the host, ensure the host files are writable by "
        "the user running CML (for local Docker Compose from the repo root: "
        "sudo chown -R $(id -u):$(id -g) packages/models/trained_models).",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

print("[entrypoint] Model download complete.")
PY
    else
        echo "[entrypoint] Models already present in $MODELS_DIR — skipping download."
    fi
else
    echo "[entrypoint] Auto-download disabled (CML_MODELS_AUTO_DOWNLOAD=$AUTO_DOWNLOAD)."
fi

# Hand off to the container CMD (e.g. uvicorn, pytest).
exec "$@"
