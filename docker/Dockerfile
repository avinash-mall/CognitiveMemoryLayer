# Cognitive Memory Layer - Multi-stage build
# Stage 1: build image with CUDA toolkit (nvcc) for compiling megablocks kernels.
# The devel variant includes nvcc, needed for megablocks CUDA extensions.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./
COPY pyproject.toml hatch_build.py ./
COPY packages/py-cml ./packages/py-cml

RUN pip install --no-cache-dir --break-system-packages -r requirements-runtime.txt \
    && pip install --no-cache-dir --break-system-packages "." \
    && pip install --no-cache-dir --break-system-packages "huggingface_hub>=0.25"

RUN python3 -m spacy download en_core_web_sm

RUN pip install --no-build-isolation --no-cache-dir --break-system-packages \
       stanford-stk git+https://github.com/nomic-ai/megablocks.git

# Stage 1b: Dashboard frontend build (neovis.js bundle for offline graph)
FROM node:20-alpine AS dashboard
WORKDIR /build
COPY src/dashboard/package.json src/dashboard/vite.config.js ./
COPY src/dashboard/static ./static
RUN npm install && npm run build

# Stage 2: production runtime image
# Uses the runtime variant (smaller, no nvcc) — still has CUDA shared libs
# so pre-compiled megablocks .so files work at runtime.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    CC=gcc \
    CXX=g++

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libpq5 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all installed Python packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src ./src
COPY --from=dashboard /build/static/js/bundle.js ./src/dashboard/static/js/bundle.js
COPY scripts ./scripts
COPY alembic.ini ./
COPY migrations ./migrations/
COPY docker/entrypoint.sh /entrypoint.sh

ENV PYTHONPATH=/app \
    NER__MODEL=en_core_web_sm \
    CML_MODELS_DIR=/app/packages/models/trained_models

# Create models directory so entrypoint can write to it.
RUN mkdir -p /app/packages/models/trained_models

# Non-root user for production
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/entrypoint.sh"]

# Health check (override in compose if needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Default: run API server (compose overrides for app test runner)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: test image with test dependencies and test sources
FROM builder AS test-dependencies

COPY requirements-test.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements-test.txt

FROM test-dependencies AS test

COPY pyproject.toml README.md hatch_build.py ./
COPY evaluation ./evaluation
COPY examples ./examples
COPY src ./src
COPY tests ./tests
COPY packages ./packages
COPY scripts ./scripts
COPY alembic.ini ./
COPY migrations ./migrations/
COPY docker/entrypoint.sh /entrypoint.sh

# Install package with the extras required by the full GitHub test suite.
RUN pip install --no-cache-dir --break-system-packages -e ".[embedded,eval,modeling]"

ENV PYTHONPATH=/app \
    NER__MODEL=en_core_web_sm \
    CML_MODELS_DIR=/app/packages/models/trained_models

ENTRYPOINT ["/entrypoint.sh"]

CMD ["pytest", "tests", "packages/py-cml/tests", "-v", "--tb=short"]

# Default build target remains runtime image.
FROM production AS final
