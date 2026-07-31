# Cognitive Memory Layer - Multi-stage build
# Stage 1: builder — installs runtime deps into site-packages.
# No CUDA toolkit: torch's pip wheels vendor their own CUDA runtime libraries,
# and GPU access comes from nvidia-container-toolkit at run time (see
# docker-compose.gpu.yml), so nothing here needs nvcc.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./
COPY pyproject.toml hatch_build.py ./
COPY packages/py-cml ./packages/py-cml

RUN pip install -r requirements-runtime.txt \
    && pip install "."

# Stage 2: production runtime image
FROM python:3.12-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all installed Python packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src ./src
COPY scripts ./scripts
COPY alembic.ini ./
COPY migrations ./migrations/
COPY docker/entrypoint.sh /entrypoint.sh

# Model/tokenizer caches live on volumes so weights survive container recreation.
# The dirs must exist and be appuser-owned before the USER switch: Docker gives a
# fresh named volume the ownership of its mount point in the image, so creating
# them later (or not at all) makes every download fail with EACCES.
ENV PYTHONPATH=/app \
    HF_HOME=/cache/hf \
    TIKTOKEN_CACHE_DIR=/cache/tiktoken

RUN useradd -m appuser \
    && mkdir -p /cache/hf /cache/tiktoken \
    && chown -R appuser:appuser /app /cache
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
RUN pip install -r requirements-test.txt

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
# The repo-root markers find_repo_root() scans for include docker/docker-compose.yml,
# so /app must look like a real checkout or eval path defaults silently become None.
COPY docker ./docker
COPY docker/entrypoint.sh /entrypoint.sh

# Install package with the extras required by the full GitHub test suite.
RUN pip install -e ".[embedded,eval]"

ENV PYTHONPATH=/app

ENTRYPOINT ["/entrypoint.sh"]

CMD ["pytest", "tests", "packages/py-cml/tests", "-v", "--tb=short"]

# Default build target remains runtime image.
FROM production AS final
