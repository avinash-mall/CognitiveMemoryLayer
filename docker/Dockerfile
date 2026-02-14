# Cognitive Memory Layer - Multi-stage build (Phase 10)
# Stage 1: Base image and system deps
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base AS dependencies

COPY pyproject.toml README.md ./
COPY alembic.ini ./
COPY migrations ./migrations/
COPY requirements-docker.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Stage 3: Production image
FROM base AS production

COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

COPY pyproject.toml README.md hatch_build.py ./
COPY src ./src
COPY tests ./tests
COPY packages ./packages
COPY scripts ./scripts
COPY alembic.ini ./
COPY migrations ./migrations/

# Install package in editable form so src is on path (no extra deps)
RUN pip install --no-cache-dir -e . --no-deps

ENV PYTHONPATH=/app

# Non-root user for production
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check (override in compose if needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Default: run API server (compose overrides for app test runner)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
