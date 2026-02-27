# Cognitive Memory Layer - Multi-stage build
# Stage 1: base image and system deps
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

# Stage 2: runtime dependencies
FROM base AS runtime-dependencies

COPY requirements-runtime.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-runtime.txt \
    && python -m spacy download en_core_web_sm

# Stage 2b: Dashboard frontend build (neovis.js bundle for offline graph)
FROM node:20-alpine AS dashboard
WORKDIR /build
COPY src/dashboard/package.json src/dashboard/vite.config.js ./
COPY src/dashboard/static ./static
RUN npm install && npm run build

# Stage 3: production runtime image (no tests/training/evaluation tooling)
FROM base AS production

COPY --from=runtime-dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=runtime-dependencies /usr/local/bin /usr/local/bin

COPY src ./src
COPY --from=dashboard /build/static/js/bundle.js ./src/dashboard/static/js/bundle.js
COPY scripts ./scripts
COPY alembic.ini ./
COPY migrations ./migrations/

ENV PYTHONPATH=/app \
    NER__MODEL=en_core_web_sm

# Non-root user for production
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check (override in compose if needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Default: run API server (compose overrides for app test runner)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: test image with test dependencies and test sources
FROM runtime-dependencies AS test-dependencies

COPY requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements-test.txt

FROM test-dependencies AS test

COPY pyproject.toml README.md hatch_build.py ./
COPY src ./src
COPY tests ./tests
COPY packages ./packages
COPY scripts ./scripts
COPY alembic.ini ./
COPY migrations ./migrations/

# Install package in editable form so src/packages are importable in tests.
RUN pip install --no-cache-dir -e . --no-deps

ENV PYTHONPATH=/app \
    NER__MODEL=en_core_web_sm

CMD ["pytest", "tests", "packages/py-cml/tests", "-v", "--tb=short"]

# Default build target remains runtime image.
FROM production AS final
