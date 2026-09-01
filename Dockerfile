# ─── Stage 1: Build Dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Pre-install CPU-only PyTorch to avoid massive CUDA overhead in container,
# then install remaining pinned dependencies into isolated virtual environment.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ─── Stage 2: Production Runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy isolated virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Prepare writable runtime data directories with non-root ownership
RUN mkdir -p /app/data/forms && chown -R appuser:appgroup /app

# Copy application source code, scripts, and authoritative statutory PDF
COPY --chown=appuser:appgroup backend/ /app/backend/
COPY --chown=appuser:appgroup scripts/ /app/scripts/
COPY --chown=appuser:appgroup data/forms/forms_manifest.json /app/data/forms/forms_manifest.json
COPY --chown=appuser:appgroup ["BNS bare act 2023.pdf", "/app/BNS bare act 2023.pdf"]

# Pre-generate 58 statutory form PDFs at build time into /app/data/forms/
RUN python scripts/extract_forms.py --pdf-path "/app/BNS bare act 2023.pdf" --output-dir "/app/data/forms" && \
    chown -R appuser:appgroup /app/data/forms

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
