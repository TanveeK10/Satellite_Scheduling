# Satellite Downlink Scheduler - Dockerfile
# Optimized for Hugging Face Spaces (Docker SDK)

FROM python:3.11-slim

# ── Non-root user ────────────────────────────────────────────
# Required for HF Spaces Dev Mode and security.
# uid 1000 matches the HF Spaces runner uid.
RUN useradd -m -u 1000 user

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ───────────────────────────────────────
# Copy requirements from root
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────
# .dockerignore prevents bloating the image with venv/git
COPY --chown=user . /app

# ── Generate scenarios ────────────────────────────────────────
# Required for Task 1/2/3 to have JSON window data available.
RUN python scripts/generate_windows.py

# ── Install the package in editable mode ──────────────────────
# This allows 'import src.envs.satellite_env' to work as a package
RUN pip install --no-cache-dir -e .

# ── Switch to non-root user ───────────────────────────────────
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── Runtime env vars ──────────────────────────────────────────
ENV SATELLITE_TASK=task1 \
    SATELLITE_SEED=42 \
    WORKERS=2 \
    PORT=7860 \
    HOST=0.0.0.0

# ── Expose port ───────────────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Start server ──────────────────────────────────────────────
# Using the module path relative to the root
#CMD uvicorn src.envs.satellite_env.server.app:app \
#        --host ${HOST} \
#        --port ${PORT} \
#        --workers ${WORKERS}
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]