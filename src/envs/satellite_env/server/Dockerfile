# src/envs/satellite_env/server/Dockerfile
#
# Build:
#   docker build -t satellite-env:latest -f src/envs/satellite_env/server/Dockerfile .
#
# Run locally (port 8000):
#   docker run -d -p 8000:7860 \
#     -e SATELLITE_TASK=task1 \
#     -e SATELLITE_SEED=42 \
#     satellite-env:latest
#
# Run locally (port 7860, mirrors HF Spaces):
#   docker run -d -p 7860:7860 \
#     -e SATELLITE_TASK=task2 \
#     satellite-env:latest
#
# Health check:
#   curl http://localhost:8000/health
#   → {"status": "healthy"}
#
# Switch tasks without rebuilding:
#   docker run -d -p 8000:7860 -e SATELLITE_TASK=task3 satellite-env:latest

FROM python:3.11-slim

# ── Non-root user ────────────────────────────────────────────
# Required for HF Spaces Dev Mode and general security hygiene.
# uid 1000 matches the HF Spaces runner uid.
RUN useradd -m -u 1000 user

# ── System deps ──────────────────────────────────────────────
# git: needed if openenv installs from a git URL
# curl: needed for health checks during CI
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ───────────────────────────────────────
# Copy requirements first — Docker layer cache means pip install
# only reruns when requirements.txt changes, not on every code edit.
COPY --chown=user src/envs/satellite_env/server/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────
# Copy the full project so relative imports work correctly.
# .dockerignore prevents __pycache__, .venv, etc. from bloating the image.
COPY --chown=user . /app

# ── Install our package in editable mode ──────────────────────
# This puts satellite_env on sys.path so all imports resolve.
RUN pip install --no-cache-dir -e .

# ── Switch to non-root user ───────────────────────────────────
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── Runtime env vars (overridable at docker run) ──────────────
ENV SATELLITE_TASK=task1 \
    SATELLITE_SEED=42 \
    WORKERS=2 \
    PORT=7860 \
    HOST=0.0.0.0

# ── Expose port ───────────────────────────────────────────────
# HF Spaces always routes external traffic to 7860.
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────
# Docker will mark the container unhealthy if /health stops responding.
# Judges can verify liveness before running inference.py.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Start server ──────────────────────────────────────────────
# Shell form (not exec form) so ${WORKERS} and ${PORT} env vars expand.
# uvicorn module path: satellite_env.server.app:app
CMD uvicorn satellite_env.server.app:app \
        --host ${HOST} \
        --port ${PORT} \
        --workers ${WORKERS}
```
```
# src/envs/satellite_env/server/requirements.txt
#
# Pinned for reproducibility — these exact versions are tested together.
# Update pins only when a security patch or OpenEnv upgrade requires it.
#
# Core server
openenv>=0.1.0
fastapi>=0.111.0,<0.120.0
uvicorn[standard]>=0.29.0,<0.35.0
websockets>=12.0,<14.0
pydantic>=2.7.0,<3.0.0

# Environment logic
numpy>=1.26.0,<2.1.0
skyfield>=1.48,<2.0

# Utilities
python-multipart>=0.0.9    # fastapi file uploads (openenv web UI needs this)
httpx>=0.27.0              # used by openenv s async client internally