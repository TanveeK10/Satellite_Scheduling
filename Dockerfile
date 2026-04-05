# Dockerfile (project root)
#
# Builds the Satellite Downlink Scheduler environment.
# Primary image for serving Task 1, 2, or 3 via SATELLITE_TASK env var.
#
# Build:
#     docker build -t satellite-env .
#
# Run (local, serving task1):
#     docker run -it -p 8000:8000 satellite-env
#
# Run (HF Spaces standard):
#     docker run -it -p 7860:7860 -e SATELLITE_TASK=task3 satellite-env

FROM python:3.12-slim

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
# NOTE: we do NOT copy the 'data' directory. Scenarios are 
# generated at runtime (or download logic could be added).
COPY src/ src/
COPY openenv.yaml .
COPY inference.py .

# Generate scenarios for all 3 tasks on startup
COPY scripts/ scripts/
RUN python scripts/generate_windows.py

# Expose port (7860 for HF Spaces / Challenge Standard)
EXPOSE 7860

# Metadata used by the runner to start the server
ENV PYTHONPATH=/app
ENV SATELLITE_TASK=task1
ENV SATELLITE_SEED=42

# Command to start the FastAPI server
CMD ["uvicorn", "src.envs.satellite_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
