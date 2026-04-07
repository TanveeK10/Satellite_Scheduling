# src/envs/satellite_env/server/app.py
"""
FastAPI server entry point.

Three lines — that's all app.py ever needs to be.
create_fastapi_app() auto-generates all endpoints:
    GET  /health
    POST /reset
    POST /step
    GET  /state
    WS   /ws        ← primary transport used by the client
    GET  /docs      ← OpenAPI docs
    GET  /web       ← Gradio debug UI (when ENABLE_WEB_INTERFACE=1)

Start locally:
    uvicorn satellite_env.server.app:app --host 0.0.0.0 --port 8000 --reload

Inside Docker (HF Spaces port):
    uvicorn satellite_env.server.app:app --host 0.0.0.0 --port 7860
"""

import os
from openenv.core.env_server import create_fastapi_app

from src.envs.satellite_env.models import SatelliteAction, SatelliteObservation
from src.envs.satellite_env.server.environment import SatelliteEnvironment

# Task is selected via environment variable so the same Docker image
# serves all three tasks — judges switch tasks by changing SATELLITE_TASK.
# Defaults to task1 so a bare `docker run` works out of the box.
_task = os.getenv("SATELLITE_TASK", "task1").strip()
_seed = int(os.getenv("SATELLITE_SEED", "42").strip())

# Global persistent instance to support dynamic task switching via reset()
shared_env = SatelliteEnvironment(task=_task, seed=_seed)

app = create_fastapi_app(
    lambda: shared_env,
    SatelliteAction,
    SatelliteObservation,
)

@app.get("/")
def welcome():
    """Friendly welcome message for HF Spaces."""
    return {
        "status": "ready",
        "project": "Satellite Downlink Scheduler",
        "benchmark": "Meta-OpenEnv",
        "author": "Dhruvk14",
        "endpoints": ["/reset", "/step", "/state"]
    }

def create_app():
    """Entry point for [project.scripts] — returns the FastAPI app."""
    return app


# --- Required for OpenEnv Multi-mode deployment
def main():
    pass

if __name__ == '__main__':
    main()
