---
title: Satellite Downlink Scheduler
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - scheduling
  - satellite
  - real-world
  - reinforcement-learning
short_description: OpenEnv environment — LEO satellite downlink scheduling (v12.0 Stable)
---

# Satellite Downlink Scheduler 🛰️

This project simulates a real-world operational challenge: scheduling contact windows for a mass-constellation of satellites to maximize data download. It is built for the *Meta-OpenEnv Hackathon*.

---

## 1. Project Overview

The simulation is a clock-driven environment where time moves in 10-minute ticks. A full episode represents a 24-hour planning horizon (144 ticks).

### The Physical Model (v12.0)

* **Constellation (15 Satellites)**: High-concurrency fleet with diverse onboard data buffers.
* **Ground Network (6 Stations)**: Svalbard, McMurdo, Bangalore, Fairbanks, Singapore, and Perth. 
* **Downlink Model**: 100 Mbps "Slow Pipe" creates realistic bandwidth contention and buffer backlogs.
* **Action Model**: **Batch (N / tick)** — The agent can coordinate multiple simultaneous downlinks across the entire ground network in a single step.

### The Challenges

* **High-Concurrency Contention**: 15 satellites competing for 6 shared ground station antennas.
* **Weather (Tasks 2 & 3)**: Dynamically degrades link quality based on local conditions.
* **Priority-Based Triage**: Routine (w=1), Important (w=10), and Emergency (w=100) data layers.
* **Emergency Bursts (Task 3)**: Simultaneous **8-Burst Clusters** of urgent data with strict deadlines and late-delivery penalties.

---

## 2. Project Structure (Submission-Ready)

```text
Satellite/
├── Dockerfile                # REQUIRED (root-level)
├── requirements.txt          # REQUIRED (dependencies for Docker)
├── pyproject.toml
├── inference.py              # REQUIRED (entry for evaluation)
├── openenv.yaml              # REQUIRED (OpenEnv manifest)
├── README.md
├── .dockerignore
├── .gitignore
├── data/                     # REQUIRED (precomputed scenarios)
└── src/
    └── envs/
        └── satellite_env/
            ├── models.py     # Shared Pydantic models
            ├── client.py     # OpenEnv client implementation
            └── server/
                ├── app.py           # FastAPI entrypoint
                ├── environment.py   # Core logic
                ├── graders.py       # Scoring functions
                ├── scheduler.py     # Conflict detection
                └── weather.py       # Weather simulation
```

---

## 3. API Requirements (MANDATORY FOR VALIDATION)

The server MUST expose the following REST endpoints:

* **POST /reset** → Initialize environment
* **POST /step** → Execute action
* **GET /state** → Return current state
* **GET /health** → Docker health check

⚠️ WebSocket (`/ws`) support is available but does NOT replace REST endpoints.

---

## 4. Installation & Setup

### Prerequisites

* Python 3.11+
* Docker
* (Optional) Ollama for local LLM testing

### Local Setup

```bash
# 1. Clone and install
git clone <your-repo>
cd Satellite
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate

# 2. Install editable package
pip install -e .
```

---

## 5. Running Locally

### Terminal 1: Environment Server

The server handles physics, weather dropouts, and batch de-confliction.

```powershell
# Windows PowerShell
$env:SATELLITE_TASK = "task3"  # Set to Task 3 for the full 15-sat stress test
venv\Scripts\uvicorn src.envs.satellite_env.server.app:app --port 7860

# Bash (Mac/Linux)
export SATELLITE_TASK=task3
uvicorn src.envs.satellite_env.server.app:app --port 7860
```

### Terminal 2: Inference Agent

Run the autonomous mission controller using a local LLM or HF router.

```powershell
# Windows PowerShell
$env:ENV_URL = "http://localhost:7860"
$env:API_BASE_URL = "http://localhost:11434/v1"  # Ollama endpoint
$env:MODEL_NAME = "qwen2.5:7b-instruct-q4_k_m"

venv\Scripts\python.exe inference.py
```

---

## 6. Environment Variables

| Variable | Purpose |
| :--- | :--- |
| `API_BASE_URL` | LLM endpoint (Ollama or HF router) |
| `MODEL_NAME` | Model used for inference |
| `HF_TOKEN` | Authentication token |
| `SATELLITE_TASK` | Task to run (task1, task2, task3) |

---

## 11. Mandatory Logging Format

Your `inference.py` MUST emit EXACTLY:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

⚠️ Any deviation will cause evaluation failure.