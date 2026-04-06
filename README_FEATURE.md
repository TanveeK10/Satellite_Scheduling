# Satellite Constellation Stress-Testing (v12.0)
### High-Concurrency & Extreme Triage Benchmark

This branch (`feature/concurrent-scheduling`) evolves the Satellite Scheduling environment into a professional-grade constellation simulation. It is designed to test an AI agent's ability to coordinate a 15-satellite fleet across a 6-ground station network under severe bandwidth contention (100 Mbps).

---

## 🚀 Architectural Evolution

| Capability | Baseline (Master) | Stress-Test (v12.0) | Rationale |
| :--- | :--- | :--- | :--- |
| **Fleet Size** | 2-4 Satellites | **15 Satellites** | Tests high-concurrency coordination. |
| **GS Network** | 4 Stations | **6 Stations** | Increases antenna-mapping complexity. |
| **Downlink Rate**| 450 Mbps | **100 Mbps** | Forced "Slow Pipe" creates buffer backlogs. |
| **Action Model** | Serial (1 / tick) | **Batch (N / tick)** | Enables simultaneous multi-station usage. |
| **Emergency Load**| Single Injection | **8-Burst Clusters** | Demands immediate preemption of routine data. |

---

## 🛠️ Step-by-Step Operational Guide

### 1. Environment Initialization
Ensure your local environment is correctly configured for the 15-satellite scenarios.

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (OpenEnv + FastAPI Core)
pip install -e .
```

### 2. High-Fidelity Constellation Hosting
The server must be running to host the 15-satellite simulation.

```powershell
# Start the Satellite Environment Server (Port 7860)
# This handles the physics, weather dropouts, and batch de-confliction.
$env:SATELLITE_TASK = "task3"  # Set to Task 3 for the full 8-burst stress test
venv\Scripts\uvicorn src.envs.satellite_env.server.app:app --port 7860
```

### 3. Executing the AI Mission Controller
Run the agent following the mandatory OpenEnv telemetry format.

```powershell
# Set environment variables for the agent
$env:ENV_URL = "http://localhost:7860"
$env:API_BASE_URL = "http://localhost:11434/v1"  # Or your local Ollama/vLLM endpoint
$env:MODEL_NAME = "qwen2.5:7b-instruct-q4_k_m"

# Execute the 144-tick mission
venv\Scripts\python.exe inference.py > mission_output.log 2>&1
```

### 4. Real-Time Telemetry Tracking
Use a separate terminal to watch the "Congestion" behavior as the agent schedules the fleet.

```powershell
# Watch the live step-rewards and action distribution
Get-Content mission_output.log -Wait -Tail 20
```

---

## 🧹 System Reset & Maintenance
If the server hangs or you hit a "CAPACITY_REACHED" error (Windows WebSocket issue), run the following cleanup sequence:

```powershell
# Hard kill all uvicorn and python sessions
Get-Process | Where-Object { $_.ProcessName -match "uvicorn" -or $_.ProcessName -match "python" } | Stop-Process -Force -ErrorAction SilentlyContinue

# Verify port 7860 is cleared
netstat -ano | findstr :7860
```

---

## ⚠️ Known Issue: `KeyError: 'chunk'`
**Condition**: Occurs during Step 92+ of Task 1 or Task 3.
**Status**: **BLOCKED (Unsolved)**

**Technical Analysis**:
*   The new **8-burst cluster** scenario format uses a `chunks` (plural) array to support simultaneous alerts.
*   The legacy mission grader in `graders.py` is hard-coded to look for a singular `inj['chunk']` key.
*   **Result**: The simulation runs perfectly, but the "Mission Completion" score calculation fails when it hits a burst injection, resulting in a session crash.

---

## 🏆 Benchmark Scoring Criteria
*   **Mission Score**: Total P1+P2+P3 completion / Total Bytes available.
*   **Emergency Integrity**: Must download P3 data before the `deadline_min`.
*   **Concurrency Bonus**: Rewards given for using multiple ground stations in a single tick.
*   **Quality Bonus**: Rewards high-elevation passes (Peak passes > 60°).
