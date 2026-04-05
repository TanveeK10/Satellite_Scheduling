# inference.py - Satellite Downlink Scheduler Baseline
# Strictly follows the OpenEnv mandatory logging format.

import json
import os
import sys
import textwrap
import time
from typing import List, Optional
from openai import OpenAI

# ── Imports (after path setup) ────────────────────────────────
# Add the project root to sys.path to resolve 'src' as a package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.envs.satellite_env.client import SatelliteEnv
from src.envs.satellite_env.models import SatelliteAction, SatelliteObservation

# ── Mandatory Environment Configuration ──────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b-instruct-q4_k_m")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# ── Inference Parameters ─────────────────────────────────────
MAX_STEPS = 144
TEMPERATURE = 0.2
MAX_TOKENS = 512
BENCHMARK = "satellite_downlink_scheduler"

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous satellite mission planner.
    You control a downlink scheduling environment.

    Your goal: maximise priority-weighted bytes downloaded from ALL satellites.
    Priority weights: routine=1, important=2, emergency=3.
    Emergency chunks (priority 3) have deadlines — download them FIRST.

    Available actions (respond with EXACTLY ONE JSON object):
        {"action_type": "schedule",  "sat_id": int, "station_id": int, "window_id": str}
        {"action_type": "preempt",   "schedule_id": str}
        {"action_type": "hold",      "sat_id": int}
        {"action_type": "noop"}

    PLANNING RULES:
    1. STATION CONSTRAINT: One station can only talk to ONE satellite at a time.
    2. REDUNDANCY: Do NOT re-schedule a satellite that is already in your CURRENT SCHEDULE.
    3. PRIORITY: Always check SATELLITES REMAINING and PRIORITY QUEUES. Follow deadlines strictly.
    4. NOOP (Wait Strategy): If SATELLITES REMAINING is not empty but TOP WINDOWS is empty, use {"action_type": "noop"} to fast-forward time until a satellite flies over a ground station.
    5. COVERAGE: You MUST schedule every satellite that has data (buf > 0). If windows are available in TOP WINDOWS, you are forbidden from using noop.
    6. TERMINATION: Only use noop repeatedly once SATELLITES REMAINING is empty.

    Respond with ONLY a valid JSON object. No explanation, no markdown fences.
""").strip()

# ── Helper functions for the automated judge ─────────────────

def _format_action_tag(action: SatelliteAction) -> str:
    """Format action as type(params) for the judge tag."""
    params = []
    if action.sat_id is not None: params.append(f"sat_id={action.sat_id}")
    if action.station_id is not None: params.append(f"station_id={action.station_id}")
    if action.window_id is not None: params.append(f"window_id={action.window_id}")
    if action.schedule_id is not None: params.append(f"schedule_id={action.schedule_id}")
    return f"{action.action_type}({', '.join(params)})"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: SatelliteAction, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_str = _format_action_tag(action)
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def _obs_to_prompt(obs: SatelliteObservation, step: int) -> str:
    # Compact serialisation for the agent
    avail = obs.station_availability
    # Filter out satellites with empty buffers — prevents -0.01 penalty loops
    buf_bytes = obs.satellite_buffer_bytes
    active_sats = {sid for sid, buf in buf_bytes.items() if buf > 0}
    max_buf = max(buf_bytes.values(), default=1)

    # Per-satellite diversity: pick best window for EACH active satellite,
    # then rank those representatives so the agent covers all satellites.
    best_per_sat: dict = {}
    for w in obs.pass_windows:
        sid_str = str(w.sat_id)
        if sid_str not in active_sats:
            continue
        score = w.link_quality * float(avail.get(str(w.station_id), 1.0))
        if sid_str not in best_per_sat or score > best_per_sat[sid_str][0]:
            best_per_sat[sid_str] = (score, w)

    ranked = sorted(
        best_per_sat.values(),
        key=lambda t: t[0] * (buf_bytes.get(t[1].window_id.split("_")[1].replace("s",""), 0) or max_buf) / max_buf,
        reverse=True
    )
    ranked = [t[1] for t in ranked][:8]

    windows_text = "\n".join(
        f"  id={w.window_id} sat={w.sat_id} stn={w.station_id} q={w.link_quality:.2f} elev={w.elevation_deg:.1f}deg buf={buf_bytes.get(str(w.sat_id),0)//1_000_000}MB"
        for w in ranked
    ) or "  (none)"
    
    # Critical: Metadata for emergency prioritization
    queues_text = []
    for sid, chunks in obs.data_priority_queues.items():
        if chunks:
            top = max(chunks, key=lambda c: c.priority)
            deadline = f" deadline={top.deadline_min}min" if top.deadline_min else ""
            queues_text.append(f"  sat{sid}: p{top.priority}{deadline}")
    
    queue_summary = "\n".join(queues_text) if queues_text else "  (empty)"
    
    # Current schedule
    sched_text = "\n".join(
        f"  {e.schedule_id}: sat{e.sat_id}->stn{e.station_id} tick={e.tick}"
        for e in obs.current_schedule[:5]
    ) or "  (empty)"
    
    # SATELLITES REMAINING summary — makes agent coverage obligation explicit
    remaining_sats = [
        f"sat{sid}={buf//1_000_000}MB" 
        for sid, buf in sorted(buf_bytes.items()) if buf > 0
    ]
    remaining_summary = ", ".join(remaining_sats) if remaining_sats else "ALL EMPTY — use noop"

    return textwrap.dedent(f"""
        Step {step} | t={obs.current_time_min}min | reward_so_far={obs.reward:.4f}
        SATELLITES REMAINING (must schedule all): {remaining_summary}
        TOP WINDOWS (one per satellite with data):
        {windows_text}
        PRIORITY QUEUES (Heads):
        {queue_summary}
        CURRENT SCHEDULE:
        {sched_text}
        EMERGENCY: {obs.info.get('emergency_injection', False)}
        LAST ERROR: {obs.info.get('action_error', 'none')}
    """).strip()

def get_action(client: OpenAI, obs: SatelliteObservation, step: int) -> SatelliteAction:
    user_prompt = _obs_to_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Basic JSON extraction
        if "{" in text and "}" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        
        data = json.loads(text)
        return SatelliteAction(**data)
    except Exception:
        return SatelliteAction(action_type="noop")

def run_task(client: OpenAI, task_name: str):
    rewards_list: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        with SatelliteEnv(base_url=ENV_URL).sync() as env:
            result = env.reset(task=task_name)
            obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                if obs.done:
                    break

                action = get_action(client, obs, step)
                
                result = env.step(action)
                obs = result.observation
                
                reward = obs.info.get("reward_last_tick", 0.0)
                done = obs.done
                error = obs.info.get("action_error") if obs.info.get("conflict") else None

                rewards_list.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action, reward=reward, done=done, error=error)

                if done:
                    break

            final_state = env.state()
            score = final_state.final_score
            score = min(max(score, 0.0), 1.0)  # clamp to [0, 1] as required
            success = score > 0.0 # Standard condition for success label

    except Exception as e:
        print(f"[DEBUG] Task failed: {e}", file=sys.stderr)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards_list)

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Run all three tasks as defined in the deployment guide
    for task in ["task1", "task2", "task3"]:
        run_task(client, task)

if __name__ == '__main__':
    main()
