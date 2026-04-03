# inference.py  (project root — mandatory name and location)
"""
Baseline inference script for the Satellite Downlink Scheduling environment.

Mandatory env vars (set before running):
    API_BASE_URL   — LLM endpoint  e.g. https://router.huggingface.co/v1
                                    or  http://localhost:11434/v1  (Ollama)
    MODEL_NAME     — model id      e.g. meta-llama/Llama-3.1-8B-Instruct
    HF_TOKEN       — API key       (your HF token, or "ollama" for local)

Optional env vars:
    ENV_URL        — environment server URL
                     default: http://localhost:8000
                     production: https://<your-hf-username>-satellite-env.hf.space
    MAX_STEPS      — max ticks per episode (default: 144)
    TEMPERATURE    — LLM temperature      (default: 0.2)
    DEBUG          — set to "1" for verbose per-step output
"""

import os
import re
import sys
import textwrap
import time
from openai import OpenAI

from src.envs.satellite_env.client import SatelliteEnv
from src.envs.satellite_env.models import SatelliteAction, SatelliteObservation

sys.path.insert(0, "src")

# ── Mandatory env vars ────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "qwen:latest")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "ollama")

# ── Optional env vars ─────────────────────────────────────────
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:8000")
MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "144"))
TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0.2"))
DEBUG: bool = os.environ.get("DEBUG", "0") == "1"
SEED: int = 42

_MISSING = [v for v, val in [
    ("API_BASE_URL", API_BASE_URL),
    ("MODEL_NAME", MODEL_NAME),
    ("HF_TOKEN", HF_TOKEN),
] if not val]
if _MISSING:
    print(f"[ERROR] Missing required env vars: {', '.join(_MISSING)}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# System prompt — numbered choice (prevents window_id hallucination)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a satellite downlink scheduler. At each step you see a numbered list
    of available contact windows ranked by quality. Pick the BEST one to schedule.

    Respond with ONLY a single number (the window choice), or 0 for no action.
    No JSON, no explanation, just the number.

    Strategy:
    - Pick windows for satellites with the most high-priority data
    - Prefer higher link quality and station availability
    - If an EMERGENCY flag is set, prioritize emergency data immediately
    - Respond 0 only when no good windows exist
""").strip()


# ─────────────────────────────────────────────────────────────
# Build ranked window choices from the observation
# ─────────────────────────────────────────────────────────────

def _build_choices(obs: SatelliteObservation) -> list[dict]:
    """Build a ranked list of actionable window choices."""
    avail = obs.station_availability
    choices = []

    for w in obs.pass_windows:
        # Skip satellites with no data to download
        sat_buffer = obs.satellite_buffer_bytes.get(str(w.satellite_id), 0)
        if sat_buffer <= 0:
            continue

        sta_avail = float(avail.get(str(w.station_id), 1.0))
        score = w.link_quality * sta_avail * (w.max_bytes / 1e9)

        sat_chunks = obs.data_priority_queues.get(str(w.satellite_id), [])
        max_pri = max((c.priority for c in sat_chunks), default=1)
        has_deadline = any(c.deadline_min is not None for c in sat_chunks)
        priority_boost = max_pri * (3.0 if has_deadline else 1.0)
        score *= priority_boost

        choices.append({
            "window": w,
            "score": score,
            "sta_avail": sta_avail,
            "max_pri": max_pri,
            "has_deadline": has_deadline,
            "buffer_mb": sat_buffer / 1e6,
        })

    choices.sort(key=lambda c: c["score"], reverse=True)
    return choices[:8]


# ─────────────────────────────────────────────────────────────
# Observation → compact prompt with numbered choices
# ─────────────────────────────────────────────────────────────

def _obs_to_prompt(obs: SatelliteObservation, step: int,
                   history: list[str], choices: list[dict]) -> str:
    choices_text = ""
    for i, c in enumerate(choices, 1):
        w = c["window"]
        choices_text += (
            f"  [{i}] sat{w.satellite_id}→stn{w.station_id}  "
            f"quality={w.link_quality:.2f}  avail={c['sta_avail']:.2f}  "
            f"bytes={w.max_bytes / 1e6:.0f}MB  "
            f"pri={c['max_pri']}"
            + ("  ⚠DEADLINE" if c["has_deadline"] else "")
            + "\n"
        )
    if not choices_text:
        choices_text = "  (no windows)\n"

    emg = "⚠ EMERGENCY — prioritize satellite with priority-3 data NOW\n" if obs.info.get("emergency_injection") else ""
    conflict = f"⚠ Last action rejected: {obs.info.get('action_error')}\n" if obs.info.get("conflict") else ""

    buffers = {k: v for k, v in obs.satellite_buffer_bytes.items() if v > 0}
    buf_text = "  ".join(f"s{k}={v/1e6:.0f}MB" for k, v in sorted(buffers.items()))

    hist = "\n".join(f"  {h}" for h in history[-3:]) or "  (none)"

    return (
        f"Step {step} | t={obs.current_time_min}min | reward={obs.reward:.4f}\n"
        f"{emg}{conflict}"
        f"\nBuffers: {buf_text}\n"
        f"\nWindows (pick number, or 0 for noop):\n{choices_text}"
        f"\nRecent:\n{hist}\n"
        f"\nRespond with ONLY a number (1-{len(choices)}) or 0:"
    )


# ─────────────────────────────────────────────────────────────
# Parse the LLM's numeric choice → SatelliteAction
# ─────────────────────────────────────────────────────────────

FALLBACK_ACTION = SatelliteAction(action_type="noop")


def _parse_choice(response_text: str, choices: list[dict]) -> SatelliteAction:
    """Parse LLM's number → schedule action for that window."""
    text = response_text.strip()
    match = re.search(r'\d+', text)
    if not match:
        return FALLBACK_ACTION

    n = int(match.group())
    if n == 0 or n > len(choices):
        return FALLBACK_ACTION

    w = choices[n - 1]["window"]
    return SatelliteAction(
        action_type="schedule",
        sat_id=w.satellite_id,
        station_id=w.station_id,
        window_id=w.window_id,
    )


# ─────────────────────────────────────────────────────────────
# Single episode runner
# ─────────────────────────────────────────────────────────────

def run_episode(llm: OpenAI, env: SatelliteEnv, task: str) -> dict:
    result = env.reset(task=task)
    obs: SatelliteObservation = result.observation
    history: list[str] = []
    step = 0
    llm_calls = 0
    t_start = time.time()

    print(f"\n{'─' * 60}")
    print(f"  Task: {task.upper()}  |  t=0  |  windows={len(obs.pass_windows)}")
    print(f"{'─' * 60}")

    while not obs.done and step < MAX_STEPS:
        step += 1

        # Build ranked choices
        choices = _build_choices(obs)

        if not choices and not obs.info.get("emergency_injection", False):
            # No windows — auto-noop, no LLM call
            action = SatelliteAction(action_type="noop")
        else:
            # Call LLM with numbered choices
            user_prompt = _obs_to_prompt(obs, step, history, choices)
            try:
                completion = llm.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=TEMPERATURE,
                    max_tokens=16,  # just need a number
                    stream=False,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw_text = completion.choices[0].message.content or ""
                llm_calls += 1
            except Exception as exc:
                print(f"  [warn] LLM call failed at step {step}: {exc}")
                raw_text = ""

            action = _parse_choice(raw_text, choices)

            if DEBUG:
                print(f"  step {step:3d}: LLM='{raw_text.strip()}' → {action.action_type}"
                      + (f" sat{action.sat_id}→stn{action.station_id}" if action.action_type == "schedule" else ""))

        # Step environment
        result = env.step(action)
        obs = result.observation

        # Log
        bytes_dl = obs.info.get("bytes_downloaded", 0)
        r_tick = obs.info.get("reward_last_tick", 0.0)
        conflict = obs.info.get("conflict", False)
        emg = obs.info.get("emergency_injection", False)

        history_line = (
            f"t={obs.current_time_min:4d}min  "
            f"{action.action_type:8s}  "
            f"reward={r_tick:+.4f}  "
            f"bytes={bytes_dl / 1e6:6.1f}MB"
            + ("  [CONFLICT]" if conflict else "")
            + ("  [EMERGENCY]" if emg else "")
        )
        history.append(history_line)

        if DEBUG:
            err_msg = obs.info.get("action_error", "")
            print(f"  step {step:3d}: {history_line}"
                  + (f"  err={err_msg}" if err_msg else ""))
        elif step % 20 == 0 or emg or conflict:
            print(f"  step {step:3d}: t={obs.current_time_min}min  "
                  f"reward_total={obs.reward:.4f}"
                  + ("  *** EMERGENCY ***" if emg else ""))

    # Final state
    final_state = env.state()
    duration = time.time() - t_start

    print(f"\n  Done: {step} steps, {llm_calls} LLM calls, {duration:.1f}s")

    return {
        "task": task,
        "steps": step,
        "llm_calls": llm_calls,
        "total_reward": round(obs.reward, 4),
        "final_score": round(final_state.get("final_score", 0.0), 4),
        "breakdown": final_state.get("breakdown", {}),
        "duration_s": round(duration, 1),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Satellite Downlink Scheduler — Baseline Inference")
    print("=" * 60)
    print(f"  Model:   {MODEL_NAME}")
    print(f"  API:     {API_BASE_URL}")
    print(f"  Env:     {ENV_URL}")
    print(f"  Seed:    {SEED}")
    print(f"  Debug:   {DEBUG}")

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks = ["task1", "task2", "task3"]
    results = []

    try:
        with SatelliteEnv(base_url=ENV_URL).sync() as env:
            for task in tasks:
                print(f"\nRunning {task}...")
                try:
                    r = run_episode(llm=llm, env=env, task=task)
                    results.append(r)
                except Exception as exc:
                    print(f"  [ERROR] {task} failed: {exc}")
                    results.append({
                        "task": task, "final_score": 0.0,
                        "breakdown": {}, "error": str(exc),
                    })
    except Exception as exc:
        print(f"  [ERROR] Connection failed: {exc}")
        for task in tasks:
            results.append({
                "task": task, "final_score": 0.0,
                "breakdown": {}, "error": str(exc),
            })

    # Results table
    print(f"\n{'=' * 60}")
    print("  BASELINE RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Task':<10}  {'Score':>7}  {'Steps':>6}  {'LLM':>5}  {'Reward':>8}  {'Time':>7}")
    print(f"  {'─'*10}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*7}")

    for r in results:
        score = r.get("final_score", 0.0)
        steps = r.get("steps", 0)
        calls = r.get("llm_calls", 0)
        rew = r.get("total_reward", 0.0)
        dur = r.get("duration_s", 0.0)
        err = "  ERROR" if "error" in r else ""
        print(f"  {r['task']:<10}  {score:>7.4f}  {steps:>6}  {calls:>5}  {rew:>8.4f}  {dur:>6.1f}s{err}")

    print(f"\n  Model:  {MODEL_NAME}")
    print(f"  Seed:   {SEED}")

    # Breakdown
    print(f"\n{'=' * 60}")
    print("  SCORE BREAKDOWNS")
    print(f"{'=' * 60}")
    for r in results:
        print(f"\n  {r['task'].upper()}:")
        bd = r.get("breakdown", {})
        if not bd:
            print(f"    error: {r.get('error', 'unknown')}")
            continue
        print(f"    final_score:          {bd.get('final_score', 0):.4f}")
        print(f"    bytes_downloaded:     {bd.get('bytes_downloaded', 0) / 1e6:.1f} MB")
        print(f"    bytes_available:      {bd.get('bytes_available', 0) / 1e6:.1f} MB")
        print(f"    throughput:           {bd.get('throughput_pct', 0):.1f}%")
        if "priority_efficiency" in bd:
            print(f"    priority_efficiency:  {bd['priority_efficiency']:.4f}")
        if "emergency_score" in bd:
            print(f"    emergency_score:      {bd['emergency_score']:.4f}")

    if any(r.get("final_score", 0.0) == 0.0 for r in results):
        print("\n  [warn] One or more tasks scored 0.0 — check server connection.")
        sys.exit(1)

    print("\n  Done.")


if __name__ == "__main__":
    main()
