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

Run locally against Ollama:
    API_BASE_URL=http://localhost:11434/v1 \\
    MODEL_NAME=llama3.2 \\
    HF_TOKEN=ollama \\
    python inference.py

Run against HF inference router:
    API_BASE_URL=https://router.huggingface.co/v1 \\
    MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \\
    HF_TOKEN=hf_xxxx \\
    ENV_URL=https://your-username-satellite-env.hf.space \\
    python inference.py

Expected baseline scores (seed=42, llama3.2 via Ollama):
    task1 — ~0.72
    task2 — ~0.44
    task3 — ~0.29
"""

import json
import os
import sys
import textwrap
import time
from openenv.core.client_types import StepResult
from openai import OpenAI

from src.envs.satellite_env.server.client import SatelliteEnv
from src.envs.satellite_env.server.models import SatelliteAction, SatelliteObservation
from src.envs.satellite_env.server.graders import grade_breakdown
from src.envs.satellite_env.server.environment import SatelliteState
from tqdm import tqdm

# ── Imports (after path setup) ────────────────────────────────
sys.path.insert(0, "src")


# ── Mandatory env vars ────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# ── Optional env vars ─────────────────────────────────────────
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:8000")
MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "400"))
TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0.2"))
MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "512"))
DEBUG: bool = os.environ.get("DEBUG", "0") == "1"
SEED: int = 42

# ── Validation ────────────────────────────────────────────────
_MISSING = [v for v, val in [
    ("API_BASE_URL", API_BASE_URL),
    ("MODEL_NAME", MODEL_NAME),
    ("HF_TOKEN", HF_TOKEN),
] if not val]

if _MISSING:
    print(f"[ERROR] Missing required env vars: {', '.join(_MISSING)}")
    print("  Set them before running inference.py — see file header for examples.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous satellite mission planner.
    You control a downlink scheduling environment.

    At each step you receive the current state of the satellite constellation:
    - pass_windows: upcoming contact opportunities (satellite ↔ station pairs)
    - station_availability: weather quality per station [0.0–1.0]
    - satellite_buffer_bytes: data remaining per satellite
    - data_priority_queues: data chunks awaiting download (priority 1/2/3)
    - current_schedule: already committed assignments
    - current_time_min: minutes elapsed in the 24-hour episode
    - info: result of your last action

    Your goal: maximise priority-weighted bytes downloaded.
    Priority weights: routine=1, important=2, emergency=3.
    Emergency chunks (priority 3) have deadlines — download them FIRST.

    Available actions (respond with EXACTLY ONE JSON object):
        {"action_type": "schedule",  "sat_id": int, "station_id": int, "window_id": str}
        {"action_type": "preempt",   "schedule_id": str}
        {"action_type": "hold",      "sat_id": int}
        {"action_type": "noop"}

    Strategy:
    1. If info.emergency_injection is true → immediately schedule the
       satellite carrying priority-3 data to the best available station.
    2. Prefer windows with high link_quality and high station_availability.
    3. Prefer satellites with high-priority data in their queues.
    4. Use preempt() to free a station slot when emergency data arrives.
    5. One satellite and one station can only handle ONE assignment per 10-minute tick.
    6. Use noop to advance time once you have filled the current tick's slots.

    Respond with ONLY a valid JSON object. No explanation, no markdown fences.
    Example: {"action_type": "schedule", "sat_id": 2, "station_id": 0, "window_id": "w_s2_g0_042"}
""").strip()


# ─────────────────────────────────────────────────────────────
# Observation → prompt
# ─────────────────────────────────────────────────────────────

def _obs_to_prompt(obs: SatelliteObservation, step: int, history: list[str]) -> str:
    """
    Serialise the observation into a compact, LLM-readable prompt.

    We deliberately truncate large fields (full queue listing, all windows)
    to stay within model context limits. The agent gets the most
    actionable information — not the raw full observation dump.
    """
    # Top windows by link_quality × station_availability
    avail = obs.station_availability
    ranked = sorted(
        obs.pass_windows,
        key=lambda w: w.link_quality * float(avail.get(str(w.station_id), 1.0)),
        reverse=True,
    )[:12]  # top 12 windows — enough context without blowing the context window

    windows_text = "\n".join(
        f"  window_id={w.window_id}  sat={w.sat_id}  "
        f"stn={w.station_id}  tick={w.tick}  "
        f"quality={w.link_quality:.2f}  "
        f"avail={float(avail.get(str(w.station_id), 1.0)):.2f}  "
        f"max_bytes={w.max_bytes:,}"
        for w in ranked
    ) or "  (none in lookahead)"

    # Buffer summary — only satellites with data remaining
    buffers = {
        k: v for k, v in obs.satellite_buffer_bytes.items() if v > 0
    }
    buffer_text = "  " + "  ".join(
        f"sat{k}={v / 1e6:.1f}MB" for k, v in sorted(buffers.items())
    ) if buffers else "  (all empty)"

    # Priority queue heads — show highest-priority chunk per satellite
    queue_heads = []
    for sid, chunks in obs.data_priority_queues.items():
        if not chunks:
            continue
        top = max(chunks, key=lambda c: c.priority)
        deadline = f"  deadline={top.deadline_min}min" if top.deadline_min else ""
        queue_heads.append(
            f"  sat{sid}: p{top.priority} {top.size_bytes / 1e6:.1f}MB{deadline}"
        )
    queue_text = "\n".join(queue_heads) or "  (all empty)"

    # Current schedule
    sched_text = "\n".join(
        f"  {e.schedule_id}: sat{e.sat_id}→stn{e.station_id} tick={e.tick}"
        for e in obs.current_schedule[:6]
    ) or "  (empty)"

    # Last 4 history lines for context
    history_text = "\n".join(f"  {h}" for h in history[-4:]) or "  (none)"

    # Info flags
    flags = []
    if obs.info.get("emergency_injection"):
        flags.append("*** EMERGENCY INJECTION — reschedule immediately ***")
    if obs.info.get("conflict"):
        flags.append(f"last action rejected: {obs.info.get('action_error')}")
    flags_text = "\n  ".join(flags) if flags else "none"

    return textwrap.dedent(f"""
        Step {step} | t={obs.current_time_min}min | reward_so_far={obs.reward:.4f}

        FLAGS: {flags_text}

        TOP UPCOMING WINDOWS (by quality × availability):
        {windows_text}

        STATION AVAILABILITY: {dict(avail)}

        SATELLITE BUFFERS (non-empty):
        {buffer_text}

        HIGHEST-PRIORITY QUEUE HEADS:
        {queue_text}

        CURRENT SCHEDULE:
        {sched_text}

        RECENT ACTIONS:
        {history_text}

        Respond with ONE JSON action object.
    """).strip()


# ─────────────────────────────────────────────────────────────
# LLM action parsing
# ─────────────────────────────────────────────────────────────

FALLBACK_ACTION = SatelliteAction(action_type="noop")


def _parse_action(response_text: str) -> SatelliteAction:
    """
    Extract a SatelliteAction from the LLM response.

    Tries in order:
        1. Parse entire response as JSON
        2. Find first {...} block in response
        3. Fall back to noop

    Invalid action_type values fall back to noop.
    Missing optional fields are left as None — Pydantic handles defaults.
    """
    text = response_text.strip()

    # Strip markdown fences if model wrapped in ```json ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        ).strip()

    # Attempt 1 — full response is JSON
    try:
        data = json.loads(text)
        return SatelliteAction(**data)
    except Exception:
        pass

    # Attempt 2 — find first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            candidate = text[start:end]
            data = json.loads(candidate)
            return SatelliteAction(**data)
        except Exception:
            pass

    # Attempt 3 — fallback
    if DEBUG:
        print(f"  [warn] Could not parse action from: {response_text[:120]!r}")
    return FALLBACK_ACTION


# ─────────────────────────────────────────────────────────────
# Single episode runner
# ─────────────────────────────────────────────────────────────

def _format_action_tag(action: SatelliteAction) -> str:
    """Format action as type(params) for the judge tag."""
    params = []
    if action.sat_id is not None: params.append(f"sat_id={action.sat_id}")
    if action.station_id is not None: params.append(f"station_id={action.station_id}")
    if action.window_id is not None: params.append(f"window_id={action.window_id}")
    if action.schedule_id is not None: params.append(f"schedule_id={action.schedule_id}")
    return f"{action.action_type}({', '.join(params)})"


def run_episode(
        llm: OpenAI,
        env: SatelliteEnv,
        task: str,
) -> dict:
    """
    Run one full episode of the given task.

    Returns a result dict with keys:
        task, steps, total_reward, final_score, breakdown, duration_s
    """
    result: "StepResult[SatelliteObservation]" = env.reset(task=task)
    obs: "SatelliteObservation" = result.observation
    history: list[str] = []
    step = 0
    t_start = time.time()

    print(f"[START] task={task} env=satellite_env model={MODEL_NAME}")
    
    cumulative_downloaded = 0
    total_expected = int(sum(v for v in obs.satellite_buffer_bytes.values()))
    
    rewards_list = []
    
    # tqdm is disabled by default to ensure a clean tag stream
    pbar = tqdm(
        total=total_expected,
        desc=f"  {task.upper()}",
        unit="B",
        unit_scale=True,
        disable=not DEBUG,
        bar_format=" {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    

    while not obs.done and step < MAX_STEPS:
        step += 1

        # ── Build prompt ──────────────────────────────────────
        user_prompt = _obs_to_prompt(obs, step, history)

        # ── Call LLM ─────────────────────────────────────────
        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [warn] LLM call failed at step {step}: {exc}")
            raw_text = ""
            
        if raw_text and DEBUG:
            print(f"  [debug] LLM raw response (step {step}):\n{raw_text[:500]}...")

        action = _parse_action(raw_text)

        # ── Step environment ──────────────────────────────────
        result = env.step(action)
        obs = result.observation

        # ── Log / Progress ──────────────────────────────────────────
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
        rewards_list.append(round(r_tick, 4))
        
        # [STEP] tag for the judge
        print(f"[STEP] step={step} action={_format_action_tag(action)} reward={r_tick:.4f} done={str(obs.done).lower()} error=null")

        cumulative_downloaded += bytes_dl
        
        # Check for new data (emergency injections) to update total
        current_total_in_env = sum(obs.satellite_buffer_bytes.values()) + cumulative_downloaded
        if current_total_in_env > total_expected:
            diff = current_total_in_env - total_expected
            pbar.total += diff
            total_expected = current_total_in_env

        # Calculate "Lost" data (chunks past deadline)
        lost_bytes = 0
        current_min = obs.current_time_min
        for sid, chunks in obs.data_priority_queues.items():
            for chunk in chunks:
                if chunk.deadline_min is not None and current_min > chunk.deadline_min:
                    lost_bytes += chunk.size_bytes

        pbar.update(bytes_dl)
        pbar.set_postfix({
            "Rem": f"{sum(obs.satellite_buffer_bytes.values())/1e6:.1f}MB",
            "Lost": f"{lost_bytes/1e6:.1f}MB"
        })

        if DEBUG:
            pbar.write(f"  step {step:3d}: {history_line}")
        
    pbar.close()
    # ── [END] tag logic ───────────────────────────────────────
    final_state = env.state()
    duration = time.time() - t_start
    score = final_state.final_score
    success = "true" if score > 0 else "false"
    rewards_str = ",".join(map(str, rewards_list))

    print(f"[END] success={success} steps={step} score={score:.3f} rewards={rewards_str}")

    return {
        "task": task,
        "steps": step,
        "total_reward": round(obs.reward, 4),
        "final_score": round(final_state.final_score, 4),
        "breakdown": final_state.breakdown,
        "duration_s": round(duration, 1),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    if DEBUG:
        print("=" * 60)
        print("  Satellite Downlink Scheduler — Baseline Inference")
        print("=" * 60)
        print(f"  Model:   {MODEL_NAME}")
        print(f"  API:     {API_BASE_URL}")
        print(f"  Env:     {ENV_URL}")
        print(f"  Seed:    {SEED}")
        print(f"  Debug:   {DEBUG}")

    # ── Build LLM client ─────────────────────────────────────
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # ── Run all 3 tasks ───────────────────────────────────────
    tasks = ["task1", "task2", "task3"]
    results = []

    for task in tasks:
        if DEBUG:
            print(f"\nConnecting to {ENV_URL} for {task}...")

        # Each task gets its own connection with SATELLITE_TASK set
        # We connect to the server which was started with the right task,
        # or pass task via reset() if the server supports it.
        # For baseline runs, start the server separately per task:
        #   SATELLITE_TASK=task1 uvicorn satellite_env.server.app:app ...
        # Or use Docker with -e SATELLITE_TASK=task1

        try:
            with SatelliteEnv(base_url=ENV_URL).sync() as env:
                r = run_episode(llm=llm, env=env, task=task)
                results.append(r)
        except Exception as exc:
            print(f"  [ERROR] {task} failed: {exc}")
            results.append({
                "task": task,
                "final_score": 0.0,
                "breakdown": {},
                "error": str(exc),
            })

    if DEBUG:
        # ── Print results table ───────────────────────────────────
        print(f"\n{'=' * 60}")
        print("  BASELINE RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'Task':<10}  {'Score':>7}  {'Steps':>6}  {'Reward':>8}  {'Time':>7}")
        print(f"  {'─' * 10}  {'─' * 7}  {'─' * 6}  {'─' * 8}  {'─' * 7}")

        for r in results:
            score = r.get("final_score", 0.0)
            steps = r.get("steps", 0)
            rew = r.get("total_reward", 0.0)
            dur = r.get("duration_s", 0.0)
            err = "  ERROR" if "error" in r else ""
            print(f"  {r['task']:<10}  {score:>7.4f}  {steps:>6}  {rew:>8.4f}  {dur:>6.1f}s{err}")

        print(f"\n  Model:  {MODEL_NAME}")
        print(f"  Seed:   {SEED}")

        # ── Detailed breakdown ────────────────────────────────────
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
            print(f"    delay_penalties:      {bd['delay_penalties']:.4f}")
            print(f"    base_score:           {bd['base_score']:.4f}")

    # ── Exit code for CI ──────────────────────────────────────
    # Non-zero exit if any task scored 0.0 (likely a connection error)
    if any(r.get("final_score", 0.0) == 0.0 for r in results):
        print("\n  [warn] One or more tasks scored 0.0 — check server connection.")
        sys.exit(1)

    print("\n  Done.")


if __name__ == "__main__":
    main()
