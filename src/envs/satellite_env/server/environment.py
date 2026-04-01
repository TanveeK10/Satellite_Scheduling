# src/envs/satellite_env/server/environment.py
"""
SatelliteEnvironment — core episode logic.

Implements the three OpenEnv abstract methods:
    reset()  → SatelliteObservation
    step()   → SatelliteObservation
    state    → SatelliteState  (@property)

Wires together:
    WeatherSampler  (weather.py)   — per-station availability
    Scheduler       (scheduler.py) — conflict detection + downlink execution

Does NOT know about HTTP, WebSockets, or FastAPI.
Fully testable as a plain Python object.
"""

from __future__ import annotations

import json
import pathlib
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openenv.core.env_server import Environment

from src.envs.satellite_env.models import (
    DataChunkModel,
    PassWindowModel,
    SatelliteAction,
    SatelliteObservation
)
from src.envs.satellite_env.server.scheduler import Scheduler
from src.envs.satellite_env.server.weather import WeatherSampler


# ─────────────────────────────────────────────────────────────
# State dataclass (episode-level metadata)
# ─────────────────────────────────────────────────────────────

@dataclass
class SatelliteState:
    """
    Episode metadata returned by state().
    Judges and the inference script use this to inspect progress
    without re-parsing the full observation.
    """
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = 0
    task: str = "task1"
    current_time_min: int = 0
    done: bool = False
    total_reward: float = 0.0
    seed: int = 42
    # Grader score updated at episode end
    final_score: float = 0.0


# ─────────────────────────────────────────────────────────────
# Reward weights — defined once, used by both step() and graders
# ─────────────────────────────────────────────────────────────

PRIORITY_WEIGHT = {1: 1.0, 2: 2.0, 3: 3.0}
CONFLICT_PENALTY = -0.05
DELAY_PENALTY_MAX = -0.10
LOOKAHEAD_TICKS = 24  # 4-hour window  (24 × 10 min)
DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent.parent / "data"


# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────

class SatelliteEnvironment(Environment):
    """
    Full satellite downlink scheduling environment.

    Construction:
        env = SatelliteEnvironment(task="task2", seed=42)

    The task controls:
        task1 — 2 satellites, 2 stations, clear weather
        task2 — 8 satellites, 4 stations, weather dropout
        task3 — task2 + emergency chunk injections at t=240 and t=480

    Episode lifecycle:
        obs        = env.reset()
        while not obs.done:
            action = agent.decide(obs)
            obs    = env.step(action)
        score = env.state.final_score
    """

    def __init__(self, task: str = "task1", seed: int = 42) -> None:
        super().__init__()
        self._task = task
        self._seed = seed

        # Load scenario from pre-baked JSON
        scenario_path = DATA_DIR / "scenarios" / f"{task}_seed{seed}.json"
        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario file not found: {scenario_path}\n"
                f"Run scripts/generate_windows.py first."
            )
        self._scenario = json.loads(scenario_path.read_text())

        # Build static lookup structures from scenario
        self._all_windows: List[dict] = self._scenario["pass_windows"]
        self._active_sats: List[int] = self._scenario["active_satellites"]
        self._active_stations: List[int] = self._scenario["active_stations"]
        self._sat_meta: List[dict] = self._scenario["satellite_meta"]
        self._injections: List[dict] = self._scenario.get("emergency_injections", [])

        # Normalizer: total priority-weighted bytes available this episode
        # Pre-computed once — used in every reward calculation
        self._normalizer: float = self._compute_normalizer()

        # Mutable episode state — reset in reset()
        self._tick: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._episode_id: str = str(uuid.uuid4())
        self._injected_ids: set[str] = set()  # chunk_ids already injected

        # Sub-systems — initialized in reset()
        self._weather: Optional[WeatherSampler] = None
        self._scheduler: Optional[Scheduler] = None

        # Run reset() immediately so the object is usable right after __init__
        self._boot()

    # ------------------------------------------------------------------
    # OpenEnv interface — three required methods
    # ------------------------------------------------------------------

    def reset(self) -> SatelliteObservation:  # type: ignore[override]
        """
        Start a fresh episode.

        - Reseeds weather sampler
        - Restores all satellite queues to initial state
        - Clears the schedule
        - Returns tick-0 observation
        """
        self._tick = 0
        self._done = False
        self._total_reward = 0.0
        self._episode_id = str(uuid.uuid4())
        self._injected_ids = set()

        self._weather.reset()
        self._scheduler.reset()

        return self._build_observation(
            info={
                "conflict": False,
                "bytes_downloaded": 0,
                "reward_last_tick": 0.0,
                "emergency_injection": False,
                "action_error": None,
            }
        )

    def step(self, action: SatelliteAction) -> SatelliteObservation:  # type: ignore[override]
        """
        Process one agent action and advance the clock by one tick.

        Order of operations per tick:
            1. Validate action type
            2. Dispatch action to scheduler
            3. Fire any emergency injections due this tick
            4. Execute all scheduled windows for this tick
            5. Compute per-tick reward
            6. Advance clock
            7. Check terminal condition
            8. Build and return observation

        Returns the observation for the NEW tick (after advancement).
        The agent sees the consequences of its action immediately.
        """
        if self._done:
            # Episode already finished — return terminal observation
            return self._build_observation(info={
                "conflict": False, "bytes_downloaded": 0,
                "reward_last_tick": 0.0, "emergency_injection": False,
                "action_error": "Episode already done — call reset()",
            })

        # ── 1. Dispatch action ────────────────────────────────────────
        info = {
            "conflict": False,
            "bytes_downloaded": 0,
            "reward_last_tick": 0.0,
            "emergency_injection": False,
            "action_error": None,
        }
        step_reward = 0.0

        action_result = self._dispatch_action(action)
        if not action_result["accepted"]:
            info["conflict"] = action_result.get("conflict", False)
            info["action_error"] = action_result.get("error")
            step_reward += CONFLICT_PENALTY

        # ── 2. Emergency injections ───────────────────────────────────
        current_min = self._tick * 10
        injected_now = self._fire_injections(current_min)
        if injected_now:
            info["emergency_injection"] = True

        # ── 3. Execute scheduled windows for this tick ────────────────
        availability = self._weather.get(self._tick)
        results = self._scheduler.execute_tick(self._tick, availability)

        # ── 4. Compute reward ─────────────────────────────────────────
        tick_weighted_bytes = 0.0
        total_bytes_this_tick = 0

        for r in results:
            for chunk_log in r.chunks_downloaded:
                w = PRIORITY_WEIGHT.get(chunk_log["priority"], 1.0)
                tick_weighted_bytes += w * chunk_log["bytes_taken"]
                total_bytes_this_tick += chunk_log["bytes_taken"]

                # Delay penalty for emergency chunks downloaded past deadline
                if chunk_log["deadline_min"] is not None:
                    download_min = self._tick * 10
                    if download_min > chunk_log["deadline_min"]:
                        delay_min = download_min - chunk_log["deadline_min"]
                        penalty = DELAY_PENALTY_MAX * min(delay_min / 60.0, 1.0)
                        step_reward += penalty

        # Normalise to [0, 1] range
        if self._normalizer > 0:
            step_reward += tick_weighted_bytes / self._normalizer

        info["bytes_downloaded"] = total_bytes_this_tick
        info["reward_last_tick"] = round(step_reward, 6)

        # ── 5. Advance clock ──────────────────────────────────────────
        self._tick += 1
        self._total_reward += step_reward

        # ── 6. Check terminal condition ───────────────────────────────
        # Done when: 144 ticks elapsed OR all pass windows passed AND buffers empty
        all_windows_past = self._tick >= 144
        buffers_empty = self._scheduler.all_buffers_empty()
        self._done = all_windows_past or (buffers_empty and self._tick > 0)

        # ── 7. Compute final score on terminal step ───────────────────
        if self._done:
            from src.envs.satellite_env.server.graders import grade
            final_score = grade(
                task=self._task,
                download_log=self._scheduler.get_download_log(),
                all_chunks=self._all_initial_chunks(),
                emergency_injections=self._injections,
            )
            self._final_score = final_score
        else:
            self._final_score = 0.0

        return self._build_observation(info=info)

    @property
    def state(self) -> SatelliteState:
        """Episode metadata snapshot. Safe to call at any point."""
        return SatelliteState(
            episode_id=self._episode_id,
            step_count=self._tick,
            task=self._task,
            current_time_min=self._tick * 10,
            done=self._done,
            total_reward=round(self._total_reward, 6),
            seed=self._seed,
            final_score=getattr(self, "_final_score", 0.0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _boot(self) -> None:
        """
        Build sub-systems from scenario data.
        Called once from __init__. Separated so reset() can
        reinitialise without re-reading JSON or rebuilding lookup tables.
        """
        # Build initial queues from scenario
        initial_queues: Dict[int, List[DataChunkModel]] = {}
        for sat_id_str, chunks in self._scenario["initial_queues"].items():
            sat_id = int(sat_id_str)
            initial_queues[sat_id] = [DataChunkModel(**c) for c in chunks]

        downlink_rates = {
            m["id"]: m["downlink_rate_bps"]
            for m in self._sat_meta
        }

        self._weather = WeatherSampler(
            seed=self._seed,
            task=self._task,
        )
        self._scheduler = Scheduler(
            initial_queues=initial_queues,
            downlink_rates_bps=downlink_rates,
        )

        # Run reset to set tick / done / reward to initial values
        self._tick = 0
        self._done = False
        self._total_reward = 0.0
        self._episode_id = str(uuid.uuid4())
        self._injected_ids = set()
        self._final_score = 0.0

    def _dispatch_action(self, action: SatelliteAction) -> dict:
        """
        Route action to the appropriate scheduler method.
        Returns a dict with keys: accepted, conflict, error.
        """
        t = action.action_type

        if t == "schedule":
            if None in (action.sat_id, action.station_id, action.window_id):
                return {"accepted": False, "conflict": False,
                        "error": "schedule requires sat_id, station_id, window_id"}
            result = self._scheduler.schedule(
                sat_id=action.sat_id,
                station_id=action.station_id,
                window_id=action.window_id,
                tick=self._tick,
            )

        elif t == "preempt":
            if action.schedule_id is None:
                return {"accepted": False, "conflict": False,
                        "error": "preempt requires schedule_id"}
            result = self._scheduler.preempt(
                schedule_id=action.schedule_id,
                current_tick=self._tick,
            )

        elif t == "hold":
            if action.sat_id is None:
                return {"accepted": False, "conflict": False,
                        "error": "hold requires sat_id"}
            result = self._scheduler.hold(action.sat_id)

        elif t == "noop":
            return {"accepted": True}

        else:
            return {"accepted": False, "conflict": False,
                    "error": f"Unknown action_type: '{t}'"}

        return {
            "accepted": result.accepted,
            "conflict": result.conflict,
            "error": result.error,
            "schedule_id": result.schedule_id,
        }

    def _fire_injections(self, current_min: int) -> List[DataChunkModel]:
        """
        Check whether any emergency injections are due at current_min.
        Each injection fires exactly once (tracked by chunk_id in _injected_ids).
        Returns the list of newly injected chunks.
        """
        injected = []
        for inj in self._injections:
            if inj["inject_at_min"] != current_min:
                continue
            chunk_id = inj["chunk"]["chunk_id"]
            if chunk_id in self._injected_ids:
                continue
            chunk = DataChunkModel(**inj["chunk"])
            self._scheduler.inject_chunks(inj["satellite_id"], [chunk])
            self._injected_ids.add(chunk_id)
            injected.append(chunk)
        return injected

    def _build_observation(self, info: dict) -> SatelliteObservation:
        """
        Assemble a SatelliteObservation from current environment state.

        pass_windows filtered to [current_tick, current_tick + LOOKAHEAD_TICKS).
        All dict keys are strings for JSON / Pydantic compatibility.
        """
        current_tick = self._tick
        lookahead_end = current_tick + LOOKAHEAD_TICKS

        # Filter windows to lookahead window
        visible_windows = [
            PassWindowModel(
                window_id=f"w_s{w['satellite_id']}_g{w['station_id']}_{w['tick']:03d}",
                satellite_id=w["satellite_id"],
                station_id=w["station_id"],
                tick=w["tick"],
                duration_s=w["duration_s"],
                max_rate_mbps=w["max_rate_mbps"],
                elevation_deg=w["elevation_deg"],
                link_quality=w["link_quality"],
                max_bytes=w["max_bytes"],
            )
            for w in self._all_windows
            if current_tick <= w["tick"] < lookahead_end
               and w["satellite_id"] in self._active_sats
               and w["station_id"] in self._active_stations
        ]

        availability = self._weather.get_str_keys(current_tick) \
            if self._weather else {str(s): 1.0 for s in self._active_stations}

        return SatelliteObservation(
            current_time_min=current_tick * 10,
            done=self._done,
            reward=round(self._total_reward, 6),
            pass_windows=visible_windows,
            station_availability=availability,
            satellite_buffer_bytes=self._scheduler.get_buffer_bytes()
            if self._scheduler else {},
            data_priority_queues={
                sid: chunks
                for sid, chunks in (
                    self._scheduler.get_queues().items()
                    if self._scheduler else {}.items()
                )
            },
            downlink_rates_bps={
                str(m["id"]): m["downlink_rate_bps"]
                for m in self._sat_meta
            },
            current_schedule=self._scheduler.get_schedule()
            if self._scheduler else [],
            info=info,
        )

    def _compute_normalizer(self) -> float:
        """
        Sum of priority_weight × size_bytes across ALL chunks in the episode.
        This is the denominator in the reward formula — computed once at init.
        """
        total = 0.0
        for chunks in self._scenario["initial_queues"].values():
            for c in chunks:
                total += PRIORITY_WEIGHT.get(c["priority"], 1.0) * c["size_bytes"]
        # Add emergency injection chunks
        for inj in self._injections:
            c = inj["chunk"]
            total += PRIORITY_WEIGHT.get(c["priority"], 1.0) * c["size_bytes"]
        return total

    def _all_initial_chunks(self) -> List[dict]:
        """
        Flat list of all chunks (initial + injections) for graders.
        """
        chunks = []
        for chunk_list in self._scenario["initial_queues"].values():
            chunks.extend(chunk_list)
        for inj in self._injections:
            chunks.append(inj["chunk"])
        return chunks
