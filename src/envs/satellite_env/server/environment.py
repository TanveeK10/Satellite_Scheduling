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
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from src.envs.satellite_env.models import (
    DataChunkModel,
    PassWindowModel,
    RewardModel,
    SatelliteAction,
    SatelliteObservation
)
from src.envs.satellite_env.server.scheduler import Scheduler
from src.envs.satellite_env.server.weather import WeatherSampler


# ─────────────────────────────────────────────────────────────
# State dataclass (episode-level metadata)
# ─────────────────────────────────────────────────────────────

class SatelliteState(BaseModel):
    """
    Episode metadata returned by state().
    Judges and the inference script use this to inspect progress
    without re-parsing the full observation.
    """
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = 0
    task: str = "task1"
    current_time_min: int = 0
    done: bool = False
    total_reward: float = 0.0
    seed: int = 42
    # Grader score updated at episode end
    final_score: float = 0.0
    breakdown: Dict[str, Any] = Field(default_factory=dict)

SatelliteState.model_rebuild()


# ─────────────────────────────────────────────────────────────
# Reward weights — defined once, used by both step() and graders
# ─────────────────────────────────────────────────────────────

PRIORITY_WEIGHT = {1: 1.0, 2: 2.0, 3: 3.0}
CONFLICT_PENALTY = -0.05       # Double-booking station/sat
INVALID_ACTION_PENALTY = -0.01  # Empty buffer/missing ID (protocol error)
DELAY_PENALTY_MAX = -0.10
LOOKAHEAD_TICKS = 24  # 4-hour window  (24 × 10 min)
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent / "data"


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
        self._scenario = {}
        self._injected_ids: set[str] = set()

        # Initial scenario load
        self._load_scenario(task, seed)
        
        # Static episode metadata — reset in reset()
        self._tick: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._episode_id: str = str(uuid.uuid4())
        self._actions_this_tick: int = 0

        # Sub-systems — initialized in reset()
        self._weather: Optional[WeatherSampler] = None
        self._scheduler: Optional[Scheduler] = None

        # Run reset() immediately so the object is usable right after __init__
        self._boot()

    def _load_scenario(self, task: str, seed: int) -> None:
        """Load scenario data from JSON."""
        scenario_path = DATA_DIR / "scenarios" / f"{task.strip()}_seed{seed}.json"
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        
        self._scenario = json.loads(scenario_path.read_text())
        self._all_windows = self._scenario["pass_windows"]
        self._active_sats = self._scenario["active_satellites"]
        self._active_stations = self._scenario["active_stations"]
        self._sat_meta = self._scenario["satellite_meta"]
        self._injections = self._scenario.get("emergency_injections", [])
        self._normalizer = self._compute_normalizer()

    # ------------------------------------------------------------------
    # OpenEnv interface — three required methods
    # ------------------------------------------------------------------

    def reset(self, task: str = None, seed: int = None) -> SatelliteObservation:  # type: ignore[override]
        """
        Reset to the beginning of the episode.
        Allows switching tasks dynamically during reset.
        """
        if task is not None: self._task = task
        if seed is not None: self._seed = seed
        
        if task is not None or seed is not None:
            self._load_scenario(self._task, self._seed)

        self._boot()
        return self._build_observation(
            info={
                "conflict": False,
                "bytes_downloaded": 0,
                "reward_last_tick": 0.0,
                "reward_breakdown": {"priority_1": 0.0, "priority_2": 0.0, "priority_3": 0.0, "penalties": 0.0},
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
            
            # Apply appropriate penalty
            if info["conflict"]:
                step_reward += CONFLICT_PENALTY
            else:
                step_reward += INVALID_ACTION_PENALTY

        # ── 2. Emergency injections ───────────────────────────────────
        current_min = self._tick * 10
        injected_now = self._fire_injections(current_min)
        if injected_now:
            info["emergency_injection"] = True

        # ── 3. Execute scheduled windows for this tick ────────────────
        availability = self._weather.get(self._tick)
        window_elevations = {
            f"w_s{w['sat_id']}_g{w['station_id']}_{w['tick']:03d}": w["elevation_deg"]
            for w in self._all_windows if w["tick"] == self._tick
        }
        results = self._scheduler.execute_tick(self._tick, availability, window_elevations)

        # ── 4. Compute reward ─────────────────────────────────────────
        tick_weighted_bytes = 0.0
        total_bytes_this_tick = 0
        breakdown = {"priority_1": 0.0, "priority_2": 0.0, "priority_3": 0.0, "penalties": 0.0}

        def _get(obj, key, default=None):
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)

        for r in results:
            chunks_downloaded = _get(r, "chunks_downloaded", [])
            for chunk_log in chunks_downloaded:
                p = _get(chunk_log, "priority", 1)
                bt = _get(chunk_log, "bytes_taken", 0)
                deadline = _get(chunk_log, "deadline_min")
                
                w = PRIORITY_WEIGHT.get(p, 1.0)
                weighted_val = (w * bt) / self._normalizer if self._normalizer > 0 else 0.0
                
                tick_weighted_bytes += w * bt
                total_bytes_this_tick += bt
                
                # Add to breakdown
                key = f"priority_{p}"
                breakdown[key] = breakdown.get(key, 0.0) + weighted_val

                # Delay penalty for emergency chunks downloaded past deadline
                if deadline is not None:
                    download_min = self._tick * 10
                    if download_min > deadline:
                        delay_min = download_min - deadline
                        penalty = DELAY_PENALTY_MAX * min(delay_min / 60.0, 1.0)
                        step_reward += penalty
                        breakdown["penalties"] += penalty


        # Normalise to [0, 1] range
        if self._normalizer > 0:
            step_reward += tick_weighted_bytes / self._normalizer

        info["bytes_downloaded"] = total_bytes_this_tick
        info["reward_last_tick"] = round(step_reward, 6)
        info["reward_breakdown"] = breakdown

        # ── 5. Advance clock ──────────────────────────────────────────
        # Only advance the tick if the agent explicitly 'noop's (waits)
        # or if they've made too many actions in a single tick (safety).
        tick_advanced = False
        if action.action_type == "noop":
            self._tick += 1
            tick_advanced = True
        
        # Safety: auto-advance if agent is spamming actions without noop
        self._actions_this_tick += 1
        if self._actions_this_tick > 50:
            self._tick += 1
            self._actions_this_tick = 0
            tick_advanced = True
        
        if tick_advanced:
            self._actions_this_tick = 0
            # Reset conflict flag for next tick's first action
            # (only if we want to clear the 'last action rejected' state)
            pass

        self._total_reward += step_reward

        # ── 6. Check terminal condition ───────────────────────────────
        # Done when: 144 ticks elapsed OR 
        # (all pass windows passed AND buffers empty AND no pending injections)
        all_windows_past = self._tick >= 144
        buffers_empty = self._scheduler.all_buffers_empty()
        
        # Check if there are any emergency injections still scheduled for the future
        pending_injections = len(self._injected_ids) < len(self._injections)
        
        self._done = all_windows_past or (buffers_empty and self._tick > 0 and not pending_injections)

        # ── 7. Compute final score on terminal step ───────────────────
        if self._done:
            from src.envs.satellite_env.server.graders import grade, grade_breakdown
            final_data = {
                "download_log": [d.__dict__ for d in self._scheduler.get_download_log()],
                "all_chunks": self._all_initial_chunks(),
                "emergency_injections": self._injections,
            }
            self._final_score = grade(task=self._task, **final_data)
            self._final_breakdown = grade_breakdown(task=self._task, **final_data)
        else:
            self._final_score = 0.0
            self._final_breakdown = {}

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
            breakdown=getattr(self, "_final_breakdown", {}),
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
        self._actions_this_tick = 0

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
            self._scheduler.inject_chunks(inj["sat_id"], [chunk])
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
                window_id=f"w_s{w['sat_id']}_g{w['station_id']}_{w['tick']:03d}",
                sat_id=w["sat_id"],
                station_id=w["station_id"],
                start_min=w["tick"] * 10,
                end_min=(w["tick"] + 1) * 10,
                tick=w["tick"],
                duration_s=w["duration_s"],
                max_rate_mbps=w["max_rate_mbps"],
                elevation_deg=w["elevation_deg"],
                link_quality=w["link_quality"],
                max_bytes=w["max_bytes"],
            )
            for w in self._all_windows
            if current_tick <= w["tick"] < lookahead_end
               and w["sat_id"] in self._active_sats
               and w["station_id"] in self._active_stations
        ]

        availability = self._weather.get_str_keys(current_tick) \
            if self._weather else {str(s): 1.0 for s in self._active_stations}

        # Build Reward object from info or running total
        reward_obj = RewardModel(
            value=round(self._total_reward, 6),
            breakdown=info.get("reward_breakdown", {}),
            metadata={
                "episode_data": {
                    "download_log": [d.__dict__ for d in self._scheduler.get_download_log()] if self._done else [],
                    "all_chunks": self._all_initial_chunks() if self._done else [],
                    "emergency_injections": self._injections if self._done else [],
                } if self._done else {}
            }
        )

        return SatelliteObservation(
            current_time_min=current_tick * 10,
            done=self._done,
            reward=reward_obj.value,
            reward_obj=reward_obj,
            pass_windows=visible_windows,
            station_availability=availability,
            satellite_buffer_bytes={
                str(sid): val
                for sid, val in (
                    self._scheduler.get_buffer_bytes().items()
                    if self._scheduler else {}.items()
                )
            },
            data_priority_queues={
                str(sid): chunks
                for sid, chunks in (
                    self._scheduler.get_queues().items()
                    if self._scheduler else {}.items()
                )
            },
            downlink_rates_bps={
                str(sid): val
                for sid, val in (
                    self._scheduler.get_rates_bps().items()
                    if self._scheduler else {}.items()
                )
            },
            current_schedule=self._scheduler.get_schedule()
            if self._scheduler else [],
            info_dict=info,
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
