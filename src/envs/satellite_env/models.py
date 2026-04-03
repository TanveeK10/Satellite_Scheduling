# src/envs/satellite_env/models.py
"""
Typed contracts for the Satellite Downlink Scheduling environment.

All three classes (Action, Observation, State) follow the OpenEnv spec:
  - Action    → subclasses openenv.core.env_server.types.Action
  - Observation → subclasses openenv.core.env_server.types.Observation
  - State     → uses openenv.core.env_server.types.State directly
                (no custom fields needed — step_count + episode_id suffice)

Import path used by both server and client:
    from satellite_env.models import (
        SatelliteAction, SatelliteObservation,
        PassWindowModel, DataChunkModel
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


# ─────────────────────────────────────────────
# Sub-models (not Action/Observation themselves —
# these are nested inside the observation)
# ─────────────────────────────────────────────

class PassWindowModel(Observation):
    """
    A single contact opportunity between one satellite and one station
    within the agent's lookahead window.

    max_bytes is pre-computed as:
        int(max_rate_mbps * 1e6 / 8 * duration_s * link_quality)
    The agent should use this directly rather than recomputing.
    """
    window_id: str = Field(..., description="Unique ID e.g. 'w_s2_g1_042'")
    satellite_id: int = Field(..., ge=0, le=7)
    station_id: int = Field(..., ge=0, le=3)
    tick: int = Field(..., ge=0, lt=144, description="Tick this window is active")
    duration_s: float = Field(..., gt=0, description="Seconds the link is open")
    max_rate_mbps: float = Field(..., ge=0, description="Peak link rate at this elevation")
    elevation_deg: float = Field(..., ge=5.0, le=90.0)
    link_quality: float = Field(..., ge=0.0, le=1.0)
    max_bytes: int = Field(..., ge=0, description="Max downloadable bytes, clear sky")


class DataChunkModel(Observation):
    """
    One unit of observation data sitting in a satellite's onboard buffer.

    priority:
        1 = routine      weight w(1) = 1.0
        2 = important    weight w(2) = 2.0
        3 = emergency    weight w(3) = 3.0  — may have a hard deadline

    deadline_min is only set for priority-3 emergency chunks (Task 3).
    None means no deadline — download anytime before episode end.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    priority: int = Field(..., ge=1, le=3)
    size_bytes: int = Field(..., ge=0)
    injected_at_min: int = Field(0, ge=0, description="0 = pre-loaded at episode start")
    deadline_min: Optional[int] = Field(None, description="Hard deadline — priority-3 only")


class ScheduleEntryModel(Observation):
    """
    One committed assignment in the current schedule.
    Created by schedule(), cancelled by preempt().
    """
    schedule_id: str = Field(..., description="Unique assignment ID e.g. 'sch_s2_g0_042'")
    satellite_id: int = Field(..., ge=0, le=7)
    station_id: int = Field(..., ge=0, le=3)
    window_id: str = Field(...)
    tick: int = Field(...)
    status: str = Field("committed", description="'committed' | 'executing' | 'done'")


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

class SatelliteAction(Action):
    """
    One decision the agent makes per tick.

    action_type must be exactly one of:
        "schedule"  — assign a satellite to a station for a window
        "preempt"   — cancel a future committed assignment
        "hold"      — mark a satellite as held (block auto-suggestions)
        "noop"      — advance the clock with no change

    Field requirements by action_type:
        schedule  → sat_id, station_id, window_id  (required)
        preempt   → schedule_id                    (required)
        hold      → sat_id                         (required)
        noop      → no extra fields needed

    Invalid combinations (e.g. schedule without window_id) are caught
    inside environment.step() and returned as a conflict penalty.
    """
    action_type: str = Field(
        ...,
        description="One of: schedule | preempt | hold | noop"
    )
    sat_id: Optional[int] = Field(None, ge=0, le=7)
    station_id: Optional[int] = Field(None, ge=0, le=3)
    window_id: Optional[str] = Field(None)
    schedule_id: Optional[str] = Field(None, description="Used by preempt only")


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class SatelliteObservation(Observation):
    """
    Everything the agent sees at the start of each tick.

    Pass windows are filtered to a 4-hour lookahead from current_time_min.
    The agent never sees future weather values — only current availability.

    info dict keys:
        "conflict"             bool   — last action was rejected for conflict
        "bytes_downloaded"     int    — bytes downloaded in the previous tick
        "reward_last_tick"     float  — reward earned in the previous tick
        "emergency_injection"  bool   — new priority-3 chunk(s) just appeared
        "action_error"         str|None — human-readable reason if last action failed
    """
    current_time_min: int = Field(..., ge=0, le=1440)
    done: bool = Field(False)
    reward: float = Field(0.0)

    # Core scheduling state
    pass_windows: List[PassWindowModel] = Field(
        default_factory=list,
        description="Upcoming windows within 4-hour lookahead"
    )
    station_availability: Dict[str, float] = Field(
        default_factory=dict,
        description="station_id (str) → float [0.0, 1.0]. Str keys for JSON compat."
    )
    satellite_buffer_bytes: Dict[str, int] = Field(
        default_factory=dict,
        description="satellite_id (str) → remaining bytes in buffer"
    )
    data_priority_queues: Dict[str, List[DataChunkModel]] = Field(
        default_factory=dict,
        description="satellite_id (str) → ordered chunk list, highest priority first"
    )
    downlink_rates_bps: Dict[str, int] = Field(
        default_factory=dict,
        description="satellite_id (str) → max bits per second"
    )
    current_schedule: List[ScheduleEntryModel] = Field(
        default_factory=list,
        description="All future committed window assignments"
    )

    # Step metadata
    info: Dict[str, object] = Field(
        default_factory=lambda: {
            "conflict": False,
            "bytes_downloaded": 0,
            "reward_last_tick": 0.0,
            "emergency_injection": False,
            "action_error": None,
        }
    )
