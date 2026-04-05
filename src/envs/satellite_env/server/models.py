# src/envs/satellite_env/models.py
"""
Pydantic Models for the Satellite Downlink Scheduling environment.
Standalone versions of Observation/Action to bypass strict library validation.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import Field, BaseModel, ConfigDict, field_validator

# Standalone base models with extra="allow" 
# to ensure server-side validation never fails for extra fields.
class Observation(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    done: bool = Field(False)
    reward: float = Field(0.0)
    info_dict: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("reward", mode="before")
    @classmethod
    def coerce_reward(cls, v: Any) -> float:
        if isinstance(v, dict):
            return float(v.get("value", 0.0))
        if hasattr(v, "value"):
            return float(v.value)
        return float(v)

class Action(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)


# ─────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────

class PassWindowModel(BaseModel):
    window_id: str = Field(...)
    sat_id: int = Field(...)
    station_id: int = Field(...)
    start_min: int = Field(...)
    end_min: int = Field(...)
    tick: int = Field(...)
    duration_s: float = Field(...)
    max_rate_mbps: float = Field(...)
    elevation_deg: float = Field(...)
    link_quality: float = Field(...)
    max_bytes: int = Field(...)

class DataChunkModel(BaseModel):
    chunk_id: str = Field(...)
    priority: int = Field(...)
    size_bytes: int = Field(...)
    injected_at_min: int = Field(0)
    deadline_min: Optional[int] = Field(None)

class ScheduleEntryModel(BaseModel):
    schedule_id: str = Field(...)
    sat_id: int = Field(...)
    station_id: int = Field(...)
    window_id: str = Field(...)
    tick: int = Field(...)
    status: str = Field("committed")

class RewardModel(BaseModel):
    value: float = Field(...)
    breakdown: Dict[str, float] = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

class SatelliteAction(Action):
    action_type: str = Field(...)
    sat_id: Optional[int] = None
    station_id: Optional[int] = None
    window_id: Optional[str] = None
    schedule_id: Optional[str] = None


# ─────────────────────────────────────────────
# Observation (The 'Thick' Model)
# ─────────────────────────────────────────────

class SatelliteObservation(Observation):
    """
    Complete observation model. All fields are at the top level.
    By inheriting from our lenient Observation base, we pass validation.
    """
    current_time_min: int = Field(0)
    reward_obj: RewardModel = Field(default_factory=lambda: RewardModel(value=0.0, breakdown={}))
    pass_windows: List[PassWindowModel] = Field(default_factory=list)
    station_availability: Dict[str, float] = Field(default_factory=dict)
    satellite_buffer_bytes: Dict[str, int] = Field(default_factory=dict)
    data_priority_queues: Dict[str, List[DataChunkModel]] = Field(default_factory=dict)
    downlink_rates_bps: Dict[str, int] = Field(default_factory=dict)
    current_schedule: List[ScheduleEntryModel] = Field(default_factory=list)

    @property
    def info(self) -> Dict[str, Any]:
        """Alias for info_dict for backward compatibility."""
        return self.info_dict
