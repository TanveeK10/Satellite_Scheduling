# src/envs/satellite_env/client.py
"""
SatelliteEnv — typed WebSocket client for the satellite downlink environment.

Usage (sync — for inference.py and testing):
    from satellite_env.client import SatelliteEnv
    from satellite_env.models import SatelliteAction

    with SatelliteEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        while not result.done:
            action = SatelliteAction(action_type="noop")
            result = env.step(action)
        print(result.observation.metadata)

Usage (async — for training loops):
    async with SatelliteEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        result = await env.step(SatelliteAction(action_type="noop"))
"""

from __future__ import annotations

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from src.envs.satellite_env.server.models import (
    DataChunkModel,
    PassWindowModel,
    RewardModel,
    SatelliteAction,
    SatelliteObservation,
    ScheduleEntryModel,
)
from src.envs.satellite_env.server.environment import SatelliteState


class SatelliteEnv(EnvClient[SatelliteAction, SatelliteObservation, SatelliteState]):
    """
    Typed client for the SatelliteEnvironment server.

    Inherits from EnvClient which handles WebSocket connectivity.
    Now that we've bypassed the library's strict validation by defining 
    lenient base models, our wire format is 'thick' (top-level fields).
    """

    # ------------------------------------------------------------------
    # Required: serialize action → dict for WebSocket transport
    # ------------------------------------------------------------------

    def _step_payload(self, action: SatelliteAction) -> dict:
        """
        Convert a SatelliteAction to a JSON-serializable dict.
        """
        payload: dict = {"action_type": action.action_type}
        if action.sat_id is not None: payload["sat_id"] = action.sat_id
        if action.station_id is not None: payload["station_id"] = action.station_id
        if action.window_id is not None: payload["window_id"] = action.window_id
        if action.schedule_id is not None: payload["schedule_id"] = action.schedule_id
        return payload

    # ------------------------------------------------------------------
    # Required: deserialize server response → StepResult
    # ------------------------------------------------------------------

    def _parse_result(self, payload: dict) -> StepResult[SatelliteObservation]:
        """
        Reconstruct the rich SatelliteObservation from the server payload.
        Now that models.py is lenient, fields arrive at the top level.
        """
        # OpenEnv puts the observation dict in "observation" or uses the payload itself
        obs_data = payload.get("observation", payload)
        # 1. Reconstruct nested model lists
        pass_windows = [
            PassWindowModel(**w)
            for w in obs_data.get("pass_windows", [])
        ]
        
        data_priority_queues = {
            sid: [DataChunkModel(**c) for c in chunks]
            for sid, chunks in obs_data.get("data_priority_queues", {}).items()
        }

        current_schedule = [
            ScheduleEntryModel(**e)
            for e in obs_data.get("current_schedule", [])
        ]

        reward_data = obs_data.get("reward_obj", {"value": 0.0, "breakdown": {}})
        reward_obj = RewardModel(**reward_data)

        # 2. Build the 'thick' client-side observation object
        # We MUST pull 'done' and 'reward' from the top-level payload 
        # because the server places them alongside the observation dict.
        done_flag = payload.get("done", False)
        reward_val = float(payload.get("reward", 0.0))

        obs = SatelliteObservation(
            done=done_flag,
            reward=reward_val,
            info_dict=obs_data.get("info_dict", {}),
            current_time_min=obs_data.get("current_time_min", 0),
            reward_obj=reward_obj,
            pass_windows=pass_windows,
            station_availability=obs_data.get("station_availability", {}),
            satellite_buffer_bytes=obs_data.get("satellite_buffer_bytes", {}),
            data_priority_queues=data_priority_queues,
            downlink_rates_bps=obs_data.get("downlink_rates_bps", {}),
            current_schedule=current_schedule,
        )

        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=done_flag,
        )

    # ------------------------------------------------------------------
    # Required: deserialize state response → SatelliteState
    # ------------------------------------------------------------------

    def _parse_state(self, payload: dict) -> SatelliteState:
        """
        Parse the JSON payload from GET /state.
        SatelliteState is now a Pydantic model.
        """
        return SatelliteState.model_validate(payload)
