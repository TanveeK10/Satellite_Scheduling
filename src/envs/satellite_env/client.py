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
        print(result.observation.info)

Usage (async — for training loops):
    async with SatelliteEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        result = await env.step(SatelliteAction(action_type="noop"))

From HF Space:
    async with SatelliteEnv(base_url="https://your-name-satellite-env.hf.space") as env:
        result = await env.reset()
"""

from __future__ import annotations

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from src.envs.satellite_env.models import (
    DataChunkModel,
    PassWindowModel,
    SatelliteAction,
    SatelliteObservation,
    ScheduleEntryModel,
)


class SatelliteEnv(EnvClient[SatelliteAction, SatelliteObservation, dict]):
    """
    Typed client for the SatelliteEnvironment server.

    Inherits from HTTPEnvClient which handles:
        - WebSocket connection lifecycle
        - Reconnection and error handling
        - .sync() wrapper for synchronous usage
        - .from_docker_image() for auto-starting containers
        - .from_env() for pulling and running HF Spaces

    We only implement three methods:
        _step_payload  — Action → JSON dict (sent to server)
        _parse_result  — JSON dict → StepResult[SatelliteObservation]
        _parse_state   — JSON dict → SatelliteState (dict form)
    """

    # ------------------------------------------------------------------
    # Required: serialize action → dict for WebSocket transport
    # ------------------------------------------------------------------

    def _step_payload(self, action: SatelliteAction) -> dict:
        """
        Convert a SatelliteAction to a JSON-serializable dict.
        Only include non-None fields to keep the payload compact.
        The server's Pydantic model handles missing optional fields.
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
        Parse the JSON payload returned by the server after reset() or step().

        The server serialises SatelliteObservation to JSON automatically.
        We reconstruct the nested Pydantic models manually because the
        nested types (PassWindowModel, DataChunkModel, ScheduleEntryModel)
        arrive as plain dicts over the wire.

        payload keys (from SatelliteObservation):
            current_time_min, done, reward,
            pass_windows, station_availability,
            satellite_buffer_bytes, data_priority_queues,
            downlink_rates_bps, current_schedule, info
        """
        obs_data = payload.get("observation", payload)

        # Reconstruct nested model lists from plain dicts
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

        obs = SatelliteObservation(
            current_time_min=obs_data.get("current_time_min", 0),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            pass_windows=pass_windows,
            station_availability=obs_data.get("station_availability", {}),
            satellite_buffer_bytes=obs_data.get("satellite_buffer_bytes", {}),
            data_priority_queues=data_priority_queues,
            downlink_rates_bps=obs_data.get("downlink_rates_bps", {}),
            current_schedule=current_schedule,
            info=obs_data.get("info", {}),
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            done=payload.get("done", obs_data.get("done", False)),
        )

    # ------------------------------------------------------------------
    # Required: deserialize state response → dict (SatelliteState)
    # ------------------------------------------------------------------

    def _parse_state(self, payload: dict) -> dict:
        """
        Parse the JSON payload from GET /state.
        Returns the state as a plain dict — the server serialises
        SatelliteState automatically, so keys match the dataclass fields.
        """
        return payload
