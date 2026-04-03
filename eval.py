import sys
import os

from src.envs.satellite_env.server.environment import SatelliteEnvironment
from src.envs.satellite_env.models import SatelliteAction, SatelliteObservation

def greedy_policy(obs: SatelliteObservation, tick: int) -> SatelliteAction:
    if obs.done:
        return SatelliteAction(action_type="noop")
    
    # Get availability
    avail = obs.station_availability
    
    # Filter windows active THIS tick
    active = [w for w in obs.pass_windows if w.tick == tick]
    if not active:
        return SatelliteAction(action_type="noop")
        
    # Sort by quality
    active.sort(key=lambda w: w.link_quality * float(avail.get(str(w.station_id), 1.0)), reverse=True)
    
    # Get buffers
    buffers = obs.satellite_buffer_bytes
    
    # Get schedule to avoid duplicate station schedules
    scheduled_stn = {e.station_id for e in obs.current_schedule if e.tick == tick}
    scheduled_sat = {e.satellite_id for e in obs.current_schedule if e.tick == tick}
    
    for w in active:
        # Check buffers
        if int(buffers.get(str(w.satellite_id), 0)) <= 0:
            continue
        # Check conflicts locally
        if w.station_id in scheduled_stn or w.satellite_id in scheduled_sat:
            continue
            
        return SatelliteAction(
            action_type="schedule",
            sat_id=w.satellite_id,
            station_id=w.station_id,
            window_id=w.window_id
        )
        
    return SatelliteAction(action_type="noop")


def run_eval(task: str):
    env = SatelliteEnvironment(task=task, seed=42)
    obs = env.reset()
    tick = 0
    
    while not obs.done and tick < 144:
        action = greedy_policy(obs, tick)
        obs = env.step(action)
        tick += 1
        
    state = env.state
    print(f"Task: {task}")
    print(f"  Score:  {state.final_score:.4f}")
    print(f"  Reward: {state.total_reward:.4f}")

if __name__ == "__main__":
    for task in ["task1", "task2", "task3"]:
        try:
            run_eval(task)
        except Exception as e:
            import traceback
            traceback.print_exc()
