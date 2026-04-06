import json
import random
import os

def scale_scenario(task_name, target_sats, target_stations):
    path = f"data/scenarios/{task_name}_seed42.json"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    # 1. Active Lists
    data['active_satellites'] = list(range(target_sats))
    data['active_stations'] = list(range(target_stations))

    # 2. Metadata & Queues
    base_meta = data['satellite_meta'][0]
    base_queue = data['initial_queues']['0']

    new_meta_list = []
    new_queues = {}
    for sid in range(target_sats):
        m = dict(base_meta)
        m['id'] = sid
        m['name'] = f"Sat_{sid:02d}"
        new_meta_list.append(m)
        
        q = []
        for c in base_queue:
            nc = dict(c)
            nc['chunk_id'] = f"c_s{sid}_{c['chunk_id'].split('_')[-1]}"
            q.append(nc)
        new_queues[str(sid)] = q
    
    data['satellite_meta'] = new_meta_list
    data['initial_queues'] = new_queues

    # 3. Enhanced Emergency Injections (The PREEMPTION Pressure)
    # We want 8 randomized emergency bursts per mission.
    # Each burst adds P3 data to 2-3 random satellites.
    injections = []
    # Force 8 distinct burst times across the 144 ticks
    burst_ticks = sorted(random.sample(range(10, 130), 8))
    
    for b_tick in burst_ticks:
        affected_sats = random.sample(range(target_sats), 3)
        for sid in affected_sats:
            inj = {
                "sat_id": sid,
                "tick": b_tick,
                "chunks": [
                    {
                        "chunk_id": f"inj_s{sid}_t{b_tick}",
                        "priority": 3,
                        "size_bytes": 12000000000, # 12 GB (requires ~5-6 ticks at 100 Mbps)
                        "deadline_min": b_tick * 10 + 60 # 60-min deadline
                    }
                ]
            }
            injections.append(inj)
    
    data['emergency_injections'] = injections

    # 4. Moderate-Overlap Window Generation
    new_windows = []
    for tick in range(144):
        # Allow up to 3 satellites to have windows at once
        active_sids = random.sample(range(target_sats), 3)
        # Distribute across the 6 stations
        active_gids = random.sample(range(target_stations), 3)
        
        for i in range(len(active_sids)):
            sid = active_sids[i]
            gid = active_gids[i]
            
            win = {
                "window_id": f"w_s{sid}_g{gid}_{tick:03d}",
                "sat_id": sid,
                "station_id": gid,
                "tick": tick,
                "duration_s": 600.0,
                "max_rate_mbps": 100.0, # Synchronized with scheduler
                "elevation_deg": random.uniform(20, 80),
                "link_quality": 1.0,
                "max_bytes": 7500000000 # 100Mbps * 600s / 8
            }
            new_windows.append(win)

    data['pass_windows'] = new_windows

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Successfully scaled {task_name} with 8 Emergency Bursts.")

# Execute scaling
scale_scenario("task1", target_sats=8, target_stations=4)
scale_scenario("task2", target_sats=15, target_stations=6)
scale_scenario("task3", target_sats=15, target_stations=6)
