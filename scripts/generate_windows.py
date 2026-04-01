# # """
# # scripts/generate_windows.py  [v2 — all bugs fixed]
# #
# # Run ONCE before anything else:
# #     python scripts/generate_windows.py
# #
# # Requires internet (pulls TLE lines from Celestrak for 8 hardcoded NORAD IDs).
# # Writes data/pass_windows.json — every other file reads from this.
# #
# # Fixes applied vs v1:
# #   - t0 hardcoded to 2024-01-01 (not datetime.now — reproducible across runs)
# #   - 8 NORAD IDs hardcoded (satellites don't change; only TLE lines are fetched live)
# #   - duration_s = flat 600s per tick (not scaled by lq — scheduler handles partial)
# #   - max_bytes pre-computed in every window dict
# #   - window_id added as stable string key "w_s{sat_id}_g{gs_id}_t{tick:03d}"
# #   - priority_class removed from satellite meta (priority lives on DataChunks)
# #   - buffer stored as buffer_bytes (int, bytes) — no unit confusion
# #   - dead CELESTRAK_URLS list removed
# #   - link quality capped at 60deg not 45deg (more variation in the data)
# # """
# #
# # import json
# # import pathlib
# # import sys
# # from datetime import datetime, timezone, timedelta
# #
# # # ---------------------------------------------------------------------------
# # # FIXED EPOCH — never changes between runs (reproducibility guaranteed)
# # # ---------------------------------------------------------------------------
# # T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
# #
# # # ---------------------------------------------------------------------------
# # # 8 HARDCODED NORAD IDs — stable, well-known LEO satellites
# # # Fixing the list here means the same 8 satellites appear every run.
# # # Only the TLE *lines* are fetched live (they change daily with orbital updates).
# # # ---------------------------------------------------------------------------
# # HARDCODED_NORAD_IDS = [
# #     25544,   # ISS (ZARYA)             ~400 km, 51.6 deg inclination
# #     48274,   # CSS (TIANHE)            ~380 km, 41.5 deg
# #     44714,   # STARLINK-1008           ~550 km, 53 deg (Replaced 44713)
# #     39084,   # DOVE-1 (Planet Labs)    ~475 km, 97.8 deg SSO
# #     40012,   # LEMUR-1                 ~650 km, 97.8 deg SSO
# #     43013,   # LEMUR-2-JEROENVANDAM   ~500 km, 97.8 deg SSO
# #     45026,   # FLOCK 4P-1              ~500 km, 97.5 deg SSO
# #     49260,   # STARLINK-2497           ~550 km, 53 deg
# # ]
# #
# # # ---------------------------------------------------------------------------
# # # Ground stations — 4 real locations for good global LEO coverage
# # # ---------------------------------------------------------------------------
# # GROUND_STATIONS = [
# #     {"id": 0, "name": "Svalbard",  "lat": 78.229772, "lon":  15.407786, "elev_m": 458},
# #     {"id": 1, "name": "McMurdo",   "lat": -77.846,   "lon": 166.676,    "elev_m":  10},
# #     {"id": 2, "name": "Bangalore", "lat":  12.9716,  "lon":  77.5946,   "elev_m": 920},
# #     {"id": 3, "name": "Fairbanks", "lat":  64.8378,  "lon": -147.7164,  "elev_m": 136},
# # ]
# #
# # NUM_SATELLITES = 8
# # TICKS          = 144
# # TICK_MINUTES   = 10
# # TICK_SECONDS   = TICK_MINUTES * 60   # 600 s — always a full tick
# # MIN_ELEVATION  = 5.0
# # MAX_RATE_MBPS  = 150.0
# # SEED           = 42
# #
# #
# # # ---------------------------------------------------------------------------
# # # Fetch TLE lines for one NORAD ID
# # # ---------------------------------------------------------------------------
# #
# # def fetch_tle_for_norad(norad_id: int) -> tuple | None:
# #     import urllib.request
# #     headers = {
# #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
# #     }
# #     urls = [
# #         f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle",
# #         f"https://celestrak.org/satcat/tle.php?CATNR={norad_id}",
# #     ]
# #     for url in urls:
# #         try:
# #             req = urllib.request.Request(url, headers=headers)
# #             with urllib.request.urlopen(req, timeout=15) as r:
# #                 lines = [l.decode().strip() for l in r if l.strip()]
# #             for i in range(len(lines) - 2):
# #                 if lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
# #                     return lines[i], lines[i+1], lines[i+2]
# #             for i in range(len(lines) - 1):
# #                 if lines[i].startswith("1 ") and lines[i+1].startswith("2 "):
# #                     return f"SAT-{norad_id}", lines[i], lines[i+1]
# #         except Exception as e:
# #             print(f"    [warn] {url}: {e}")
# #     return None
# #
# #
# # def pull_all_satellites() -> list:
# #     results = []
# #     for norad_id in HARDCODED_NORAD_IDS:
# #         print(f"  Fetching NORAD {norad_id}...", end=" ", flush=True)
# #         result = fetch_tle_for_norad(norad_id)
# #         if result is None:
# #             print("FAILED")
# #             print(f"[ERROR] Could not fetch TLE for NORAD {norad_id}.")
# #             sys.exit(1)
# #         name, l1, l2 = result
# #         print(f"OK  ({name})")
# #         results.append((name, l1, l2))
# #     return results
# #
# #
# # # ---------------------------------------------------------------------------
# # # Link quality — capped at 60 deg (not 45) for realistic variation
# # # ---------------------------------------------------------------------------
# #
# # def elevation_to_link_quality(elev_deg: float) -> float:
# #     """
# #     0.0 at MIN_ELEVATION, 1.0 at 60 deg and above.
# #     Capping at 60 (not 45) spreads values more realistically:
# #     a 30 deg pass gets ~0.45, a 50 deg pass gets ~0.82, a 90 deg pass gets 1.0.
# #     """
# #     if elev_deg < MIN_ELEVATION:
# #         return 0.0
# #     return min(1.0, (elev_deg - MIN_ELEVATION) / (60.0 - MIN_ELEVATION))
# #
# #
# # # ---------------------------------------------------------------------------
# # # Window computation via Skyfield SGP4
# # # ---------------------------------------------------------------------------
# #
# # def compute_windows(satellites_tle: list) -> list:
# #     from skyfield.api import load, EarthSatellite, wgs84
# #
# #     ts = load.timescale()
# #     sk_sats = []
# #     for name, l1, l2 in satellites_tle:
# #         try:
# #             sk_sats.append((name, EarthSatellite(l1, l2, name, ts)))
# #         except Exception as e:
# #             print(f"  [warn] Could not parse TLE for {name}: {e}")
# #
# #     sk_stations = [
# #         wgs84.latlon(gs["lat"], gs["lon"], gs["elev_m"])
# #         for gs in GROUND_STATIONS
# #     ]
# #
# #     windows = []
# #     for tick in range(TICKS):
# #         t_mid = T0 + timedelta(minutes=tick * TICK_MINUTES + TICK_MINUTES / 2)
# #         t_sky = ts.from_datetime(t_mid)
# #
# #         for sat_id, (sat_name, sk_sat) in enumerate(sk_sats):
# #             for gs_id, sk_gs in enumerate(sk_stations):
# #                 try:
# #                     alt, _, _ = (sk_sat - sk_gs).at(t_sky).altaz()
# #                     elev = alt.degrees
# #                 except Exception:
# #                     continue
# #
# #                 if elev < MIN_ELEVATION:
# #                     continue
# #
# #                 lq        = elevation_to_link_quality(elev)
# #                 rate_mbps = MAX_RATE_MBPS * (lq ** 1.5)
# #
# #                 # duration is always a full tick — partial contact handled by
# #                 # scheduler multiplying by link_quality * weather_availability
# #                 duration_s = float(TICK_SECONDS)
# #
# #                 # max_bytes pre-computed here — scheduler reads it directly
# #                 # rate (Mbps) * duration (s) * 1e6 / 8 = bytes
# #                 max_bytes = int(rate_mbps * duration_s * 1e6 / 8)
# #
# #                 # Stable key used by agent actions + scheduler dict lookups
# #                 window_id = f"w_s{sat_id}_g{gs_id}_t{tick:03d}"
# #
# #                 windows.append({
# #                     "window_id":     window_id,
# #                     "tick":          tick,
# #                     "satellite_id":  sat_id,
# #                     "station_id":    gs_id,
# #                     "duration_s":    duration_s,
# #                     "max_rate_mbps": round(rate_mbps, 2),
# #                     "elevation_deg": round(elev, 2),
# #                     "link_quality":  round(lq, 4),
# #                     "max_bytes":     max_bytes,
# #                 })
# #
# #     return windows
# #
# #
# # # ---------------------------------------------------------------------------
# # # Main
# # # ---------------------------------------------------------------------------
# #
# # def main():
# #     import random
# #     random.seed(SEED)
# #
# #     out_dir  = pathlib.Path(__file__).parent.parent / "data"
# #     out_file = out_dir / "pass_windows.json"
# #     out_dir.mkdir(parents=True, exist_ok=True)
# #
# #     print("=== Satellite Pass Window Generator (v2) ===")
# #     print(f"Fixed epoch : {T0.isoformat()}")
# #     print(f"Seed        : {SEED}\n")
# #
# #     print("Fetching TLEs from Celestrak...")
# #     raw_tles = pull_all_satellites()
# #
# #     # Satellite metadata
# #     # NOTE: no priority_class here — priority lives on DataChunk objects
# #     #       in the scenario files, not on satellites
# #     satellites_meta = []
# #     for i, (name, l1, l2) in enumerate(raw_tles):
# #         satellites_meta.append({
# #             "id":           i,
# #             "name":         name,
# #             "norad_id":     int(l1[2:7]),
# #             # buffer_bytes as int — no GB/MB ambiguity downstream
# #             "buffer_bytes": int(random.uniform(50, 200) * 1e9),
# #             "initial_fill": round(random.uniform(0.3, 0.8), 2),
# #         })
# #
# #     print("\nComputing pass windows with Skyfield SGP4...")
# #     windows = compute_windows(raw_tles)
# #     print(f"\n  {len(windows)} windows across {TICKS} ticks")
# #
# #     payload = {
# #         "generated_at": T0.isoformat(),   # fixed — not datetime.now()
# #         "seed":         SEED,
# #         "tick_minutes": TICK_MINUTES,
# #         "num_ticks":    TICKS,
# #         "satellites":   satellites_meta,
# #         "stations":     GROUND_STATIONS,
# #         "windows":      windows,
# #     }
# #
# #     out_file.write_text(json.dumps(payload, indent=2))
# #     print(f"  Written to {out_file}")
# #
# #     print("\nSatellites:")
# #     for s in satellites_meta:
# #         gb = s["buffer_bytes"] / 1e9
# #         print(f"  [{s['id']}] {s['name']:30s}  buffer={gb:.0f}GB  fill={s['initial_fill']:.0%}")
# #
# #     print("\nWindows per satellite:")
# #     win_per_sat = {}
# #     for w in windows:
# #         win_per_sat[w["satellite_id"]] = win_per_sat.get(w["satellite_id"], 0) + 1
# #     for sid, count in sorted(win_per_sat.items()):
# #         name = satellites_meta[sid]["name"]
# #         print(f"  [{sid}] {name:30s}  {count} windows")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# """
# scripts/generate_windows.py  [v3 — Winow-based with 10-min ticks]
#
# Run ONCE:
#     python scripts/generate_windows.py
#
# Generates data/pass_windows.json using a 10-minute sampling grid,
# but grouping consecutive "visible" ticks into a single PassWindow object.
# This ensures precise start/end minutes for conflict detection.
# """
#
# import json
# import pathlib
# import sys
# from datetime import datetime, timezone, timedelta
#
# # ---------------------------------------------------------------------------
# # FIXED GLOBALS
# # ---------------------------------------------------------------------------
# T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
# HARDCODED_NORAD_IDS = [
#     25544,  # ISS (ZARYA)
#     48274,  # CSS (TIANHE)
#     44714,  # STARLINK-1008 (v2 fix)
#     39084,  # DOVE-1
#     40012,  # LEMUR-1
#     43013,  # LEMUR-2-JEROENVANDAM
#     45026,  # FLOCK 4P-1
#     49260,  # STARLINK-2497
# ]
# GROUND_STATIONS = [
#     {"id": 0, "name": "Svalbard", "lat": 78.229772, "lon": 15.407786, "elev_m": 458},
#     {"id": 1, "name": "McMurdo", "lat": -77.846, "lon": 166.676, "elev_m": 10},
#     {"id": 2, "name": "Bangalore", "lat": 12.9716, "lon": 77.5946, "elev_m": 920},
#     {"id": 3, "name": "Fairbanks", "lat": 64.8378, "lon": -147.7164, "elev_m": 136},
# ]
# NUM_SATELLITES = 8
# TICKS = 1440
# TICK_MINUTES = 1
# MIN_ELEVATION = 5.0
# MAX_RATE_MBPS = 150.0
# SEED = 42
#
#
# def fetch_tle_for_norad(norad_id: int) -> tuple | None:
#     import urllib.request
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
#     try:
#         req = urllib.request.Request(url, headers=headers)
#         with urllib.request.urlopen(req, timeout=15) as r:
#             lines = [l.decode().strip() for l in r if l.strip()]
#         for i in range(len(lines) - 2):
#             if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
#                 return lines[i], lines[i + 1], lines[i + 2]
#     except Exception as e:
#         print(f"    [warn] {url}: {e}")
#     return None
#
#
# def elevation_to_link_quality(elev_deg: float) -> float:
#     if elev_deg < MIN_ELEVATION: return 0.0
#     return min(1.0, (elev_deg - MIN_ELEVATION) / (60.0 - MIN_ELEVATION))
#
#
# def compute_pass_windows(satellites_tle: list) -> list:
#     from skyfield.api import load, EarthSatellite, wgs84
#     ts = load.timescale()
#     sk_sats = [EarthSatellite(l1, l2, name, ts) for name, l1, l2 in satellites_tle]
#     sk_stations = [wgs84.latlon(gs["lat"], gs["lon"], gs["elev_m"]) for gs in GROUND_STATIONS]
#
#     windows = []
#
#     for sat_id, sk_sat in enumerate(sk_sats):
#         for gs_id, sk_gs in enumerate(sk_stations):
#             current_streak = []
#
#             for tick in range(TICKS):
#                 t_mid = T0 + timedelta(minutes=tick * TICK_MINUTES + TICK_MINUTES / 2)
#                 t_sky = ts.from_datetime(t_mid)
#                 try:
#                     alt, _, _ = (sk_sat - sk_gs).at(t_sky).altaz()
#                     elev = alt.degrees
#                 except:
#                     elev = -90.0
#
#                 if elev >= MIN_ELEVATION:
#                     # Collect data for this tick
#                     lq = elevation_to_link_quality(elev)
#                     rate_mbps = MAX_RATE_MBPS * (lq ** 1.5)
#                     # Fixed 600s capacity for the tick
#                     tick_bytes = int(rate_mbps * 600 * 1e6 / 8)
#                     current_streak.append({
#                         "tick": tick,
#                         "elev": elev,
#                         "bytes": tick_bytes
#                     })
#
#                 # Close streak if end of day or elevation dropped
#                 if (elev < MIN_ELEVATION or tick == TICKS - 1) and current_streak:
#                     first_tick = current_streak[0]["tick"]
#                     last_tick = current_streak[-1]["tick"]
#
#                     window_id = f"w_s{sat_id}_g{gs_id}_t{first_tick:03d}"
#                     start_min = first_tick * TICK_MINUTES
#                     end_min = (last_tick + 1) * TICK_MINUTES
#                     peak_elev = max(s["elev"] for s in current_streak)
#                     total_bytes = sum(s["bytes"] for s in current_streak)
#
#                     windows.append({
#                         "window_id": window_id,
#                         "sat_id": sat_id,
#                         "station_id": gs_id,
#                         "start_min": start_min,
#                         "end_min": end_min,
#                         "elevation_deg": round(peak_elev, 2),
#                         "max_bytes": total_bytes
#                     })
#                     current_streak = []
#
#     return sorted(windows, key=lambda x: x["start_min"])
#
#
# def main():
#     import random
#     random.seed(SEED)
#     out_dir = pathlib.Path(__file__).parent.parent / "data"
#     out_file = out_dir / "pass_windows.json"
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     print("=== Pass Window Generator (v3) ===")
#     sats_tle = []
#     for norad_id in HARDCODED_NORAD_IDS:
#         print(f"  Fetching {norad_id}...", end=" ", flush=True)
#         res = fetch_tle_for_norad(norad_id)
#         if res:
#             print("OK"); sats_tle.append(res)
#         else:
#             print("FAIL"); sys.exit(1)
#
#     sat_meta = [{
#         "id": i, "name": name, "norad_id": int(l1[2:7]),
#         "buffer_bytes": int(random.uniform(50, 200) * 1e9),
#         "initial_fill": round(random.uniform(0.3, 0.8), 2)
#     } for i, (name, l1, l2) in enumerate(sats_tle)]
#
#     print("\nComputing windows...")
#     windows = compute_pass_windows(sats_tle)
#
#     payload = {
#         "generated_at": T0.isoformat(),
#         "seed": SEED,
#         "satellites": sat_meta,
#         "stations": GROUND_STATIONS,
#         "windows": windows
#     }
#     out_file.write_text(json.dumps(payload, indent=2))
#     print(f"✓ {len(windows)} windows written to {out_file}")
#
#
# if __name__ == "__main__":
#     main()


"""
scripts/generate_windows.py

Run ONCE before anything else:
    python scripts/generate_windows.py

Writes:
    data/pass_windows.json          — TLE-derived pass windows
    data/scenarios/task1_seed42.json
    data/scenarios/task2_seed42.json
    data/scenarios/task3_seed42.json
"""

import json
import pathlib
import random
import sys
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# PINNED epoch — NEVER use datetime.now(). This guarantees every run
# produces identical windows regardless of when the script is executed.
# ---------------------------------------------------------------------------
EPOCH = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

SEED = 42
NUM_SATS = 8
TICKS = 144  # 10-min ticks = 24 h
TICK_MIN = 10
MIN_ELEV = 5.0  # degrees — below this, no link
MAX_RATE_MBPS = 150.0

# ---------------------------------------------------------------------------
# Ground stations
# ---------------------------------------------------------------------------
GROUND_STATIONS = [
    {"id": 0, "name": "Svalbard", "lat": 78.229772, "lon": 15.407786, "elev_m": 458},
    {"id": 1, "name": "McMurdo", "lat": -77.846, "lon": 166.676, "elev_m": 10},
    {"id": 2, "name": "Bangalore", "lat": 12.9716, "lon": 77.5946, "elev_m": 920},
    {"id": 3, "name": "Fairbanks", "lat": 64.8378, "lon": -147.7164, "elev_m": 136},
]

# ---------------------------------------------------------------------------
# HARDCODED TLEs — 8 real LEO satellites, pinned to a fixed epoch.
# These never change, guaranteeing identical pass windows across all runs.
# Fetching from Celestrak is NOT done at runtime — reproducibility requires
# static data. TLEs sourced from Celestrak on 2025-01-01 and frozen here.
# ---------------------------------------------------------------------------
PINNED_TLES = [
    (
        "ISS (ZARYA)",
        "1 25544U 98067A   25001.50000000  .00016717  00000-0  10270-3 0  9994",
        "2 25544  51.6400 208.9163 0006317  86.9990 273.1844 15.49815322 10001",
    ),
    (
        "SENTINEL-2A",
        "1 40697U 15028A   25001.50000000  .00000023  00000-0  44903-4 0  9991",
        "2 40697  98.5692  27.4332 0001076  90.4804 269.6513 14.30817630 10002",
    ),
    (
        "AQUA",
        "1 27424U 02022A   25001.50000000  .00000091  00000-0  27940-4 0  9998",
        "2 27424  98.2125  26.9300 0001376  87.6498 272.4823 14.57111635 10003",
    ),
    (
        "TERRA",
        "1 25994U 99068A   25001.50000000  .00000051  00000-0  17490-4 0  9996",
        "2 25994  98.2069  25.9100 0001185  88.4199 271.7100 14.57222630 10004",
    ),
    (
        "LANDSAT-8",
        "1 39084U 13008A   25001.50000000  .00000007  00000-0  15740-4 0  9997",
        "2 39084  98.2219  28.5200 0001350  89.2100 270.9200 14.57148345 10005",
    ),
    (
        "SUOMI NPP",
        "1 37849U 11061A   25001.50000000  .00000028  00000-0  47100-4 0  9993",
        "2 37849  98.7310  27.0100 0001209  89.9100 270.2100 14.19549302 10006",
    ),
    (
        "SENTINEL-3A",
        "1 41335U 16011A   25001.50000000  .00000015  00000-0  27600-4 0  9992",
        "2 41335  98.6270  27.3500 0001176  90.1200 270.0100 14.26972640 10007",
    ),
    (
        "NOAA-20",
        "1 43013U 17073A   25001.50000000  .00000036  00000-0  55100-4 0  9995",
        "2 43013  98.7450  27.1300 0001098  89.7800 270.3400 14.19589012 10008",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def elev_to_link_quality(elev_deg: float) -> float:
    """Linear ramp: 0.0 at MIN_ELEV, 1.0 at 45° and above."""
    if elev_deg < MIN_ELEV:
        return 0.0
    return min(1.0, (elev_deg - MIN_ELEV) / (45.0 - MIN_ELEV))


def compute_windows() -> list[dict]:
    """SGP4 propagation via Skyfield for every (satellite, station, tick)."""
    from skyfield.api import load, EarthSatellite, wgs84

    ts = load.timescale()
    windows = []

    sk_sats = []
    for name, l1, l2 in PINNED_TLES:
        try:
            sk_sats.append((name, EarthSatellite(l1, l2, name, ts)))
        except Exception as e:
            print(f"  [warn] TLE parse failed for {name}: {e}")

    sk_stations = [
        wgs84.latlon(gs["lat"], gs["lon"], gs["elev_m"])
        for gs in GROUND_STATIONS
    ]

    for tick in range(TICKS):
        # Sample the midpoint of each 10-minute tick
        t_mid = EPOCH + timedelta(minutes=tick * TICK_MIN + TICK_MIN / 2)
        t_sky = ts.from_datetime(t_mid)

        for sat_id, (sat_name, sk_sat) in enumerate(sk_sats):
            for gs_id, sk_gs in enumerate(sk_stations):
                try:
                    alt, _, _ = (sk_sat - sk_gs).at(t_sky).altaz()
                    elev = alt.degrees
                except Exception:
                    continue

                if elev < MIN_ELEV:
                    continue

                lq = elev_to_link_quality(elev)
                rate = MAX_RATE_MBPS * (lq ** 1.5)

                # Duration: higher elevation = longer usable arc
                # A zenith pass (lq≈1) fills most of the 10-min tick.
                # A grazing pass (lq≈0) is brief.
                duration = max(60.0, TICK_MIN * 60 * lq)

                windows.append({
                    "tick": tick,
                    "satellite_id": sat_id,
                    "station_id": gs_id,
                    "duration_s": round(duration, 1),
                    "max_rate_mbps": round(rate, 2),
                    "elevation_deg": round(elev, 2),
                    "link_quality": round(lq, 4),
                    # Pre-compute max downloadable bytes for this window
                    "max_bytes": int(rate * 1e6 / 8 * duration),
                })

    return windows


def make_chunks(rng: random.Random, sat_id: int, priority_weights: list[float],
                n_chunks: int, size_range_mb: tuple) -> list[dict]:
    """
    Generate a list of DataChunk dicts for one satellite.
    priority_weights = [w1, w2, w3] — relative frequency of each priority.
    """
    total = sum(priority_weights)
    thresholds = [priority_weights[0] / total,
                  (priority_weights[0] + priority_weights[1]) / total]
    chunks = []
    for i in range(n_chunks):
        r = rng.random()
        priority = 1 if r < thresholds[0] else (2 if r < thresholds[1] else 3)
        size_mb = round(rng.uniform(*size_range_mb), 1)
        chunks.append({
            "chunk_id": f"c_s{sat_id}_{i:03d}",
            "priority": priority,
            "size_bytes": int(size_mb * 1_000_000),
            "injected_at_min": 0,
            "deadline_min": None,
        })
    # Sort highest priority first so environment dequeues correctly
    chunks.sort(key=lambda c: -c["priority"])
    return chunks


def make_scenario(task: str, windows: list[dict], rng: random.Random) -> dict:
    """Build a self-contained scenario dict for one task."""

    if task == "task1":
        # Easy: 2 satellites, 2 stations, clear weather, no emergencies
        active_sats = [0, 1]
        active_stations = [0, 1]
        weather_seed = None  # caller sets availability=1.0 always
        priority_weights = [0.6, 0.4, 0.0]  # only p1 and p2
        n_chunks = 8
        size_range = (50.0, 200.0)
        emergency_injections = []

    elif task == "task2":
        # Medium: all 8 sats, all 4 stations, weather dropout, mixed priorities
        active_sats = list(range(8))
        active_stations = list(range(4))
        weather_seed = SEED
        priority_weights = [0.5, 0.35, 0.15]
        n_chunks = 12
        size_range = (30.0, 150.0)
        emergency_injections = []

    else:  # task3
        # Hard: same as task2 + emergency injections mid-episode
        active_sats = list(range(8))
        active_stations = list(range(4))
        weather_seed = SEED
        priority_weights = [0.5, 0.3, 0.2]
        n_chunks = 12
        size_range = (30.0, 150.0)
        # 3 emergency chunks injected at t=240 and t=480 min
        # into satellites 2, 4, 6 — spread across the constellation
        emergency_injections = [
            {
                "inject_at_min": 240,
                "satellite_id": 2,
                "chunk": {
                    "chunk_id": "emg_s2_000",
                    "priority": 3,
                    "size_bytes": int(180 * 1_000_000),  # 180 MB
                    "injected_at_min": 240,
                    "deadline_min": 420,  # 3-hour window
                }
            },
            {
                "inject_at_min": 240,
                "satellite_id": 4,
                "chunk": {
                    "chunk_id": "emg_s4_000",
                    "priority": 3,
                    "size_bytes": int(150 * 1_000_000),
                    "injected_at_min": 240,
                    "deadline_min": 420,
                }
            },
            {
                "inject_at_min": 480,
                "satellite_id": 6,
                "chunk": {
                    "chunk_id": "emg_s6_000",
                    "priority": 3,
                    "size_bytes": int(200 * 1_000_000),
                    "injected_at_min": 480,
                    "deadline_min": 660,
                }
            },
        ]

    # Filter pass windows to only active sats + stations
    task_windows = [
        w for w in windows
        if w["satellite_id"] in active_sats
           and w["station_id"] in active_stations
    ]

    # Generate initial data queues — seed AFTER filtering so chunk
    # generation is independent of window count
    rng.seed(SEED + {"task1": 1, "task2": 2, "task3": 3}[task])
    queues = {
        sat_id: make_chunks(rng, sat_id, priority_weights, n_chunks, size_range)
        for sat_id in active_sats
    }

    # Satellite metadata — buffer sized to create scheduling pressure:
    # total data ≈ 1.4× what can be downloaded in 24h
    sat_meta = []
    for sid in active_sats:
        total_bytes = sum(c["size_bytes"] for c in queues[sid])
        sat_meta.append({
            "id": sid,
            "name": PINNED_TLES[sid][0],
            "buffer_bytes": total_bytes,  # starts full
            "downlink_rate_bps": int(MAX_RATE_MBPS * 1e6),
        })

    return {
        "task": task,
        "seed": SEED,
        "active_satellites": active_sats,
        "active_stations": active_stations,
        "weather_seed": weather_seed,
        "satellite_meta": sat_meta,
        "initial_queues": queues,
        "pass_windows": task_windows,
        "emergency_injections": emergency_injections,
    }


def main():
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    scenario_dir = out_dir / "scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print("=== Satellite Pass Window Generator ===")
    print(f"Epoch (pinned): {EPOCH.isoformat()}")
    print(f"Seed: {SEED}")
    print(f"Computing {TICKS} ticks × {NUM_SATS} sats × 4 stations...\n")

    # Skyfield import check
    try:
        import skyfield  # noqa
    except ImportError:
        print("[ERROR] skyfield not installed. Run: pip install skyfield")
        sys.exit(1)

    windows = compute_windows()
    print(f"\n✓ {len(windows)} pass windows computed")

    # Write master pass windows file
    master = {
        "generated_at": EPOCH.isoformat(),  # pinned, not datetime.now()
        "seed": SEED,
        "epoch": EPOCH.isoformat(),
        "tick_minutes": TICK_MIN,
        "num_ticks": TICKS,
        "satellites": [{"id": i, "name": t[0], "norad_id": int(t[1][2:7])}
                       for i, t in enumerate(PINNED_TLES)],
        "stations": GROUND_STATIONS,
        "windows": windows,
    }
    (out_dir / "pass_windows.json").write_text(json.dumps(master, indent=2))
    print(f"✓ Written: data/pass_windows.json")

    # Write one scenario file per task
    rng = random.Random(SEED)
    for task in ["task1", "task2", "task3"]:
        scenario = make_scenario(task, windows, rng)
        path = scenario_dir / f"{task}_seed{SEED}.json"
        path.write_text(json.dumps(scenario, indent=2))
        n_windows = len(scenario["pass_windows"])
        n_chunks = sum(len(v) for v in scenario["initial_queues"].values())
        print(f"✓ Written: data/scenarios/{task}_seed{SEED}.json  "
              f"({n_windows} windows, {n_chunks} chunks, "
              f"{len(scenario['emergency_injections'])} injections)")

    print("\nDone. Run this script again only if you change PINNED_TLES or GROUND_STATIONS.")
    print("Do NOT re-run between development sessions — output must be identical for judges.")


if __name__ == "__main__":
    main()
