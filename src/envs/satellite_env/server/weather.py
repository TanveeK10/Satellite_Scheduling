# src/envs/satellite_env/server/weather.py
"""
Seeded weather availability sampler.

Each ground station has an independent availability float in [0.0, 1.0]
resampled every 30 minutes (every 3 ticks).

Distribution: Beta(α=5, β=2)  →  mean ≈ 0.71, skewed toward clear sky.
Task 1 always returns 1.0 (clear sky, no weather).
Tasks 2 and 3 use the seeded sampler.

Usage:
    sampler = WeatherSampler(seed=42, task="task2")
    availability = sampler.get(tick=0)   # {0: 0.82, 1: 0.71, 2: 0.90, 3: 0.65}
    availability = sampler.get(tick=1)   # identical — same 30-min window
    availability = sampler.get(tick=3)   # new sample drawn
    sampler.reset()                      # back to tick-0 state
"""

from __future__ import annotations

import random
from typing import Dict

RESAMPLE_EVERY = 3  # ticks — 3 × 10 min = 30-min weather windows
NUM_STATIONS = 4
ALPHA = 4.0
BETA = 4.0


def _beta_sample(rng: random.Random) -> float:
    """
    Beta(α, β) via stdlib only — no scipy in the Docker image.

    Uses the Gamma-ratio identity:
        X = Gamma(α,1) / (Gamma(α,1) + Gamma(β,1))  →  X ~ Beta(α,β)

    random.gammavariate(shape, scale) is in the stdlib and produces
    accurate samples. This keeps our Docker image minimal.
    """
    g1 = rng.gammavariate(ALPHA, 1.0)
    g2 = rng.gammavariate(BETA, 1.0)
    return g1 / (g1 + g2)


class WeatherSampler:
    """
    Deterministic per-station availability sampler.

    Design:
    -------
    One RNG per station, each seeded independently. This means:
      - Station availabilities are uncorrelated (realistic — weather
        in Svalbard is independent of weather in Bangalore)
      - Querying stations in any order produces the same sequence
      - Each station's RNG advances exactly once per 30-min window

    A draw_count per station tracks how many Beta samples have been
    drawn from that station's RNG. On get(tick), if sample_idx is
    ahead of draw_count, we fast-forward by drawing and discarding.
    This is safe because the RNG is deterministic — draw order = result.

    Cache:
    ------
    _cache[sample_idx] stores the Dict[int, float] once computed.
    Subsequent calls with the same tick return the cached value
    without touching the RNG — get() is fully idempotent.
    """

    def __init__(self, seed: int, task: str) -> None:
        self._seed = seed
        self._task = task
        self._cache: Dict[int, Dict[int, float]] = {}
        # Per-station RNG and draw counter
        self._rngs: list[random.Random] = []
        self._draw_counts: list[int] = []
        self._init_rngs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, tick: int) -> Dict[int, float]:
        """
        Availability for all stations at the given tick.

        Returns Dict[int, float] — station_id → value in [0.0, 1.0].
        Always 1.0 for task1 regardless of tick.
        """
        if self._task == "task1":
            return {sid: 1.0 for sid in range(NUM_STATIONS)}

        sample_idx = tick // RESAMPLE_EVERY
        if sample_idx not in self._cache:
            self._cache[sample_idx] = self._draw(sample_idx)
        return dict(self._cache[sample_idx])  # return copy — caller can't mutate cache

    def get_str_keys(self, tick: int) -> Dict[str, float]:
        """
        Same as get() but with string keys for JSON / Pydantic compatibility.
        SatelliteObservation.station_availability is Dict[str, float].
        """
        return {str(k): v for k, v in self.get(tick).items()}

    def reset(self) -> None:
        """
        Restore sampler to its initial state.
        Must be called by environment.reset() so every episode starts
        from the same weather sequence — critical for reproducibility.
        """
        self._cache = {}
        self._init_rngs()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_rngs(self) -> None:
        """Seed one RNG per station. Called on __init__ and reset()."""
        self._rngs = [
            random.Random(self._seed + sid * 1_000)
            for sid in range(NUM_STATIONS)
        ]
        self._draw_counts = [0] * NUM_STATIONS

    def _draw(self, sample_idx: int) -> Dict[int, float]:
        """
        Produce one Beta sample per station for sample window sample_idx.

        For each station:
          - If draw_count < sample_idx: fast-forward by drawing and
            discarding (sample_idx - draw_count) values. This handles
            any non-sequential access pattern safely.
          - Then draw the actual sample and increment draw_count.

        Fast-forward is O(sample_idx) in the worst case but in normal
        episode execution get() is called sequentially tick-by-tick so
        fast-forward is never needed — each draw_count already equals
        sample_idx when we arrive.
        """
        result = {}
        for sid in range(NUM_STATIONS):
            rng = self._rngs[sid]
            count = self._draw_counts[sid]

            # Discard samples we've skipped over
            skip = sample_idx - count
            for _ in range(skip):
                _beta_sample(rng)  # draw and discard
            self._draw_counts[sid] += skip

            # Draw the real sample
            value = _beta_sample(rng)
            self._draw_counts[sid] += 1
            result[sid] = round(value, 4)

        return result
