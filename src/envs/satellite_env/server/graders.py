# src/envs/satellite_env/server/graders.py
"""
Graders — deterministic episode scoring for all three tasks.

Public API:
    grade(task, download_log, all_chunks, emergency_injections) -> float

Each task grader returns a float in [0.0, 1.0].
Same inputs always produce the same output — no randomness here.

Grader breakdown:
    task1 — raw bytes downloaded / bytes available
    task2 — priority-weighted bytes / total priority-weighted bytes
    task3 — 0.4 × task2_score + 0.6 × emergency_score − delay_penalties
"""

from __future__ import annotations

from typing import List, Dict, Optional

from src.envs.satellite_env.server.scheduler import DownlinkResult

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

PRIORITY_WEIGHT: Dict[int, float] = {1: 1.0, 2: 2.0, 3: 3.0}

# Maximum delay penalty per emergency chunk (subtracted from score)
DELAY_PENALTY_MAX = 0.10

# Weight split for task3
TASK3_BASE_WEIGHT = 0.4
TASK3_EMERGENCY_WEIGHT = 0.6


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def grade(
        task: str,
        download_log: List[DownlinkResult],
        all_chunks: List[dict],
        emergency_injections: List[dict],
) -> float:
    """
    Score a completed episode.

    Args:
        task                 — "task1" | "task2" | "task3"
        download_log         — Scheduler.get_download_log() output
        all_chunks           — flat list of every DataChunk dict in the episode
                               (initial queues + injected emergency chunks)
        emergency_injections — scenario["emergency_injections"] list
                               (used for deadline tracking in task3)

    Returns float in [0.0, 1.0].
    Clipped at 0.0 — penalties cannot push score below zero.
    """
    if task == "task1":
        return _grade_task1(download_log, all_chunks)
    elif task == "task2":
        return _grade_task2(download_log, all_chunks)
    elif task == "task3":
        return _grade_task3(download_log, all_chunks, emergency_injections)
    else:
        raise ValueError(f"Unknown task: '{task}'. Must be task1, task2, or task3.")


# ─────────────────────────────────────────────────────────────
# Task 1 — raw bytes
# ─────────────────────────────────────────────────────────────

def _grade_task1(
        download_log: List[DownlinkResult],
        all_chunks: List[dict],
) -> float:
    """
    Score = bytes_downloaded / bytes_available

    No priority weighting — pure throughput efficiency.
    Measures whether the agent assigned available windows at all.

    Expected scores:
        noop agent   → 0.0
        greedy agent → ~0.75
        perfect agent → 1.0
    """
    bytes_available = sum(c["size_bytes"] for c in all_chunks)
    bytes_downloaded = _total_bytes(download_log)

    if bytes_available == 0:
        return 0.0

    score = bytes_downloaded / bytes_available
    return _clip(score)


# ─────────────────────────────────────────────────────────────
# Task 2 — priority-weighted bytes
# ─────────────────────────────────────────────────────────────

def _grade_task2(
        download_log: List[DownlinkResult],
        all_chunks: List[dict],
) -> float:
    """
    Score = Σ w(priority) × bytes_downloaded
          / Σ w(priority) × bytes_available

    Rewards the agent for downloading high-priority data first.
    Downloading 1 MB of priority-3 scores the same as 3 MB of priority-1.

    Expected scores:
        noop agent     → 0.0
        greedy agent   → ~0.45  (ignores priority, weather hurts it)
        priority-aware → ~0.70
        weather ceiling → ~0.88 (no agent can exceed this — weather limits throughput)
    """
    weighted_available = _weighted_total(all_chunks)
    weighted_downloaded = _weighted_downloaded(download_log)

    if weighted_available == 0:
        return 0.0

    score = weighted_downloaded / weighted_available
    return _clip(score)


# ─────────────────────────────────────────────────────────────
# Task 3 — emergency performance + delay penalties
# ─────────────────────────────────────────────────────────────

def _grade_task3(
        download_log: List[DownlinkResult],
        all_chunks: List[dict],
        emergency_injections: List[dict],
) -> float:
    """
    Score = 0.4 × base_score
          + 0.6 × emergency_score
          − Σ delay_penalties

    base_score      — same as task2 (priority-weighted throughput)
    emergency_score — fraction of emergency bytes downloaded before deadline
    delay_penalties — 0.0 to 0.10 per emergency chunk downloaded late or missed

    This weighting means:
        An agent that ignores emergency data entirely scores at most 0.4 × 0.88 ≈ 0.35
        An agent that handles emergencies perfectly can reach ~0.75
        (weather cap on base_score prevents reaching 1.0)

    Expected scores:
        noop agent              → ~−0.30  (clipped to 0.0)
        ignores emergencies     → ~0.30
        handles emergencies well → ~0.65
        strong agent             → ~0.75
    """
    # Base component — reuse task2 formula
    base_score = _grade_task2(download_log, all_chunks)

    # Build lookup of emergency chunk metadata keyed by chunk_id
    emg_meta: Dict[str, dict] = {}
    for inj in emergency_injections:
        c = inj["chunk"]
        emg_meta[c["chunk_id"]] = c

    if not emg_meta:
        # No emergency chunks in this scenario — fall back to task2
        return base_score

    # ── Emergency score component ─────────────────────────────────────
    # Total weighted emergency bytes available
    emg_total = sum(
        PRIORITY_WEIGHT[3] * c["size_bytes"]
        for c in emg_meta.values()
    )

    # Bytes from emergency chunks actually downloaded (from download_log)
    emg_downloaded = 0.0
    for result in download_log:
        for chunk_log in result.chunks_downloaded:
            if chunk_log["chunk_id"] in emg_meta:
                emg_downloaded += PRIORITY_WEIGHT[3] * chunk_log["bytes_taken"]

    emergency_score = emg_downloaded / emg_total if emg_total > 0 else 0.0

    # ── Delay penalties ───────────────────────────────────────────────
    # For each emergency chunk: when was it downloaded vs its deadline?
    # We reconstruct actual download time from the download_log.
    delay_penalties = _compute_delay_penalties(download_log, emg_meta)

    raw = (
            TASK3_BASE_WEIGHT * base_score
            + TASK3_EMERGENCY_WEIGHT * emergency_score
            - delay_penalties
    )
    return _clip(raw)


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _total_bytes(download_log: List[DownlinkResult]) -> int:
    """Sum of all bytes_downloaded across all DownlinkResults."""
    return sum(r.bytes_downloaded for r in download_log)


def _weighted_total(chunks: List[dict]) -> float:
    """
    Σ w(priority) × size_bytes for a flat list of chunk dicts.
    Used as the denominator in task2/3 scoring.
    """
    return sum(
        PRIORITY_WEIGHT.get(c["priority"], 1.0) * c["size_bytes"]
        for c in chunks
    )


def _weighted_downloaded(download_log: List[DownlinkResult]) -> float:
    """
    Σ w(priority) × bytes_taken across all chunks in the download log.
    Used as the numerator in task2/3 scoring.
    """
    total = 0.0
    for result in download_log:
        for chunk_log in result.chunks_downloaded:
            w = PRIORITY_WEIGHT.get(chunk_log["priority"], 1.0)
            total += w * chunk_log["bytes_taken"]
    return total


def _compute_delay_penalties(
        download_log: List[DownlinkResult],
        emg_meta: Dict[str, dict],
) -> float:
    """
    Compute total delay penalties for emergency chunks.

    For each emergency chunk:
        - If downloaded before deadline  → penalty = 0.0
        - If downloaded after deadline   → penalty = DELAY_PENALTY_MAX
                                           × min(delay_minutes / 60, 1.0)
        - If never downloaded at all     → penalty = DELAY_PENALTY_MAX (full)

    Returns total penalty to subtract from task3 score.
    """
    # Track which emergency chunks were downloaded and when
    # chunk_id → earliest tick it was (partially) downloaded
    first_download_tick: Dict[str, int] = {}
    for result in download_log:
        for chunk_log in result.chunks_downloaded:
            cid = chunk_log["chunk_id"]
            if cid in emg_meta:
                if cid not in first_download_tick:
                    first_download_tick[cid] = result.tick

    total_penalty = 0.0
    for chunk_id, meta in emg_meta.items():
        deadline_min: Optional[int] = meta.get("deadline_min")
        if deadline_min is None:
            # No deadline on this chunk — no penalty
            continue

        if chunk_id not in first_download_tick:
            # Never downloaded — full penalty
            total_penalty += DELAY_PENALTY_MAX
            continue

        # Downloaded — check whether it was on time
        download_min = first_download_tick[chunk_id] * 10
        if download_min > deadline_min:
            delay_min = download_min - deadline_min
            penalty = DELAY_PENALTY_MAX * min(delay_min / 60.0, 1.0)
            total_penalty += penalty

    return total_penalty


def _clip(score: float) -> float:
    """Clip to [0.0, 1.0]. Penalties cannot push score below zero."""
    return round(max(0.0, min(1.0, score)), 4)


# ─────────────────────────────────────────────────────────────
# Convenience: score breakdown for logging / README
# ─────────────────────────────────────────────────────────────

def grade_breakdown(
        task: str,
        download_log: List[DownlinkResult],
        all_chunks: List[dict],
        emergency_injections: List[dict],
) -> dict:
    """
    Same as grade() but returns a detailed breakdown dict.
    Used by inference.py to print human-readable results.

    Returns:
    {
        "task":             str,
        "final_score":      float,
        "bytes_downloaded": int,
        "bytes_available":  int,
        "throughput_pct":   float,
        "weighted_downloaded": float,
        "weighted_available":  float,
        "priority_efficiency": float,  # task2/3 only
        "emergency_score":     float,  # task3 only
        "delay_penalties":     float,  # task3 only
        "base_score":          float,  # task3 only
    }
    """
    final = grade(task, download_log, all_chunks, emergency_injections)

    breakdown = {
        "task": task,
        "final_score": final,
        "bytes_downloaded": _total_bytes(download_log),
        "bytes_available": sum(c["size_bytes"] for c in all_chunks),
        "throughput_pct": round(
            _total_bytes(download_log)
            / max(1, sum(c["size_bytes"] for c in all_chunks))
            * 100, 2
        ),
    }

    if task in ("task2", "task3"):
        w_dl = _weighted_downloaded(download_log)
        w_tot = _weighted_total(all_chunks)
        breakdown["weighted_downloaded"] = round(w_dl, 0)
        breakdown["weighted_available"] = round(w_tot, 0)
        breakdown["priority_efficiency"] = round(w_dl / max(1, w_tot), 4)

    if task == "task3":
        emg_meta = {
            inj["chunk"]["chunk_id"]: inj["chunk"]
            for inj in emergency_injections
        }
        emg_total = sum(
            PRIORITY_WEIGHT[3] * c["size_bytes"]
            for c in emg_meta.values()
        )
        emg_dl = sum(
            PRIORITY_WEIGHT[3] * cl["bytes_taken"]
            for r in download_log
            for cl in r.chunks_downloaded
            if cl["chunk_id"] in emg_meta
        )
        breakdown["base_score"] = round(_grade_task2(download_log, all_chunks), 4)
        breakdown["emergency_score"] = round(emg_dl / max(1, emg_total), 4)
        breakdown["delay_penalties"] = round(_compute_delay_penalties(download_log, emg_meta), 4)

    return breakdown
