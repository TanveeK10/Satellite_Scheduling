# src/envs/satellite_env/server/scheduler.py
"""
Scheduler — conflict detection, window assignment, and downlink execution.

Responsibilities:
    1. Accept / reject schedule() actions (conflict detection)
    2. Execute all active windows at the start of each tick
    3. Dequeue bytes from satellite priority queues (highest priority first)
    4. Return a DownlinkResult per executed window (feeds reward calculation)

Nothing in this file knows about HTTP, WebSockets, or rewards.
It is pure domain logic — fully testable without a running server.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.envs.satellite_env.models import DataChunkModel, ScheduleEntryModel


# ─────────────────────────────────────────────────────────────
# Return types (internal — not exposed over the wire)
# ─────────────────────────────────────────────────────────────

@dataclass
class DownlinkResult:
    """
    What actually happened when one scheduled window executed.
    One of these is produced per window that fires in a tick.
    The environment aggregates these to compute per-tick reward.
    """
    schedule_id:    str
    sat_id:   int
    station_id:     int
    tick:           int
    bytes_downloaded: int
    chunks_downloaded: List[dict]   # [{chunk_id, priority, bytes_taken}]
    availability:   float           # station availability this tick
    was_conflict:   bool = False


@dataclass
class ActionResult:
    """
    What happened when the agent submitted an action.
    Returned by every public method so environment.step() can
    build the info dict without knowing scheduler internals.
    """
    accepted:     bool
    schedule_id:  Optional[str] = None   # set on successful schedule()
    conflict:     bool = False
    error:        Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────

class Scheduler:
    """
    Manages the current schedule and executes downlinks each tick.

    State:
        _schedule: Dict[schedule_id → ScheduleEntryModel]
            All committed, not-yet-executed assignments.

        _queues: Dict[sat_id → List[DataChunkModel]]
            Live priority queues. Mutated as bytes are downloaded.
            Highest priority first. Within same priority, FIFO.

        _buffer_bytes: Dict[sat_id → int]
            Running total of bytes remaining per satellite.
            Derived from _queues but cached for O(1) observation access.

        _held_sats: Set[int]
            Satellites marked as held via hold(). Prevents auto-scheduling
            suggestions but does NOT cancel existing assignments.

        _download_log: List[DownlinkResult]
            Append-only log used by graders at episode end.
    """

    def __init__(
        self,
        initial_queues: Dict[int, List[DataChunkModel]],
        downlink_rates_bps: Dict[int, int],
    ) -> None:
        # Deep-copy queues so reset() can restore from originals
        self._original_queues = {
            sid: [c.model_copy() for c in chunks]
            for sid, chunks in initial_queues.items()
        }
        self._downlink_rates = downlink_rates_bps

        self._schedule:     Dict[str, ScheduleEntryModel] = {}
        self._held_sats:    set[int] = set()
        self._download_log: List[DownlinkResult] = []
        self._queues:       Dict[int, List[DataChunkModel]] = {}
        self._buffer_bytes: Dict[int, int] = {}

        self._init_queues()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore to initial state. Called by environment.reset()."""
        self._schedule = {}
        self._held_sats = set()
        self._download_log = []
        self._init_queues()

    def inject_chunks(self, sat_id: int, chunks: List[DataChunkModel]) -> None:
        """
        Insert emergency chunks into a satellite's queue mid-episode.
        Emergency chunks (priority=3) are inserted at the front, so they
        are downloaded before any existing lower-priority data.
        Called by environment.step() when an emergency injection fires.
        """
        emergency = [c for c in chunks if c.priority == 3]
        normal = [c for c in chunks if c.priority < 3]

        # Emergency at front, then existing queue, then any injected normal
        self._queues[sat_id] = (
            emergency
            + self._queues.get(sat_id, [])
            + normal
        )
        self._buffer_bytes[sat_id] = sum(
            c.size_bytes for c in self._queues[sat_id]
        )

    # ------------------------------------------------------------------
    # Agent actions
    # ------------------------------------------------------------------

    def schedule(
        self,
        sat_id:     int,
        station_id: int,
        window_id:  str,
        tick:       int,
    ) -> ActionResult:
        """
        Assign satellite sat_id to station_id for the window at tick.

        Rejection conditions (in order):
            1. tick is in the past (already executed)
            2. station already has a committed assignment at tick
            3. satellite already has a committed assignment at tick
            4. satellite buffer is empty (nothing to download)

        On rejection: return ActionResult(accepted=False, conflict=True)
        and leave the schedule unchanged.
        On acceptance: create a ScheduleEntryModel, add to _schedule,
        return ActionResult(accepted=True, schedule_id=...).
        """
        # Check for conflicts
        conflict_reason = self._find_conflict(sat_id, station_id, tick)
        if conflict_reason:
            return ActionResult(
                accepted=False,
                conflict=True,
                error=conflict_reason,
            )

        # Check buffer
        if self._buffer_bytes.get(sat_id, 0) == 0:
            return ActionResult(
                accepted=False,
                conflict=False,
                error=f"Satellite {sat_id} buffer is empty — nothing to download",
            )

        schedule_id = f"sch_s{sat_id}_g{station_id}_{tick:03d}_{uuid.uuid4().hex[:4]}"
        entry = ScheduleEntryModel(
            schedule_id=schedule_id,
            sat_id=sat_id,
            station_id=station_id,
            window_id=window_id,
            tick=tick,
            status="committed",
        )
        self._schedule[schedule_id] = entry
        return ActionResult(accepted=True, schedule_id=schedule_id)

    def preempt(self, schedule_id: str, current_tick: int) -> ActionResult:
        """
        Cancel a committed (not yet started) assignment.

        Rejection conditions:
            1. schedule_id does not exist
            2. The window's tick <= current_tick (already executing or done)

        No bytes are lost — only the opportunity cost of the freed window.
        The station slot becomes available for a new schedule() call.
        """
        entry = self._schedule.get(schedule_id)
        if entry is None:
            return ActionResult(
                accepted=False,
                error=f"schedule_id '{schedule_id}' not found",
            )
        if entry.tick <= current_tick:
            return ActionResult(
                accepted=False,
                conflict=True,
                error=f"Cannot preempt tick {entry.tick} — already at tick {current_tick}",
            )

        del self._schedule[schedule_id]
        return ActionResult(accepted=True)

    def hold(self, sat_id: int) -> ActionResult:
        """
        Mark satellite as held. Does not cancel existing assignments.
        The environment uses this to suppress auto-schedule suggestions
        in the observation (future feature — no-op for now but accepted).
        """
        self._held_sats.add(sat_id)
        return ActionResult(accepted=True)

    def unhold(self, sat_id: int) -> None:
        """Remove hold. Called if agent schedules a held satellite."""
        self._held_sats.discard(sat_id)

    # ------------------------------------------------------------------
    # Tick execution
    # ------------------------------------------------------------------

    def execute_tick(
        self,
        tick: int,
        availability: Dict[int, float],
        window_elevations: Dict[str, float] = None,
        window_durations: Dict[str, float] = None,
    ) -> List[DownlinkResult]:
        """
        Execute all windows scheduled for this tick.

        For each matching ScheduleEntryModel:
            1. Look up station availability (float multiplier)
            2. Look up max_bytes from the window definition (actual duration x elevation scalar)
            3. Dequeue bytes from the satellite's priority queue
            4. Record a DownlinkResult
            5. Mark the entry as done and remove from active schedule
        """
        results: List[DownlinkResult] = []

        # Find all entries scheduled for this exact tick
        firing = [e for e in self._schedule.values() if e.tick == tick]

        for entry in firing:
            avail = availability.get(entry.station_id, 1.0)
            rate_bps = self._downlink_rates.get(entry.sat_id, 100_000_000)
            
            # Elevation & Duration Fidelity
            import math
            elev_deg = window_elevations.get(entry.window_id, 90.0) if window_elevations else 90.0
            dur_s = window_durations.get(entry.window_id, 600.0) if window_durations else 600.0
            
            # Harsher Scalar floor (0.1 instead of 0.3)
            elev_scalar = 0.1 + 0.9 * math.sin(math.radians(max(0, elev_deg)))
            
            # max_bytes = rate × duration × availability × elevation efficiency
            window_bytes = int((rate_bps / 8) * dur_s * avail * elev_scalar)

            bytes_downloaded, chunks_log = self._dequeue(
                entry.sat_id, window_bytes
            )

            result = DownlinkResult(
                schedule_id=entry.schedule_id,
                sat_id=entry.sat_id,
                station_id=entry.station_id,
                tick=tick,
                bytes_downloaded=bytes_downloaded,
                chunks_downloaded=chunks_log,
                availability=avail,
            )
            results.append(result)
            self._download_log.append(result)

            # Mark done and remove from active schedule
            entry.status = "done"
            del self._schedule[entry.schedule_id]

        return results

    # ------------------------------------------------------------------
    # Read-only accessors (for observation building)
    # ------------------------------------------------------------------

    def get_schedule(self) -> List[ScheduleEntryModel]:
        """All committed future assignments. Returns a snapshot list."""
        return list(self._schedule.values())

    def get_buffer_bytes(self) -> Dict[str, int]:
        """sat_id (str key) → remaining bytes. JSON-safe."""
        return {str(k): v for k, v in self._buffer_bytes.items()}

    def get_queues(self) -> Dict[str, List[DataChunkModel]]:
        """sat_id (str key) → chunk list. JSON-safe."""
        return {str(k): list(v) for k, v in self._queues.items()}

    def get_download_log(self) -> List[DownlinkResult]:
        """Full append-only log. Used by graders at episode end."""
        return list(self._download_log)

    def get_rates_bps(self) -> Dict[str, int]:
        """sat_id (str key) → downlink rate. JSON-safe."""
        return {str(k): v for k, v in self._downlink_rates.items()}

    def is_held(self, sat_id: int) -> bool:
        return sat_id in self._held_sats

    def all_buffers_empty(self) -> bool:
        """True when every satellite has nothing left to download."""
        return all(b == 0 for b in self._buffer_bytes.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_queues(self) -> None:
        """Restore queues and buffer_bytes from original deep copy."""
        self._queues = {
            sid: [c.model_copy() for c in chunks]
            for sid, chunks in self._original_queues.items()
        }
        self._buffer_bytes = {
            sid: sum(c.size_bytes for c in chunks)
            for sid, chunks in self._queues.items()
        }

    def _find_conflict(
        self,
        sat_id:     int,
        station_id: int,
        tick:       int,
    ) -> Optional[str]:
        """
        Return a human-readable conflict reason or None if no conflict.

        Two conflict types:
            Station conflict: station already serving another satellite at tick.
            Satellite conflict: satellite already assigned to another station at tick.
        """
        for entry in self._schedule.values():
            if entry.tick != tick:
                continue
            if entry.station_id == station_id and entry.sat_id != sat_id:
                return (
                    f"Station {station_id} already assigned to "
                    f"satellite {entry.sat_id} at tick {tick}"
                )
            if entry.sat_id == sat_id and entry.station_id != station_id:
                return (
                    f"Satellite {sat_id} already assigned to "
                    f"station {entry.station_id} at tick {tick}"
                )
        return None

    def _dequeue(
        self,
        sat_id:      int,
        max_bytes:   int,
    ) -> Tuple[int, List[dict]]:
        """
        Download up to max_bytes from satellite sat_id's queue.
        Dequeues front-first (highest priority first — queue is pre-sorted).

        Partial chunk downloads are supported:
            If a chunk is larger than remaining capacity, download what fits
            and reduce the chunk's size_bytes in place.

        Returns:
            (total_bytes_downloaded, chunks_log)
            chunks_log: [{chunk_id, priority, bytes_taken}]
        """
        queue = self._queues.get(sat_id, [])
        remaining = max_bytes
        total = 0
        log: List[dict] = []

        while queue and remaining > 0:
            chunk = queue[0]
            take = min(chunk.size_bytes, remaining)

            log.append({
                "chunk_id":    chunk.chunk_id,
                "priority":    chunk.priority,
                "bytes_taken": take,
                "deadline_min": chunk.deadline_min,
            })

            total += take
            remaining -= take
            chunk.size_bytes -= take

            if chunk.size_bytes == 0:
                queue.pop(0)   # chunk fully downloaded — remove from queue

        # Update buffer total
        self._buffer_bytes[sat_id] = sum(c.size_bytes for c in queue)
        return total, log
