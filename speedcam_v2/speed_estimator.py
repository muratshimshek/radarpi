"""Track-based state machine and speed estimation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .homography import HomographyManager
from .line_crossing import CrossingEvent
from .tracker import TrackedObject


@dataclass
class MeasurementSnapshot:
    """Finalized measurement result."""

    track_id: int
    speed_kmh: float | None
    status: str
    reason: str
    timestamp: float
    capture_pending: bool = False


@dataclass
class TrackMeasurementState:
    """State for one tracked vehicle."""

    track_id: int
    state: str = "IDLE"
    start_time: float | None = None
    start_track_id: int | None = None
    start_point: tuple[float, float] | None = None
    end_time: float | None = None
    speed_kmh: float | None = None
    status_message: str = "Bekleniyor"
    invalid_reason: str = ""
    last_update: float = 0.0
    done_until: float = 0.0
    invalid_until: float = 0.0
    start_committed: bool = False
    measurement_locked: bool = False


class SpeedEstimator:
    """Manage measurement state across active tracks."""

    def __init__(self, config: dict, homography: HomographyManager | None = None) -> None:
        self.config = config
        self.homography = homography
        self.states: dict[int, TrackMeasurementState] = {}

    def _distance_meters(self, start_point: tuple[float, float], end_point: tuple[float, float]) -> float:
        """Return calibrated distance between start and finish references."""
        if self.homography is not None and self.homography.enabled and self.homography.matrix is not None:
            measured = self.homography.distance_meters(start_point, end_point)
            if measured > 0.1:
                return measured
        return float(self.config["measurement"]["real_distance_meters"])

    def get_or_create(self, track_id: int) -> TrackMeasurementState:
        """Return existing or create new state."""
        state = self.states.get(track_id)
        if state is None:
            state = TrackMeasurementState(track_id=track_id)
            self.states[track_id] = state
        return state

    def update(self, tracked_object: TrackedObject, crossing: CrossingEvent, stable: bool) -> MeasurementSnapshot | None:
        """Advance the track state machine."""
        state = self.get_or_create(tracked_object.track_id)
        state.last_update = tracked_object.timestamp

        if state.state == "IDLE":
            state.state = "TRACKING"
            state.status_message = "Bekleniyor"

        if state.state == "TRACKING" and crossing.crossed_yellow:
            if not stable or tracked_object.is_predicted:
                state.state = "INVALID"
                state.invalid_reason = "Measurement Invalid"
                state.status_message = "Measurement Invalid"
                state.invalid_until = tracked_object.timestamp + float(self.config["measurement"]["done_display_seconds"])
                return MeasurementSnapshot(tracked_object.track_id, None, state.state, state.invalid_reason, tracked_object.timestamp)
            state.state = "ARMED"
            state.start_time = time.monotonic()
            state.start_track_id = tracked_object.track_id
            state.start_point = tracked_object.bottom_center
            state.start_committed = True
            state.status_message = "Olcum Basladi"
            return None

        if state.state == "ARMED":
            if tracked_object.is_predicted and tracked_object.lost_frames > int(round(self.config["tracking"]["lost_tolerance_seconds"] * self.config["camera"]["fps"])):
                state.state = "INVALID"
                state.invalid_reason = "Tracking Lost"
                state.status_message = "Tracking Lost"
                state.invalid_until = tracked_object.timestamp + float(self.config["measurement"]["done_display_seconds"])
                return MeasurementSnapshot(tracked_object.track_id, None, state.state, state.invalid_reason, tracked_object.timestamp)
            if crossing.crossed_red and state.start_time is not None and state.start_track_id == tracked_object.track_id:
                end_time = time.monotonic()
                elapsed = end_time - state.start_time
                if elapsed < float(self.config["measurement"]["min_elapsed_seconds"]) or elapsed > float(self.config["measurement"]["max_elapsed_seconds"]):
                    state.state = "INVALID"
                    state.invalid_reason = "Measurement Invalid"
                    state.status_message = "Measurement Invalid"
                    state.invalid_until = tracked_object.timestamp + float(self.config["measurement"]["done_display_seconds"])
                    return MeasurementSnapshot(tracked_object.track_id, None, state.state, state.invalid_reason, tracked_object.timestamp)
                distance = self._distance_meters(state.start_point or tracked_object.bottom_center, tracked_object.bottom_center)
                speed_ms = distance / max(elapsed, 1e-6)
                speed_kmh = speed_ms * 3.6
                state.state = "MEASURED"
                state.end_time = end_time
                state.speed_kmh = speed_kmh
                state.measurement_locked = True
                state.done_until = tracked_object.timestamp + float(self.config["measurement"]["done_display_seconds"])
                state.status_message = "Ihlal Kaydedildi" if speed_kmh > float(self.config["measurement"]["speed_limit_kmh"]) else "Olcum Tamamlandi"
                return MeasurementSnapshot(tracked_object.track_id, speed_kmh, state.state, state.status_message, tracked_object.timestamp, capture_pending=True)

        if state.state == "MEASURED":
            state.state = "DONE"
            return None

        if state.state == "DONE" and tracked_object.timestamp > state.done_until:
            state.status_message = "Bekleniyor"

        return None

    def cleanup(self, active_track_ids: set[int], now_timestamp: float) -> None:
        """Remove inactive or expired state."""
        stale_ids: list[int] = []
        for track_id, state in self.states.items():
            if track_id in active_track_ids:
                continue
            if state.state == "DONE" and now_timestamp <= state.done_until:
                continue
            if state.state == "INVALID" and now_timestamp <= state.invalid_until:
                continue
            stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.states[track_id]
