"""Line geometry and crossing detection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


Point = tuple[float, float]


def point_side(point: Point, line_start: Point, line_end: Point) -> float:
    """Return signed side of a point relative to a directed line."""
    return ((line_end[0] - line_start[0]) * (point[1] - line_start[1])) - (
        (line_end[1] - line_start[1]) * (point[0] - line_start[0])
    )


def segment_crossed(previous_point: Point, current_point: Point, line_start: Point, line_end: Point) -> bool:
    """Detect whether a moving point crossed a segment's infinite line between frames."""
    previous_side = point_side(previous_point, line_start, line_end)
    current_side = point_side(current_point, line_start, line_end)
    if previous_side == 0.0 or current_side == 0.0:
        return False
    return (previous_side < 0 <= current_side) or (previous_side > 0 >= current_side)


def point_in_polygon(point: Point, polygon: list[list[float]] | list[Point]) -> bool:
    """Ray-casting point in polygon test."""
    px, py = point
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        intersects = ((y1 > py) != (y2 > py)) and (
            px < ((x2 - x1) * (py - y1) / max(y2 - y1, 1e-6)) + x1
        )
        if intersects:
            inside = not inside
    return inside


@dataclass
class TrackCrossingState:
    """Per-track line crossing memory."""

    last_point: Optional[Point] = None
    yellow_crossed: bool = False
    red_crossed: bool = False
    inside_lane: bool = False


@dataclass
class CrossingEvent:
    """Crossing result for one track in one frame."""

    crossed_yellow: bool = False
    crossed_red: bool = False
    inside_lane: bool = False


@dataclass
class LineCrossingDetector:
    """Track-aware crossing detector using bottom-center reference points."""

    yellow_line: list[list[float]]
    red_line: list[list[float]]
    lane_polygon: list[list[float]]
    track_states: dict[int, TrackCrossingState] = field(default_factory=dict)

    def update(self, track_id: int, point: Point) -> CrossingEvent:
        """Update crossing state for a track."""
        state = self.track_states.setdefault(track_id, TrackCrossingState())
        event = CrossingEvent()
        event.inside_lane = point_in_polygon(point, self.lane_polygon)
        state.inside_lane = event.inside_lane
        if state.last_point is not None:
            if not state.yellow_crossed and segment_crossed(state.last_point, point, tuple(self.yellow_line[0]), tuple(self.yellow_line[1])):
                state.yellow_crossed = True
                event.crossed_yellow = True
            if state.yellow_crossed and not state.red_crossed and segment_crossed(state.last_point, point, tuple(self.red_line[0]), tuple(self.red_line[1])):
                state.red_crossed = True
                event.crossed_red = True
        state.last_point = point
        return event

    def cleanup(self, active_ids: set[int]) -> None:
        """Drop crossing state for inactive tracks."""
        stale_ids = [track_id for track_id in self.track_states if track_id not in active_ids]
        for track_id in stale_ids:
            del self.track_states[track_id]

