"""Overlay drawing for the web stream and optional local preview."""

from __future__ import annotations

import cv2
import numpy as np

from .speed_estimator import TrackMeasurementState
from .tracker import TrackedObject


STATE_COLORS = {
    "IDLE": (180, 180, 180),
    "TRACKING": (0, 255, 0),
    "ARMED": (0, 255, 255),
    "MEASURED": (0, 128, 255),
    "DONE": (0, 0, 255),
    "INVALID": (90, 90, 90),
}


def draw_profile_guides(frame: np.ndarray, profile: dict, show_roi_overlay: bool = True) -> None:
    """Draw lane polygon and start/finish lines."""
    if show_roi_overlay:
        lane = np.array(profile["lane_polygon"], dtype=np.int32)
        cv2.polylines(frame, [lane], True, (255, 0, 0), 2)
    yellow = profile["yellow_line"]
    red = profile["red_line"]
    cv2.line(frame, tuple(map(int, yellow[0])), tuple(map(int, yellow[1])), (0, 255, 255), 2)
    cv2.line(frame, tuple(map(int, red[0])), tuple(map(int, red[1])), (0, 0, 255), 2)


def draw_track(frame: np.ndarray, track: TrackedObject, state: TrackMeasurementState | None, fps: float) -> None:
    """Draw one tracked object."""
    state_name = state.state if state is not None else "TRACKING"
    color = STATE_COLORS.get(state_name, (0, 255, 0))
    thickness = 1 if track.is_predicted else 2
    x1, y1, x2, y2 = track.bbox_xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    bottom_center = (int(track.bottom_center[0]), int(track.bottom_center[1]))
    cv2.circle(frame, bottom_center, 4, color, -1)

    lines = [
        f"ID:{track.track_id}",
        state.status_message if state is not None else "Bekleniyor",
        f"Conf:{track.confidence:.2f}",
    ]
    if state is not None and state.speed_kmh is not None and state.state in {"MEASURED", "DONE"}:
        lines.insert(1, f"Hiz:{state.speed_kmh:.1f} km/h")
    origin_y = max(18, y1 - 6)
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x1, origin_y - (index * 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_hud(frame: np.ndarray, active_tracks: int, fps: float, status_text: str, last_speed_kmh: float | None) -> None:
    """Draw global HUD."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Aktif Track: {active_tracks}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, status_text, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    speed_text = "Son Hiz: --"
    if last_speed_kmh is not None:
        speed_text = f"Son Hiz: {last_speed_kmh:.1f} km/h"
    cv2.putText(frame, speed_text, (frame.shape[1] - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
