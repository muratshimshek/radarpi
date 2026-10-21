"""Tracking layer with Kalman prediction and association control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .detector import Detection


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter_area / max(area_a + area_b - inter_area, 1.0)


@dataclass
class TrackedObject:
    """Tracker output object."""

    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    timestamp: float
    age_frames: int = 1
    hit_streak: int = 1
    lost_frames: int = 0
    is_predicted: bool = False
    state: str = "TRACKING"
    source_backend: str = ""
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0,
        )

    @property
    def width(self) -> int:
        return max(1, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> int:
        return max(1, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        return float(self.width * self.height)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0, float(self.bbox_xyxy[3]))


class InternalTrack:
    """Internal track state with Kalman filter."""

    def __init__(self, track_id: int, detection: Detection, smoothing_alpha: float) -> None:
        self.track_id = track_id
        self.object = TrackedObject(
            track_id=track_id,
            bbox_xyxy=detection.bbox_xyxy,
            confidence=detection.confidence,
            timestamp=detection.timestamp,
            source_backend=detection.source_backend,
            history=[detection.bottom_center],
        )
        self.kalman = cv2.KalmanFilter(6, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
            np.float32,
        )
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.4
        cx, cy = self.object.center
        self.kalman.statePost = np.array([[cx], [cy], [0], [0], [self.object.width], [self.object.height]], np.float32)
        self.smoothing_alpha = smoothing_alpha

    def predict(self) -> tuple[float, float]:
        """Predict center for the next step."""
        predicted = self.kalman.predict()
        return float(predicted[0][0]), float(predicted[1][0])

    def update(self, detection: Detection, max_scale_ratio: float, min_area_ratio: float) -> None:
        """Update track from matched detection with smoothing and scale clamps."""
        cx, cy = detection.center
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(measurement)

        prev_box = self.object.bbox_xyxy
        prev_width = self.object.width
        prev_height = self.object.height
        det_w = detection.width
        det_h = detection.height

        if det_w < int(prev_width * min_area_ratio):
            det_w = max(det_w, int(prev_width * min_area_ratio))
        if det_h < int(prev_height * min_area_ratio):
            det_h = max(det_h, int(prev_height * min_area_ratio))
        det_w = min(det_w, int(prev_width * max_scale_ratio))
        det_h = min(det_h, int(prev_height * max_scale_ratio))

        smoothed_cx = ((1.0 - self.smoothing_alpha) * self.object.center[0]) + (self.smoothing_alpha * cx)
        smoothed_cy = ((1.0 - self.smoothing_alpha) * self.object.center[1]) + (self.smoothing_alpha * cy)
        smoothed_w = int(round(((1.0 - self.smoothing_alpha) * prev_width) + (self.smoothing_alpha * det_w)))
        smoothed_h = int(round(((1.0 - self.smoothing_alpha) * prev_height) + (self.smoothing_alpha * det_h)))
        x1 = int(round(smoothed_cx - (smoothed_w / 2.0)))
        y1 = int(round(smoothed_cy - (smoothed_h / 2.0)))
        new_box = (x1, y1, x1 + smoothed_w, y1 + smoothed_h)

        self.object.bbox_xyxy = new_box
        self.object.confidence = detection.confidence
        self.object.timestamp = detection.timestamp
        self.object.age_frames += 1
        self.object.hit_streak += 1
        self.object.lost_frames = 0
        self.object.is_predicted = False
        self.object.source_backend = detection.source_backend
        self.object.history.append(self.object.bottom_center)
        if len(self.object.history) > 20:
            self.object.history.pop(0)

    def mark_predicted(self, timestamp: float, damping: float) -> None:
        """Advance track with predicted center when no detection is matched."""
        predicted_cx, predicted_cy = self.predict()
        current_w = self.object.width
        current_h = self.object.height
        x1 = int(round(predicted_cx - (current_w / 2.0)))
        y1 = int(round(predicted_cy - (current_h / 2.0)))
        self.object.bbox_xyxy = (x1, y1, x1 + current_w, y1 + current_h)
        self.object.timestamp = timestamp
        self.object.age_frames += 1
        self.object.hit_streak = 0
        self.object.lost_frames += 1
        self.object.is_predicted = True
        self.object.confidence *= damping
        self.object.history.append(self.object.bottom_center)
        if len(self.object.history) > 20:
            self.object.history.pop(0)


class MultiObjectTracker:
    """Simple Pi-friendly tracker with controlled association logic."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.next_track_id = 1
        self.tracks: dict[int, InternalTrack] = {}
        self.fps = max(1.0, float(config["camera"]["fps"]))
        self.lost_tolerance_frames = max(1, int(round(config["tracking"]["lost_tolerance_seconds"] * self.fps)))
        self.max_center_jump = float(config["tracking"]["max_center_jump_px"])
        self.max_lateral_jump = float(config["tracking"].get("max_lateral_jump_px", 42))
        self.max_reverse_motion = float(config["tracking"].get("max_reverse_motion_px", 18))
        self.max_scale_ratio = float(config["tracking"]["max_scale_ratio"])
        self.min_area_ratio = float(config["tracking"]["min_area_ratio_of_prev"])
        self.smoothing_alpha = float(config["tracking"]["smoothing_alpha"])
        self.prediction_damping = float(config["tracking"]["prediction_damping"])
        self.stable_hits = int(config["tracking"]["stable_hits"])
        self.active_direction = "down"

    def set_active_profile(self, profile: dict) -> None:
        """Update direction hints from active profile."""
        self.active_direction = str(profile.get("direction", "down")).lower()

    def _match_score(self, track: InternalTrack, detection: Detection) -> float:
        """Association score between a track and a detection."""
        pred_cx, pred_cy = track.predict()
        det_cx, det_cy = detection.center
        center_distance = float(((pred_cx - det_cx) ** 2 + (pred_cy - det_cy) ** 2) ** 0.5)
        if center_distance > (self.max_center_jump * (2.0 if track.object.is_predicted else 1.0)):
            return -1e9
        dx = det_cx - track.object.center[0]
        dy = det_cy - track.object.center[1]
        lateral_limit = self.max_lateral_jump * (1.35 if track.object.is_predicted else 1.0)
        if track.object.hit_streak >= self.stable_hits:
            lateral_limit *= 0.75
        if abs(dx) > lateral_limit:
            return -1e9
        if self.active_direction == "down" and dy < -self.max_reverse_motion:
            return -1e9
        if self.active_direction == "up" and dy > self.max_reverse_motion:
            return -1e9
        iou = bbox_iou(track.object.bbox_xyxy, detection.bbox_xyxy)
        scale_ratio = max(detection.area, 1.0) / max(track.object.area, 1.0)
        scale_ratio = scale_ratio if scale_ratio >= 1.0 else 1.0 / max(scale_ratio, 1e-6)
        if scale_ratio > max(2.2, self.max_scale_ratio * 1.3):
            return -1e9
        return (
            (iou * 100.0)
            - center_distance
            - (abs(dx) * 0.9)
            - ((scale_ratio - 1.0) * 20.0)
            + (max(dy, -5.0) * 0.25)
            + (detection.confidence * 10.0)
        )

    def update(self, detections: list[Detection], timestamp: float) -> list[TrackedObject]:
        """Associate detections to tracks and return active tracked objects."""
        unmatched_detections = set(range(len(detections)))
        assignments: dict[int, int] = {}

        for track_id, track in list(self.tracks.items()):
            best_detection_index: Optional[int] = None
            best_score = -1e9
            for det_index in unmatched_detections:
                score = self._match_score(track, detections[det_index])
                if score > best_score:
                    best_score = score
                    best_detection_index = det_index
            if best_detection_index is not None and best_score > -100.0:
                assignments[track_id] = best_detection_index
                unmatched_detections.discard(best_detection_index)

        for track_id, det_index in assignments.items():
            self.tracks[track_id].update(detections[det_index], self.max_scale_ratio, self.min_area_ratio)

        for det_index in unmatched_detections:
            detection = detections[det_index]
            self.tracks[self.next_track_id] = InternalTrack(self.next_track_id, detection, self.smoothing_alpha)
            self.next_track_id += 1

        stale_ids: list[int] = []
        for track_id, track in self.tracks.items():
            if track_id in assignments:
                continue
            if track.object.track_id >= self.next_track_id:
                continue
            if track.object.timestamp != timestamp:
                track.mark_predicted(timestamp, self.prediction_damping)
            if track.object.lost_frames > self.lost_tolerance_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.tracks[track_id]

        objects = [track.object for track in self.tracks.values()]
        objects.sort(key=lambda item: item.track_id)
        return objects
