#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raspberry Pi 3 icin hafif hiz tespit sistemi.

Ozellikler:
- OpenCV ile hareket algilama
- SOL / SAG presetleri ile yol tarafina gore hizli yerlesim degistirme
- Iki sanal cizgi arasinda sure olcumu
- Aracin on kenarina gore hiz hesabi
- Hiz limiti asiminda fotograf kaydi
- HDMI / dokunmatik ekranda canli goruntu, FPS ve overlay bilgileri
"""

from __future__ import annotations

import logging
import math
import sys
import time
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

try:
    from flask import Flask, Response, redirect, render_template_string, request, url_for
except Exception:
    Flask = None
    Response = None
    redirect = None
    render_template_string = None
    request = None
    url_for = None

cv2.setUseOptimized(True)


# ============================================================
# AYARLAR
# ============================================================

CAMERA_INDEX = 0
LEGACY_FRAME_WIDTH = 424
LEGACY_FRAME_HEIGHT = 240
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
TARGET_FPS = 30
WINDOW_NAME = "RadarPi Hiz Tespit Sistemi"
ENABLE_DESKTOP_GUI = False
ENABLE_WEB_STREAM = True
WEB_STREAM_PORT = 5001
WEB_STREAM_FPS_LIMIT = 8.0
WEB_JPEG_QUALITY = 72
FPS_LOG_INTERVAL_SECONDS = 1.0
DETECTION_INTERVAL_IDLE = 1
DETECTION_INTERVAL_ACTIVE = 2
OVERLAY_RENDER_INTERVAL = 4
ENABLE_DEBUG_CONTOUR_DRAW = False

DISTANCE_M = 10.0
SPEED_LIMIT_KMH = 30.0
COOLDOWN_SECONDS = 3.0
CALIBRATION_CARD_LENGTH_M = 1.15
CALIBRATION_CARD_WIDTH_M = 0.03
CALIBRATION_ROUND_STEP_M = 5.0

# Cizgi yukseklikleri.
START_LINE_Y = int(round(90 * (FRAME_HEIGHT / float(LEGACY_FRAME_HEIGHT))))
FINISH_LINE_Y = int(round(186 * (FRAME_HEIGHT / float(LEGACY_FRAME_HEIGHT))))

# Genel ROI sekli icin perspektif genislikler.
ROI_TOP_WIDTH = int(round(118 * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH))))
ROI_BOTTOM_WIDTH = int(round(174 * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH))))
FOCUS_BAND_HEIGHT = int(round(22 * (FRAME_HEIGHT / float(LEGACY_FRAME_HEIGHT))))

# Profil merkezleri. SOL profili solda, SAG profili sagda calisir.
PROFILE_CENTER_SOL = int(round(162 * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH))))
PROFILE_CENTER_SAG = int(round(262 * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH))))
PROFILE_LANE_HALF_WIDTH = int(round(74 * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH))))

# Gurultu ve contour ayarlari. Onceki surume gore biraz daha hassas.
MIN_CONTOUR_AREA = 360
MAX_CONTOUR_AREA = 52000
MIN_BBOX_WIDTH = 16
MIN_BBOX_HEIGHT = 12
MIN_ASPECT_RATIO = 0.55
MAX_ASPECT_RATIO = 5.20
GAUSSIAN_BLUR_KERNEL = (1, 1)
MORPH_KERNEL_SIZE = 3
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 1
MORPH_DILATE_ITERATIONS = 0
THRESHOLD_VALUE = 140
BG_HISTORY = 220
BG_VAR_THRESHOLD = 20
BG_DETECT_SHADOWS = False
WARMUP_FRAMES = 30
WARMUP_FRAMES_RUNTIME = WARMUP_FRAMES

# Olcum guvenligi.
MIN_VALID_TIME = 0.20
MAX_VALID_TIME = 12.00
TRACK_LOST_TIMEOUT = 2.10
MAX_CENTER_JUMP = 94
MAX_FRONT_JUMP = 78
MIN_TRACK_FRAMES = 3
TRACK_CONFIRM_FRAMES = 3
TRACK_MAX_MISSED_FRAMES = 16
TRACK_CONFIRM_HOLD_MISSED_FRAMES = 7
TRACK_MATCH_CENTER_DISTANCE = 132
TRACK_MATCH_FRONT_DISTANCE = 122
TRACK_SMOOTHING = 0.65
START_RECOVERY_NEAR_PX = 6
START_TRACK_GRACE_SECONDS = 1.00
MAX_RECOVERY_GAP_SECONDS = 0.30
FINISH_RECOVERY_MARGIN_PX = 12
FINISH_RECOVERY_NEAR_PX = 22

DEFAULT_PROCESS_WIDTH = PROCESS_WIDTH
DEFAULT_PROCESS_HEIGHT = PROCESS_HEIGHT
DEFAULT_MIN_CONTOUR_AREA = MIN_CONTOUR_AREA
DEFAULT_MIN_BBOX_WIDTH = MIN_BBOX_WIDTH
DEFAULT_MIN_BBOX_HEIGHT = MIN_BBOX_HEIGHT
DEFAULT_TRACK_CONFIRM_FRAMES = TRACK_CONFIRM_FRAMES
DEFAULT_MAX_CENTER_JUMP = MAX_CENTER_JUMP
DEFAULT_MAX_FRONT_JUMP = MAX_FRONT_JUMP
DEFAULT_DETECTION_INTERVAL_ACTIVE = DETECTION_INTERVAL_ACTIVE
DEFAULT_OVERLAY_RENDER_INTERVAL = OVERLAY_RENDER_INTERVAL

DETECTION_MODE_PRESETS: Dict[str, Dict[str, int]] = {
    "KARARLI": {
        "MIN_CONTOUR_AREA": 460,
        "MIN_BBOX_WIDTH": 17,
        "MIN_BBOX_HEIGHT": 13,
        "TRACK_CONFIRM_FRAMES": 3,
        "MAX_CENTER_JUMP": 92,
        "MAX_FRONT_JUMP": 76,
        "WARMUP_FRAMES": 24,
    },
    "DENGELI": {
        "MIN_CONTOUR_AREA": 320,
        "MIN_BBOX_WIDTH": 14,
        "MIN_BBOX_HEIGHT": 10,
        "TRACK_CONFIRM_FRAMES": 2,
        "MAX_CENTER_JUMP": 104,
        "MAX_FRONT_JUMP": 86,
        "WARMUP_FRAMES": 14,
    },
    "HASSAS": {
        "MIN_CONTOUR_AREA": 220,
        "MIN_BBOX_WIDTH": 12,
        "MIN_BBOX_HEIGHT": 8,
        "TRACK_CONFIRM_FRAMES": 2,
        "MAX_CENTER_JUMP": 116,
        "MAX_FRONT_JUMP": 96,
        "WARMUP_FRAMES": 8,
    },
}

PERFORMANCE_MODE_PRESETS: Dict[str, Dict[str, int]] = {
    "NORMAL": {
        "PROCESS_WIDTH": DEFAULT_PROCESS_WIDTH,
        "PROCESS_HEIGHT": DEFAULT_PROCESS_HEIGHT,
        "DETECTION_INTERVAL_ACTIVE": DEFAULT_DETECTION_INTERVAL_ACTIVE,
        "OVERLAY_RENDER_INTERVAL": DEFAULT_OVERLAY_RENDER_INTERVAL,
    },
    "HIZLI": {
        "PROCESS_WIDTH": 320,
        "PROCESS_HEIGHT": 180,
        "DETECTION_INTERVAL_ACTIVE": 3,
        "OVERLAY_RENDER_INTERVAL": 5,
    },
}

# Arayuz ayarlari.
TOP_BAR_HEIGHT = 26
BUTTON_Y1 = 3
BUTTON_Y2 = 20
BUTTON_WIDTH = 54
BUTTON_GAP = 8

# Klasorler.
BASE_DIR = Path("/home/radarpi/radar")
VIOLATION_DIR = BASE_DIR / "ihlaller"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "radar.log"
DISTANCE_FILE = BASE_DIR / "distance_m.txt"
LAYOUT_CONFIG_FILE = BASE_DIR / "layout_config.json"

# Masaustu pencere debug amacli acilabilir; varsayilan headless + web stream.
SHOW_WINDOW = ENABLE_DESKTOP_GUI
DEFAULT_UPSTREAM_OFFSET_PX = 60
FRAME_BUFFER_SIZE = 8
HANDLE_RADIUS = 6
SELECTED_HANDLE_RADIUS = 8
LINE_HIT_TOLERANCE = 14


@dataclass(frozen=True)
class LayoutProfile:
    """Cihazin yolun hangi tarafinda olduguna gore overlay yerlesimi."""

    name: str
    lane_center_x: int
    lane_half_width: int
    start_p1: Tuple[int, int]
    start_p2: Tuple[int, int]
    finish_p1: Tuple[int, int]
    finish_p2: Tuple[int, int]
    focus_p1: Tuple[int, int]
    focus_p2: Tuple[int, int]


@dataclass
class BufferFrame:
    frame: np.ndarray
    score: float
    timestamp: float


@dataclass
class WebStreamState:
    latest_jpeg: Optional[bytes] = None
    latest_fps: float = 0.0
    latest_status: str = "Baslatiliyor"
    latest_speed_kmh: float = 0.0
    last_publish_at: float = 0.0
    last_enqueue_at: float = 0.0
    pending_job: Optional[Dict[str, Any]] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class Detection:
    """Tek bir hareketli nesnenin secilen ozeti."""

    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    front_point: Tuple[int, int]
    front_y: int
    area: float
    width: int
    height: int
    aspect_ratio: float


@dataclass
class Track:
    """Kararlarin ayni araca baglanmasi icin hafif tek-arac track yapisi."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    front_point: Tuple[int, int]
    front_y: int
    area: float
    width: int
    height: int
    aspect_ratio: float
    last_seen_at: float
    previous_seen_at: Optional[float] = None
    seen_frames: int = 1
    missed_frames: int = 0
    status: str = "provisional"
    measurement_state: str = "armed"
    start_time: Optional[float] = None
    was_above_start_line: bool = False
    direction_down: Optional[bool] = None
    last_speed_kmh: float = 0.0
    velocity_center: Tuple[int, int] = (0, 0)
    velocity_front: Tuple[int, int] = (0, 0)
    previous_front_point: Optional[Tuple[int, int]] = None


class RadarState:
    """Tek hedefli basit durum makinesi."""

    def __init__(self) -> None:
        self.mode = "WAITING_FOR_DETECTION"
        self.start_time: Optional[float] = None
        self.last_speed_kmh: float = 0.0
        self.last_violation_at: float = 0.0
        self.last_seen_at: float = 0.0
        self.last_front_y: Optional[int] = None
        self.last_center: Optional[Tuple[int, int]] = None
        self.direction_down: Optional[bool] = None
        self.track_frames: int = 0
        self.was_above_start_line: bool = False
        self.finish_time: Optional[float] = None
        self.start_hold_until: float = 0.0

    def reset_tracking(self) -> None:
        self.mode = "WAITING_FOR_DETECTION"
        self.start_time = None
        self.finish_time = None
        self.start_hold_until = 0.0
        self.last_seen_at = 0.0
        self.last_front_y = None
        self.last_center = None
        self.direction_down = None
        self.track_frames = 0
        self.was_above_start_line = False

    def enter_cooldown(self, speed_kmh: float, now: float) -> None:
        self.mode = "cooldown"
        self.last_speed_kmh = speed_kmh
        self.last_violation_at = now
        self.start_time = None
        self.last_seen_at = now
        self.last_front_y = None
        self.last_center = None
        self.direction_down = None
        self.track_frames = 0
        self.was_above_start_line = False
        self.finish_time = None

    def update_cooldown(self, now: float) -> None:
        if self.mode == "cooldown" and (now - self.last_violation_at) >= COOLDOWN_SECONDS:
            self.mode = "idle"


class UIContext:
    """Dokunmatik butonlar ve aktif profil secimi icin hafif baglam."""

    def __init__(self, initial_profile: str) -> None:
        self.active_profile_name = initial_profile
        self.pending_profile_change = False
        self.pending_save_config = False
        self.is_fullscreen = True
        self.pending_window_mode = False
        self.pending_fullscreen_mode = False
        self.exit_requested = False
        self.menu_open = False
        self.menu_status = "idle"
        self.active_tool = "none"
        self.pending_calibration = False
        self.pending_distance_reset = False
        self.pending_calibration_source: Optional[str] = None
        self.calibration_mode = "idle"
        self.calibration_first_point: Optional[Tuple[int, int]] = None
        self.calibration_second_point: Optional[Tuple[int, int]] = None
        self.calibration_candidate_bbox: Optional[Tuple[int, int, int, int]] = None
        self.calibration_candidate_length_px: Optional[float] = None
        self.calibration_feedback_point: Optional[Tuple[int, int]] = None
        self.calibration_feedback_until: float = 0.0
        self.calibration_status = "idle"
        self.calibration_label = "Kalibre Et"
        self.manual_distance_mode = False
        self.manual_distance_text = ""
        self.pending_manual_distance_save = False
        self.manual_distance_status = "idle"
        self.pending_external_distance: Optional[float] = None
        self.manual_speed_mode = False
        self.manual_speed_text = ""
        self.pending_manual_speed_save = False
        self.manual_speed_status = "idle"
        self.pending_external_speed_limit: Optional[float] = None
        self.dragging_handle: Optional[str] = None
        self.dragging_line: Optional[str] = None
        self.drag_last_point: Optional[Tuple[int, int]] = None
        self.calibration_point_drag_radius = 1.0
        self.show_roi_overlay = False
        self.detection_mode = "DENGELI"
        self.performance_mode = "NORMAL"
        self.pending_runtime_mode_apply = False

    def select_profile(self, profile_name: str) -> None:
        if self.active_profile_name != profile_name:
            self.active_profile_name = profile_name
            self.pending_profile_change = True

    def request_window_mode(self) -> None:
        self.pending_window_mode = True

    def request_fullscreen_mode(self) -> None:
        self.pending_fullscreen_mode = True

    def request_exit(self) -> None:
        self.exit_requested = True

    def open_menu(self) -> None:
        self.menu_open = True
        self.menu_status = "pending"
        self.active_tool = "none"
        self.cancel_manual_distance()
        self.cancel_manual_speed()

    def close_menu(self) -> None:
        self.menu_open = False
        self.active_tool = "none"
        self.dragging_handle = None
        self.dragging_line = None
        self.drag_last_point = None

    def request_save(self) -> None:
        self.pending_save_config = True

    def request_start_line_edit(self) -> None:
        self.menu_open = True
        self.active_tool = "edit_start"
        self.dragging_handle = None
        self.dragging_line = None
        self.drag_last_point = None
        self.calibration_mode = "idle"
        self.pending_calibration = False
        self.pending_calibration_source = None
        self.manual_distance_mode = False

    def request_finish_line_edit(self) -> None:
        self.menu_open = True
        self.active_tool = "edit_finish"
        self.dragging_handle = None
        self.dragging_line = None
        self.drag_last_point = None
        self.calibration_mode = "idle"
        self.pending_calibration = False
        self.pending_calibration_source = None
        self.manual_distance_mode = False

    def request_reset_active_mode(self) -> None:
        self.menu_open = True
        self.active_tool = "reset_active"

    def request_detection_mode_menu(self) -> None:
        self.menu_open = True
        self.active_tool = "detection_mode"

    def request_performance_mode_menu(self) -> None:
        self.menu_open = True
        self.active_tool = "performance_mode"

    def set_detection_mode(self, mode_name: str) -> None:
        self.detection_mode = mode_name.upper()
        self.pending_runtime_mode_apply = True
        self.pending_save_config = True
        self.menu_open = True
        self.active_tool = "none"

    def set_performance_mode(self, mode_name: str) -> None:
        self.performance_mode = mode_name.upper()
        self.pending_runtime_mode_apply = True
        self.pending_save_config = True
        self.menu_open = True
        self.active_tool = "none"

    def request_calibration(self) -> None:
        self.menu_open = False
        self.active_tool = "calibration"
        self.dragging_handle = None
        self.dragging_line = None
        self.drag_last_point = None
        self.manual_distance_mode = False
        self.pending_calibration = False
        self.pending_calibration_source = None
        self.pending_distance_reset = True
        self.calibration_mode = "auto_wait_tap"
        self.calibration_first_point = None
        self.calibration_second_point = None
        self.calibration_candidate_bbox = None
        self.calibration_candidate_length_px = None
        self.calibration_feedback_point = None
        self.calibration_feedback_until = 0.0
        self.calibration_status = "pending"
        self.calibration_label = "Nesne Sec"

    def request_calibration_retry(self) -> None:
        self.menu_open = False
        self.active_tool = "calibration"
        self.dragging_handle = None
        self.dragging_line = None
        self.drag_last_point = None
        self.manual_distance_mode = False
        self.pending_calibration = False
        self.pending_calibration_source = None
        self.calibration_mode = "auto_wait_tap"
        self.calibration_first_point = None
        self.calibration_second_point = None
        self.calibration_candidate_bbox = None
        self.calibration_candidate_length_px = None
        self.calibration_feedback_point = None
        self.calibration_feedback_until = 0.0
        self.calibration_status = "pending"
        self.calibration_label = "Nesne Sec"

    def request_manual_distance(self) -> None:
        self.menu_open = False
        self.active_tool = "manual_distance"
        self.manual_distance_mode = True
        self.manual_distance_text = ""
        self.pending_manual_distance_save = False
        self.manual_distance_status = "idle"
        self.manual_speed_mode = False
        self.pending_manual_speed_save = False

    def cancel_manual_distance(self) -> None:
        self.manual_distance_mode = False
        self.pending_manual_distance_save = False
        self.manual_distance_status = "idle"
        self.menu_open = True
        self.active_tool = "none"

    def request_speed_limit(self) -> None:
        self.menu_open = False
        self.active_tool = "manual_speed_limit"
        self.manual_speed_mode = True
        self.manual_speed_text = ""
        self.pending_manual_speed_save = False
        self.manual_speed_status = "idle"
        self.manual_distance_mode = False
        self.pending_manual_distance_save = False

    def cancel_manual_speed(self) -> None:
        self.manual_speed_mode = False
        self.pending_manual_speed_save = False
        self.manual_speed_status = "idle"
        self.menu_open = True
        self.active_tool = "none"

    def append_manual_distance_char(self, char: str) -> None:
        if char == ".":
            if "." in self.manual_distance_text:
                return
            self.manual_distance_text = "0." if self.manual_distance_text == "" else self.manual_distance_text + "."
            return
        if char.isdigit():
            self.manual_distance_text += char

    def backspace_manual_distance(self) -> None:
        self.manual_distance_text = self.manual_distance_text[:-1]

    def append_manual_speed_char(self, char: str) -> None:
        if char == ".":
            if "." in self.manual_speed_text:
                return
            self.manual_speed_text = "0." if self.manual_speed_text == "" else self.manual_speed_text + "."
            return
        if char.isdigit():
            self.manual_speed_text += char

    def backspace_manual_speed(self) -> None:
        self.manual_speed_text = self.manual_speed_text[:-1]


class RadarTracker:
    """Tek bir baskin adayi kareler boyunca eslestirip surdurur."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.next_track_id = 1
        self.active_track: Optional[Track] = None
        self.logger = logger

    def reset(self) -> None:
        self.active_track = None

    def _candidate_priority(self, detection: Detection, profile: LayoutProfile) -> float:
        lane_distance = abs(detection.center[0] - profile.lane_center_x)
        focus_bonus = 420 if point_is_in_focus_band(detection.front_point, profile) else 0
        start_line_y = line_y_at_x(profile.start_p1, profile.start_p2, detection.front_point[0])
        near_start_bonus = 260 if detection.front_y <= (start_line_y + 22) else 0
        upstream_bonus = 140 if detection.front_y < start_line_y else 0
        return (
            (detection.front_y * 2)
            + detection.height
            + int(detection.area // 90)
            - (lane_distance // 3)
            + focus_bonus
            + near_start_bonus
            + upstream_bonus
        )

    def _create_track(self, detection: Detection, now: float, profile: LayoutProfile) -> Track:
        track = Track(
            track_id=self.next_track_id,
            bbox=detection.bbox,
            center=detection.center,
            front_point=detection.front_point,
            front_y=detection.front_y,
            area=detection.area,
            width=detection.width,
            height=detection.height,
            aspect_ratio=detection.aspect_ratio,
            last_seen_at=now,
            previous_seen_at=now,
        )
        start_line_y = line_y_at_x(profile.start_p1, profile.start_p2, detection.front_point[0])
        track.was_above_start_line = detection.front_y < start_line_y
        if point_is_in_focus_band(detection.front_point, profile) or detection.front_y <= (start_line_y + 18):
            track.seen_frames = 3
        self.next_track_id += 1
        return track

    def _bbox_iou(self, bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        union = max(1, area_a + area_b - inter_area)
        return inter_area / float(union)

    def _predict_point(self, point: Tuple[int, int], velocity: Tuple[int, int], missed_frames: int) -> Tuple[int, int]:
        scale = min(2, missed_frames + 1)
        return (point[0] + (velocity[0] * scale), point[1] + (velocity[1] * scale))

    def _smooth_velocity(self, old_velocity: Tuple[int, int], new_delta: Tuple[int, int]) -> Tuple[int, int]:
        vx = int(round((old_velocity[0] * 0.45) + (new_delta[0] * 0.55)))
        vy = int(round((old_velocity[1] * 0.45) + (new_delta[1] * 0.55)))
        return vx, vy

    def _match_score(self, track: Track, detection: Detection, profile: LayoutProfile) -> Optional[float]:
        predicted_center = self._predict_point(track.center, track.velocity_center, track.missed_frames)
        predicted_front = self._predict_point(track.front_point, track.velocity_front, track.missed_frames)
        center_dx = detection.center[0] - predicted_center[0]
        center_dy = detection.center[1] - predicted_center[1]
        center_distance = float((center_dx * center_dx + center_dy * center_dy) ** 0.5)
        front_distance = abs(detection.front_y - predicted_front[1])
        area_ratio = max(detection.area, 1.0) / max(track.area, 1.0)
        area_ratio = area_ratio if area_ratio >= 1.0 else 1.0 / max(area_ratio, 1e-6)
        width_ratio = max(detection.width, 1) / max(track.width, 1)
        width_ratio = width_ratio if width_ratio >= 1.0 else 1.0 / max(width_ratio, 1e-6)
        height_ratio = max(detection.height, 1) / max(track.height, 1)
        height_ratio = height_ratio if height_ratio >= 1.0 else 1.0 / max(height_ratio, 1e-6)
        iou = self._bbox_iou(track.bbox, detection.bbox)
        tx, ty, tw, th = track.bbox
        dx, dy, dw, dh = detection.bbox
        detection_inside_track = (
            dx >= tx - 4
            and dy >= ty - 4
            and (dx + dw) <= (tx + tw + 4)
            and (dy + dh) <= (ty + th + 4)
        )
        start_line_y = line_y_at_x(profile.start_p1, profile.start_p2, detection.front_point[0])
        near_start_zone = detection.front_y <= (start_line_y + 40)

        center_limit = TRACK_MATCH_CENTER_DISTANCE + min(track.missed_frames, 4) * 22
        front_limit = TRACK_MATCH_FRONT_DISTANCE + min(track.missed_frames, 4) * 18
        if track.status in ("confirmed", "lost"):
            center_limit += 30
            front_limit += 24
        if point_is_in_focus_band(detection.front_point, profile):
            center_limit += 18
            front_limit += 16

        if center_distance > center_limit:
            return None
        if front_distance > front_limit:
            return None
        max_area_ratio = 4.0 if track.status in ("confirmed", "lost") else 3.2
        if point_is_in_focus_band(detection.front_point, profile):
            max_area_ratio += 0.5
        if area_ratio > max_area_ratio:
            return None
        if area_ratio > 2.4 and iou < 0.12:
            return None
        close_near_start_match = center_distance <= 32.0 and front_distance <= 24.0
        if (
            track.status == "confirmed"
            and near_start_zone
            and detection_inside_track
            and detection.area < (track.area * 0.58)
            and detection.width < max(18, int(track.width * 0.72))
            and detection.height < max(16, int(track.height * 0.72))
            and iou < 0.42
        ):
            return None
        if track.status == "confirmed" and near_start_zone and iou < 0.25:
            if close_near_start_match:
                if area_ratio > 2.35 or width_ratio > 2.10 or height_ratio > 2.25:
                    return None
            elif area_ratio > 1.55 or width_ratio > 1.55 or height_ratio > 1.70:
                return None
        if track.status == "confirmed" and near_start_zone and detection.width >= int(PROCESS_WIDTH * 0.34) and iou < 0.38:
            return None
        if track.status in ("confirmed", "lost") and iou < 0.16:
            if width_ratio > 2.4 or height_ratio > 2.4:
                return None
        if track.direction_down is True and detection.front_y < (track.front_y - 6) and not point_is_in_focus_band(detection.front_point, profile):
            return None

        lane_distance = abs(detection.center[0] - profile.lane_center_x)
        focus_bonus = 52.0 if point_is_in_focus_band(detection.front_point, profile) else 0.0
        sticky_bonus = 26.0 if track.was_above_start_line else 0.0
        confirmed_bonus = 18.0 if track.status == "confirmed" else (10.0 if track.status == "lost" else 0.0)
        return (
            (130.0 - center_distance)
            + (85.0 - front_distance)
            - lane_distance * 0.15
            - (area_ratio - 1.0) * 15.0
            + (iou * 90.0)
            + focus_bonus
            + sticky_bonus
            + confirmed_bonus
        )

    def _smooth_point(self, old_point: Tuple[int, int], new_point: Tuple[int, int]) -> Tuple[int, int]:
        ox, oy = old_point
        nx, ny = new_point
        smoothed_x = int(round((TRACK_SMOOTHING * nx) + ((1.0 - TRACK_SMOOTHING) * ox)))
        smoothed_y = int(round((TRACK_SMOOTHING * ny) + ((1.0 - TRACK_SMOOTHING) * oy)))
        return smoothed_x, smoothed_y

    def _clamp_bbox_to_frame(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox
        w = max(1, min(PROCESS_WIDTH, int(w)))
        h = max(1, min(PROCESS_HEIGHT, int(h)))
        x = max(0, min(PROCESS_WIDTH - w, int(x)))
        y = max(0, min(PROCESS_HEIGHT - h, int(y)))
        return x, y, w, h

    def update(self, detections: list[Detection], now: float, profile: LayoutProfile) -> Optional[Track]:
        active = self.active_track

        if active is None:
            if not detections:
                return None
            best_detection = max(detections, key=lambda det: self._candidate_priority(det, profile))
            self.active_track = self._create_track(best_detection, now, profile)
            return self.active_track

        best_match: Optional[Detection] = None
        best_score: Optional[float] = None

        for detection in detections:
            score = self._match_score(active, detection, profile)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_match = detection

        if best_match is None:
            active.missed_frames += 1
            if active.missed_frames > TRACK_MAX_MISSED_FRAMES or (now - active.last_seen_at) > TRACK_LOST_TIMEOUT:
                self.active_track = None
                return None
            if active.status in ("confirmed", "lost"):
                previous_status = active.status
                active.previous_front_point = active.front_point
                start_line_y = line_y_at_x(profile.start_p1, profile.start_p2, active.front_point[0])
                near_start_hold = active.front_y <= (start_line_y + 68)
                hold_limit = TRACK_CONFIRM_HOLD_MISSED_FRAMES + (7 if near_start_hold else 0)
                if active.missed_frames <= hold_limit and previous_status == "confirmed":
                    active.status = "confirmed"
                    return active
                else:
                    active.status = "lost"
                active.center = self._predict_point(active.center, active.velocity_center, 1)
                active.front_point = self._predict_point(active.front_point, active.velocity_front, 1)
                active.front_y = active.front_point[1]
                if active.status == "lost" and previous_status != "lost" and self.logger is not None:
                    self.logger.info(
                        "track_lost_after_expand track=%s area=%.1f w=%s h=%s missed=%s",
                        active.track_id,
                        active.area,
                        active.width,
                        active.height,
                        active.missed_frames,
                    )
            return active

        old_front_y = active.front_y
        old_center = active.center
        old_front_point = active.front_point
        active.previous_front_point = active.front_point
        smoothed_center = self._smooth_point(active.center, best_match.center)
        smoothed_front_point = self._smooth_point(active.front_point, best_match.front_point)
        stabilized_bbox = best_match.bbox
        stabilized_width = best_match.width
        stabilized_height = best_match.height
        stabilized_front_y = smoothed_front_point[1]
        start_line_y = line_y_at_x(profile.start_p1, profile.start_p2, active.front_point[0])
        near_start_stabilize = (
            active.status == "confirmed"
            and active.front_y <= (start_line_y + 72)
        )
        raw_center_distance = float(
            (((best_match.center[0] - old_center[0]) ** 2) + ((best_match.center[1] - old_center[1]) ** 2)) ** 0.5
        )
        raw_front_distance = abs(best_match.front_y - old_front_y)
        if near_start_stabilize and raw_center_distance <= 46.0 and raw_front_distance <= 34:
            min_width = max(18, int(round(active.width * 0.82)))
            min_height = max(16, int(round(active.height * 0.82)))
            max_width = max(min_width, int(round(active.width * 1.30)))
            max_height = max(min_height, int(round(active.height * 1.30)))
            stabilized_width = max(min_width, min(max_width, best_match.width))
            stabilized_height = max(min_height, min(max_height, best_match.height))
            if active.direction_down is True:
                stabilized_front_y = max(stabilized_front_y, old_front_y - 2)
            stabilized_bbox = self._clamp_bbox_to_frame(
                (
                    int(round(smoothed_center[0] - (stabilized_width / 2.0))),
                    int(round(stabilized_front_y - stabilized_height)),
                    stabilized_width,
                    stabilized_height,
                )
            )

        active.center = smoothed_center
        active.bbox = stabilized_bbox
        active.front_point = (
            int(round(stabilized_bbox[0] + (stabilized_bbox[2] / 2.0))),
            int(round(stabilized_bbox[1] + stabilized_bbox[3])),
        )
        active.front_y = active.front_point[1]
        active.area = float(stabilized_bbox[2] * stabilized_bbox[3])
        active.width = stabilized_bbox[2]
        active.height = stabilized_bbox[3]
        active.aspect_ratio = active.width / max(active.height, 1)
        active.previous_seen_at = active.last_seen_at
        active.last_seen_at = now
        active.seen_frames += 1
        active.missed_frames = 0
        active.velocity_center = self._smooth_velocity(
            active.velocity_center,
            (active.center[0] - old_center[0], active.center[1] - old_center[1]),
        )
        active.velocity_front = self._smooth_velocity(
            active.velocity_front,
            (active.front_point[0] - old_front_point[0], active.front_point[1] - old_front_point[1]),
        )

        if active.direction_down is None:
            active.direction_down = active.front_y >= old_front_y
        elif active.front_y >= old_front_y - 2:
            active.direction_down = True

        active.was_above_start_line = active.was_above_start_line or (
            active.front_y < line_y_at_x(profile.start_p1, profile.start_p2, active.front_point[0])
        )

        start_line_y = line_y_at_x(profile.start_p1, profile.start_p2, active.front_point[0])
        if active.seen_frames >= TRACK_CONFIRM_FRAMES or (
            point_is_in_focus_band(active.front_point, profile) and active.seen_frames >= 2
        ) or (
            active.was_above_start_line and active.front_y <= (start_line_y + 28) and active.seen_frames >= 2
        ) or (
            active.was_above_start_line and active.seen_frames >= 2 and abs(active.center[0] - profile.lane_center_x) <= int(profile.lane_half_width * 0.9)
        ):
            active.status = "confirmed"
        else:
            active.status = "provisional"

        if abs(active.center[0] - old_center[0]) > MAX_CENTER_JUMP or abs(active.center[1] - old_center[1]) > MAX_CENTER_JUMP:
            active.status = "provisional"

        return active


def scale_legacy_point(point: Tuple[int, int]) -> Tuple[int, int]:
    return (
        int(round(point[0] * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH)))),
        int(round(point[1] * (FRAME_HEIGHT / float(LEGACY_FRAME_HEIGHT)))),
    )


PROFILES: Dict[str, LayoutProfile] = {
    "SOL": LayoutProfile(
        name="SOL",
        lane_center_x=int(round(212 * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH)))),
        lane_half_width=int(round(192 * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH)))),
        start_p1=scale_legacy_point((48, 95)),
        start_p2=scale_legacy_point((334, 79)),
        finish_p1=scale_legacy_point((77, 188)),
        finish_p2=scale_legacy_point((438, 162)),
        focus_p1=scale_legacy_point((48, 73)),
        focus_p2=scale_legacy_point((334, 57)),
    ),
    "SAG": LayoutProfile(
        name="SAG",
        lane_center_x=int(round(211 * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH)))),
        lane_half_width=int(round(192 * (FRAME_WIDTH / float(LEGACY_FRAME_WIDTH)))),
        start_p1=scale_legacy_point((89, 79)),
        start_p2=scale_legacy_point((375, 95)),
        finish_p1=scale_legacy_point((-15, 162)),
        finish_p2=scale_legacy_point((346, 188)),
        focus_p1=scale_legacy_point((89, 57)),
        focus_p2=scale_legacy_point((375, 73)),
    ),
}

DEFAULT_LAYOUTS: Dict[str, Dict[str, Tuple[int, int]]] = {
    mode_name: {
        "start_p1": profile.start_p1,
        "start_p2": profile.start_p2,
        "finish_p1": profile.finish_p1,
        "finish_p2": profile.finish_p2,
        "focus_p1": profile.focus_p1,
        "focus_p2": profile.focus_p2,
    }
    for mode_name, profile in PROFILES.items()
}
RUNTIME_DEFAULT_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "SOL": dict(DEFAULT_LAYOUTS["SOL"]),
    "SAG": dict(DEFAULT_LAYOUTS["SAG"]),
}
RUNTIME_DEFAULT_UPSTREAM_OFFSET_PX = DEFAULT_UPSTREAM_OFFSET_PX


def make_profile(name: str, start_p1: Tuple[int, int], start_p2: Tuple[int, int], finish_p1: Tuple[int, int], finish_p2: Tuple[int, int]) -> LayoutProfile:
    focus_p1 = (start_p1[0], start_p1[1] - FOCUS_BAND_HEIGHT)
    focus_p2 = (start_p2[0], start_p2[1] - FOCUS_BAND_HEIGHT)
    center_x = int(round((start_p1[0] + start_p2[0] + finish_p1[0] + finish_p2[0]) / 4.0))
    lane_half_width = max(30, int(round(max(abs(start_p2[0] - start_p1[0]), abs(finish_p2[0] - finish_p1[0])) / 2.0)) + 20)
    return LayoutProfile(
        name=name,
        lane_center_x=center_x,
        lane_half_width=lane_half_width,
        start_p1=start_p1,
        start_p2=start_p2,
        finish_p1=finish_p1,
        finish_p2=finish_p2,
        focus_p1=focus_p1,
        focus_p2=focus_p2,
    )


def build_profile_from_layout(name: str, layout: Dict[str, Any]) -> LayoutProfile:
    return make_profile(
        name,
        tuple(layout["start_p1"]),
        tuple(layout["start_p2"]),
        tuple(layout["finish_p1"]),
        tuple(layout["finish_p2"]),
    )


def export_profile_layout(profile: LayoutProfile) -> Dict[str, list[int]]:
    return {
        "start_p1": [profile.start_p1[0], profile.start_p1[1]],
        "start_p2": [profile.start_p2[0], profile.start_p2[1]],
        "finish_p1": [profile.finish_p1[0], profile.finish_p1[1]],
        "finish_p2": [profile.finish_p2[0], profile.finish_p2[1]],
    }


def default_layout_config() -> Dict[str, Any]:
    return {
        "modes": {
            "SOL": export_profile_layout(build_profile_from_layout("SOL", DEFAULT_LAYOUTS["SOL"])),
            "SAG": export_profile_layout(build_profile_from_layout("SAG", DEFAULT_LAYOUTS["SAG"])),
        },
        "upstream_offset_px": DEFAULT_UPSTREAM_OFFSET_PX,
        "meter_per_pixel": None,
        "real_distance_meter": DISTANCE_M,
        "speed_limit_kmh": SPEED_LIMIT_KMH,
        "last_active_mode": "SOL",
        "detection_mode": "DENGELI",
        "performance_mode": "NORMAL",
    }


def ensure_directories() -> None:
    """Gerekli klasorleri olusturur."""
    VIOLATION_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_layout_config(logger: logging.Logger) -> Dict[str, Any]:
    """Cizgi ve genel ayarlari JSON'dan yukler."""
    config = default_layout_config()
    if not LAYOUT_CONFIG_FILE.exists():
        return config

    try:
        loaded = json.loads(LAYOUT_CONFIG_FILE.read_text(encoding="utf-8"))
        config["upstream_offset_px"] = int(loaded.get("upstream_offset_px", config["upstream_offset_px"]))
        config["meter_per_pixel"] = loaded.get("meter_per_pixel", config["meter_per_pixel"])
        config["real_distance_meter"] = float(loaded.get("real_distance_meter", config["real_distance_meter"]))
        config["speed_limit_kmh"] = float(loaded.get("speed_limit_kmh", config["speed_limit_kmh"]))
        config["last_active_mode"] = str(loaded.get("last_active_mode", config["last_active_mode"]))
        config["detection_mode"] = str(loaded.get("detection_mode", config["detection_mode"])).upper()
        config["performance_mode"] = str(loaded.get("performance_mode", config["performance_mode"])).upper()
        loaded_modes = loaded.get("modes", {})
        for mode_name in ("SOL", "SAG"):
            if mode_name in loaded_modes:
                mode_data = loaded_modes[mode_name]
                for key in ("start_p1", "start_p2", "finish_p1", "finish_p2"):
                    if key not in mode_data:
                        raise ValueError(f"Eksik layout anahtari: {mode_name}.{key}")
                config["modes"][mode_name] = {
                    "start_p1": [int(mode_data["start_p1"][0]), int(mode_data["start_p1"][1])],
                    "start_p2": [int(mode_data["start_p2"][0]), int(mode_data["start_p2"][1])],
                    "finish_p1": [int(mode_data["finish_p1"][0]), int(mode_data["finish_p1"][1])],
                    "finish_p2": [int(mode_data["finish_p2"][0]), int(mode_data["finish_p2"][1])],
                }
        logger.info("Layout config yuklendi: %s", LAYOUT_CONFIG_FILE)
    except Exception as exc:
        logger.warning("Layout config okunamadi, varsayilanlar kullanilacak: %s", exc)
    return config


def save_layout_config(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Cizgi ve genel ayarlari JSON'a kaydeder."""
    try:
        LAYOUT_CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Layout config kaydedildi: %s", LAYOUT_CONFIG_FILE)
    except Exception as exc:
        logger.warning("Layout config kaydedilemedi: %s", exc)


def load_distance_setting(logger: logging.Logger) -> float:
    """Kayitli mesafe varsa yukler, yoksa varsayilani kullanir."""
    if not DISTANCE_FILE.exists():
        return DISTANCE_M
    try:
        raw_text = DISTANCE_FILE.read_text(encoding="utf-8").strip()
        loaded_value = float(raw_text)
        if loaded_value <= 0:
            raise ValueError("Mesafe sifirdan buyuk olmali.")
        logger.info("Kayitli mesafe yuklendi: %.1f m", loaded_value)
        return loaded_value
    except Exception as exc:
        logger.warning("Mesafe ayari okunamadi, varsayilan kullanilacak: %s", exc)
        return DISTANCE_M


def save_distance_setting(distance_m: float, logger: logging.Logger) -> None:
    """Kalibre edilen mesafeyi dosyaya kaydeder."""
    try:
        DISTANCE_FILE.write_text(f"{distance_m:.1f}\n", encoding="utf-8")
        logger.info("Mesafe kaydedildi: %.1f m", distance_m)
    except Exception as exc:
        logger.warning("Mesafe kaydedilemedi: %s", exc)


def setup_logging() -> logging.Logger:
    """Hem dosyaya hem stdout'a log yazar."""
    ensure_directories()

    logger = logging.getLogger("radarpi")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


class CameraAdapter:
    """CSI Picamera2 icin ortak read/release arayuzu."""

    def __init__(
        self,
        mode: str,
        cv_cap: Optional[cv2.VideoCapture] = None,
        picam2: Optional[Any] = None,
    ) -> None:
        self.mode = mode
        self.cv_cap = cv_cap
        self.picam2 = picam2
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = 0
        self._frame_lock = threading.Lock()
        self._reader_stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

    def _normalize_frame(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def start_reader(self) -> None:
        if self.mode != "csi" or self.picam2 is None or self._reader_thread is not None:
            return

        def _reader_loop() -> None:
            while not self._reader_stop.is_set():
                try:
                    frame = self.picam2.capture_array("main")
                    frame = self._normalize_frame(frame)
                    if frame is not None:
                        with self._frame_lock:
                            self._latest_frame = frame
                            self._latest_frame_id += 1
                except Exception:
                    time.sleep(0.005)

        self._reader_thread = threading.Thread(
            target=_reader_loop,
            name="radarpi-camera-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        if self.mode == "csi":
            if self.picam2 is None:
                return False, None, -1
            with self._frame_lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id
            if frame is None:
                return False, None, -1
            return True, frame, frame_id

        return False, None, -1

    def release(self) -> None:
        self._reader_stop.set()
        if self._reader_thread is not None:
            try:
                self._reader_thread.join(timeout=0.5)
            except Exception:
                pass
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
        if self.cv_cap is not None:
            self.cv_cap.release()


def create_camera(logger: logging.Logger) -> CameraAdapter:
    """CSI/Picamera2 ile kamerayi acar."""
    if Picamera2 is None:
        raise RuntimeError("Picamera2 modulu bulunamadi. python3-picamera2 paketini kurun.")

    picam2: Optional[Any] = None
    try:
        picam2 = Picamera2()
        preview_config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"},
            controls={"FrameRate": TARGET_FPS},
            buffer_count=1,
            queue=False,
        )
        picam2.configure(preview_config)
        picam2.start()
        time.sleep(0.6)
        test_frame = picam2.capture_array("main")
        if test_frame is not None:
            adapter = CameraAdapter(mode="csi", picam2=picam2)
            with adapter._frame_lock:
                adapter._latest_frame = adapter._normalize_frame(test_frame)
            adapter.start_reader()
            logger.info(
                "CSI kamera hazirlandi: %sx%s @ hedef %s FPS",
                FRAME_WIDTH,
                FRAME_HEIGHT,
                TARGET_FPS,
            )
            return adapter
        raise RuntimeError("CSI kameradan test karesi alinamadi.")
    except Exception as exc:
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
            try:
                picam2.close()
            except Exception:
                pass
        raise RuntimeError(f"CSI/Picamera2 kamera acilamadi: {exc}") from exc


def create_subtractor() -> cv2.BackgroundSubtractor:
    """Hafif ve stabil arka plan cikarma nesnesi."""
    return cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY,
        varThreshold=BG_VAR_THRESHOLD,
        detectShadows=BG_DETECT_SHADOWS,
    )


def apply_runtime_modes(detection_mode: str, performance_mode: str) -> None:
    global MIN_CONTOUR_AREA
    global MIN_BBOX_WIDTH
    global MIN_BBOX_HEIGHT
    global TRACK_CONFIRM_FRAMES
    global MAX_CENTER_JUMP
    global MAX_FRONT_JUMP
    global WARMUP_FRAMES_RUNTIME
    global PROCESS_WIDTH
    global PROCESS_HEIGHT
    global DETECTION_INTERVAL_ACTIVE
    global OVERLAY_RENDER_INTERVAL

    detection_key = detection_mode.upper()
    performance_key = performance_mode.upper()
    detection_preset = DETECTION_MODE_PRESETS.get(detection_key, DETECTION_MODE_PRESETS["DENGELI"])
    performance_preset = PERFORMANCE_MODE_PRESETS.get(performance_key, PERFORMANCE_MODE_PRESETS["NORMAL"])

    MIN_CONTOUR_AREA = int(detection_preset["MIN_CONTOUR_AREA"])
    MIN_BBOX_WIDTH = int(detection_preset["MIN_BBOX_WIDTH"])
    MIN_BBOX_HEIGHT = int(detection_preset["MIN_BBOX_HEIGHT"])
    TRACK_CONFIRM_FRAMES = int(detection_preset["TRACK_CONFIRM_FRAMES"])
    MAX_CENTER_JUMP = int(detection_preset["MAX_CENTER_JUMP"])
    MAX_FRONT_JUMP = int(detection_preset["MAX_FRONT_JUMP"])
    WARMUP_FRAMES_RUNTIME = int(detection_preset.get("WARMUP_FRAMES", WARMUP_FRAMES))

    PROCESS_WIDTH = int(performance_preset["PROCESS_WIDTH"])
    PROCESS_HEIGHT = int(performance_preset["PROCESS_HEIGHT"])
    DETECTION_INTERVAL_ACTIVE = int(performance_preset["DETECTION_INTERVAL_ACTIVE"])
    OVERLAY_RENDER_INTERVAL = int(performance_preset["OVERLAY_RENDER_INTERVAL"])


def get_lane_bounds(profile: LayoutProfile, frame_width: int) -> Tuple[int, int]:
    """Aktif profile gore hedef secim koridorunu hesaplar."""
    x1 = max(0, profile.lane_center_x - profile.lane_half_width)
    x2 = min(frame_width - 1, profile.lane_center_x + profile.lane_half_width)
    return x1, x2


def line_y_at_x(line_p1: Tuple[int, int], line_p2: Tuple[int, int], x: int) -> int:
    """Verilen x konumunda cizginin y degerini bulur."""
    x1, y1 = line_p1
    x2, y2 = line_p2
    if x2 == x1:
        return int((y1 + y2) / 2)

    ratio = (x - x1) / float(x2 - x1)
    ratio = max(0.0, min(1.0, ratio))
    return int(round(y1 + (y2 - y1) * ratio))


def line_x_at_y(line_p1: Tuple[int, int], line_p2: Tuple[int, int], y: int) -> int:
    """Verilen y konumunda cizginin x degerini bulur."""
    x1, y1 = line_p1
    x2, y2 = line_p2
    if y2 == y1:
        return int((x1 + x2) / 2)

    ratio = (y - y1) / float(y2 - y1)
    return int(round(x1 + (x2 - x1) * ratio))


def extend_line_to_frame(line_p1: Tuple[int, int], line_p2: Tuple[int, int], top_y: int, bottom_y: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Bir cizgiyi ekranin ust ve alt sinirina kadar uzatir."""
    top_x = line_x_at_y(line_p1, line_p2, top_y)
    bottom_x = line_x_at_y(line_p1, line_p2, bottom_y)
    return (top_x, top_y), (bottom_x, bottom_y)


def get_visual_guide_lines(profile: LayoutProfile, frame_height: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Mavi yol rehberi cizgilerini gercek koridor kenarlarindan ekrana uzatir."""
    top_y = 0 if not SHOW_WINDOW else TOP_BAR_HEIGHT
    bottom_y = frame_height - 1
    left_top, left_bottom = extend_line_to_frame(profile.start_p1, profile.finish_p1, top_y, bottom_y)
    right_top, right_bottom = extend_line_to_frame(profile.start_p2, profile.finish_p2, top_y, bottom_y)
    return left_top, left_bottom, right_top, right_bottom


def profile_signature(profile: LayoutProfile) -> Tuple[Any, ...]:
    """Profil degistiginde cache yenilemek icin sabit imza."""
    return (
        profile.name,
        profile.lane_center_x,
        profile.lane_half_width,
        profile.start_p1,
        profile.start_p2,
        profile.finish_p1,
        profile.finish_p2,
        profile.focus_p1,
        profile.focus_p2,
    )


def set_start_line_points(profile: LayoutProfile, start_p1: Tuple[int, int], start_p2: Tuple[int, int]) -> LayoutProfile:
    return make_profile(profile.name, start_p1, start_p2, profile.finish_p1, profile.finish_p2)


def set_finish_line_points(profile: LayoutProfile, finish_p1: Tuple[int, int], finish_p2: Tuple[int, int]) -> LayoutProfile:
    return make_profile(profile.name, profile.start_p1, profile.start_p2, finish_p1, finish_p2)


def get_vehicle_tracking_point(detection: Detection) -> Tuple[int, int]:
    return detection.front_point


def point_side_of_line(point: Tuple[int, int], line_p1: Tuple[int, int], line_p2: Tuple[int, int]) -> float:
    return ((line_p2[0] - line_p1[0]) * (point[1] - line_p1[1])) - ((line_p2[1] - line_p1[1]) * (point[0] - line_p1[0]))


def has_crossed_line(previous_point: Tuple[int, int], current_point: Tuple[int, int], line_p1: Tuple[int, int], line_p2: Tuple[int, int]) -> bool:
    prev_side = point_side_of_line(previous_point, line_p1, line_p2)
    curr_side = point_side_of_line(current_point, line_p1, line_p2)
    return (prev_side < 0 <= curr_side) or (prev_side > 0 >= curr_side)


def estimate_line_cross_time(
    previous_point: Tuple[int, int],
    previous_time: float,
    current_point: Tuple[int, int],
    current_time: float,
    line_p1: Tuple[int, int],
    line_p2: Tuple[int, int],
) -> Optional[float]:
    """Iki nokta arasindaki cizgi gecisini zaman ekseninde enterpole eder."""
    prev_side = point_side_of_line(previous_point, line_p1, line_p2)
    curr_side = point_side_of_line(current_point, line_p1, line_p2)

    if prev_side == 0:
        return previous_time
    if curr_side == 0:
        return current_time
    if (prev_side < 0 and curr_side < 0) or (prev_side > 0 and curr_side > 0):
        return None

    denom = abs(prev_side) + abs(curr_side)
    if denom <= 1e-6:
        return current_time

    ratio = abs(prev_side) / denom
    return previous_time + ((current_time - previous_time) * ratio)


def offset_line_upstream(line_p1: Tuple[int, int], line_p2: Tuple[int, int], offset_px: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    dx = float(line_p2[0] - line_p1[0])
    dy = float(line_p2[1] - line_p1[1])
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    normal_x = -dy / length
    normal_y = dx / length
    if normal_y > 0:
        normal_x *= -1.0
        normal_y *= -1.0
    p1 = (int(round(line_p1[0] + normal_x * offset_px)), int(round(line_p1[1] + normal_y * offset_px)))
    p2 = (int(round(line_p2[0] + normal_x * offset_px)), int(round(line_p2[1] + normal_y * offset_px)))
    return p1, p2


def line_intersection(
    a1: Tuple[int, int],
    a2: Tuple[int, int],
    b1: Tuple[int, int],
    b2: Tuple[int, int],
) -> Tuple[int, int]:
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if abs(denom) < 1e-6:
        return a1
    det_a = (x1 * y2) - (y1 * x2)
    det_b = (x3 * y4) - (y3 * x4)
    px = ((det_a * (x3 - x4)) - ((x1 - x2) * det_b)) / denom
    py = ((det_a * (y3 - y4)) - ((y1 - y2) * det_b)) / denom
    return int(round(px)), int(round(py))


def build_roi_polygon_points(
    profile: LayoutProfile,
    top_line_p1: Tuple[int, int],
    top_line_p2: Tuple[int, int],
) -> np.ndarray:
    left_top = line_intersection(profile.start_p1, profile.finish_p1, top_line_p1, top_line_p2)
    right_top = line_intersection(profile.start_p2, profile.finish_p2, top_line_p1, top_line_p2)
    return np.array(
        [
            [left_top[0], left_top[1]],
            [right_top[0], right_top[1]],
            [profile.finish_p2[0], profile.finish_p2[1]],
            [profile.finish_p1[0], profile.finish_p1[1]],
        ],
        dtype=np.int32,
    )


def build_measurement_roi(frame_shape: Tuple[int, int, int], profile: LayoutProfile) -> np.ndarray:
    frame_height, frame_width = frame_shape[:2]
    polygon = build_roi_polygon_points(profile, profile.start_p1, profile.start_p2)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, frame_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, frame_height - 1)
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def build_detection_roi(frame_shape: Tuple[int, int, int], profile: LayoutProfile, upstream_offset_px: int) -> np.ndarray:
    frame_height, frame_width = frame_shape[:2]
    detect_start_p1, detect_start_p2 = offset_line_upstream(profile.start_p1, profile.start_p2, upstream_offset_px)
    polygon = build_roi_polygon_points(profile, detect_start_p1, detect_start_p2)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, frame_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, frame_height - 1)
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def get_measurement_roi_polygon(profile: LayoutProfile) -> np.ndarray:
    return build_roi_polygon_points(profile, profile.start_p1, profile.start_p2)


def get_detection_roi_polygon(profile: LayoutProfile, upstream_offset_px: int) -> np.ndarray:
    detect_start_p1, detect_start_p2 = offset_line_upstream(profile.start_p1, profile.start_p2, upstream_offset_px)
    return build_roi_polygon_points(profile, detect_start_p1, detect_start_p2)


def draw_roi_visual_overlay(
    frame: np.ndarray,
    profile: LayoutProfile,
    upstream_offset_px: int,
    alpha: float = 0.18,
) -> np.ndarray:
    overlay = frame.copy()
    detection_polygon = get_detection_roi_polygon(profile, upstream_offset_px).astype(np.int32)
    measurement_polygon = get_measurement_roi_polygon(profile).astype(np.int32)
    cv2.fillConvexPoly(overlay, detection_polygon, (40, 220, 80))
    cv2.fillConvexPoly(overlay, measurement_polygon, (220, 80, 220))
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.polylines(frame, [detection_polygon], True, (80, 255, 120), 1, cv2.LINE_AA)
    cv2.polylines(frame, [measurement_polygon], True, (255, 120, 255), 1, cv2.LINE_AA)
    return frame


def get_polygon_bounds(polygon: np.ndarray, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    clipped = polygon.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, frame_width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, frame_height - 1)
    x, y, w, h = cv2.boundingRect(clipped.astype(np.int32))
    return int(x), int(y), int(max(1, w)), int(max(1, h))


def get_processing_roi(frame_shape: Tuple[int, int, int], profile: LayoutProfile) -> np.ndarray:
    """
    Hareket analizi icin yol perspektifine uygun ROI maskesi uretir.

    Not:
    - Goruntu ekranda tamamen gorunmeye devam eder.
    - Yalnizca bu ROI icindeki hareketler islenir.
    """
    frame_height, frame_width = frame_shape[:2]
    main_polygon = np.array(
        [
            [profile.start_p1[0], profile.start_p1[1]],
            [profile.start_p2[0], profile.start_p2[1]],
            [profile.finish_p2[0], profile.finish_p2[1]],
            [profile.finish_p1[0], profile.finish_p1[1]],
        ],
        dtype=np.int32,
    )
    focus_polygon = np.array(
        [
            [profile.focus_p1[0], profile.focus_p1[1]],
            [profile.focus_p2[0], profile.focus_p2[1]],
            [profile.start_p2[0], profile.start_p2[1]],
            [profile.start_p1[0], profile.start_p1[1]],
        ],
        dtype=np.int32,
    )

    main_polygon[:, 0] = np.clip(main_polygon[:, 0], 0, frame_width - 1)
    main_polygon[:, 1] = np.clip(main_polygon[:, 1], 0, frame_height - 1)
    focus_polygon[:, 0] = np.clip(focus_polygon[:, 0], 0, frame_width - 1)
    focus_polygon[:, 1] = np.clip(focus_polygon[:, 1], 0, frame_height - 1)

    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, main_polygon, 255)
    cv2.fillConvexPoly(mask, focus_polygon, 255)
    return mask


def scale_point(point: Tuple[int, int], scale_x: float, scale_y: float) -> Tuple[int, int]:
    """Bir noktayi hedef koordinat sistemine tasir."""
    return int(round(point[0] * scale_x)), int(round(point[1] * scale_y))


def scale_layout_profile(
    profile: LayoutProfile,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> LayoutProfile:
    """Cizgi ve odak koordinatlarini isleme boyutuna gore olceklendirir."""
    scale_x = dst_width / float(src_width)
    scale_y = dst_height / float(src_height)
    return LayoutProfile(
        name=profile.name,
        lane_center_x=int(round(profile.lane_center_x * scale_x)),
        lane_half_width=max(8, int(round(profile.lane_half_width * scale_x))),
        start_p1=scale_point(profile.start_p1, scale_x, scale_y),
        start_p2=scale_point(profile.start_p2, scale_x, scale_y),
        finish_p1=scale_point(profile.finish_p1, scale_x, scale_y),
        finish_p2=scale_point(profile.finish_p2, scale_x, scale_y),
        focus_p1=scale_point(profile.focus_p1, scale_x, scale_y),
        focus_p2=scale_point(profile.focus_p2, scale_x, scale_y),
    )


def scale_track_for_display(
    track: Optional[Track],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> Optional[Track]:
    """Isleme boyutundaki track'i gosterim boyutuna tasir."""
    if track is None:
        return None

    scale_x = dst_width / float(src_width)
    scale_y = dst_height / float(src_height)
    x, y, w, h = track.bbox

    return Track(
        track_id=track.track_id,
        bbox=(
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            max(1, int(round(w * scale_x))),
            max(1, int(round(h * scale_y))),
        ),
        center=scale_point(track.center, scale_x, scale_y),
        front_point=scale_point(track.front_point, scale_x, scale_y),
        front_y=int(round(track.front_y * scale_y)),
        area=track.area * scale_x * scale_y,
        width=max(1, int(round(track.width * scale_x))),
        height=max(1, int(round(track.height * scale_y))),
        aspect_ratio=track.aspect_ratio,
        last_seen_at=track.last_seen_at,
        previous_seen_at=track.previous_seen_at,
        seen_frames=track.seen_frames,
        missed_frames=track.missed_frames,
        status=track.status,
        measurement_state=track.measurement_state,
        start_time=track.start_time,
        was_above_start_line=track.was_above_start_line,
        direction_down=track.direction_down,
        last_speed_kmh=track.last_speed_kmh,
        velocity_center=scale_point(track.velocity_center, scale_x, scale_y),
        velocity_front=scale_point(track.velocity_front, scale_x, scale_y),
        previous_front_point=scale_point(track.previous_front_point, scale_x, scale_y) if track.previous_front_point is not None else None,
    )


def build_active_polygon(profile: LayoutProfile, frame_height: int) -> np.ndarray:
    """Gorsel ve islemsel koridoru ayni geometriyle uretir."""
    left_guide_top, left_guide_bottom, right_guide_top, right_guide_bottom = get_visual_guide_lines(profile, frame_height)
    return np.array(
        [
            [left_guide_top[0], left_guide_top[1]],
            [right_guide_top[0], right_guide_top[1]],
            [right_guide_bottom[0], right_guide_bottom[1]],
            [left_guide_bottom[0], left_guide_bottom[1]],
        ],
        dtype=np.int32,
    )


def preprocess_frame(
    frame: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """Eski tam-frame uyumluluk yardimcisi."""
    return preprocess_frame_roi(
        frame,
        subtractor,
        roi_mask,
        (0, 0, frame.shape[1], frame.shape[0]),
    )[0]


def preprocess_frame_roi(
    frame: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    roi_mask: np.ndarray,
    roi_bounds: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Yalnizca ROI crop uzerinde hareket maskesi uretir."""
    x, y, w, h = roi_bounds
    roi_frame = frame[y : y + h, x : x + w]
    roi_mask_crop = roi_mask[y : y + h, x : x + w]
    timings = {
        "roi_time": 0.0,
        "mask_time": 0.0,
        "threshold_time": 0.0,
        "morphology_time": 0.0,
    }

    t0 = time.perf_counter()
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    if GAUSSIAN_BLUR_KERNEL[0] > 1 and GAUSSIAN_BLUR_KERNEL[1] > 1:
        gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
    timings["roi_time"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    fg_mask = subtractor.apply(gray)
    masked_fg = cv2.bitwise_and(fg_mask, roi_mask_crop)
    timings["mask_time"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    _, thresholded = cv2.threshold(masked_fg, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    timings["threshold_time"] = (time.perf_counter() - t0) * 1000.0

    cleaned = thresholded
    if MORPH_OPEN_ITERATIONS > 0 or MORPH_CLOSE_ITERATIONS > 0 or MORPH_DILATE_ITERATIONS > 0:
        t0 = time.perf_counter()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        )
        if MORPH_OPEN_ITERATIONS > 0:
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_OPEN,
                kernel,
                iterations=MORPH_OPEN_ITERATIONS,
            )
        if MORPH_CLOSE_ITERATIONS > 0:
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=MORPH_CLOSE_ITERATIONS,
            )
        if MORPH_DILATE_ITERATIONS > 0:
            cleaned = cv2.dilate(cleaned, kernel, iterations=MORPH_DILATE_ITERATIONS)
        timings["morphology_time"] = (time.perf_counter() - t0) * 1000.0

    return cleaned, timings


def contour_to_detection(contour: np.ndarray) -> Optional[Detection]:
    """Contour'u uygulamanin kullandigi ozete donusturur."""
    area = cv2.contourArea(contour)
    soft_min_area = max(90, int(round(MIN_CONTOUR_AREA * 0.45)))
    if area < soft_min_area or area > MAX_CONTOUR_AREA:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None

    soft_min_width = max(8, int(round(MIN_BBOX_WIDTH * 0.65)))
    soft_min_height = max(6, int(round(MIN_BBOX_HEIGHT * 0.65)))
    if w < soft_min_width or h < soft_min_height:
        return None

    aspect_ratio = w / float(h)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return None

    center_x = x + (w // 2)
    center_y = y + (h // 2)

    return Detection(
        bbox=(x, y, w, h),
        center=(center_x, center_y),
        front_point=(center_x, y + h),
        front_y=y + h,
        area=area,
        width=w,
        height=h,
        aspect_ratio=aspect_ratio,
    )


def point_crosses_horizontal_segment(
    point: Tuple[int, int],
    line_p1: Tuple[int, int],
    line_p2: Tuple[int, int],
) -> bool:
    """Nokta, verilen egimli/yatay cizgiyi gecti mi kontrol eder."""
    px, py = point
    min_x = min(line_p1[0], line_p2[0])
    max_x = max(line_p1[0], line_p2[0])
    if not (min_x <= px <= max_x):
        return False
    return py >= line_y_at_x(line_p1, line_p2, px)


def point_is_in_focus_band(point: Tuple[int, int], profile: LayoutProfile) -> bool:
    """Ilk tespit icin baslangic cizgisinin hemen ustundeki odak bandi."""
    px, py = point
    top_y = line_y_at_x(profile.focus_p1, profile.focus_p2, px)
    bottom_y = line_y_at_x(profile.start_p1, profile.start_p2, px)
    min_x = min(profile.focus_p1[0], profile.focus_p2[0], profile.start_p1[0], profile.start_p2[0])
    max_x = max(profile.focus_p1[0], profile.focus_p2[0], profile.start_p1[0], profile.start_p2[0])
    return min_x <= px <= max_x and top_y <= py <= bottom_y


def validate_detection_candidate(detection: Detection, profile: LayoutProfile, frame_width: int) -> bool:
    """
    Ileride hafif AI veya ek dogrulama baglanabilsin diye ayrik hook.

    Su an yalnizca klasik OpenCV akisi icinde her adayi kabul eder.
    """
    if detection.width <= 0 or detection.height <= 0:
        return False

    start_line_y = line_y_at_x(profile.start_p1, profile.start_p2, detection.front_point[0])
    near_start_or_upstream = detection.front_y <= (start_line_y + 22)
    dynamic_min_area = MIN_CONTOUR_AREA
    dynamic_min_width = MIN_BBOX_WIDTH
    dynamic_min_height = MIN_BBOX_HEIGHT
    if near_start_or_upstream:
        dynamic_min_area = max(160, int(round(MIN_CONTOUR_AREA * 0.62)))
        dynamic_min_width = max(14, int(round(MIN_BBOX_WIDTH * 0.90)))
        dynamic_min_height = max(10, int(round(MIN_BBOX_HEIGHT * 0.88)))

    if detection.area < dynamic_min_area:
        return False
    if detection.width < dynamic_min_width or detection.height < dynamic_min_height:
        return False

    bbox_area = float(detection.width * detection.height)
    fill_ratio = detection.area / max(bbox_area, 1.0)
    if fill_ratio < 0.16:
        return False
    if bbox_area >= float(frame_width * PROCESS_HEIGHT) * 0.08 and fill_ratio < 0.30:
        return False

    if detection.aspect_ratio > 3.4 and fill_ratio < 0.30:
        return False

    if detection.width >= int(frame_width * 0.40):
        return False
    if detection.height >= int(PROCESS_HEIGHT * 0.30) and fill_ratio < 0.28:
        return False
    if detection.width >= int(frame_width * 0.32) and detection.height <= int(PROCESS_HEIGHT * 0.12) and fill_ratio < 0.34:
        return False

    return True


def collect_detections(
    mask: np.ndarray,
    profile: LayoutProfile,
    frame_width: int,
    roi_offset: Tuple[int, int] = (0, 0),
) -> list[Detection]:
    """Aktif profile gore butun uygun adaylari toplar."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    lane_x1, lane_x2 = get_lane_bounds(profile, frame_width)
    detections: list[Detection] = []
    offset_x, offset_y = roi_offset

    for contour in contours:
        detection = contour_to_detection(contour)
        if detection is None:
            continue
        if not validate_detection_candidate(detection, profile, frame_width):
            continue

        if offset_x != 0 or offset_y != 0:
            x, y, w, h = detection.bbox
            center_x, center_y = detection.center
            front_x, front_y = detection.front_point
            detection = Detection(
                bbox=(x + offset_x, y + offset_y, w, h),
                center=(center_x + offset_x, center_y + offset_y),
                front_point=(front_x + offset_x, front_y + offset_y),
                front_y=front_y + offset_y,
                area=detection.area,
                width=detection.width,
                height=detection.height,
                aspect_ratio=detection.aspect_ratio,
            )

        center_x, _ = detection.center
        if not (lane_x1 <= center_x <= lane_x2):
            continue

        detections.append(detection)

    return detections


def round_up_distance(distance_m: float) -> float:
    """Kalibrasyon mesafesini yuvarlamadan, dogrudan olculen deger olarak dondurur."""
    return max(0.1, float(distance_m))


def line_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Iki sonsuz dogrunun kesisimini bulur."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if abs(denom) < 1e-6:
        return None

    det1 = (x1 * y2) - (y1 * x2)
    det2 = (x3 * y4) - (y3 * x4)
    px = ((det1 * (x3 - x4)) - ((x1 - x2) * det2)) / denom
    py = ((det1 * (y3 - y4)) - ((y1 - y2) * det2)) / denom
    return float(px), float(py)


def compute_distance_from_reference_px(profile: LayoutProfile, reference_px: float) -> Optional[float]:
    """Referans uzunlugunun piksel karsiligindan iki cizgi arasi mesafeyi bulur."""
    if reference_px < 4.0:
        return None
    center_x = int(round((profile.start_p1[0] + profile.start_p2[0] + profile.finish_p1[0] + profile.finish_p2[0]) / 4.0))
    start_y = line_y_at_x(profile.start_p1, profile.start_p2, center_x)
    finish_y = line_y_at_x(profile.finish_p1, profile.finish_p2, center_x)
    pixel_distance = abs(finish_y - start_y)
    if pixel_distance <= 1.0:
        return None
    pixels_per_meter = reference_px / CALIBRATION_CARD_LENGTH_M
    if pixels_per_meter <= 0:
        return None
    measured_distance = pixel_distance / pixels_per_meter
    return round_up_distance(measured_distance)


def manual_calibration_from_two_points(
    profile: LayoutProfile,
    first_point: Tuple[int, int],
    second_point: Tuple[int, int],
) -> Optional[float]:
    """
    Referans nesnenin uzun kenarinin iki kosesi secilerek iki cizgi arasi mesafeyi bulur.

    Varsayim:
    - Kullanici 115 cm'lik referansin iki ucuna dokunur.
    - Referans yol duzleminde ve mavi cizgilere paralel yerlestirilir.
    """
    x1, y1 = first_point
    x2, y2 = second_point
    reference_px = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if reference_px < 12.0:
        return None

    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    dir_x = (x2 - x1) / reference_px
    dir_y = (y2 - y1) / reference_px

    long_line_p1 = (mid_x - (dir_x * 800.0), mid_y - (dir_y * 800.0))
    long_line_p2 = (mid_x + (dir_x * 800.0), mid_y + (dir_y * 800.0))

    start_hit = line_intersection(long_line_p1, long_line_p2, profile.start_p1, profile.start_p2)
    finish_hit = line_intersection(long_line_p1, long_line_p2, profile.finish_p1, profile.finish_p2)
    if start_hit is None or finish_hit is None:
        return None

    pixel_distance = float(((finish_hit[0] - start_hit[0]) ** 2 + (finish_hit[1] - start_hit[1]) ** 2) ** 0.5)
    if pixel_distance <= 1.0:
        return None

    pixels_per_meter = reference_px / CALIBRATION_CARD_LENGTH_M
    if pixels_per_meter <= 0:
        return None

    measured_distance = pixel_distance / pixels_per_meter
    return round_up_distance(measured_distance)


def try_auto_calibration_from_tap(
    frame: np.ndarray,
    profile: LayoutProfile,
    target_point: Tuple[int, int],
) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    """Dokunulan nesnenin etrafinda otomatik kutu arar."""
    active_polygon = build_active_polygon(profile, frame.shape[0])
    active_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(active_mask, active_polygon, 255)

    masked_frame = cv2.bitwise_and(frame, frame, mask=active_mask)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    tx, ty = target_point
    roi_half_w = 110
    roi_half_h = 90
    roi_x1 = max(0, tx - roi_half_w)
    roi_y1 = max(TOP_BAR_HEIGHT, ty - roi_half_h)
    roi_x2 = min(frame.shape[1], tx + roi_half_w)
    roi_y2 = min(frame.shape[0], ty + roi_half_h)
    if (roi_x2 - roi_x1) < 20 or (roi_y2 - roi_y1) < 20:
        return None

    gray_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if gray_roi.size == 0:
        return None

    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    _, otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        17,
        3,
    )
    combined = cv2.bitwise_or(edges, cv2.bitwise_or(otsu_inv, cv2.bitwise_xor(otsu, adaptive)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.dilate(combined, kernel, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    local_x = tx - roi_x1
    local_y = ty - roi_y1
    expected_ratio = CALIBRATION_CARD_LENGTH_M / CALIBRATION_CARD_WIDTH_M
    best_score = float("inf")
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_long_edge = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 70 or area > 22000:
            continue

        rect = cv2.minAreaRect(contour)
        (cx_local, cy_local), (w, h), _ = rect
        if w <= 0 or h <= 0:
            continue

        long_edge = float(max(w, h))
        short_edge = float(min(w, h))
        if short_edge < 4:
            continue

        aspect_ratio = long_edge / max(short_edge, 1e-6)
        if not (1.5 <= aspect_ratio <= 22.0):
            continue

        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
        contour_contains_tap = cv2.pointPolygonTest(contour, (float(local_x), float(local_y)), False) >= 0
        if not contour_contains_tap:
            continue

        cx = cx_local + roi_x1
        cy = cy_local + roi_y1
        if cv2.pointPolygonTest(active_polygon, (float(cx), float(cy)), False) < 0:
            continue

        center_distance = float(((cx - tx) * (cx - tx) + (cy - ty) * (cy - ty)) ** 0.5)
        if center_distance > 96:
            continue

        touch_bonus = 26.0 if center_distance <= 28.0 else 0.0
        ratio_penalty = abs(aspect_ratio - expected_ratio) * 13.0
        score = center_distance - touch_bonus - 118.0 - long_edge * 0.48 - min(area, 7000.0) * 0.018 + ratio_penalty
        if score < best_score:
            best_score = score
            best_bbox = (bbox_x + roi_x1, bbox_y + roi_y1, bbox_w, bbox_h)
            best_long_edge = long_edge

    if best_bbox is None or best_long_edge <= 0:
        fallback_masks = [otsu_inv, otsu, adaptive]
        for candidate_mask in fallback_masks:
            mask_u8 = candidate_mask.astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
            if num_labels <= 1:
                continue

            if not (0 <= local_x < labels.shape[1] and 0 <= local_y < labels.shape[0]):
                continue

            label_id = int(labels[local_y, local_x])
            if label_id <= 0:
                continue

            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            if area < 70 or w < 10 or h < 10:
                continue

            long_edge = float(max(w, h))
            short_edge = float(min(w, h))
            if short_edge < 4:
                continue

            aspect_ratio = long_edge / max(short_edge, 1e-6)
            if not (1.4 <= aspect_ratio <= 26.0):
                continue

            cx = roi_x1 + x + (w / 2.0)
            cy = roi_y1 + y + (h / 2.0)
            if cv2.pointPolygonTest(active_polygon, (float(cx), float(cy)), False) < 0:
                continue

            best_bbox = (roi_x1 + x, roi_y1 + y, w, h)
            best_long_edge = long_edge
            break

    if best_bbox is None or best_long_edge <= 0:
        return None
    return best_bbox, best_long_edge


def update_measurement(
    track: Optional[Track],
    state: RadarState,
    now: float,
    logger: logging.Logger,
    profile: LayoutProfile,
    distance_m: float,
) -> Optional[float]:
    """Olcum durumunu gunceller ve gerekiyorsa hiz dondurur."""
    state.update_cooldown(now)

    if state.mode == "cooldown":
        return None

    if track is None:
        return None

    if track.status != "confirmed":
        return None

    current_front_point = track.front_point
    if track.direction_down is False:
        return None

    if track.measurement_state == "armed" and point_crosses_horizontal_segment(
        current_front_point,
        profile.start_p1,
        profile.start_p2,
    ):
        if track.seen_frames < MIN_TRACK_FRAMES:
            return None
        if not track.was_above_start_line:
            return None
        track.measurement_state = "timing"
        track.start_time = now
        state.mode = "timing"
        logger.info(
            "Olcum basladi. profil=%s track=%s front=(%s,%s)",
            profile.name,
            track.track_id,
            current_front_point[0],
            current_front_point[1],
        )
        return None

    if track.measurement_state == "timing" and track.start_time is not None and point_crosses_horizontal_segment(
        current_front_point,
        profile.finish_p1,
        profile.finish_p2,
    ):
        elapsed = now - track.start_time
        if elapsed < MIN_VALID_TIME or elapsed > MAX_VALID_TIME:
            logger.warning("Supheli olcum reddedildi. sure=%.3f sn", elapsed)
            track.measurement_state = "armed"
            track.start_time = None
            return None

        speed_ms = distance_m / elapsed
        speed_kmh = speed_ms * 3.6
        state.last_speed_kmh = speed_kmh
        state.mode = "idle"
        track.last_speed_kmh = speed_kmh
        track.measurement_state = "measured"
        logger.info("Olcum tamamlandi. track=%s sure=%.3f sn hiz=%.2f km/s", track.track_id, elapsed, speed_kmh)
        return speed_kmh

    return None


def update_speed_state(
    track: Optional[Track],
    state: RadarState,
    now: float,
    logger: logging.Logger,
    profile: LayoutProfile,
    distance_m: float,
) -> Optional[float]:
    state.update_cooldown(now)
    if state.mode == "cooldown":
        return None
    if track is None:
        if state.mode == "WAITING_FOR_START" and now <= state.start_hold_until:
            return None
        if state.mode != "cooldown":
            state.mode = "WAITING_FOR_DETECTION"
            state.start_hold_until = 0.0
        return None
    if track.status not in ("confirmed", "lost"):
        if state.mode == "WAITING_FOR_START" and now <= state.start_hold_until:
            return None
        return None
    if track.status == "lost" and state.mode not in ("WAITING_FOR_START", "TRACKING"):
        return None

    current_point = track.front_point
    previous_point = track.previous_front_point or current_point
    if track.status == "lost":
        previous_time = track.last_seen_at
        current_time = now
    else:
        current_time = track.last_seen_at or now
        previous_time = track.previous_seen_at if track.previous_seen_at is not None else current_time
    current_start_line_y = line_y_at_x(profile.start_p1, profile.start_p2, current_point[0])
    current_finish_line_y = line_y_at_x(profile.finish_p1, profile.finish_p2, current_point[0])
    previous_start_line_y = line_y_at_x(profile.start_p1, profile.start_p2, previous_point[0])
    previous_finish_line_y = line_y_at_x(profile.finish_p1, profile.finish_p2, previous_point[0])
    expected_dx = ((profile.finish_p1[0] + profile.finish_p2[0]) - (profile.start_p1[0] + profile.start_p2[0])) / 2.0
    movement_dx = current_point[0] - previous_point[0]
    movement_dy = current_point[1] - previous_point[1]
    effective_dx = movement_dx if abs(movement_dx) >= 2 else track.velocity_front[0]
    effective_dy = movement_dy if abs(movement_dy) >= 2 else track.velocity_front[1]
    direction_matches_profile = True
    if abs(expected_dx) >= 12 and abs(effective_dx) >= 2:
        direction_matches_profile = (effective_dx * expected_dx) >= 0
    start_cross_time = estimate_line_cross_time(
        previous_point,
        previous_time,
        current_point,
        current_time,
        profile.start_p1,
        profile.start_p2,
    )
    finish_cross_time = estimate_line_cross_time(
        previous_point,
        previous_time,
        current_point,
        current_time,
        profile.finish_p1,
        profile.finish_p2,
    )
    start_crossed = start_cross_time is not None
    finish_crossed = finish_cross_time is not None
    current_is_past_start = current_point[1] >= current_start_line_y
    current_is_past_finish = current_point[1] >= current_finish_line_y
    current_near_start = current_point[1] >= (current_start_line_y - START_RECOVERY_NEAR_PX)
    previous_near_start = previous_point[1] >= (previous_start_line_y - START_RECOVERY_NEAR_PX)
    current_is_safely_past_finish = current_point[1] >= (current_finish_line_y + FINISH_RECOVERY_MARGIN_PX)
    previous_near_finish = previous_point[1] >= (previous_finish_line_y - FINISH_RECOVERY_NEAR_PX)
    recovery_gap_ok = (current_time - previous_time) <= MAX_RECOVERY_GAP_SECONDS

    if not direction_matches_profile and state.mode in ("WAITING_FOR_DETECTION", "WAITING_FOR_START"):
        return None

    if state.mode == "WAITING_FOR_DETECTION":
        state.mode = "WAITING_FOR_START"

    if state.mode == "WAITING_FOR_START":
        state.start_hold_until = max(state.start_hold_until, now + START_TRACK_GRACE_SECONDS)
        if start_crossed:
            state.mode = "TRACKING"
            state.start_time = start_cross_time
            state.start_hold_until = 0.0
            logger.info(
                "start_crossed track=%s prev=(%s,%s) curr=(%s,%s)",
                track.track_id,
                previous_point[0],
                previous_point[1],
                current_point[0],
                current_point[1],
            )
        elif track.was_above_start_line and current_is_past_start:
            state.mode = "TRACKING"
            state.start_time = previous_time if previous_time < current_time else current_time
            state.start_hold_until = 0.0
            logger.info(
                "start_recovered_late_confirm track=%s curr=(%s,%s)",
                track.track_id,
                current_point[0],
                current_point[1],
            )
        elif (
            track.status == "lost"
            and recovery_gap_ok
            and track.missed_frames <= 10
            and track.was_above_start_line
            and current_is_past_start
        ):
            state.mode = "TRACKING"
            state.start_time = previous_time if previous_time < current_time else current_time
            state.start_hold_until = 0.0
            logger.info(
                "start_recovered_after_lost track=%s curr=(%s,%s)",
                track.track_id,
                current_point[0],
                current_point[1],
            )
        elif (
            track.status == "lost"
            and recovery_gap_ok
            and track.missed_frames <= 10
            and (track.was_above_start_line or current_near_start or previous_near_start)
            and previous_near_start
            and current_point[1] >= (current_start_line_y - max(2, START_RECOVERY_NEAR_PX // 2))
            and effective_dy >= 4
        ):
            state.mode = "TRACKING"
            state.start_time = previous_time if previous_time < current_time else current_time
            state.start_hold_until = 0.0
            logger.info(
                "start_recovered_near_line track=%s prev=(%s,%s) curr=(%s,%s) status=%s",
                track.track_id,
                previous_point[0],
                previous_point[1],
                current_point[0],
                current_point[1],
                track.status,
            )

    if track.status == "lost":
        if (
            state.mode == "TRACKING"
            and state.start_time is not None
            and recovery_gap_ok
            and current_is_safely_past_finish
            and previous_near_finish
            and effective_dy >= 2
            and direction_matches_profile
        ):
            finish_time = current_time
            elapsed = finish_time - state.start_time
            if MIN_VALID_TIME <= elapsed <= MAX_VALID_TIME:
                speed_ms = distance_m / elapsed
                speed_kmh = speed_ms * 3.6
                state.finish_time = finish_time
                state.last_speed_kmh = speed_kmh
                state.mode = "FINISHED"
                logger.info("finish_recovered_after_lost track=%s curr=(%s,%s)", track.track_id, current_point[0], current_point[1])
                logger.info("Olcum tamamlandi. track=%s sure=%.3f sn hiz=%.2f km/s", track.track_id, elapsed, speed_kmh)
                return speed_kmh
        return None

    if state.mode == "TRACKING" and state.start_time is not None and direction_matches_profile and (finish_crossed or current_is_past_finish):
        finish_time = finish_cross_time if finish_cross_time is not None else current_time
        elapsed = finish_time - state.start_time
        if elapsed < MIN_VALID_TIME or elapsed > MAX_VALID_TIME:
            logger.warning("measurement_rejected_invalid_elapsed track=%s sure=%.3f sn", track.track_id, elapsed)
            state.reset_tracking()
            return None
        speed_ms = distance_m / elapsed
        speed_kmh = speed_ms * 3.6
        state.finish_time = finish_time
        state.last_speed_kmh = speed_kmh
        state.mode = "FINISHED"
        if finish_crossed:
            logger.info("finish_crossed track=%s", track.track_id)
        logger.info("Olcum tamamlandi. track=%s sure=%.3f sn hiz=%.2f km/s", track.track_id, elapsed, speed_kmh)
        return speed_kmh

    return None


def build_violation_frame(frame: np.ndarray, speed_kmh: float, profile_name: str) -> np.ndarray:
    """Kaydedilecek ihlal fotografini hazirlar."""
    violation_frame = frame.copy()
    timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv2.putText(
        violation_frame,
        "IHLAL",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        violation_frame,
        f"Hiz: {speed_kmh:.1f} km/s",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        violation_frame,
        f"Mod: {profile_name}",
        (10, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        violation_frame,
        timestamp_text,
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return violation_frame


def save_violation(frame: np.ndarray, speed_kmh: float, logger: logging.Logger, profile_name: str) -> Path:
    """Ihlal fotografini kaydeder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ihlal_{profile_name.lower()}_{timestamp}_{int(round(speed_kmh))}kmh.jpg"
    output_path = VIOLATION_DIR / filename

    violation_frame = build_violation_frame(frame, speed_kmh, profile_name)
    success = cv2.imwrite(str(output_path), violation_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise RuntimeError(f"Ihlal fotografi kaydedilemedi: {output_path}")

    logger.warning("Ihlal kaydedildi: %s (%.2f km/s)", output_path, speed_kmh)
    return output_path


def start_frame_buffer() -> Deque[BufferFrame]:
    return deque(maxlen=FRAME_BUFFER_SIZE)


def clear_frame_buffer(frame_buffer: Deque[BufferFrame]) -> None:
    frame_buffer.clear()


def crop_plate_region(frame: np.ndarray, track: Optional[Track], profile: LayoutProfile) -> np.ndarray:
    return frame


def append_frame_buffer(frame_buffer: Deque[BufferFrame], frame: np.ndarray, track: Optional[Track], profile: LayoutProfile, now: float) -> None:
    full_frame = crop_plate_region(frame, track, profile)
    gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var()) + (full_frame.shape[0] * full_frame.shape[1] * 0.001)
    frame_buffer.append(BufferFrame(frame=full_frame.copy(), score=score, timestamp=now))


def select_best_plate_frame(frame_buffer: Deque[BufferFrame]) -> Optional[np.ndarray]:
    if not frame_buffer:
        return None
    best = max(frame_buffer, key=lambda item: item.score)
    return best.frame.copy()


def capture_plate_frame_from_buffer(frame_buffer: Deque[BufferFrame]) -> Optional[np.ndarray]:
    return select_best_plate_frame(frame_buffer)


def get_button_rects(frame_width: int, ui_context: Optional[UIContext] = None) -> Dict[str, Tuple[int, int, int, int]]:
    """SOL / SAG / MENU ve pencere kontrol buton alanlarini hesaplar."""
    compact = frame_width <= 340
    side_button_width = 48 if compact else BUTTON_WIDTH
    icon_width = 28 if compact else 34
    gap = 6 if compact else BUTTON_GAP

    left_x1 = 12
    left_x2 = left_x1 + side_button_width
    left2_x1 = left_x2 + BUTTON_GAP
    left2_x2 = left2_x1 + side_button_width

    close_x2 = frame_width - 12
    close_x1 = close_x2 - icon_width
    toggle_x2 = close_x1 - gap
    toggle_x1 = toggle_x2 - icon_width

    button_map = {
        "SOL": (left_x1, BUTTON_Y1, left_x2, BUTTON_Y2),
        "SAG": (left2_x1, BUTTON_Y1, left2_x2, BUTTON_Y2),
        "TOGGLE": (toggle_x1, BUTTON_Y1, toggle_x2, BUTTON_Y2),
        "X": (close_x1, BUTTON_Y1, close_x2, BUTTON_Y2),
    }

    if ui_context is not None and ui_context.calibration_mode == "auto_detected_wait_confirm":
        onay_width = 48 if compact else 62
        dene_width = 64 if compact else 88
        total_width = onay_width + gap + dene_width
        start_x = left2_x2 + max(10, ((toggle_x1 - left2_x2) - total_width) // 2)
        button_map["ONAY"] = (start_x, BUTTON_Y1, start_x + onay_width, BUTTON_Y2)
        dene_x1 = start_x + onay_width + gap
        button_map["DENE"] = (dene_x1, BUTTON_Y1, dene_x1 + dene_width, BUTTON_Y2)
    elif ui_context is not None and ui_context.menu_open and ui_context.active_tool in ("edit_start", "edit_finish"):
        save_width = 84 if compact else 96
        save_x2 = toggle_x1 - 8
        save_x1 = max(left2_x2 + 10, save_x2 - save_width)
        button_map["SUBSAVE"] = (save_x1, BUTTON_Y1, save_x2, BUTTON_Y2)
    else:
        menu_width = 84 if compact else 108
        menu_x1 = left2_x2 + max(10, ((toggle_x1 - left2_x2) - menu_width) // 2)
        button_map["MENU"] = (menu_x1, BUTTON_Y1, menu_x1 + menu_width, BUTTON_Y2)

    return button_map


def get_menu_panel_rects(frame_width: int, frame_height: int) -> Dict[str, Tuple[int, int, int, int]]:
    panel_w = 284
    panel_h = 128
    x1 = max(10, (frame_width - panel_w) // 2)
    y1 = 46
    x2 = x1 + panel_w
    y2 = y1 + panel_h
    button_h = 18
    gap = 4
    rows = [
        ("SARI", "KIRMIZI"),
        ("KALIB", "MESAFE"),
        ("LIMIT", "TESPIT"),
        ("PERF", "ROI"),
        ("VARSAY", "CIK"),
    ]
    rects: Dict[str, Tuple[int, int, int, int]] = {"PANEL": (x1, y1, x2, y2)}
    current_y = y1 + 10
    inner_x1 = x1 + 12
    inner_x2 = x2 - 12
    inner_w = inner_x2 - inner_x1
    col_gap = 8
    col_w = (inner_w - col_gap) // 2
    for left_label, right_label in rows:
        rects[left_label] = (inner_x1, current_y, inner_x1 + col_w, current_y + button_h)
        right_x1 = inner_x1 + col_w + col_gap
        rects[right_label] = (right_x1, current_y, right_x1 + col_w, current_y + button_h)
        current_y += button_h + gap
    return rects


def get_manual_keypad_rects(frame_width: int, frame_height: int) -> Dict[str, Tuple[int, int, int, int]]:
    """Elle gir paneli icin dokunmatik tus alanlarini hesaplar."""
    panel_w = 214
    panel_h = 142
    panel_x1 = (frame_width - panel_w) // 2
    panel_y1 = 50
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h

    key_w = 46
    key_h = 24
    gap = 6
    grid_x = panel_x1 + 16
    grid_y = panel_y1 + 42

    rects: Dict[str, Tuple[int, int, int, int]] = {
        "PANEL": (panel_x1, panel_y1, panel_x2, panel_y2),
        "DISPLAY": (panel_x1 + 16, panel_y1 + 10, panel_x2 - 16, panel_y1 + 32),
    }

    labels = [
        ["7", "8", "9"],
        ["4", "5", "6"],
        ["1", "2", "3"],
        [".", "0", "SIL"],
    ]
    for row_idx, row in enumerate(labels):
        for col_idx, label in enumerate(row):
            x1 = grid_x + col_idx * (key_w + gap)
            y1 = grid_y + row_idx * (key_h + gap)
            rects[label] = (x1, y1, x1 + key_w, y1 + key_h)

    action_y1 = grid_y + 4 * (key_h + gap) + 2
    rects["IPTAL"] = (panel_x1 + 16, action_y1, panel_x1 + 98, action_y1 + 24)
    rects["KAYDET"] = (panel_x2 - 98, action_y1, panel_x2 - 16, action_y1 + 24)
    return rects


def get_mode_selector_rects(frame_width: int, frame_height: int, mode_kind: str) -> Dict[str, Tuple[int, int, int, int]]:
    panel_w = 214
    panel_h = 106
    panel_x1 = (frame_width - panel_w) // 2
    panel_y1 = 62
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h
    rects: Dict[str, Tuple[int, int, int, int]] = {"PANEL": (panel_x1, panel_y1, panel_x2, panel_y2)}
    row_y = panel_y1 + 34
    if mode_kind == "detection":
        keys = ["KARARLI", "DENGELI", "HASSAS"]
        widths = [58, 58, 58]
    else:
        keys = ["NORMAL", "HIZLI"]
        widths = [82, 82]
    gap = 8
    total_w = sum(widths) + gap * (len(widths) - 1)
    start_x = panel_x1 + ((panel_w - total_w) // 2)
    current_x = start_x
    for key, width in zip(keys, widths):
        rects[key] = (current_x, row_y, current_x + width, row_y + 24)
        current_x += width + gap
    rects["IPTAL"] = (panel_x1 + 66, panel_y2 - 28, panel_x2 - 66, panel_y2 - 6)
    return rects


def draw_button(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    is_active: bool,
    ui_context: Optional[UIContext] = None,
) -> None:
    """Overlay uzerinde basit dokunmatik buton cizer."""
    x1, y1, x2, y2 = rect
    display_label = label
    if label == "TOGGLE":
        display_label = "_" if is_active else "[]"

    if label == "X":
        fill_color = (70, 30, 30)
        border_color = (255, 200, 200)
    elif label == "ONAY":
        fill_color = (40, 135, 40)
        border_color = (220, 255, 220)
    elif label == "DENE":
        fill_color = (55, 55, 180)
        border_color = (220, 220, 255)
    elif label == "SUBSAVE":
        display_label = "KAYDET"
        fill_color = (40, 135, 40)
        border_color = (220, 255, 220)
    elif label == "ELLE":
        fill_color = (120, 85, 40) if is_active else (70, 60, 45)
        border_color = (255, 230, 200) if is_active else (190, 170, 150)
    elif label == "TOGGLE":
        fill_color = (40, 40, 75)
        border_color = (210, 210, 255)
    elif label == "KAL" and ui_context is not None:
        display_label = ui_context.calibration_label
        if ui_context.calibration_status == "success":
            fill_color = (40, 135, 40)
            border_color = (220, 255, 220)
        elif ui_context.calibration_status == "error":
            fill_color = (40, 40, 180)
            border_color = (220, 220, 255)
        elif ui_context.calibration_status == "pending":
            fill_color = (60, 110, 170)
            border_color = (220, 240, 255)
        else:
            fill_color = (75, 75, 75)
            border_color = (185, 185, 185)
    else:
        fill_color = (40, 170, 255) if is_active else (60, 60, 60)
        border_color = (255, 255, 255) if is_active else (145, 145, 145)

    cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)

    font_scale = 0.33 if len(display_label) > 14 else (0.37 if len(display_label) > 10 else (0.40 if len(display_label) > 6 else (0.44 if display_label == "[]" else 0.48)))
    text_size, _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    text_x = x1 + ((x2 - x1 - text_size[0]) // 2)
    text_y = y1 + ((y2 - y1 + text_size[1]) // 2) + 1
    cv2.putText(frame, display_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

def get_line_handles(profile: LayoutProfile) -> Dict[str, Tuple[int, int]]:
    return {
        "start_p1": profile.start_p1,
        "start_p2": profile.start_p2,
        "finish_p1": profile.finish_p1,
        "finish_p2": profile.finish_p2,
    }


def point_near_segment(point: Tuple[int, int], seg_a: Tuple[int, int], seg_b: Tuple[int, int], tolerance: int = LINE_HIT_TOLERANCE) -> bool:
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) <= (tolerance * tolerance)
    t = max(0.0, min(1.0, (((px - ax) * dx) + ((py - ay) * dy)) / float(dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) <= (tolerance * tolerance)


def clamp_point(point: Tuple[int, int]) -> Tuple[int, int]:
    return (
        int(max(0, min(FRAME_WIDTH - 1, point[0]))),
        int(max(TOP_BAR_HEIGHT + 2, min(FRAME_HEIGHT - 1, point[1]))),
    )


def update_profile_handle(profile: LayoutProfile, handle_name: str, point: Tuple[int, int]) -> LayoutProfile:
    point = clamp_point(point)
    if handle_name == "start_p1":
        return set_start_line_points(profile, point, profile.start_p2)
    if handle_name == "start_p2":
        return set_start_line_points(profile, profile.start_p1, point)
    if handle_name == "finish_p1":
        return set_finish_line_points(profile, point, profile.finish_p2)
    if handle_name == "finish_p2":
        return set_finish_line_points(profile, profile.finish_p1, point)
    return profile


def move_profile_line(profile: LayoutProfile, line_name: str, delta_x: int, delta_y: int) -> LayoutProfile:
    if line_name == "start":
        p1 = clamp_point((profile.start_p1[0] + delta_x, profile.start_p1[1] + delta_y))
        p2 = clamp_point((profile.start_p2[0] + delta_x, profile.start_p2[1] + delta_y))
        return set_start_line_points(profile, p1, p2)
    if line_name == "finish":
        p1 = clamp_point((profile.finish_p1[0] + delta_x, profile.finish_p1[1] + delta_y))
        p2 = clamp_point((profile.finish_p2[0] + delta_x, profile.finish_p2[1] + delta_y))
        return set_finish_line_points(profile, p1, p2)
    return profile


def handle_drag_events(event: int, x: int, y: int, ui_context: UIContext) -> None:
    if ui_context.active_tool not in ("edit_start", "edit_finish"):
        return

    profile = PROFILES[ui_context.active_profile_name]
    handles = get_line_handles(profile)
    target_handles = ("start_p1", "start_p2") if ui_context.active_tool == "edit_start" else ("finish_p1", "finish_p2")

    if event == cv2.EVENT_LBUTTONDOWN:
        for handle_name in target_handles:
            hx, hy = handles[handle_name]
            if ((x - hx) ** 2 + (y - hy) ** 2) <= (SELECTED_HANDLE_RADIUS * SELECTED_HANDLE_RADIUS):
                ui_context.dragging_handle = handle_name
                ui_context.drag_last_point = (x, y)
                return
        if ui_context.active_tool == "edit_start" and point_near_segment((x, y), profile.start_p1, profile.start_p2):
            ui_context.dragging_line = "start"
            ui_context.drag_last_point = (x, y)
            return
        if ui_context.active_tool == "edit_finish" and point_near_segment((x, y), profile.finish_p1, profile.finish_p2):
            ui_context.dragging_line = "finish"
            ui_context.drag_last_point = (x, y)
            return

    if event == cv2.EVENT_MOUSEMOVE and ui_context.drag_last_point is not None:
        if ui_context.dragging_handle is not None:
            PROFILES[ui_context.active_profile_name] = update_profile_handle(profile, ui_context.dragging_handle, (x, y))
            return
        if ui_context.dragging_line is not None:
            dx = x - ui_context.drag_last_point[0]
            dy = y - ui_context.drag_last_point[1]
            PROFILES[ui_context.active_profile_name] = move_profile_line(profile, ui_context.dragging_line, dx, dy)
            ui_context.drag_last_point = (x, y)
            return

    if event == cv2.EVENT_LBUTTONUP:
        ui_context.dragging_handle = None
        ui_context.dragging_line = None
        ui_context.drag_last_point = None


def handle_mouse_click(event: int, x: int, y: int, flags: int, ui_context: UIContext) -> None:
    """Dokunmatik / fare tiklamalarini ve suruklemeyi yonetir."""
    del flags
    if ui_context.menu_open and ui_context.active_tool in ("edit_start", "edit_finish"):
        handle_drag_events(event, x, y, ui_context)
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if ui_context.manual_distance_mode or ui_context.manual_speed_mode:
        keypad_rects = get_manual_keypad_rects(FRAME_WIDTH, FRAME_HEIGHT)
        clickable_keys = [key for key in keypad_rects.keys() if key not in ("PANEL", "DISPLAY")]
        for key in clickable_keys:
            rect = keypad_rects[key]
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key in "0123456789.":
                    if ui_context.manual_distance_mode:
                        ui_context.append_manual_distance_char(key)
                    else:
                        ui_context.append_manual_speed_char(key)
                elif key == "SIL":
                    if ui_context.manual_distance_mode:
                        ui_context.backspace_manual_distance()
                    else:
                        ui_context.backspace_manual_speed()
                elif key == "IPTAL":
                    if ui_context.manual_distance_mode:
                        ui_context.cancel_manual_distance()
                    else:
                        ui_context.cancel_manual_speed()
                elif key == "KAYDET":
                    if ui_context.manual_distance_mode:
                        ui_context.pending_manual_distance_save = True
                    else:
                        ui_context.pending_manual_speed_save = True
                return
        return

    if ui_context.menu_open and ui_context.active_tool in ("detection_mode", "performance_mode"):
        rects = get_mode_selector_rects(
            FRAME_WIDTH,
            FRAME_HEIGHT,
            "detection" if ui_context.active_tool == "detection_mode" else "performance",
        )
        for key, rect in rects.items():
            if key == "PANEL":
                continue
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key == "IPTAL":
                    ui_context.active_tool = "none"
                elif ui_context.active_tool == "detection_mode":
                    ui_context.set_detection_mode(key)
                else:
                    ui_context.set_performance_mode(key)
                return

    if ui_context.menu_open and ui_context.active_tool == "none":
        for key, rect in get_menu_panel_rects(FRAME_WIDTH, FRAME_HEIGHT).items():
            if key == "PANEL":
                continue
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key == "SARI":
                    ui_context.request_start_line_edit()
                elif key == "KIRMIZI":
                    ui_context.request_finish_line_edit()
                elif key == "KALIB":
                    ui_context.request_calibration()
                elif key == "MESAFE":
                    ui_context.request_manual_distance()
                elif key == "LIMIT":
                    ui_context.request_speed_limit()
                elif key == "TESPIT":
                    ui_context.request_detection_mode_menu()
                elif key == "PERF":
                    ui_context.request_performance_mode_menu()
                elif key == "ROI":
                    ui_context.show_roi_overlay = not ui_context.show_roi_overlay
                elif key == "VARSAY":
                    ui_context.request_reset_active_mode()
                elif key == "CIK":
                    ui_context.close_menu()
                return

    for profile_name, rect in get_button_rects(FRAME_WIDTH, ui_context).items():
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            if profile_name == "TOGGLE":
                if ui_context.is_fullscreen:
                    ui_context.request_window_mode()
                else:
                    ui_context.request_fullscreen_mode()
            elif profile_name == "X":
                ui_context.request_exit()
            elif profile_name == "ONAY":
                if ui_context.calibration_candidate_bbox is not None and ui_context.calibration_candidate_length_px is not None:
                    ui_context.pending_calibration = True
                    ui_context.pending_calibration_source = "auto_confirm"
                    ui_context.calibration_label = "Kalibre..."
            elif profile_name == "SUBSAVE":
                ui_context.request_save()
                ui_context.active_tool = "none"
            elif profile_name == "DENE":
                ui_context.request_calibration_retry()
            elif profile_name == "MENU":
                ui_context.open_menu()
            else:
                ui_context.select_profile(profile_name)
            return

    if y <= TOP_BAR_HEIGHT:
        return

    if ui_context.calibration_mode == "auto_wait_tap":
        ui_context.calibration_feedback_point = (x, y)
        ui_context.calibration_feedback_until = time.time() + 2.0
        ui_context.pending_calibration = True
        ui_context.pending_calibration_source = "auto_detect"
        ui_context.calibration_first_point = (x, y)
        ui_context.calibration_label = "Kalibre..."
        return

    if ui_context.calibration_mode == "manual_point_1":
        ui_context.calibration_first_point = (x, y)
        ui_context.calibration_feedback_point = (x, y)
        ui_context.calibration_feedback_until = time.time() + 2.0
        ui_context.calibration_label = "2. Kose"
        ui_context.calibration_mode = "manual_point_2"
        return

    if ui_context.calibration_mode == "manual_point_2":
        ui_context.calibration_second_point = (x, y)
        ui_context.calibration_feedback_point = (x, y)
        ui_context.calibration_feedback_until = time.time() + 2.0
        ui_context.pending_calibration = True
        ui_context.pending_calibration_source = "manual_points"
        ui_context.calibration_label = "Kalibre..."
        return


def draw_overlay(
    frame: np.ndarray,
    track: Optional[Track],
    fps: float,
    state: RadarState,
    warmup_remaining: int,
    roi_mask: np.ndarray,
    profile: LayoutProfile,
    ui_context: UIContext,
    distance_m: float,
    upstream_offset_px: int,
) -> np.ndarray:
    """Canli goruntu ustune bilgi cizer."""
    output = frame.copy()
    has_motion = track is not None and track.missed_frames == 0
    button_rects = get_button_rects(frame.shape[1], ui_context)

    # Ust bar
    cv2.rectangle(output, (0, 0), (frame.shape[1] - 1, TOP_BAR_HEIGHT), (24, 24, 24), -1)
    cv2.line(output, (0, TOP_BAR_HEIGHT), (frame.shape[1] - 1, TOP_BAR_HEIGHT), (80, 80, 80), 1)

    # Cizgiler
    if ui_context.show_roi_overlay:
        draw_roi_visual_overlay(output, profile, upstream_offset_px)
    cv2.line(output, profile.start_p1, profile.start_p2, (0, 255, 255), 2)
    cv2.line(output, profile.finish_p1, profile.finish_p2, (0, 0, 255), 2)
    left_guide_top, left_guide_bottom, right_guide_top, right_guide_bottom = get_visual_guide_lines(profile, frame.shape[0])
    cv2.line(output, left_guide_top, left_guide_bottom, (255, 0, 0), 2)
    cv2.line(output, right_guide_top, right_guide_bottom, (255, 0, 0), 2)

    start_label_x = min(profile.start_p1[0], profile.start_p2[0]) + 10
    start_label_y = min(profile.start_p1[1], profile.start_p2[1]) - 8
    finish_label_x = min(profile.finish_p1[0], profile.finish_p2[0]) + 10
    finish_label_y = min(profile.finish_p1[1], profile.finish_p2[1]) - 8
    cv2.putText(output, "Baslangic", (start_label_x, start_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(output, "Bitis", (finish_label_x, finish_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if track is not None:
        x, y, w, h = track.bbox
        box_color = (0, 255, 0) if track.status == "confirmed" else (150, 150, 150)
        cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
        cv2.circle(output, track.center, 4, (255, 0, 0), -1)
        cv2.circle(output, track.front_point, 4, (0, 165, 255), -1)

    if ui_context.calibration_candidate_bbox is not None:
        x, y, w, h = ui_context.calibration_candidate_bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (80, 255, 80), 2)
        cv2.putText(
            output,
            "Nesne algilandi",
            (x, max(TOP_BAR_HEIGHT + 16, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 255, 80),
            1,
            cv2.LINE_AA,
        )

    for profile_name, rect in button_rects.items():
        if profile_name == "TOGGLE":
            is_active = ui_context.is_fullscreen
        elif profile_name == "MENU":
            is_active = ui_context.menu_open
        else:
            is_active = ui_context.active_profile_name == profile_name
        draw_button(output, rect, profile_name, is_active, ui_context)

    status_text = "Durum: Menu" if ui_context.menu_open else ("Durum: Hareket var" if has_motion else "Durum: Bosta")
    speed_text = f"Son hiz: {state.last_speed_kmh:.1f} km/s"
    fps_text = f"FPS: {fps:.1f}"
    takip_text = "Takip: yok"
    if track is not None:
        if track.measurement_state == "timing":
            takip_text = f"Takip: olculuyor #{track.track_id}"
        elif track.status == "confirmed":
            takip_text = f"Takip: aktif #{track.track_id}"
        elif track.status == "lost":
            takip_text = f"Takip: kayip #{track.track_id}"
        else:
            takip_text = f"Takip: aday #{track.track_id}"

    if profile.name == "SOL":
        text_anchor_x = frame.shape[1] - 152
        text_y_base = frame.shape[0] - 70
    else:
        text_anchor_x = 12
        text_y_base = frame.shape[0] - 70
    cv2.putText(output, status_text, (text_anchor_x, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, takip_text, (text_anchor_x, text_y_base + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (190, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(output, speed_text, (text_anchor_x, text_y_base + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"Limit: {SPEED_LIMIT_KMH:.1f} km/s", (text_anchor_x, text_y_base + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"Mesafe: {distance_m:.0f} m", (text_anchor_x, text_y_base + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1, cv2.LINE_AA)
    fps_x = 10 if profile.name == "SOL" else frame.shape[1] - 78
    cv2.putText(output, fps_text, (fps_x, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    if warmup_remaining > 0 and not ui_context.menu_open:
        cv2.putText(output, f"Isinma: {warmup_remaining} kare", (text_anchor_x, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 255), 1, cv2.LINE_AA)

    if ui_context.calibration_mode == "auto_wait_tap":
        guide_text = "Kalibrasyon icin nesneye dokun"
        cv2.putText(
            output,
            guide_text,
            (12, frame.shape[0] - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 255, 255),
            1,
            cv2.LINE_AA,
        )
    elif ui_context.calibration_mode == "auto_detected_wait_confirm":
        cv2.putText(
            output,
            "Nesne algilandi: Onayla veya Yeniden Dene",
            (12, frame.shape[0] - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (80, 255, 255),
            1,
            cv2.LINE_AA,
        )
    elif ui_context.calibration_mode in ("manual_point_1", "manual_point_2"):
        guide_text = "Kalibrasyon icin 1. koseye dokun" if ui_context.calibration_mode == "manual_point_1" else "Kalibrasyon icin 2. koseye dokun"
        cv2.putText(
            output,
            guide_text,
            (12, frame.shape[0] - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if ui_context.calibration_first_point is not None:
        cv2.circle(output, ui_context.calibration_first_point, 2, (0, 255, 255), 1)
        cv2.drawMarker(output, ui_context.calibration_first_point, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 4, 1, cv2.LINE_AA)
    if ui_context.calibration_second_point is not None:
        cv2.circle(output, ui_context.calibration_second_point, 2, (0, 255, 255), 1)
        cv2.drawMarker(output, ui_context.calibration_second_point, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 4, 1, cv2.LINE_AA)
    if ui_context.calibration_first_point is not None and ui_context.calibration_second_point is not None:
        cv2.line(output, ui_context.calibration_first_point, ui_context.calibration_second_point, (0, 255, 255), 2)
    elif (
        ui_context.calibration_feedback_point is not None
        and time.time() <= ui_context.calibration_feedback_until
    ):
        px, py = ui_context.calibration_feedback_point
        cv2.circle(output, (px, py), 3, (255, 255, 255), 1)
        cv2.circle(output, (px, py), 1, (0, 255, 255), 1)
        cv2.drawMarker(output, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)

    if ui_context.manual_distance_mode or ui_context.manual_speed_mode:
        keypad_rects = get_manual_keypad_rects(frame.shape[1], frame.shape[0])
        px1, py1, px2, py2 = keypad_rects["PANEL"]
        is_distance_mode = ui_context.manual_distance_mode
        cv2.putText(output, "Elle Mesafe Gir" if is_distance_mode else "Hiz Limiti Gir", (px1 + 16, py1 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        dx1, dy1, dx2, dy2 = keypad_rects["DISPLAY"]
        cv2.rectangle(output, (dx1, dy1), (dx2, dy2), (34, 34, 40), -1)
        cv2.rectangle(output, (dx1, dy1), (dx2, dy2), (170, 170, 180), 1)
        display_text = ui_context.manual_distance_text if is_distance_mode else ui_context.manual_speed_text
        display_text = display_text if display_text else "0"
        cv2.putText(output, f"{display_text}{' m' if is_distance_mode else ' km/s'}", (dx1 + 8, dy2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
        status_is_error = ui_context.manual_distance_status == "error" if is_distance_mode else ui_context.manual_speed_status == "error"
        if status_is_error:
            cv2.putText(output, "Gecersiz mesafe" if is_distance_mode else "Gecersiz limit", (dx1 + 8, py2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 80, 255), 1, cv2.LINE_AA)

        for key, rect in keypad_rects.items():
            if key in ("PANEL", "DISPLAY"):
                continue
            x1, y1, x2, y2 = rect
            if key == "KAYDET":
                fill = (40, 135, 40)
                border = (220, 255, 220)
            elif key == "IPTAL":
                fill = (70, 30, 30)
                border = (255, 220, 220)
            else:
                fill = (62, 62, 72)
                border = (180, 180, 190)
            cv2.rectangle(output, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(output, (x1, y1), (x2, y2), border, 1)
            label = "<" if key == "SIL" else key
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_x = x1 + ((x2 - x1 - text_size[0]) // 2)
            text_y = y1 + ((y2 - y1 + text_size[1]) // 2)
            cv2.putText(output, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if ui_context.menu_open and ui_context.active_tool == "none":
        menu_rects = get_menu_panel_rects(frame.shape[1], frame.shape[0])
        x1, y1, x2, y2 = menu_rects["PANEL"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (28, 28, 28), -1)
        cv2.rectangle(output, (x1, y1), (x2, y2), (170, 170, 170), 1)
        cv2.putText(output, "Menu", (x1 + 8, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        label_map = {
            "SARI": "Sari Cizgi Ayarla",
            "KIRMIZI": "Kirmizi Cizgi Ayarla",
            "KALIB": "Kalibrasyon Yap",
            "MESAFE": "Mesafe Gir",
            "LIMIT": f"Hiz Limiti: {SPEED_LIMIT_KMH:.0f}",
            "TESPIT": f"Tespit: {ui_context.detection_mode.title()}",
            "PERF": f"Performans: {'Hizli' if ui_context.performance_mode == 'HIZLI' else 'Normal'}",
            "ROI": "Tespit Alani Kapat" if ui_context.show_roi_overlay else "Tespit Alani Ac",
            "VARSAY": "Varsayilana Don",
            "CIK": "Menuden Cik",
        }
        for key, label in label_map.items():
            rect = menu_rects[key]
            active = (
                (key == "SARI" and ui_context.active_tool == "edit_start")
                or (key == "KIRMIZI" and ui_context.active_tool == "edit_finish")
                or (key == "KALIB" and ui_context.active_tool == "calibration")
                or (key == "MESAFE" and ui_context.manual_distance_mode)
                or (key == "LIMIT" and ui_context.manual_speed_mode)
                or (key == "TESPIT" and ui_context.active_tool == "detection_mode")
                or (key == "PERF" and ui_context.active_tool == "performance_mode")
                or (key == "ROI" and ui_context.show_roi_overlay)
            )
            draw_button(output, rect, label, active, ui_context)

    if ui_context.menu_open and ui_context.active_tool in ("detection_mode", "performance_mode"):
        mode_kind = "detection" if ui_context.active_tool == "detection_mode" else "performance"
        mode_rects = get_mode_selector_rects(frame.shape[1], frame.shape[0], mode_kind)
        x1, y1, x2, y2 = mode_rects["PANEL"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (28, 28, 28), -1)
        cv2.rectangle(output, (x1, y1), (x2, y2), (170, 170, 170), 1)
        title = "Tespit Modu" if mode_kind == "detection" else "Performans Modu"
        cv2.putText(output, title, (x1 + 8, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        option_keys = ("KARARLI", "DENGELI", "HASSAS") if mode_kind == "detection" else ("NORMAL", "HIZLI")
        current_value = ui_context.detection_mode if mode_kind == "detection" else ui_context.performance_mode
        for key in option_keys:
            draw_button(output, mode_rects[key], key.title(), key == current_value, ui_context)
        draw_button(output, mode_rects["IPTAL"], "Iptal", False, ui_context)

    if ui_context.menu_open and ui_context.active_tool in ("edit_start", "edit_finish"):
        handles = get_line_handles(profile)
        target_handles = ("start_p1", "start_p2") if ui_context.active_tool == "edit_start" else ("finish_p1", "finish_p2")
        for handle_name in target_handles:
            hx, hy = handles[handle_name]
            selected = ui_context.dragging_handle == handle_name
            radius = SELECTED_HANDLE_RADIUS if selected else HANDLE_RADIUS
            fill_color = (0, 220, 255) if "start" in handle_name else (0, 0, 255)
            cv2.circle(output, (hx, hy), radius, (255, 255, 255), 1)
            cv2.circle(output, (hx, hy), max(2, radius - 3), fill_color, -1)

    return output


def draw_web_overlay(
    frame: np.ndarray,
    track: Optional[Track],
    fps: float,
    state: RadarState,
    warmup_remaining: int,
    profile: LayoutProfile,
    distance_m: float,
    show_roi_overlay: bool,
    upstream_offset_px: int,
) -> np.ndarray:
    """Web stream icin hafif overlay. Gereksiz menu/UI cizmez."""
    output = frame.copy()
    has_motion = track is not None and track.missed_frames == 0

    if show_roi_overlay:
        draw_roi_visual_overlay(output, profile, upstream_offset_px)
    cv2.line(output, profile.start_p1, profile.start_p2, (0, 255, 255), 2)
    cv2.line(output, profile.finish_p1, profile.finish_p2, (0, 0, 255), 2)
    left_guide_top, left_guide_bottom, right_guide_top, right_guide_bottom = get_visual_guide_lines(profile, frame.shape[0])
    cv2.line(output, left_guide_top, left_guide_bottom, (255, 0, 0), 2)
    cv2.line(output, right_guide_top, right_guide_bottom, (255, 0, 0), 2)

    if track is not None:
        x, y, w, h = track.bbox
        box_color = (0, 255, 0) if track.status == "confirmed" else (150, 150, 150)
        cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
        cv2.circle(output, track.front_point, 3, (0, 165, 255), -1)
        cv2.circle(output, track.center, 3, (255, 0, 0), -1)

    status_text = "Durum: Hareket var" if has_motion else "Durum: Bosta"
    takip_text = "Takip: yok"
    if track is not None:
        if track.measurement_state == "timing":
            takip_text = f"Takip: olculuyor #{track.track_id}"
        elif track.status == "confirmed":
            takip_text = f"Takip: aktif #{track.track_id}"
        elif track.status == "lost":
            takip_text = f"Takip: kayip #{track.track_id}"
        else:
            takip_text = f"Takip: aday #{track.track_id}"

    text_x = 10 if profile.name == "SAG" else frame.shape[1] - 148
    text_y = frame.shape[0] - 58
    cv2.putText(output, status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, takip_text, (text_x, text_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (190, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"Son hiz: {state.last_speed_kmh:.1f} km/s", (text_x, text_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"Limit: {SPEED_LIMIT_KMH:.1f} km/s", (text_x, text_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"Mesafe: {distance_m:.0f} m", (text_x, text_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1, cv2.LINE_AA)
    fps_x = frame.shape[1] - 78 if profile.name == "SAG" else 10
    cv2.putText(output, f"FPS: {fps:.1f}", (fps_x, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    if warmup_remaining > 0:
        cv2.putText(output, f"Isinma: {warmup_remaining} kare", (10, TOP_BAR_HEIGHT + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 255), 1, cv2.LINE_AA)

    return output


def maybe_publish_web_frame(
    web_state: Optional[WebStreamState],
    frame: np.ndarray,
    track: Optional[Track],
    fps: float,
    state: RadarState,
    warmup_remaining: int,
    profile: LayoutProfile,
    distance_m: float,
    show_roi_overlay: bool,
    upstream_offset_px: int,
    now: float,
) -> None:
    if web_state is None or not ENABLE_WEB_STREAM:
        return
    min_interval = 1.0 / max(WEB_STREAM_FPS_LIMIT, 1.0)
    status_text = "Hareket var" if (track is not None and track.missed_frames == 0) else "Bosta"
    with web_state.lock:
        if (now - web_state.last_enqueue_at) < min_interval:
            return
        web_state.pending_job = {
            "frame": frame,
            "track": track,
            "fps": fps,
            "state_speed_kmh": state.last_speed_kmh,
            "warmup_remaining": warmup_remaining,
            "profile": profile,
            "distance_m": distance_m,
            "show_roi_overlay": show_roi_overlay,
            "upstream_offset_px": upstream_offset_px,
            "status_text": status_text,
            "now": now,
        }
        web_state.last_enqueue_at = now


def apply_web_action(ui_context: UIContext, action: str) -> None:
    action = action.upper()
    if action == "SOL":
        ui_context.select_profile("SOL")
    elif action == "SAG":
        ui_context.select_profile("SAG")
    elif action == "MENU":
        if ui_context.menu_open:
            ui_context.close_menu()
        else:
            ui_context.open_menu()
    elif action == "SARI":
        ui_context.request_start_line_edit()
    elif action == "KIRMIZI":
        ui_context.request_finish_line_edit()
    elif action == "KALIB":
        ui_context.request_calibration()
    elif action == "ONAY":
        if ui_context.calibration_candidate_bbox is not None and ui_context.calibration_candidate_length_px is not None:
            ui_context.pending_calibration = True
            ui_context.pending_calibration_source = "auto_confirm"
            ui_context.calibration_label = "Kalibre..."
    elif action == "DENE":
        ui_context.request_calibration_retry()
    elif action == "MESAFE":
        ui_context.request_manual_distance()
    elif action == "LIMIT":
        ui_context.request_speed_limit()
    elif action == "TESPITMODE":
        ui_context.request_detection_mode_menu()
    elif action == "PERFMODE":
        ui_context.request_performance_mode_menu()
    elif action in ("KARARLI", "DENGELI", "HASSAS"):
        ui_context.set_detection_mode(action)
    elif action in ("NORMAL", "HIZLI"):
        ui_context.set_performance_mode(action)
    elif action == "ROI":
        ui_context.show_roi_overlay = not ui_context.show_roi_overlay
    elif action == "VARSAY":
        ui_context.request_reset_active_mode()
    elif action == "SAVEWEB":
        ui_context.request_save()
        ui_context.menu_open = True
        ui_context.active_tool = "none"
        ui_context.dragging_handle = None
        ui_context.dragging_line = None
        ui_context.drag_last_point = None
    elif action == "CANCELWEB":
        ui_context.menu_open = True
        ui_context.active_tool = "none"
        ui_context.dragging_handle = None
        ui_context.dragging_line = None
        ui_context.drag_last_point = None
        ui_context.manual_distance_mode = False
        ui_context.manual_speed_mode = False
    elif action == "CIK":
        ui_context.close_menu()
    elif action == "TOGGLE":
        if ui_context.is_fullscreen:
            ui_context.request_window_mode()
        else:
            ui_context.request_fullscreen_mode()
    elif action == "X":
        ui_context.request_exit()


def create_web_app(
    web_state: WebStreamState,
    ui_context: UIContext,
    ui_lock: threading.Lock,
    distance_state: Dict[str, float],
) -> Any:
    if Flask is None or Response is None or render_template_string is None or redirect is None or request is None or url_for is None:
        raise RuntimeError("Flask kurulu degil. python3-flask paketini kurun.")

    app = Flask(__name__)

    page = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RadarPi Stream</title>
  <style>
    html, body { background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; padding:0; min-height:100vh; }
    body { text-align:center; }
    .shell { padding:8px 8px 10px; }
    .shell:fullscreen, .shell:-webkit-full-screen {
      width:100vw; height:100vh; padding:0; box-sizing:border-box;
      display:flex; flex-direction:column; gap:0; background:#111;
      position:relative; overflow:hidden;
    }
    img {
      display:block;
      width:100%;
      max-width:none;
      border:1px solid #333;
      border-radius:8px;
      margin:0 auto;
      object-fit:contain;
      max-height:calc(100vh - 74px);
      background:#000;
    }
    .row { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin:0 0 8px; }
    .row form { margin:0; }
    .stack { display:flex; flex-direction:column; gap:8px; align-items:center; }
    button {
      background:#2d3f75; color:#fff; border:1px solid #9fb4ff; border-radius:6px;
      padding:8px 12px; min-width:84px; cursor:pointer; font-size:15px;
    }
    button.active { background:#c8aa2b; color:#111; border-color:#fff2b0; }
    button.warn { background:#5c2d2d; border-color:#ffcaca; }
    button.dim { background:#2d2d2d; border-color:#777; }
    .panel {
      background:#1a1a1a; border:1px solid #444; border-radius:8px; padding:10px; margin:0 0 8px;
      max-width:960px; margin-left:auto; margin-right:auto;
    }
    .panel h3 { margin:0 0 8px; font-size:16px; }
    .panel p { margin:0 0 8px; color:#bbb; font-size:13px; }
    .panel input {
      background:#0f0f0f; color:#fff; border:1px solid #666; border-radius:6px;
      padding:8px 10px; font-size:16px; width:140px; text-align:center;
    }
    .viewer { position:relative; width:100%; max-width:960px; margin:0 auto; }
    .shell:fullscreen .viewer, .shell:-webkit-full-screen .viewer {
      flex:1 1 auto;
      width:100%;
      height:100vh;
      max-width:none;
      display:flex;
      align-items:center;
      justify-content:center;
      margin:0;
    }
    .shell:fullscreen img, .shell:-webkit-full-screen img {
      width:auto;
      height:100%;
      max-width:100%;
      max-height:100%;
      object-fit:contain;
    }
    .shell:fullscreen > .row:first-child, .shell:-webkit-full-screen > .row:first-child {
      position:absolute;
      top:8px;
      left:8px;
      right:8px;
      z-index:20;
      margin:0;
      pointer-events:none;
    }
    .shell:fullscreen > .row:first-child form, .shell:-webkit-full-screen > .row:first-child form,
    .shell:fullscreen > .row:first-child button, .shell:-webkit-full-screen > .row:first-child button {
      pointer-events:auto;
    }
    .shell:fullscreen .panel, .shell:-webkit-full-screen .panel,
    .shell:fullscreen > .row:nth-child(2), .shell:-webkit-full-screen > .row:nth-child(2) {
      position:absolute;
      top:58px;
      left:8px;
      right:8px;
      z-index:20;
      margin:0 auto;
      max-width:960px;
    }
    .viewer canvas {
      position:absolute; inset:0; width:100%; height:100%; touch-action:none;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="row">
      <form method="post" action="/action/SOL"><button class="{{ 'active' if profile == 'SOL' else '' }}">SOL</button></form>
      <form method="post" action="/action/SAG"><button class="{{ 'active' if profile == 'SAG' else '' }}">SAG</button></form>
      <form method="post" action="/action/MENU"><button class="{{ 'active' if menu_open else '' }}">MENU</button></form>
      <button type="button" id="fullscreenBtn" class="dim">Tam Ekran</button>
    </div>
    {% if menu_open and active_tool == 'none' %}
    <div class="row">
      <form method="post" action="/action/SARI"><button>Sari Cizgi</button></form>
      <form method="post" action="/action/KIRMIZI"><button>Kirmizi Cizgi</button></form>
      <form method="post" action="/action/KALIB"><button>Kalibrasyon</button></form>
      <form method="post" action="/action/MESAFE"><button>Mesafe Gir</button></form>
      <form method="post" action="/action/LIMIT"><button>Hiz Limiti: {{ speed_limit_kmh|round(0)|int }}</button></form>
      <form method="post" action="/action/TESPITMODE"><button>Tespit: {{ detection_mode_title }}</button></form>
      <form method="post" action="/action/PERFMODE"><button>Performans: {{ performance_mode_title }}</button></form>
      <form method="post" action="/action/ROI"><button>{{ 'Tespit Alani Kapat' if show_roi_overlay else 'Tespit Alani Ac' }}</button></form>
      <form method="post" action="/action/VARSAY"><button>Varsayilan</button></form>
      <form method="post" action="/action/CIK"><button class="dim">Kapat</button></form>
    </div>
    {% endif %}
    {% if active_tool == 'manual_distance' %}
    <div class="panel">
      <h3>Mesafe Gir</h3>
      <p>Sari ve kirmizi cizgi arasi gercek mesafeyi metre olarak girin.</p>
      <form class="stack" method="post" action="/set_distance">
        <input type="number" step="0.1" min="0.1" name="distance" value="{{ distance_m }}">
        <div class="row">
          <button type="submit">Kaydet</button>
          <button type="submit" formaction="/action/CANCELWEB" formmethod="post" class="dim">Iptal</button>
        </div>
      </form>
    </div>
    {% elif active_tool == 'manual_speed_limit' %}
    <div class="panel">
      <h3>Hiz Limiti Gir</h3>
      <p>Limit degerini km/s olarak girin. Bu deger asilirsa fotograf tetiklenir.</p>
      <form class="stack" method="post" action="/set_speed_limit">
        <input type="number" step="0.1" min="1" name="speed_limit" value="{{ speed_limit_kmh }}">
        <div class="row">
          <button type="submit">Kaydet</button>
          <button type="submit" formaction="/action/CANCELWEB" formmethod="post" class="dim">Iptal</button>
        </div>
      </form>
    </div>
    {% elif active_tool in ['edit_start', 'edit_finish'] %}
    <div class="panel">
      <h3>{{ 'Sari Cizgi Ayarla' if active_tool == 'edit_start' else 'Kirmizi Cizgi Ayarla' }}</h3>
      <p>Uclari veya cizginin govdesini parmaginizla surukleyin. Sonra Kaydet'e basin.</p>
      <div class="row">
        <form method="post" action="/action/SAVEWEB"><button>Kaydet</button></form>
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
    </div>
    {% elif active_tool == 'calibration' %}
    <div class="panel">
      <h3>Kalibrasyon</h3>
      {% if calibration_mode == 'auto_wait_tap' %}
      <p>Referans nesneye dokunun. Sistem nesneyi bulmaya calisacak.</p>
      <div class="row">
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
      {% elif calibration_mode == 'auto_detected_wait_confirm' %}
      <p>Nesne bulundu. Onayla ile mesafeyi hesapla, olmazsa yeniden dene.</p>
      <div class="row">
        <form method="post" action="/action/ONAY"><button>Onayla</button></form>
        <form method="post" action="/action/DENE"><button>Yeniden Dene</button></form>
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
      {% elif calibration_mode in ['manual_point_1', 'manual_point_2'] %}
      <p>{{ '1. koseye dokunun' if calibration_mode == 'manual_point_1' else '2. koseye dokunun' }}</p>
      <div class="row">
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
      {% else %}
      <p>Kalibrasyon hazir.</p>
      <div class="row">
        <form method="post" action="/action/CANCELWEB"><button class="dim">Kapat</button></form>
      </div>
      {% endif %}
    </div>
    {% elif active_tool == 'detection_mode' %}
    <div class="panel">
      <h3>Tespit Modu</h3>
      <p>Yakalama karakterini secin. Hassas daha erken yakalar, Kararli daha secicidir.</p>
      <div class="row">
        <form method="post" action="/action/KARARLI"><button class="{{ 'active' if detection_mode == 'KARARLI' else '' }}">Kararli</button></form>
        <form method="post" action="/action/DENGELI"><button class="{{ 'active' if detection_mode == 'DENGELI' else '' }}">Dengeli</button></form>
        <form method="post" action="/action/HASSAS"><button class="{{ 'active' if detection_mode == 'HASSAS' else '' }}">Hassas</button></form>
      </div>
      <div class="row">
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
    </div>
    {% elif active_tool == 'performance_mode' %}
    <div class="panel">
      <h3>Performans Modu</h3>
      <p>Hizli mod daha az islem yapar, Normal mod daha guvenli takip sunar.</p>
      <div class="row">
        <form method="post" action="/action/NORMAL"><button class="{{ 'active' if performance_mode == 'NORMAL' else '' }}">Normal</button></form>
        <form method="post" action="/action/HIZLI"><button class="{{ 'active' if performance_mode == 'HIZLI' else '' }}">Hizli</button></form>
      </div>
      <div class="row">
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
    </div>
    {% endif %}
    <div class="viewer">
      <img id="stream" src="/video_feed" alt="RadarPi stream">
      {% if active_tool in ['edit_start', 'edit_finish', 'calibration'] %}
      <canvas id="editorCanvas"></canvas>
      {% endif %}
    </div>
  </div>
  <script>
    (function () {
      const shell = document.querySelector(".shell");
      const btn = document.getElementById("fullscreenBtn");
      if (!shell || !btn) return;

      async function toggleFullscreen() {
        try {
          const active = document.fullscreenElement || document.webkitFullscreenElement;
          if (active) {
            if (document.exitFullscreen) await document.exitFullscreen();
            else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
            return;
          }
          if (shell.requestFullscreen) await shell.requestFullscreen();
          else if (shell.webkitRequestFullscreen) shell.webkitRequestFullscreen();
        } catch (err) {
          console.warn("Tam ekran acilamadi:", err);
        }
      }

      function syncLabel() {
        const active = document.fullscreenElement || document.webkitFullscreenElement;
        btn.textContent = active ? "Kapat" : "Tam Ekran";
      }

      btn.addEventListener("click", toggleFullscreen);
      document.addEventListener("fullscreenchange", syncLabel);
      document.addEventListener("webkitfullscreenchange", syncLabel);
      syncLabel();
    })();
  </script>
  {% if active_tool in ['edit_start', 'edit_finish', 'calibration'] %}
  <script>
    const FRAME_W = {{ frame_width }};
    const FRAME_H = {{ frame_height }};
    const ACTIVE_TOOL = "{{ active_tool }}";
    const img = document.getElementById("stream");
    const canvas = document.getElementById("editorCanvas");
    const ctx = canvas.getContext("2d");
    let layout = null;
    let drag = null;
    let lastSent = 0;
    let imageBox = {x: 0, y: 0, width: 1, height: 1};

    function resizeCanvas() {
      const rect = img.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      const frameAspect = FRAME_W / FRAME_H;
      const rectAspect = rect.width / Math.max(rect.height, 1);
      let drawWidth = rect.width;
      let drawHeight = rect.height;
      if (rectAspect > frameAspect) {
        drawHeight = rect.height;
        drawWidth = drawHeight * frameAspect;
      } else {
        drawWidth = rect.width;
        drawHeight = drawWidth / frameAspect;
      }
      imageBox = {
        x: (rect.width - drawWidth) / 2,
        y: (rect.height - drawHeight) / 2,
        width: drawWidth,
        height: drawHeight,
      };
      draw();
    }

    function toCanvas(pt) {
      return {
        x: imageBox.x + pt[0] * (imageBox.width / FRAME_W),
        y: imageBox.y + pt[1] * (imageBox.height / FRAME_H),
      };
    }

    function toFrame(x, y) {
      const clampedX = Math.max(imageBox.x, Math.min(imageBox.x + imageBox.width, x));
      const clampedY = Math.max(imageBox.y, Math.min(imageBox.y + imageBox.height, y));
      return {
        x: Math.max(0, Math.min(FRAME_W - 1, Math.round((clampedX - imageBox.x) * (FRAME_W / imageBox.width)))),
        y: Math.max({{ top_bar_height + 2 }}, Math.min(FRAME_H - 1, Math.round((clampedY - imageBox.y) * (FRAME_H / imageBox.height)))),
      };
    }

    function near(a, b, r=18) {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      return (dx*dx + dy*dy) <= r*r;
    }

    function pointNearSegment(p, a, b, tolerance=16) {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      if (dx === 0 && dy === 0) return near(p, a, tolerance);
      let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / ((dx*dx) + (dy*dy));
      t = Math.max(0, Math.min(1, t));
      const proj = {x: a.x + t*dx, y: a.y + t*dy};
      return near(p, proj, tolerance);
    }

    async function loadLayout() {
      const res = await fetch("/api/layout");
      layout = await res.json();
      draw();
    }

    function draw() {
      if (!layout) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const start1 = toCanvas(layout.start_p1);
      const start2 = toCanvas(layout.start_p2);
      const finish1 = toCanvas(layout.finish_p1);
      const finish2 = toCanvas(layout.finish_p2);
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#ffe600";
      ctx.beginPath(); ctx.moveTo(start1.x, start1.y); ctx.lineTo(start2.x, start2.y); ctx.stroke();
      ctx.strokeStyle = "#ff4040";
      ctx.beginPath(); ctx.moveTo(finish1.x, finish1.y); ctx.lineTo(finish2.x, finish2.y); ctx.stroke();
      ctx.strokeStyle = "#2f7dff";
      ctx.beginPath(); ctx.moveTo(start1.x, start1.y); ctx.lineTo(finish1.x, finish1.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(start2.x, start2.y); ctx.lineTo(finish2.x, finish2.y); ctx.stroke();

      const handles = ACTIVE_TOOL === "edit_start"
        ? [{k:"start_p1", p:start1}, {k:"start_p2", p:start2}]
        : [{k:"finish_p1", p:finish1}, {k:"finish_p2", p:finish2}];
      for (const h of handles) {
        ctx.fillStyle = "#ffffff";
        ctx.beginPath(); ctx.arc(h.p.x, h.p.y, 9, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = ACTIVE_TOOL === "edit_start" ? "#ffe600" : "#ff4040";
        ctx.beginPath(); ctx.arc(h.p.x, h.p.y, 5, 0, Math.PI*2); ctx.fill();
      }
    }

    async function pushLayout() {
      const now = Date.now();
      if ((now - lastSent) < 50) return;
      lastSent = now;
      await fetch("/api/layout", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(layout)
      });
    }

    function pointerDown(ev) {
      if (!layout) return;
      const rect = canvas.getBoundingClientRect();
      const p = {x: ev.clientX - rect.left, y: ev.clientY - rect.top};
      if (ACTIVE_TOOL === "calibration") {
        const fp = toFrame(p.x, p.y);
        fetch("/api/calibration_tap", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({x: fp.x, y: fp.y})
        }).finally(() => {
          window.setTimeout(() => window.location.reload(), 350);
        });
        return;
      }
      const start1 = toCanvas(layout.start_p1);
      const start2 = toCanvas(layout.start_p2);
      const finish1 = toCanvas(layout.finish_p1);
      const finish2 = toCanvas(layout.finish_p2);
      if (ACTIVE_TOOL === "edit_start") {
        if (near(p, start1)) drag = {type:"handle", key:"start_p1"};
        else if (near(p, start2)) drag = {type:"handle", key:"start_p2"};
        else if (pointNearSegment(p, start1, start2)) drag = {type:"line", line:"start", last:p};
      } else {
        if (near(p, finish1)) drag = {type:"handle", key:"finish_p1"};
        else if (near(p, finish2)) drag = {type:"handle", key:"finish_p2"};
        else if (pointNearSegment(p, finish1, finish2)) drag = {type:"line", line:"finish", last:p};
      }
    }

    function pointerMove(ev) {
      if (!drag || !layout) return;
      const rect = canvas.getBoundingClientRect();
      const p = {x: ev.clientX - rect.left, y: ev.clientY - rect.top};
      if (drag.type === "handle") {
        const fp = toFrame(p.x, p.y);
        layout[drag.key] = [fp.x, fp.y];
      } else if (drag.type === "line") {
        const dx = p.x - drag.last.x;
        const dy = p.y - drag.last.y;
        drag.last = p;
        const keys = drag.line === "start" ? ["start_p1","start_p2"] : ["finish_p1","finish_p2"];
        for (const key of keys) {
          const cp = toCanvas(layout[key]);
          const fp = toFrame(cp.x + dx, cp.y + dy);
          layout[key] = [fp.x, fp.y];
        }
      }
      draw();
      pushLayout();
    }

    function pointerUp() { drag = null; }

    window.addEventListener("resize", resizeCanvas);
    img.addEventListener("load", resizeCanvas);
    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointerleave", pointerUp);
    loadLayout();
  </script>
  {% endif %}
</body>
</html>"""

    @app.get("/")
    def index() -> Any:
        with ui_lock:
            profile = ui_context.active_profile_name
            menu_open = ui_context.menu_open
            active_tool = ui_context.active_tool
            distance_m = float(distance_state.get("value", 0.0))
        return render_template_string(
            page,
            profile=profile,
            menu_open=menu_open,
            active_tool=active_tool,
            calibration_mode=ui_context.calibration_mode,
            detection_mode=ui_context.detection_mode,
            performance_mode=ui_context.performance_mode,
            detection_mode_title=ui_context.detection_mode.title(),
            performance_mode_title=("Hizli" if ui_context.performance_mode == "HIZLI" else "Normal"),
            show_roi_overlay=ui_context.show_roi_overlay,
            distance_m=f"{distance_m:.1f}",
            speed_limit_kmh=SPEED_LIMIT_KMH,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            top_bar_height=TOP_BAR_HEIGHT,
        )

    @app.post("/action/<action>")
    def action(action: str) -> Any:
        with ui_lock:
            apply_web_action(ui_context, action)
        return redirect(url_for("index"))

    @app.post("/set_distance")
    def set_distance() -> Any:
        raw_value = str(request.form.get("distance", "")).strip().replace(",", ".")
        try:
            distance_value = float(raw_value)
            if distance_value <= 0:
                raise ValueError("Mesafe pozitif olmali.")
        except Exception:
            return redirect(url_for("index"))
        with ui_lock:
            ui_context.pending_external_distance = distance_value
            ui_context.manual_distance_mode = False
            ui_context.manual_speed_mode = False
            ui_context.menu_open = True
            ui_context.active_tool = "none"
        return redirect(url_for("index"))

    @app.post("/set_speed_limit")
    def set_speed_limit() -> Any:
        raw_value = str(request.form.get("speed_limit", "")).strip().replace(",", ".")
        try:
            speed_value = float(raw_value)
            if speed_value <= 0:
                raise ValueError("Limit pozitif olmali.")
        except Exception:
            return redirect(url_for("index"))
        with ui_lock:
            ui_context.pending_external_speed_limit = speed_value
            ui_context.manual_speed_mode = False
            ui_context.manual_distance_mode = False
            ui_context.menu_open = True
            ui_context.active_tool = "none"
        return redirect(url_for("index"))

    @app.get("/api/layout")
    def api_layout() -> Any:
        with ui_lock:
            profile = PROFILES[ui_context.active_profile_name]
            payload = {
                "profile": ui_context.active_profile_name,
                "active_tool": ui_context.active_tool,
                "start_p1": [profile.start_p1[0], profile.start_p1[1]],
                "start_p2": [profile.start_p2[0], profile.start_p2[1]],
                "finish_p1": [profile.finish_p1[0], profile.finish_p1[1]],
                "finish_p2": [profile.finish_p2[0], profile.finish_p2[1]],
            }
        return payload

    @app.post("/api/layout")
    def api_layout_update() -> Any:
        payload = request.get_json(silent=True) or {}
        with ui_lock:
            profile = PROFILES[ui_context.active_profile_name]
            try:
                if ui_context.active_tool == "edit_start":
                    p1 = clamp_point((int(payload["start_p1"][0]), int(payload["start_p1"][1])))
                    p2 = clamp_point((int(payload["start_p2"][0]), int(payload["start_p2"][1])))
                    PROFILES[ui_context.active_profile_name] = set_start_line_points(profile, p1, p2)
                elif ui_context.active_tool == "edit_finish":
                    p1 = clamp_point((int(payload["finish_p1"][0]), int(payload["finish_p1"][1])))
                    p2 = clamp_point((int(payload["finish_p2"][0]), int(payload["finish_p2"][1])))
                    PROFILES[ui_context.active_profile_name] = set_finish_line_points(profile, p1, p2)
            except Exception:
                pass
        return {"ok": True}

    @app.post("/api/calibration_tap")
    def api_calibration_tap() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            x = int(payload.get("x", 0))
            y = int(payload.get("y", 0))
        except Exception:
            return {"ok": False}
        with ui_lock:
            x, y = clamp_point((x, y))
            if ui_context.active_tool != "calibration":
                return {"ok": False}
            if ui_context.calibration_mode == "auto_wait_tap":
                ui_context.calibration_feedback_point = (x, y)
                ui_context.calibration_feedback_until = time.time() + 2.0
                ui_context.pending_calibration = True
                ui_context.pending_calibration_source = "auto_detect"
                ui_context.calibration_first_point = (x, y)
                ui_context.calibration_label = "Kalibre..."
                return {"ok": True}
            if ui_context.calibration_mode == "manual_point_1":
                ui_context.calibration_first_point = (x, y)
                ui_context.calibration_feedback_point = (x, y)
                ui_context.calibration_feedback_until = time.time() + 2.0
                ui_context.calibration_label = "2. Kose"
                ui_context.calibration_mode = "manual_point_2"
                return {"ok": True}
            if ui_context.calibration_mode == "manual_point_2":
                ui_context.calibration_second_point = (x, y)
                ui_context.calibration_feedback_point = (x, y)
                ui_context.calibration_feedback_until = time.time() + 2.0
                ui_context.pending_calibration = True
                ui_context.pending_calibration_source = "manual_points"
                ui_context.calibration_label = "Kalibre..."
                return {"ok": True}
        return {"ok": False}

    @app.get("/video_feed")
    def video_feed() -> Any:
        def generate() -> Any:
            last_frame: Optional[bytes] = None
            while True:
                with web_state.lock:
                    current = web_state.latest_jpeg
                if current is not None:
                    last_frame = current
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + current + b"\r\n"
                    )
                else:
                    time.sleep(0.05)
                    continue
                time.sleep(1.0 / max(WEB_STREAM_FPS_LIMIT, 1.0))

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def start_web_server(
    web_state: WebStreamState,
    ui_context: UIContext,
    ui_lock: threading.Lock,
    distance_state: Dict[str, float],
    logger: logging.Logger,
) -> threading.Thread:
    app = create_web_app(web_state, ui_context, ui_lock, distance_state)

    def run_encoder() -> None:
        while True:
            job: Optional[Dict[str, Any]] = None
            with web_state.lock:
                if web_state.pending_job is not None:
                    job = web_state.pending_job
                    web_state.pending_job = None
            if job is None:
                time.sleep(0.002)
                continue

            # draw_web_overlay yalniz last speed kullandigi icin gecici state nesnesine aktar.
            web_state_holder = RadarState()
            web_state_holder.last_speed_kmh = float(job["state_speed_kmh"])
            web_frame = draw_web_overlay(
                job["frame"],
                job["track"],
                float(job["fps"]),
                web_state_holder,
                int(job["warmup_remaining"]),
                job["profile"],
                float(job["distance_m"]),
                bool(job["show_roi_overlay"]),
                int(job["upstream_offset_px"]),
            )
            ok, encoded = cv2.imencode(".jpg", web_frame, [int(cv2.IMWRITE_JPEG_QUALITY), WEB_JPEG_QUALITY])
            if not ok:
                continue
            with web_state.lock:
                web_state.latest_jpeg = encoded.tobytes()
                web_state.latest_fps = float(job["fps"])
                web_state.latest_status = str(job["status_text"])
                web_state.latest_speed_kmh = float(job["state_speed_kmh"])
                web_state.last_publish_at = float(job["now"])

    def run_server() -> None:
        logger.info("Flask web stream baslatiliyor: http://0.0.0.0:%s", WEB_STREAM_PORT)
        app.run(host="0.0.0.0", port=WEB_STREAM_PORT, debug=False, threaded=True, use_reloader=False)

    encoder_thread = threading.Thread(target=run_encoder, name="radarpi-web-encoder", daemon=True)
    encoder_thread.start()
    thread = threading.Thread(target=run_server, name="radarpi-web", daemon=True)
    thread.start()
    return thread


def open_display_window(ui_context: UIContext) -> None:
    """OpenCV pencere ayarini baslatir."""
    if not SHOW_WINDOW:
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_click, ui_context)


def apply_window_mode(ui_context: UIContext) -> None:
    """Pencereyi tam ekran ile normal mod arasinda gecirir."""
    if not SHOW_WINDOW:
        return
    if ui_context.is_fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH * 2, FRAME_HEIGHT * 2)


def main() -> int:
    global SPEED_LIMIT_KMH
    global RUNTIME_DEFAULT_LAYOUTS
    global RUNTIME_DEFAULT_UPSTREAM_OFFSET_PX
    logger = setup_logging()
    logger.info("Radar uygulamasi baslatiliyor.")
    layout_config = load_layout_config(logger)
    current_distance_m = float(layout_config.get("real_distance_meter", load_distance_setting(logger)))
    SPEED_LIMIT_KMH = float(layout_config.get("speed_limit_kmh", SPEED_LIMIT_KMH))
    for mode_name in ("SOL", "SAG"):
        PROFILES[mode_name] = build_profile_from_layout(mode_name, layout_config["modes"][mode_name])
    upstream_offset_px = int(layout_config.get("upstream_offset_px", DEFAULT_UPSTREAM_OFFSET_PX))
    RUNTIME_DEFAULT_LAYOUTS = {
        "SOL": export_profile_layout(PROFILES["SOL"]),
        "SAG": export_profile_layout(PROFILES["SAG"]),
    }
    RUNTIME_DEFAULT_UPSTREAM_OFFSET_PX = upstream_offset_px

    cap: Optional[CameraAdapter] = None
    try:
        cap = create_camera(logger)
        subtractor = create_subtractor()
        state = RadarState()
        tracker = RadarTracker(logger)
        ui_context = UIContext(initial_profile=str(layout_config.get("last_active_mode", "SOL")))
        ui_context.detection_mode = str(layout_config.get("detection_mode", "DENGELI")).upper()
        ui_context.performance_mode = str(layout_config.get("performance_mode", "NORMAL")).upper()
        apply_runtime_modes(ui_context.detection_mode, ui_context.performance_mode)
        ui_lock = threading.Lock()
        distance_state = {"value": current_distance_m}
        web_state = WebStreamState() if ENABLE_WEB_STREAM else None
        if web_state is not None:
            start_web_server(web_state, ui_context, ui_lock, distance_state, logger)
        open_display_window(ui_context)

        prev_time = time.time()
        last_fps_log_at = prev_time
        fps = 0.0
        frame_count = 0
        detection_roi_mask: Optional[np.ndarray] = None
        measurement_roi_mask: Optional[np.ndarray] = None
        detection_roi_bounds: Optional[Tuple[int, int, int, int]] = None
        measurement_roi_bounds: Optional[Tuple[int, int, int, int]] = None
        cached_profile_signature: Optional[Tuple[Any, ...]] = None
        cached_process_profile: Optional[LayoutProfile] = None
        frame_buffer = start_frame_buffer()
        overlay_cache: Optional[np.ndarray] = None
        timing_stats = {
            "capture_time": 0.0,
            "resize_time": 0.0,
            "roi_time": 0.0,
            "mask_time": 0.0,
            "threshold_time": 0.0,
            "morphology_time": 0.0,
            "contour_time": 0.0,
            "tracking_time": 0.0,
            "overlay_time": 0.0,
            "buffer_time": 0.0,
            "total_loop_time": 0.0,
        }
        timing_sample_count = 0
        last_processed_frame_id = -1

        while True:
            loop_started_at = time.perf_counter()
            if ui_context.exit_requested:
                logger.info("X butonu ile cikis istendi.")
                break

            if ui_context.pending_window_mode:
                ui_context.is_fullscreen = False
                apply_window_mode(ui_context)
                ui_context.pending_window_mode = False
                logger.info("Pencere modu acildi.")

            if ui_context.pending_fullscreen_mode:
                ui_context.is_fullscreen = True
                apply_window_mode(ui_context)
                ui_context.pending_fullscreen_mode = False
                logger.info("Tam ekran modu acildi.")

            if ui_context.pending_profile_change:
                logger.info("Profil degisti: %s", ui_context.active_profile_name)
                state.reset_tracking()
                subtractor = create_subtractor()
                tracker.reset()
                detection_roi_mask = None
                measurement_roi_mask = None
                detection_roi_bounds = None
                measurement_roi_bounds = None
                frame_count = 0
                ui_context.pending_profile_change = False
                layout_config["last_active_mode"] = ui_context.active_profile_name
                ui_context.pending_calibration = False
                ui_context.pending_distance_reset = False
                ui_context.pending_calibration_source = None
                ui_context.calibration_mode = "idle"
                ui_context.calibration_first_point = None
                ui_context.calibration_second_point = None
                ui_context.calibration_candidate_bbox = None
                ui_context.calibration_candidate_length_px = None
                ui_context.calibration_feedback_point = None
                ui_context.calibration_feedback_until = 0.0
                ui_context.calibration_status = "idle"
                ui_context.calibration_label = "Kalibre Et"
                ui_context.manual_distance_mode = False
                ui_context.manual_distance_text = ""
                ui_context.pending_manual_distance_save = False
                ui_context.manual_distance_status = "idle"
                clear_frame_buffer(frame_buffer)

            active_profile = PROFILES[ui_context.active_profile_name]
            active_profile_signature = profile_signature(active_profile)
            if cached_process_profile is None or cached_profile_signature != active_profile_signature:
                cached_process_profile = scale_layout_profile(
                    active_profile,
                    FRAME_WIDTH,
                    FRAME_HEIGHT,
                    PROCESS_WIDTH,
                    PROCESS_HEIGHT,
                )
                cached_profile_signature = active_profile_signature
                detection_roi_mask = None
                measurement_roi_mask = None
            process_profile = cached_process_profile

            capture_started_at = time.perf_counter()
            ok, frame, frame_id = cap.read()
            now = time.time()
            capture_time_ms = (time.perf_counter() - capture_started_at) * 1000.0

            if not ok or frame is None:
                logger.warning("Kameradan kare alinamadi, tekrar denenecek.")
                time.sleep(0.05)
                continue
            if frame_id == last_processed_frame_id:
                time.sleep(0.0015)
                continue
            last_processed_frame_id = frame_id

            if ui_context.pending_distance_reset:
                current_distance_m = 0.0
                distance_state["value"] = current_distance_m
                ui_context.pending_distance_reset = False

            if ui_context.pending_external_distance is not None:
                current_distance_m = float(ui_context.pending_external_distance)
                distance_state["value"] = current_distance_m
                save_distance_setting(current_distance_m, logger)
                layout_config["real_distance_meter"] = current_distance_m
                save_layout_config(layout_config, logger)
                ui_context.pending_external_distance = None
                ui_context.manual_distance_status = "success"
                logger.info("Web uzerinden mesafe girildi: %.2f m", current_distance_m)

            if ui_context.pending_external_speed_limit is not None:
                SPEED_LIMIT_KMH = float(ui_context.pending_external_speed_limit)
                layout_config["speed_limit_kmh"] = SPEED_LIMIT_KMH
                save_layout_config(layout_config, logger)
                ui_context.pending_external_speed_limit = None
                ui_context.manual_speed_status = "success"
                logger.info("Web uzerinden hiz limiti girildi: %.2f km/s", SPEED_LIMIT_KMH)

            if ui_context.pending_manual_distance_save:
                try:
                    entered_distance = float(ui_context.manual_distance_text.strip().replace(",", "."))
                    if entered_distance <= 0:
                        raise ValueError("Mesafe sifirdan buyuk olmali.")
                    current_distance_m = entered_distance
                    distance_state["value"] = current_distance_m
                    save_distance_setting(current_distance_m, logger)
                    layout_config["real_distance_meter"] = current_distance_m
                    save_layout_config(layout_config, logger)
                    ui_context.manual_distance_mode = False
                    ui_context.pending_manual_distance_save = False
                    ui_context.manual_distance_status = "success"
                    ui_context.menu_open = True
                    ui_context.active_tool = "none"
                    logger.info("Elle mesafe girildi: %.2f m", current_distance_m)
                except Exception as exc:
                    ui_context.pending_manual_distance_save = False
                    ui_context.manual_distance_status = "error"
                    logger.warning("Elle mesafe girisi gecersiz: %s", exc)

            if ui_context.pending_manual_speed_save:
                try:
                    entered_speed_limit = float(ui_context.manual_speed_text.strip().replace(",", "."))
                    if entered_speed_limit <= 0:
                        raise ValueError("Limit sifirdan buyuk olmali.")
                    SPEED_LIMIT_KMH = entered_speed_limit
                    layout_config["speed_limit_kmh"] = SPEED_LIMIT_KMH
                    save_layout_config(layout_config, logger)
                    ui_context.manual_speed_mode = False
                    ui_context.pending_manual_speed_save = False
                    ui_context.manual_speed_status = "success"
                    ui_context.menu_open = True
                    ui_context.active_tool = "none"
                    logger.info("Elle hiz limiti girildi: %.2f km/s", SPEED_LIMIT_KMH)
                except Exception as exc:
                    ui_context.pending_manual_speed_save = False
                    ui_context.manual_speed_status = "error"
                    logger.warning("Elle hiz limiti girisi gecersiz: %s", exc)

            if ui_context.pending_calibration:
                calibrated_distance: Optional[float] = None
                source = ui_context.pending_calibration_source

                if source == "auto_detect" and ui_context.calibration_first_point is not None:
                    candidate = try_auto_calibration_from_tap(
                        frame,
                        active_profile,
                        ui_context.calibration_first_point,
                    )
                    if candidate is not None:
                        ui_context.calibration_candidate_bbox, ui_context.calibration_candidate_length_px = candidate
                        ui_context.calibration_mode = "auto_detected_wait_confirm"
                        ui_context.calibration_status = "pending"
                        ui_context.calibration_label = "Nesne Algilandi"
                        logger.info("Kalibrasyon adayi bulundu. onay bekleniyor.")
                    else:
                        ui_context.calibration_candidate_bbox = None
                        ui_context.calibration_candidate_length_px = None
                        ui_context.calibration_mode = "manual_point_1"
                        ui_context.calibration_status = "pending"
                        ui_context.calibration_label = "1. Kose"
                        ui_context.calibration_first_point = None
                        logger.warning("Otomatik kalibrasyon adayi bulunamadi, manuel secime gecildi.")

                elif source == "auto_confirm" and ui_context.calibration_candidate_length_px is not None:
                    calibrated_distance = compute_distance_from_reference_px(
                        active_profile,
                        ui_context.calibration_candidate_length_px,
                    )

                elif (
                    source == "manual_points"
                    and ui_context.calibration_first_point is not None
                    and ui_context.calibration_second_point is not None
                ):
                    calibrated_distance = manual_calibration_from_two_points(
                        active_profile,
                        ui_context.calibration_first_point,
                        ui_context.calibration_second_point,
                    )

                if source in ("auto_confirm", "manual_points"):
                    if calibrated_distance is not None:
                        current_distance_m = calibrated_distance
                        distance_state["value"] = current_distance_m
                        save_distance_setting(current_distance_m, logger)
                        layout_config["real_distance_meter"] = current_distance_m
                        save_layout_config(layout_config, logger)
                        ui_context.calibration_status = "success"
                        ui_context.calibration_label = "Kalibre Edildi"
                        ui_context.calibration_mode = "idle"
                        ui_context.menu_open = True
                        ui_context.active_tool = "none"
                        logger.info("Kalibrasyon basarili: %.0f m", current_distance_m)
                    else:
                        ui_context.calibration_status = "error"
                        ui_context.calibration_label = "Kalibrasyon Hatasi"
                        ui_context.calibration_mode = "idle"
                        ui_context.menu_open = True
                        ui_context.active_tool = "none"
                        logger.warning("Kalibrasyon basarisiz.")

                ui_context.pending_calibration = False
                ui_context.pending_calibration_source = None

                if ui_context.calibration_mode == "idle":
                    ui_context.calibration_first_point = None
                    ui_context.calibration_second_point = None
                    ui_context.calibration_candidate_bbox = None
                    ui_context.calibration_candidate_length_px = None
            if ui_context.pending_save_config:
                layout_config["modes"]["SOL"] = export_profile_layout(PROFILES["SOL"])
                layout_config["modes"]["SAG"] = export_profile_layout(PROFILES["SAG"])
                layout_config["upstream_offset_px"] = upstream_offset_px
                layout_config["real_distance_meter"] = current_distance_m
                layout_config["speed_limit_kmh"] = SPEED_LIMIT_KMH
                layout_config["last_active_mode"] = ui_context.active_profile_name
                layout_config["detection_mode"] = ui_context.detection_mode
                layout_config["performance_mode"] = ui_context.performance_mode
                save_layout_config(layout_config, logger)
                ui_context.pending_save_config = False
                ui_context.menu_status = "saved"
            if ui_context.pending_runtime_mode_apply:
                apply_runtime_modes(ui_context.detection_mode, ui_context.performance_mode)
                cached_process_profile = None
                detection_roi_mask = None
                measurement_roi_mask = None
                ui_context.pending_runtime_mode_apply = False
            if ui_context.active_tool == "reset_active":
                PROFILES["SOL"] = build_profile_from_layout("SOL", RUNTIME_DEFAULT_LAYOUTS["SOL"])
                PROFILES["SAG"] = build_profile_from_layout("SAG", RUNTIME_DEFAULT_LAYOUTS["SAG"])
                upstream_offset_px = RUNTIME_DEFAULT_UPSTREAM_OFFSET_PX
                current_distance_m = DISTANCE_M
                distance_state["value"] = current_distance_m
                SPEED_LIMIT_KMH = 30.0
                ui_context.detection_mode = "DENGELI"
                ui_context.performance_mode = "NORMAL"
                ui_context.show_roi_overlay = False
                ui_context.manual_distance_mode = False
                ui_context.manual_speed_mode = False
                ui_context.manual_distance_text = ""
                ui_context.manual_speed_text = ""
                ui_context.pending_manual_distance_save = False
                ui_context.pending_manual_speed_save = False
                ui_context.manual_distance_status = "idle"
                ui_context.manual_speed_status = "idle"
                ui_context.pending_external_distance = None
                ui_context.pending_external_speed_limit = None
                ui_context.pending_runtime_mode_apply = True
                detection_roi_mask = None
                measurement_roi_mask = None
                save_distance_setting(current_distance_m, logger)
                layout_config["modes"]["SOL"] = export_profile_layout(PROFILES["SOL"])
                layout_config["modes"]["SAG"] = export_profile_layout(PROFILES["SAG"])
                layout_config["upstream_offset_px"] = upstream_offset_px
                layout_config["real_distance_meter"] = current_distance_m
                layout_config["speed_limit_kmh"] = SPEED_LIMIT_KMH
                layout_config["last_active_mode"] = ui_context.active_profile_name
                layout_config["detection_mode"] = ui_context.detection_mode
                layout_config["performance_mode"] = ui_context.performance_mode
                save_layout_config(layout_config, logger)
                ui_context.active_tool = "none"

            active_track: Optional[Track] = None
            resize_time_ms = 0.0
            roi_time_ms = 0.0
            mask_time_ms = 0.0
            threshold_time_ms = 0.0
            morphology_time_ms = 0.0
            contour_time_ms = 0.0
            tracking_time_ms = 0.0
            overlay_time_ms = 0.0
            buffer_time_ms = 0.0
            processing_paused = (
                ui_context.menu_open
                or ui_context.manual_distance_mode
                or ui_context.manual_speed_mode
                or ui_context.active_tool in ("calibration", "edit_start", "edit_finish")
                or ui_context.calibration_mode != "idle"
            )
            if not processing_paused:
                frame_count += 1
                tracker_has_active = tracker.active_track is not None and tracker.active_track.status in ("confirmed", "lost")
                detection_interval = DETECTION_INTERVAL_ACTIVE if tracker_has_active else DETECTION_INTERVAL_IDLE
                run_detection_this_frame = (
                    frame_count <= WARMUP_FRAMES
                    or detection_interval <= 1
                    or (frame_count % detection_interval) == 0
                )

                if run_detection_this_frame:
                    resize_started_at = time.perf_counter()
                    processing_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    resize_time_ms = (time.perf_counter() - resize_started_at) * 1000.0
                    process_upstream_offset = max(10, int(round(upstream_offset_px * (PROCESS_HEIGHT / float(FRAME_HEIGHT)))))
                    if detection_roi_mask is None or detection_roi_mask.shape[:2] != processing_frame.shape[:2]:
                        detection_roi_mask = build_detection_roi(
                            processing_frame.shape,
                            process_profile,
                            process_upstream_offset,
                        )
                        detection_roi_bounds = get_polygon_bounds(
                            get_detection_roi_polygon(process_profile, process_upstream_offset),
                            processing_frame.shape,
                        )
                    if measurement_roi_mask is None or measurement_roi_mask.shape[:2] != processing_frame.shape[:2]:
                        measurement_roi_mask = build_measurement_roi(processing_frame.shape, process_profile)
                        measurement_roi_bounds = get_polygon_bounds(
                            get_measurement_roi_polygon(process_profile),
                            processing_frame.shape,
                        )
                    active_mask = measurement_roi_mask if state.mode in ("WAITING_FOR_START", "TRACKING", "FINISHED") else detection_roi_mask
                    active_bounds = measurement_roi_bounds if state.mode in ("WAITING_FOR_START", "TRACKING", "FINISHED") else detection_roi_bounds
                    if active_mask is None or active_bounds is None:
                        continue
                    mask, preprocess_timings = preprocess_frame_roi(processing_frame, subtractor, active_mask, active_bounds)
                    roi_time_ms = preprocess_timings["roi_time"]
                    mask_time_ms = preprocess_timings["mask_time"]
                    threshold_time_ms = preprocess_timings["threshold_time"]
                    morphology_time_ms = preprocess_timings["morphology_time"]
                    contour_started_at = time.perf_counter()
                    detections = [] if frame_count <= WARMUP_FRAMES_RUNTIME else collect_detections(
                        mask,
                        process_profile,
                        processing_frame.shape[1],
                        roi_offset=(active_bounds[0], active_bounds[1]),
                    )
                    contour_time_ms = (time.perf_counter() - contour_started_at) * 1000.0
                    tracking_started_at = time.perf_counter()
                    active_track = tracker.update(detections, now, process_profile)
                    tracking_time_ms = (time.perf_counter() - tracking_started_at) * 1000.0
                else:
                    active_track = tracker.active_track

                if state.mode == "TRACKING" and active_track is not None:
                    buffer_started_at = time.perf_counter()
                    append_frame_buffer(frame_buffer, frame, scale_track_for_display(active_track, PROCESS_WIDTH, PROCESS_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT), active_profile, now)
                    buffer_time_ms = (time.perf_counter() - buffer_started_at) * 1000.0
                elif state.mode not in ("WAITING_FOR_START", "TRACKING"):
                    clear_frame_buffer(frame_buffer)

                speed_kmh = update_speed_state(active_track, state, now, logger, process_profile, current_distance_m)
                if speed_kmh is not None and speed_kmh > SPEED_LIMIT_KMH:
                    plate_frame = capture_plate_frame_from_buffer(frame_buffer)
                    if plate_frame is None:
                        plate_frame = frame
                    save_violation(plate_frame, speed_kmh, logger, active_profile.name)
                    state.enter_cooldown(speed_kmh, now)
                    tracker.reset()
                    clear_frame_buffer(frame_buffer)
                elif speed_kmh is not None:
                    state.last_speed_kmh = speed_kmh
                    state.reset_tracking()
                    tracker.reset()
                    clear_frame_buffer(frame_buffer)
            else:
                if state.mode == "TRACKING":
                    clear_frame_buffer(frame_buffer)

            warmup_remaining = max(0, WARMUP_FRAMES_RUNTIME - frame_count)
            display_track = scale_track_for_display(
                active_track,
                PROCESS_WIDTH,
                PROCESS_HEIGHT,
                FRAME_WIDTH,
                FRAME_HEIGHT,
            )
            should_render_overlay = (
                overlay_cache is None
                or (frame_count % OVERLAY_RENDER_INTERVAL) == 0
                or processing_paused
                or ui_context.menu_open
                or ui_context.active_tool != "none"
            )
            if should_render_overlay:
                overlay_started_at = time.perf_counter()
                overlay_cache = draw_overlay(
                    frame,
                    display_track,
                    fps,
                    state,
                    warmup_remaining,
                    measurement_roi_mask if measurement_roi_mask is not None else np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH), dtype=np.uint8),
                    active_profile,
                    ui_context,
                    current_distance_m,
                    upstream_offset_px,
                )
                overlay_time_ms = (time.perf_counter() - overlay_started_at) * 1000.0

            web_publish_started_at = time.perf_counter()
            maybe_publish_web_frame(
                web_state,
                frame,
                display_track,
                fps,
                state,
                warmup_remaining,
                active_profile,
                current_distance_m,
                ui_context.show_roi_overlay,
                upstream_offset_px,
                now,
            )
            overlay_time_ms += (time.perf_counter() - web_publish_started_at) * 1000.0

            if SHOW_WINDOW:
                if overlay_cache is None:
                    overlay_cache = frame
                cv2.imshow(WINDOW_NAME, overlay_cache)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    logger.info("ESC ile cikis istendi.")
                    break

            dt = max(now - prev_time, 1e-6)
            fps = 1.0 / dt
            prev_time = now
            total_loop_time_ms = (time.perf_counter() - loop_started_at) * 1000.0
            timing_stats["capture_time"] += capture_time_ms
            timing_stats["resize_time"] += resize_time_ms
            timing_stats["roi_time"] += roi_time_ms
            timing_stats["mask_time"] += mask_time_ms
            timing_stats["threshold_time"] += threshold_time_ms
            timing_stats["morphology_time"] += morphology_time_ms
            timing_stats["contour_time"] += contour_time_ms
            timing_stats["tracking_time"] += tracking_time_ms
            timing_stats["overlay_time"] += overlay_time_ms
            timing_stats["buffer_time"] += buffer_time_ms
            timing_stats["total_loop_time"] += total_loop_time_ms
            timing_sample_count += 1

            if (now - last_fps_log_at) >= FPS_LOG_INTERVAL_SECONDS:
                sample_div = max(1, timing_sample_count)
                avg_stats = {key: value / sample_div for key, value in timing_stats.items()}
                bottleneck_name = max(
                    ("capture_time", "resize_time", "roi_time", "mask_time", "threshold_time", "morphology_time", "contour_time", "tracking_time", "overlay_time", "buffer_time"),
                    key=lambda key: avg_stats[key],
                )
                logger.info(
                    "FPS=%.1f capture=%.1fms threshold=%.1fms contour=%.1fms overlay=%.1fms total=%.1fms bottleneck=%s(%.1fms) durum=%s takip=%s profil=%s",
                    fps,
                    avg_stats["capture_time"],
                    avg_stats["threshold_time"],
                    avg_stats["contour_time"],
                    avg_stats["overlay_time"],
                    avg_stats["total_loop_time"],
                    bottleneck_name,
                    avg_stats[bottleneck_name],
                    state.mode,
                    active_track.status if active_track is not None else "yok",
                    active_profile.name,
                )
                last_fps_log_at = now
                timing_stats = {key: 0.0 for key in timing_stats}
                timing_sample_count = 0

        return 0

    except KeyboardInterrupt:
        logger.info("Uygulama kullanici tarafindan durduruldu.")
        return 0
    except Exception as exc:
        logger.exception("Beklenmeyen hata: %s", exc)
        return 1
    finally:
        if cap is not None:
            cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        logger.info("Radar uygulamasi kapandi.")


if __name__ == "__main__":
    raise SystemExit(main())
