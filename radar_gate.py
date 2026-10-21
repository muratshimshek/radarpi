#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Raspberry Pi 5 speed measurement and post-capture vehicle validation."""

from __future__ import annotations

import logging
import json
import threading
import time
import csv
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
try:
    from vehicle_validation_worker import (
        add_vehicle_validation_job,
        start_vehicle_validation_worker,
        stop_vehicle_validation_worker,
    )
except Exception:
    add_vehicle_validation_job = None
    start_vehicle_validation_worker = None
    stop_vehicle_validation_worker = None

try:
    from flask import Flask, Response, abort, jsonify, render_template_string, request, send_from_directory
except Exception:
    Flask = None
    Response = None
    abort = None
    jsonify = None
    render_template_string = None
    request = None
    send_from_directory = None

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None


# ============================================================
# PI ODAKLI AYARLAR
# ============================================================

BASE_DIR = Path("/home/radarpi/radar")
VIOLATION_DIR = BASE_DIR / "ihlaller"
CONFIG_FILE = BASE_DIR / "radar_gate_config.json"

USE_PICAMERA2 = True
CAMERA_SOURCE = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
ANALYSIS_WIDTH = 480
ANALYSIS_HEIGHT = 360
TARGET_FPS = 60
FORCE_CAMERA_FPS = True

WEB_HOST = "0.0.0.0"
WEB_PORT = 5001
WEB_JPEG_QUALITY = 60
WEB_STREAM_MAX_FPS = 8.0
SHOW_GLOBAL_REF_OVERLAY = False
USE_COLOR_DEBUG_PREVIEW = False
COLOR_PREVIEW_EVERY_N_FRAMES = 6
USE_HIGH_RES_DEBUG_PREVIEW = True
HIGH_RES_PREVIEW_EVERY_N_FRAMES = 2
DEBUG_PREVIEW_COLOR = False

ACTIVE_PROFILE = "SAG"

# Fiziksel kapi mesafesi metre cinsinden.
GATE_DISTANCE_M = 3.0
SPEED_LIMIT_KMH = 50.0

# Baslangicta background otursun.
WARMUP_SECONDS = 2.5

# Cizgiden otomatik uretilen ROI bandi kalinligi.
BAND_HALF_THICKNESS_PX = 24
CENTERLINE_THICKNESS_PX = 5
FINISH_CENTERLINE_THICKNESS_PX = 9

# Her gate tetigi en az bu kadar kare ust uste gorulmeli.
CONFIRM_FRAMES = 3

# Piksel farki bu esigi gecerse "degismis" sayilir.
PIXEL_DIFF_THRESHOLD = 28

# ROI tetik esikleri.
ROI_MEAN_THRESHOLD = 12.0
ROI_RATIO_THRESHOLD = 0.06

# Global isik degisimi esikleri.
GLOBAL_MEAN_THRESHOLD = 10.0
GLOBAL_RATIO_THRESHOLD = 0.10

# State machine zaman sinirlari.
WAIT_FINISH_TIMEOUT_SECONDS = 6.0
COOLDOWN_SECONDS = 1.5
MIN_VALID_DELTA_SECONDS = 0.20
MAX_VALID_DELTA_SECONDS = 6.0

# Adaptive background hizlari.
BG_ALPHA_IDLE = 0.02
BG_ALPHA_GLOBAL = 0.03
BG_ALPHA_COOLDOWN = 0.005

TEST_MODE_DEFAULT = True
CSV_LOG_PATH = VIOLATION_DIR / "event_log.csv"

ROI_RATIO_HIGH = ROI_RATIO_THRESHOLD
ROI_RATIO_LOW = 0.03
ROI_MEAN_HIGH = ROI_MEAN_THRESHOLD
ROI_MEAN_LOW = 8.0
SUPPRESSION_RATIO_MAX = 0.90

# Kirmizi FINISH kapisi pratikte gec tetiklenmeye daha yatkin oldugu icin
# local esikleri biraz daha toleransli tutulur.
FINISH_RATIO_HIGH = 0.045
FINISH_RATIO_LOW = 0.022
FINISH_MEAN_HIGH = 10.0
FINISH_MEAN_LOW = 7.0

# Merkez dogru temasinda FINISH biraz daha toleranslidir.
START_CENTERLINE_MIN_PIXELS = 2
START_CENTERLINE_MIN_RATIO = 0.03
FINISH_CENTERLINE_MIN_PIXELS = 1
FINISH_CENTERLINE_MIN_RATIO = 0.02

# Ardisik kareler arasi gercek hareket esikleri.
# Sabit duran ama background'dan farkli nesnelerin candidate olmasini engeller.
TEMPORAL_RATIO_HIGH = 0.015
TEMPORAL_RATIO_LOW = 0.006
TEMPORAL_MEAN_HIGH = 5.0
TEMPORAL_MEAN_LOW = 2.0

# ============================================================
# PROFILLER
# ============================================================

# START / FINISH cizgileri ROI poligonuna otomatik cevrilir.
# Global referans ROI ayri tanimlidir ve gate'lerle cakismamalidir.
DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "SAG": {
        "start_line_p1": (120, 92),
        "start_line_p2": (565, 118),
        "finish_line_p1": (35, 196),
        "finish_line_p2": (500, 244),
        # Yol disi / sakin alan olarak ayri bir referans ROI.
        "global_ref_polygon": [(6, 20), (170, 20), (170, 88), (6, 88)],
    },
    "SOL": {
        "start_line_p1": (72, 118),
        "start_line_p2": (520, 92),
        "finish_line_p1": (110, 244),
        "finish_line_p2": (575, 196),
        "global_ref_polygon": [(470, 20), (634, 20), (634, 88), (470, 88)],
    },
}


# ============================================================
# VERI SINIFLARI
# ============================================================


@dataclass
class MotionScore:
    """Tek ROI icin hareket olcum sonucu."""

    mean_abs_diff: float = 0.0
    changed_pixel_ratio: float = 0.0
    pixel_count: int = 0


@dataclass
class ROIRegion:
    """Bir ROI poligonunun crop ve local mask bilgisi."""

    polygon: np.ndarray
    bbox: tuple[int, int, int, int]
    local_mask: np.ndarray
    centerline_mask: np.ndarray


@dataclass
class GateState:
    """State machine ve debug verileri."""

    state: str = "IDLE"
    last_message: str = "WARMUP"
    t1: Optional[float] = None
    t2: Optional[float] = None
    delta_time: Optional[float] = None
    speed_kmh: Optional[float] = None
    start_confirm_count: int = 0
    finish_confirm_count: int = 0
    wait_finish_deadline: float = 0.0
    cooldown_until: float = 0.0
    last_process_ms: float = 0.0
    fps: float = 0.0
    last_frame_ts: Optional[float] = None
    warmup_until: float = 0.0
    last_violation_text: str = "--"
    violation_saved_this_cycle: bool = False
    start_trigger_flash_until: float = 0.0
    finish_trigger_flash_until: float = 0.0
    start_candidate_latched: bool = False
    finish_candidate_latched: bool = False
    start_candidate_since: Optional[float] = None
    finish_candidate_since: Optional[float] = None
    start_centerline_since: Optional[float] = None
    finish_centerline_since: Optional[float] = None
    finish_exit_pending: bool = False
    current_event_id: str = ""
    start_peak_ratio: float = 0.0
    finish_peak_ratio: float = 0.0
    global_peak_ratio: float = 0.0
    start_peak_mean: float = 0.0
    finish_peak_mean: float = 0.0
    event_log_pending: bool = False
    pending_event_valid: bool = False
    pending_event_message: str = ""


@dataclass
class WebState:
    """Flask thread ile paylasilan son debug goruntusu."""

    latest_jpeg: bytes = b""
    latest_status: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class ControlState:
    """Web arayuzu ile paylasilan kalibrasyon ve profil durumu."""

    active_profile: str = ACTIVE_PROFILE
    profiles: dict[str, dict[str, Any]] = field(default_factory=lambda: deepcopy(DEFAULT_PROFILES))
    gate_distance_m: float = GATE_DISTANCE_M
    speed_limit_kmh: float = SPEED_LIMIT_KMH
    menu_open: bool = False
    edit_mode: str = "none"
    test_mode: bool = TEST_MODE_DEFAULT
    version: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class CapturedFrame:
    """Kamera thread'inin urettigi son frame."""

    gray: np.ndarray
    color_frame: Optional[np.ndarray]
    captured_at: float


@dataclass
class FrameStore:
    """Kamera thread'i ile isleme thread'i arasinda son frame'i tutar."""

    latest: Optional[CapturedFrame] = None
    sequence: int = 0
    stopped: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def put(self, frame: CapturedFrame) -> None:
        """Yeni frame'i atomik olarak yayinlar."""
        with self.lock:
            self.latest = frame
            self.sequence += 1

    def get_if_newer(self, last_sequence: int) -> tuple[Optional[CapturedFrame], int]:
        """Yeni frame varsa dondurur, yoksa None verir."""
        with self.lock:
            if self.latest is None or self.sequence == last_sequence:
                return None, last_sequence
            return self.latest, self.sequence


# ============================================================
# KAMERA
# ============================================================


class CameraHandle:
    """Kameranin ortak arayuzu."""

    def __init__(self, mode: str, main: Any, lores_size: tuple[int, int]) -> None:
        self.mode = mode
        self.main = main
        self.lores_size = lores_size


def init_camera(logger: logging.Logger) -> CameraHandle:
    """Picamera2 oncelikli kamera baslatir, olmazsa OpenCV fallback kullanir."""
    lores_size = (ANALYSIS_WIDTH, ANALYSIS_HEIGHT)

    if USE_PICAMERA2 and Picamera2 is not None:
        try:
            picam2 = Picamera2()
            controls: dict[str, Any] = {}
            if FORCE_CAMERA_FPS and TARGET_FPS > 0:
                frame_us = int(round(1_000_000 / float(TARGET_FPS)))
                # Sensoru hedef FPS'e zorla; dusuk isikta goruntu kararabilir ama FPS sabitlenir.
                controls["FrameDurationLimits"] = (frame_us, frame_us)
            config = picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"},
                lores={"size": lores_size, "format": "YUV420"},
                controls=controls,
                buffer_count=4,
                queue=False,
            )
            picam2.configure(config)
            picam2.start()
            logger.info("Picamera2 baslatildi.")
            return CameraHandle(mode="picamera2", main=picam2, lores_size=lores_size)
        except Exception as exc:
            logger.warning("Picamera2 acilamadi, OpenCV fallback denenecek: %s", exc)

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    if not cap.isOpened():
        raise RuntimeError("Kamera acilamadi.")
    logger.info("OpenCV VideoCapture baslatildi.")
    return CameraHandle(mode="opencv", main=cap, lores_size=lores_size)


def get_luminance_frame(camera: CameraHandle) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Motion icin tek kanalli luminance frame, gerekiyorsa renkli frame dondurur."""
    if camera.mode == "picamera2":
        picam2 = camera.main

        try:
            yuv = picam2.capture_array("lores")
            # Picamera2 YUV420 donerse Y duzlemi ilk HEIGHT satirindadir.
            gray = yuv[: ANALYSIS_HEIGHT, : ANALYSIS_WIDTH].copy()
            if gray.ndim == 2 and gray.shape == (ANALYSIS_HEIGHT, ANALYSIS_WIDTH):
                return None, gray
        except Exception:
            # YUV alma olmazsa guvenli sekilde grayscale fallback yap.
            pass

        color_frame = picam2.capture_array("main")
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT), interpolation=cv2.INTER_LINEAR)
        return color_frame, gray

    cap = camera.main
    ok, color_frame = cap.read()
    if not ok or color_frame is None:
        raise RuntimeError("Kameradan kare okunamadi.")
    if color_frame.shape[1] != FRAME_WIDTH or color_frame.shape[0] != FRAME_HEIGHT:
        color_frame = cv2.resize(color_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return color_frame, gray


def get_color_frame(camera: CameraHandle) -> Optional[np.ndarray]:
    """Yalnizca kayit gerektiginde renkli kare alir."""
    try:
        if camera.mode == "picamera2":
            return camera.main.capture_array("main")
        return None
    except Exception:
        return None


def start_camera_thread(camera: CameraHandle, frame_store: FrameStore, logger: logging.Logger) -> threading.Thread:
    """Kameradan surekli frame okuyup en son kareyi paylasir."""

    def runner() -> None:
        last_color_frame: Optional[np.ndarray] = None
        frame_index = 0
        while not frame_store.stopped:
            try:
                color_frame, gray = get_luminance_frame(camera)
                frame_index += 1
                if USE_COLOR_DEBUG_PREVIEW or USE_HIGH_RES_DEBUG_PREVIEW:
                    if color_frame is not None:
                        last_color_frame = color_frame
                    else:
                        sample_every = COLOR_PREVIEW_EVERY_N_FRAMES if USE_COLOR_DEBUG_PREVIEW else HIGH_RES_PREVIEW_EVERY_N_FRAMES
                        if frame_index % max(1, sample_every) == 0:
                            sampled_color = get_color_frame(camera)
                            if sampled_color is not None:
                                last_color_frame = sampled_color
                    color_frame = last_color_frame
                else:
                    color_frame = None
                frame_store.put(CapturedFrame(gray=gray, color_frame=color_frame, captured_at=time.monotonic()))
            except Exception as exc:
                logger.warning("Kamera frame okunamadi: %s", exc)
                time.sleep(0.05)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


# ============================================================
# ROI / BACKGROUND
# ============================================================


def build_line_band_polygon(
    p1: tuple[int, int],
    p2: tuple[int, int],
    half_thickness_px: int,
) -> np.ndarray:
    """Egik cizgiden otomatik 4 noktalı bant poligonu uretir."""
    x1, y1 = p1
    x2, y2 = p2
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)

    # Cizgiye dik normal vektor.
    nx = -dy / length
    ny = dx / length

    ox = nx * float(half_thickness_px)
    oy = ny * float(half_thickness_px)

    polygon = np.array(
        [
            [int(round(x1 + ox)), int(round(y1 + oy))],
            [int(round(x2 + ox)), int(round(y2 + oy))],
            [int(round(x2 - ox)), int(round(y2 - oy))],
            [int(round(x1 - ox)), int(round(y1 - oy))],
        ],
        dtype=np.int32,
    )
    return polygon


def build_roi_mask(frame_shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """Verilen poligon icin maske uretir."""
    mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def build_roi_region(frame_shape: tuple[int, int], polygon: np.ndarray) -> ROIRegion:
    """ROI poligonunu daha hizli crop-islem yapisi haline getirir."""
    x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
    x = max(0, min(x, frame_shape[1] - 1))
    y = max(0, min(y, frame_shape[0] - 1))
    w = max(1, min(w, frame_shape[1] - x))
    h = max(1, min(h, frame_shape[0] - y))

    local_polygon = polygon.astype(np.int32).copy()
    local_polygon[:, 0] -= x
    local_polygon[:, 1] -= y

    local_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(local_mask, local_polygon, 255)
    centerline_mask = np.zeros((h, w), dtype=np.uint8)
    return ROIRegion(polygon=polygon, bbox=(x, y, w, h), local_mask=local_mask, centerline_mask=centerline_mask)


def set_region_centerline(
    region: ROIRegion,
    p1: tuple[int, int],
    p2: tuple[int, int],
    thickness: int = CENTERLINE_THICKNESS_PX,
) -> None:
    """ROI crop icinde merkez cizgisi maskesini olusturur."""
    x, y, w, h = region.bbox
    mask = np.zeros((h, w), dtype=np.uint8)
    lp1 = (int(round(p1[0] - x)), int(round(p1[1] - y)))
    lp2 = (int(round(p2[0] - x)), int(round(p2[1] - y)))
    cv2.line(mask, lp1, lp2, 255, thickness=max(1, thickness), lineType=cv2.LINE_AA)
    region.centerline_mask = cv2.bitwise_and(mask, region.local_mask)


def scale_polygon(
    polygon: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    """Bir poligonu kaynak boyuttan hedef boyuta oransal tasir."""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    scaled = polygon.astype(np.float32).copy()
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy
    return np.round(scaled).astype(np.int32)


def compute_roi_motion_score(
    current_crop: np.ndarray,
    background_crop: np.ndarray,
    local_mask: np.ndarray,
) -> MotionScore:
    """Maskeli bolgedeki fark ortalamasi ve degisen piksel oranini hesaplar."""
    diff = cv2.absdiff(current_crop, background_crop)
    roi_pixels = diff[local_mask > 0]
    if roi_pixels.size == 0:
        return MotionScore()

    changed = roi_pixels > PIXEL_DIFF_THRESHOLD
    return MotionScore(
        mean_abs_diff=float(np.mean(roi_pixels)),
        changed_pixel_ratio=float(np.count_nonzero(changed) / float(roi_pixels.size)),
        pixel_count=int(roi_pixels.size),
    )


def centerline_touched(
    current_crop: np.ndarray,
    background_crop: np.ndarray,
    centerline_mask: np.ndarray,
    min_changed_pixels: int = START_CENTERLINE_MIN_PIXELS,
    min_ratio: float = START_CENTERLINE_MIN_RATIO,
) -> bool:
    """Merkez cizgisinde anlamli foreground var mi kontrol eder."""
    line_pixels = centerline_mask > 0
    total = int(np.count_nonzero(line_pixels))
    if total == 0:
        return False
    diff = cv2.absdiff(current_crop, background_crop)
    changed = np.count_nonzero((diff > PIXEL_DIFF_THRESHOLD) & line_pixels)
    ratio = changed / float(total)
    return changed >= max(1, min_changed_pixels) and ratio >= float(min_ratio)


def update_background(
    background_crop: np.ndarray,
    current_crop: np.ndarray,
    local_mask: np.ndarray,
    alpha: float,
) -> None:
    """Maskeli pikselleri yavas EMA ile background'a yaklastirir."""
    if alpha <= 0.0:
        return
    roi = local_mask > 0
    background_crop[roi] = ((1.0 - alpha) * background_crop[roi]) + (alpha * current_crop[roi])


def get_profile_geometry(control_state: ControlState) -> dict[str, Any]:
    """Aktif profilin tum cizgi ve poligon geometrisini dondurur."""
    with control_state.lock:
        profile = deepcopy(control_state.profiles[control_state.active_profile])

    start_p1 = tuple(profile["start_line_p1"])
    start_p2 = tuple(profile["start_line_p2"])
    finish_p1 = tuple(profile["finish_line_p1"])
    finish_p2 = tuple(profile["finish_line_p2"])
    global_polygon = np.array(profile["global_ref_polygon"], dtype=np.int32)
    start_polygon = build_line_band_polygon(start_p1, start_p2, BAND_HALF_THICKNESS_PX)
    finish_polygon = build_line_band_polygon(finish_p1, finish_p2, BAND_HALF_THICKNESS_PX)
    left_start = segment_midpoint(start_polygon[0], start_polygon[3])
    left_finish = segment_midpoint(finish_polygon[0], finish_polygon[3])
    right_start = segment_midpoint(start_polygon[1], start_polygon[2])
    right_finish = segment_midpoint(finish_polygon[1], finish_polygon[2])
    left_full = clip_infinite_line_to_frame(left_start, left_finish, FRAME_WIDTH, FRAME_HEIGHT)
    right_full = clip_infinite_line_to_frame(right_start, right_finish, FRAME_WIDTH, FRAME_HEIGHT)

    return {
        "start_p1": start_p1,
        "start_p2": start_p2,
        "finish_p1": finish_p1,
        "finish_p2": finish_p2,
        "start_polygon": start_polygon,
        "finish_polygon": finish_polygon,
        "global_polygon": global_polygon,
        "left_full": left_full,
        "right_full": right_full,
    }


def line_center(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
    """Cizginin orta noktasini verir."""
    return (int(round((p1[0] + p2[0]) / 2.0)), int(round((p1[1] + p2[1]) / 2.0)))


def segment_midpoint(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Iki noktanin orta noktasini verir."""
    return (int(round((int(a[0]) + int(b[0])) / 2.0)), int(round((int(a[1]) + int(b[1])) / 2.0)))


def clip_infinite_line_to_frame(
    p1: tuple[int, int],
    p2: tuple[int, int],
    width: int,
    height: int,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Sonsuz dogruyu frame sinirlarina kirpar."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1
    points: list[tuple[int, int]] = []

    if abs(dx) > 1e-6:
        for x in (0.0, float(width - 1)):
            t = (x - x1) / dx
            y = y1 + (t * dy)
            if 0.0 <= y <= float(height - 1):
                points.append((int(round(x)), int(round(y))))

    if abs(dy) > 1e-6:
        for y in (0.0, float(height - 1)):
            t = (y - y1) / dy
            x = x1 + (t * dx)
            if 0.0 <= x <= float(width - 1):
                points.append((int(round(x)), int(round(y))))

    uniq: list[tuple[int, int]] = []
    for point in points:
        if point not in uniq:
            uniq.append(point)
    if len(uniq) < 2:
        return None
    return uniq[0], uniq[1]


def center_distance_px(start_p1: tuple[int, int], start_p2: tuple[int, int], finish_p1: tuple[int, int], finish_p2: tuple[int, int]) -> float:
    """START ve FINISH merkezleri arasindaki piksel mesafesi."""
    c1 = line_center(start_p1, start_p2)
    c2 = line_center(finish_p1, finish_p2)
    return float(((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2) ** 0.5)


# ============================================================
# STATE MACHINE / HIZ
# ============================================================


def calculate_speed(delta_time: float, gate_distance_m: float) -> float:
    """Fiziksel gate mesafesinden km/h hesaplar."""
    speed_mps = gate_distance_m / max(delta_time, 1e-6)
    return speed_mps * 3.6


def generate_event_id() -> str:
    """Her olay icin tekil bir event kimligi uretir."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def reset_event_peaks(runtime: GateState) -> None:
    """Yeni event basinda peak metriklerini sifirlar."""
    runtime.start_peak_ratio = 0.0
    runtime.finish_peak_ratio = 0.0
    runtime.global_peak_ratio = 0.0
    runtime.start_peak_mean = 0.0
    runtime.finish_peak_mean = 0.0


EVENT_LOG_FIELDS = [
    "event_id",
    "timestamp",
    "profile",
    "gate_distance_m",
    "speed_limit_kmh",
    "delta",
    "speed_kmh",
    "start_peak_ratio",
    "finish_peak_ratio",
    "global_peak_ratio",
    "start_peak_mean",
    "finish_peak_mean",
    "fps",
    "process_ms",
    "valid",
    "message",
    "violation_saved",
    "image_filename",
]


def ensure_event_log_header() -> None:
    """CSV log dosyasi yoksa veya bossa header yazar."""
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (not CSV_LOG_PATH.exists()) or CSV_LOG_PATH.stat().st_size == 0
    if not needs_header:
        return
    with CSV_LOG_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENT_LOG_FIELDS)
        writer.writeheader()


def append_event_log(event_data: dict[str, Any]) -> bool:
    """Event satirini CSV log dosyasina ekler."""
    try:
        ensure_event_log_header()
        row = {field: event_data.get(field, "") for field in EVENT_LOG_FIELDS}
        with CSV_LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=EVENT_LOG_FIELDS)
            writer.writerow(row)
        return True
    except Exception:
        return False


def update_state_machine(
    runtime: GateState,
    now: float,
    start_triggered: bool,
    finish_triggered: bool,
    start_center_hit: bool,
    finish_center_hit: bool,
    gate_distance_m: float,
) -> None:
    """START / FINISH tetiklerine gore state machine'i gunceller."""
    runtime.last_message = runtime.state

    if now < runtime.warmup_until:
        runtime.state = "WARMUP"
        runtime.last_message = "WARMUP"
        runtime.start_confirm_count = 0
        runtime.finish_confirm_count = 0
        runtime.start_candidate_latched = False
        runtime.finish_candidate_latched = False
        runtime.start_candidate_since = None
        runtime.finish_candidate_since = None
        runtime.start_centerline_since = None
        runtime.finish_centerline_since = None
        runtime.finish_exit_pending = False
        return

    if runtime.state == "WARMUP":
        runtime.state = "IDLE"
        runtime.last_message = "IDLE"

    if runtime.state == "COOLDOWN":
        if now >= runtime.cooldown_until:
            runtime.state = "IDLE"
            runtime.last_message = "IDLE"
            runtime.violation_saved_this_cycle = False
        else:
            runtime.last_message = "COOLDOWN"
        runtime.start_confirm_count = 0
        runtime.finish_confirm_count = 0
        runtime.start_candidate_latched = False
        runtime.finish_candidate_latched = False
        runtime.start_candidate_since = None
        runtime.finish_candidate_since = None
        runtime.start_centerline_since = None
        runtime.finish_centerline_since = None
        runtime.finish_exit_pending = False
        return

    if runtime.state == "IDLE":
        runtime.finish_confirm_count = 0
        if start_center_hit and runtime.start_centerline_since is None:
            runtime.start_centerline_since = now
        if start_triggered:
            if runtime.start_confirm_count == 0 or runtime.start_candidate_since is None:
                runtime.start_candidate_since = now
            runtime.start_confirm_count += 1
        else:
            runtime.start_confirm_count = 0
            runtime.start_candidate_since = None
            if not start_center_hit:
                runtime.start_centerline_since = None

        if runtime.start_confirm_count >= CONFIRM_FRAMES:
            runtime.current_event_id = generate_event_id()
            reset_event_peaks(runtime)
            runtime.t1 = (
                runtime.start_centerline_since
                if runtime.start_centerline_since is not None
                else runtime.start_candidate_since if runtime.start_candidate_since is not None else now
            )
            runtime.t2 = None
            runtime.delta_time = None
            runtime.speed_kmh = None
            runtime.wait_finish_deadline = now + WAIT_FINISH_TIMEOUT_SECONDS
            runtime.state = "WAIT_FINISH"
            runtime.last_message = "START TRIGGERED"
            runtime.start_trigger_flash_until = now + 0.7
            runtime.start_confirm_count = 0
            runtime.finish_confirm_count = 0
            runtime.event_log_pending = False
            runtime.pending_event_valid = False
            runtime.pending_event_message = ""
            runtime.finish_candidate_since = None
            runtime.finish_centerline_since = None
            runtime.finish_exit_pending = False
        else:
            runtime.last_message = "IDLE"
        return

    if runtime.state == "WAIT_FINISH":
        if now > runtime.wait_finish_deadline:
            runtime.state = "IDLE"
            runtime.last_message = "TIMEOUT"
            runtime.start_confirm_count = 0
            runtime.finish_confirm_count = 0
            runtime.pending_event_valid = False
            runtime.pending_event_message = "TIMEOUT"
            runtime.event_log_pending = True
            runtime.t1 = None
            runtime.t2 = None
            runtime.speed_kmh = None
            runtime.start_candidate_latched = False
            runtime.finish_candidate_latched = False
            runtime.start_candidate_since = None
            runtime.finish_candidate_since = None
            runtime.start_centerline_since = None
            runtime.finish_centerline_since = None
            runtime.finish_exit_pending = False
            return

        if finish_center_hit and runtime.finish_centerline_since is None:
            runtime.finish_centerline_since = now
        if finish_triggered:
            if runtime.finish_confirm_count == 0 or runtime.finish_candidate_since is None:
                runtime.finish_candidate_since = now
            runtime.finish_confirm_count += 1
        else:
            runtime.finish_confirm_count = 0
            runtime.finish_candidate_since = None
            if not finish_center_hit:
                runtime.finish_centerline_since = None

        if runtime.finish_confirm_count >= CONFIRM_FRAMES and runtime.t1 is not None:
            runtime.t2 = (
                runtime.finish_centerline_since
                if runtime.finish_centerline_since is not None
                else runtime.finish_candidate_since if runtime.finish_candidate_since is not None else now
            )
            runtime.delta_time = runtime.t2 - runtime.t1
            runtime.finish_trigger_flash_until = now + 0.7

            if runtime.delta_time < MIN_VALID_DELTA_SECONDS or runtime.delta_time > MAX_VALID_DELTA_SECONDS:
                runtime.speed_kmh = None
                runtime.last_message = "INVALID DELTA"
                runtime.pending_event_valid = False
                runtime.pending_event_message = "INVALID DELTA"
            else:
                runtime.speed_kmh = calculate_speed(runtime.delta_time, gate_distance_m)
                runtime.last_message = "FINISH TRIGGERED"
                runtime.pending_event_valid = True
                runtime.pending_event_message = "FINISH TRIGGERED"

            runtime.state = "COOLDOWN"
            runtime.cooldown_until = now + COOLDOWN_SECONDS
            runtime.start_confirm_count = 0
            runtime.finish_confirm_count = 0
            runtime.event_log_pending = True
            runtime.start_candidate_latched = False
            runtime.finish_candidate_latched = False
            runtime.start_candidate_since = None
            runtime.finish_candidate_since = None
            runtime.start_centerline_since = None
            runtime.finish_centerline_since = None
            runtime.finish_exit_pending = False
        else:
            runtime.last_message = "WAITING FINISH"


def save_violation_frame(frame: np.ndarray, speed_kmh: float, event_id: str, logger: logging.Logger) -> str:
    """Ihlal goruntusunu zaman damgasi ile kaydeder."""
    try:
        VIOLATION_DIR.mkdir(parents=True, exist_ok=True)
        ts = event_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = VIOLATION_DIR / f"ihlal_{ts}_{int(round(speed_kmh))}kmh.jpg"
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            logger.error("Ihlal kaydedilemedi: %s", path)
            return ""
        logger.warning("Ihlal kaydedildi: %s (%.2f km/h)", path, speed_kmh)
        return str(path)
    except Exception as exc:
        logger.error("Ihlal kayit hatasi: %s", exc)
        return ""


def annotate_violation_frame(frame: np.ndarray, speed_kmh: float) -> np.ndarray:
    """Ihlal kaydi icin sade bilgi kutusu ekler."""
    output = frame.copy()
    ts_text = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    lines = [
        ("Ihlal", (0, 0, 255), 1.0, 3),
        (f"Tarih/Saat: {ts_text}", (255, 255, 255), 0.72, 2),
        (f"Hiz: {speed_kmh:.2f} km/s", (255, 255, 255), 0.72, 2),
    ]

    x = 18
    y = 38
    for text, color, scale, thickness in lines:
        cv2.putText(
            output,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += 34
    return output


def default_control_state() -> ControlState:
    """Varsayilan kontrol durumunu uretir."""
    return ControlState()


def load_control_state(logger: logging.Logger) -> ControlState:
    """Kalibrasyon ve profil ayarlarini diskten yukler."""
    state = default_control_state()
    if not CONFIG_FILE.exists():
        return state

    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        state.active_profile = str(raw.get("active_profile", state.active_profile)).upper()
        state.gate_distance_m = float(raw.get("gate_distance_m", state.gate_distance_m))
        state.speed_limit_kmh = float(raw.get("speed_limit_kmh", state.speed_limit_kmh))
        state.test_mode = bool(raw.get("test_mode", state.test_mode))
        loaded_profiles = raw.get("profiles", {})
        for name in ("SOL", "SAG"):
            if name not in loaded_profiles:
                continue
            src = loaded_profiles[name]
            dst = state.profiles[name]
            for key in ("start_line_p1", "start_line_p2", "finish_line_p1", "finish_line_p2"):
                if key in src and len(src[key]) == 2:
                    dst[key] = (int(src[key][0]), int(src[key][1]))
            if "global_ref_polygon" in src and len(src["global_ref_polygon"]) >= 3:
                dst["global_ref_polygon"] = [(int(p[0]), int(p[1])) for p in src["global_ref_polygon"]]
        logger.info("Radar gate config yuklendi: %s", CONFIG_FILE)
    except Exception as exc:
        logger.warning("Radar gate config okunamadi, varsayilan kullaniliyor: %s", exc)
    return state


def save_control_state(control_state: ControlState, logger: logging.Logger) -> None:
    """Kalibrasyon ve profil ayarlarini diskte saklar."""
    try:
        with control_state.lock:
            payload = {
                "active_profile": control_state.active_profile,
                "gate_distance_m": control_state.gate_distance_m,
                "speed_limit_kmh": control_state.speed_limit_kmh,
                "test_mode": control_state.test_mode,
                "profiles": control_state.profiles,
            }
        CONFIG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Radar gate config kaydedildi: %s", CONFIG_FILE)
    except Exception as exc:
        logger.warning("Radar gate config kaydedilemedi: %s", exc)


# ============================================================
# DEBUG GORUNTUSU
# ============================================================


def _draw_polygon(frame: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int], alpha: float = 0.18) -> None:
    """Debug icin yari seffaf ROI cizer."""
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, polygon, color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)


def draw_debug_overlay(
    color_frame: np.ndarray,
    runtime: GateState,
    control_state: ControlState,
    start_p1: tuple[int, int],
    start_p2: tuple[int, int],
    finish_p1: tuple[int, int],
    finish_p2: tuple[int, int],
    left_full: Optional[tuple[tuple[int, int], tuple[int, int]]],
    right_full: Optional[tuple[tuple[int, int], tuple[int, int]]],
    start_polygon: np.ndarray,
    finish_polygon: np.ndarray,
    global_polygon: np.ndarray,
    start_score: MotionScore,
    finish_score: MotionScore,
    global_score: MotionScore,
    process_ms: float,
) -> np.ndarray:
    """Canli debug icin yalnizca ROI bantlarini cizer."""
    frame = color_frame.copy()

    start_color = (0, 255, 255) if time.monotonic() < runtime.start_trigger_flash_until else (0, 220, 220)
    finish_color = (0, 0, 255) if time.monotonic() < runtime.finish_trigger_flash_until else (0, 0, 220)
    _draw_polygon(frame, start_polygon, start_color)
    _draw_polygon(frame, finish_polygon, finish_color)
    if left_full is not None:
        cv2.line(frame, left_full[0], left_full[1], (255, 120, 0), 2, cv2.LINE_AA)
    if right_full is not None:
        cv2.line(frame, right_full[0], right_full[1], (255, 120, 0), 2, cv2.LINE_AA)
    for point in (start_p1, start_p2, finish_p1, finish_p2):
        cv2.circle(frame, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 8, (255, 120, 0), 2, cv2.LINE_AA)
    if SHOW_GLOBAL_REF_OVERLAY:
        cv2.polylines(frame, [global_polygon], True, (255, 128, 0), 2, cv2.LINE_AA)

    with control_state.lock:
        if control_state.menu_open:
            cv2.putText(
                frame,
                f"Profil: {control_state.active_profile} | Mod: {control_state.edit_mode}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return frame


def encode_frame(frame: np.ndarray) -> bytes:
    """MJPEG stream icin JPEG encode eder."""
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), WEB_JPEG_QUALITY])
    return encoded.tobytes() if ok else b""


# ============================================================
# WEB DEBUG
# ============================================================


def create_web_app(web_state: WebState, control_state: ControlState, logger: logging.Logger) -> Any:
    """Cok sade debug web arayuzu."""
    if Flask is None:
        return None

    html = """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Radar Gate Debug</title>
  <style>
    body { background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; padding:12px; overflow-x:hidden; }
    .wrap { max-width:1600px; margin:0 auto; }
    .toolbar { display:flex; gap:10px; margin:0 auto 14px auto; flex-wrap:wrap; align-items:center; justify-content:center; }
    .btn { background:#394487; color:#fff; border:0; border-radius:12px; padding:8px 14px; font-size:14px; min-width:84px; cursor:pointer; white-space:nowrap; }
    .btn.active { background:#d0c149; color:#111; }
    .btn.secondary { background:#4b4b4b; }
    .panel { background:#1b1b1b; border-radius:12px; padding:12px; }
    .grid { display:block; }
    .grid > .panel:last-child { display:none; }
    .menu { display:none; margin-bottom:14px; gap:10px; align-items:center; flex-wrap:wrap; justify-content:center; }
    .menu.open { display:flex; }
    .menu-group { display:flex; align-items:center; gap:8px; }
    .viewer { position:relative; width:min(100%, calc((100vh - 185px) * 1.3333)); max-height:calc(100vh - 185px); aspect-ratio:4/3; margin:0 auto; border-radius:10px; overflow:hidden; background:#000; }
    img { width:100%; border-radius:10px; background:#000; display:block; user-select:none; -webkit-user-drag:none; touch-action:none; pointer-events:none; }
    canvas { position:absolute; inset:0; width:100%; height:100%; touch-action:none; cursor:crosshair; z-index:2; }
    .hud { position:absolute; left:12px; right:12px; bottom:12px; z-index:3; display:flex; justify-content:center; pointer-events:none; }
    .hud-box { background:rgba(0,0,0,0.58); color:#fff; border-radius:12px; padding:8px 12px; font-size:14px; line-height:1.45; display:flex; gap:16px; flex-wrap:wrap; justify-content:center; text-shadow:0 1px 2px rgba(0,0,0,0.45); }
    .hud-box span { white-space:nowrap; }
    .aux-links { display:flex; justify-content:center; gap:18px; margin-top:10px; font-size:15px; }
    .aux-links a { color:#8ab4ff; text-decoration:none; }
    pre { white-space:pre-wrap; word-break:break-word; line-height:1.55; }
    @media (max-width: 1024px), (orientation: landscape) {
      body { padding:10px; }
      .toolbar { gap:8px; margin-bottom:10px; }
      .btn { padding:7px 12px; font-size:13px; min-width:76px; }
      .menu.open { margin-bottom:10px; gap:8px; }
      .viewer { width:min(100%, calc((100vh - 150px) * 1.3333)); max-height:calc(100vh - 150px); }
      .hud-box { font-size:12px; padding:6px 10px; gap:12px; }
      .aux-links { margin-top:8px; font-size:13px; gap:12px; }
    }
    html.is-fullscreen body { padding:8px; }
    html.is-fullscreen .wrap { max-width:none; }
    html.is-fullscreen .toolbar { margin-bottom:8px; }
    html.is-fullscreen .menu.open { margin-bottom:8px; }
    html.is-fullscreen .viewer { width:min(100%, calc((100vh - 105px) * 1.3333)); max-height:calc(100vh - 105px); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <button id="btnSOL" class="btn">SOL</button>
      <button id="btnSAG" class="btn">SAG</button>
      <button id="btnMENU" class="btn">MENU</button>
      <button id="btnViolations" class="btn secondary">Violations</button>
      <button id="btnFullscreen" class="btn secondary">Fullscreen</button>
    </div>
    <div id="menuPanel" class="menu">
      <div class="menu-group">
        <button class="btn" data-mode="edit_start">Edit Start Gate</button>
      </div>
      <div class="menu-group">
        <button class="btn" data-mode="edit_finish">Edit Finish Gate</button>
      </div>
      <div class="menu-group">
        <button id="btnDistance" class="btn">Mesafe Gir</button>
      </div>
      <div class="menu-group">
        <button id="btnSpeedLimit" class="btn">Set Speed Limit</button>
      </div>
      <div class="menu-group">
        <button id="btnTestMode" class="btn">Test Mode</button>
      </div>
      <div class="menu-group">
        <button id="btnClose" class="btn secondary">Close</button>
      </div>
    </div>
    <div class="panel">
      <div class="viewer">
        <img id="stream" src="/video_feed" alt="debug stream" draggable="false">
        <canvas id="overlay"></canvas>
        <div class="hud">
          <div id="hudBox" class="hud-box">Yukleniyor...</div>
        </div>
      </div>
      <div class="aux-links">
        <a id="lastViolationLink" href="#" target="_blank" rel="noopener noreferrer" style="display:none;">Latest Violation</a>
        <a href="/violations" target="_blank" rel="noopener noreferrer">All Violations</a>
      </div>
      <pre id="status" style="display:none;">Yukleniyor...</pre><!--
          <a id="lastViolationLink" href="#" target="_blank" rel="noopener noreferrer" style="color:#8ab4ff; text-decoration:none;">Son İhlal</a>
          <div style="margin-top:8px;">
            <a href="/violations" target="_blank" rel="noopener noreferrer" style="color:#8ab4ff; text-decoration:none;">Tum İhlaller</a>
          </div>
        </div>-->
      </div>
    </div>
  </div>
  <script>
    const FRAME_W = __FRAME_W__;
    const FRAME_H = __FRAME_H__;
    let latestStatus = null;
    let uiHandles = null;
    let dragHandle = null;

    async function postJson(url, body) {
      await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {})
      });
    }

    async function postAction(name) {
      await fetch('/action/' + name, { method: 'POST' });
    }

    async function refreshStatus() {
      if (dragHandle) return;
      try {
        const res = await fetch('/status');
        const data = await res.json();
        latestStatus = data;
        uiHandles = JSON.parse(JSON.stringify(data.handles || {}));
        document.getElementById('status').textContent =
          'state: ' + data.state + '\\n' +
          'message: ' + data.message + '\\n' +
          'fps: ' + data.fps + '\\n' +
          'process_ms: ' + data.process_ms + '\\n' +
          'start_ratio: ' + data.start_ratio + '\\n' +
          'finish_ratio: ' + data.finish_ratio + '\\n' +
          'global_ratio: ' + data.global_ratio + '\\n' +
          'start_peak_ratio: ' + data.start_peak_ratio + '\\n' +
          'finish_peak_ratio: ' + data.finish_peak_ratio + '\\n' +
          'global_peak_ratio: ' + data.global_peak_ratio + '\\n' +
          'start_peak_mean: ' + data.start_peak_mean + '\\n' +
          'finish_peak_mean: ' + data.finish_peak_mean + '\\n' +
          'delta: ' + data.delta + '\\n' +
          'speed_kmh: ' + data.speed_kmh + '\\n' +
          'profile: ' + data.active_profile + '\\n' +
          'edit_mode: ' + data.edit_mode + '\\n' +
          'test_mode: ' + data.test_mode + '\\n' +
          'center_distance_px: ' + data.center_distance_px + '\\n' +
          'gate_distance_m: ' + data.gate_distance_m + '\\n' +
          'speed_limit_kmh: ' + data.speed_limit_kmh + '\\n' +
          'last_violation: ' + data.last_violation;
        document.getElementById('hudBox').innerHTML =
          '<span>FPS: ' + data.fps + '</span>' +
          '<span>process_ms: ' + data.process_ms + '</span>' +
          '<span>start_ratio: ' + data.start_ratio + '</span>' +
          '<span>finish_ratio: ' + data.finish_ratio + '</span>' +
          '<span>delta: ' + data.delta + '</span>' +
          '<span>speed: ' + data.speed_kmh + ' km/h</span>';
        document.getElementById('menuPanel').classList.toggle('open', !!data.menu_open);
        document.getElementById('btnSOL').classList.toggle('active', data.active_profile === 'SOL');
        document.getElementById('btnSAG').classList.toggle('active', data.active_profile === 'SAG');
        document.getElementById('btnMENU').classList.toggle('active', !!data.menu_open);
        const btnTestMode = document.getElementById('btnTestMode');
        btnTestMode.classList.toggle('active', !!data.test_mode);
        btnTestMode.textContent = data.test_mode ? 'Disable Test Mode' : 'Enable Test Mode';
        redrawOverlay();
        const link = document.getElementById('lastViolationLink');
        if (data.last_violation_url) {
          link.style.display = 'inline';
          link.href = data.last_violation_url;
          link.textContent = 'Latest Violation: ' + (data.last_violation || 'Open');
          link.textContent = 'Son İhlal: ' + (data.last_violation || 'Ac');
          link.textContent = 'Latest Violation: ' + (data.last_violation || 'Open');
        } else {
          link.style.display = 'none';
          link.href = '#';
        }
      } catch (err) {
        document.getElementById('status').textContent = 'Status unavailable: ' + err;
        document.getElementById('hudBox').textContent = 'Status unavailable: ' + err;
      }
    }

    function eventToFramePoint(ev) {
      const rect = document.getElementById('overlay').getBoundingClientRect();
      const rx = Math.min(Math.max((ev.clientX - rect.left) / rect.width, 0), 1);
      const ry = Math.min(Math.max((ev.clientY - rect.top) / rect.height, 0), 1);
      return {
        x: Math.round(rx * FRAME_W),
        y: Math.round(ry * FRAME_H)
      };
    }

    function nearestHandle(point) {
      if (!uiHandles) return null;
      const candidates = latestStatus.edit_mode === 'edit_start'
        ? ['start_p1', 'start_p2']
        : latestStatus.edit_mode === 'edit_finish'
          ? ['finish_p1', 'finish_p2']
          : ['start_p1', 'start_p2', 'finish_p1', 'finish_p2'];
      let best = null;
      let bestDist = 1e9;
      for (const name of candidates) {
        const value = uiHandles[name];
        if (!value) continue;
        const dx = value[0] - point.x;
        const dy = value[1] - point.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestDist) {
          bestDist = d;
          best = name;
        }
      }
      return bestDist <= 40 ? best : null;
    }

    document.getElementById('btnSOL').onclick = () => postAction('SOL');
    document.getElementById('btnSAG').onclick = () => postAction('SAG');
    document.getElementById('btnMENU').onclick = () => postAction('MENU');
    document.getElementById('btnViolations').onclick = () => window.open('/violations', '_blank', 'noopener,noreferrer');
    document.getElementById('btnFullscreen').onclick = async () => {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    };
    document.getElementById('btnClose').onclick = () => postAction('KAPAT');
    document.getElementById('btnDistance').onclick = async () => {
      const current = latestStatus ? latestStatus.gate_distance_m : 3.0;
      const value = window.prompt('START-FINISH merkezler arasi mesafe (metre)', String(current));
      if (!value) return;
      const parsed = Number(String(value).replace(',', '.'));
      if (!Number.isFinite(parsed) || parsed <= 0) return;
      await postJson('/api/set_distance', { gate_distance_m: parsed });
      refreshStatus();
    };
    document.getElementById('btnSpeedLimit').onclick = async () => {
      const current = latestStatus ? latestStatus.speed_limit_kmh : 50.0;
      const value = window.prompt('Speed limit (km/h)', String(current));
      if (!value) return;
      const parsed = Number(String(value).replace(',', '.'));
      if (!Number.isFinite(parsed) || parsed <= 0) return;
      await postJson('/api/set_speed_limit', { speed_limit_kmh: parsed });
      refreshStatus();
    };
    document.getElementById('btnTestMode').onclick = async () => {
      await postJson('/api/toggle_test_mode', {});
      refreshStatus();
    };
    document.querySelectorAll('[data-mode]').forEach((btn) => {
      btn.onclick = async () => {
        await postJson('/api/set_edit_mode', { edit_mode: btn.dataset.mode });
        refreshStatus();
      };
    });

    function frameToCanvasPoint(x, y) {
      const canvas = document.getElementById('overlay');
      return {
        x: (x / FRAME_W) * canvas.width,
        y: (y / FRAME_H) * canvas.height
      };
    }

    function drawGuide(ctx, a, b) {
      const pa = frameToCanvasPoint(a[0], a[1]);
      const pb = frameToCanvasPoint(b[0], b[1]);
      ctx.strokeStyle = '#3d8bff';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    function redrawOverlay() {
      const canvas = document.getElementById('overlay');
      const img = document.getElementById('stream');
      const rect = img.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width));
      canvas.height = Math.max(1, Math.round(rect.height));
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!uiHandles) return;

      const h = uiHandles;
      if (h.guide_left_a && h.guide_left_b) drawGuide(ctx, h.guide_left_a, h.guide_left_b);
      if (h.guide_right_a && h.guide_right_b) drawGuide(ctx, h.guide_right_a, h.guide_right_b);

      for (const name of ['start_p1', 'start_p2', 'finish_p1', 'finish_p2']) {
        const p = frameToCanvasPoint(h[name][0], h[name][1]);
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#3d8bff';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    const overlay = document.getElementById('overlay');
    let pendingPoint = null;

    overlay.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      if (!latestStatus || latestStatus.edit_mode === 'none') return;
      dragHandle = nearestHandle(eventToFramePoint(ev));
      if (dragHandle) {
        document.body.style.cursor = 'grabbing';
        overlay.setPointerCapture(ev.pointerId);
      }
    });
    overlay.addEventListener('pointermove', (ev) => {
      if (!dragHandle) return;
      ev.preventDefault();
      const point = eventToFramePoint(ev);
      pendingPoint = point;
      uiHandles[dragHandle] = [point.x, point.y];
      if (dragHandle === 'start_p1' || dragHandle === 'finish_p1') {
        uiHandles.guide_left_a = uiHandles.start_p1;
        uiHandles.guide_left_b = uiHandles.finish_p1;
      }
      if (dragHandle === 'start_p2' || dragHandle === 'finish_p2') {
        uiHandles.guide_right_a = uiHandles.start_p2;
        uiHandles.guide_right_b = uiHandles.finish_p2;
      }
      redrawOverlay();
    });
    overlay.addEventListener('pointerup', async (ev) => {
      if (!dragHandle) return;
      ev.preventDefault();
      const point = pendingPoint || eventToFramePoint(ev);
      const handle = dragHandle;
      dragHandle = null;
      pendingPoint = null;
      document.body.style.cursor = 'default';
      try { overlay.releasePointerCapture(ev.pointerId); } catch (err) {}
      await postJson('/api/move_handle', { handle, x: point.x, y: point.y });
      refreshStatus();
    });
    overlay.addEventListener('pointercancel', () => { dragHandle = null; pendingPoint = null; document.body.style.cursor = 'default'; });

    document.getElementById('stream').addEventListener('load', redrawOverlay);
    window.addEventListener('resize', redrawOverlay);
    document.addEventListener('fullscreenchange', () => {
      document.documentElement.classList.toggle('is-fullscreen', !!document.fullscreenElement);
      setTimeout(redrawOverlay, 30);
    });

    setInterval(refreshStatus, 300);
    refreshStatus();
  </script>
</body>
</html>
"""

    html = html.replace("__FRAME_W__", str(FRAME_WIDTH)).replace("__FRAME_H__", str(FRAME_HEIGHT))

    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(html)

    @app.get("/status")
    def status() -> Any:
        with web_state.lock:
            return jsonify(dict(web_state.latest_status))

    @app.post("/action/<name>")
    def action(name: str) -> Any:
        action_name = str(name).upper()
        with control_state.lock:
            if action_name in ("SOL", "SAG"):
                control_state.active_profile = action_name
                control_state.version += 1
            elif action_name == "MENU":
                control_state.menu_open = not control_state.menu_open
            elif action_name == "KAPAT":
                control_state.menu_open = False
                control_state.edit_mode = "none"
            elif action_name == "KALIBRASYON":
                control_state.menu_open = True
                control_state.edit_mode = "edit_start"
            elif action_name == "RESET":
                control_state.profiles = deepcopy(DEFAULT_PROFILES)
                control_state.gate_distance_m = GATE_DISTANCE_M
                control_state.active_profile = ACTIVE_PROFILE
                control_state.edit_mode = "none"
                control_state.version += 1
        save_control_state(control_state, logger)
        return jsonify({"ok": True})

    @app.post("/api/set_distance")
    def set_distance() -> Any:
        data = request.get_json(silent=True) or {}
        try:
            value = float(data.get("gate_distance_m"))
            if value <= 0:
                raise ValueError("Mesafe sifirdan buyuk olmali")
            with control_state.lock:
                control_state.gate_distance_m = value
            save_control_state(control_state, logger)
            return jsonify({"ok": True})
        except Exception:
            return jsonify({"ok": False}), 400

    @app.post("/api/set_edit_mode")
    def set_edit_mode() -> Any:
        data = request.get_json(silent=True) or {}
        mode = str(data.get("edit_mode", "none"))
        if mode not in ("none", "edit_start", "edit_finish"):
            return jsonify({"ok": False}), 400
        with control_state.lock:
            control_state.edit_mode = mode
            control_state.menu_open = True
        return jsonify({"ok": True})

    @app.post("/api/set_speed_limit")
    def set_speed_limit() -> Any:
        data = request.get_json(silent=True) or {}
        try:
            value = float(data.get("speed_limit_kmh"))
            if value <= 0:
                raise ValueError("Limit sifirdan buyuk olmali")
            with control_state.lock:
                control_state.speed_limit_kmh = value
            save_control_state(control_state, logger)
            return jsonify({"ok": True})
        except Exception:
            return jsonify({"ok": False}), 400

    @app.post("/api/toggle_test_mode")
    def toggle_test_mode() -> Any:
        with control_state.lock:
            control_state.test_mode = not control_state.test_mode
        save_control_state(control_state, logger)
        return jsonify({"ok": True})

    @app.post("/api/move_handle")
    def move_handle() -> Any:
        data = request.get_json(silent=True) or {}
        handle = str(data.get("handle", ""))
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        x = max(0, min(FRAME_WIDTH - 1, x))
        y = max(0, min(FRAME_HEIGHT - 1, y))
        handle_map = {
            "start_p1": "start_line_p1",
            "start_p2": "start_line_p2",
            "finish_p1": "finish_line_p1",
            "finish_p2": "finish_line_p2",
        }
        if handle not in handle_map:
            return jsonify({"ok": False}), 400
        with control_state.lock:
            profile = control_state.profiles[control_state.active_profile]
            profile[handle_map[handle]] = (x, y)
            control_state.version += 1
        save_control_state(control_state, logger)
        return jsonify({"ok": True})

    @app.get("/video_feed")
    def video_feed() -> Any:
        def generate() -> Any:
            while True:
                with web_state.lock:
                    frame = web_state.latest_jpeg
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.03)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/violations")
    def violations_index() -> Any:
        VIOLATION_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(VIOLATION_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = []
        for path in files[:500]:
            items.append(
                f'<li><a href="/violations/{escape(path.name)}" target="_blank" rel="noopener noreferrer">{escape(path.name)}</a></li>'
            )
        html_page = f"""
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Violations</title>
  <style>
    body {{ background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; padding:24px; }}
    .wrap {{ max-width:980px; margin:0 auto; }}
    a {{ color:#8ab4ff; text-decoration:none; }}
    ul {{ line-height:1.8; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Violations</h2>
    <p><a href="/">Geri Don</a></p>
    <ul>{''.join(items) if items else '<li>No violations recorded.</li>'}</ul>
    <div style="margin-top:18px;">
      <p><a href="/violations/csv/event_log.csv" target="_blank" rel="noopener noreferrer">CSV Event Logunu Ac</a></p>
      <form method="post" action="/violations/csv/clear" onsubmit="return confirm('CSV event log icerigi temizlensin mi?');" style="margin-top:10px;">
        <button type="submit">CSV Logunu Temizle</button>
      </form>
    </div>
  </div>
</body>
</html>
"""
        return html_page

    @app.get("/violations/csv/<path:filename>")
    def violations_csv(filename: str) -> Any:
        if filename != "event_log.csv":
            return abort(404)
        ensure_event_log_header()
        try:
            text = CSV_LOG_PATH.read_text(encoding="utf-8")
        except Exception:
            return abort(500)
        return Response(text, mimetype="text/plain; charset=utf-8")

    @app.post("/violations/csv/clear")
    def violations_csv_clear() -> Any:
        try:
            with CSV_LOG_PATH.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=EVENT_LOG_FIELDS)
                writer.writeheader()
        except Exception:
            return abort(500)
        return (
            "<!doctype html><html><head><meta http-equiv='refresh' content='0; url=/violations'></head><body></body></html>"
        )

    @app.get("/violations/raw/<path:filename>")
    def violation_raw(filename: str) -> Any:
        full_path = (VIOLATION_DIR / filename).resolve()
        try:
            full_path.relative_to(VIOLATION_DIR.resolve())
        except Exception:
            return abort(404)
        if not full_path.exists():
            return abort(404)
        return send_from_directory(VIOLATION_DIR, filename)

    @app.get("/violations/<path:filename>")
    def violation_viewer(filename: str) -> Any:
        files = sorted(VIOLATION_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        names = [p.name for p in files]
        if filename not in names:
            return abort(404)

        idx = names.index(filename)
        prev_name = names[idx - 1] if idx > 0 else None
        next_name = names[idx + 1] if idx < len(names) - 1 else None

        prev_link = f'/violations/{escape(prev_name)}' if prev_name else '#'
        next_link = f'/violations/{escape(next_name)}' if next_name else '#'
        prev_disabled = 'disabled' if prev_name is None else ''
        next_disabled = 'disabled' if next_name is None else ''

        html_page = f"""
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(filename)}</title>
  <style>
    body {{ background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; }}
    .topbar {{ padding:14px 18px; display:flex; gap:12px; align-items:center; }}
    .btn {{ background:#394487; color:#fff; border-radius:12px; padding:10px 16px; text-decoration:none; }}
    .btn.secondary {{ background:#4b4b4b; }}
    .btn.disabled {{ background:#2a2a2a; color:#888; pointer-events:none; }}
    .stage {{ position:relative; width:100%; height:calc(100vh - 70px); display:flex; align-items:center; justify-content:center; }}
    img {{ max-width:100%; max-height:100%; object-fit:contain; display:block; }}
    .nav {{ position:absolute; top:50%; transform:translateY(-50%); background:rgba(20,20,20,0.55); color:#fff; border-radius:14px; padding:16px 18px; text-decoration:none; font-size:28px; }}
    .nav.left {{ left:18px; }}
    .nav.right {{ right:18px; }}
    .nav.disabled {{ pointer-events:none; opacity:0.25; }}
  </style>
</head>
<body>
  <div class="topbar">
    <a class="btn secondary" href="/violations">All Violations</a>
    <a class="btn secondary" href="/violations/raw/{escape(filename)}" target="_blank" rel="noopener noreferrer">Raw File</a>
    <form method="post" action="/violations/delete/{escape(filename)}" onsubmit="return confirm('Delete this violation?');" style="margin:0;">
      <button class="btn secondary" type="submit">Delete</button>
    </form>
    <form method="post" action="/violations/delete_all" onsubmit="return confirm('Delete all violations?');" style="margin:0;">
      <button class="btn secondary" type="submit">Delete All</button>
    </form>
    <span>{escape(filename)}</span>
  </div>
  <div class="stage">
    <a class="nav left {prev_disabled}" href="{prev_link}">&#8592;</a>
    <img src="/violations/raw/{escape(filename)}" alt="{escape(filename)}">
    <a class="nav right {next_disabled}" href="{next_link}">&#8594;</a>
  </div>
</body>
</html>
"""
        return html_page

    @app.post("/violations/delete/<path:filename>")
    def violation_delete(filename: str) -> Any:
        files = sorted(VIOLATION_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        names = [p.name for p in files]
        if filename not in names:
            return abort(404)

        target = (VIOLATION_DIR / filename).resolve()
        try:
            target.relative_to(VIOLATION_DIR.resolve())
        except Exception:
            return abort(404)

        next_target = None
        idx = names.index(filename)
        if idx + 1 < len(names):
            next_target = names[idx + 1]
        elif idx > 0:
            next_target = names[idx - 1]

        try:
            target.unlink(missing_ok=False)
        except Exception:
            return abort(500)

        if next_target is not None and next_target != filename:
            return (
                "<!doctype html><html><head><meta http-equiv='refresh' content='0; url=/violations/"
                + escape(next_target)
                + "'></head><body></body></html>"
            )
        return (
            "<!doctype html><html><head><meta http-equiv='refresh' content='0; url=/violations'></head><body></body></html>"
        )

    @app.post("/violations/delete_all")
    def violation_delete_all() -> Any:
        try:
            for path in VIOLATION_DIR.glob("*.jpg"):
                path.unlink(missing_ok=True)
        except Exception:
            return abort(500)
        return (
            "<!doctype html><html><head><meta http-equiv='refresh' content='0; url=/violations'></head><body></body></html>"
        )

    return app


def start_web_thread(web_state: WebState, control_state: ControlState, logger: logging.Logger) -> Optional[threading.Thread]:
    """Flask varsa arka planda web debug sunucusunu baslatir."""
    app = create_web_app(web_state, control_state, logger)
    if app is None:
        logger.warning("Flask bulunamadi, web debug kapali.")
        return None

    def runner() -> None:
        logger.info("Web debug baslatiliyor: http://%s:%s", WEB_HOST, WEB_PORT)
        app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True, use_reloader=False)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


# ============================================================
# ANA AKIS
# ============================================================


def build_logger() -> logging.Logger:
    """Sade logger kurar."""
    logger = logging.getLogger("radar_gate")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def main() -> None:
    """Uygulamayi calistirir."""
    logger = build_logger()
    if start_vehicle_validation_worker is not None:
        try:
            start_vehicle_validation_worker()
        except Exception as exc:
            logger.error("Arac dogrulama worker baslatilamadi: %s", exc)
    else:
        logger.warning("Arac dogrulama worker modulu yuklenemedi; AI/OCR kapali.")
    control_state = load_control_state(logger)
    geometry = get_profile_geometry(control_state)
    start_p1 = geometry["start_p1"]
    start_p2 = geometry["start_p2"]
    finish_p1 = geometry["finish_p1"]
    finish_p2 = geometry["finish_p2"]
    left_full = geometry["left_full"]
    right_full = geometry["right_full"]
    start_polygon = geometry["start_polygon"]
    finish_polygon = geometry["finish_polygon"]
    global_polygon = geometry["global_polygon"]

    analysis_size = (ANALYSIS_WIDTH, ANALYSIS_HEIGHT)
    display_size = (FRAME_WIDTH, FRAME_HEIGHT)

    start_polygon_analysis = scale_polygon(start_polygon, display_size, analysis_size)
    finish_polygon_analysis = scale_polygon(finish_polygon, display_size, analysis_size)
    global_polygon_analysis = scale_polygon(global_polygon, display_size, analysis_size)
    start_p1_analysis = tuple(scale_polygon(np.array([start_p1], dtype=np.int32), display_size, analysis_size)[0])
    start_p2_analysis = tuple(scale_polygon(np.array([start_p2], dtype=np.int32), display_size, analysis_size)[0])
    finish_p1_analysis = tuple(scale_polygon(np.array([finish_p1], dtype=np.int32), display_size, analysis_size)[0])
    finish_p2_analysis = tuple(scale_polygon(np.array([finish_p2], dtype=np.int32), display_size, analysis_size)[0])

    start_region = build_roi_region((ANALYSIS_HEIGHT, ANALYSIS_WIDTH), start_polygon_analysis)
    finish_region = build_roi_region((ANALYSIS_HEIGHT, ANALYSIS_WIDTH), finish_polygon_analysis)
    global_region = build_roi_region((ANALYSIS_HEIGHT, ANALYSIS_WIDTH), global_polygon_analysis)
    set_region_centerline(start_region, start_p1_analysis, start_p2_analysis, thickness=CENTERLINE_THICKNESS_PX)
    set_region_centerline(finish_region, finish_p1_analysis, finish_p2_analysis, thickness=FINISH_CENTERLINE_THICKNESS_PX)

    camera = init_camera(logger)
    frame_store = FrameStore()
    start_camera_thread(camera, frame_store, logger)
    web_state = WebState()
    start_web_thread(web_state, control_state, logger)

    runtime = GateState(warmup_until=time.monotonic() + WARMUP_SECONDS)
    last_frame_sequence = 0
    applied_version = -1

    start_bg: Optional[np.ndarray] = None
    finish_bg: Optional[np.ndarray] = None
    global_bg: Optional[np.ndarray] = None
    prev_start_crop_u8: Optional[np.ndarray] = None
    prev_finish_crop_u8: Optional[np.ndarray] = None
    prev_global_crop_u8: Optional[np.ndarray] = None
    last_web_publish = 0.0

    try:
        while True:
            captured, last_frame_sequence = frame_store.get_if_newer(last_frame_sequence)
            if captured is None:
                time.sleep(0.002)
                continue

            loop_started = time.perf_counter()
            with control_state.lock:
                current_version = control_state.version
                current_distance_m = control_state.gate_distance_m
                current_speed_limit_kmh = control_state.speed_limit_kmh
                current_test_mode = control_state.test_mode
                current_active_profile = control_state.active_profile
                current_menu_open = control_state.menu_open
                current_edit_mode = control_state.edit_mode
            if current_version != applied_version:
                geometry = get_profile_geometry(control_state)
                start_p1 = geometry["start_p1"]
                start_p2 = geometry["start_p2"]
                finish_p1 = geometry["finish_p1"]
                finish_p2 = geometry["finish_p2"]
                left_full = geometry["left_full"]
                right_full = geometry["right_full"]
                start_polygon = geometry["start_polygon"]
                finish_polygon = geometry["finish_polygon"]
                global_polygon = geometry["global_polygon"]
                start_polygon_analysis = scale_polygon(start_polygon, display_size, analysis_size)
                finish_polygon_analysis = scale_polygon(finish_polygon, display_size, analysis_size)
                global_polygon_analysis = scale_polygon(global_polygon, display_size, analysis_size)
                start_p1_analysis = tuple(scale_polygon(np.array([start_p1], dtype=np.int32), display_size, analysis_size)[0])
                start_p2_analysis = tuple(scale_polygon(np.array([start_p2], dtype=np.int32), display_size, analysis_size)[0])
                finish_p1_analysis = tuple(scale_polygon(np.array([finish_p1], dtype=np.int32), display_size, analysis_size)[0])
                finish_p2_analysis = tuple(scale_polygon(np.array([finish_p2], dtype=np.int32), display_size, analysis_size)[0])
                start_region = build_roi_region((ANALYSIS_HEIGHT, ANALYSIS_WIDTH), start_polygon_analysis)
                finish_region = build_roi_region((ANALYSIS_HEIGHT, ANALYSIS_WIDTH), finish_polygon_analysis)
                global_region = build_roi_region((ANALYSIS_HEIGHT, ANALYSIS_WIDTH), global_polygon_analysis)
                set_region_centerline(start_region, start_p1_analysis, start_p2_analysis, thickness=CENTERLINE_THICKNESS_PX)
                set_region_centerline(finish_region, finish_p1_analysis, finish_p2_analysis, thickness=FINISH_CENTERLINE_THICKNESS_PX)
                start_bg = None
                finish_bg = None
                global_bg = None
                prev_start_crop_u8 = None
                prev_finish_crop_u8 = None
                prev_global_crop_u8 = None
                runtime.state = "WARMUP"
                runtime.last_message = "WARMUP"
                runtime.warmup_until = time.monotonic() + WARMUP_SECONDS
                runtime.t1 = None
                runtime.t2 = None
                runtime.delta_time = None
                runtime.speed_kmh = None
                runtime.start_confirm_count = 0
                runtime.finish_confirm_count = 0
                runtime.start_candidate_latched = False
                runtime.finish_candidate_latched = False
                runtime.start_candidate_since = None
                runtime.finish_candidate_since = None
                runtime.start_centerline_since = None
                runtime.finish_centerline_since = None
                runtime.finish_exit_pending = False
                runtime.current_event_id = ""
                runtime.event_log_pending = False
                runtime.pending_event_valid = False
                runtime.pending_event_message = ""
                reset_event_peaks(runtime)
                applied_version = current_version

            gray = captured.gray
            if captured.color_frame is not None and (USE_COLOR_DEBUG_PREVIEW or USE_HIGH_RES_DEBUG_PREVIEW):
                if DEBUG_PREVIEW_COLOR:
                    debug_base_frame = captured.color_frame.copy()
                else:
                    preview_gray = cv2.cvtColor(captured.color_frame, cv2.COLOR_BGR2GRAY)
                    debug_base_frame = cv2.cvtColor(preview_gray, cv2.COLOR_GRAY2BGR)
            else:
                debug_gray = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
                debug_base_frame = cv2.cvtColor(debug_gray, cv2.COLOR_GRAY2BGR)
            gray_f32 = gray.astype(np.float32)
            now = captured.captured_at

            start_x, start_y, start_w, start_h = start_region.bbox
            finish_x, finish_y, finish_w, finish_h = finish_region.bbox
            global_x, global_y, global_w, global_h = global_region.bbox

            start_crop = gray_f32[start_y:start_y + start_h, start_x:start_x + start_w]
            finish_crop = gray_f32[finish_y:finish_y + finish_h, finish_x:finish_x + finish_w]
            global_crop = gray_f32[global_y:global_y + global_h, global_x:global_x + global_w]
            start_crop_u8 = start_crop.astype(np.uint8)
            finish_crop_u8 = finish_crop.astype(np.uint8)
            global_crop_u8 = global_crop.astype(np.uint8)

            # Ilk karelerde background'u dogrudan olustur.
            if start_bg is None:
                start_bg = start_crop.copy()
                finish_bg = finish_crop.copy()
                global_bg = global_crop.copy()
                prev_start_crop_u8 = start_crop_u8.copy()
                prev_finish_crop_u8 = finish_crop_u8.copy()
                prev_global_crop_u8 = global_crop_u8.copy()

            assert finish_bg is not None and global_bg is not None

            start_score = compute_roi_motion_score(
                start_crop_u8,
                start_bg.astype(np.uint8),
                start_region.local_mask,
            )
            finish_score = compute_roi_motion_score(
                finish_crop_u8,
                finish_bg.astype(np.uint8),
                finish_region.local_mask,
            )
            global_score = compute_roi_motion_score(
                global_crop_u8,
                global_bg.astype(np.uint8),
                global_region.local_mask,
            )
            start_temporal_score = compute_roi_motion_score(
                start_crop_u8,
                prev_start_crop_u8 if prev_start_crop_u8 is not None else start_crop_u8,
                start_region.local_mask,
            )
            finish_temporal_score = compute_roi_motion_score(
                finish_crop_u8,
                prev_finish_crop_u8 if prev_finish_crop_u8 is not None else finish_crop_u8,
                finish_region.local_mask,
            )
            start_center_hit = centerline_touched(
                start_crop_u8,
                start_bg.astype(np.uint8),
                start_region.centerline_mask,
                min_changed_pixels=START_CENTERLINE_MIN_PIXELS,
                min_ratio=START_CENTERLINE_MIN_RATIO,
            )
            finish_center_hit = centerline_touched(
                finish_crop_u8,
                finish_bg.astype(np.uint8),
                finish_region.centerline_mask,
                min_changed_pixels=FINISH_CENTERLINE_MIN_PIXELS,
                min_ratio=FINISH_CENTERLINE_MIN_RATIO,
            )

            global_triggered = (
                global_score.mean_abs_diff >= GLOBAL_MEAN_THRESHOLD
                or global_score.changed_pixel_ratio >= GLOBAL_RATIO_THRESHOLD
            )

            start_suppression_ratio = global_score.changed_pixel_ratio / (start_score.changed_pixel_ratio + 1e-6)
            finish_suppression_ratio = global_score.changed_pixel_ratio / (finish_score.changed_pixel_ratio + 1e-6)

            start_high = (
                start_score.mean_abs_diff >= ROI_MEAN_HIGH
                or start_score.changed_pixel_ratio >= ROI_RATIO_HIGH
            )
            start_motion_high = (
                start_temporal_score.mean_abs_diff >= TEMPORAL_MEAN_HIGH
                or start_temporal_score.changed_pixel_ratio >= TEMPORAL_RATIO_HIGH
            )
            start_low_released = (
                start_score.mean_abs_diff < ROI_MEAN_LOW
                and start_score.changed_pixel_ratio < ROI_RATIO_LOW
            )
            finish_high = (
                finish_score.mean_abs_diff >= FINISH_MEAN_HIGH
                or finish_score.changed_pixel_ratio >= FINISH_RATIO_HIGH
            )
            finish_motion_high = (
                finish_temporal_score.mean_abs_diff >= TEMPORAL_MEAN_HIGH
                or finish_temporal_score.changed_pixel_ratio >= TEMPORAL_RATIO_HIGH
            )
            finish_low_released = (
                finish_score.mean_abs_diff < FINISH_MEAN_LOW
                and finish_score.changed_pixel_ratio < FINISH_RATIO_LOW
            )

            if not runtime.start_candidate_latched:
                runtime.start_candidate_latched = start_high and start_motion_high
            elif start_low_released:
                runtime.start_candidate_latched = False
                runtime.start_candidate_since = None
                runtime.start_centerline_since = None

            if not runtime.finish_candidate_latched:
                runtime.finish_candidate_latched = finish_high and finish_motion_high
            elif finish_low_released:
                runtime.finish_candidate_latched = False
                runtime.finish_candidate_since = None
                runtime.finish_centerline_since = None

            start_candidate = (
                runtime.start_candidate_latched
                and not global_triggered
                and start_suppression_ratio <= SUPPRESSION_RATIO_MAX
            )
            finish_candidate = (
                runtime.finish_candidate_latched
                and not global_triggered
                and finish_suppression_ratio <= SUPPRESSION_RATIO_MAX
            )
            start_center_ready = (
                start_center_hit
                and not global_triggered
                and start_suppression_ratio <= SUPPRESSION_RATIO_MAX
                and start_motion_high
            )
            finish_center_ready = (
                finish_center_hit
                and not global_triggered
                and finish_suppression_ratio <= SUPPRESSION_RATIO_MAX
                and finish_motion_high
            )

            if runtime.state in ("WAIT_FINISH", "COOLDOWN") or runtime.current_event_id:
                runtime.start_peak_ratio = max(runtime.start_peak_ratio, start_score.changed_pixel_ratio)
                runtime.finish_peak_ratio = max(runtime.finish_peak_ratio, finish_score.changed_pixel_ratio)
                runtime.global_peak_ratio = max(runtime.global_peak_ratio, global_score.changed_pixel_ratio)
                runtime.start_peak_mean = max(runtime.start_peak_mean, start_score.mean_abs_diff)
                runtime.finish_peak_mean = max(runtime.finish_peak_mean, finish_score.mean_abs_diff)

            update_state_machine(
                runtime,
                now,
                start_candidate,
                finish_candidate,
                start_center_ready,
                finish_center_ready,
                current_distance_m,
            )

            # Background guncelleme kurallari durum bazli.
            if now < runtime.warmup_until:
                update_background(start_bg, start_crop, start_region.local_mask, BG_ALPHA_IDLE)
                update_background(finish_bg, finish_crop, finish_region.local_mask, BG_ALPHA_IDLE)
                update_background(global_bg, global_crop, global_region.local_mask, BG_ALPHA_GLOBAL)
            elif runtime.state == "IDLE":
                if not start_candidate:
                    update_background(start_bg, start_crop, start_region.local_mask, BG_ALPHA_IDLE)
                if not finish_candidate:
                    update_background(finish_bg, finish_crop, finish_region.local_mask, BG_ALPHA_IDLE)
                if not global_triggered:
                    update_background(global_bg, global_crop, global_region.local_mask, BG_ALPHA_GLOBAL)
            elif runtime.state == "WAIT_FINISH":
                # START background'u bekleme boyunca tamamen dondur.
                if not finish_candidate:
                    update_background(finish_bg, finish_crop, finish_region.local_mask, BG_ALPHA_IDLE)
                if not global_triggered:
                    update_background(global_bg, global_crop, global_region.local_mask, BG_ALPHA_GLOBAL)
            elif runtime.state == "COOLDOWN":
                update_background(start_bg, start_crop, start_region.local_mask, BG_ALPHA_COOLDOWN)
                update_background(finish_bg, finish_crop, finish_region.local_mask, BG_ALPHA_COOLDOWN)
                if not global_triggered:
                    update_background(global_bg, global_crop, global_region.local_mask, BG_ALPHA_GLOBAL)

            prev_start_crop_u8 = start_crop_u8.copy()
            prev_finish_crop_u8 = finish_crop_u8.copy()
            prev_global_crop_u8 = global_crop_u8.copy()

            saved_path = ""
            # Gecerli hiz varsa ve limit asildiysa bu dongude tek kez kaydet.
            if (
                runtime.state == "COOLDOWN"
                and runtime.speed_kmh is not None
                and runtime.speed_kmh > current_speed_limit_kmh
                and not runtime.violation_saved_this_cycle
            ):
                save_base = get_color_frame(camera)
                if save_base is None:
                    save_base = debug_base_frame.copy()
                annotated_for_save = annotate_violation_frame(save_base, runtime.speed_kmh)
                saved_path = save_violation_frame(annotated_for_save, runtime.speed_kmh, runtime.current_event_id, logger)
                if saved_path:
                    runtime.last_violation_text = Path(saved_path).name
                    runtime.last_message = "VIOLATION SAVED"
                    runtime.violation_saved_this_cycle = True
                    if add_vehicle_validation_job is not None:
                        try:
                            queued = add_vehicle_validation_job(
                                saved_path,
                                float(runtime.speed_kmh),
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                runtime.current_event_id,
                            )
                            if not queued:
                                logger.error("Arac dogrulama kuyrugu dolu; fotograf korundu: %s", saved_path)
                        except Exception as exc:
                            logger.error("Arac dogrulama isi kuyruga atilamadi; fotograf korundu: %s", exc)

            if runtime.event_log_pending:
                violation_saved = bool(runtime.violation_saved_this_cycle)
                image_filename = runtime.last_violation_text if violation_saved and runtime.last_violation_text not in ("", "--") else ""
                should_log = current_test_mode or violation_saved
                if should_log:
                    log_ok = append_event_log(
                        {
                            "event_id": runtime.current_event_id or generate_event_id(),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "profile": current_active_profile,
                            "gate_distance_m": current_distance_m,
                            "speed_limit_kmh": current_speed_limit_kmh,
                            "delta": "" if runtime.delta_time is None else round(runtime.delta_time, 3),
                            "speed_kmh": "" if runtime.speed_kmh is None else round(runtime.speed_kmh, 2),
                            "start_peak_ratio": round(runtime.start_peak_ratio, 4),
                            "finish_peak_ratio": round(runtime.finish_peak_ratio, 4),
                            "global_peak_ratio": round(runtime.global_peak_ratio, 4),
                            "start_peak_mean": round(runtime.start_peak_mean, 3),
                            "finish_peak_mean": round(runtime.finish_peak_mean, 3),
                            "fps": round(runtime.fps, 1),
                            "process_ms": round(runtime.last_process_ms, 1),
                            "valid": "true" if runtime.pending_event_valid else "false",
                            "message": runtime.pending_event_message,
                            "violation_saved": "true" if violation_saved else "false",
                            "image_filename": image_filename,
                        }
                    )
                    if not log_ok:
                        logger.error("CSV event log yazilamadi.")
                runtime.event_log_pending = False
                runtime.current_event_id = ""

            if runtime.last_frame_ts is not None:
                dt = now - runtime.last_frame_ts
                if dt > 0:
                    current_fps = 1.0 / dt
                    runtime.fps = current_fps if runtime.fps == 0.0 else ((runtime.fps * 0.8) + (current_fps * 0.2))
            runtime.last_frame_ts = now

            runtime.last_process_ms = (time.perf_counter() - loop_started) * 1000.0

            debug_frame = draw_debug_overlay(
                debug_base_frame,
                runtime,
                control_state,
                start_p1,
                start_p2,
                finish_p1,
                finish_p2,
                left_full,
                right_full,
                start_polygon,
                finish_polygon,
                global_polygon,
                start_score,
                finish_score,
                global_score,
                runtime.last_process_ms,
            )

            last_violation_name = runtime.last_violation_text
            if last_violation_name not in ("", "--"):
                last_violation_path = VIOLATION_DIR / last_violation_name
                if not last_violation_path.is_file():
                    runtime.last_violation_text = "--"
                    last_violation_name = "--"

            with web_state.lock:
                if now - last_web_publish >= (1.0 / WEB_STREAM_MAX_FPS):
                    web_state.latest_jpeg = encode_frame(debug_frame)
                    last_web_publish = now
                web_state.latest_status = {
                    "state": runtime.state,
                    "message": runtime.last_message,
                    "fps": round(runtime.fps, 1),
                    "process_ms": round(runtime.last_process_ms, 1),
                    "start_ratio": round(start_score.changed_pixel_ratio, 4),
                    "finish_ratio": round(finish_score.changed_pixel_ratio, 4),
                    "global_ratio": round(global_score.changed_pixel_ratio, 4),
                    "start_peak_ratio": round(runtime.start_peak_ratio, 4),
                    "finish_peak_ratio": round(runtime.finish_peak_ratio, 4),
                    "global_peak_ratio": round(runtime.global_peak_ratio, 4),
                    "start_peak_mean": round(runtime.start_peak_mean, 3),
                    "finish_peak_mean": round(runtime.finish_peak_mean, 3),
                    "delta": "--" if runtime.delta_time is None else round(runtime.delta_time, 3),
                    "speed_kmh": "--" if runtime.speed_kmh is None else round(runtime.speed_kmh, 2),
                    "last_violation": last_violation_name,
                    "last_violation_url": f"/violations/{last_violation_name}" if last_violation_name not in ("", "--") else "",
                    "active_profile": current_active_profile,
                    "test_mode": current_test_mode,
                    "menu_open": current_menu_open,
                    "edit_mode": current_edit_mode,
                    "gate_distance_m": round(current_distance_m, 2),
                    "speed_limit_kmh": round(current_speed_limit_kmh, 2),
                    "center_distance_px": round(center_distance_px(start_p1, start_p2, finish_p1, finish_p2), 1),
                    "handles": {
                        "start_p1": [start_p1[0], start_p1[1]],
                        "start_p2": [start_p2[0], start_p2[1]],
                        "finish_p1": [finish_p1[0], finish_p1[1]],
                        "finish_p2": [finish_p2[0], finish_p2[1]],
                        "guide_left_a": [left_full[0][0], left_full[0][1]] if left_full is not None else [start_p1[0], start_p1[1]],
                        "guide_left_b": [left_full[1][0], left_full[1][1]] if left_full is not None else [finish_p1[0], finish_p1[1]],
                        "guide_right_a": [right_full[0][0], right_full[0][1]] if right_full is not None else [start_p2[0], start_p2[1]],
                        "guide_right_b": [right_full[1][0], right_full[1][1]] if right_full is not None else [finish_p2[0], finish_p2[1]],
                    },
                }

    except KeyboardInterrupt:
        logger.info("Kapatiliyor...")
    finally:
        frame_store.stopped = True
        if stop_vehicle_validation_worker is not None:
            try:
                stop_vehicle_validation_worker()
            except Exception:
                pass
        if camera.mode == "picamera2":
            try:
                camera.main.stop()
            except Exception:
                pass
        else:
            try:
                camera.main.release()
            except Exception:
                pass


if __name__ == "__main__":
    main()
