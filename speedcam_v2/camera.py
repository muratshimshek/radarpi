"""Threaded CSI/OpenCV camera capture source."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2

try:
    from picamera2 import Picamera2  # type: ignore
except ImportError:  # pragma: no cover - only on Pi
    Picamera2 = None


@dataclass
class TimedFrame:
    """One captured frame with monotonic timestamp."""

    frame: any
    timestamp: float


class CameraSource(threading.Thread):
    """Dedicated capture thread feeding a bounded queue."""

    def __init__(self, config: dict, output_queue: "queue.Queue[TimedFrame]", logger: logging.Logger) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.output_queue = output_queue
        self.logger = logger
        self._stop_event = threading.Event()
        self.capture = None
        self.picam = None
        self.is_video_file = False
        self.frame_interval_seconds = 0.0
        self.next_frame_due = 0.0
        self.output_width = int(config["camera"]["width"])
        self.output_height = int(config["camera"]["height"])

    def _apply_crop(self, frame):
        """Crop and resize a frame when camera.crop is configured."""
        crop = self.config["camera"].get("crop") or []
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            x, y, w, h = [int(value) for value in crop]
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            frame = frame[y:y + h, x:x + w]
        if frame.shape[1] != self.output_width or frame.shape[0] != self.output_height:
            frame = cv2.resize(frame, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR)
        return frame

    def _open(self) -> None:
        """Open either Picamera2 or cv2.VideoCapture."""
        camera_cfg = self.config["camera"]
        width = int(camera_cfg["width"])
        height = int(camera_cfg["height"])
        fps = int(camera_cfg["fps"])
        if camera_cfg.get("use_picamera2", False) and Picamera2 is not None:
            self.picam = Picamera2()
            preview_config = self.picam.create_preview_configuration(main={"size": (width, height), "format": "BGR888"})
            self.picam.configure(preview_config)
            self.picam.start()
            self.logger.info("Picamera2 capture baslatildi.")
            return
        source = camera_cfg["source"]
        self.is_video_file = isinstance(source, str) and "://" not in source and not str(source).isdigit()
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        if not self.capture.isOpened():
            raise RuntimeError("Kamera acilamadi.")
        if self.is_video_file and camera_cfg.get("replay_realtime", True):
            detected_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
            target_fps = detected_fps if detected_fps > 1.0 else float(fps)
            speed_multiplier = float(camera_cfg.get("replay_speed_multiplier", 1.0))
            effective_fps = max(target_fps * max(speed_multiplier, 0.1), 1.0)
            self.frame_interval_seconds = 1.0 / effective_fps
            self.next_frame_due = time.monotonic()
            self.logger.info(
                "Replay dosyasi gercek zamanli oynatilacak: kaynak %.2f FPS, efektif %.2f FPS",
                target_fps,
                effective_fps,
            )
        self.logger.info("OpenCV VideoCapture baslatildi.")

    def run(self) -> None:
        """Continuously capture frames and keep queue fresh."""
        self._open()
        while not self._stop_event.is_set():
            try:
                if self.picam is not None:
                    frame = self.picam.capture_array()
                else:
                    if self.is_video_file and self.frame_interval_seconds > 0.0:
                        now = time.monotonic()
                        if now < self.next_frame_due:
                            time.sleep(min(self.next_frame_due - now, 0.02))
                    ok, frame = self.capture.read()
                    if not ok:
                        if self.is_video_file and self.config["camera"].get("replay_loop", True):
                            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.next_frame_due = time.monotonic()
                            continue
                        time.sleep(0.02)
                        continue
                    if self.is_video_file and self.frame_interval_seconds > 0.0:
                        self.next_frame_due = max(self.next_frame_due + self.frame_interval_seconds, time.monotonic())
                frame = self._apply_crop(frame)
                item = TimedFrame(frame=frame, timestamp=time.monotonic())
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.output_queue.put_nowait(item)
            except Exception:  # pragma: no cover - runtime protection
                self.logger.exception("Capture thread hata verdi.")
                time.sleep(0.1)

    def stop(self) -> None:
        """Signal thread to stop and release capture resources."""
        self._stop_event.set()
        if self.picam is not None:
            self.picam.stop()
        if self.capture is not None:
            self.capture.release()
