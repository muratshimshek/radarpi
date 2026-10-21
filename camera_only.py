#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    from flask import Flask, Response, render_template_string
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Flask kurulu degil. sudo apt install python3-flask -y") from exc

try:
    from picamera2 import Picamera2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Picamera2 kurulu degil. sudo apt install python3-picamera2 -y") from exc


FRAME_WIDTH = 424
FRAME_HEIGHT = 240
TARGET_FPS = 30
STREAM_FPS_LIMIT = 15.0
JPEG_QUALITY = 72
HTTP_PORT = 5001
FPS_LOG_INTERVAL_SECONDS = 1.0


@dataclass
class StreamState:
    latest_jpeg: Optional[bytes] = None
    latest_fps: float = 0.0
    last_publish_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("camera_only")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def draw_overlay(frame: np.ndarray, fps: float) -> np.ndarray:
    output = frame.copy()
    cv2.putText(
        output,
        f"Camera FPS: {fps:.1f}",
        (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def create_app(state: StreamState) -> Flask:
    app = Flask(__name__)

    page = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Only</title>
  <style>
    body { background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; padding:16px; text-align:center; }
    img { width:100%; max-width:900px; border:2px solid #444; border-radius:8px; }
    .meta { margin:10px 0 16px; font-size:14px; color:#bbb; }
  </style>
</head>
<body>
  <h2>Camera Only Test</h2>
  <div class="meta">Port: 5001 | Sadece CSI kamera + web stream</div>
  <img src="/video_feed" alt="Camera only stream">
</body>
</html>"""

    @app.get("/")
    def index() -> str:
        return render_template_string(page)

    @app.get("/video_feed")
    def video_feed() -> Response:
        def generate() -> bytes:
            while True:
                with state.lock:
                    current = state.latest_jpeg
                if current is None:
                    time.sleep(0.03)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + current + b"\r\n"
                )
                time.sleep(1.0 / max(STREAM_FPS_LIMIT, 1.0))

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def camera_loop(state: StreamState, logger: logging.Logger) -> None:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"},
        controls={"FrameRate": TARGET_FPS},
        buffer_count=2,
        queue=False,
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.6)

    prev_time = time.time()
    last_log_at = prev_time

    try:
        while True:
            frame = picam2.capture_array("main")
            if frame is None:
                time.sleep(0.01)
                continue

            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = frame[:, :, ::-1].copy()

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if (now - last_log_at) >= FPS_LOG_INTERVAL_SECONDS:
                logger.info("Camera-only FPS=%.1f", fps)
                last_log_at = now

            min_interval = 1.0 / max(STREAM_FPS_LIMIT, 1.0)
            with state.lock:
                should_publish = (now - state.last_publish_at) >= min_interval

            if should_publish:
                overlay = draw_overlay(frame, fps)
                ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok:
                    with state.lock:
                        state.latest_jpeg = encoded.tobytes()
                        state.latest_fps = fps
                        state.last_publish_at = now
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass


def main() -> int:
    logger = setup_logging()
    state = StreamState()

    app = create_app(state)
    capture_thread = threading.Thread(target=camera_loop, args=(state, logger), name="camera-only-loop", daemon=True)
    capture_thread.start()

    logger.info("Camera-only test baslatildi: http://0.0.0.0:%s", HTTP_PORT)
    app.run(host="0.0.0.0", port=HTTP_PORT, debug=False, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
