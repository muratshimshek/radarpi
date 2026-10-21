"""SpeedCam V2 runtime entrypoint."""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from .camera import CameraSource, TimedFrame
from .capture import FrameBuffer, save_capture_set
from .config import BASE_DIR, DEFAULT_CONFIG_PATH, load_config
from .detector import VehicleDetector
from .homography import HomographyManager
from .line_crossing import LineCrossingDetector
from .speed_estimator import SpeedEstimator
from .tracker import MultiObjectTracker
from .ui import draw_hud, draw_profile_guides, draw_track
from .webui import WebState, create_app, encode_frame


class ProcessingEngine(threading.Thread):
    """Consume captured frames, run pipeline, and publish output."""

    def __init__(
        self,
        config: dict,
        frame_queue: "queue.Queue[TimedFrame]",
        web_state: WebState,
        logger: logging.Logger,
        max_runtime_seconds: float | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.frame_queue = frame_queue
        self.web_state = web_state
        self.logger = logger
        self.stop_event = threading.Event()
        self.started_at = time.monotonic()
        self.max_runtime_seconds = max_runtime_seconds
        self.detector = VehicleDetector(config, logger)
        self.tracker = MultiObjectTracker(config)
        self.homography = HomographyManager(
            enabled=bool(config["homography"]["enabled"]),
            save_path=BASE_DIR / config["homography"]["save_path"],
            image_points=config["homography"]["image_points"],
            world_points=config["homography"]["world_points"],
        )
        self.speed_estimator = SpeedEstimator(config, self.homography)
        self.frame_buffer = FrameBuffer(config["capture"]["buffer_size"])
        self.crossing_detectors: dict[int, LineCrossingDetector] = {}
        self.last_timestamp = None
        self.smoothed_fps = 0.0
        self.last_speed_kmh: float | None = None
        self.last_speed_until: float = 0.0
        self.global_status_text = "Bekleniyor"

    def stop(self) -> None:
        """Signal thread to stop."""
        self.stop_event.set()

    def run(self) -> None:
        """Main processing loop."""
        while not self.stop_event.is_set():
            if self.max_runtime_seconds is not None and (time.monotonic() - self.started_at) >= self.max_runtime_seconds:
                self.logger.info("Maksimum replay suresi doldu, isleme durduruluyor.")
                self.stop_event.set()
                break
            try:
                timed_frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame = timed_frame.frame
            timestamp = timed_frame.timestamp
            self.frame_buffer.push(frame, timestamp)

            active_profile_name = self.config["app"]["active_profile"]
            profile = self.config["profiles"][active_profile_name]
            self.detector.set_active_profile(active_profile_name)
            self.tracker.set_active_profile(profile)

            detections = self.detector.detect(frame, timestamp, profile)
            tracks = self.tracker.update(detections, timestamp)

            annotated = frame.copy()
            if self.config["app"].get("draw_guides", True):
                draw_profile_guides(
                    annotated,
                    profile,
                    show_roi_overlay=bool(self.config["app"].get("show_roi_overlay", True)),
                )
            active_ids: set[int] = set()

            for track in tracks:
                active_ids.add(track.track_id)
                if track.track_id not in self.crossing_detectors:
                    self.crossing_detectors[track.track_id] = LineCrossingDetector(
                        yellow_line=profile["yellow_line"],
                        red_line=profile["red_line"],
                        lane_polygon=profile["lane_polygon"],
                    )
                detector = self.crossing_detectors[track.track_id]
                detector.yellow_line = profile["yellow_line"]
                detector.red_line = profile["red_line"]
                detector.lane_polygon = profile["lane_polygon"]
                crossing = detector.update(track.track_id, track.bottom_center)
                stable = track.hit_streak >= int(self.config["tracking"]["stable_hits"]) and not track.is_predicted
                snapshot = self.speed_estimator.update(track, crossing, stable)
                state = self.speed_estimator.get_or_create(track.track_id)
                draw_track(annotated, track, state, self.smoothed_fps)

                if snapshot is not None:
                    self.global_status_text = snapshot.reason or snapshot.status
                    if snapshot.speed_kmh is not None:
                        self.last_speed_kmh = snapshot.speed_kmh
                        self.last_speed_until = track.timestamp + float(self.config["measurement"]["done_display_seconds"])
                    if snapshot.capture_pending:
                        best = self.frame_buffer.best_frame(track.bbox_xyxy)
                        if best is not None:
                            ts_text = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            save_capture_set(
                                output_dir=BASE_DIR / self.config["capture"]["save_dir"],
                                track_id=track.track_id,
                                timestamp=ts_text,
                                frame=best.frame,
                                bbox_xyxy=track.bbox_xyxy,
                                plate_crop_ratio=float(self.config["capture"]["plate_crop_ratio"]),
                            )

            self.speed_estimator.cleanup(active_ids, timestamp)
            stale_crossing_ids = [track_id for track_id in self.crossing_detectors if track_id not in active_ids]
            for track_id in stale_crossing_ids:
                del self.crossing_detectors[track_id]

            if self.last_speed_kmh is not None and timestamp > self.last_speed_until:
                self.last_speed_kmh = None

            if self.last_timestamp is not None:
                delta = timestamp - self.last_timestamp
                if delta > 0:
                    current_fps = 1.0 / delta
                    self.smoothed_fps = current_fps if self.smoothed_fps == 0.0 else ((self.smoothed_fps * 0.8) + (current_fps * 0.2))
            self.last_timestamp = timestamp

            draw_hud(
                annotated,
                active_tracks=len(active_ids),
                fps=self.smoothed_fps,
                status_text=self.global_status_text,
                last_speed_kmh=self.last_speed_kmh,
            )

            backend_name = self.detector.primary_backend.name
            if hasattr(self.detector.primary_backend, "loaded") and not getattr(self.detector.primary_backend, "loaded"):
                if self.detector.fallback_backend is not None:
                    backend_name = f"{self.detector.primary_backend.name}->{self.detector.fallback_backend.name}"

            with self.web_state.lock:
                self.web_state.latest_jpeg = encode_frame(annotated, int(self.config["performance"]["jpeg_quality"]))
                self.web_state.latest_status = {
                    "profile": active_profile_name,
                    "fps": round(self.smoothed_fps, 1),
                    "active_tracks": len(active_ids),
                    "status_text": self.global_status_text,
                    "last_speed_kmh": self.last_speed_kmh,
                    "last_speed_text": f"{self.last_speed_kmh:.1f} km/h" if self.last_speed_kmh is not None else "--",
                    "backend_name": backend_name,
                    "control_mode": self.web_state.control_mode,
                }

            if self.config["app"]["show_window"]:
                cv2.imshow(self.config["app"]["name"], annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()


def build_logger(config: dict) -> logging.Logger:
    """Build app logger."""
    logger = logging.getLogger("speedcam_v2")
    if logger.handlers:
        return logger
    level = getattr(logging, str(config["app"]["log_level"]).upper(), logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def main(
    config_path: str | Path | None = None,
    no_web: bool = False,
    max_runtime_seconds: float | None = None,
) -> None:
    """Run the SpeedCam V2 app."""
    config = load_config(config_path or DEFAULT_CONFIG_PATH)
    logger = build_logger(config)

    frame_queue: "queue.Queue[TimedFrame]" = queue.Queue(maxsize=int(config["performance"]["queue_size"]))
    web_state = WebState()
    camera = CameraSource(config, frame_queue, logger)
    processor = ProcessingEngine(config, frame_queue, web_state, logger, max_runtime_seconds=max_runtime_seconds)
    app = create_app(config, web_state)

    capture_dir = BASE_DIR / config["capture"]["save_dir"]
    capture_dir.mkdir(parents=True, exist_ok=True)

    camera.start()
    processor.start()

    try:
        if no_web:
            while processor.is_alive() and not processor.stop_event.is_set():
                time.sleep(0.2)
        else:
            app.run(
                host=config["app"]["web_host"],
                port=int(config["app"]["web_port"]),
                debug=False,
                threaded=True,
                use_reloader=False,
            )
    finally:
        processor.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--max-runtime", type=float, default=None)
    args = parser.parse_args()
    main(
        config_path=args.config,
        no_web=args.no_web,
        max_runtime_seconds=args.max_runtime,
    )
