#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Asynchronous VISTA/YOLOX vehicle validation for saved violations.

The radar timing loop never waits for this module.  A persistent child process
loads the ONNX model once, validates saved images, narrows the OCR search to the
selected vehicle, and then runs the existing plate reader.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


BASE_DIR = Path(os.environ.get("RADAR_BASE_DIR", "/home/radarpi/radar"))
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "yolox_s.onnx"
AI_LOG_PATH = BASE_DIR / "violations" / "vehicle_validation.log"
AI_METADATA_DIR = BASE_DIR / "violations" / "ai_metadata"
AI_CROP_DIR = BASE_DIR / "violations" / "vehicle_crops"
OCR_REVIEW_DIR = BASE_DIR / "violations" / "ocr_review"
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}
_SENTINEL = "__STOP_VEHICLE_VALIDATOR__"


@dataclass
class VehicleValidationJob:
    image_path: str
    speed_kmh: float
    timestamp: str
    event_id: str = ""


@dataclass
class VehicleDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


def _logger() -> logging.Logger:
    logger = logging.getLogger("vehicle_validation")
    if logger.handlers:
        return logger
    AI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(AI_LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def _iou(a: VehicleDetection, b: VehicleDetection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    w = max(0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = float(w * h)
    union = float((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1)) - intersection
    return intersection / union if union > 0 else 0.0


def _class_aware_nms(detections: list[VehicleDetection], threshold: float = 0.45) -> list[VehicleDetection]:
    accepted: list[VehicleDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(detection.class_id == other.class_id and _iou(detection, other) > threshold for other in accepted):
            continue
        accepted.append(detection)
    return accepted


class YoloXVehicleDetector:
    """OpenCV-DNN implementation of VISTA's raw YOLOX-s contract."""

    def __init__(self, model_path: Path, confidence: float = 0.25, threads: int = 2) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLOX model bulunamadi: {model_path}")
        cv2.setNumThreads(max(1, int(threads)))
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.confidence = float(confidence)

    def detect(self, image: np.ndarray) -> list[VehicleDetection]:
        height, width = image.shape[:2]
        scale = min(640.0 / width, 640.0 / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[:resized_height, :resized_width] = cv2.resize(image, (resized_width, resized_height))
        # VISTA contract: BGR8 -> RGB float CHW, values remain in 0..255.
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0, size=(640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = np.asarray(self.net.forward(), dtype=np.float32)
        if output.size != 8400 * 85:
            raise RuntimeError(f"Beklenmeyen YOLOX cikisi: {output.shape}; [1,8400,85] gerekli")
        rows = output.reshape(8400, 85)
        candidates: list[VehicleDetection] = []
        row_index = 0
        for stride in (8, 16, 32):
            grid_size = 640 // stride
            for gy in range(grid_size):
                for gx in range(grid_size):
                    values = rows[row_index]
                    row_index += 1
                    class_id = int(np.argmax(values[5:85]))
                    if class_id not in VEHICLE_CLASSES:
                        continue
                    score = float(values[4] * values[5 + class_id])
                    if not np.isfinite(score) or score < self.confidence or score > 1.0:
                        continue
                    if not np.isfinite(values[2]) or not np.isfinite(values[3]) or abs(values[2]) > 20 or abs(values[3]) > 20:
                        continue
                    cx = float((values[0] + gx) * stride / scale)
                    cy = float((values[1] + gy) * stride / scale)
                    box_width = float(np.exp(values[2]) * stride / scale)
                    box_height = float(np.exp(values[3]) * stride / scale)
                    x1 = max(0, min(width, int(round(cx - box_width / 2))))
                    y1 = max(0, min(height, int(round(cy - box_height / 2))))
                    x2 = max(0, min(width, int(round(cx + box_width / 2))))
                    y2 = max(0, min(height, int(round(cy + box_height / 2))))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    candidates.append(VehicleDetection(class_id, VEHICLE_CLASSES[class_id], score, (x1, y1, x2, y2)))
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return _class_aware_nms(candidates[:1000])


def _select_vehicle(detections: list[VehicleDetection], image_shape: tuple[int, ...]) -> Optional[VehicleDetection]:
    """Prefer a confident, large road vehicle near the lower image area."""
    if not detections:
        return None
    image_h, image_w = image_shape[:2]
    image_area = max(1.0, float(image_h * image_w))

    def score(item: VehicleDetection) -> float:
        x1, y1, x2, y2 = item.bbox_xyxy
        area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
        bottom_ratio = y2 / max(1.0, float(image_h))
        return item.confidence * 2.0 + min(area_ratio, 0.6) + bottom_ratio * 0.20

    return max(detections, key=score)


def _write_vehicle_crops(image: np.ndarray, detection: VehicleDetection, stem: str) -> tuple[Path, Path]:
    AI_CROP_DIR.mkdir(parents=True, exist_ok=True)
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = detection.bbox_xyxy
    pad_x = int(round((x2 - x1) * 0.08))
    pad_y = int(round((y2 - y1) * 0.08))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(image_w, x2 + pad_x), min(image_h, y2 + pad_y)
    vehicle = image[y1:y2, x1:x2]
    if vehicle.size == 0:
        raise RuntimeError("Secilen arac kutusu bos")
    lower_start = int(round(vehicle.shape[0] * 0.42))
    lower = vehicle[lower_start:, :]
    vehicle_path = AI_CROP_DIR / f"{stem}_vehicle.jpg"
    lower_path = AI_CROP_DIR / f"{stem}_vehicle_lower.jpg"
    if not cv2.imwrite(str(vehicle_path), vehicle) or not cv2.imwrite(str(lower_path), lower):
        raise RuntimeError("Arac OCR crop dosyalari yazilamadi")
    return lower_path, vehicle_path


def _choose_ocr_result(results: list[dict[str, Any]], single_min_confidence: float = 0.98) -> tuple[Optional[str], Optional[float], str]:
    """Accept consensus, not a single overconfident OCR hallucination."""
    valid = [item for item in results if item.get("plate")]
    if not valid:
        return None, None, "no_candidate"
    counts = Counter(str(item["plate"]) for item in valid)
    plate, votes = counts.most_common(1)[0]
    matching = [item for item in valid if str(item["plate"]) == plate]
    confidences = [float(item["confidence"]) for item in matching if item.get("confidence") is not None]
    confidence = max(confidences) if confidences else None
    is_six_digit_numeric = len(plate) == 6 and plate.isdigit()
    if votes >= 2:
        return plate, confidence, "six_digit_multi_crop_consensus" if is_six_digit_numeric else "multi_crop_consensus"
    # A six-digit numeric candidate is never accepted from one crop alone;
    # road signs, timestamps and speed overlays frequently contain only digits.
    if is_six_digit_numeric:
        return None, None, "six_digit_candidate_requires_consensus"
    if len(valid) == 1 and confidence is not None and confidence >= single_min_confidence:
        return plate, confidence, "single_very_high_confidence"
    return None, None, "conflicting_or_weak_candidates"


def _run_ocr(paths: list[Path], job: VehicleValidationJob, logger: logging.Logger) -> None:
    from plate_reader import read_plate
    from violation_logger import log_violation

    attempts: list[dict[str, Any]] = []
    for candidate_path in paths:
        result = read_plate(str(candidate_path))
        attempts.append({"path": str(candidate_path), "plate": result.get("plate"), "confidence": result.get("confidence")})
    accepted_plate, confidence, decision = _choose_ocr_result(
        attempts,
        single_min_confidence=float(os.environ.get("RADAR_OCR_SINGLE_MIN_CONFIDENCE", "0.98")),
    )
    plate = accepted_plate or "OKUNAMADI"
    status = "OK" if accepted_plate else "OKUNAMADI"
    log_violation(job.timestamp, job.speed_kmh, plate, job.image_path, confidence, status)
    OCR_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review_path = OCR_REVIEW_DIR / f"{Path(job.image_path).stem}.json"
    review_path.write_text(json.dumps({
        "schema": "radarpi.ocr-review.v1",
        "event_id": job.event_id,
        "image_path": job.image_path,
        "decision": decision,
        "accepted_plate": accepted_plate,
        "attempts": attempts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("OCR tamamlandi event=%s plate=%s decision=%s", job.event_id, plate, decision)


def _write_metadata(job: VehicleValidationJob, status: str, detections: list[VehicleDetection], selected: Optional[VehicleDetection], **extra: Any) -> None:
    AI_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "radarpi.vehicle-validation.v1",
        "event_id": job.event_id,
        "timestamp": job.timestamp,
        "image_path": job.image_path,
        "status": status,
        "detections": [asdict(item) for item in detections],
        "selected": None if selected is None else asdict(selected),
    }
    payload.update(extra)
    target = AI_METADATA_DIR / f"{Path(job.image_path).stem}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _status_filename(path: Path, status_suffix: str) -> str:
    """Add one status suffix without duplicating it on retries."""
    suffix = status_suffix.upper()
    if path.stem.upper().endswith(suffix):
        return path.name
    return f"{path.stem}{status_suffix}{path.suffix}"


def _reject_image(job: VehicleValidationJob, logger: logging.Logger) -> tuple[str, Optional[str]]:
    path = Path(job.image_path)
    mode = os.environ.get("RADAR_AI_REJECT_MODE", "move").strip().lower()
    if mode == "delete":
        path.unlink(missing_ok=True)
        logger.info("Arac bulunamadi; fotograf silindi: %s", path)
        return "rejected_deleted", None
    if mode == "keep":
        destination = path.with_name(_status_filename(path, "_ARAC_YOK"))
        if path != destination and path.exists():
            path.rename(destination)
        logger.info("Arac bulunamadi; fotograf ARAC_YOK olarak korundu: %s", destination)
        return "rejected_kept", str(destination)
    rejected_dir = path.parent / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    destination = rejected_dir / _status_filename(path, "_ARAC_YOK")
    if destination.exists():
        destination = rejected_dir / f"{path.stem}_ARAC_YOK_{int(time.time())}{path.suffix}"
    shutil.move(str(path), str(destination))
    logger.info("Arac bulunamadi; fotograf rejected klasorune tasindi: %s", destination)
    return "rejected_moved", str(destination)


def _worker_main(job_queue: Any) -> None:
    logger = _logger()
    try:
        if hasattr(os, "nice"):
            os.nice(int(os.environ.get("RADAR_AI_NICE", "10")))
    except Exception:
        pass
    model_path = Path(os.environ.get("RADAR_AI_MODEL", str(DEFAULT_MODEL_PATH)))
    detector: Optional[YoloXVehicleDetector] = None
    try:
        detector = YoloXVehicleDetector(
            model_path,
            confidence=float(os.environ.get("RADAR_AI_CONFIDENCE", "0.25")),
            threads=int(os.environ.get("RADAR_AI_THREADS", "2")),
        )
        logger.info("VISTA YOLOX modeli yuklendi: %s", model_path)
    except Exception as exc:
        # Fail open: validation must never destroy evidence when AI is unavailable.
        logger.exception("VISTA modeli yuklenemedi; fotograflar korunup mevcut OCR calisacak: %s", exc)

    while True:
        item = job_queue.get()
        if item == _SENTINEL:
            break
        job = VehicleValidationJob(**item)
        path = Path(job.image_path)
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Ihlal fotografi okunamadi: {path}")
            if detector is None:
                _write_metadata(job, "ai_unavailable_fail_open", [], None)
                _run_ocr([path], job, logger)
                continue
            detections = detector.detect(image)
            selected = _select_vehicle(detections, image.shape)
            if selected is None:
                reject_status, rejected_path = _reject_image(job, logger)
                _write_metadata(job, reject_status, detections, None, rejected_path=rejected_path, ocr_skipped=True)
                continue
            lower_path, vehicle_path = _write_vehicle_crops(image, selected, path.stem)
            _write_metadata(
                job,
                "vehicle_confirmed",
                detections,
                selected,
                vehicle_label=f"ARAC_VAR_{selected.class_name.upper()}",
                ocr_skipped=False,
                ocr_candidates=[str(lower_path), str(vehicle_path), str(path)],
            )
            _run_ocr([lower_path, vehicle_path, path], job, logger)
        except Exception as exc:
            logger.exception("Arac dogrulama hatasi event=%s image=%s: %s", job.event_id, path, exc)
            try:
                _write_metadata(job, "validation_error_fail_open", [], None, error=str(exc))
                if path.is_file():
                    _run_ocr([path], job, logger)
            except Exception:
                logger.exception("Fail-open OCR da basarisiz: %s", path)


class VehicleValidationWorker:
    def __init__(self, maxsize: int = 8) -> None:
        self.context = mp.get_context("spawn")
        self.queue = self.context.Queue(maxsize=maxsize)
        self.process: Optional[mp.Process] = None

    def start(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.process = self.context.Process(target=_worker_main, args=(self.queue,), name="vehicle-validator", daemon=True)
        self.process.start()

    def add_job(self, image_path: str, speed_kmh: float, timestamp: str, event_id: str = "") -> bool:
        try:
            self.queue.put_nowait(asdict(VehicleValidationJob(image_path, float(speed_kmh), timestamp, event_id)))
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            self.queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)


_WORKER: Optional[VehicleValidationWorker] = None


def start_vehicle_validation_worker() -> None:
    global _WORKER
    if _WORKER is None:
        _WORKER = VehicleValidationWorker()
    _WORKER.start()


def add_vehicle_validation_job(image_path: str, speed_kmh: float, timestamp: str, event_id: str = "") -> bool:
    if _WORKER is None:
        return False
    return _WORKER.add_job(image_path, speed_kmh, timestamp, event_id)


def stop_vehicle_validation_worker() -> None:
    global _WORKER
    if _WORKER is not None:
        _WORKER.stop()
        _WORKER = None


def _cli() -> int:
    """Pi kurulumunda modeli ve tek fotografi OCR calistirmadan test et."""
    import argparse

    parser = argparse.ArgumentParser(description="RadarPi VISTA arac dogrulama testi")
    parser.add_argument("image", help="Test edilecek fotograf")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLOX-s ONNX model yolu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Fotograf okunamadi: {args.image}")
    started = time.perf_counter()
    detector = YoloXVehicleDetector(Path(args.model), args.confidence, args.threads)
    detections = detector.detect(image)
    selected = _select_vehicle(detections, image.shape)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(json.dumps({
        "status": "vehicle_confirmed" if selected else "no_vehicle",
        "elapsed_ms_including_model_load": round(elapsed_ms, 1),
        "detections": [asdict(item) for item in detections],
        "selected": None if selected is None else asdict(selected),
    }, ensure_ascii=False, indent=2))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(_cli())

