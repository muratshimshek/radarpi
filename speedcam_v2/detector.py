"""Detection backends and detector-level filtering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .line_crossing import point_in_polygon

try:
    import ncnn  # type: ignore
except ImportError:  # pragma: no cover - optional on dev machines
    ncnn = None

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover - optional on dev machines
    YOLO = None


@dataclass
class Detection:
    """Single detector output before tracking."""

    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    timestamp: float
    source_backend: str

    @property
    def width(self) -> int:
        return max(0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        return float(self.width * self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0,
        )

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0,
            float(self.bbox_xyxy[3]),
        )

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


class BaseDetectorBackend:
    """Base detector backend contract."""

    name = "base"

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        raise NotImplementedError


class NullDetectorBackend(BaseDetectorBackend):
    """Calibration-only backend that intentionally returns no detections."""

    name = "none"
    loaded = True

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        """Return no detections in calibration mode."""
        return []


class OpenCvFallbackBackend(BaseDetectorBackend):
    """Simple contour-based detector for debug and pipeline bring-up."""

    name = "opencv_fallback"

    def __init__(self, config: dict, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=48,
            detectShadows=False,
        )

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        """Detect moving blobs as a temporary fallback."""
        fg_mask = self.background_subtractor.apply(frame)
        blurred = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 12 or h < 12:
                continue
            area = float(w * h)
            if area < 150:
                continue
            detections.append(
                Detection(
                    bbox_xyxy=(x, y, x + w, y + h),
                    confidence=min(0.95, 0.35 + min(area / 12000.0, 0.55)),
                    class_id=0,
                    timestamp=timestamp,
                    source_backend=self.name,
                )
            )
        return detections


class NcnnVehicleBackend(BaseDetectorBackend):
    """NCNN backend skeleton for Pi deployment."""

    name = "ncnn"

    def __init__(self, model_path: str | Path, config: dict, logger: logging.Logger) -> None:
        self.model_path = Path(model_path)
        self.config = config
        self.logger = logger
        self.net = None
        self.loaded = False
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load the NCNN model folder."""
        if ncnn is None:
            self.logger.warning("ncnn Python binding bulunamadi; fallback kullanilacak.")
            return
        param_path = self.model_path / "model.ncnn.param"
        bin_path = self.model_path / "model.ncnn.bin"
        if not param_path.exists() or not bin_path.exists():
            self.logger.warning("NCNN model dosyalari eksik: %s", self.model_path)
            return
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = 4
        if net.load_param(str(param_path)) != 0:
            self.logger.warning("NCNN param dosyasi yuklenemedi.")
            return
        if net.load_model(str(bin_path)) != 0:
            self.logger.warning("NCNN bin dosyasi yuklenemedi.")
            return
        self.net = net
        self.loaded = True

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        """Return detections from NCNN model if available.

        The decode logic is intentionally isolated here so it can be
        replaced with the exact model-specific postprocess once the final
        exported model is fixed on the Pi.
        """
        if not self.loaded or self.net is None:
            return []
        raise NotImplementedError(
            "NCNN decode katmani model export ciktisina gore tamamlanacak. "
            "Ilk fazda OpenCV fallback ile pipeline dogrulanir."
        )


class UltralyticsVehicleBackend(BaseDetectorBackend):
    """Official Ultralytics backend for quick bring-up with ready YOLO weights."""

    name = "ultralytics"

    def __init__(self, model_path: str | Path, config: dict, logger: logging.Logger) -> None:
        self.model_path = str(model_path)
        self.config = config
        self.logger = logger
        self.model = None
        self.loaded = False
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load an Ultralytics model."""
        if YOLO is None:
            self.logger.warning("ultralytics paketi bulunamadi; fallback kullanilacak.")
            return
        try:
            self.model = YOLO(self.model_path)
            self.loaded = True
            self.logger.info("Ultralytics model yuklendi: %s", self.model_path)
        except Exception:  # pragma: no cover - runtime protection
            self.logger.exception("Ultralytics model yuklenemedi: %s", self.model_path)
            self.model = None
            self.loaded = False

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        """Run official YOLO inference and return only configured classes."""
        if not self.loaded or self.model is None:
            return []
        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=float(self.config["model"]["conf_threshold"]),
            iou=float(self.config["model"]["nms_threshold"]),
            imgsz=int(self.config["model"]["input_size"]),
            classes=self.config["model"].get("classes"),
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
                detections.append(
                    Detection(
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        timestamp=timestamp,
                        source_backend=self.name,
                    )
                )
        return detections


class VehicleDetector:
    """Detector facade with NCNN primary and OpenCV fallback."""

    def __init__(self, config: dict, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.model_config = config["model"]
        self.tracking_config = config["tracking"]
        self.active_profile_name = config["app"]["active_profile"]
        self.last_accepted_by_backend: dict[str, Detection] = {}

        backend_name = str(self.model_config.get("backend", "ncnn")).lower()
        if backend_name == "none":
            primary = NullDetectorBackend()
        elif backend_name == "ultralytics":
            primary = UltralyticsVehicleBackend(self.model_config["path"], config, logger)
        else:
            primary = NcnnVehicleBackend(self.model_config["path"], config, logger)
        self.primary_backend: BaseDetectorBackend = primary
        self.fallback_backend: Optional[BaseDetectorBackend] = None
        if self.model_config.get("opencv_fallback_enabled", False):
            self.fallback_backend = OpenCvFallbackBackend(config, logger)

    def set_active_profile(self, profile_name: str) -> None:
        """Update active profile used for filtering."""
        self.active_profile_name = profile_name

    def _passes_sanity(self, detection: Detection, frame_shape: tuple[int, int, int], lane_polygon: list[list[float]]) -> bool:
        """Apply detector-level filtering rules."""
        frame_h, frame_w = frame_shape[:2]
        frame_area = float(frame_h * frame_w)
        area_ratio = detection.area / max(frame_area, 1.0)
        if area_ratio < 0.0008 or area_ratio > 0.65:
            return False
        if not (0.3 <= detection.aspect_ratio <= 4.0):
            return False
        if not point_in_polygon(detection.bottom_center, lane_polygon):
            return False

        previous = self.last_accepted_by_backend.get(detection.source_backend)
        if previous is not None:
            cx0, cy0 = previous.center
            cx1, cy1 = detection.center
            jump = float(((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5)
            if jump > self.tracking_config["max_center_jump_px"]:
                return False
            if previous.area > 0.0 and detection.area < (previous.area * self.tracking_config["min_area_ratio_of_prev"]):
                return False
        return True

    def detect(self, frame: np.ndarray, timestamp: float, profile: dict) -> list[Detection]:
        """Run primary backend and fallback if needed."""
        detections: list[Detection] = []
        errors: list[Exception] = []
        self.last_accepted_by_backend.clear()
        try:
            detections = self.primary_backend.detect(frame, timestamp)
        except NotImplementedError as exc:
            errors.append(exc)
        except Exception as exc:  # pragma: no cover - runtime protection
            errors.append(exc)
            self.logger.exception("Primary detector backend failed.")

        if not detections and self.fallback_backend is not None:
            detections = self.fallback_backend.detect(frame, timestamp)
            if errors:
                self.logger.info("Fallback detector aktif: %s", self.fallback_backend.name)

        filtered: list[Detection] = []
        for detection in detections:
            if detection.confidence < self.model_config["conf_threshold"]:
                continue
            if self._passes_sanity(detection, frame.shape, profile["lane_polygon"]):
                filtered.append(detection)
                self.last_accepted_by_backend[detection.source_backend] = detection
        return filtered
