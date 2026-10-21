"""Homography calibration and projection utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class HomographyManager:
    """Manage image-to-world homography."""

    enabled: bool
    save_path: Path
    image_points: list[list[float]]
    world_points: list[list[float]]

    def __post_init__(self) -> None:
        self.matrix: np.ndarray | None = None
        if self.enabled:
            self.load_or_compute()

    def load_or_compute(self) -> None:
        """Load saved matrix or compute from configured points."""
        if self.save_path.exists():
            with self.save_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.image_points = payload.get("image_points", self.image_points)
            self.world_points = payload.get("world_points", self.world_points)
            matrix = payload.get("matrix")
            if matrix:
                self.matrix = np.array(matrix, dtype=np.float32)
                return
        self.compute()

    def compute(self) -> None:
        """Compute matrix from current image/world points."""
        if len(self.image_points) != 4 or len(self.world_points) != 4:
            self.matrix = None
            return
        src = np.array(self.image_points, dtype=np.float32)
        dst = np.array(self.world_points, dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(src, dst)
        self.save()

    def save(self) -> None:
        """Persist homography configuration and matrix."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "image_points": self.image_points,
            "world_points": self.world_points,
            "matrix": self.matrix.tolist() if self.matrix is not None else None,
        }
        with self.save_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def image_to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        """Project image point to world point."""
        if self.matrix is None:
            return point
        src = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.matrix)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def distance_meters(self, point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        """Compute Euclidean distance in world plane if matrix exists."""
        if self.matrix is None:
            return 0.0
        world_a = self.image_to_world(point_a)
        world_b = self.image_to_world(point_b)
        dx = world_a[0] - world_b[0]
        dy = world_a[1] - world_b[1]
        return float((dx * dx + dy * dy) ** 0.5)

