"""Frame buffering and blur scoring for capture output."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class BufferedFrame:
    """Frame saved in the rolling buffer."""

    frame: np.ndarray
    timestamp: float


class FrameBuffer:
    """Keep the latest frames and choose the sharpest one for saving."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max(1, int(max_size))
        self.frames: deque[BufferedFrame] = deque(maxlen=self.max_size)

    def push(self, frame: np.ndarray, timestamp: float) -> None:
        """Append a frame copy to the buffer."""
        self.frames.append(BufferedFrame(frame=frame.copy(), timestamp=timestamp))

    @staticmethod
    def blur_score(frame: np.ndarray) -> float:
        """Use Laplacian variance as a blur score."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def best_frame(self, crop_box: tuple[int, int, int, int] | None = None) -> Optional[BufferedFrame]:
        """Return the sharpest frame, optionally based on a crop region."""
        best: Optional[BufferedFrame] = None
        best_score = -1.0
        for buffered in self.frames:
            sample = buffered.frame
            if crop_box is not None:
                x1, y1, x2, y2 = crop_box
                x1 = max(0, min(sample.shape[1] - 1, x1))
                x2 = max(x1 + 1, min(sample.shape[1], x2))
                y1 = max(0, min(sample.shape[0] - 1, y1))
                y2 = max(y1 + 1, min(sample.shape[0], y2))
                sample = sample[y1:y2, x1:x2]
                if sample.size == 0:
                    continue
            score = self.blur_score(sample)
            if score > best_score:
                best_score = score
                best = buffered
        return best


def save_capture_set(
    output_dir: Path,
    track_id: int,
    timestamp: str,
    frame: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    plate_crop_ratio: float,
) -> dict[str, Path]:
    """Save full frame, vehicle crop, and plate candidate crop."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    vehicle = frame[y1:y2, x1:x2]
    plate_y1 = int(round(y2 - ((y2 - y1) * plate_crop_ratio)))
    plate = frame[max(y1, plate_y1):y2, x1:x2]

    full_path = output_dir / f"full_track_{track_id}_{timestamp}.jpg"
    vehicle_path = output_dir / f"vehicle_track_{track_id}_{timestamp}.jpg"
    plate_path = output_dir / f"plate_track_{track_id}_{timestamp}.jpg"

    cv2.imwrite(str(full_path), frame)
    if vehicle.size > 0:
        cv2.imwrite(str(vehicle_path), vehicle)
    if plate.size > 0:
        cv2.imwrite(str(plate_path), plate)
    return {"full": full_path, "vehicle": vehicle_path, "plate": plate_path}

