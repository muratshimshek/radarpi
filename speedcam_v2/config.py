"""Configuration loading and persistence helpers for SpeedCam V2."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BASE_DIR / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "name": "SpeedCam V2",
        "active_profile": "SAG",
        "web_host": "0.0.0.0",
        "web_port": 5001,
        "show_window": False,
        "draw_guides": True,
        "show_roi_overlay": True,
        "detection_mode": "DENGELI",
        "performance_mode": "NORMAL",
        "log_level": "INFO",
        "line_move_step_px": 4,
        "distance_adjust_step_m": 0.5,
        "speed_limit_step_kmh": 5.0,
    },
    "camera": {
        "source": 0,
        "width": 640,
        "height": 360,
        "fps": 30,
        "use_picamera2": True,
        "replay_realtime": True,
        "replay_loop": True,
        "replay_speed_multiplier": 1.0,
        "crop": [],
    },
    "model": {
        "backend": "ncnn",
        "path": "models/yolov8n_car_ncnn_model",
        "conf_threshold": 0.45,
        "nms_threshold": 0.45,
        "input_size": 320,
        "classes": [2],
        "opencv_fallback_enabled": True,
    },
    "tracking": {
        "stable_hits": 5,
        "lost_tolerance_seconds": 0.5,
        "max_center_jump_px": 80,
        "max_lateral_jump_px": 42,
        "max_reverse_motion_px": 18,
        "max_scale_ratio": 1.8,
        "min_area_ratio_of_prev": 0.30,
        "smoothing_alpha": 0.35,
        "prediction_damping": 0.92,
    },
    "measurement": {
        "real_distance_meters": 10.0,
        "min_elapsed_seconds": 0.50,
        "max_elapsed_seconds": 8.0,
        "speed_limit_kmh": 50.0,
        "done_display_seconds": 3.0,
    },
    "capture": {
        "buffer_size": 10,
        "save_dir": "captures",
        "plate_crop_ratio": 0.30,
    },
    "profiles": {
        "SOL": {
            "lane_polygon": [[32, 50], [520, 50], [610, 358], [0, 358]],
            "yellow_line": [[72, 118], [520, 92]],
            "red_line": [[110, 244], [575, 196]],
            "direction": "down",
        },
        "SAG": {
            "lane_polygon": [[120, 50], [608, 50], [640, 358], [35, 358]],
            "yellow_line": [[120, 92], [565, 118]],
            "red_line": [[35, 196], [500, 244]],
            "direction": "down",
        },
    },
    "homography": {
        "enabled": False,
        "save_path": "data/homography.json",
        "image_points": [],
        "world_points": [],
    },
    "performance": {
        "queue_size": 2,
        "jpeg_quality": 85,
        "status_history_size": 120,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two nested dictionaries without mutating either input."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, falling back to defaults for missing keys."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        save_config(DEFAULT_CONFIG, config_path)
        config = deepcopy(DEFAULT_CONFIG)
        config["_config_path"] = str(config_path)
        return config
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = deep_merge(DEFAULT_CONFIG, loaded)
    config["_config_path"] = str(config_path)
    return config


def save_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    """Persist YAML config to disk."""
    config_path = Path(path) if path is not None else Path(config.get("_config_path", DEFAULT_CONFIG_PATH))
    ensure_parent(config_path)
    to_save = deepcopy(config)
    to_save.pop("_config_path", None)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_save, handle, sort_keys=False, allow_unicode=False)
    return config_path
