#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

BASE_DIR = Path(os.environ.get("RADAR_BASE_DIR", "/home/radarpi/radar"))
VIOLATIONS_DIR = BASE_DIR / "violations"
CSV_PATH = VIOLATIONS_DIR / "violations.csv"
JSONL_PATH = VIOLATIONS_DIR / "violations.jsonl"
OCR_LOG_PATH = VIOLATIONS_DIR / "ocr_system.log"
CSV_FIELDS = ["timestamp", "speed_kmh", "plate", "image_path", "ocr_confidence", "status"]
WRITE_LOCK = threading.Lock()


def build_logger() -> logging.Logger:
    """OCR modulleri icin ortak rotating logger kurar."""
    logger = logging.getLogger("ocr_system")
    if logger.handlers:
        return logger
    VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        OCR_LOG_PATH,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger


LOGGER = build_logger()


def _ensure_csv_header() -> None:
    """CSV dosyasi yoksa header yazar."""
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        return
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()


def log_violation(
    timestamp: str,
    speed_kmh: float,
    plate: str,
    image_path: str,
    ocr_confidence: Optional[float],
    status: str,
) -> None:
    """OCR sonucunu thread-safe sekilde CSV ve JSONL'e yazar."""
    try:
        VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        with WRITE_LOCK:
            _ensure_csv_header()
            row = {
                "timestamp": timestamp,
                "speed_kmh": f"{float(speed_kmh):.2f}",
                "plate": plate,
                "image_path": image_path,
                "ocr_confidence": "" if ocr_confidence is None else f"{float(ocr_confidence):.2f}",
                "status": "OK" if status == "OK" else "OKUNAMADI",
            }
            with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_fh:
                writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
                writer.writerow(row)

            payload = {
                "timestamp": timestamp,
                "speed_kmh": round(float(speed_kmh), 2),
                "plate": plate,
                "image_path": image_path,
                "ocr_confidence": None if ocr_confidence is None else round(float(ocr_confidence), 2),
                "status": row["status"],
            }
            with JSONL_PATH.open("a", encoding="utf-8") as jsonl_fh:
                jsonl_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        LOGGER.exception("Ihlal log yazim hatasi: %s", exc)
