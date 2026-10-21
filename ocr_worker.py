#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from plate_reader import read_plate
from violation_logger import log_violation

BASE_DIR = Path(os.environ.get("RADAR_BASE_DIR", "/home/radarpi/radar"))
VIOLATIONS_DIR = BASE_DIR / "violations"
OCR_LOG_PATH = VIOLATIONS_DIR / "ocr_system.log"
QUEUE_MAXSIZE = 32
_SENTINEL = object()


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


@dataclass
class OCRJob:
    image_path: str
    speed_kmh: float
    timestamp: str


class OCRWorker:
    """OCR islerini arka planda queue ile isleyen worker."""

    def __init__(self, maxsize: int = QUEUE_MAXSIZE) -> None:
        self.queue: queue.Queue[object] = queue.Queue(maxsize=maxsize)
        self.thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._running and self.thread is not None and self.thread.is_alive():
                return
            self._running = True
            self.thread = threading.Thread(target=self._run, name="ocr-worker", daemon=True)
            self.thread.start()
            LOGGER.info("OCR worker baslatildi.")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            try:
                self.queue.put_nowait(_SENTINEL)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except Exception:
                    pass
                try:
                    self.queue.put_nowait(_SENTINEL)
                except Exception:
                    pass
            thread = self.thread
        if thread is not None:
            thread.join(timeout=2.0)
        LOGGER.info("OCR worker durduruldu.")

    def add_job(self, image_path: str, speed_kmh: float, timestamp: str) -> bool:
        """Ana thread'i bekletmeden OCR isini kuyruga atar."""
        try:
            self.queue.put_nowait(OCRJob(image_path=image_path, speed_kmh=float(speed_kmh), timestamp=timestamp))
            return True
        except queue.Full:
            LOGGER.warning("OCR kuyrugu dolu, is dusuruldu: %s", image_path)
            return False
        except Exception as exc:
            LOGGER.exception("OCR isi kuyruga atilamadi (%s): %s", image_path, exc)
            return False

    def _run(self) -> None:
        while True:
            try:
                item = self.queue.get()
            except Exception as exc:
                LOGGER.exception("OCR kuyruk alma hatasi: %s", exc)
                continue

            if item is _SENTINEL:
                self.queue.task_done()
                break

            assert isinstance(item, OCRJob)
            try:
                result = read_plate(item.image_path)
                plate = result.get("plate")
                confidence = result.get("confidence")
                if plate:
                    status = "OK"
                else:
                    plate = "OKUNAMADI"
                    confidence = None
                    status = "OKUNAMADI"
                if confidence is not None:
                    LOGGER.info(
                        "OCR sonucu: plate=%s final_conf=%.2f",
                        plate,
                        float(confidence),
                    )
                log_violation(
                    timestamp=item.timestamp,
                    speed_kmh=item.speed_kmh,
                    plate=plate,
                    image_path=item.image_path,
                    ocr_confidence=confidence,
                    status=status,
                )
            except Exception as exc:
                LOGGER.exception("OCR isleme hatasi (%s): %s", item.image_path, exc)
                try:
                    log_violation(
                        timestamp=item.timestamp,
                        speed_kmh=item.speed_kmh,
                        plate="OKUNAMADI",
                        image_path=item.image_path,
                        ocr_confidence=None,
                        status="OKUNAMADI",
                    )
                except Exception as nested_exc:
                    LOGGER.exception("OCR fallback log hatasi: %s", nested_exc)
            finally:
                self.queue.task_done()


_WORKER_LOCK = threading.Lock()
_WORKER: Optional[OCRWorker] = None


def _get_worker() -> OCRWorker:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = OCRWorker()
        return _WORKER


def start_ocr_worker() -> None:
    _get_worker().start()


def stop_ocr_worker() -> None:
    global _WORKER
    with _WORKER_LOCK:
        worker = _WORKER
    if worker is not None:
        worker.stop()


def add_ocr_job(image_path: str, speed_kmh: float, timestamp: str) -> None:
    _get_worker().add_job(image_path=image_path, speed_kmh=speed_kmh, timestamp=timestamp)
