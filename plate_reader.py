#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract

BASE_DIR = Path(os.environ.get("RADAR_BASE_DIR", "/home/radarpi/radar"))
OCR_DIR = BASE_DIR / "violations"
OCR_LOG_PATH = OCR_DIR / "ocr_system.log"
TESSERACT_CONFIGS = [
    "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVYZ0123456789",
    "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVYZ0123456789",
    "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVYZ0123456789",
    "--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVYZ0123456789",
]
DIGIT_TESSERACT_CONFIGS = [
    "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
    "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
    "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
]
PLATE_REGEX = re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])([A-Z]{1,3})([0-9]{2,4})$")
SIX_DIGIT_PLATE_REGEX = re.compile(r"^[0-9]{6}$")
MIN_PLATE_AREA = 320
MAX_CANDIDATES = 12
MIN_PLATE_WIDTH_RATIO = 0.035
MAX_PLATE_WIDTH_RATIO = 0.45
MIN_PLATE_HEIGHT_RATIO = 0.018
MAX_PLATE_HEIGHT_RATIO = 0.20
DIRECT_PLATE_MAX_CANDIDATES = 10
OCR_TARGET_MIN_WIDTH = 220
OCR_TARGET_MIN_HEIGHT = 70
PROD_MAX_VARIANTS_PER_ROI = 3
DEBUG_MAX_VARIANTS_PER_ROI = 8
PROD_MAX_DIGIT_VARIANTS = 2
DEBUG_MAX_DIGIT_VARIANTS = 5
EARLY_ACCEPT_CONFIDENCE = 0.78
EARLY_ACCEPT_SCORE = 3.10
STAGE_LIMIT_PANEL = 4
STAGE_LIMIT_DIRECT = 3
STAGE_LIMIT_GENERIC = 2
PROD_MAX_TOTAL_CANDIDATES = 12
MIN_PANEL_QUALITY_FOR_OCR = 0.22
SOFT_ACCEPT_PANEL_QUALITY = 0.58
SOFT_ACCEPT_NORMALIZED_SCORE = 1.45
PROD_MIN_BLUR_SCORE = 22.0
PROD_MIN_CONTRAST_SCORE = 12.0
PROD_MIN_ASPECT_RATIO = 2.2
PROD_MAX_ASPECT_RATIO = 6.8
PROD_MIN_AREA_RATIO = 0.0022
PROD_MAX_AREA_RATIO = 0.1100
STRONG_TAIL_REFINEMENT_PANEL_QUALITY = 0.28
STRONG_TAIL_REFINEMENT_SCORE = 1.10
DEFAULT_OCR_BACKEND = "rapidocr"
DEFAULT_OCR_FALLBACK = "tesseract"
PADDLE_LITE_MAX_CANDIDATES = 8
PADDLE_LITE_MIN_ACCEPT_CONFIDENCE = 0.45
PADDLE_LITE_CMD_ENV = "RADARPI_PADDLE_LITE_CMD"
PADDLE_LITE_DEFAULT_PATHS = [
    "/home/radarpi/PaddleOCR-Lite-License/PaddleOCR-Lite-License",
    "/home/radarpi/PaddleOCR-Lite-License/build/PaddleOCR-Lite-License",
]
RAPIDOCR_MAX_CANDIDATES = 4
RAPIDOCR_MIN_ACCEPT_CONFIDENCE = 0.42


def build_logger() -> logging.Logger:
    """OCR modulleri icin ortak rotating logger kurar."""
    logger = logging.getLogger("ocr_system")
    if logger.handlers:
        return logger
    OCR_DIR.mkdir(parents=True, exist_ok=True)
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
_RAPIDOCR_INSTANCE: Any = None
_RAPIDOCR_IMPORT_FAILED = False
_PADDLE_OCR_INSTANCE: Any = None
_PADDLE_OCR_IMPORT_FAILED = False


def _cleanup_text(text: str) -> str:
    """OCR metnini alfanumerik buyuk harf formuna getirir."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "")
    return cleaned.upper()


def _apply_digit_corrections(text: str) -> str:
    """Rakam bolumleri icin harf-karakter karisimlarini duzeltir."""
    table = str.maketrans({
        "O": "0",
        "I": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
    })
    return text.translate(table)


def _apply_letter_corrections(text: str) -> str:
    """Harf bolumu icin rakam-karakter karisimlarini duzeltir."""
    table = str.maketrans({
        "0": "O",
        "1": "I",
        "5": "S",
        "8": "B",
        "2": "Z",
    })
    return text.translate(table)


def _normalize_plate_candidate(text: str) -> Optional[str]:
    """Metni olasi TR plaka formatina gore normalize eder."""
    cleaned = _cleanup_text(text)
    if len(cleaned) < 5:
        return None

    if SIX_DIGIT_PLATE_REGEX.fullmatch(cleaned):
        return cleaned

    candidates_to_try: list[str] = [cleaned]
    if len(cleaned) > 9:
        seen: set[str] = set()
        sliced: list[str] = []
        for length in (9, 8, 7, 6, 5):
            if len(cleaned) < length:
                continue
            for start in range(0, len(cleaned) - length + 1):
                piece = cleaned[start : start + length]
                if piece not in seen:
                    seen.add(piece)
                    sliced.append(piece)
        candidates_to_try = sliced

    for candidate_text in candidates_to_try:
        for letter_len in (3, 2, 1):
            for tail_len in (4, 3, 2):
                if 2 + letter_len + tail_len != len(candidate_text):
                    continue
                city = _apply_digit_corrections(candidate_text[:2])
                letters = _apply_letter_corrections(candidate_text[2 : 2 + letter_len])
                tail = _apply_digit_corrections(candidate_text[2 + letter_len :])
                candidate = f"{city}{letters}{tail}"
                if PLATE_REGEX.match(candidate):
                    return candidate
    return None


def _valid_city_code(code: str) -> bool:
    """TR il kodu araliginda iki haneli deger kontrolu yapar."""
    return bool(re.match(r"^(0[1-9]|[1-7][0-9]|8[01])$", code))


def _extract_partial_plate_components(text: str) -> list[tuple[str, str, str]]:
    """Ham OCR metninden sehir/harf/tail parcaciklari cikarir."""
    cleaned = _cleanup_text(text)
    if len(cleaned) < 5:
        return []

    results: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for start in range(0, len(cleaned)):
        for letter_len in (3, 2, 1):
            for tail_len in (4, 3, 2):
                total_len = 2 + letter_len + tail_len
                end = start + total_len
                if end > len(cleaned):
                    continue
                chunk = cleaned[start:end]
                city = _apply_digit_corrections(chunk[:2])
                letters = _apply_letter_corrections(chunk[2 : 2 + letter_len])
                tail = _apply_digit_corrections(chunk[2 + letter_len :])
                if not _valid_city_code(city):
                    continue
                if not re.match(r"^[A-Z]{1,3}$", letters):
                    continue
                if not re.match(r"^[0-9]{2,4}$", tail):
                    continue
                key = (city, letters, tail)
                if key in seen:
                    continue
                seen.add(key)
                results.append(key)
    return results


def _extract_loose_plate_components(text: str) -> dict[str, set[str]]:
    """Ham OCR icinden sehir, harf ve kuyruk parcaciklarini daha gevsek toplar."""
    cleaned = _cleanup_text(text)
    result: dict[str, set[str]] = {
        "city": set(),
        "letters": set(),
        "tail": set(),
    }
    if len(cleaned) < 3:
        return result

    for start in range(0, max(0, len(cleaned) - 1)):
        city = _apply_digit_corrections(cleaned[start : start + 2])
        if len(city) == 2 and _valid_city_code(city):
            result["city"].add(city)

    for start in range(0, len(cleaned)):
        for letter_len in (3, 2, 1):
            for tail_len in (4, 3, 2):
                end = start + letter_len + tail_len
                if end > len(cleaned):
                    continue
                letters = _apply_letter_corrections(cleaned[start : start + letter_len])
                tail = _apply_digit_corrections(cleaned[start + letter_len : end])
                if re.match(r"^[A-Z]{3}$", letters):
                    result["letters"].add(letters)
                if re.match(r"^[0-9]{3,4}$", tail):
                    result["tail"].add(tail)
    return result


def _plate_shape_bonus(letters: str, tail: str) -> float:
    """Turkiye'de daha yaygin plaka uzunluklarini one cikarir."""
    bonus = 0.0
    if len(letters) == 3:
        bonus += 0.80
    elif len(letters) == 2:
        bonus += 0.20

    if len(tail) == 3:
        bonus += 0.75
    elif len(tail) == 4:
        bonus += 0.25

    if len(letters) == 3 and len(tail) == 3:
        bonus += 0.55
    return bonus


def _split_plate_components(plate: str) -> Optional[tuple[str, str, str]]:
    """Normalize plaka metnini sehir/harf/tail parcalarina ayirir."""
    match = PLATE_REGEX.match(plate or "")
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _refine_plate_from_raw_evidence(
    plate: str,
    raw_records: list[tuple[str, float]],
) -> str:
    """Secilen plakayi, ayni harf grubunu tasiyan ham OCR kanitlariyla iyilestirir."""
    parts = _split_plate_components(plate)
    if parts is None:
        return plate

    city, letters, tail = parts
    if len(letters) != 3:
        return plate

    current_score = 0.0
    tail_scores: dict[str, float] = defaultdict(float)

    for cleaned_text, weight in raw_records:
        if not cleaned_text or letters not in cleaned_text:
            continue
        idx = cleaned_text.find(letters)
        suffix = cleaned_text[idx + len(letters) :]
        prefix = cleaned_text[:idx]

        for match in re.finditer(r"[0-9OSIBZ]{3,4}", suffix):
            candidate_tail = _apply_digit_corrections(match.group(0))
            if re.match(r"^[0-9]{3,4}$", candidate_tail):
                bonus = weight
                if candidate_tail == tail:
                    current_score += bonus
                tail_scores[candidate_tail] += bonus

        for match in re.finditer(r"[0-9OSIBZ]{2}$", prefix[-2:]):
            candidate_city = _apply_digit_corrections(match.group(0))
            if candidate_city == city:
                current_score += weight * 0.35

    if not tail_scores:
        return plate

    best_tail, best_tail_score = max(tail_scores.items(), key=lambda item: item[1])
    if best_tail != tail and best_tail_score >= (current_score + 0.25):
        return f"{city}{letters}{best_tail}"

    sibling_candidates = [
        (candidate_tail, tail_score)
        for candidate_tail, tail_score in tail_scores.items()
        if candidate_tail != tail and len(candidate_tail) == len(tail) and candidate_tail[:2] == tail[:2]
    ]
    if sibling_candidates:
        sibling_candidates.sort(key=lambda item: item[1], reverse=True)
        sibling_tail, sibling_score = sibling_candidates[0]
        if sibling_score >= (current_score * 0.72):
            return f"{city}{letters}{sibling_tail}"
    return plate


def _preprocess_digit_variants(image: np.ndarray) -> list[np.ndarray]:
    """Plakanin sayi blogu icin sade ama agresif OCR varyantlari uretir."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharp = cv2.addWeighted(enhanced, 1.8, cv2.GaussianBlur(enhanced, (0, 0), 1.0), -0.8, 0)
    otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = cv2.bitwise_not(otsu)
    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    return [enhanced, sharp, otsu, inv, adaptive]


def _extract_digit_text_candidates_with_configs(
    image: np.ndarray,
    configs: list[str],
    call_counter: Optional[dict[str, int]] = None,
) -> list[tuple[str, Optional[float], str]]:
    """Digits-only OCR sonuclari uretir."""
    candidates: list[tuple[str, Optional[float], str]] = []
    for config in configs:
        try:
            if call_counter is not None:
                call_counter["count"] = call_counter.get("count", 0) + 1
            data = pytesseract.image_to_data(
                image,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            texts: list[str] = []
            confidences: list[float] = []
            for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
                text = re.sub(r"[^0-9]", "", str(raw_text or ""))
                if not text:
                    continue
                try:
                    conf_value = float(raw_conf)
                except Exception:
                    conf_value = -1.0
                if conf_value >= 0:
                    confidences.append(conf_value)
                texts.append(text)
            combined = "".join(texts)
            confidence = None
            if confidences:
                confidence = max(0.0, min(1.0, float(sum(confidences) / len(confidences)) / 100.0))
            if combined:
                candidates.append((combined, confidence, config))
        except Exception:
            pass

        try:
            if call_counter is not None:
                call_counter["count"] = call_counter.get("count", 0) + 1
            raw = pytesseract.image_to_string(image, config=config)
            cleaned = re.sub(r"[^0-9]", "", raw or "")
            if cleaned:
                candidates.append((cleaned, None, config + " string"))
        except Exception:
            pass
    return candidates


def _extract_digit_text_candidates(image: np.ndarray) -> list[tuple[str, Optional[float], str]]:
    """Geriye donuk uyumluluk icin varsayilan digit OCR wrapper'i."""
    return _extract_digit_text_candidates_with_configs(image, list(DIGIT_TESSERACT_CONFIGS))


def _select_tesseract_configs(debug_enabled: bool) -> list[str]:
    """Uretimde daha hafif, debug'da daha genis OCR config seti sec."""
    if debug_enabled:
        return list(TESSERACT_CONFIGS)
    return [
        TESSERACT_CONFIGS[0],  # psm 7
        TESSERACT_CONFIGS[2],  # psm 6
        TESSERACT_CONFIGS[3],  # psm 13
    ]


def _select_digit_tesseract_configs(debug_enabled: bool) -> list[str]:
    """Sayi blogu OCR icin uretim/debug config secimi."""
    if debug_enabled:
        return list(DIGIT_TESSERACT_CONFIGS)
    return [
        DIGIT_TESSERACT_CONFIGS[0],  # psm 7
        DIGIT_TESSERACT_CONFIGS[2],  # psm 13
    ]


def _select_digit_variant_subset(variants: list[np.ndarray], debug_enabled: bool) -> list[np.ndarray]:
    """Sayi blogu OCR varyantlarini debug/uretim icin sinirla."""
    if debug_enabled:
        return variants[:DEBUG_MAX_DIGIT_VARIANTS]
    preferred_indices = [1, 2]
    selected: list[np.ndarray] = []
    for idx in preferred_indices:
        if idx < len(variants):
            selected.append(variants[idx])
        if len(selected) >= PROD_MAX_DIGIT_VARIANTS:
            break
    return selected if selected else variants[:PROD_MAX_DIGIT_VARIANTS]


def _digit_mask(image: np.ndarray) -> np.ndarray:
    """Rakam ROI'sinden koyu rakam maskesi uretir."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if len(np.unique(gray)) <= 2:
        # Beyaz zemin / siyah rakam bekleniyor.
        mask = (gray < 128).astype("uint8") * 255
    else:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        white_ratio = float(np.count_nonzero(thresh)) / float(max(1, thresh.size))
        mask = cv2.bitwise_not(thresh) if white_ratio > 0.50 else thresh
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    return mask


def _classify_last_digit_shape(image: np.ndarray) -> Optional[str]:
    """Son rakamin sekline bakarak 2/7 ayrimi yapmaya calisir."""
    mask = _digit_mask(image)
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[tuple[int, int, int, int, int]] = []
    h, w = mask.shape[:2]
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < max(20, (h * w) // 100):
            continue
        if ch < max(8, h // 3):
            continue
        components.append((x, y, cw, ch, area))

    if not components:
        return None

    components.sort(key=lambda item: item[0])
    x, y, cw, ch, _area = components[-1]
    digit = mask[y : y + ch, x : x + cw]
    if digit.size == 0 or cw < 6 or ch < 12:
        return None

    norm = cv2.resize(digit, (48, 72), interpolation=cv2.INTER_NEAREST)
    occ = norm > 0
    top = occ[:18, :]
    mid = occ[18:54, :]
    bottom = occ[54:, :]
    left_bottom = bottom[:, :24]
    right_bottom = bottom[:, 24:]
    right_col = occ[:, 32:]

    top_ratio = float(np.count_nonzero(top)) / float(max(1, top.size))
    mid_ratio = float(np.count_nonzero(mid)) / float(max(1, mid.size))
    bottom_ratio = float(np.count_nonzero(bottom)) / float(max(1, bottom.size))
    left_bottom_ratio = float(np.count_nonzero(left_bottom)) / float(max(1, left_bottom.size))
    right_bottom_ratio = float(np.count_nonzero(right_bottom)) / float(max(1, right_bottom.size))
    right_ratio = float(np.count_nonzero(right_col)) / float(max(1, right_col.size))

    score_2 = (bottom_ratio * 2.6) + (left_bottom_ratio * 1.7) + (mid_ratio * 0.8)
    score_7 = (top_ratio * 2.0) + (right_ratio * 1.3) - (left_bottom_ratio * 1.2) - (bottom_ratio * 0.7)

    if score_2 >= (score_7 + 0.18):
        return "2"
    if score_7 >= (score_2 + 0.18):
        return "7"
    return None


def _refine_tail_with_digit_crop(
    plate: str,
    plate_roi: Optional[np.ndarray],
    debug_base: Optional[Path],
    call_counter: Optional[dict[str, int]] = None,
) -> str:
    """Plakanin sag sayi blogunu ayri OCR ile okuyup son haneyi duzeltmeye calisir."""
    parts = _split_plate_components(plate)
    if parts is None or plate_roi is None or plate_roi.size == 0:
        return plate

    city, letters, tail = parts
    if len(letters) != 3 or len(tail) != 3:
        return plate

    h, w = plate_roi.shape[:2]
    if h < 12 or w < 36:
        return plate

    x0 = int(round(w * 0.58))
    x1 = int(round(w * 0.98))
    y0 = int(round(h * 0.08))
    y1 = int(round(h * 0.92))
    digit_roi = plate_roi[y0:y1, x0:x1]
    if digit_roi.size == 0:
        return plate

    debug_enabled = debug_base is not None
    variants = _select_digit_variant_subset(_preprocess_digit_variants(digit_roi), debug_enabled=debug_enabled)
    tail_scores: dict[str, float] = defaultdict(float)
    tail_hits: dict[str, int] = defaultdict(int)
    last_digit_votes: dict[str, float] = defaultdict(float)

    for idx, variant in enumerate(variants, start=1):
        if debug_base is not None:
            _save_debug_image(debug_base / f"tail_variant_{idx:02d}.png", variant)
        shape_digit = _classify_last_digit_shape(variant)
        if shape_digit is not None:
            last_digit_votes[shape_digit] += 1.0
        for raw_digits, confidence, _source in _extract_digit_text_candidates_with_configs(
            variant,
            _select_digit_tesseract_configs(debug_enabled=debug_enabled),
            call_counter=call_counter,
        ):
            for match in re.finditer(r"[0-9]{3}", raw_digits):
                candidate_tail = match.group(0)
                score = 1.0 + (confidence or 0.0)
                if candidate_tail == tail:
                    score += 0.15
                tail_scores[candidate_tail] += score
                tail_hits[candidate_tail] += 1

    if not tail_scores:
        return plate

    best_tail, best_score = max(
        tail_scores.items(),
        key=lambda item: (tail_hits.get(item[0], 0), item[1]),
    )
    current_score = tail_scores.get(tail, 0.0)
    best_hits = tail_hits.get(best_tail, 0)
    current_hits = tail_hits.get(tail, 0)

    if best_tail != tail and best_hits >= 2 and best_hits > current_hits:
        return f"{city}{letters}{best_tail}"

    if best_tail != tail and best_score >= (current_score + 0.20):
        return f"{city}{letters}{best_tail}"

    # 507/502 benzeri ayrimlarda ayni ilk iki haneyi tasiyan alternatif kuyrugu tercih et.
    sibling_candidates = [
        (candidate_tail, tail_hits.get(candidate_tail, 0), score)
        for candidate_tail, score in tail_scores.items()
        if candidate_tail != tail and candidate_tail[:2] == tail[:2]
    ]
    if sibling_candidates:
        sibling_candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
        sibling_tail, sibling_hits, sibling_score = sibling_candidates[0]
        if sibling_hits >= current_hits and sibling_score >= (current_score * 0.70):
            return f"{city}{letters}{sibling_tail}"

    if tail.endswith("7") and last_digit_votes.get("2", 0.0) >= (last_digit_votes.get("7", 0.0) + 1.0):
        return f"{city}{letters}{tail[:2]}2"

    if best_tail != tail and best_score >= (current_score + 0.45):
        return f"{city}{letters}{best_tail}"
    return plate


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """OCR oncesi ROI kontrastini ve keskinligini artirir."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    thresh = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    upscaled = cv2.resize(thresh, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return upscaled


def _preprocess_variants(image: np.ndarray) -> list[np.ndarray]:
    """Ayni ROI icin birden fazla OCR denemesi uretir."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    upscale = max(
        1.0,
        OCR_TARGET_MIN_WIDTH / float(max(1, w)),
        OCR_TARGET_MIN_HEIGHT / float(max(1, h)),
    )
    gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 7, 35, 35)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.7, blurred, -0.7, 0)
    denoised_sharp = cv2.addWeighted(denoised, 1.6, cv2.GaussianBlur(denoised, (0, 0), 1.2), -0.6, 0)
    natural_gray = cv2.resize(enhanced, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    upscaled_gray = cv2.resize(sharpened, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    denoised_otsu = cv2.threshold(denoised_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inverted_otsu = cv2.bitwise_not(otsu)
    morph = cv2.morphologyEx(
        otsu,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    blackhat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
    )
    blackhat_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    variants = [
        natural_gray,
        upscaled_gray,
        cv2.resize(adaptive, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        cv2.resize(otsu, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        cv2.resize(denoised_otsu, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        cv2.resize(inverted_otsu, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        cv2.resize(morph, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
        cv2.resize(blackhat_norm, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC),
    ]
    return variants


def _select_variant_subset(variants: list[np.ndarray], debug_enabled: bool) -> list[np.ndarray]:
    """Uretimde daha az ama daha yararli preprocess varyanti sec."""
    if debug_enabled:
        return variants[:DEBUG_MAX_VARIANTS_PER_ROI]
    preferred_indices = [0, 3, 4]
    selected: list[np.ndarray] = []
    for idx in preferred_indices:
        if idx < len(variants):
            selected.append(variants[idx])
        if len(selected) >= PROD_MAX_VARIANTS_PER_ROI:
            break
    return selected if selected else variants[:PROD_MAX_VARIANTS_PER_ROI]


def _largest_component_ratio(mask: np.ndarray) -> float:
    """Maskede en buyuk bagli bileşenin alana oranini verir."""
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    except Exception:
        return 0.0
    if num_labels <= 1:
        return 0.0
    largest = 0
    for idx in range(1, num_labels):
        largest = max(largest, int(stats[idx, cv2.CC_STAT_AREA]))
    total = int(mask.shape[0] * mask.shape[1])
    return 0.0 if total <= 0 else float(largest) / float(total)


def _plate_panel_quality(roi: np.ndarray) -> float:
    """ROI'nin beyaz plaka paneline benzerligini 0..1 arasi puanlar."""
    if roi is None or roi.size == 0:
        return 0.0
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    h, w = gray.shape[:2]
    if h < 8 or w < 16:
        return 0.0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = cv2.bitwise_not(otsu)

    white_ratio = float(np.count_nonzero(otsu)) / float(max(1, w * h))
    dark_ratio = float(np.count_nonzero(inv)) / float(max(1, w * h))
    largest_white = _largest_component_ratio(otsu)

    col_dark = np.mean(inv > 0, axis=0)
    char_columns = float(np.count_nonzero((col_dark > 0.06) & (col_dark < 0.72))) / float(max(1, w))

    mid_y0 = int(round(h * 0.18))
    mid_y1 = int(round(h * 0.82))
    mid_band = inv[mid_y0:mid_y1, :]
    outer_top = inv[: max(1, int(round(h * 0.14))), :]
    outer_bottom = inv[min(h, int(round(h * 0.86))) :, :]
    mid_dark_ratio = float(np.count_nonzero(mid_band)) / float(max(1, mid_band.size))
    outer_pixels = np.concatenate([outer_top.reshape(-1), outer_bottom.reshape(-1)]) if outer_top.size or outer_bottom.size else np.array([], dtype=np.uint8)
    outer_dark_ratio = 0.0 if outer_pixels.size == 0 else float(np.count_nonzero(outer_pixels)) / float(outer_pixels.size)

    ratio = float(w) / float(max(1, h))
    ratio_pref = 1.0 - min(1.0, abs(ratio - 4.5) / 1.8)
    white_pref = 1.0 - min(1.0, abs(white_ratio - 0.72) / 0.30)
    dark_pref = 1.0 - min(1.0, abs(dark_ratio - 0.22) / 0.18)
    blob_pref = min(1.0, largest_white / 0.55)
    char_pref = 1.0 - min(1.0, abs(char_columns - 0.52) / 0.38)
    band_pref = min(1.0, max(0.0, (mid_dark_ratio - outer_dark_ratio) + 0.18) / 0.30)

    return max(
        0.0,
        min(
            1.0,
            (ratio_pref * 0.18)
            + (white_pref * 0.22)
            + (dark_pref * 0.18)
            + (blob_pref * 0.18)
            + (char_pref * 0.16)
            + (band_pref * 0.08),
        ),
    )


def _roi_blur_score(roi: np.ndarray) -> float:
    """ROI netligini hizli bir Laplacian varyansi ile olcer."""
    if roi is None or roi.size == 0:
        return 0.0
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    if gray.shape[0] < 8 or gray.shape[1] < 16:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _roi_contrast_score(roi: np.ndarray) -> float:
    """ROI kontrastini gri ton stddev ile olcer."""
    if roi is None or roi.size == 0:
        return 0.0
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    return float(np.std(gray))


def _production_candidate_metrics(
    roi: np.ndarray,
    roi_used: bool,
    refined_used: bool,
    image_shape: tuple[int, ...],
) -> dict[str, float]:
    """Uretim modunda ROI eleme ve siralama icin hafif metrikler hesaplar."""
    h, w = roi.shape[:2]
    image_h, image_w = image_shape[:2]
    panel_quality = _plate_panel_quality(roi)
    blur_score = _roi_blur_score(roi)
    contrast_score = _roi_contrast_score(roi)
    aspect_ratio = float(w) / float(max(1, h))
    area_ratio = float(w * h) / float(max(1, image_w * image_h))
    support_bonus = 0.14 if roi_used else 0.0
    support_bonus += 0.12 if refined_used else 0.0
    ranking_score = (
        panel_quality * 0.58
        + min(1.0, blur_score / 180.0) * 0.16
        + min(1.0, contrast_score / 42.0) * 0.10
        + (1.0 - min(1.0, abs(aspect_ratio - 4.4) / 2.8)) * 0.12
        + support_bonus
    )
    return {
        "panel_quality": panel_quality,
        "blur_score": blur_score,
        "contrast_score": contrast_score,
        "aspect_ratio": aspect_ratio,
        "area_ratio": area_ratio,
        "ranking_score": ranking_score,
    }


def _passes_production_roi_filter(metrics: dict[str, float]) -> bool:
    """Uretim modunda zayif ROI'leri OCR oncesi eler."""
    if metrics["aspect_ratio"] < PROD_MIN_ASPECT_RATIO or metrics["aspect_ratio"] > PROD_MAX_ASPECT_RATIO:
        return False
    if metrics["area_ratio"] < PROD_MIN_AREA_RATIO or metrics["area_ratio"] > PROD_MAX_AREA_RATIO:
        return False
    if metrics["blur_score"] < PROD_MIN_BLUR_SCORE:
        return False
    if metrics["contrast_score"] < PROD_MIN_CONTRAST_SCORE:
        return False
    if metrics["panel_quality"] < MIN_PANEL_QUALITY_FOR_OCR and metrics["ranking_score"] < 0.42:
        return False
    return True


def _region_score(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> float:
    """Plaka adayina konum + oran + boyut tabanli skor verir."""
    if h <= 0 or w <= 0:
        return -1.0
    ratio = float(w) / float(h)
    area = float(w * h)
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    normalized_center_x = center_x / float(image_w)
    normalized_center_y = center_y / float(image_h)
    width_ratio = w / float(image_w)
    height_ratio = h / float(image_h)
    horizontal_pref = 1.0 - min(1.0, abs(normalized_center_x - 0.58) / 0.42)
    vertical_pref = 1.0 - min(1.0, abs(normalized_center_y - 0.58) / 0.34)
    ratio_pref = 1.0 - min(1.0, abs(ratio - 4.1) / 1.8)
    size_pref = min(1.0, area / 12000.0)
    shape_bonus = 1.0
    if width_ratio < MIN_PLATE_WIDTH_RATIO or width_ratio > MAX_PLATE_WIDTH_RATIO:
        shape_bonus *= 0.45
    if height_ratio < MIN_PLATE_HEIGHT_RATIO or height_ratio > MAX_PLATE_HEIGHT_RATIO:
        shape_bonus *= 0.45
    if normalized_center_y < 0.22:
        shape_bonus *= 0.55
    if normalized_center_x < 0.10 or normalized_center_x > 0.92:
        shape_bonus *= 0.65
    return ((ratio_pref * 3.2) + (horizontal_pref * 2.0) + (vertical_pref * 3.0) + (size_pref * 1.4)) * shape_bonus


def _extract_candidate_regions(image: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Goruntuden birden cok olasi plaka ROI'si cikarir."""
    image_h, image_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    search_regions = [
        (int(image_w * 0.40), int(image_h * 0.18), int(image_w * 0.56), int(image_h * 0.34)),
        (int(image_w * 0.22), int(image_h * 0.24), int(image_w * 0.66), int(image_h * 0.34)),
        (int(image_w * 0.18), int(image_h * 0.48), int(image_w * 0.64), int(image_h * 0.30)),
        (int(image_w * 0.22), int(image_h * 0.42), int(image_w * 0.56), int(image_h * 0.34)),
        (0, int(image_h * 0.35), image_w, int(image_h * 0.50)),
        (0, 0, image_w, image_h),
    ]

    scored_rects: list[tuple[float, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for rx, ry, rw, rh in search_regions:
        if rw <= 0 or rh <= 0:
            continue
        region_gray = gray[ry:ry + rh, rx:rx + rw]
        if region_gray.size == 0:
            continue

        filtered = cv2.bilateralFilter(region_gray, 9, 75, 75)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, rect_kernel)
        grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = np.absolute(grad_x)
        max_grad = float(grad_x.max())
        if max_grad > 0:
            grad_x = (255.0 * grad_x / max_grad).astype("uint8")
        else:
            grad_x = grad_x.astype("uint8")
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5)),
            iterations=1,
        )
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=2)

        edges = cv2.Canny(filtered, 60, 180)
        combined = cv2.bitwise_or(thresh, edges)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            if area < MIN_PLATE_AREA:
                continue
            ratio = float(w) / float(max(h, 1))
            if ratio < 1.6 or ratio > 7.4:
                continue

            gx = rx + x
            gy = ry + y
            pad_x = int(round(w * 0.10))
            pad_y = int(round(h * 0.18))
            x0 = max(0, gx - pad_x)
            y0 = max(0, gy - pad_y)
            x1 = min(image_w, gx + w + pad_x)
            y1 = min(image_h, gy + h + pad_y)
            rect = (x0, y0, x1 - x0, y1 - y0)
            key = (rect[0] // 8, rect[1] // 8, rect[2] // 8, rect[3] // 8)
            if key in seen:
                continue
            seen.add(key)
            score = _region_score(rect[0], rect[1], rect[2], rect[3], image_w, image_h)
            scored_rects.append((score, rect))

    scored_rects.sort(key=lambda item: item[0], reverse=True)
    rois: list[tuple[np.ndarray, bool]] = []
    for _, (x, y, w, h) in scored_rects[:MAX_CANDIDATES]:
        roi = image[y:y + h, x:x + w]
        if roi.size:
            rois.append((roi, True))

    # Son fallback'ler: alt-orta ve tam goruntu.
    fallback_boxes = [
        (int(image_w * 0.40), int(image_h * 0.22), int(image_w * 0.42), int(image_h * 0.18)),
        (int(image_w * 0.26), int(image_h * 0.28), int(image_w * 0.54), int(image_h * 0.20)),
        (int(image_w * 0.28), int(image_h * 0.54), int(image_w * 0.44), int(image_h * 0.18)),
        (int(image_w * 0.04), int(image_h * 0.58), int(image_w * 0.60), int(image_h * 0.26)),
        (int(image_w * 0.08), int(image_h * 0.54), int(image_w * 0.68), int(image_h * 0.30)),
        (int(image_w * 0.22), int(image_h * 0.50), int(image_w * 0.56), int(image_h * 0.28)),
        (0, 0, image_w, image_h),
    ]
    for x, y, w, h in fallback_boxes:
        roi = image[y:y + h, x:x + w]
        if roi.size:
            rois.append((roi, False))
    return rois


def _extract_direct_plate_regions(image: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Tam goruntu icinde beyaz plaka panelini dogrudan arar."""
    image_h, image_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    search_regions = [
        (int(image_w * 0.42), int(image_h * 0.18), int(image_w * 0.52), int(image_h * 0.28)),
        (int(image_w * 0.24), int(image_h * 0.22), int(image_w * 0.62), int(image_h * 0.28)),
        (int(image_w * 0.18), int(image_h * 0.48), int(image_w * 0.64), int(image_h * 0.34)),
        (int(image_w * 0.22), int(image_h * 0.42), int(image_w * 0.56), int(image_h * 0.34)),
        (0, int(image_h * 0.40), image_w, int(image_h * 0.45)),
    ]

    scored_rects: list[tuple[float, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for rx, ry, rw, rh in search_regions:
        region = enhanced[ry:ry + rh, rx:rx + rw]
        if region.size == 0:
            continue

        blur = cv2.GaussianBlur(region, (5, 5), 0)
        bright = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY)[1]
        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        masks = [bright, otsu]

        for mask in masks:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
                iterations=1,
            )
            mask = cv2.dilate(mask, None, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 0 or h <= 0:
                    continue
                area = float(w * h)
                if area < 220:
                    continue
                ratio = float(w) / float(h)
                if ratio < 2.0 or ratio > 7.2:
                    continue

                gx = rx + x
                gy = ry + y
                center_x = (gx + (w / 2.0)) / float(image_w)
                center_y = (gy + (h / 2.0)) / float(image_h)
                width_ratio = w / float(image_w)
                height_ratio = h / float(image_h)
                if width_ratio < 0.035 or width_ratio > 0.62:
                    continue
                if height_ratio < 0.018 or height_ratio > 0.22:
                    continue
                if center_y < 0.16 or center_y > 0.92:
                    continue

                roi_gray = enhanced[gy:gy + h, gx:gx + w]
                if roi_gray.size == 0:
                    continue
                white_ratio = float(np.count_nonzero(mask[y:y + h, x:x + w])) / float(max(1, w * h))
                edge_density = float(
                    np.count_nonzero(cv2.Canny(roi_gray, 60, 180))
                ) / float(max(1, w * h))

                horizontal_pref = 1.0 - min(1.0, abs(center_x - 0.56) / 0.38)
                vertical_pref = 1.0 - min(1.0, abs(center_y - 0.54) / 0.30)
                ratio_pref = 1.0 - min(1.0, abs(ratio - 4.4) / 1.5)
                white_pref = 1.0 - min(1.0, abs(white_ratio - 0.70) / 0.45)
                edge_pref = min(1.0, edge_density / 0.18)
                size_pref = min(1.0, area / 2500.0)
                score = (
                    (horizontal_pref * 2.0)
                    + (vertical_pref * 3.2)
                    + (ratio_pref * 2.8)
                    + (white_pref * 1.8)
                    + (edge_pref * 1.6)
                    + (size_pref * 1.0)
                )

                pad_x = int(round(w * 0.08))
                pad_y = int(round(h * 0.20))
                x0 = max(0, gx - pad_x)
                y0 = max(0, gy - pad_y)
                x1 = min(image_w, gx + w + pad_x)
                y1 = min(image_h, gy + h + pad_y)
                rect = (x0, y0, x1 - x0, y1 - y0)
                key = (rect[0] // 8, rect[1] // 8, rect[2] // 8, rect[3] // 8)
                if key in seen:
                    continue
                seen.add(key)
                scored_rects.append((score, rect))

    scored_rects.sort(key=lambda item: item[0], reverse=True)
    rois: list[tuple[np.ndarray, bool]] = []
    for _, (x, y, w, h) in scored_rects[:DIRECT_PLATE_MAX_CANDIDATES]:
        roi = image[y:y + h, x:x + w]
        if roi.size:
            rois.append((roi, True))
    return rois


def _extract_plate_panel_candidates(image: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Plakanin beyaz panelini onceleyen daha dar adaylar cikarir."""
    image_h, image_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    search_regions = [
        (int(image_w * 0.40), int(image_h * 0.18), int(image_w * 0.52), int(image_h * 0.32)),
        (int(image_w * 0.22), int(image_h * 0.24), int(image_w * 0.60), int(image_h * 0.32)),
        (int(image_w * 0.18), int(image_h * 0.46), int(image_w * 0.64), int(image_h * 0.38)),
        (int(image_w * 0.26), int(image_h * 0.30), int(image_w * 0.50), int(image_h * 0.42)),
    ]
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    scored_rects: list[tuple[float, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for rx, ry, rw, rh in search_regions:
        region = enhanced[ry:ry + rh, rx:rx + rw]
        if region.size == 0:
            continue
        blackhat = cv2.morphologyEx(region, cv2.MORPH_BLACKHAT, rect_kernel)
        grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = np.absolute(grad_x)
        max_grad = float(grad_x.max())
        if max_grad > 0:
            grad_x = (255.0 * grad_x / max_grad).astype("uint8")
        else:
            grad_x = grad_x.astype("uint8")
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel, iterations=1)
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            area = float(w * h)
            if area < 220:
                continue
            ratio = float(w) / float(h)
            if ratio < 2.0 or ratio > 7.2:
                continue

            gx = rx + x
            gy = ry + y
            width_ratio = w / float(image_w)
            height_ratio = h / float(image_h)
            center_x = (gx + (w / 2.0)) / float(image_w)
            center_y = (gy + (h / 2.0)) / float(image_h)
            if width_ratio < 0.04 or width_ratio > 0.62:
                continue
            if height_ratio < 0.018 or height_ratio > 0.22:
                continue
            if center_x < 0.08 or center_x > 0.92:
                continue
            if center_y < 0.16 or center_y > 0.94:
                continue

            roi_gray = enhanced[gy:gy + h, gx:gx + w]
            if roi_gray.size == 0:
                continue
            local_otsu = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            white_ratio = float(np.count_nonzero(local_otsu)) / float(max(1, w * h))
            if white_ratio < 0.08 or white_ratio > 0.97:
                continue
            edge_density = float(np.count_nonzero(cv2.Canny(roi_gray, 70, 180))) / float(max(1, w * h))
            if edge_density < 0.025:
                continue

            ratio_pref = 1.0 - min(1.0, abs(ratio - 4.1) / 1.6)
            horizontal_pref = 1.0 - min(1.0, abs(center_x - 0.56) / 0.34)
            vertical_pref = 1.0 - min(1.0, abs(center_y - 0.54) / 0.30)
            white_pref = 1.0 - min(1.0, abs(white_ratio - 0.62) / 0.28)
            edge_pref = min(1.0, edge_density / 0.16)
            score = (
                (ratio_pref * 3.0)
                + (horizontal_pref * 2.4)
                + (vertical_pref * 2.4)
                + (white_pref * 2.0)
                + (edge_pref * 2.0)
            )

            pad_x = int(round(w * 0.08))
            pad_y = int(round(h * 0.22))
            x0 = max(0, gx - pad_x)
            y0 = max(0, gy - pad_y)
            x1 = min(image_w, gx + w + pad_x)
            y1 = min(image_h, gy + h + pad_y)
            rect = (x0, y0, x1 - x0, y1 - y0)
            key = (rect[0] // 8, rect[1] // 8, rect[2] // 8, rect[3] // 8)
            if key in seen:
                continue
            seen.add(key)
            scored_rects.append((score, rect))

    scored_rects.sort(key=lambda item: item[0], reverse=True)
    rois: list[tuple[np.ndarray, bool]] = []
    for _, (x, y, w, h) in scored_rects[:DIRECT_PLATE_MAX_CANDIDATES]:
        roi = image[y:y + h, x:x + w]
        if roi.size:
            rois.append((roi, True))
    return rois


def _refine_plate_subroi(roi: np.ndarray) -> tuple[np.ndarray, bool]:
    """Aday ROI icinden daha dar bir plaka kutusu ayiklamaya calisir."""
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # TR plakada beyaz panel cok baskin; onu once hedefleyelim.
    bright_thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)[1]
    otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh_inv = cv2.bitwise_not(otsu_thresh)

    masks = [bright_thresh, otsu_thresh, thresh_inv]
    best_rect: Optional[tuple[int, int, int, int]] = None
    best_score = -1.0
    roi_h, roi_w = gray.shape[:2]

    for mask in masks:
        closed = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
            iterations=1,
        )
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            area = float(w * h)
            if area < 120:
                continue
            ratio = float(w) / float(h)
            if ratio < 1.9 or ratio > 7.0:
                continue
            width_ratio = w / float(roi_w)
            height_ratio = h / float(roi_h)
            if width_ratio < 0.12 or width_ratio > 0.95:
                continue
            if height_ratio < 0.05 or height_ratio > 0.42:
                continue
            center_y = (y + (h / 2.0)) / float(roi_h)
            center_x = (x + (w / 2.0)) / float(roi_w)
            vertical_pref = 1.0 - min(1.0, abs(center_y - 0.60) / 0.38)
            horizontal_pref = 1.0 - min(1.0, abs(center_x - 0.48) / 0.46)
            ratio_pref = 1.0 - min(1.0, abs(ratio - 4.4) / 1.6)
            size_pref = min(1.0, area / 3000.0)
            white_ratio = float(np.count_nonzero(mask[y:y + h, x:x + w])) / float(max(1, w * h))
            white_pref = 1.0 - min(1.0, abs(white_ratio - 0.72) / 0.50)
            panel_quality = _plate_panel_quality(roi[y:y + h, x:x + w])
            if panel_quality < 0.18:
                continue
            score = (
                (vertical_pref * 3.2)
                + (horizontal_pref * 1.6)
                + (ratio_pref * 2.5)
                + (size_pref * 1.2)
                + (white_pref * 1.8)
                + (panel_quality * 4.0)
            )
            if score > best_score:
                best_score = score
                best_rect = (x, y, w, h)

    if best_rect is None:
        return roi, False

    x, y, w, h = best_rect
    pad_x = int(round(w * 0.08))
    pad_y = int(round(h * 0.20))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(roi.shape[1], x + w + pad_x)
    y1 = min(roi.shape[0], y + h + pad_y)
    refined = roi[y0:y1, x0:x1]
    if refined.size == 0:
        return roi, False
    return refined, True


def _save_debug_image(path: Path, image: np.ndarray) -> None:
    """Debug inceleme icin ROI veya preprocess goruntusu yazar."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
    except Exception as exc:
        LOGGER.warning("Debug goruntu yazilamadi (%s): %s", path, exc)


def _save_debug_text(path: Path, lines: list[str]) -> None:
    """Debug inceleme icin metin trace dosyasi yazar."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Debug metni yazilamadi (%s): %s", path, exc)


def _debug_dir(cli_debug_dir: Optional[str]) -> Optional[Path]:
    """Debug klasoru CLI argumani veya env degiskeninden belirler."""
    if cli_debug_dir:
        return Path(cli_debug_dir)
    env_value = os.environ.get("PLATE_READER_DEBUG_DIR", "").strip()
    if env_value:
        return Path(env_value)
    return None


def _finalize_plate_result(plate: Optional[str], confidence: Optional[float]) -> dict[str, Any]:
    """Ortak read_plate sozlesmesine uygun sonuc dondur."""
    return {
        "plate": plate if plate else None,
        "confidence": 0.0 if confidence is None else max(0.0, min(0.99, float(confidence))),
    }


def _resolve_backend(backend: Optional[str], fallback: Optional[str]) -> tuple[str, Optional[str]]:
    """OCR backend secimini arguman veya environment uzerinden cozer."""
    selected = (backend or os.environ.get("PLATE_READER_BACKEND") or DEFAULT_OCR_BACKEND).strip().lower()
    selected_fallback = (fallback or os.environ.get("PLATE_READER_FALLBACK") or DEFAULT_OCR_FALLBACK).strip().lower()
    if selected == selected_fallback:
        selected_fallback = None
    return selected, selected_fallback


def _collect_plate_candidates(
    image: np.ndarray,
    debug_enabled: bool,
) -> list[dict[str, Any]]:
    """Plaka ROI adaylarini OCR backend'lerinden bagimsiz toplar."""
    candidates: list[dict[str, Any]] = []
    candidate_stages = _build_candidate_stages(image)
    prod_processed = 0
    roi_index = 0
    for stage_index, stage_regions in enumerate(candidate_stages, start=1):
        stage_candidates = _select_stage_candidates(stage_regions, image.shape, debug_enabled=debug_enabled)
        if not debug_enabled:
            remaining = PROD_MAX_TOTAL_CANDIDATES - prod_processed
            if remaining <= 0:
                break
            stage_candidates = stage_candidates[:remaining]
        for working_roi, roi_used, roi_metrics, refined_used in stage_candidates:
            roi_index += 1
            prod_processed += 1
            candidates.append(
                {
                    "roi": working_roi,
                    "roi_index": roi_index,
                    "stage_index": stage_index,
                    "roi_used": bool(roi_used or refined_used),
                    "panel_quality": float(roi_metrics["panel_quality"]),
                    "ranking_score": float(roi_metrics["ranking_score"]),
                    "blur_score": float(roi_metrics["blur_score"]),
                    "contrast_score": float(roi_metrics["contrast_score"]),
                    "aspect_ratio": float(roi_metrics["aspect_ratio"]),
                    "area_ratio": float(roi_metrics["area_ratio"]),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["ranking_score"],
            item["panel_quality"],
            item["contrast_score"],
        ),
        reverse=True,
    )
    return candidates if debug_enabled else candidates[:RAPIDOCR_MAX_CANDIDATES]


def _build_light_ocr_variants(roi: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Hafif OCR backend'leri icin az sayida ROI varyanti uret."""
    base_variants = _preprocess_variants(roi)
    variants: list[tuple[str, np.ndarray]] = [("roi", roi)]
    if base_variants:
        variants.append(("gray", base_variants[0]))
    if len(base_variants) > 3:
        variants.append(("otsu", base_variants[3]))
    return variants[:3]


def _build_rapidocr_variants(roi: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """RapidOCR icin detector'a daha uygun, buyutulmus ve padded girisler uret."""
    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return []
    scale = max(2.0, 160.0 / float(max(1, h)))
    enlarged = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad_y = max(16, enlarged.shape[0] // 6)
    pad_x = max(24, enlarged.shape[1] // 12)
    padded = cv2.copyMakeBorder(
        enlarged,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    variants: list[tuple[str, np.ndarray]] = [("padded", padded)]
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    gray_big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray_padded = cv2.copyMakeBorder(
        gray_big,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    variants.append(("gray_padded", gray_padded))
    return variants[:2]


def _build_rapidocr_global_fallbacks(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """RapidOCR tamamen bos donerse, tam goruntuden 1-2 genis bant dener."""
    h, w = image.shape[:2]
    boxes = [
        ("global_lower", (0, int(h * 0.50), w, int(h * 0.38))),
        ("global_lower_left", (0, int(h * 0.52), int(w * 0.78), int(h * 0.34))),
    ]
    outputs: list[tuple[str, np.ndarray]] = []
    for name, (x, y, bw, bh) in boxes:
        crop = image[y : y + bh, x : x + bw]
        if crop.size == 0:
            continue
        outputs.extend((f"{name}_{variant_name}", variant) for variant_name, variant in _build_rapidocr_variants(crop))
    return outputs[:4]


def _iter_paddle_text_candidates(payload: Any) -> list[tuple[str, float]]:
    """PaddleOCR sonuc yapilarindan text/conf ciftleri cikarir."""
    extracted: list[tuple[str, float]] = []
    if payload is None:
        return extracted
    if isinstance(payload, (list, tuple)):
        if len(payload) == 2 and isinstance(payload[0], str):
            try:
                extracted.append((payload[0], float(payload[1])))
            except Exception:
                extracted.append((payload[0], 0.0))
            return extracted
        if len(payload) == 2 and isinstance(payload[1], (list, tuple)) and len(payload[1]) >= 2:
            text = str(payload[1][0])
            try:
                conf = float(payload[1][1])
            except Exception:
                conf = 0.0
            extracted.append((text, conf))
            return extracted
        for item in payload:
            extracted.extend(_iter_paddle_text_candidates(item))
    return extracted


def _iter_rapidocr_text_candidates(payload: Any) -> list[tuple[str, float]]:
    """RapidOCR sonuc yapilarindan text/conf ciftleri cikarir."""
    extracted: list[tuple[str, float]] = []
    if payload is None:
        return extracted
    if hasattr(payload, "txts"):
        txts = list(getattr(payload, "txts") or [])
        scores = list(getattr(payload, "scores", []) or [])
        for idx, text in enumerate(txts):
            score = 0.0
            if idx < len(scores):
                try:
                    score = float(scores[idx])
                except Exception:
                    score = 0.0
            extracted.append((str(text), score))
        return extracted
    if hasattr(payload, "rec_txts"):
        txts = list(getattr(payload, "rec_txts") or [])
        scores = list(getattr(payload, "rec_scores", []) or [])
        for idx, text in enumerate(txts):
            score = 0.0
            if idx < len(scores):
                try:
                    score = float(scores[idx])
                except Exception:
                    score = 0.0
            extracted.append((str(text), score))
        return extracted
    if isinstance(payload, tuple):
        for item in payload:
            extracted.extend(_iter_rapidocr_text_candidates(item))
        return extracted
    if isinstance(payload, list):
        for item in payload:
            # rapidocr_onnxruntime 1.x: [box_points, text, confidence].
            # This must be recognized before descending into box_points;
            # otherwise numeric x/y coordinates are mistaken for OCR text.
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 3
                and isinstance(item[1], str)
                and isinstance(item[2], (float, int))
            ):
                extracted.append((item[1], float(item[2])))
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], (float, int)):
                # Compact [text, confidence] form; reject coordinate pairs.
                if isinstance(item[0], str):
                    extracted.append((item[0], float(item[1])))
                continue
            extracted.extend(_iter_rapidocr_text_candidates(item))
        return extracted
    return extracted


def _get_rapidocr_instance() -> Any:
    """RapidOCR backend'ini lazim oldugunda gec ac."""
    global _RAPIDOCR_INSTANCE, _RAPIDOCR_IMPORT_FAILED
    if _RAPIDOCR_INSTANCE is not None:
        return _RAPIDOCR_INSTANCE
    if _RAPIDOCR_IMPORT_FAILED:
        return None
    try:
        from rapidocr import RapidOCR  # type: ignore
    except Exception:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except Exception as exc:
            _RAPIDOCR_IMPORT_FAILED = True
            LOGGER.warning("RapidOCR backend yuklenemedi, fallback kullanilacak: %s", exc)
            return None
    try:
        _RAPIDOCR_INSTANCE = RapidOCR()
        LOGGER.info("RapidOCR backend hazirlandi.")
        return _RAPIDOCR_INSTANCE
    except Exception as exc:
        _RAPIDOCR_IMPORT_FAILED = True
        LOGGER.warning("RapidOCR instance olusturulamadi, fallback kullanilacak: %s", exc)
        return None


def _run_rapidocr(image: np.ndarray, recognition_only: bool = False) -> list[tuple[str, float]]:
    """RapidOCR backend'ini kullanarak OCR denemesi yapar."""
    engine = _get_rapidocr_instance()
    if engine is None:
        return []
    try:
        if recognition_only:
            # Candidate ROI is already localized as a plate. Running text
            # detection again loses very small/blurred plates (15-25 px high).
            result = engine(image, use_det=False, use_cls=False, use_rec=True)
        else:
            result = engine(image)
    except Exception as exc:
        LOGGER.warning("RapidOCR cagrisi basarisiz: %s", exc)
        return []
    return _iter_rapidocr_text_candidates(result)


def _get_paddleocr_instance() -> Any:
    """Python PaddleOCR backend'i lazim oldugunda gec ac."""
    global _PADDLE_OCR_INSTANCE, _PADDLE_OCR_IMPORT_FAILED
    if _PADDLE_OCR_INSTANCE is not None:
        return _PADDLE_OCR_INSTANCE
    if _PADDLE_OCR_IMPORT_FAILED:
        return None
    try:
        from paddleocr import PaddleOCR  # type: ignore

        _PADDLE_OCR_INSTANCE = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False,
        )
        LOGGER.info("PaddleOCR Python backend hazirlandi.")
        return _PADDLE_OCR_INSTANCE
    except Exception as exc:
        _PADDLE_OCR_IMPORT_FAILED = True
        LOGGER.warning("PaddleOCR Python backend yuklenemedi, fallback kullanilacak: %s", exc)
        return None


def _run_python_paddle_ocr(image: np.ndarray) -> list[tuple[str, float]]:
    """PaddleOCR Python paketini kullanarak OCR denemesi yapar."""
    ocr = _get_paddleocr_instance()
    if ocr is None:
        return []
    try:
        result = ocr.ocr(image, cls=False)
    except TypeError:
        result = ocr.ocr(image)
    except Exception as exc:
        LOGGER.warning("PaddleOCR python cagrisi basarisiz: %s", exc)
        return []
    return _iter_paddle_text_candidates(result)


def _candidate_paddle_commands() -> list[str]:
    """Qengineering Paddle Lite executable yollarini cozer."""
    commands: list[str] = []
    env_value = os.environ.get(PADDLE_LITE_CMD_ENV, "").strip()
    if env_value:
        commands.append(env_value)
    for path in PADDLE_LITE_DEFAULT_PATHS:
        if Path(path).exists():
            commands.append(path)
    which_hit = shutil.which("PaddleOCR-Lite-License")
    if which_hit:
        commands.append(which_hit)
    deduped: list[str] = []
    seen: set[str] = set()
    for cmd in commands:
        if cmd not in seen:
            seen.add(cmd)
            deduped.append(cmd)
    return deduped


def _run_external_paddle_lite(crop_path: Path) -> list[tuple[str, float]]:
    """Harici Paddle Lite executable varsa onu cagirir."""
    text_candidates: list[tuple[str, float]] = []
    for command in _candidate_paddle_commands():
        try:
            completed = subprocess.run(
                [command, str(crop_path)],
                capture_output=True,
                text=True,
                timeout=8.0,
                check=False,
            )
        except Exception as exc:
            LOGGER.warning("Harici paddle_lite komutu calismadi (%s): %s", command, exc)
            continue
        merged_output = "\n".join(
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        )
        if not merged_output:
            continue
        for line in merged_output.splitlines():
            normalized = _normalize_plate_candidate(line)
            if normalized is not None:
                text_candidates.append((normalized, 0.80))
        if text_candidates:
            break
    return text_candidates


def _read_plate_paddle_lite(
    image_path: str,
    image: np.ndarray,
    debug_base: Optional[Path],
) -> dict[str, Any]:
    """Yeni hizli OCR yolu: once ROI sec, sonra dar sayida Paddle OCR uygula."""
    debug_enabled = debug_base is not None
    started_at = time.perf_counter()
    trace_lines: list[str] = []
    candidates = _collect_plate_candidates(image, debug_enabled=debug_enabled)
    best_plate: Optional[str] = None
    best_confidence = 0.0
    best_source = ""
    temp_external_crop = OCR_DIR / "_paddle_lite_crop.png"

    if debug_base is not None:
        trace_lines.append(f"backend=paddle_lite candidates={len(candidates)} image={image_path}")

    for candidate in candidates:
        roi = candidate["roi"]
        roi_index = int(candidate["roi_index"])
        if debug_base is not None:
            _save_debug_image(debug_base / f"paddle_roi_{roi_index:02d}.jpg", roi)
        for variant_name, variant in _build_light_ocr_variants(roi):
            if debug_base is not None:
                _save_debug_image(debug_base / f"paddle_roi_{roi_index:02d}_{variant_name}.png", variant)

            texts = _run_python_paddle_ocr(variant)
            if not texts:
                _save_debug_image(temp_external_crop, variant)
                texts = _run_external_paddle_lite(temp_external_crop)

            for raw_text, raw_conf in texts:
                cleaned = _cleanup_text(raw_text)
                normalized = _normalize_plate_candidate(raw_text)
                plate_like = _plate_like_score(cleaned)
                confidence = max(0.0, min(0.99, float(raw_conf)))
                final_score = (
                    confidence * 0.62
                    + plate_like * 0.18
                    + candidate["panel_quality"] * 0.12
                    + min(1.0, candidate["ranking_score"] / 2.5) * 0.08
                )
                if debug_base is not None:
                    trace_lines.append(
                        f"ROI {roi_index:02d} variant={variant_name} raw={cleaned or '-'} "
                        f"normalized={normalized or '-'} conf={confidence:.3f} "
                        f"panel={candidate['panel_quality']:.3f} rank={candidate['ranking_score']:.3f}"
                    )
                if normalized is None:
                    continue
                if final_score > best_confidence:
                    best_plate = normalized
                    best_confidence = final_score
                    best_source = f"roi={roi_index:02d}/{variant_name}"
                if best_confidence >= 0.92:
                    break
            if best_confidence >= 0.92:
                break
        if best_confidence >= 0.92:
            break

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    if debug_base is not None:
        trace_lines.append(
            f"backend=paddle_lite best_plate={best_plate or '-'} confidence={best_confidence:.3f} "
            f"elapsed_ms={elapsed_ms:.1f} source={best_source or '-'}"
        )
        _save_debug_text(debug_base / "paddle_trace.txt", trace_lines)

    if best_plate is None or best_confidence < PADDLE_LITE_MIN_ACCEPT_CONFIDENCE:
        LOGGER.info(
            "paddle_lite sonuc yetersiz: %s conf=%.2f elapsed_ms=%.1f",
            image_path,
            best_confidence,
            elapsed_ms,
        )
        return _finalize_plate_result(None, 0.0)

    LOGGER.info(
        "paddle_lite plaka okundu: %s -> %s conf=%.2f elapsed_ms=%.1f source=%s",
        image_path,
        best_plate,
        best_confidence,
        elapsed_ms,
        best_source,
    )
    return _finalize_plate_result(best_plate, best_confidence)


def _read_plate_rapidocr(
    image_path: str,
    image: np.ndarray,
    debug_base: Optional[Path],
) -> dict[str, Any]:
    """Ana backend: hizli ROI secimi + RapidOCR OCR."""
    debug_enabled = debug_base is not None
    started_at = time.perf_counter()
    trace_lines: list[str] = []
    candidates = _collect_plate_candidates(image, debug_enabled=debug_enabled)
    best_plate: Optional[str] = None
    best_confidence = 0.0
    best_source = ""

    if debug_base is not None:
        trace_lines.append(f"backend=rapidocr candidates={len(candidates)} image={image_path}")

    for candidate in candidates:
        roi = candidate["roi"]
        roi_index = int(candidate["roi_index"])
        if debug_base is not None:
            _save_debug_image(debug_base / f"rapidocr_roi_{roi_index:02d}.jpg", roi)
        for variant_name, variant in _build_rapidocr_variants(roi):
            if debug_base is not None:
                _save_debug_image(debug_base / f"rapidocr_roi_{roi_index:02d}_{variant_name}.png", variant)
            texts = _run_rapidocr(variant, recognition_only=True)
            for raw_text, raw_conf in texts:
                cleaned = _cleanup_text(raw_text)
                normalized = _normalize_plate_candidate(raw_text)
                plate_like = _plate_like_score(cleaned)
                confidence = max(0.0, min(0.99, float(raw_conf)))
                final_score = (
                    confidence * 0.68
                    + plate_like * 0.18
                    + candidate["panel_quality"] * 0.10
                    + min(1.0, candidate["ranking_score"] / 2.5) * 0.04
                )
                if debug_base is not None:
                    trace_lines.append(
                        f"ROI {roi_index:02d} variant={variant_name} raw={cleaned or '-'} "
                        f"normalized={normalized or '-'} conf={confidence:.3f} "
                        f"panel={candidate['panel_quality']:.3f} rank={candidate['ranking_score']:.3f}"
                    )
                if normalized is None:
                    continue
                if final_score > best_confidence:
                    best_plate = normalized
                    best_confidence = final_score
                    best_source = f"roi={roi_index:02d}/{variant_name}"
                if best_confidence >= 0.90:
                    break
            if best_confidence >= 0.90:
                break
        if best_confidence >= 0.90:
            break

    if best_plate is None:
        for variant_name, variant in _build_rapidocr_global_fallbacks(image):
            if debug_base is not None:
                _save_debug_image(debug_base / f"{variant_name}.png", variant)
            texts = _run_rapidocr(variant)
            for raw_text, raw_conf in texts:
                cleaned = _cleanup_text(raw_text)
                normalized = _normalize_plate_candidate(raw_text)
                plate_like = _plate_like_score(cleaned)
                confidence = max(0.0, min(0.99, float(raw_conf)))
                final_score = (confidence * 0.72) + (plate_like * 0.28)
                if debug_base is not None:
                    trace_lines.append(
                        f"GLOBAL variant={variant_name} raw={cleaned or '-'} normalized={normalized or '-'} conf={confidence:.3f}"
                    )
                if normalized is None:
                    continue
                if final_score > best_confidence:
                    best_plate = normalized
                    best_confidence = final_score
                    best_source = variant_name

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    if debug_base is not None:
        trace_lines.append(
            f"backend=rapidocr best_plate={best_plate or '-'} confidence={best_confidence:.3f} "
            f"elapsed_ms={elapsed_ms:.1f} source={best_source or '-'}"
        )
        _save_debug_text(debug_base / "rapidocr_trace.txt", trace_lines)

    if best_plate is None or best_confidence < RAPIDOCR_MIN_ACCEPT_CONFIDENCE:
        LOGGER.info(
            "rapidocr sonuc yetersiz: %s conf=%.2f elapsed_ms=%.1f",
            image_path,
            best_confidence,
            elapsed_ms,
        )
        return _finalize_plate_result(None, 0.0)

    LOGGER.info(
        "rapidocr plaka okundu: %s -> %s conf=%.2f elapsed_ms=%.1f source=%s",
        image_path,
        best_plate,
        best_confidence,
        elapsed_ms,
        best_source,
    )
    return _finalize_plate_result(best_plate, best_confidence)


def _extract_text_candidates_with_configs(
    image: np.ndarray,
    configs: list[str],
    call_counter: Optional[dict[str, int]] = None,
) -> list[tuple[str, Optional[float], str]]:
    """Bir preprocess cikti icin birden fazla Tesseract modundan adaylar dondurur."""
    candidates: list[tuple[str, Optional[float], str]] = []
    for config in configs:
        try:
            if call_counter is not None:
                call_counter["count"] = call_counter.get("count", 0) + 1
            data = pytesseract.image_to_data(
                image,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            texts: list[str] = []
            confidences: list[float] = []
            for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
                text = _cleanup_text(str(raw_text))
                if not text:
                    continue
                try:
                    conf_value = float(raw_conf)
                except Exception:
                    conf_value = -1.0
                if conf_value >= 0:
                    confidences.append(conf_value)
                texts.append(text)
            combined = "".join(texts)
            confidence = None
            if confidences:
                confidence = max(0.0, min(1.0, float(sum(confidences) / len(confidences)) / 100.0))
            if combined:
                candidates.append((combined, confidence, config))
        except Exception:
            pass

        try:
            if call_counter is not None:
                call_counter["count"] = call_counter.get("count", 0) + 1
            raw = pytesseract.image_to_string(image, config=config)
            cleaned = _cleanup_text(raw)
            if cleaned:
                candidates.append((cleaned, None, config + " string"))
        except Exception:
            pass
    return candidates


def _extract_text_candidates(image: np.ndarray) -> list[tuple[str, Optional[float], str]]:
    """Geriye donuk uyumluluk icin varsayilan OCR wrapper'i."""
    return _extract_text_candidates_with_configs(image, list(TESSERACT_CONFIGS))


def _plate_like_score(cleaned_text: str) -> float:
    """Regex olmasa bile plaka benzerligini puanlar."""
    score = 0.0
    if len(cleaned_text) >= 7:
        score += 0.2
    if re.match(r"^(0[1-9]|[1-7][0-9]|8[01])", cleaned_text):
        score += 0.35
    letter_match = re.search(r"[A-Z]{1,3}", cleaned_text[2:6] if len(cleaned_text) >= 4 else cleaned_text)
    if letter_match:
        score += 0.2
    digit_match = re.search(r"[0-9]{2,4}$", cleaned_text)
    if digit_match:
        score += 0.25
    return score


def _is_reliable_match(normalized: Optional[str], confidence: Optional[float], roi_used: bool, plate_like: float) -> bool:
    """Yanlis fallback eslesmeleri bastirmak icin kabul kurali."""
    if normalized is None:
        return False
    conf = confidence or 0.0
    if roi_used:
        return conf >= 0.08 or plate_like >= 0.92
    return conf >= 0.35 and plate_like >= 0.95


def _merge_partial_votes(
    partial_votes: dict[str, dict[str, float]],
    normalized_votes: dict[str, float],
) -> tuple[Optional[str], float]:
    """Parca oylarindan gecerli bir TR plaka adayi uretmeye calisir."""
    best_normalized = max(normalized_votes.items(), key=lambda item: item[1])[0] if normalized_votes else None
    best_score = normalized_votes.get(best_normalized, 0.0) if best_normalized else -1.0

    city_items = sorted(partial_votes["city"].items(), key=lambda item: item[1], reverse=True)[:6]
    letter_items = sorted(partial_votes["letters"].items(), key=lambda item: item[1], reverse=True)[:8]
    tail_items = sorted(partial_votes["tail"].items(), key=lambda item: item[1], reverse=True)[:8]

    strong_letter_items = [item for item in letter_items if len(item[0]) == 3]
    strong_tail_items = [item for item in tail_items if len(item[0]) == 3]
    if strong_letter_items:
        letter_items = strong_letter_items
    if strong_tail_items:
        tail_items = strong_tail_items

    for city, city_score in city_items:
        for letters, letters_score in letter_items:
            for tail, tail_score in tail_items:
                candidate = f"{city}{letters}{tail}"
                if not PLATE_REGEX.match(candidate):
                    continue
                combo_score = city_score + letters_score + tail_score + _plate_shape_bonus(letters, tail)
                if candidate in normalized_votes:
                    combo_score += normalized_votes[candidate]
                if len(letters) < 3:
                    combo_score -= 1.0
                if len(tail) < 3:
                    combo_score -= 1.0
                if combo_score > best_score:
                    best_normalized = candidate
                    best_score = combo_score

    # Eger ayni sehir+harf icin tail'ler yarisiyorsa, tail oyunu daha guclu olan
    # 3 haneli kuyrugu one cek. Bu sahnede "507" ve "502" gibi ayrimlar icin yardimci olur.
    if best_normalized is not None:
        parts = _split_plate_components(best_normalized)
        if parts is not None:
            best_city, best_letters, best_tail = parts
            if len(best_letters) == 3 and len(best_tail) == 3:
                current_tail_score = partial_votes["tail"].get(best_tail, 0.0)
                for tail, tail_score in tail_items:
                    if len(tail) != 3 or tail == best_tail:
                        continue
                    candidate = f"{best_city}{best_letters}{tail}"
                    if not PLATE_REGEX.match(candidate):
                        continue
                    combo_score = (
                        partial_votes["city"].get(best_city, 0.0)
                        + partial_votes["letters"].get(best_letters, 0.0)
                        + tail_score
                        + _plate_shape_bonus(best_letters, tail)
                        + (normalized_votes.get(candidate, 0.0) * 0.35)
                    )
                    tail_advantage = tail_score - current_tail_score
                    if tail_advantage >= 0.45 and combo_score >= (best_score - 0.60):
                        best_normalized = candidate
                        best_score = combo_score
                        break
    return best_normalized, best_score


def _is_final_result_reliable(best_match: Optional[dict[str, Any]], normalized_votes: dict[str, float]) -> bool:
    """Son aday fallback kaynakli zayif bir yanlis-pozitif mi kontrol eder."""
    if best_match is None:
        return False
    plate = best_match.get("plate")
    if not plate:
        return False

    roi_used = bool(best_match.get("roi_used"))
    confidence = best_match.get("confidence")
    source = str(best_match.get("source") or "")
    score = float(best_match.get("score") or 0.0)
    normalized_score = float(normalized_votes.get(plate, 0.0))
    hit_count = int(best_match.get("hit_count") or 0)
    panel_quality = float(best_match.get("panel_quality") or 0.0)

    if roi_used and hit_count >= 2 and normalized_score >= 1.8:
        return True
    if roi_used and hit_count >= 1 and panel_quality >= SOFT_ACCEPT_PANEL_QUALITY and normalized_score >= max(1.85, SOFT_ACCEPT_NORMALIZED_SCORE + 0.40):
        return True
    if roi_used and confidence is not None and float(confidence) >= 0.82 and panel_quality >= 0.45:
        return True
    if "merged-votes" in source:
        return (hit_count >= 2 and normalized_score >= 2.8) or score >= 3.2
    return hit_count >= 2 and score >= 2.35 and normalized_score >= 2.0


def _compute_final_confidence(best_match: Optional[dict[str, Any]], normalized_votes: dict[str, float]) -> Optional[float]:
    """Ham OCR'den bagimsiz, son karar icin anlamli 0..1 guven skoru uretir."""
    if best_match is None or not best_match.get("plate"):
        return None

    plate = str(best_match["plate"])
    source = str(best_match.get("source") or "")
    raw_conf = best_match.get("confidence")
    score = float(best_match.get("score") or 0.0)
    hit_count = int(best_match.get("hit_count") or 0)
    panel_quality = float(best_match.get("panel_quality") or 0.0)
    normalized_score = float(normalized_votes.get(plate, 0.0))

    confidence = 0.20
    confidence += min(0.24, normalized_score / 8.0)
    confidence += min(0.18, score / 12.0)
    confidence += min(0.14, hit_count * 0.05)
    confidence += min(0.14, panel_quality * 0.20)

    if raw_conf is not None:
        confidence += min(0.10, float(raw_conf) * 0.20)

    if "digit-tail-refine" in source:
        confidence += 0.10
    elif "raw-tail-refine" in source:
        confidence += 0.07
    elif "merged-votes" in source:
        confidence += 0.05

    if best_match.get("roi_used"):
        confidence += 0.05

    return max(0.0, min(0.99, confidence))


def _build_candidate_pool(image: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """En guvenilir adaylari onceleyen ROI havuzu uretir."""
    direct_panel = _extract_plate_panel_candidates(image)
    if direct_panel:
        return direct_panel + _extract_direct_plate_regions(image)
    direct = _extract_direct_plate_regions(image)
    if direct:
        return direct + _extract_candidate_regions(image)
    return _extract_candidate_regions(image)


def _build_candidate_stages(image: np.ndarray) -> list[list[tuple[np.ndarray, bool]]]:
    """Adaylari guven seviyesine gore asamali islemek icin gruplar."""
    panel = _extract_plate_panel_candidates(image)[:STAGE_LIMIT_PANEL]
    direct = _extract_direct_plate_regions(image)[:STAGE_LIMIT_DIRECT]
    generic_all = _extract_candidate_regions(image)
    generic_primary = [item for item in generic_all if item[1]][: max(1, STAGE_LIMIT_GENERIC - 1)]
    generic_fallback = [item for item in generic_all if not item[1]][:2]
    generic = (generic_primary + generic_fallback)[:STAGE_LIMIT_GENERIC]

    stages: list[list[tuple[np.ndarray, bool]]] = []
    if panel:
        stages.append(panel)
    if direct:
        stages.append(direct)
    if generic:
        stages.append(generic)
    return stages


def _select_stage_candidates(
    stage_regions: list[tuple[np.ndarray, bool]],
    image_shape: tuple[int, ...],
    debug_enabled: bool,
) -> list[tuple[np.ndarray, bool, dict[str, float], bool]]:
    """Her asama icin adaylari uretim veya debug ihtiyacina gore hazirlar."""
    selected: list[tuple[np.ndarray, bool, dict[str, float], bool]] = []
    for roi, roi_used in stage_regions:
        working_roi, refined_used = _refine_plate_subroi(roi)
        metrics = _production_candidate_metrics(working_roi, roi_used, refined_used, image_shape)
        passes_filter = _passes_production_roi_filter(metrics)
        if debug_enabled or passes_filter:
            selected.append((working_roi, roi_used, metrics, refined_used))

    if debug_enabled:
        return selected

    selected.sort(
        key=lambda item: (
            item[2]["ranking_score"],
            item[2]["panel_quality"],
            item[2]["contrast_score"],
        ),
        reverse=True,
    )
    return selected


def _is_strong_early_match(match: Optional[dict[str, Any]]) -> bool:
    """Yeterince guclu bir ROI bulundugunda sonraki adaylari isleme."""
    if not match or not match.get("plate"):
        return False
    if not bool(match.get("roi_used")):
        return False
    confidence = match.get("confidence")
    score = float(match.get("score") or 0.0)
    hit_count = int(match.get("hit_count") or 0)
    panel_quality = float(match.get("panel_quality") or 0.0)
    if hit_count < 2:
        return False
    if confidence is not None and float(confidence) >= EARLY_ACCEPT_CONFIDENCE:
        return True
    return score >= EARLY_ACCEPT_SCORE and panel_quality >= 0.40


def _should_run_tail_digit_refine(match: Optional[dict[str, Any]]) -> bool:
    """Digits-only ikinci pass'i yalnizca guclu adaylarda calistir."""
    if not match or not match.get("plate") or match.get("plate_roi") is None:
        return False
    parts = _split_plate_components(str(match.get("plate") or ""))
    if parts is not None:
        _city, letters, tail = parts
        if len(letters) == 3 and len(tail) == 3 and bool(match.get("roi_used")):
            return True
    return (
        float(match.get("panel_quality") or 0.0) >= STRONG_TAIL_REFINEMENT_PANEL_QUALITY
        or float(match.get("score") or 0.0) >= STRONG_TAIL_REFINEMENT_SCORE
    )


def _read_plate_tesseract(image_path: str, debug_dir: Optional[str] = None) -> dict[str, Any]:
    """Tesseract tabanli mevcut OCR akisini koruyan fallback yol."""
    started_at = time.perf_counter()
    metrics = {
        "mode": "prod",
        "roi_count": 0,
        "selected_candidates": 0,
        "tesseract_call_count": 0,
    }
    try:
        image = cv2.imread(image_path)
        if image is None:
            LOGGER.warning("Goruntu okunamadi: %s", image_path)
            return _finalize_plate_result(None, 0.0)

        debug_base = _debug_dir(debug_dir)
        debug_enabled = debug_base is not None
        metrics["mode"] = "debug" if debug_enabled else "prod"
        best_match: Optional[dict[str, Any]] = None
        normalized_votes: dict[str, float] = defaultdict(float)
        strong_normalized_votes: dict[str, float] = defaultdict(float)
        normalized_hits: dict[str, int] = defaultdict(int)
        strong_normalized_hits: dict[str, int] = defaultdict(int)
        partial_votes: dict[str, dict[str, float]] = {
            "city": defaultdict(float),
            "letters": defaultdict(float),
            "tail": defaultdict(float),
        }
        strong_partial_votes: dict[str, dict[str, float]] = {
            "city": defaultdict(float),
            "letters": defaultdict(float),
            "tail": defaultdict(float),
        }
        strong_raw_records: list[tuple[str, float]] = []
        debug_trace_lines: list[str] = []
        candidate_stages = _build_candidate_stages(image)
        roi_index = 0
        prod_processed = 0
        ocr_call_counter = {"count": 0}
        for stage_regions in candidate_stages:
            stage_candidates = _select_stage_candidates(
                stage_regions,
                image.shape,
                debug_enabled=debug_enabled,
            )
            if not debug_enabled:
                remaining = PROD_MAX_TOTAL_CANDIDATES - prod_processed
                if remaining <= 0:
                    break
                stage_candidates = stage_candidates[:remaining]

            for working_roi, roi_used, roi_metrics, refined_used in stage_candidates:
                roi_index += 1
                metrics["roi_count"] += 1
                metrics["selected_candidates"] += 1
                prod_processed += 1
                support_is_strong = bool(roi_used or refined_used)
                panel_quality = roi_metrics["panel_quality"]
                if debug_base is not None:
                    debug_trace_lines.append(
                        "ROI "
                        f"{roi_index:02d} candidate roi_used={roi_used} refined_used={refined_used} "
                        f"panel_quality={panel_quality:.3f} blur={roi_metrics['blur_score']:.1f} "
                        f"contrast={roi_metrics['contrast_score']:.1f} "
                        f"ratio={roi_metrics['aspect_ratio']:.2f} area={roi_metrics['area_ratio']:.4f} "
                        f"shape={working_roi.shape[1]}x{working_roi.shape[0]}"
                    )
                if debug_base is not None:
                    _save_debug_image(
                        debug_base / f"roi_{roi_index:02d}_{'candidate' if roi_used else 'fallback'}.jpg",
                        working_roi,
                    )
                    if refined_used:
                        _save_debug_image(debug_base / f"roi_{roi_index:02d}_refined.jpg", working_roi)
                processed_variants = _select_variant_subset(_preprocess_variants(working_roi), debug_enabled=debug_enabled)
                for variant_index, processed in enumerate(processed_variants, start=1):
                    if debug_base is not None:
                        _save_debug_image(
                            debug_base / f"roi_{roi_index:02d}_variant_{variant_index:02d}.png",
                            processed,
                        )
                    for raw_text, confidence, source in _extract_text_candidates_with_configs(
                        processed,
                        _select_tesseract_configs(debug_enabled=debug_enabled),
                        call_counter=ocr_call_counter,
                    ):
                        cleaned_text = _cleanup_text(raw_text)
                        normalized = _normalize_plate_candidate(raw_text)
                        plate_like = _plate_like_score(cleaned_text)
                        if debug_base is not None:
                            debug_trace_lines.append(
                                f"ROI {roi_index:02d} VAR {variant_index:02d} src={source} conf={confidence if confidence is not None else 'None'} raw={cleaned_text or '-'} normalized={normalized or '-'} plate_like={plate_like:.3f}"
                            )
                        base_score = (
                            (confidence or 0.0)
                            + plate_like
                            + (0.40 if roi_used else 0.0)
                            + (0.40 if refined_used else 0.0)
                            + (panel_quality * 1.35)
                        )
                        if support_is_strong and len(cleaned_text) >= 6:
                            strong_raw_records.append((cleaned_text, base_score))
                        for city, letters, tail in _extract_partial_plate_components(cleaned_text):
                            partial_votes["city"][city] += base_score
                            partial_votes["letters"][letters] += base_score
                            partial_votes["tail"][tail] += base_score
                            if support_is_strong:
                                strong_partial_votes["city"][city] += base_score
                                strong_partial_votes["letters"][letters] += base_score
                                strong_partial_votes["tail"][tail] += base_score
                        if support_is_strong and len(cleaned_text) >= 6 and (plate_like >= 0.60 or panel_quality >= 0.52):
                            loose_components = _extract_loose_plate_components(cleaned_text)
                            for city in loose_components["city"]:
                                strong_partial_votes["city"][city] += base_score * 0.65
                            for letters in loose_components["letters"]:
                                strong_partial_votes["letters"][letters] += base_score * 0.80
                            for tail in loose_components["tail"]:
                                strong_partial_votes["tail"][tail] += base_score * 0.80
                        if normalized is None and plate_like < 0.7:
                            continue
                        if normalized is not None and not _is_reliable_match(normalized, confidence, roi_used, plate_like):
                            continue
                        score = base_score
                        if normalized is not None:
                            score += 1.0
                            normalized_votes[normalized] += score
                            normalized_hits[normalized] += 1
                            if support_is_strong:
                                strong_normalized_votes[normalized] += score
                                strong_normalized_hits[normalized] += 1
                        if best_match is None or score > best_match["score"]:
                            best_match = {
                                "plate": normalized,
                                "confidence": confidence,
                                "roi_used": roi_used or refined_used,
                                "score": score,
                                "raw_text": raw_text,
                                "source": source,
                                "hit_count": 0 if normalized is None else strong_normalized_hits[normalized] if support_is_strong else normalized_hits[normalized],
                                "panel_quality": panel_quality,
                                "plate_roi": working_roi.copy(),
                            }
                if _is_strong_early_match(best_match):
                    break
            if _is_strong_early_match(best_match):
                break
            if not debug_enabled and prod_processed >= PROD_MAX_TOTAL_CANDIDATES:
                break

        merged_plate, merged_score = _merge_partial_votes(strong_partial_votes, strong_normalized_votes)
        if merged_plate is not None:
            merged_hits = strong_normalized_hits.get(merged_plate, 0)
            if merged_score >= 3.2 and (
                best_match is None
                or best_match.get("plate") is None
                or merged_score >= (best_match.get("score", 0.0) - 0.20)
            ):
                best_match = {
                    "plate": merged_plate,
                    "confidence": None if best_match is None else best_match.get("confidence"),
                    "roi_used": True,
                    "score": max(merged_score, 1.0),
                    "raw_text": None if best_match is None else best_match.get("raw_text"),
                    "source": "merged-votes" if best_match is None else f"{best_match.get('source', 'ocr')} + merged-votes",
                    "hit_count": merged_hits,
                    "panel_quality": 0.0 if best_match is None else best_match.get("panel_quality", 0.0),
                    "plate_roi": None if best_match is None else best_match.get("plate_roi"),
                }

        if best_match is not None and best_match.get("plate"):
            refined_plate = _refine_plate_from_raw_evidence(str(best_match["plate"]), strong_raw_records)
            if refined_plate != best_match["plate"]:
                best_match["plate"] = refined_plate
                best_match["source"] = f"{best_match.get('source', 'ocr')} + raw-tail-refine"

        if _should_run_tail_digit_refine(best_match):
            digit_refined_plate = _refine_tail_with_digit_crop(
                str(best_match["plate"]),
                best_match.get("plate_roi"),
                debug_base,
                call_counter=ocr_call_counter,
            )
            if digit_refined_plate != best_match["plate"]:
                best_match["plate"] = digit_refined_plate
                best_match["source"] = f"{best_match.get('source', 'ocr')} + digit-tail-refine"

        if debug_base is not None:
            _save_debug_text(debug_base / "ocr_trace.txt", debug_trace_lines)

        metrics["tesseract_call_count"] = ocr_call_counter["count"]
        total_ms = (time.perf_counter() - started_at) * 1000.0
        if not _is_final_result_reliable(best_match, strong_normalized_votes):
            LOGGER.info(
                "Plaka okunamadi: %s mode=%s total_ms=%.1f roi_count=%d selected_candidates=%d tesseract_call_count=%d",
                image_path,
                metrics["mode"],
                total_ms,
                metrics["roi_count"],
                metrics["selected_candidates"],
                metrics["tesseract_call_count"],
            )
            return _finalize_plate_result(None, 0.0)

        final_confidence = _compute_final_confidence(best_match, strong_normalized_votes)

        LOGGER.info(
            "Plaka okundu: %s -> %s (raw=%s, source=%s, final_conf=%.2f, mode=%s, total_ms=%.1f, roi_count=%d, selected_candidates=%d, tesseract_call_count=%d)",
            image_path,
            best_match["plate"],
            best_match.get("raw_text"),
            best_match.get("source"),
            -1.0 if final_confidence is None else float(final_confidence),
            metrics["mode"],
            total_ms,
            metrics["roi_count"],
            metrics["selected_candidates"],
            metrics["tesseract_call_count"],
        )
        return _finalize_plate_result(best_match["plate"], final_confidence)
    except Exception as exc:
        LOGGER.exception("Plate reader hatasi (%s): %s", image_path, exc)
        return _finalize_plate_result(None, 0.0)


def read_plate(
    image_path: str,
    debug_dir: Optional[str] = None,
    backend: Optional[str] = None,
    fallback: Optional[str] = DEFAULT_OCR_FALLBACK,
) -> dict[str, Any]:
    """Plate reader ana giris noktasi; varsayilan olarak paddle_lite dener."""
    selected_backend, selected_fallback = _resolve_backend(backend, fallback)
    image = cv2.imread(image_path)
    if image is None:
        LOGGER.warning("Goruntu okunamadi: %s", image_path)
        return _finalize_plate_result(None, 0.0)

    debug_base = _debug_dir(debug_dir)
    backends_to_try = [selected_backend]
    if selected_fallback:
        backends_to_try.append(selected_fallback)

    tried: list[str] = []
    rapidocr_available = _get_rapidocr_instance() is not None if selected_backend == "rapidocr" or selected_fallback == "rapidocr" else False
    paddle_available = (
        _get_paddleocr_instance() is not None or bool(_candidate_paddle_commands())
        if selected_backend == "paddle_lite" or selected_fallback == "paddle_lite"
        else False
    )
    for backend_name in backends_to_try:
        if backend_name in tried:
            continue
        tried.append(backend_name)
        if backend_name == "rapidocr":
            if not rapidocr_available:
                continue
            result = _read_plate_rapidocr(image_path, image, debug_base)
            if result.get("plate"):
                return _finalize_plate_result(result.get("plate"), result.get("confidence"))
            continue
        elif backend_name == "paddle_lite":
            if not paddle_available:
                continue
            result = _read_plate_paddle_lite(image_path, image, debug_base)
            if result.get("plate"):
                return _finalize_plate_result(result.get("plate"), result.get("confidence"))
            continue
        elif backend_name == "tesseract":
            result = _read_plate_tesseract(image_path, debug_dir=debug_dir)
        else:
            LOGGER.warning("Bilinmeyen OCR backend'i istendi: %s", backend_name)
            continue

        if result.get("plate"):
            return _finalize_plate_result(result.get("plate"), result.get("confidence"))

    return _finalize_plate_result(None, 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RadarPi plaka okuyucu",
    )
    parser.add_argument("image_path", help="Islenecek goruntu yolu")
    parser.add_argument(
        "legacy_debug_dir",
        nargs="?",
        help="Geriye uyumluluk icin opsiyonel debug klasoru",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Genis debug taramasi ve debug ciktilari ac",
    )
    parser.add_argument(
        "--debug-dir",
        dest="debug_dir",
        help="Debug ciktilarinin kaydedilecegi klasor",
    )
    parser.add_argument(
        "--backend",
        choices=["rapidocr", "paddle_lite", "tesseract"],
        default=None,
        help="Tercih edilen OCR backend'i",
    )
    parser.add_argument(
        "--fallback",
        choices=["rapidocr", "paddle_lite", "tesseract"],
        default=DEFAULT_OCR_FALLBACK,
        help="Birincil backend basarisiz olursa dusulecek backend",
    )
    args = parser.parse_args()

    debug_dir_arg = args.debug_dir or args.legacy_debug_dir
    if args.debug and debug_dir_arg is None:
        debug_dir_arg = os.environ.get("PLATE_READER_DEBUG_DIR", "").strip() or "plate_debug"

    result = read_plate(
        args.image_path,
        debug_dir=debug_dir_arg,
        backend=args.backend,
        fallback=args.fallback,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
