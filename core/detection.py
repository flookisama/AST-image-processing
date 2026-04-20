"""
Disk detection pipeline — adaptive, multi-strategy.

Detection hierarchy:
  1. Analyze image characteristics (dark vs bright background, contrast)
  2. Choose primary strategy (bright-blob or dark-circle)
  3. Run Hough as secondary strategy
  4. Merge and deduplicate candidates
  5. Validate each candidate with an adaptive circularity test

This replaces the brittle single-strategy Hough + dark-circle filter
that failed on any image not matching the expected appearance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import AppConfig

log = logging.getLogger(__name__)

# ((cx, cy), radius_px)
Candidate = Tuple[Tuple[float, float], float]


# ── Image utilities ────────────────────────────────────────────────────────────

def to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("Empty or invalid image")
    out = img
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]
    if out.ndim == 3:
        out = cv2.cvtColor(
            out.astype(np.uint8) if out.dtype != np.uint8 else out,
            cv2.COLOR_RGB2GRAY,
        )
    if out.dtype != np.uint8:
        out = (out * 255).astype(np.uint8) if out.max() <= 1.0 else out.astype(np.uint8)
    return out


def enhance(gray: np.ndarray, cfg: AppConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=float(cfg.clahe_clip),
        tileGridSize=(int(cfg.clahe_grid), int(cfg.clahe_grid)),
    )
    return clahe.apply(gray)


# ── Image analysis ─────────────────────────────────────────────────────────────

@dataclass
class ImageStats:
    dark_background: bool   # True if most of image is dark (< 80)
    low_contrast: bool      # True if std(center) < 25
    mean_brightness: float
    has_vignette: bool      # True if center is significantly brighter than edges

    @property
    def disk_appearance(self) -> str:
        """Expected visual appearance of antibiotic paper disks."""
        # Dark background → disks appear bright white against dark zones
        # Bright background → disks are dark pellets on bright agar
        return "bright" if self.dark_background else "dark"


def analyze_image(gray: np.ndarray) -> ImageStats:
    h, w = gray.shape
    # Sample center region (avoid edges)
    cy, cx = h // 2, w // 2
    rh, rw = h // 4, w // 4
    center = gray[cy - rh : cy + rh, cx - rw : cx + rw]
    edge_strips = np.concatenate([
        gray[:h // 8, :].ravel(),
        gray[-h // 8 :, :].ravel(),
        gray[:, :w // 8].ravel(),
        gray[:, -w // 8 :].ravel(),
    ])
    mean_center = float(np.mean(center))
    mean_edge = float(np.mean(edge_strips))
    std_center = float(np.std(center))

    return ImageStats(
        dark_background=float(np.median(gray)) < 80,
        low_contrast=std_center < 25.0,
        mean_brightness=mean_center,
        has_vignette=(mean_center - mean_edge) > 30,
    )


# ── Petri dish crop ────────────────────────────────────────────────────────────

def crop_petri(gray: np.ndarray, cfg: AppConfig) -> Tuple[np.ndarray, int, int]:
    """
    Detect and crop the Petri dish.
    Returns (cropped_gray, offset_x, offset_y).

    Tries three strategies in order:
      1. Hough large circle
      2. Largest bright/dark contour (adapts to background)
      3. Full image (no crop)
    """
    h, w = gray.shape
    enhanced = enhance(gray, cfg)
    blurred = cv2.GaussianBlur(enhanced, (21, 21), 3)

    # Strategy 1: Hough large circle
    for param2 in [30, 20, 15]:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=min(w, h) // 2,
            param1=50,
            param2=param2,
            minRadius=min(w, h) // 6,
            maxRadius=min(w, h) // 2,
        )
        if circles is not None:
            # Pick largest circle (most likely the dish)
            c = sorted(circles[0], key=lambda x: x[2], reverse=True)[0]
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            margin = int(r * 0.06)
            x1 = max(0, cx - r - margin)
            y1 = max(0, cy - r - margin)
            x2 = min(w, cx + r + margin)
            y2 = min(h, cy + r + margin)
            crop = gray[y1:y2, x1:x2]
            if crop.size > 120 * 120:
                log.debug("crop_petri: Hough (r=%d, param2=%d)", r, param2)
                return crop, x1, y1

    # Strategy 2: largest contour (adapts to dark/bright background)
    median_val = float(np.median(blurred))
    th_input = (255 - blurred) if median_val < 80 else blurred
    _, th = cv2.threshold(th_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.10 * h * w:   # at least 10% of image
            x, y, bw, bh = cv2.boundingRect(largest)
            margin = int(0.02 * max(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            bw = min(w - x, bw + 2 * margin)
            bh = min(h - y, bh + 2 * margin)
            crop = gray[y : y + bh, x : x + bw]
            if crop.size > 120 * 120:
                log.debug("crop_petri: contour fallback")
                return crop, x, y

    log.debug("crop_petri: no crop (full image)")
    return gray, 0, 0


# ── Candidate validation ───────────────────────────────────────────────────────

def _score_candidate(
    gray: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    stats: ImageStats,
) -> float:
    """
    Score a candidate circle on a 0-1 scale.
    Adaptive: accepts both bright disks (on dark agar) and dark disks (on bright agar).
    Returns 0 = reject, >0 = keep (higher = more confident).
    """
    h, w = gray.shape
    yg, xg = np.ogrid[:h, :w]
    dist = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)

    inner_mask = dist < r * 0.60
    ring_mask = (dist >= r * 1.2) & (dist < r * 2.2)

    if not np.any(inner_mask) or not np.any(ring_mask):
        return 0.5   # can't evaluate, give neutral score

    inner_vals = gray[inner_mask].astype(float)
    ring_vals = gray[ring_mask].astype(float)

    inner_mean = float(np.mean(inner_vals))
    ring_p50 = float(np.percentile(ring_vals, 50))

    # Contrast between disk and surrounding
    contrast = abs(inner_mean - ring_p50) / 255.0

    if contrast < 0.04:      # completely uniform — not a disk
        return 0.0

    if stats.dark_background:
        # Expect bright disk (paper disk) on dark inhibition zone
        if inner_mean > ring_p50:
            return min(1.0, contrast * 3)
        # Could still be valid if it's an inhibition zone — give partial credit
        return contrast * 0.5
    else:
        # Expect dark disk on bright agar
        if inner_mean < ring_p50:
            return min(1.0, contrast * 3)
        return contrast * 0.5


def _is_valid_candidate(
    gray: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    stats: ImageStats,
    min_score: float = 0.08,
) -> bool:
    return _score_candidate(gray, cx, cy, r, stats) >= min_score


# ── Strategy 1: bright blob detection (for dark backgrounds) ──────────────────

def _detect_bright_blobs(
    gray: np.ndarray,
    cfg: AppConfig,
    stats: ImageStats,
) -> List[Candidate]:
    """
    Find bright circular blobs via morphological opening + contour fitting.
    Designed for images where paper disks appear white on dark inhibition zones.
    """
    h, w = gray.shape
    min_r = max(5, int(min(w, h) * cfg.hough_min_r_ratio))
    max_r = max(min_r + 20, int(min(w, h) * cfg.hough_max_r_ratio))

    # Normalize and threshold to isolate bright objects
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, bright = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove tiny blobs, keep disk-sized ones
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, min_r // 2), max(3, min_r // 2)))
    opened = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k_open, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Candidate] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * (min_r * 0.6) ** 2:
            continue
        if area > np.pi * (max_r * 1.8) ** 2:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)

        # Circularity check
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        if circularity < 0.45:
            continue

        if min_r * 0.5 <= r <= max_r * 1.8:
            candidates.append(((float(cx), float(cy)), float(r)))

    log.debug("bright-blob: %d candidates", len(candidates))
    return candidates


# ── Strategy 2: Hough circles ─────────────────────────────────────────────────

def _detect_hough(
    gray: np.ndarray,
    cfg: AppConfig,
) -> List[Candidate]:
    h, w = gray.shape
    min_r = max(5, int(min(w, h) * cfg.hough_min_r_ratio))
    max_r = max(min_r + 20, int(min(w, h) * cfg.hough_max_r_ratio))

    enhanced = enhance(gray, cfg)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

    best: Optional[np.ndarray] = None
    base_p2 = float(cfg.hough_param2)
    for p2 in [base_p2, max(15.0, base_p2 - 10), min(60.0, base_p2 + 10)]:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=float(cfg.hough_dp),
            minDist=min_r * 2,
            param1=float(cfg.hough_param1),
            param2=p2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            if best is None or len(circles[0]) > len(best[0]):
                best = circles
            if len(circles[0]) >= 4:
                break

    candidates: List[Candidate] = []
    if best is not None:
        best = np.around(best).astype(int)
        for c in best[0]:
            candidates.append(((float(c[0]), float(c[1])), float(c[2])))

    log.debug("hough: %d candidates", len(candidates))
    return candidates


# ── Strategy 3: dark-zone center detection ────────────────────────────────────

def _detect_dark_zones(
    gray: np.ndarray,
    cfg: AppConfig,
) -> List[Candidate]:
    """
    Detect large dark circular zones (inhibition areas) and return their centers.
    Useful when the zone itself is more visible than the disk.
    """
    h, w = gray.shape
    # Zone radius is typically 1.5× to 5× disk radius
    disk_min = max(5, int(min(w, h) * cfg.hough_min_r_ratio))
    zone_min = int(disk_min * 1.5)
    zone_max = int(min(w, h) * 0.40)

    enhanced = enhance(gray, cfg)
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)

    # Invert: dark zones become bright blobs
    inv = 255 - blurred
    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zone_min, zone_min))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Candidate] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * zone_min ** 2 * 0.4:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        if circularity < 0.4:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        # Return zone center — caller needs to find disk inside
        # Estimate disk radius from zone (assume zone is 3-5× disk)
        est_disk_r = r / 3.5
        if disk_min * 0.4 <= est_disk_r <= disk_min * 2.5:
            candidates.append(((float(cx), float(cy)), float(est_disk_r)))

    log.debug("dark-zone: %d candidates", len(candidates))
    return candidates


# ── Candidate merging (non-maximum suppression) ────────────────────────────────

def _nms_candidates(
    candidates: List[Candidate],
    min_dist_ratio: float = 1.2,
) -> List[Candidate]:
    """
    Remove duplicate candidates that are too close to each other.
    Keeps the one with the largest radius among overlapping detections.
    """
    if not candidates:
        return []

    sorted_c = sorted(candidates, key=lambda c: c[1], reverse=True)
    kept: List[Candidate] = []

    for (cx, cy), r in sorted_c:
        too_close = False
        for (kx, ky), kr in kept:
            dist = np.hypot(cx - kx, cy - ky)
            if dist < (r + kr) * min_dist_ratio * 0.5:
                too_close = True
                break
        if not too_close:
            kept.append(((cx, cy), r))

    return kept


# ── Main entry point ───────────────────────────────────────────────────────────

def find_disk_candidates(
    img: np.ndarray,
    cfg: AppConfig,
) -> List[Candidate]:
    """
    Adaptive multi-strategy disk detection.

    Selects detection strategies based on image characteristics,
    runs all applicable ones, merges results and validates each.
    """
    gray = to_gray_uint8(img)
    work, ox, oy = crop_petri(gray, cfg)
    stats = analyze_image(work)

    all_candidates: List[Candidate] = []

    # ── Primary strategy based on image type ──────────────────────────────────
    if stats.dark_background:
        # Look for bright paper disks
        blobs = _detect_bright_blobs(work, cfg, stats)
        all_candidates.extend(blobs)
    else:
        # Look for dark inhibition zone centers (classic approach)
        zones = _detect_dark_zones(work, cfg)
        all_candidates.extend(zones)

    # ── Secondary: Hough always runs ─────────────────────────────────────────
    hough = _detect_hough(work, cfg)
    all_candidates.extend(hough)

    # ── Translate back to full-image coordinates ──────────────────────────────
    shifted = [((cx + ox, cy + oy), r) for (cx, cy), r in all_candidates]

    # ── Deduplicate ───────────────────────────────────────────────────────────
    merged = _nms_candidates(shifted)

    # ── Adaptive validation ───────────────────────────────────────────────────
    image_stats = analyze_image(gray)
    validated = [
        c for c in merged
        if _is_valid_candidate(gray, c[0][0], c[0][1], c[1], image_stats)
    ]

    log.debug(
        "find_disk_candidates: raw=%d merged=%d validated=%d (dark_bg=%s)",
        len(all_candidates), len(merged), len(validated), stats.dark_background,
    )

    return validated
