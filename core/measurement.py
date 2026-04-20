"""
Zone measurement — multi-angle with outlier rejection.

Algorithm:
  1. Cast N radial rays from disk center
  2. Find the zone edge along each ray (first intensity threshold crossing)
  3. Reject outlier radii using Median Absolute Deviation (handles overlapping zones)
  4. Fit the final diameter from the clean sample mean
  5. Confidence is derived from angular coverage + radial consistency

This replaces the single radial profile approach, which broke on:
  - Overlapping zones (only one side visible)
  - Non-circular zones (uneven agar, tilted disk)
  - Asymmetric lighting (one side brighter)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import AppConfig
from .models import DISK_DIAMETER_MM

log = logging.getLogger(__name__)

# ── Scale conversion ───────────────────────────────────────────────────────────

def px_per_mm(disk_radius_px: float) -> float:
    if disk_radius_px <= 0:
        return 0.0
    return (2.0 * disk_radius_px) / DISK_DIAMETER_MM


# ── Background normalization ───────────────────────────────────────────────────

def normalize_background(gray: np.ndarray, disk_r: float) -> np.ndarray:
    """
    Subtract a morphological estimate of the background (agar surface).
    Removes vignetting and uneven illumination so the threshold is consistent.
    """
    # Use a large structuring element (much larger than the zone)
    ksize = max(int(disk_r * 8) | 1, 31)   # odd, at least 31px
    ksize = min(ksize, min(gray.shape) - 2)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    normalized = cv2.subtract(background, gray)   # bright zones → brighter
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


# ── Single-ray edge finder ─────────────────────────────────────────────────────

def _ray_edge(
    gray: np.ndarray,
    cx: float,
    cy: float,
    angle: float,
    start_r: int,
    max_r: int,
    threshold_ratio: float,
    smooth_k: int = 7,
) -> Optional[int]:
    """
    Sample intensity along one radial ray and return the edge radius in pixels.
    Returns None if no clear edge is found.
    """
    h, w = gray.shape
    radii = np.arange(start_r, max_r)
    if len(radii) < smooth_k + 2:
        return None

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    xs = np.clip((cx + radii * cos_a).astype(int), 0, w - 1)
    ys = np.clip((cy + radii * sin_a).astype(int), 0, h - 1)
    profile = gray[ys, xs].astype(np.float32)

    # Smooth to suppress bacteria-colony noise
    profile_s = cv2.GaussianBlur(
        profile.reshape(-1, 1), (smooth_k, 1), 0
    ).ravel()

    low = float(np.min(profile_s))
    high = float(np.max(profile_s))
    if high - low < 8.0:       # not enough contrast along this ray
        return None

    thresh = low + threshold_ratio * (high - low)

    # Find first crossing from clear zone into bacteria (low → high)
    for i in range(1, len(profile_s)):
        if profile_s[i] >= thresh:
            return int(radii[i])

    return None


# ── Multi-angle zone measurement ──────────────────────────────────────────────

def measure_zone(
    gray: np.ndarray,
    cx: float,
    cy: float,
    disk_r: float,
    scale: float,
    cfg: AppConfig,
    n_angles: int = 36,
) -> Tuple[float, float]:
    """
    Measure the inhibition zone diameter by multi-angle radial edge detection.

    Samples n_angles rays uniformly, finds the zone edge along each,
    rejects outliers (from overlapping zones or reflections) using MAD,
    and returns (diameter_mm, confidence).
    """
    if scale <= 0:
        return DISK_DIAMETER_MM, 0.0

    h, w = gray.shape
    start_r = max(int(disk_r) + 2, 4)
    max_r = int(min(28 * scale, 0.45 * min(h, w)))
    max_r = max(max_r, start_r + 10)

    # Normalize background for more consistent thresholding
    try:
        work = normalize_background(gray, disk_r)
    except Exception:
        work = gray

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    edge_px: List[int] = []

    for angle in angles:
        edge = _ray_edge(
            work, cx, cy, angle,
            start_r, max_r,
            threshold_ratio=cfg.zone_threshold_ratio,
        )
        if edge is not None:
            edge_px.append(edge)

    # ── Need at least 25% angular coverage ────────────────────────────────────
    if len(edge_px) < max(4, n_angles // 4):
        log.debug("measure_zone: too few rays (%d/%d) — fallback", len(edge_px), n_angles)
        return _fallback_measure(work, cx, cy, disk_r, scale, cfg)

    radii = np.array(edge_px, dtype=float)

    # ── Outlier rejection via MAD ──────────────────────────────────────────────
    median_r = float(np.median(radii))
    mad = float(np.median(np.abs(radii - median_r)))
    # Allow up to 3.5 MADs — generous to handle slight asymmetry
    cutoff = max(3.5 * mad, max(5.0, disk_r * 0.2))   # at least 5px or 20% of disk
    valid = radii[np.abs(radii - median_r) <= cutoff]

    if len(valid) < 3:
        valid = radii    # last resort: use everything

    mean_r_px = float(np.mean(valid))
    diameter_mm = float(np.clip(2.0 * mean_r_px / scale, DISK_DIAMETER_MM, 50.0))

    # ── Confidence ────────────────────────────────────────────────────────────
    # Coverage: what fraction of rays had a clear edge
    coverage = len(edge_px) / n_angles
    # Consistency: how tight the valid radii are (CV)
    cv = float(np.std(valid)) / (mean_r_px + 1e-6)
    consistency = float(np.clip(1.0 - cv * 4, 0.0, 1.0))
    # Outlier fraction: how many rays were rejected
    inlier_frac = len(valid) / len(edge_px) if edge_px else 0.0

    confidence = float(np.clip(coverage * consistency * inlier_frac * 1.5, 0.0, 1.0))

    log.debug(
        "measure_zone (%.0f,%.0f): rays=%d/%d valid=%d r=%.1fpx → %.1fmm conf=%.2f",
        cx, cy, len(edge_px), n_angles, len(valid), mean_r_px, diameter_mm, confidence,
    )

    return diameter_mm, confidence


# ── Fallback: single radial profile ───────────────────────────────────────────

def _fallback_measure(
    gray: np.ndarray,
    cx: float,
    cy: float,
    disk_r: float,
    scale: float,
    cfg: AppConfig,
) -> Tuple[float, float]:
    """
    Classic single radial profile — used only when multi-angle fails.
    Returns (diameter_mm, confidence).
    """
    if scale <= 0:
        return DISK_DIAMETER_MM, 0.0

    h, w = gray.shape
    max_r = int(min(28 * scale, 0.45 * min(h, w)))
    max_r = max(max_r, int(disk_r) + 5)

    # Build radial profile (mean intensity per radius)
    yg, xg = np.ogrid[:h, :w]
    dist = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    r_int = np.clip(dist.astype(int), 0, max_r).ravel()
    vals = gray.ravel().astype(np.float64)
    profile = np.bincount(r_int, weights=vals, minlength=max_r + 1)
    count = np.maximum(np.bincount(r_int, minlength=max_r + 1), 1)
    profile = profile / count

    start = int(disk_r) + 2
    if start >= len(profile) - 4:
        return DISK_DIAMETER_MM, 0.0

    segment = profile[start:]
    if len(segment) < 6:
        return DISK_DIAMETER_MM, 0.0

    kernel_size = max(3, (min(9, len(segment) // 3 * 2 + 1) | 1))
    smooth = cv2.GaussianBlur(
        segment.astype(np.float32).reshape(-1, 1), (kernel_size, 1), 2.0
    ).ravel()

    low, high = float(np.min(smooth)), float(np.max(smooth))
    if high <= low + 1:
        return DISK_DIAMETER_MM, 0.0

    thresh = low + cfg.zone_threshold_ratio * (high - low)
    grad = np.gradient(smooth)

    edge_idx = start
    for i in range(2, len(smooth) - 1):
        if smooth[i] >= thresh and grad[i] > 0:
            edge_idx = start + i
            break
    else:
        for i in range(1, len(smooth)):
            if smooth[i] >= thresh:
                edge_idx = start + i
                break
        else:
            edge_idx = start + len(smooth) - 1

    diameter_mm = float(np.clip(2 * edge_idx / scale, DISK_DIAMETER_MM, 50.0))
    confidence = float(np.clip((high - low) / 255.0 * 1.5, 0.0, 1.0)) * 0.6  # penalize fallback
    return diameter_mm, confidence
