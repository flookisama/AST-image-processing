"""
Zone measurement — pure functions, explicit config.
"""
from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from .config import AppConfig
from .models import DISK_DIAMETER_MM

log = logging.getLogger(__name__)


def px_per_mm(disk_radius_px: float) -> float:
    """Convert disk radius in pixels to scale px/mm using known disk diameter."""
    if disk_radius_px <= 0:
        return 0.0
    return (2.0 * disk_radius_px) / DISK_DIAMETER_MM


def _radial_profile(gray: np.ndarray, cx: float, cy: float, max_r: int) -> np.ndarray:
    h, w = gray.shape
    yg, xg = np.ogrid[:h, :w]
    dist = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    r_int = np.clip(dist.astype(int), 0, max_r).ravel()
    vals = gray.ravel().astype(np.float64)
    profile = np.bincount(r_int, weights=vals, minlength=max_r + 1)
    count = np.maximum(np.bincount(r_int, minlength=max_r + 1), 1)
    return profile / count


def measure_zone(
    gray: np.ndarray,
    cx: float,
    cy: float,
    disk_r: float,
    scale: float,
    cfg: AppConfig,
) -> Tuple[float, float]:
    """
    Measure inhibition zone diameter from a radial intensity profile.

    Returns (diameter_mm, confidence 0-1).
    Falls back to DISK_DIAMETER_MM with 0 confidence on failure.
    """
    if scale <= 0:
        return DISK_DIAMETER_MM, 0.0

    h, w = gray.shape
    max_r = min(int(28 * scale), int(0.45 * min(h, w)))
    max_r = max(max_r, int(disk_r) + 5)

    profile = _radial_profile(gray, cx, cy, max_r)
    start = int(disk_r) + 2

    if start >= len(profile) - 4:
        return DISK_DIAMETER_MM, 0.0

    segment = profile[start:]
    if len(segment) < 6:
        return DISK_DIAMETER_MM, 0.0

    kernel_size = (min(9, len(segment) // 3 * 2 + 1) | 1)
    smooth = cv2.GaussianBlur(
        segment.astype(np.float32).reshape(-1, 1), (kernel_size, 1), 2.0
    ).ravel()

    low, high = float(np.min(smooth)), float(np.max(smooth))
    if high <= low + 1:
        return DISK_DIAMETER_MM, 0.0

    thresh = low + cfg.zone_threshold_ratio * (high - low)
    grad = np.gradient(smooth)

    edge_idx = start
    # Primary: first point above threshold where intensity is increasing
    for i in range(2, len(smooth) - 1):
        if smooth[i] >= thresh and grad[i] > 0:
            edge_idx = start + i
            break
    else:
        # Fallback: first threshold crossing
        for i in range(1, len(smooth)):
            if smooth[i] >= thresh:
                edge_idx = start + i
                break
        else:
            edge_idx = start + len(smooth) - 1

    radius_mm = edge_idx / scale
    diameter_mm = float(np.clip(2 * radius_mm, DISK_DIAMETER_MM, 50.0))
    contrast = (high - low) / 255.0
    confidence = float(np.clip(contrast * 1.5, 0.0, 1.0))
    return diameter_mm, confidence
