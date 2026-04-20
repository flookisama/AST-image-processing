"""
Disk detection pipeline.

Functions take an explicit AppConfig rather than reading global state.
All functions are pure (no side effects, no I/O).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import AppConfig

log = logging.getLogger(__name__)

# type alias
Candidate = Tuple[Tuple[float, float], float]   # ((cx, cy), radius_px)


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


# ── Petri dish crop ────────────────────────────────────────────────────────────

def crop_petri(gray: np.ndarray, cfg: AppConfig) -> Tuple[np.ndarray, int, int]:
    """
    Detect and return (cropped_gray, offset_x, offset_y).
    Falls back to the whole image when no dish is found.
    """
    h, w = gray.shape
    enhanced = enhance(gray, cfg)
    blurred = cv2.GaussianBlur(enhanced, (21, 21), 3)

    # ── Strategy 1: Hough large circle ──────────────────────────────────────
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min(w, h) // 2,
        param1=50,
        param2=30,
        minRadius=min(w, h) // 5,
        maxRadius=min(w, h) // 2,
    )
    if circles is not None:
        c = circles[0, 0]
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        margin = int(r * 0.06)
        x1 = max(0, cx - r - margin)
        y1 = max(0, cy - r - margin)
        x2 = min(w, cx + r + margin)
        y2 = min(h, cy + r + margin)
        crop = gray[y1:y2, x1:x2]
        if crop.size > 100 * 100:
            return crop, x1, y1

    # ── Strategy 2: contour on Otsu threshold ───────────────────────────────
    # Invert when background is dark so the bright dish becomes the largest contour
    median_val = float(np.median(blurred))
    th_input = 255 - blurred if median_val < 80 else blurred
    _, th = cv2.threshold(th_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        margin = int(0.02 * max(w, h))
        x = max(0, x - margin)
        y = max(0, y - margin)
        bw = min(w - x, bw + 2 * margin)
        bh = min(h - y, bh + 2 * margin)
        crop = gray[y : y + bh, x : x + bw]
        if crop.size > 100 * 100:
            return crop, x, y

    return gray, 0, 0


# ── Dark-circle validation ─────────────────────────────────────────────────────

def is_dark_circle(
    gray: np.ndarray, cx: float, cy: float, r: float, cfg: AppConfig
) -> bool:
    """
    Return True if the circle interior is darker than its neighbourhood.
    When dark_validation is disabled, always returns True (accept all).
    """
    if not cfg.dark_validation:
        return True
    h, w = gray.shape
    yg, xg = np.ogrid[:h, :w]
    dist = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    inner_mask = dist < r * 0.65
    outer_mask = (dist >= r * 1.3) & (dist < r * 2.5)
    if not np.any(inner_mask) or not np.any(outer_mask):
        return True
    inner_mean = float(np.mean(gray[inner_mask]))
    outer_perc = float(np.percentile(gray[outer_mask], cfg.dark_percentile))
    return inner_mean < outer_perc


# ── Disk detection ─────────────────────────────────────────────────────────────

def find_disk_candidates(
    img: np.ndarray,
    cfg: AppConfig,
) -> List[Candidate]:
    """
    Detect disk candidates with Hough + dark-circle filter.
    Returns a list of ((cx, cy), radius_px).
    """
    gray = to_gray_uint8(img)
    work, ox, oy = crop_petri(gray, cfg)
    h, w = work.shape

    min_r = max(5, int(min(w, h) * cfg.hough_min_r_ratio))
    max_r = max(min_r + 20, int(min(w, h) * cfg.hough_max_r_ratio))

    enhanced = enhance(work, cfg)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

    # Try three sensitivity levels; keep whichever finds the most circles
    best: Optional[np.ndarray] = None
    base_p2 = float(cfg.hough_param2)
    for p2 in [base_p2, max(20.0, base_p2 - 10), min(55.0, base_p2 + 10)]:
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
            cx = float(c[0]) + ox
            cy = float(c[1]) + oy
            r = float(c[2])
            if is_dark_circle(gray, cx, cy, r, cfg):
                candidates.append(((cx, cy), r))

    return candidates
