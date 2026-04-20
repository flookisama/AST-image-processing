"""
Process antibiogram images: detect petri dish, antibiotic disks, and measure zone diameters.
Uses OpenCV only (no dependency on the C++ astimplib).
"""
from __future__ import annotations

import json
import logging
import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)

DISK_DIAMETER_MM = 6.0
_CONFIG_PATH = Path(__file__).parent / "data" / "calibration.json"

_DEFAULT_PARAMS = {
    "clahe_clip": 2.0,
    "clahe_grid": 8,
    "hough_dp": 1,
    "hough_param1": 80,
    "hough_param2": 35,
    "hough_min_r_ratio": 0.02,
    "hough_max_r_ratio": 0.15,
    "zone_threshold_ratio": 0.40,
    "dark_validation": True,
    "dark_percentile": 40,
}

_params: Optional[dict] = None


def load_params() -> dict:
    global _params
    if _params is not None:
        return _params
    if _CONFIG_PATH.is_file():
        try:
            with open(_CONFIG_PATH) as f:
                loaded = json.load(f)
            _params = {**_DEFAULT_PARAMS, **loaded}
            return _params
        except Exception as exc:
            log.warning("Could not load calibration config: %s — using defaults", exc)
    _params = dict(_DEFAULT_PARAMS)
    return _params


def save_params(p: dict) -> None:
    global _params
    _params = {**_DEFAULT_PARAMS, **p}
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(_params, f, indent=2)


def reset_params() -> None:
    global _params
    _params = None
    if _CONFIG_PATH.is_file():
        _CONFIG_PATH.unlink()


@dataclass
class DetectedDisk:
    center: Tuple[float, float]
    radius_px: float
    zone_diameter_mm: float
    confidence: float
    label: str
    zone_radius_px: float = field(default=0.0)


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("Empty or invalid image")
    out = img
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]
    if out.ndim == 3:
        out = cv2.cvtColor(out.astype(np.uint8) if out.dtype != np.uint8 else out, cv2.COLOR_RGB2GRAY)
    if out.dtype != np.uint8:
        out = (out * 255).astype(np.uint8) if out.max() <= 1.0 else out.astype(np.uint8)
    return out


def _enhance(gray: np.ndarray) -> np.ndarray:
    p = load_params()
    grid = int(p["clahe_grid"])
    clahe = cv2.createCLAHE(clipLimit=float(p["clahe_clip"]), tileGridSize=(grid, grid))
    return clahe.apply(gray)


def _crop_petri(gray: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Detect and crop the petri dish region. Returns (cropped_gray, offset_x, offset_y)."""
    h, w = gray.shape
    enhanced = _enhance(gray)
    blurred_large = cv2.GaussianBlur(enhanced, (21, 21), 3)

    # Try to find the petri dish as a large circle
    circles = cv2.HoughCircles(
        blurred_large,
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

    # Fallback: contour-based crop on Otsu threshold
    # On dark backgrounds, invert so the bright dish becomes the largest contour
    median_val = float(np.median(blurred_large))
    th_input = 255 - blurred_large if median_val < 80 else blurred_large
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


def _is_dark_circle(gray: np.ndarray, cx: float, cy: float, r: float) -> bool:
    """Antibiotic disks are dark pellets — reject bright circles (reflections, bubbles)."""
    p = load_params()
    if not p.get("dark_validation", True):
        return True
    h, w = gray.shape
    yg, xg = np.ogrid[:h, :w]
    dist = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    inner_mask = dist < r * 0.65
    outer_mask = (dist >= r * 1.3) & (dist < r * 2.5)
    if not np.any(inner_mask) or not np.any(outer_mask):
        return True
    inner_mean = float(np.mean(gray[inner_mask]))
    outer_perc = float(np.percentile(gray[outer_mask], int(p.get("dark_percentile", 40))))
    return inner_mean < outer_perc


def find_disks(img: np.ndarray, use_ml: bool = True) -> List[Tuple[Tuple[float, float], float]]:
    """Detect antibiotic disks (dark circles). Returns list of (center_xy, radius_px).

    When use_ml=True and a trained ML classifier exists, applies it as a
    post-processing filter to remove false positives.
    """
    p = load_params()
    gray = _to_gray_uint8(img)
    work, ox, oy = _crop_petri(gray)
    h, w = work.shape

    min_r = max(5, int(min(w, h) * float(p["hough_min_r_ratio"])))
    max_r = max(min_r + 20, int(min(w, h) * float(p["hough_max_r_ratio"])))

    enhanced = _enhance(work)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

    # Try multiple sensitivity levels; keep the one that finds the most circles
    best: Optional[np.ndarray] = None
    base_p2 = float(p["hough_param2"])
    for p2 in [base_p2, max(20.0, base_p2 - 10), min(55.0, base_p2 + 10)]:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=float(p["hough_dp"]),
            minDist=min_r * 2,
            param1=float(p["hough_param1"]),
            param2=p2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            if best is None or len(circles[0]) > len(best[0]):
                best = circles
            if len(circles[0]) >= 4:
                break

    out: List[Tuple[Tuple[float, float], float]] = []
    if best is not None:
        best = np.around(best).astype(int)
        for c in best[0]:
            cx, cy, r = float(c[0]) + ox, float(c[1]) + oy, float(c[2])
            if _is_dark_circle(gray, cx, cy, r):
                out.append(((cx, cy), r))

    # ML post-filtering (removes false positives from reflections, bubbles, etc.)
    if use_ml and out:
        try:
            from ml_detector import load_classifier
            clf = load_classifier()
            if clf is not None:
                filtered = clf.filter(img, out)
                log.debug("ML filter: %d → %d candidates", len(out), len(filtered))
                out = filtered
        except Exception as exc:
            log.warning("ML filter failed (%s) — keeping Hough results", exc)

    return out


def _radial_profile(img: np.ndarray, cx: float, cy: float, max_r: int) -> np.ndarray:
    """Fast vectorized mean intensity vs radius."""
    h, w = img.shape
    yg, xg = np.ogrid[:h, :w]
    dist = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    r_int = np.clip(dist.astype(int), 0, max_r).ravel()
    vals = img.ravel().astype(np.float64)
    profile = np.bincount(r_int, weights=vals, minlength=max_r + 1)
    count = np.maximum(np.bincount(r_int, minlength=max_r + 1), 1)
    return profile / count


def _measure_zone(
    gray: np.ndarray, cx: float, cy: float, disk_r: float, px_per_mm: float
) -> Tuple[float, float]:
    """
    Measure inhibition zone diameter using gradient-based edge detection on radial profile.
    Returns (diameter_mm, confidence 0-1).
    """
    if px_per_mm <= 0:
        return DISK_DIAMETER_MM, 0.0
    p = load_params()
    h, w = gray.shape
    max_r = min(int(28 * px_per_mm), int(0.45 * min(h, w)))
    max_r = max(max_r, int(disk_r) + 5)

    profile = _radial_profile(gray, cx, cy, max_r)
    start = int(disk_r) + 2
    if start >= len(profile) - 4:
        return DISK_DIAMETER_MM, 0.0

    segment = profile[start:]
    if len(segment) < 6:
        return DISK_DIAMETER_MM, 0.0

    # Smooth profile
    kernel_size = min(9, len(segment) // 3 * 2 + 1) | 1  # ensure odd
    smooth = cv2.GaussianBlur(
        segment.astype(np.float32).reshape(-1, 1), (kernel_size, 1), 2.0
    ).ravel()

    low, high = float(np.min(smooth)), float(np.max(smooth))
    if high <= low + 1:
        return DISK_DIAMETER_MM, 0.0

    thresh = low + float(p["zone_threshold_ratio"]) * (high - low)
    grad = np.gradient(smooth)

    # Zone edge: first point above threshold where intensity is increasing
    edge_idx = start
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

    radius_mm = edge_idx / px_per_mm
    diameter_mm = float(np.clip(2 * radius_mm, DISK_DIAMETER_MM, 50.0))
    contrast = (high - low) / 255.0
    confidence = float(np.clip(contrast * 1.5, 0.0, 1.0))
    return diameter_mm, confidence


def _px_per_mm(radius_px: float) -> float:
    return (2.0 * radius_px) / DISK_DIAMETER_MM


def process_antibiogram(
    img: np.ndarray,
    disk_labels: Optional[List[str]] = None,
    use_ml: bool = True,
) -> Tuple[List[DetectedDisk], float]:
    """Full pipeline: find disks, measure zones. Returns (disks, px_per_mm)."""
    if img is None or img.size == 0:
        return [], 0.0
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    gray = _to_gray_uint8(img)
    circles = find_disks(img, use_ml=use_ml)
    if not circles:
        return [], 0.0

    circles.sort(key=lambda c: (round(c[0][1] / 30), c[0][0]))
    median_r = float(np.median([r for _, r in circles]))
    if median_r <= 0:
        log.warning("median disk radius is 0 — cannot compute scale")
        return [], 0.0
    scale = _px_per_mm(median_r)

    results: List[DetectedDisk] = []
    for i, (center, r_px) in enumerate(circles):
        zone_mm, conf = _measure_zone(gray, center[0], center[1], r_px, scale)
        zone_r_px = (zone_mm / 2.0) * scale
        lbl = (disk_labels[i] if disk_labels and i < len(disk_labels) else None) or f"Disk {i + 1}"
        results.append(
            DetectedDisk(
                center=center,
                radius_px=r_px,
                zone_diameter_mm=zone_mm,
                confidence=conf,
                label=lbl,
                zone_radius_px=zone_r_px,
            )
        )
    return results, scale


def process_with_params(
    img: np.ndarray,
    params_override: dict,
    disk_labels: Optional[List[str]] = None,
) -> Tuple[List[DetectedDisk], float]:
    """
    Run the full pipeline with a temporary parameter override.
    Does not persist any changes to the on-disk calibration file.
    """
    global _params
    old = _params
    _params = {**_DEFAULT_PARAMS, **(old or {}), **params_override}
    try:
        return process_antibiogram(img, disk_labels)
    finally:
        _params = old


def draw_results(img: np.ndarray, disks: List[DetectedDisk], px_per_mm: float) -> np.ndarray:
    """Draw disk and zone circles with labels on the image."""
    out = np.asarray(img).copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    for d in disks:
        cx, cy = int(d.center[0]), int(d.center[1])
        r_disk = int(d.radius_px)
        r_zone = int((d.zone_diameter_mm / 2.0) * px_per_mm)

        # Confidence-based color for zone circle
        if d.confidence > 0.65:
            zone_color = (0, 200, 0)
        elif d.confidence > 0.35:
            zone_color = (0, 165, 255)
        else:
            zone_color = (0, 0, 220)

        cv2.circle(out, (cx, cy), r_disk, (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), r_zone, zone_color, 2)
        label_text = f"{d.label}  {d.zone_diameter_mm:.1f}mm"
        cv2.putText(
            out,
            label_text,
            (cx - 45, cy - r_disk - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Convert BGR back to RGB for Streamlit
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
