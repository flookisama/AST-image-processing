"""
Process antibiogram images: detect petri dish, antibiotic disks, and measure zone diameters.
Uses OpenCV only (no dependency on the C++ astimplib).
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Standard antibiotic disk diameter in mm (used for scale)
DISK_DIAMETER_MM = 6.0


@dataclass
class DetectedDisk:
    center: Tuple[float, float]  # (x, y) in image
    radius_px: float
    zone_diameter_mm: float
    confidence: float  # 0-1
    label: str  # e.g. "AMX25" or "Disk 1"


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("Empty or invalid image")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def _crop_petri_region(gray: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Optional: focus on largest circular/elliptical region (petri). Returns (cropped, offset_x, offset_y)."""
    h, w = gray.shape
    # Use Otsu to get rough foreground (plate often lighter than background)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray, 0, 0
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    # Add margin
    margin = int(0.02 * max(w, h))
    x = max(0, x - margin)
    y = max(0, y - margin)
    bw = min(w - x, bw + 2 * margin)
    bh = min(h - y, bh + 2 * margin)
    crop = gray[y : y + bh, x : x + bw]
    if crop.size < 100 * 100:
        return gray, 0, 0
    return crop, x, y


def find_disks(img: np.ndarray) -> List[Tuple[Tuple[float, float], float]]:
    """
    Detect antibiotic disks (dark circles). Returns list of (center_xy, radius_px).
    Uses Hough Circle Transform and filters by size.
    """
    gray = _ensure_uint8(img)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(gray, dtype=np.uint8)

    # Optional crop to petri
    work, ox, oy = _crop_petri_region(gray)
    h, w = work.shape
    min_r = int(min(w, h) * 0.02)
    max_r = int(min(w, h) * 0.15)
    min_r = max(5, min_r)
    max_r = max(max_r, min_r + 20)

    # Blur to reduce noise for Hough
    blurred = cv2.GaussianBlur(work, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_r * 2,
        param1=80,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )
    out: List[Tuple[Tuple[float, float], float]] = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cx, cy, r = float(c[0]) + ox, float(c[1]) + oy, float(c[2])
            out.append(((cx, cy), r))
    return out


def _px_per_mm_from_disk_radius(radius_px: float) -> float:
    """Assume disk diameter = 6 mm."""
    return (2 * radius_px) / DISK_DIAMETER_MM


def _radial_profile(img: np.ndarray, cx: float, cy: float, max_r_px: int) -> np.ndarray:
    """Mean intensity vs radius (in pixels)."""
    h, w = img.shape
    yg, xg = np.ogrid[:h, :w]
    r = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    r_int = np.clip(r.astype(int), 0, max_r_px)
    profile = np.zeros(max_r_px + 1)
    count = np.zeros(max_r_px + 1)
    for ri in range(max_r_px + 1):
        mask = r_int == ri
        if np.any(mask):
            profile[ri] = np.mean(img[mask])
            count[ri] = np.sum(mask)
    return profile


def _measure_zone_diameter(
    gray: np.ndarray, cx: float, cy: float, disk_radius_px: float, px_per_mm: float
) -> Tuple[float, float]:
    """
    Measure inhibition zone by radial profile: find edge where intensity rises (bacteria).
    Returns (diameter_mm, confidence).
    """
    h, w = gray.shape
    max_r = min(int(25 * px_per_mm), int(0.45 * min(h, w)))
    max_r = max(max_r, int(disk_radius_px) + 5)
    profile = _radial_profile(gray, cx, cy, max_r)
    # Skip disk region
    start = int(disk_radius_px) + 2
    if start >= len(profile) - 2:
        return DISK_DIAMETER_MM, 0.0
    segment = profile[start:]
    if len(segment) < 3:
        return DISK_DIAMETER_MM, 0.0
    # Zone = clear (low intensity); bacteria = higher. Find radius where intensity rises.
    smooth = cv2.GaussianBlur(segment.astype(np.float32).reshape(-1, 1), (5, 1), 1).ravel()
    low = np.min(smooth)
    high = np.max(smooth)
    if high <= low:
        return DISK_DIAMETER_MM, 0.0
    # Threshold at mid-level; zone edge ≈ first point above threshold going outward
    thresh = low + 0.4 * (high - low)
    edge_idx = start
    for i in range(1, len(smooth)):
        if smooth[i] >= thresh and smooth[i] > smooth[i - 1]:
            edge_idx = start + i
            break
    else:
        edge_idx = start + len(smooth) - 1
    radius_mm = edge_idx / px_per_mm
    diameter_mm = 2 * radius_mm
    diameter_mm = max(DISK_DIAMETER_MM, min(50.0, diameter_mm))
    # Confidence from contrast
    contrast = (high - low) / 255.0 if high > low else 0
    confidence = min(1.0, float(contrast) * 1.5)
    return diameter_mm, confidence


def process_antibiogram(
    img: np.ndarray,
    disk_labels: Optional[List[str]] = None,
) -> Tuple[List[DetectedDisk], float]:
    """
    Full pipeline: find disks, measure zones, optionally label.
    Returns (list of DetectedDisk, px_per_mm used).
    """
    if img is None or img.size == 0:
        return [], 0.0
    if isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    gray = _ensure_uint8(img)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)

    circles = find_disks(gray)
    if not circles:
        return [], 0.0

    # Sort by position (top-left to bottom-right) for stable ordering
    circles.sort(key=lambda c: (c[0][1], c[0][0]))
    # Use median disk radius for scale
    radii = [r for _, r in circles]
    median_r = float(np.median(radii))
    px_per_mm = _px_per_mm_from_disk_radius(median_r)

    results: List[DetectedDisk] = []
    for i, (center, radius_px) in enumerate(circles):
        zone_mm, conf = _measure_zone_diameter(gray, center[0], center[1], radius_px, px_per_mm)
        label = (disk_labels[i] if disk_labels and i < len(disk_labels) else None) or f"Disk {i + 1}"
        results.append(
            DetectedDisk(
                center=center,
                radius_px=radius_px,
                zone_diameter_mm=zone_mm,
                confidence=conf,
                label=label,
            )
        )
    return results, px_per_mm


def draw_results(img: np.ndarray, disks: List[DetectedDisk], px_per_mm: float) -> np.ndarray:
    """Draw disk circles and zone circles on a copy of the image."""
    out = np.asarray(img).copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for d in disks:
        cx, cy = int(d.center[0]), int(d.center[1])
        r_disk = int(d.radius_px)
        r_zone_px = (d.zone_diameter_mm / 2) * px_per_mm
        cv2.circle(out, (cx, cy), r_disk, (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), int(r_zone_px), (0, 255, 0), 2)
        cv2.putText(
            out, f"{d.label} {d.zone_diameter_mm:.1f}mm",
            (cx - 40, cy - r_disk - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
    return out
