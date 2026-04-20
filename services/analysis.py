"""
Analysis service — full antibiogram pipeline.

Orchestrates: detection → ML filter → zone measurement → results.
Accepts explicit AppConfig; never reads or writes global state.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from core.config import AppConfig
from core.detection import find_disk_candidates, to_gray_uint8
from core.measurement import measure_zone, px_per_mm
from core.models import AnalysisResult, DetectedDisk

log = logging.getLogger(__name__)


def run_analysis(
    img: np.ndarray,
    cfg: Optional[AppConfig] = None,
    use_ml: bool = True,
    disk_labels: Optional[List[str]] = None,
) -> AnalysisResult:
    """
    Full antibiogram analysis pipeline.

    Args:
        img:         RGB image as numpy array.
        cfg:         AppConfig — defaults to AppConfig.load() if None.
        use_ml:      Whether to apply HOG+SVM post-filter.
        disk_labels: Optional pre-assigned labels for each disk.

    Returns:
        AnalysisResult with detected disks and scale.
    """
    if cfg is None:
        cfg = AppConfig.load()

    if img is None or img.size == 0:
        return AnalysisResult.empty()

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    shape = (img.shape[0], img.shape[1])

    # ── Step 1: Hough + dark-circle detection ─────────────────────────────────
    candidates = find_disk_candidates(img, cfg)
    log.debug("Hough found %d candidates", len(candidates))

    # ── Step 2: ML post-filter ─────────────────────────────────────────────────
    if use_ml and candidates:
        candidates = _apply_ml_filter(img, candidates)

    if not candidates:
        return AnalysisResult.empty(shape)

    # ── Step 3: Compute scale from median disk radius ──────────────────────────
    median_r = float(np.median([r for _, r in candidates]))
    if median_r <= 0:
        log.warning("Median disk radius is 0 — cannot compute scale")
        return AnalysisResult.empty(shape)

    scale = px_per_mm(median_r)

    # ── Step 4: Sort top-left → bottom-right ──────────────────────────────────
    candidates.sort(key=lambda c: (round(c[0][1] / 30), c[0][0]))

    # ── Step 5: Measure zones ─────────────────────────────────────────────────
    gray = to_gray_uint8(img)
    disks: List[DetectedDisk] = []
    for i, (center, r_px) in enumerate(candidates):
        zone_mm, conf = measure_zone(gray, center[0], center[1], r_px, scale, cfg)
        zone_r_px = (zone_mm / 2.0) * scale
        label = (
            (disk_labels[i] if disk_labels and i < len(disk_labels) else None)
            or f"Disk {i + 1}"
        )
        disks.append(
            DetectedDisk(
                center=center,
                radius_px=r_px,
                zone_diameter_mm=zone_mm,
                confidence=conf,
                label=label,
                zone_radius_px=zone_r_px,
            )
        )

    return AnalysisResult(disks=disks, px_per_mm=scale, image_shape=shape, used_ml=use_ml)


def _apply_ml_filter(img: np.ndarray, candidates: list) -> list:
    """Apply HOG+SVM classifier. Returns original candidates on failure."""
    try:
        from ml_detector import load_classifier
        clf = load_classifier()
        if clf is not None:
            filtered = clf.filter(img, candidates)
            log.debug("ML filter: %d → %d candidates", len(candidates), len(filtered))
            return filtered
    except Exception as exc:
        log.warning("ML filter failed (%s) — using Hough results only", exc)
    return candidates
