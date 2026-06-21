"""
Analysis service — full antibiogram pipeline.

Orchestrates: detection → ML filter → scale estimation → zone measurement.
Stateless: accepts explicit AppConfig, no global state.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from core.config import AppConfig
from core.detection import find_disk_candidates, to_gray_uint8, analyze_image
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

    Steps:
      1. Adaptive disk candidate detection (multi-strategy)
      2. HOG+SVM post-filter (optional)
      3. Robust scale estimation (outlier-resistant median)
      4. Multi-angle zone measurement per disk
    """
    if cfg is None:
        cfg = AppConfig.load()

    if img is None or img.size == 0:
        return AnalysisResult.empty()

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    shape = (img.shape[0], img.shape[1])

    # ── Step 1: Detection ─────────────────────────────────────────────────────
    candidates = find_disk_candidates(img, cfg)
    log.info("Detection: %d disk candidates found", len(candidates))

    if not candidates:
        return AnalysisResult.empty(shape)

    # ── Step 2: ML post-filter ────────────────────────────────────────────────
    if use_ml and candidates:
        n_before = len(candidates)
        candidates = _apply_ml_filter(img, candidates)
        log.info("ML filter: %d → %d candidates", n_before, len(candidates))

    if not candidates:
        return AnalysisResult.empty(shape)

    # ── Step 3: Robust scale estimation ──────────────────────────────────────
    scale = _estimate_scale(candidates)
    if scale <= 0:
        log.warning("Could not estimate scale (all disk radii zero)")
        return AnalysisResult.empty(shape)

    # ── Step 4: Sort spatially (top-left → bottom-right) ─────────────────────
    candidates.sort(key=lambda c: (round(c[0][1] / 30), c[0][0]))

    # ── Step 5: Zone measurement ──────────────────────────────────────────────
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

    log.info(
        "Analysis complete: %d disks, scale=%.1f px/mm, zones=%s mm",
        len(disks),
        scale,
        [f"{d.zone_diameter_mm:.1f}" for d in disks],
    )

    return AnalysisResult(disks=disks, px_per_mm=scale, image_shape=shape, used_ml=use_ml)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _estimate_scale(
    candidates: list,
    mad_factor: float = 2.5,
) -> float:
    """
    Estimate px/mm from disk radii using outlier-resistant median + MAD filter.
    Uses the known paper disk diameter (6mm) as reference.
    """
    radii = np.array([r for _, r in candidates], dtype=float)
    if len(radii) == 0:
        return 0.0

    if len(radii) == 1:
        return px_per_mm(float(radii[0]))

    median_r = float(np.median(radii))
    mad = float(np.median(np.abs(radii - median_r)))
    cutoff = max(mad_factor * mad, 2.0)   # at least 2px tolerance
    valid = radii[np.abs(radii - median_r) <= cutoff]

    if len(valid) == 0:
        valid = radii

    return px_per_mm(float(np.mean(valid)))


def _apply_ml_filter(img: np.ndarray, candidates: list) -> list:
    """Apply HOG+SVM classifier. Returns original candidates on failure."""
    try:
        from ml_detector import load_classifier
        clf = load_classifier()
        if clf is not None:
            return clf.filter(img, candidates)
    except Exception as exc:
        log.warning("ML filter failed (%s) — using detection results only", exc)
    return candidates
