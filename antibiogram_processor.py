"""
antibiogram_processor.py — backward-compatibility shim.

All logic lives in core/ and services/.
This module re-exports the public API so existing imports keep working.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Re-export core types so callers keep working ───────────────────────────────
from core.models import DetectedDisk, DISK_DIAMETER_MM          # noqa: F401
from core.drawing import draw_results                            # noqa: F401
from core.detection import to_gray_uint8 as _to_gray_uint8      # noqa: F401
from core.config import AppConfig

# ── Config helpers (dict-based API kept for Calibration page) ─────────────────

_CONFIG_PATH = Path(__file__).parent / "data" / "calibration.json"

_DEFAULT_PARAMS = {
    "clahe_clip": 2.0,
    "clahe_grid": 8,
    "hough_dp": 1.0,
    "hough_param1": 80,
    "hough_param2": 35,
    "hough_min_r_ratio": 0.02,
    "hough_max_r_ratio": 0.15,
    "zone_threshold_ratio": 0.40,
    "dark_validation": True,
    "dark_percentile": 40,
}


def load_params() -> dict:
    return AppConfig.load().to_dict()


def save_params(p: dict) -> None:
    cfg = AppConfig(**{k: v for k, v in p.items() if k in AppConfig.__dataclass_fields__})
    cfg.save()


def reset_params() -> None:
    AppConfig.reset()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def find_disks(
    img: np.ndarray,
    use_ml: bool = True,
) -> List[Tuple[Tuple[float, float], float]]:
    from core.detection import find_disk_candidates
    from services.analysis import _apply_ml_filter
    cfg = AppConfig.load()
    candidates = find_disk_candidates(img, cfg)
    if use_ml and candidates:
        candidates = _apply_ml_filter(img, candidates)
    return candidates


def process_antibiogram(
    img: np.ndarray,
    disk_labels: Optional[List[str]] = None,
    use_ml: bool = True,
) -> Tuple[List[DetectedDisk], float]:
    from services.analysis import run_analysis
    cfg = AppConfig.load()
    result = run_analysis(img, cfg=cfg, use_ml=use_ml, disk_labels=disk_labels)
    return result.disks, result.px_per_mm


def process_with_params(
    img: np.ndarray,
    params_override: dict,
    disk_labels: Optional[List[str]] = None,
) -> Tuple[List[DetectedDisk], float]:
    """Run the full pipeline with a temporary parameter override (no disk write)."""
    from services.analysis import run_analysis
    cfg = AppConfig(**{k: v for k, v in params_override.items() if k in AppConfig.__dataclass_fields__})
    result = run_analysis(img, cfg=cfg, disk_labels=disk_labels)
    return result.disks, result.px_per_mm
