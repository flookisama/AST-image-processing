"""
Annotation drawing — overlay disks and zones on an image.
Returns RGB numpy array suitable for Streamlit st.image().
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .models import DetectedDisk


def _confidence_color(confidence: float) -> tuple:
    if confidence > 0.65:
        return (0, 200, 0)    # green
    if confidence > 0.35:
        return (0, 165, 255)  # orange
    return (0, 0, 220)        # red


def draw_results(
    img: np.ndarray,
    disks: List[DetectedDisk],
    px_per_mm: float,
) -> np.ndarray:
    """
    Draw disk outlines and zone circles on img.
    Returns a copy in RGB colour space (Streamlit-compatible).
    """
    out = np.asarray(img).copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    for d in disks:
        cx, cy = int(d.center[0]), int(d.center[1])
        r_disk = int(d.radius_px)
        r_zone = int((d.zone_diameter_mm / 2.0) * px_per_mm)
        zone_color = _confidence_color(d.confidence)

        cv2.circle(out, (cx, cy), r_disk, (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), r_zone, zone_color, 2)
        cv2.putText(
            out,
            f"{d.label}  {d.zone_diameter_mm:.1f}mm",
            (cx - 45, cy - r_disk - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
