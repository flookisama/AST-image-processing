"""Domain models — shared across core/, services/ and pages/."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

DISK_DIAMETER_MM: float = 6.0


@dataclass
class DetectedDisk:
    center: Tuple[float, float]     # (x, y) in pixels
    radius_px: float                # disk radius in pixels
    zone_diameter_mm: float         # inhibition zone diameter
    confidence: float               # 0–1, measurement quality
    label: str                      # e.g. "CIP", "Disk 3"
    zone_radius_px: float = field(default=0.0)


@dataclass
class AnalysisResult:
    disks: List[DetectedDisk]
    px_per_mm: float
    image_shape: Tuple[int, int]    # (height, width)
    used_ml: bool = True

    @property
    def disk_count(self) -> int:
        return len(self.disks)

    @property
    def has_results(self) -> bool:
        return bool(self.disks)

    @staticmethod
    def empty(image_shape: Tuple[int, int] = (0, 0)) -> "AnalysisResult":
        return AnalysisResult(disks=[], px_per_mm=0.0, image_shape=image_shape)
