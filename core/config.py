"""
Application configuration.

Replaces the global _params dict antipattern with an explicit dataclass.
No module-level mutable state — callers own the config object.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "data" / "calibration.json"


@dataclass
class AppConfig:
    # CLAHE preprocessing
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    # Hough circle detection
    hough_dp: float = 1.0
    hough_param1: int = 80
    hough_param2: int = 35
    hough_min_r_ratio: float = 0.02
    hough_max_r_ratio: float = 0.15
    # Zone measurement
    zone_threshold_ratio: float = 0.40
    # Dark-circle validation
    dark_validation: bool = True
    dark_percentile: int = 40

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = _CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        log.info("Config saved to %s", path)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path = _CONFIG_PATH) -> "AppConfig":
        if path.is_file():
            try:
                with open(path) as f:
                    data = json.load(f)
                valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                return cls(**valid)
            except Exception as exc:
                log.warning("Could not load config from %s: %s — using defaults", path, exc)
        return cls()

    @classmethod
    def default(cls) -> "AppConfig":
        return cls()

    @classmethod
    def reset(cls, path: Path = _CONFIG_PATH) -> "AppConfig":
        if path.is_file():
            path.unlink()
            log.info("Config reset — deleted %s", path)
        return cls()
