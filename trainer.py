"""
Auto-calibration training engine.

Given a set of (image, ground-truth zones) pairs, performs a grid search over
key detection parameters and returns the combination that minimises zone MAE.

The search is split into two fast sequential stages:
  Stage 1 — disk detection  : sweep hough_param2 & clahe_clip
  Stage 2 — zone measurement: sweep zone_threshold_ratio (no re-detection)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict, Any

# ── Search grids ───────────────────────────────────────────────────────────────
PARAM2_GRID        = [15, 20, 25, 30, 35, 40, 45, 50]
CLAHE_CLIP_GRID    = [1.0, 1.5, 2.0, 2.5, 3.0]
THRESHOLD_GRID     = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class TrainingSample:
    name: str
    image: np.ndarray
    gt_zones: List[float]          # ground-truth zone diameters (mm)
    gt_antibiotics: List[str] = field(default_factory=list)
    gt_sir: List[str] = field(default_factory=list)


@dataclass
class CalibrationResult:
    best_params: Dict[str, Any]
    best_mae: float
    baseline_mae: float            # MAE with default parameters
    improvement_pct: float
    grid_df: pd.DataFrame          # full grid search results
    per_sample_df: pd.DataFrame    # per-image results with best params


# ── Helpers ────────────────────────────────────────────────────────────────────
def _match_zones(detected: List[float], gt: List[float]) -> List[Tuple[float, float]]:
    """
    Best-effort matching of detected ↔ ground-truth zones.
    Both lists are sorted and paired by index up to min(len).
    """
    n = min(len(detected), len(gt))
    if n == 0:
        return []
    return list(zip(sorted(detected)[:n], sorted(gt)[:n]))


def _mae(detected: List[float], gt: List[float]) -> Optional[float]:
    pairs = _match_zones(detected, gt)
    if not pairs:
        return None
    return float(np.mean([abs(d - g) for d, g in pairs]))


def _run(img: np.ndarray, params: dict) -> Tuple[List[float], int]:
    """Run process_with_params and return (zone_list, n_disks)."""
    from antibiogram_processor import process_with_params
    try:
        disks, _ = process_with_params(img, params)
        return [d.zone_diameter_mm for d in disks], len(disks)
    except Exception:
        return [], 0


def _score_params(samples: List[TrainingSample], params: dict) -> Tuple[float, float]:
    """
    Return (mean_MAE, mean_disk_count) across all samples.
    Samples with no detection contribute MAE = 50 (penalty).
    """
    maes, counts = [], []
    for s in samples:
        zones, n = _run(s.image, params)
        counts.append(n)
        mae = _mae(zones, s.gt_zones)
        maes.append(mae if mae is not None else 50.0)
    return float(np.mean(maes)), float(np.mean(counts))


# ── Main calibration function ──────────────────────────────────────────────────
def auto_calibrate(
    samples: List[TrainingSample],
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> CalibrationResult:
    """
    Two-stage grid search calibration.

    progress_cb(fraction 0-1, status_message) is called after each step.
    Returns a CalibrationResult with best parameters and evaluation DataFrames.
    """
    from antibiogram_processor import load_params

    base_params = load_params()

    def _report(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    # ── Baseline (current params) ──────────────────────────────────────────
    _report(0.0, "Calcul de la baseline avec les paramètres actuels…")
    baseline_mae, _ = _score_params(samples, base_params)

    # ── Stage 1 : sweep hough_param2 × clahe_clip ─────────────────────────
    stage1_results = []
    total_s1 = len(PARAM2_GRID) * len(CLAHE_CLIP_GRID)
    step = 0

    for p2 in PARAM2_GRID:
        for clip in CLAHE_CLIP_GRID:
            params = {**base_params, "hough_param2": p2, "clahe_clip": clip}
            mae, avg_n = _score_params(samples, params)
            stage1_results.append({"hough_param2": p2, "clahe_clip": clip, "MAE": mae, "avg_disks": avg_n})
            step += 1
            _report(0.05 + 0.45 * step / total_s1, f"Étape 1/2 — param2={p2}, clahe={clip:.1f} → MAE={mae:.2f}mm")

    stage1_df = pd.DataFrame(stage1_results).sort_values("MAE")
    best_s1 = stage1_df.iloc[0]
    best_p2 = int(best_s1["hough_param2"])
    best_clip = float(best_s1["clahe_clip"])

    # ── Stage 2 : sweep zone_threshold_ratio (best p2+clip fixed) ─────────
    stage2_results = []
    total_s2 = len(THRESHOLD_GRID)

    for thr in THRESHOLD_GRID:
        params = {**base_params, "hough_param2": best_p2, "clahe_clip": best_clip, "zone_threshold_ratio": thr}
        mae, avg_n = _score_params(samples, params)
        stage2_results.append({"zone_threshold_ratio": thr, "MAE": mae, "avg_disks": avg_n})
        step_s2 = THRESHOLD_GRID.index(thr) + 1
        _report(0.50 + 0.45 * step_s2 / total_s2, f"Étape 2/2 — seuil={thr:.2f} → MAE={mae:.2f}mm")

    stage2_df = pd.DataFrame(stage2_results).sort_values("MAE")
    best_s2 = stage2_df.iloc[0]
    best_threshold = float(best_s2["zone_threshold_ratio"])

    # ── Best combined params ───────────────────────────────────────────────
    best_params = {
        **base_params,
        "hough_param2": best_p2,
        "clahe_clip": best_clip,
        "zone_threshold_ratio": best_threshold,
    }
    best_mae, _ = _score_params(samples, best_params)

    # ── Per-sample evaluation with best params ─────────────────────────────
    _report(0.96, "Évaluation par échantillon avec les meilleurs paramètres…")
    per_sample_rows = []
    for s in samples:
        zones, n_found = _run(s.image, best_params)
        baseline_zones, _ = _run(s.image, base_params)

        mae_new = _mae(zones, s.gt_zones)
        mae_base = _mae(baseline_zones, s.gt_zones)

        pairs = _match_zones(zones, s.gt_zones)
        for det, gt in pairs:
            per_sample_rows.append({
                "Échantillon": s.name,
                "GT Zone (mm)": round(gt, 1),
                "Détectée (mm)": round(det, 1),
                "Erreur (mm)": round(abs(det - gt), 1),
                "OK (≤2mm)": "✅" if abs(det - gt) <= 2.0 else "❌",
                "Disques trouvés": n_found,
                "Disques attendus": len(s.gt_zones),
            })

        if not pairs and s.gt_zones:
            per_sample_rows.append({
                "Échantillon": s.name,
                "GT Zone (mm)": "—",
                "Détectée (mm)": "—",
                "Erreur (mm)": "N/D",
                "OK (≤2mm)": "❌",
                "Disques trouvés": n_found,
                "Disques attendus": len(s.gt_zones),
            })

    per_sample_df = pd.DataFrame(per_sample_rows)

    # Full grid table (stage1 × stage2 combined overview)
    grid_rows = []
    for r1 in stage1_results:
        for r2 in stage2_results:
            grid_rows.append({
                "param2": r1["hough_param2"],
                "clahe_clip": r1["clahe_clip"],
                "zone_threshold": r2["zone_threshold_ratio"],
                "MAE Stage1": round(r1["MAE"], 2),
                "MAE Stage2": round(r2["MAE"], 2),
            })
    grid_df = (
        stage2_df.rename(columns={"MAE": "MAE (mm)"})
        .assign(**{"param2": best_p2, "clahe_clip": best_clip})
        [["param2", "clahe_clip", "zone_threshold_ratio", "MAE (mm)", "avg_disks"]]
    )

    _report(1.0, "Calibration terminée.")
    improvement_pct = (
        100.0 * (baseline_mae - best_mae) / baseline_mae if baseline_mae > 0 else 0.0
    )

    return CalibrationResult(
        best_params=best_params,
        best_mae=best_mae,
        baseline_mae=baseline_mae,
        improvement_pct=improvement_pct,
        grid_df=grid_df,
        per_sample_df=per_sample_df,
    )


# ── Dataset helpers ────────────────────────────────────────────────────────────
def samples_from_ground_truth(
    gt_records: list,
    images: Dict[str, np.ndarray],
) -> List[TrainingSample]:
    """
    Build TrainingSample list from ground_truth.json records and image dict.
    gt_records: list of dicts as saved by the training data page.
    images: {filename: np.ndarray RGB}
    """
    samples = []
    for rec in gt_records:
        img_name = rec.get("image_name", "")
        if img_name not in images:
            continue
        gt_zones = [e["zone_mm"] for e in rec.get("entries", []) if e.get("zone_mm") is not None]
        if not gt_zones:
            continue
        samples.append(TrainingSample(
            name=img_name,
            image=images[img_name],
            gt_zones=gt_zones,
            gt_antibiotics=[e.get("antibiotic", "") for e in rec.get("entries", [])],
            gt_sir=[e.get("interpretation", "") for e in rec.get("entries", [])],
        ))
    return samples


def filename_similarity(a: str, b: str) -> float:
    """Simple similarity score between two filenames (0–1)."""
    import difflib
    a_stem = a.rsplit(".", 1)[0].lower()
    b_stem = b.rsplit(".", 1)[0].lower()
    return difflib.SequenceMatcher(None, a_stem, b_stem).ratio()


def auto_match_files(
    image_names: List[str],
    word_names: List[str],
    threshold: float = 0.4,
) -> Dict[str, Optional[str]]:
    """
    Return {image_name: best_matching_word_name or None}.
    """
    matches: Dict[str, Optional[str]] = {}
    for img in image_names:
        best_score, best_word = 0.0, None
        for wrd in word_names:
            score = filename_similarity(img, wrd)
            if score > best_score:
                best_score, best_word = score, wrd
        matches[img] = best_word if best_score >= threshold else None
    return matches
