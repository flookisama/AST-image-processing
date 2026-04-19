"""
ML-enhanced disk detector.

Trains a HOG+SVM binary classifier on disk / non-disk patches extracted from
antibiogram images. The classifier is used as a post-processing filter on top
of the existing Hough-based candidate generator, eliminating false positives
caused by reflections, bubbles, and other circular artefacts.

Usage
-----
1. Train:
    from ml_detector import DiskClassifier
    clf = DiskClassifier()
    clf.fit_from_images(images_dict)       # {name: np.ndarray RGB}
    clf.save()

2. Inference (in antibiogram_processor.py):
    from ml_detector import load_classifier
    clf = load_classifier()                # None if not trained
    candidates = find_disks_raw(img)       # Hough only
    if clf:
        candidates = clf.filter(img, candidates)
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

MODEL_PATH = Path(__file__).parent / "data" / "disk_classifier.pkl"
PATCH_SIZE = 48


# ── Patch extraction ───────────────────────────────────────────────────────────

def _extract_patch(gray: np.ndarray, cx: float, cy: float, r: float) -> Optional[np.ndarray]:
    """Crop a normalised PATCH_SIZE×PATCH_SIZE window centred on a disk."""
    h, w = gray.shape
    pad = max(int(r * 1.5), 12)
    x1, y1 = max(0, int(cx) - pad), max(0, int(cy) - pad)
    x2, y2 = min(w, int(cx) + pad), min(h, int(cy) + pad)
    crop = gray[y1:y2, x1:x2]
    if crop.size < 16 * 16:
        return None
    patch = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    # Normalise brightness
    patch = cv2.equalizeHist(patch)
    return patch


def _hog_features(patch: np.ndarray) -> np.ndarray:
    """HOG descriptor for a PATCH_SIZE×PATCH_SIZE uint8 image."""
    hog = cv2.HOGDescriptor(
        _winSize=(PATCH_SIZE, PATCH_SIZE),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(patch).ravel()


# ── Training data generation ───────────────────────────────────────────────────

def _generate_negatives(gray: np.ndarray, positives: List[Tuple[float, float, float]],
                        n: int, min_r: int = 10) -> List[np.ndarray]:
    """Sample random patches that don't overlap with any known disk."""
    h, w = gray.shape
    rng = np.random.default_rng(42)
    negs: List[np.ndarray] = []
    attempts = 0
    while len(negs) < n and attempts < n * 20:
        attempts += 1
        r = rng.integers(min_r, min_r * 3)
        cx = rng.integers(r, w - r)
        cy = rng.integers(r, h - r)
        # Check no overlap with a positive
        overlap = any(
            np.hypot(cx - px, cy - py) < (pr + r) * 1.1
            for px, py, pr in positives
        )
        if not overlap:
            patch = _extract_patch(gray, float(cx), float(cy), float(r))
            if patch is not None:
                negs.append(patch)
    return negs


def generate_training_patches(
    images: Dict[str, np.ndarray],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays from a dict of RGB images.
    Uses calibrated Hough to find positive (disk) candidates and
    random patches as negatives.
    """
    from antibiogram_processor import find_disks, _to_gray_uint8

    X_patches: List[np.ndarray] = []
    y_labels: List[int] = []

    for name, img_rgb in images.items():
        gray = _to_gray_uint8(img_rgb)
        candidates = find_disks(img_rgb)   # validated (dark-circle filtered)

        if verbose:
            print(f"  {name}: {len(candidates)} disks found", end="  ")

        pos_list: List[Tuple[float, float, float]] = []
        pos_patches: List[np.ndarray] = []
        for (cx, cy), r in candidates:
            patch = _extract_patch(gray, cx, cy, r)
            if patch is not None:
                pos_list.append((cx, cy, r))
                pos_patches.append(patch)

        n_pos = len(pos_patches)
        neg_patches = _generate_negatives(gray, pos_list, n=max(n_pos * 2, 4))

        if verbose:
            print(f"pos={n_pos} neg={len(neg_patches)}")

        for p in pos_patches:
            X_patches.append(p)
            y_labels.append(1)
        for p in neg_patches:
            X_patches.append(p)
            y_labels.append(0)

    if not X_patches:
        raise ValueError("No training patches generated — check image paths.")

    X_hog = np.array([_hog_features(p) for p in X_patches])
    y = np.array(y_labels)
    return X_hog, y


# ── Classifier ─────────────────────────────────────────────────────────────────

class DiskClassifier:
    def __init__(self):
        self._clf = None
        self._trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train SVM on pre-computed HOG features. Returns cross-val metrics."""
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=5.0, gamma="scale", probability=True)),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipeline, X, y, cv=min(5, int(np.bincount(y).min())),
                                     scoring="f1")
        pipeline.fit(X, y)
        self._clf = pipeline
        self._trained = True
        metrics = {"f1_mean": float(scores.mean()), "f1_std": float(scores.std()),
                   "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
        return metrics

    def fit_from_images(
        self,
        images: Dict[str, np.ndarray],
        verbose: bool = True,
        progress_cb=None,
    ) -> dict:
        """Full pipeline: generate patches → train. Returns metrics dict."""
        if progress_cb:
            progress_cb(0.0, "Génération des patches…")
        X, y = generate_training_patches(images, verbose=verbose)
        if progress_cb:
            progress_cb(0.6, f"Entraînement SVM sur {len(X)} patches…")
        metrics = self.fit(X, y)
        if progress_cb:
            progress_cb(1.0, "Entraînement terminé.")
        return metrics

    def filter(
        self,
        img: np.ndarray,
        candidates: List[Tuple[Tuple[float, float], float]],
        threshold: float = 0.45,
    ) -> List[Tuple[Tuple[float, float], float]]:
        """Filter Hough candidates — keep only those the SVM calls a disk."""
        if not self._trained or not candidates:
            return candidates
        from antibiogram_processor import _to_gray_uint8
        gray = _to_gray_uint8(img)
        kept = []
        for (cx, cy), r in candidates:
            patch = _extract_patch(gray, cx, cy, r)
            if patch is None:
                continue
            feat = _hog_features(patch).reshape(1, -1)
            prob = self._clf.predict_proba(feat)[0][1]
            if prob >= threshold:
                kept.append(((cx, cy), r))
        return kept

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._clf, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DiskClassifier":
        obj = cls()
        with open(path, "rb") as f:
            obj._clf = pickle.load(f)
        obj._trained = True
        return obj

    @property
    def is_trained(self) -> bool:
        return self._trained


def load_classifier() -> Optional[DiskClassifier]:
    """Return a trained DiskClassifier or None if no model file exists."""
    if MODEL_PATH.is_file():
        try:
            return DiskClassifier.load()
        except Exception:
            pass
    return None
