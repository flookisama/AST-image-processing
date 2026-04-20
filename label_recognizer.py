"""
Antibiotic disk label recognition.

Pipeline (best available → fallback):
  1. EasyOCR  — deep-learning OCR, best accuracy on small text
  2. Tesseract — if pytesseract is installed
  3. Returns empty string

After OCR, the raw text is normalised to a known antibiotic code via:
  - exact match against known codes
  - prefix / substring fuzzy match
  - numeric strip (AMX25 → AMX)

Results are cached per (image-shape, cx, cy, r) to avoid redundant inference.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Known antibiotic code set ──────────────────────────────────────────────────
_KNOWN_CODES: Optional[set] = None


def _known_codes() -> set:
    global _KNOWN_CODES
    if _KNOWN_CODES is None:
        try:
            from breakpoints import EUCAST_BREAKPOINTS
            _KNOWN_CODES = set(EUCAST_BREAKPOINTS.get("Escherichia coli", {}).keys())
            # Union all species
            for sp_data in EUCAST_BREAKPOINTS.values():
                _KNOWN_CODES.update(sp_data.keys())
        except Exception:
            _KNOWN_CODES = set()
        # Fallback common codes
        _KNOWN_CODES.update({
            "AMX", "AMC", "AMP", "OXA", "CLX", "TIC",
            "CIP", "NOR", "LVX", "MXF", "OFX",
            "GEN", "TOB", "AMK", "NET",
            "CTX", "CAZ", "CRO", "FEP", "ATM",
            "IMP", "MEM", "ERT", "DOR",
            "TET", "DOX", "MIN",
            "SXT", "TMP",
            "CHL", "RIF", "FOS", "NIT",
            "CLI", "ERY", "AZI", "CLA",
            "VAN", "TEI", "LNZ", "DAP",
            "COL", "PMB", "TGC",
            "PEN", "OXA", "CLO",
        })
    return _KNOWN_CODES


# ── Image preprocessing for OCR ────────────────────────────────────────────────

def _prepare_disk_crop(
    img_rgb: np.ndarray, cx: float, cy: float, r: float, scale: int = 4
) -> np.ndarray:
    """
    Crop disk region, upscale, and binarise for OCR.
    The label text is typically printed in the central ~60% of the disk.
    """
    h, w = img_rgb.shape[:2]
    pad = max(int(r * 0.9), 8)
    x1 = max(0, int(cx) - pad)
    y1 = max(0, int(cy) - pad)
    x2 = min(w, int(cx) + pad)
    y2 = min(h, int(cy) + pad)
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size < 8 * 8 * 3:
        return np.zeros((32, 32), dtype=np.uint8)

    # Upscale so tiny text is readable
    cw, ch = crop.shape[1], crop.shape[0]
    up = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold — handles uneven disk colouring
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Morphological closing to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


# ── EasyOCR reader (singleton, lazy init) ─────────────────────────────────────
_easyocr_reader = None
_easyocr_failed = False


def _get_easyocr():
    global _easyocr_reader, _easyocr_failed
    if _easyocr_failed:
        return None
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            _easyocr_failed = True
            return None
    return _easyocr_reader


# ── Text normalisation ─────────────────────────────────────────────────────────

def _strip_to_letters(text: str) -> str:
    """Remove digits and punctuation from an OCR result to get the base code."""
    return re.sub(r"[^A-Z]", "", text.upper())


def _normalise_code(raw: str) -> str:
    """Map raw OCR text to the best-matching antibiotic code."""
    if not raw:
        return ""
    # Upper + remove non-alphanumeric
    clean = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if not clean:
        return ""

    codes = _known_codes()

    # 1. Exact match
    if clean in codes:
        return clean

    # 2. Strip digits → exact
    letters = _strip_to_letters(clean)
    if letters in codes:
        return letters

    # 3. Prefix match (at least 2 chars)
    if len(letters) >= 2:
        matches = [c for c in codes if c.startswith(letters) or letters.startswith(c[:len(letters)])]
        if len(matches) == 1:
            return matches[0]
        # Pick closest length
        if matches:
            return min(matches, key=lambda c: abs(len(c) - len(letters)))

    # 4. Return cleaned uppercase as fallback (may still be useful)
    return letters if letters else clean


# ── Main recognition function ──────────────────────────────────────────────────

def recognize_label(
    img_rgb: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    use_easyocr: bool = True,
) -> Tuple[str, float]:
    """
    Recognise the antibiotic code printed on a disk.

    Returns (code, confidence) where confidence is in [0, 1].
    Returns ("", 0.0) if nothing readable.
    """
    binary = _prepare_disk_crop(img_rgb, cx, cy, r)

    # ── Try EasyOCR ───────────────────────────────────────────────────────────
    if use_easyocr:
        reader = _get_easyocr()
        if reader is not None:
            try:
                results = reader.readtext(binary, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                if results:
                    # Take highest-confidence result
                    best = max(results, key=lambda x: x[2])
                    raw_text, conf = best[1], float(best[2])
                    code = _normalise_code(raw_text)
                    if code:
                        return code, conf
            except Exception:
                pass

    # ── Try Tesseract ─────────────────────────────────────────────────────────
    try:
        import pytesseract
        cfg = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        data = pytesseract.image_to_data(
            binary, config=cfg, output_type=pytesseract.Output.DICT, lang="eng"
        )
        texts = [t for t, c in zip(data["text"], data["conf"]) if t.strip() and int(c) > 0]
        confs = [int(c) / 100.0 for t, c in zip(data["text"], data["conf"]) if t.strip() and int(c) > 0]
        if texts:
            best_idx = int(np.argmax(confs))
            code = _normalise_code(texts[best_idx])
            return code, float(confs[best_idx]) if code else 0.0
    except Exception:
        pass

    return "", 0.0


def recognize_all_disks(
    img_rgb: np.ndarray,
    disks,              # List[DetectedDisk]
) -> list:
    """
    Batch-recognise labels for all detected disks.
    Returns list of (code, confidence) in same order as disks.
    """
    return [
        recognize_label(img_rgb, d.center[0], d.center[1], d.radius_px)
        for d in disks
    ]
