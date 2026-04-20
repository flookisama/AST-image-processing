"""
Parse antibiogram data from Word documents (.docx).
Handles common clinical table formats (French and English).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


@dataclass
class AntibioEntry:
    antibiotic: str
    zone_mm: Optional[float] = None
    interpretation: Optional[str] = None   # "S", "I", or "R"
    disk_load: Optional[str] = None


@dataclass
class AntibiogramRecord:
    source_file: str
    sample_id: Optional[str] = None
    patient_id: Optional[str] = None
    date: Optional[str] = None
    bacteria: Optional[str] = None
    entries: List[AntibioEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "sample_id": self.sample_id,
            "patient_id": self.patient_id,
            "date": self.date,
            "bacteria": self.bacteria,
            "entries": [
                {
                    "antibiotic": e.antibiotic,
                    "zone_mm": e.zone_mm,
                    "interpretation": e.interpretation,
                    "disk_load": e.disk_load,
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AntibiogramRecord":
        rec = cls(
            source_file=d.get("source_file", ""),
            sample_id=d.get("sample_id"),
            patient_id=d.get("patient_id"),
            date=d.get("date"),
            bacteria=d.get("bacteria"),
        )
        for e in d.get("entries", []):
            rec.entries.append(
                AntibioEntry(
                    antibiotic=e.get("antibiotic", ""),
                    zone_mm=e.get("zone_mm"),
                    interpretation=e.get("interpretation"),
                    disk_load=e.get("disk_load"),
                )
            )
        return rec


# ── Keyword sets for column classification ──────────────────────────────────

_AB_KW = {"antibiotique", "antimicrobien", "antibiotic", "agent", "molecule", "molécule", "antibio"}
_ZONE_KW = {"zone", "diametre", "diamètre", "diam", "inhibition", "mm"}
_SIR_KW = {"resultat", "résultat", "interpretation", "interp", "sir", "sensib", "r/s", "s/i/r", "classif"}
_LOAD_KW = {"charge", "dose", "µg", "ug", "disk", "disque", "concentration"}
_PATIENT_KW = {"patient", "num", "prelevement", "prélèvement", "echantillon", "échantillon", "n°", "identif"}
_BACTERIA_KW = {"bacterie", "bactérie", "bacteria", "germe", "organisme", "espece", "espèce", "microorg"}


def _norm(s: str) -> str:
    """Lowercase, strip accents, remove punctuation for keyword matching."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^\w\s/]", " ", s).strip().lower()


def _classify_header(text: str) -> str:
    n = _norm(text)
    words = set(n.split())
    if words & _AB_KW or any(k in n for k in _AB_KW):
        return "antibiotic"
    if words & _ZONE_KW or any(k in n for k in _ZONE_KW):
        return "zone"
    if words & _SIR_KW or any(k in n for k in _SIR_KW):
        return "sir"
    if words & _LOAD_KW or any(k in n for k in _LOAD_KW):
        return "load"
    if words & _PATIENT_KW or any(k in n for k in _PATIENT_KW):
        return "patient"
    if words & _BACTERIA_KW or any(k in n for k in _BACTERIA_KW):
        return "bacteria"
    return "other"


def _parse_zone(s: str) -> Optional[float]:
    """Extract a zone diameter in mm from a cell string."""
    if not s or not s.strip():
        return None
    s = s.strip()
    # Number + optional "mm"
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*mm?", s, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", "."))
    # Bare number in plausible range
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?)", s)
    if m:
        v = float(m.group(1).replace(",", "."))
        if 6.0 <= v <= 55.0:
            return v
    return None


def _parse_sir(s: str) -> Optional[str]:
    """Extract S/I/R from a cell string."""
    if not s:
        return None
    s = s.strip().upper()
    if s in ("S", "I", "R"):
        return s
    n = _norm(s)
    if any(k in n for k in ("sensible", "susceptible", "suscept")):
        return "S"
    if "interm" in n:
        return "I"
    if any(k in n for k in ("resist", "résist")):
        return "R"
    return None


def _cell(cell) -> str:
    return cell.text.strip()


def _parse_table(table, source_name: str) -> List[AntibiogramRecord]:
    """
    Try to parse a Word table as an antibiogram.
    Supports:
     - Standard format: rows=antibiotics, cols=antibiotic|zone|SIR
     - Multi-patient: paired zone+SIR columns per patient
    """
    rows = table.rows
    if len(rows) < 2:
        return []

    header_cells = [_cell(c) for c in rows[0].cells]
    col_types = [_classify_header(h) for h in header_cells]

    ab_col = next((i for i, t in enumerate(col_types) if t == "antibiotic"), None)
    zone_cols = [i for i, t in enumerate(col_types) if t == "zone"]
    sir_cols = [i for i, t in enumerate(col_types) if t == "sir"]
    load_col = next((i for i, t in enumerate(col_types) if t == "load"), None)

    # Fallback: if no header recognized but table has 3+ cols, assume col0=antibiotic
    if ab_col is None and not zone_cols and not sir_cols and len(header_cells) >= 3:
        ab_col = 0
        # Check second row to guess column semantics
        sample = [_cell(c) for c in rows[1].cells] if len(rows) > 1 else []
        for i, v in enumerate(sample[1:], 1):
            if _parse_zone(v) is not None:
                zone_cols.append(i)
                break
        for i, v in enumerate(sample[1:], 1):
            if _parse_sir(v) is not None and i not in zone_cols:
                sir_cols.append(i)
                break

    if ab_col is None and not zone_cols and not sir_cols:
        return []

    if ab_col is None:
        ab_col = 0

    n_patients = max(len(zone_cols), len(sir_cols), 1)
    records = [AntibiogramRecord(source_file=source_name) for _ in range(n_patients)]

    for row in rows[1:]:
        cells = [_cell(c) for c in row.cells]
        if not cells:
            continue
        ab_name = cells[ab_col].strip() if ab_col < len(cells) else ""
        if not ab_name:
            continue
        # Skip header-like rows that leaked into data
        if _norm(ab_name) in _AB_KW or _classify_header(ab_name) == "antibiotic":
            continue

        load = cells[load_col].strip() if load_col is not None and load_col < len(cells) else None

        for pi in range(n_patients):
            zone_val: Optional[float] = None
            sir_val: Optional[str] = None

            if pi < len(zone_cols) and zone_cols[pi] < len(cells):
                raw = cells[zone_cols[pi]]
                zone_val = _parse_zone(raw)
                if zone_val is None:
                    sir_val = _parse_sir(raw)

            if pi < len(sir_cols) and sir_cols[pi] < len(cells):
                raw = cells[sir_cols[pi]]
                if sir_val is None:
                    sir_val = _parse_sir(raw)
                if zone_val is None:
                    zone_val = _parse_zone(raw)

            if zone_val is not None or sir_val is not None:
                records[pi].entries.append(
                    AntibioEntry(
                        antibiotic=ab_name,
                        zone_mm=zone_val,
                        interpretation=sir_val,
                        disk_load=load,
                    )
                )

    return [r for r in records if r.entries]


def _extract_metadata(doc, records: List[AntibiogramRecord]) -> None:
    """Enrich records with patient/bacteria info found in document paragraphs."""
    full_text = "\n".join(p.text for p in doc.paragraphs)

    patient_m = re.search(r"(?:patient|malade)\s*[:\-#]?\s*([^\n]{1,60})", full_text, re.IGNORECASE)
    date_m = re.search(r"date\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", full_text, re.IGNORECASE)
    sample_m = re.search(
        r"(?:pr[eé]l[eè]vement|echantillon|échantillon|n[°o])\s*[:\-#]?\s*([\w\-/]{1,30})",
        full_text,
        re.IGNORECASE,
    )
    bacteria_patterns = [
        r"(?:bact[eé]rie|germe|organisme|esp[eè]ce)\s*[:\-]?\s*([^\n]{1,80})",
        r"(?:isolat|micro-organisme)\s*[:\-]?\s*([^\n]{1,80})",
    ]
    bacteria_m = None
    for pat in bacteria_patterns:
        bacteria_m = re.search(pat, full_text, re.IGNORECASE)
        if bacteria_m:
            break

    for rec in records:
        if patient_m and not rec.patient_id:
            rec.patient_id = patient_m.group(1).strip()[:60]
        if date_m and not rec.date:
            rec.date = date_m.group(1).strip()
        if sample_m and not rec.sample_id:
            rec.sample_id = sample_m.group(1).strip()[:30]
        if bacteria_m and not rec.bacteria:
            rec.bacteria = bacteria_m.group(1).strip()[:80]


def parse_word_file(filepath: str | Path) -> List[AntibiogramRecord]:
    """
    Parse a .docx antibiogram file.
    Returns a list of AntibiogramRecord (one per detected patient/sample).
    Raises ImportError if python-docx is not installed.
    """
    if not HAS_DOCX:
        raise ImportError("python-docx is required — install with: pip install python-docx")

    path = Path(filepath)
    doc = Document(str(path))
    records: List[AntibiogramRecord] = []

    for table in doc.tables:
        parsed = _parse_table(table, path.name)
        records.extend(parsed)

    _extract_metadata(doc, records)
    return records


def parse_word_files(filepaths: List[str | Path]) -> List[AntibiogramRecord]:
    """Parse multiple Word files, silently skipping files that fail."""
    all_records: List[AntibiogramRecord] = []
    for fp in filepaths:
        try:
            all_records.extend(parse_word_file(fp))
        except Exception as exc:
            print(f"[word_importer] Could not parse {fp}: {exc}")
    return all_records
