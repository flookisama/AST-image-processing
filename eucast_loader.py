"""
Load EUCAST breakpoint tables from the official Excel file (e.g. v_16.0__BreakpointTables.xlsx).
Falls back to built-in breakpoints if file is missing or unreadable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Map (guideline, sheet_name) -> app species name
SHEET_TO_SPECIES = {
    "Enterobacterales": "Escherichia coli",
    "Staphylococcus": "Staphylococcus aureus",
    "Pseudomonas": "Pseudomonas aeruginosa",
    "S.pneumoniae": "Streptococcus pneumoniae",
}

# Sheets we read (EUCAST disk diffusion zone breakpoints)
EUCAST_SHEETS = ["Enterobacterales", "Staphylococcus", "Pseudomonas", "S.pneumoniae"]

# Agent name (start of string) -> short code for dropdown/lookup
AGENT_NAME_TO_CODE: List[Tuple[str, str]] = [
    ("Amoxicillin-clavulanic acid", "AMC"),
    ("Ampicillin-sulbactam", "AMS"),
    ("Piperacillin-tazobactam", "TZP"),
    ("Ticarcillin-clavulanic acid", "TCC"),
    ("Cefepime-enmetazobactam", "CFE"),
    ("Ceftazidime-avibactam", "CZA"),
    ("Ceftolozane-tazobactam", "CTL"),
    ("Trimethoprim-sulfamethoxazole", "SXT"),
    ("Co-trimoxazole", "SXT"),
    ("Benzylpenicillin", "PEN"),
    ("Ampicillin", "AMP"),
    ("Amoxicillin", "AMX"),
    ("Piperacillin", "PIP"),
    ("Temocillin", "TEM"),
    ("Phenoxymethylpenicillin", "PEN"),
    ("Oxacillin", "OXA"),
    ("Cloxacillin", "CLX"),
    ("Dicloxacillin", "DIC"),
    ("Flucloxacillin", "FLX"),
    ("Mecillinam", "MEC"),
    ("Cefaclor", "CEC"),
    ("Cefadroxil", "CFR"),
    ("Cefalexin", "LEX"),
    ("Cefazolin", "CFZ"),
    ("Cefepime", "FEP"),
    ("Cefiderocol", "FDC"),
    ("Cefixime", "CFM"),
    ("Cefotaxime", "CTX"),
    ("Cefoxitin", "FOX"),
    ("Cefpodoxime", "CPD"),
    ("Ceftaroline", "CPT"),
    ("Ceftazidime", "CAZ"),
    ("Ceftibuten", "CTB"),
    ("Ceftobiprole", "BPR"),
    ("Ceftriaxone", "CRO"),
    ("Cefuroxime", "CXM"),
    ("Ciprofloxacin", "CIP"),
    ("Levofloxacin", "LEV"),
    ("Moxifloxacin", "MXF"),
    ("Gentamicin", "GEN"),
    ("Tobramycin", "TOB"),
    ("Amikacin", "AMK"),
    ("Trimethoprim", "TMP"),
    ("Fosfomycin", "FOS"),
    ("Nitrofurantoin", "NIT"),
    ("Colistin", "COL"),
    ("Polymyxin B", "PB"),
    ("Erythromycin", "ERY"),
    ("Clarithromycin", "CLR"),
    ("Azithromycin", "AZM"),
    ("Tetracycline", "TET"),
    ("Tigecycline", "TGC"),
    ("Doxycycline", "DOX"),
    ("Vancomycin", "VAN"),
    ("Teicoplanin", "TEI"),
    ("Linezolid", "LZD"),
    ("Daptomycin", "DAP"),
    ("Chloramphenicol", "CHL"),
    ("Rifampicin", "RIF"),
    ("Clindamycin", "CLI"),
    ("Quinupristin-dalfopristin", "QDA"),
    ("Meropenem", "MEM"),
    ("Imipenem", "IPM"),
    ("Ertapenem", "ETP"),
    ("Aztreonam", "ATM"),
    ("Nalidixic acid", "NAL"),
    ("Gepotidacin", "GEP"),
]


def _agent_to_code(agent_name: str) -> str:
    """Convert EUCAST agent name to short code."""
    if not agent_name or not isinstance(agent_name, str):
        return "Unknown"
    name = agent_name.strip().split("(")[0].strip()
    for prefix, code in AGENT_NAME_TO_CODE:
        if name.startswith(prefix):
            return code
    # Fallback: first 3 letters uppercase from first word
    first = name.split()[0] if name else ""
    return first[:3].upper() if len(first) >= 2 else "UNK"


def _parse_zone_value(val) -> Optional[float]:
    """Parse S≥ or R< cell to a number. Returns None if not interpretable."""
    if val is None or (isinstance(val, float) and (val != val or val == 0)):
        return None
    s = str(val).strip().upper()
    if not s or s in ("-", "IE", "NAN", "NOTE", "ATU"):
        return None
    # Remove note letters: "14A" -> 14, "(50)A,D" -> 50, "19-20" -> take first
    s = re.sub(r"\((\d+)\)?", r"\1", s)
    s = re.sub(r"[A-Z,].*", "", s)
    s = s.strip()
    # Range like "19-20" -> use first number for S≥, second for R<
    if "-" in s:
        parts = s.split("-")
        if len(parts) >= 2 and parts[0].strip().isdigit():
            return float(parts[0].strip())
        s = parts[0].strip()
    # Single number
    m = re.match(r"^[\d.]+", s)
    if m:
        return float(m.group(0))
    return None


def load_eucast_excel(
    path: Optional[str | Path] = None,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Tuple[float, float]]], List[str]]:
    """
    Load EUCAST breakpoint tables from Excel.
    Returns (breakpoints_dict, list of agent codes for dropdown).
    breakpoints_dict[(guideline, species)][agent_code] = (r_max, s_min) for zone interpretation.
    """
    try:
        import pandas as pd
    except ImportError:
        return {}, []

    if path is None:
        # Default: project data folder, then user Downloads
        base = Path(__file__).resolve().parent
        candidates = [
            base / "data" / "v_16.0__BreakpointTables.xlsx",
            base / "v_16.0__BreakpointTables.xlsx",
            Path.home() / "Downloads" / "v_16.0__BreakpointTables.xlsx",
            Path.home() / "Téléchargements" / "v_16.0__BreakpointTables.xlsx",
        ]
        for p in candidates:
            if p.is_file():
                path = p
                break
        else:
            return {}, []

    path = Path(path)
    if not path.is_file():
        return {}, []

    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return {}, []

    result: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    all_codes: set[str] = set()

    for sheet in EUCAST_SHEETS:
        if sheet not in xl.sheet_names:
            continue
        species = SHEET_TO_SPECIES.get(sheet)
        if not species:
            continue
        key = ("EUCAST", species)
        if key not in result:
            result[key] = {}

        df = pd.read_excel(path, sheet_name=sheet, header=None)
        # Col 0 = agent name, 5 = S≥, 6 = R<
        for i in range(9, len(df)):
            agent = df.iloc[i, 0]
            s_ge = df.iloc[i, 5]
            r_lt = df.iloc[i, 6]
            if pd.isna(agent) or not str(agent).strip():
                continue
            agent_str = str(agent).strip()
            if "MIC " in agent_str or "breakpoint" in agent_str or "Zone diameter" in agent_str:
                continue
            s_val = _parse_zone_value(s_ge)
            r_val = _parse_zone_value(r_lt)
            if s_val is None and r_val is None:
                continue
            # EUCAST: S ≥ s_min, R < r_lt means zone < r_lt is R, so r_max = r_lt - 1
            if s_val is not None and r_val is not None:
                r_max = r_val - 1  # largest zone that is R
                s_min = s_val
            elif s_val is not None:
                r_max = s_val - 1
                s_min = s_val
            elif r_val is not None:
                r_max = r_val - 1
                s_min = r_val
            else:
                continue
            code = _agent_to_code(agent_str)
            if code == "Unknown":
                continue
            # Prefer first occurrence; later rows may be variants (e.g. meningitis)
            if code not in result[key]:
                result[key][code] = (r_max, s_min)
                all_codes.add(code)

    codes_sorted = sorted(all_codes)
    return result, codes_sorted
