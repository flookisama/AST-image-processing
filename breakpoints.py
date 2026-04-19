"""
EUCAST/CLSI disk diffusion breakpoints (zone diameter mm) for S/I/R interpretation.
EUCAST data can be loaded from the official Excel file (see eucast_loader.py).

Format:
  EUCAST → (r_max, s_min)  : S if zone >= s_min; R if zone <= r_max; else I
  CLSI   → (s_min, i_max, r_max) : S if zone >= s_min; R if zone <= r_max; else I
"""
from pathlib import Path

BREAKPOINTS = {
    # ── Escherichia coli ────────────────────────────────────────────────────
    ("EUCAST", "Escherichia coli"): {
        "AMX": (13, 18),
        "AMC": (13, 18),
        "AMP": (13, 17),
        "TZP": (17, 22),
        "CXM": (14, 18),
        "CTX": (14, 23),
        "CRO": (14, 23),
        "CAZ": (14, 21),
        "FEP": (14, 21),
        "ETP": (18, 22),
        "IPM": (15, 22),
        "MEM": (15, 22),
        "ATM": (14, 21),
        "CIP": (19, 26),
        "CIP5": (19, 26),
        "LEV": (15, 23),
        "GEN": (12, 17),
        "TOB": (12, 17),
        "AMK": (12, 17),
        "SXT": (9, 14),
        "TMP": (11, 16),
        "FOS": (15, 24),
        "NIT": (20, 25),
        "NAL": (13, 19),
        "TGC": (18, 25),
    },
    ("CLSI", "Escherichia coli"): {
        "AMX": (14, 17, 18),
        "AMC": (14, 17, 18),
        "AMP": (13, 14, 17),
        "TZP": (17, 20, 21),
        "CTX": (22, 22, 26),
        "CRO": (22, 22, 26),
        "CAZ": (18, 20, 21),
        "FEP": (19, 22, 25),
        "IPM": (20, 23, 24),
        "MEM": (20, 23, 24),
        "CIP": (16, 21, 21),
        "CIP5": (16, 21, 21),
        "LEV": (13, 17, 17),
        "GEN": (12, 14, 15),
        "TOB": (12, 14, 15),
        "AMK": (14, 17, 17),
        "SXT": (10, 11, 16),
        "TZP": (17, 20, 21),
    },

    # ── Klebsiella pneumoniae ───────────────────────────────────────────────
    ("EUCAST", "Klebsiella pneumoniae"): {
        "AMC": (10, 19),
        "TZP": (17, 22),
        "CXM": (13, 18),
        "CTX": (14, 23),
        "CRO": (14, 23),
        "CAZ": (14, 21),
        "FEP": (14, 21),
        "ETP": (18, 22),
        "IPM": (15, 22),
        "MEM": (15, 22),
        "ATM": (14, 21),
        "CIP": (15, 26),
        "LEV": (15, 23),
        "GEN": (11, 17),
        "TOB": (11, 17),
        "AMK": (11, 17),
        "SXT": (9, 14),
        "FOS": (15, 24),
        "TGC": (18, 25),
    },
    ("CLSI", "Klebsiella pneumoniae"): {
        "AMC": (14, 17, 18),
        "TZP": (17, 20, 21),
        "CTX": (22, 22, 26),
        "CRO": (22, 22, 26),
        "CAZ": (18, 20, 21),
        "FEP": (19, 22, 25),
        "IPM": (20, 23, 24),
        "MEM": (20, 23, 24),
        "CIP": (16, 21, 21),
        "GEN": (12, 14, 15),
        "TOB": (12, 14, 15),
        "AMK": (14, 17, 17),
        "SXT": (10, 11, 16),
    },

    # ── Staphylococcus aureus ───────────────────────────────────────────────
    ("EUCAST", "Staphylococcus aureus"): {
        "PEN": (26, 26),
        "OXA": (19, 22),
        "FOX": (22, 26),
        "CIP": (17, 22),
        "CIP5": (17, 22),
        "LEV": (17, 22),
        "MXF": (20, 24),
        "ERY": (17, 22),
        "CLI": (19, 22),
        "SXT": (11, 16),
        "GEN": (14, 18),
        "TOB": (14, 18),
        "TET": (19, 25),
        "DOX": (20, 25),
        "VAN": (17, 17),
        "TEI": (14, 17),
        "LZD": (21, 21),
        "DAP": (22, 22),
        "RIF": (17, 20),
        "CHL": (21, 26),
        "MUP": (14, 18),
        "TGC": (19, 25),
        "FUS": (22, 27),
    },
    ("CLSI", "Staphylococcus aureus"): {
        "PEN": (28, 28, 29),
        "OXA": (10, 11, 13),
        "FOX": (22, 23, 24),
        "CIP": (15, 16, 21),
        "CIP5": (15, 16, 21),
        "ERY": (13, 14, 23),
        "CLI": (14, 15, 21),
        "SXT": (10, 11, 16),
        "GEN": (12, 13, 15),
        "TOB": (12, 13, 15),
        "TET": (14, 15, 19),
        "VAN": (15, 15, 17),
        "LZD": (21, 22, 25),
        "RIF": (16, 17, 20),
    },

    # ── Pseudomonas aeruginosa ──────────────────────────────────────────────
    ("EUCAST", "Pseudomonas aeruginosa"): {
        "TZP": (17, 22),
        "CAZ": (17, 22),
        "FEP": (17, 22),
        "IPM": (16, 20),
        "MEM": (16, 20),
        "ATM": (16, 21),
        "CIP": (17, 24),
        "CIP5": (17, 24),
        "LEV": (17, 21),
        "GEN": (13, 17),
        "TOB": (13, 17),
        "AMK": (15, 18),
        "COL": (11, 11),
    },
    ("CLSI", "Pseudomonas aeruginosa"): {
        "TZP": (14, 17, 21),
        "CAZ": (15, 18, 22),
        "FEP": (15, 18, 22),
        "IPM": (16, 19, 22),
        "MEM": (16, 19, 22),
        "ATM": (16, 19, 22),
        "CIP": (16, 21, 25),
        "GEN": (12, 13, 15),
        "TOB": (12, 13, 15),
        "AMK": (14, 17, 17),
    },

    # ── Streptococcus pneumoniae ────────────────────────────────────────────
    ("EUCAST", "Streptococcus pneumoniae"): {
        "PEN": (20, 24),
        "AMX": (20, 24),
        "AMC": (20, 24),
        "CTX": (25, 29),
        "CRO": (25, 29),
        "CIP": (19, 21),
        "LEV": (17, 20),
        "MXF": (18, 22),
        "ERY": (17, 22),
        "CLI": (16, 19),
        "SXT": (15, 19),
        "TET": (22, 26),
        "VAN": (17, 17),
        "LZD": (25, 25),
        "CHL": (24, 28),
        "RIF": (16, 19),
        "TGC": (25, 30),
    },
    ("CLSI", "Streptococcus pneumoniae"): {
        "PEN": (19, 21, 24),
        "AMX": (19, 21, 24),
        "CTX": (24, 25, 28),
        "CRO": (24, 25, 28),
        "CIP": (15, 16, 21),
        "ERY": (13, 14, 23),
        "CLI": (15, 16, 21),
        "SXT": (15, 16, 19),
        "TET": (18, 19, 22),
        "VAN": (17, 17, 17),
        "LZD": (21, 22, 25),
        "LEV": (16, 17, 20),
    },

    # ── Enterococcus faecalis ───────────────────────────────────────────────
    ("EUCAST", "Enterococcus faecalis"): {
        "AMX": (15, 19),
        "AMP": (15, 19),
        "GEN": (9, 10),   # High-level synergy screening (120 µg disk)
        "STR": (9, 10),   # High-level synergy screening (300 µg disk)
        "VAN": (16, 17),
        "TEI": (9, 14),
        "LZD": (24, 25),
        "DAP": (21, 22),
        "TET": (16, 22),
        "CLI": (15, 19),
        "CIP": (16, 22),
        "NIT": (20, 25),
        "FOS": (20, 26),
    },
    ("CLSI", "Enterococcus faecalis"): {
        "AMP": (16, 17, 17),
        "VAN": (15, 15, 17),
        "LZD": (21, 22, 25),
        "GEN": (6, 9, 10),
        "TET": (12, 15, 19),
        "CIP": (15, 16, 21),
    },

    # ── Enterococcus faecium ────────────────────────────────────────────────
    ("EUCAST", "Enterococcus faecium"): {
        "AMX": (15, 19),
        "AMP": (15, 19),
        "VAN": (16, 17),
        "TEI": (9, 14),
        "LZD": (24, 25),
        "DAP": (21, 22),
        "TET": (16, 22),
        "CIP": (16, 22),
    },
    ("CLSI", "Enterococcus faecium"): {
        "AMP": (16, 17, 17),
        "VAN": (15, 15, 17),
        "LZD": (21, 22, 25),
        "TET": (12, 15, 19),
    },

    # ── Haemophilus influenzae ──────────────────────────────────────────────
    ("EUCAST", "Haemophilus influenzae"): {
        "AMP": (12, 22),
        "AMX": (12, 22),
        "AMC": (23, 24),
        "CFM": (25, 31),
        "CTX": (25, 31),
        "CRO": (25, 31),
        "AZM": (17, 23),
        "CIP": (20, 33),
        "LEV": (20, 28),
        "SXT": (9, 16),
        "CHL": (24, 30),
        "TET": (23, 28),
    },
    ("CLSI", "Haemophilus influenzae"): {
        "AMP": (18, 19, 22),
        "AMC": (17, 18, 20),
        "CTX": (26, 27, 31),
        "CRO": (26, 27, 31),
        "AZM": (12, 13, 23),
        "CIP": (25, 26, 33),
        "SXT": (10, 11, 16),
        "CHL": (26, 27, 30),
    },

    # ── Streptococcus agalactiae (Group B) ──────────────────────────────────
    ("EUCAST", "Streptococcus agalactiae"): {
        "PEN": (14, 24),
        "AMP": (14, 24),
        "AMX": (14, 24),
        "AMC": (14, 24),
        "CTX": (14, 24),
        "CRO": (14, 24),
        "ERY": (14, 18),
        "CLI": (15, 19),
        "LZD": (20, 21),
        "VAN": (16, 17),
        "TET": (19, 26),
        "CHL": (24, 28),
    },
    ("CLSI", "Streptococcus agalactiae"): {
        "PEN": (25, 26, 26),
        "AMP": (25, 26, 26),
        "CTX": (25, 26, 28),
        "ERY": (13, 14, 21),
        "CLI": (15, 16, 19),
        "LZD": (21, 22, 25),
        "VAN": (17, 17, 17),
        "TET": (18, 19, 25),
    },

    # ── Acinetobacter baumannii ─────────────────────────────────────────────
    ("EUCAST", "Acinetobacter baumannii"): {
        "IPM": (16, 20),
        "MEM": (16, 20),
        "CAZ": (16, 18),
        "FEP": (16, 18),
        "TZP": (16, 18),
        "CIP": (16, 20),
        "LEV": (16, 20),
        "GEN": (12, 15),
        "TOB": (12, 15),
        "AMK": (14, 18),
        "COL": (11, 11),
        "SXT": (10, 14),
        "RIF": (17, 22),
        "TET": (14, 20),
        "TGC": (17, 23),
    },
    ("CLSI", "Acinetobacter baumannii"): {
        "IPM": (14, 16, 22),
        "MEM": (14, 16, 22),
        "CAZ": (14, 18, 22),
        "CIP": (15, 16, 21),
        "GEN": (12, 13, 15),
        "TOB": (12, 13, 15),
        "AMK": (14, 17, 17),
        "SXT": (10, 11, 16),
        "TET": (11, 12, 15),
        "TGC": (16, 17, 23),
    },

    # ── Salmonella spp. ─────────────────────────────────────────────────────
    ("EUCAST", "Salmonella spp."): {
        "AMP": (13, 17),
        "AMX": (13, 17),
        "AMC": (13, 17),
        "CTX": (14, 23),
        "CRO": (14, 23),
        "CAZ": (14, 21),
        "CIP": (19, 26),
        "CIP5": (19, 26),
        "LEV": (15, 23),
        "AZM": (12, 17),
        "GEN": (12, 17),
        "SXT": (9, 14),
        "CHL": (16, 21),
        "TET": (12, 17),
    },
    ("CLSI", "Salmonella spp."): {
        "AMP": (13, 14, 17),
        "AMC": (14, 17, 18),
        "CTX": (22, 22, 26),
        "CRO": (22, 22, 26),
        "CIP": (16, 21, 21),
        "GEN": (12, 14, 15),
        "SXT": (10, 11, 16),
        "CHL": (18, 19, 23),
        "TET": (11, 12, 15),
    },

    # ── Proteus mirabilis ───────────────────────────────────────────────────
    ("EUCAST", "Proteus mirabilis"): {
        "AMP": (13, 17),
        "AMX": (13, 17),
        "AMC": (13, 18),
        "TZP": (17, 22),
        "CTX": (14, 23),
        "CRO": (14, 23),
        "CAZ": (14, 21),
        "FEP": (14, 21),
        "ETP": (18, 22),
        "IPM": (15, 22),
        "MEM": (15, 22),
        "CIP": (15, 26),
        "LEV": (15, 23),
        "GEN": (12, 17),
        "TOB": (12, 17),
        "AMK": (12, 17),
        "SXT": (9, 14),
    },
}


def interpret_zone(guideline: str, species: str, antibiotic_code: str, zone_mm: float) -> str:
    """Return 'S', 'I', or 'R' for a given zone diameter (mm)."""
    table = get_breakpoints_table(guideline, species)
    if not table:
        return "?"
    bp = table.get(antibiotic_code) or table.get(antibiotic_code.rstrip("0123456789"))
    if bp is None:
        return "?"
    if len(bp) == 2:
        r_max, s_min = bp
        if zone_mm >= s_min:
            return "S"
        if zone_mm <= r_max:
            return "R"
        return "I"
    else:
        s_min, _i_max, r_max = bp
        if zone_mm >= s_min:
            return "S"
        if zone_mm <= r_max:
            return "R"
        return "I"


def get_breakpoint_info(guideline: str, species: str, antibiotic_code: str, zone_mm: float) -> dict:
    sir = interpret_zone(guideline, species, antibiotic_code, zone_mm)
    labels = {"S": "Susceptible", "I": "Intermediate", "R": "Resistant", "?": "Unknown"}
    return {"sir": sir, "label": labels[sir], "zone_mm": zone_mm}


_EUCAST_FROM_EXCEL: dict = {}
_ANTIBIOTIC_CODES_FROM_EXCEL: list = []
_load_tried: bool = False


def _ensure_breakpoints_loaded() -> None:
    global _EUCAST_FROM_EXCEL, _ANTIBIOTIC_CODES_FROM_EXCEL, _load_tried
    if _load_tried:
        return
    _load_tried = True
    try:
        from eucast_loader import load_eucast_excel
        loaded, codes = load_eucast_excel()
        if loaded:
            _EUCAST_FROM_EXCEL = loaded
            _ANTIBIOTIC_CODES_FROM_EXCEL = sorted(set(codes))
    except Exception:
        pass


# ── Derived breakpoints (from TSV database) ────────────────────────────────────
_DERIVED: dict = {}
_NAME_MAP: dict = {}          # normalised full name → canonical code
_DERIVED_LOAD_TRIED: bool = False

_DATA_DIR = Path(__file__).parent / "data"

# Common antibiotic full name → code table (supplement derived map)
_BUILTIN_NAME_TO_CODE = {
    "ampicillin": "AMP", "amoxicillin": "AMX", "amoxicillin-clavulanate": "AMC",
    "piperacillin-tazobactam": "TZP", "oxacillin": "OXA", "cloxacillin": "CLX",
    "cefoxitin": "FOX", "cefuroxime": "CXM", "cefotaxime": "CTX",
    "ceftriaxone": "CRO", "ceftazidime": "CAZ", "cefepime": "FEP",
    "aztreonam": "ATM", "imipenem": "IPM", "meropenem": "MEM",
    "ertapenem": "ETP", "doripenem": "DOR",
    "ciprofloxacin": "CIP", "levofloxacin": "LEV", "norfloxacin": "NOR",
    "moxifloxacin": "MXF", "ofloxacin": "OFX", "nalidixic acid": "NAL",
    "gentamicin": "GEN", "tobramycin": "TOB", "amikacin": "AMK",
    "netilmicin": "NET", "streptomycin": "STR",
    "trimethoprim-sulfamethoxazole": "SXT", "trimethoprim": "TMP",
    "chloramphenicol": "CHL", "tetracycline": "TET", "doxycycline": "DOX",
    "minocycline": "MIN", "tigecycline": "TGC",
    "vancomycin": "VAN", "teicoplanin": "TEI", "linezolid": "LZD",
    "daptomycin": "DAP", "rifampicin": "RIF", "rifampin": "RIF",
    "fosfomycin": "FOS", "nitrofurantoin": "NIT", "colistin": "COL",
    "polymyxin b": "PMB", "clindamycin": "CLI", "erythromycin": "ERY",
    "azithromycin": "AZM", "clarithromycin": "CLA", "penicillin": "PEN",
    "sulfamethoxazole": "SXT", "cefazolin": "CFZ", "cephalothin": "CFL",
    "piperacillin": "PIP", "ticarcillin": "TIC",
}


def _norm_name(n: str) -> str:
    import re, unicodedata
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower().strip()
    n = re.sub(r"\s*[-/]\s*\d+.*$", "", n)
    return re.sub(r"\s+", " ", n)


def _load_derived() -> None:
    global _DERIVED, _NAME_MAP, _DERIVED_LOAD_TRIED
    if _DERIVED_LOAD_TRIED:
        return
    _DERIVED_LOAD_TRIED = True
    try:
        import json
        bp_path = _DATA_DIR / "derived_breakpoints.json"
        if bp_path.is_file():
            with open(bp_path) as f:
                _DERIVED = json.load(f)
        nm_path = _DATA_DIR / "antibiotic_name_map.json"
        if nm_path.is_file():
            with open(nm_path) as f:
                raw_map = json.load(f)
            # normalised_name → code  (use builtin code if known, else keep raw)
            for norm_name, raw_name in raw_map.items():
                code = _BUILTIN_NAME_TO_CODE.get(norm_name) or _BUILTIN_NAME_TO_CODE.get(_norm_name(raw_name))
                if code:
                    _NAME_MAP[norm_name] = code
    except Exception:
        pass
    # Always ensure builtin map is present
    for full, code in _BUILTIN_NAME_TO_CODE.items():
        _NAME_MAP.setdefault(full, code)


def _derived_breakpoints_for(guideline: str, species: str) -> dict:
    """
    Convert derived breakpoints (from TSV) to the same (r_max, s_min) tuple format.
    Searches for exact species match, then prefix match.
    """
    _load_derived()
    if not _DERIVED:
        return {}

    # Find species key
    sp_data = _DERIVED.get(species)
    if sp_data is None:
        sp_lower = species.lower()
        for key in _DERIVED:
            if key.lower().startswith(sp_lower[:12]) or sp_lower.startswith(key.lower()[:12]):
                sp_data = _DERIVED[key]
                break
    if sp_data is None:
        return {}

    result = {}
    for abx_full, std_dict in sp_data.items():
        # Map full name → code
        code = (_NAME_MAP.get(_norm_name(abx_full))
                or _BUILTIN_NAME_TO_CODE.get(_norm_name(abx_full)))
        if not code:
            continue
        bp_data = std_dict.get(guideline) or std_dict.get(next(iter(std_dict), ""))
        if not bp_data:
            continue
        r_max = bp_data.get("r_max")
        s_min = bp_data.get("s_min")
        if r_max is not None and s_min is not None:
            result[code] = (int(r_max), int(s_min))
    return result


def full_name_to_code(name: str) -> str:
    """Convert a full antibiotic name to its short code (e.g. 'ampicillin' → 'AMP')."""
    _load_derived()
    norm = _norm_name(name)
    return _NAME_MAP.get(norm, "").upper() or _BUILTIN_NAME_TO_CODE.get(norm, "").upper()


def get_breakpoints_table(guideline: str, species: str) -> dict:
    """Return breakpoints dict for (guideline, species), merging all data sources."""
    _ensure_breakpoints_loaded()
    key = (guideline, species)
    # Start with derived (lowest priority)
    table = _derived_breakpoints_for(guideline, species)
    # Override with hard-coded curated values
    table.update(BREAKPOINTS.get(key, {}))
    # Override with EUCAST Excel (highest priority)
    if guideline == "EUCAST" and key in _EUCAST_FROM_EXCEL:
        table.update(_EUCAST_FROM_EXCEL[key])
    return table


def get_antibiotic_options(guideline: str, species: str) -> list:
    table = get_breakpoints_table(guideline, species)
    codes = sorted(table.keys())
    if not codes and guideline == "EUCAST":
        _ensure_breakpoints_loaded()
        codes = _ANTIBIOTIC_CODES_FROM_EXCEL
    if not codes:
        codes = ["AMX", "AMC", "AMP", "CIP", "CTX", "CAZ", "GEN", "SXT", "TZP", "OXA", "PEN", "ERY", "TET"]
    return ["Unknown"] + codes


def get_all_species() -> list:
    """All species with breakpoints, including derived ones."""
    _load_derived()
    hardcoded = {s for (_, s) in BREAKPOINTS}
    derived = set(_DERIVED.keys()) if _DERIVED else set()
    return sorted(hardcoded | derived)


SPECIES_LIST = sorted({s for (_, s) in BREAKPOINTS})
