"""
EUCAST/CLSI disk diffusion breakpoints (zone diameter mm) for S/I/R interpretation.
EUCAST data can be loaded from the official Excel file (see eucast_loader.py).
"""

# Species key: (guideline, species_name)
# Antibiotic key: code (e.g. AMX, CIP)
# Value: (r_max, s_min) for EUCAST → S if zone >= s_min, R if zone <= r_max, else I
#        or (s_min, i_max, r_max) for CLSI 3-tier

# Built-in fallback when Excel is not available
BREAKPOINTS = {
    ("EUCAST", "Escherichia coli"): {
        "AMX": (13, 18),   # S>=18, R<13 (so I: 13-17)
        "AMC": (13, 18),   # amoxicillin-clavulanate
        "CIP": (19, 25),   # S>=25, R<19
        "CIP5": (19, 25),
        "CTX": (19, 25),   # cefotaxime
        "CAZ": (19, 25),   # ceftazidime
        "GEN": (14, 18),   # gentamicin
        "TMP": (11, 16),   # trimethoprim (SXT component)
        "SXT": (11, 16),   # trimethoprim-sulfamethoxazole
        "FOS": (16, 22),   # fosfomycin
        "NAL": (13, 19),   # nalidixic acid
        "TZP": (19, 25),   # piperacillin-tazobactam
    },
    ("EUCAST", "Staphylococcus aureus"): {
        "PEN": (26, 26),   # penicillin S>=26
        "OXA": (19, 22),   # oxacillin (methicillin)
        "CIP": (19, 22),
        "CIP5": (19, 22),
        "ERY": (21, 25),
        "SXT": (11, 16),
        "GEN": (14, 18),
        "VAN": (17, 17),   # vancomycin - no disk diffusion in EUCAST
        "LZD": (21, 21),   # linezolid
        "TET": (19, 22),   # tetracycline
    },
    ("EUCAST", "Pseudomonas aeruginosa"): {
        "CAZ": (19, 25),
        "CIP": (19, 25),
        "CIP5": (19, 25),
        "GEN": (13, 17),
        "TZP": (19, 25),
        "MEM": (19, 22),   # meropenem
        "IPM": (19, 22),   # imipenem
    },
    ("EUCAST", "Streptococcus pneumoniae"): {
        "PEN": (20, 24),   # meningitis: different; here non-meningitis
        "AMX": (20, 24),
        "AMC": (20, 24),
        "CTX": (25, 29),
        "CIP": (19, 21),
        "ERY": (21, 24),
        "SXT": (15, 19),
        "VAN": (17, 17),
        "LEV": (17, 20),   # levofloxacin
    },
    ("CLSI", "Escherichia coli"): {
        "AMX": (14, 17, 18),
        "AMC": (14, 17, 18),
        "CIP": (21, 16, 20),
        "CIP5": (21, 16, 20),
        "CTX": (23, 20, 22),
        "CAZ": (21, 18, 20),
        "GEN": (15, 13, 14),
        "SXT": (16, 11, 10),
        "TZP": (21, 18, 20),
    },
    ("CLSI", "Staphylococcus aureus"): {
        "PEN": (29, 28, 27),
        "OXA": (22, 21, 19),
        "CIP": (21, 16, 15),
        "CIP5": (21, 16, 15),
        "ERY": (23, 14, 13),
        "SXT": (16, 11, 10),
        "GEN": (15, 13, 12),
        "TET": (19, 15, 14),
    },
    ("CLSI", "Pseudomonas aeruginosa"): {
        "CAZ": (22, 19, 18),
        "CIP": (25, 20, 19),
        "CIP5": (25, 20, 19),
        "GEN": (15, 13, 12),
        "TZP": (21, 18, 17),
        "MEM": (22, 19, 18),
    },
    ("CLSI", "Streptococcus pneumoniae"): {
        "PEN": (24, 21, 19),
        "AMX": (24, 21, 19),
        "AMC": (24, 21, 19),
        "CTX": (28, 26, 24),
        "CIP": (21, 16, 15),
        "ERY": (23, 14, 13),
        "SXT": (19, 16, 15),
        "VAN": (17, 17, 17),
        "LEV": (20, 17, 16),
    },
}


def interpret_zone(guideline: str, species: str, antibiotic_code: str, zone_mm: float) -> str:
    """
    Return "S", "I", or "R" for a given zone diameter (mm).
    antibiotic_code: e.g. "AMX", "CIP5", "AMC".
    """
    ab = get_breakpoints_table(guideline, species)
    if not ab:
        return "?"
    # Try exact code then without dose
    bp = (
        ab.get(antibiotic_code)
        or ab.get(antibiotic_code.rstrip("0123456789"))
        or None
    )
    if bp is None:
        return "?"
    if len(bp) == 2:
        # EUCAST style: (R_max_below, S_min) → S>=S_min, R<=R_max, I between
        r_max, s_min = bp
        if zone_mm >= s_min:
            return "S"
        if zone_mm <= r_max:
            return "R"
        return "I"
    else:
        # CLSI style: (S_min, I_max, R_max) → S>=S_min, I: R_max < z < S_min, R<=R_max
        s_min, i_max, r_max = bp
        if zone_mm >= s_min:
            return "S"
        if zone_mm <= r_max:
            return "R"
        return "I"


def get_breakpoint_info(guideline: str, species: str, antibiotic_code: str, zone_mm: float) -> dict:
    """Return interpretation and optional text."""
    sir = interpret_zone(guideline, species, antibiotic_code, zone_mm)
    labels = {"S": "Susceptible", "I": "Intermediate", "R": "Resistant", "?": "Unknown"}
    return {"sir": sir, "label": labels[sir], "zone_mm": zone_mm}


# Load EUCAST from Excel when available (e.g. v_16.0__BreakpointTables.xlsx in project or Downloads)
_EUCAST_FROM_EXCEL: dict = {}
_ANTIBIOTIC_CODES_FROM_EXCEL: list = []
_load_tried: bool = False


def _ensure_breakpoints_loaded():
    """Load EUCAST breakpoints from Excel once; merge into get_breakpoints_table."""
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


def get_breakpoints_table(guideline: str, species: str) -> dict:
    """Return the breakpoints dict for (guideline, species), with EUCAST Excel data merged when available."""
    _ensure_breakpoints_loaded()
    key = (guideline, species)
    table = dict(BREAKPOINTS.get(key, {}))
    if guideline == "EUCAST" and key in _EUCAST_FROM_EXCEL:
        table.update(_EUCAST_FROM_EXCEL[key])
    return table


def get_antibiotic_options(guideline: str, species: str) -> list:
    """Return list of antibiotic codes for the dropdown (Unknown + codes from table)."""
    table = get_breakpoints_table(guideline, species)
    codes = sorted(table.keys())
    if not codes and guideline == "EUCAST":
        _ensure_breakpoints_loaded()
        codes = _ANTIBIOTIC_CODES_FROM_EXCEL
    if not codes:
        codes = ["AMX", "AMC", "CIP", "CIP5", "CTX", "CAZ", "GEN", "SXT", "TZP", "OXA", "PEN", "ERY", "TET"]
    return ["Unknown"] + codes
