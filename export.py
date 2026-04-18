"""
Export antibiogram results to CSV and PDF.
PDF generation requires fpdf2 (pip install fpdf2).
"""
from __future__ import annotations

import io
import datetime
from typing import List, Dict, Any

import pandas as pd


def results_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list-of-row dicts to a clean DataFrame."""
    return pd.DataFrame(rows)


def to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """Return UTF-8 CSV bytes from a list of result rows."""
    df = results_to_dataframe(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue().encode("utf-8")


# ── PDF export ────────────────────────────────────────────────────────────

def _sir_color(sir: str):
    """Return (R, G, B) tuple for a SIR value."""
    return {"S": (0, 160, 0), "I": (200, 130, 0), "R": (200, 0, 0)}.get(sir, (80, 80, 80))


def to_pdf_bytes(
    rows: List[Dict[str, Any]],
    species: str = "",
    guideline: str = "",
    title: str = "Antibiogram Report",
) -> bytes:
    """
    Generate a PDF report and return its bytes.
    Requires fpdf2 (pip install fpdf2).
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 is required for PDF export — pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # ── Title ──────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(2)

    # ── Metadata ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(80, 80, 80)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_parts = []
    if species:
        meta_parts.append(f"Species: {species}")
    if guideline:
        meta_parts.append(f"Guideline: {guideline}")
    meta_parts.append(f"Generated: {now}")
    pdf.cell(0, 6, "  |  ".join(meta_parts), ln=True, align="C")
    pdf.ln(4)

    # ── Table header ───────────────────────────────────────────────────────
    col_widths = {"Disk": 15, "Label": 40, "Antibiotic": 45, "Zone (mm)": 28, "Interpretation": 52}
    cols = list(col_widths.keys())

    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    for col in cols:
        pdf.cell(col_widths[col], 8, col, border=1, fill=True, align="C")
    pdf.ln()

    # ── Table rows ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", size=10)
    for i, row in enumerate(rows):
        fill_color = (245, 248, 255) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_color)

        # Extract SIR from interpretation string (may contain emoji prefix)
        interp_text = str(row.get("Interpretation", "—"))
        sir = ""
        for s in ("S", "I", "R"):
            if f" {s}" in interp_text or interp_text.endswith(s):
                sir = s
                break

        pdf.set_text_color(60, 60, 60)
        pdf.cell(col_widths["Disk"], 7, str(row.get("Disk", "")), border=1, fill=True, align="C")
        pdf.cell(col_widths["Label"], 7, str(row.get("Label", ""))[:22], border=1, fill=True)
        pdf.cell(col_widths["Antibiotic"], 7, str(row.get("Antibiotic", ""))[:25], border=1, fill=True)
        pdf.cell(col_widths["Zone (mm)"], 7, str(row.get("Zone (mm)", "—")), border=1, fill=True, align="C")

        # Color the interpretation cell
        r, g, b = _sir_color(sir)
        pdf.set_text_color(r, g, b)
        # Strip emoji characters for PDF (fpdf2 core fonts don't support them)
        clean_interp = re.sub(r"[^\x00-\x7F]", "", interp_text).strip()
        pdf.cell(col_widths["Interpretation"], 7, clean_interp[:25], border=1, fill=True, align="C")
        pdf.set_text_color(60, 60, 60)
        pdf.ln()

    # ── Legend ─────────────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "S = Susceptible   I = Intermediate   R = Resistant", ln=True)
    pdf.cell(0, 5, "Zone diameters measured by disk diffusion. Interpretation per " + (guideline or "EUCAST/CLSI") + " breakpoints.", ln=True)

    return bytes(pdf.output())


# ── re import for PDF (needed for emoji stripping) ────────────────────────
import re
