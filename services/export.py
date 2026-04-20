"""
Export service — CSV and PDF generation.
Moved from export.py; uses AnalysisResult for richer output.
"""
from __future__ import annotations

import io
import re
import datetime
from typing import List, Dict, Any

import pandas as pd


def to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    to_dataframe(rows).to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue().encode("utf-8")


def _sir_color(sir: str) -> tuple:
    return {"S": (0, 160, 0), "I": (200, 130, 0), "R": (200, 0, 0)}.get(sir, (80, 80, 80))


def to_pdf_bytes(
    rows: List[Dict[str, Any]],
    species: str = "",
    guideline: str = "",
    title: str = "Antibiogram Report",
) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 is required — pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(80, 80, 80)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = "  |  ".join(filter(None, [
        f"Species: {species}" if species else "",
        f"Guideline: {guideline}" if guideline else "",
        f"Generated: {now}",
    ]))
    pdf.cell(0, 6, meta, ln=True, align="C")
    pdf.ln(4)

    col_widths = {"Disk": 15, "Label": 40, "Antibiotic": 45, "Zone (mm)": 28, "Interpretation": 52}
    cols = list(col_widths.keys())

    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    for col in cols:
        pdf.cell(col_widths[col], 8, col, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", size=10)
    for i, row in enumerate(rows):
        fill_color = (245, 248, 255) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_color)

        interp_text = str(row.get("Interpretation", "—"))
        sir = next((s for s in ("S", "I", "R") if f" {s}" in interp_text or interp_text.endswith(s)), "")

        pdf.set_text_color(60, 60, 60)
        pdf.cell(col_widths["Disk"], 7, str(row.get("Disk", "")), border=1, fill=True, align="C")
        pdf.cell(col_widths["Label"], 7, str(row.get("Label", ""))[:22], border=1, fill=True)
        pdf.cell(col_widths["Antibiotic"], 7, str(row.get("Antibiotic", ""))[:25], border=1, fill=True)
        pdf.cell(col_widths["Zone (mm)"], 7, str(row.get("Zone (mm)", "—")), border=1, fill=True, align="C")

        r, g, b = _sir_color(sir)
        pdf.set_text_color(r, g, b)
        clean = re.sub(r"[^\x00-\x7F]", "", interp_text).strip()
        pdf.cell(col_widths["Interpretation"], 7, clean[:25], border=1, fill=True, align="C")
        pdf.set_text_color(60, 60, 60)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "S = Susceptible   I = Intermediate   R = Resistant", ln=True)
    pdf.cell(0, 5, f"Interpretation per {guideline or 'EUCAST/CLSI'} breakpoints.", ln=True)

    return bytes(pdf.output())
