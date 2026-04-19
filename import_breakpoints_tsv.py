#!/usr/bin/env python3
"""
import_breakpoints_tsv.py — Derive S/I/R breakpoints from a large antibiogram TSV.

The TSV is expected to have (at minimum) these columns:
    species, antibiotic, phenotype, measurement_sign,
    measurement_value, measurement_units, typing_method, standard

Usage:
    python3 import_breakpoints_tsv.py --tsv <file.tsv>
    python3 import_breakpoints_tsv.py --tsv <file.tsv> --method disk_diffusion --units mm

Outputs:
    data/derived_breakpoints.json  — breakpoints per (species, antibiotic, standard)
    data/antibiotic_name_map.json  — full name → normalised code lookup
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ── Column name variants ────────────────────────────────────────────────────────
_COL_ALIASES = {
    "species":           ["species", "organism", "organism_name"],
    "antibiotic":        ["antibiotic", "antibiotic_name", "antimicrobial_agent", "compound"],
    "phenotype":         ["phenotype", "resistance_phenotype", "interpretation", "sir"],
    "measurement_sign":  ["measurement_sign", "sign", "measurement_operator"],
    "measurement_value": ["measurement_value", "value", "zone_diameter", "measurement"],
    "measurement_units": ["measurement_units", "units", "unit"],
    "typing_method":     ["typing_method", "method", "test_method"],
    "standard":          ["standard", "guideline", "testing_standard"],
}


def _resolve_columns(df: pd.DataFrame) -> dict:
    """Map canonical name → actual column name in df."""
    cols = {c.lower(): c for c in df.columns}
    resolved = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in cols:
                resolved[canonical] = cols[alias.lower()]
                break
    return resolved


# ── Phenotype normalisation ────────────────────────────────────────────────────
_PHENO_MAP = {
    "susceptible": "S", "s": "S",
    "intermediate": "I", "i": "I",
    "resistant": "R", "r": "R",
}


def _norm_pheno(v: str) -> str:
    return _PHENO_MAP.get(str(v).strip().lower(), "")


# ── Antibiotic name → code normalisation ──────────────────────────────────────
def _norm_abx_name(name: str) -> str:
    """Lowercase, strip trailing concentrations/units, collapse spaces."""
    import re
    n = str(name).lower().strip()
    n = re.sub(r"\s*[-/]\s*\d+.*$", "", n)  # e.g. "amoxicillin-clavulanate 2:1" → "amoxicillin"
    n = re.sub(r"\s+", " ", n)
    return n


# ── Breakpoint derivation ──────────────────────────────────────────────────────
def _derive_breakpoints(group: pd.DataFrame) -> dict | None:
    """
    Given a group (same species+antibiotic+standard), derive S_min and R_max
    from the zone diameter distribution.

    For disk diffusion:
      - Resistant entries should have SMALLER zones
      - Susceptible entries should have LARGER zones
      - R_max_zone = max zone value labelled R
      - S_min_zone = min zone value labelled S
    """
    s_vals = group.loc[group["_pheno"] == "S", "_value"].dropna()
    r_vals = group.loc[group["_pheno"] == "R", "_value"].dropna()
    i_vals = group.loc[group["_pheno"] == "I", "_value"].dropna()

    if s_vals.empty and r_vals.empty:
        return None

    result = {"n_S": len(s_vals), "n_I": len(i_vals), "n_R": len(r_vals)}

    if not s_vals.empty:
        result["s_min"] = round(float(s_vals.quantile(0.10)), 1)   # 10th pct of S = S floor
        result["s_mean"] = round(float(s_vals.mean()), 1)

    if not r_vals.empty:
        result["r_max"] = round(float(r_vals.quantile(0.90)), 1)   # 90th pct of R = R ceiling

    if not i_vals.empty:
        result["i_mean"] = round(float(i_vals.mean()), 1)

    # Infer missing boundary from the other
    if "s_min" not in result and "r_max" in result:
        result["s_min"] = result["r_max"] + 3   # rough gap
    if "r_max" not in result and "s_min" in result:
        result["r_max"] = result["s_min"] - 3

    result["r_max"] = max(1, result.get("r_max", 6))
    result["s_min"] = max(result["r_max"] + 1, result.get("s_min", result["r_max"] + 3))

    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dériver les breakpoints depuis un TSV antibiogramme")
    parser.add_argument("--tsv", required=True, help="Fichier TSV antibiogramme")
    parser.add_argument("--method", default="disk diffusion",
                        help="Méthode de test à conserver (défaut: 'disk diffusion')")
    parser.add_argument("--units", default="mm",
                        help="Unités à conserver (défaut: 'mm')")
    parser.add_argument("--min-n", type=int, default=5,
                        help="Nombre minimum d'entrées par groupe (défaut: 5)")
    parser.add_argument("--chunksize", type=int, default=200_000,
                        help="Lignes par chunk (défaut: 200000)")
    parser.add_argument("--output", default="data/derived_breakpoints.json")
    parser.add_argument("--name-map", default="data/antibiotic_name_map.json")
    args = parser.parse_args()

    tsv_path = Path(args.tsv)
    if not tsv_path.is_file():
        print(f"❌ Fichier introuvable : {tsv_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  DÉRIVATION DES BREAKPOINTS")
    print(f"{'='*60}")
    print(f"  Fichier : {tsv_path.name}")
    print(f"  Méthode : {args.method}  |  Unités : {args.units}")

    # ── Read in chunks (file can be huge) ─────────────────────────────────
    print(f"\n  Lecture du fichier par chunks de {args.chunksize:,} lignes…")
    chunks = []
    total_rows = 0
    col_map = None

    for i, chunk in enumerate(pd.read_csv(tsv_path, sep="\t", low_memory=False,
                                           chunksize=args.chunksize, on_bad_lines="skip")):
        if col_map is None:
            col_map = _resolve_columns(chunk)
            missing = [k for k in ["species","antibiotic","phenotype","measurement_value"] if k not in col_map]
            if missing:
                print(f"❌ Colonnes manquantes : {missing}")
                print(f"   Colonnes disponibles : {list(chunk.columns)[:15]}")
                sys.exit(1)
            print(f"  Colonnes résolues : {col_map}")

        # Rename to canonical
        chunk = chunk.rename(columns={v: k for k, v in col_map.items()})

        # Filter by method
        if "typing_method" in chunk.columns:
            mask = chunk["typing_method"].str.lower().str.contains(
                args.method.lower(), na=False
            )
            chunk = chunk[mask]

        # Filter by units
        if "measurement_units" in chunk.columns:
            chunk = chunk[chunk["measurement_units"].str.lower().str.strip() == args.units.lower()]

        # Filter by sign == exact (not <=/>= MIC breakpoints)
        if "measurement_sign" in chunk.columns:
            chunk = chunk[chunk["measurement_sign"].isin(["==", "="])]

        total_rows += len(chunk)
        if len(chunk):
            chunks.append(chunk[["species", "antibiotic", "phenotype",
                                  "measurement_value",
                                  *([c] for c in ["standard"] if c in chunk.columns)]].copy()
                          if "standard" not in chunk.columns
                          else chunk[["species", "antibiotic", "phenotype", "measurement_value", "standard"]].copy())
        print(f"  Chunk {i+1}: {total_rows:,} lignes conservées après filtrage", end="\r")

    print(f"\n  ✅ Total après filtrage : {total_rows:,} lignes")

    if not chunks:
        print("❌ Aucune ligne conservée après filtrage. Vérifiez --method et --units.")
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)

    # Normalise values
    df["_pheno"] = df["phenotype"].apply(_norm_pheno)
    df["_value"] = pd.to_numeric(df["measurement_value"], errors="coerce")
    df = df[df["_pheno"].isin(["S", "I", "R"]) & df["_value"].notna() & (df["_value"] > 0)]

    # Clean species & antibiotic
    df["species"] = df["species"].str.strip()
    df["antibiotic"] = df["antibiotic"].str.strip()

    print(f"  Lignes valides (phénotype+valeur OK) : {len(df):,}")

    # Build antibiotic name map (full name → normalised)
    abx_names = sorted(df["antibiotic"].dropna().unique())
    print(f"\n  Antibiotiques distincts : {len(abx_names)}")
    name_map = {_norm_abx_name(n): n for n in abx_names}
    # Invert: normalised_name → first seen raw name
    name_map_out = {norm: raw for norm, raw in name_map.items()}

    # ── Derive breakpoints ────────────────────────────────────────────────
    print(f"\n  Dérivation des breakpoints par groupe (espèce × antibiotique × standard)…")

    group_cols = ["species", "antibiotic"]
    if "standard" in df.columns:
        group_cols.append("standard")

    breakpoints: dict = {}
    groups = df.groupby(group_cols)
    n_groups = len(groups)
    n_kept = 0

    for keys, grp in groups:
        if isinstance(keys, str):
            keys = (keys,)
        species = keys[0]
        antibiotic = keys[1]
        standard = keys[2] if len(keys) > 2 else "unknown"

        if len(grp) < args.min_n:
            continue

        bp = _derive_breakpoints(grp)
        if bp is None:
            continue

        bp["antibiotic_raw"] = antibiotic
        if species not in breakpoints:
            breakpoints[species] = {}
        if antibiotic not in breakpoints[species]:
            breakpoints[species][antibiotic] = {}
        breakpoints[species][antibiotic][standard] = bp
        n_kept += 1

    print(f"  Groupes analysés : {n_groups:,}  |  Breakpoints dérivés : {n_kept:,}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  Espèces couvertes : {len(breakpoints)}")
    for sp, abx_dict in sorted(breakpoints.items())[:10]:
        print(f"    {sp:<45} {len(abx_dict)} antibiotiques")
    if len(breakpoints) > 10:
        print(f"    … et {len(breakpoints)-10} autres espèces")

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(breakpoints, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Breakpoints sauvegardés → {out}  ({out.stat().st_size // 1024} KB)")

    nm_out = Path(args.name_map)
    with open(nm_out, "w", encoding="utf-8") as f:
        json.dump(name_map_out, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Carte des noms      → {nm_out}  ({nm_out.stat().st_size // 1024} KB)")

    print(f"\n{'='*60}")
    print("  Prochaine étape — pousser les fichiers générés :")
    print(f"    git add {out} {nm_out}")
    print( "    git commit -m \"Add derived breakpoints from TSV database\"")
    print( "    git push mygithub master")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
