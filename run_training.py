#!/usr/bin/env python3
"""
run_training.py — Pipeline d'entraînement complet en ligne de commande.

Usage:
    python3 run_training.py --images <dossier_images> --tables <dossier_tables>

Le script :
  1. Fait le matching automatique image ↔ Word par convention de nommage
  2. Parse tous les fichiers Word pour extraire les zones de référence
  3. Lance l'auto-calibration par grille de recherche
  4. Affiche les métriques et sauvegarde les meilleurs paramètres
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement antibiogramme")
    parser.add_argument("--images", required=True, help="Dossier contenant les images (.jpg/.png)")
    parser.add_argument("--tables", required=True, help="Dossier contenant les fichiers Word (.docx)")
    parser.add_argument("--output", default="data/calibration.json", help="Fichier de calibration à écrire")
    parser.add_argument("--dry-run", action="store_true", help="Parsing seulement, sans calibration")
    parser.add_argument("--max-size", type=int, default=512, help="Redimensionner les images à max N px (défaut: 512)")
    parser.add_argument("--max-samples", type=int, default=0, help="Limiter à N échantillons (0 = tous)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    tables_dir = Path(args.tables)

    if not images_dir.is_dir():
        print(f"❌ Dossier images introuvable : {images_dir}")
        sys.exit(1)
    if not tables_dir.is_dir():
        print(f"❌ Dossier tables introuvable : {tables_dir}")
        sys.exit(1)

    # ── 1. Lister les fichiers ────────────────────────────────────────────────
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".bmp")
    )
    word_files = sorted(f for f in tables_dir.iterdir() if f.suffix.lower() == ".docx")

    print(f"\n{'='*60}")
    print(f"  PIPELINE D'ENTRAÎNEMENT ANTIBIOGRAMME")
    print(f"{'='*60}")
    print(f"  Images trouvées : {len(image_files)}")
    print(f"  Fichiers Word   : {len(word_files)}")

    if not image_files:
        print("❌ Aucune image trouvée.")
        sys.exit(1)
    if not word_files:
        print("❌ Aucun fichier Word trouvé.")
        sys.exit(1)

    # ── 2. Matching automatique par convention de nommage ────────────────────
    # Ex: "1.1.1. original.jpg" ↔ "Table 1.1.1..docx"
    print(f"\n{'─'*60}")
    print("  ÉTAPE 1 — Association image ↔ tableau Word")
    print(f"{'─'*60}")

    def _extract_id(fname: str) -> str:
        """Extrait l'identifiant numérique commun (ex: '1.1.1')."""
        import re
        m = re.search(r"(\d+\.\d+\.?\d*)", fname)
        return m.group(1).rstrip(".") if m else ""

    word_by_id = {}
    for wf in word_files:
        wid = _extract_id(wf.name)
        if wid:
            word_by_id[wid] = wf

    pairs = []
    unmatched_imgs = []
    for img_f in image_files:
        iid = _extract_id(img_f.name)
        if iid and iid in word_by_id:
            pairs.append((img_f, word_by_id[iid]))
        else:
            unmatched_imgs.append(img_f.name)

    print(f"  ✅ Paires trouvées    : {len(pairs)}")
    if unmatched_imgs:
        print(f"  ⚠️  Images sans match : {unmatched_imgs}")

    if not pairs:
        print("❌ Aucune paire valide. Vérifiez la convention de nommage.")
        sys.exit(1)

    # ── 3. Parser les fichiers Word ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ÉTAPE 2 — Parsing des tableaux Word")
    print(f"{'─'*60}")

    from word_importer import parse_word_file

    gt_records = []
    for img_f, word_f in pairs:
        try:
            recs = parse_word_file(word_f)
            if recs:
                rec = recs[0].to_dict()
                rec["image_name"] = img_f.name
                rec["_image_path"] = str(img_f)
                zones = [e["zone_mm"] for e in rec.get("entries", []) if e.get("zone_mm") is not None]
                antibiotics = [e["antibiotic"] for e in rec.get("entries", []) if e.get("antibiotic")]
                print(f"  {img_f.name:<35} → {len(zones)} zones  {zones[:4]}")
                gt_records.append(rec)
            else:
                print(f"  ⚠️  {word_f.name} : aucune entrée extraite")
        except Exception as e:
            print(f"  ❌ {word_f.name} : {e}")

    # Sauvegarder la vérité terrain
    gt_path = Path("data/ground_truth.json")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_records, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Vérité terrain sauvegardée → {gt_path} ({len(gt_records)} fiches)")

    if args.dry_run:
        print("\n  Mode dry-run : arrêt avant calibration.")
        return

    # ── 4. Charger les images ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ÉTAPE 3 — Chargement des images")
    print(f"{'─'*60}")

    from trainer import TrainingSample, samples_from_ground_truth

    max_size = args.max_size
    images_arrays = {}
    for rec in gt_records:
        img_path = rec.get("_image_path")
        if img_path and Path(img_path).is_file():
            try:
                img_pil = Image.open(img_path).convert("RGB")
                # Resize for speed while keeping aspect ratio
                if max(img_pil.size) > max_size:
                    img_pil.thumbnail((max_size, max_size), Image.LANCZOS)
                arr = np.array(img_pil)
                images_arrays[rec["image_name"]] = arr
                h, w = arr.shape[:2]
                print(f"  {rec['image_name']:<35} {w}×{h}px (redimensionné)")
            except Exception as e:
                print(f"  ❌ {rec['image_name']} : {e}")

    samples = samples_from_ground_truth(gt_records, images_arrays)
    # Optional: limit number of samples
    if args.max_samples and args.max_samples < len(samples):
        import random; random.seed(42)
        samples = random.sample(samples, args.max_samples)
        print(f"\n  ✅ {len(samples)} échantillon(s) sélectionnés (sur {len(samples)} disponibles)")
    else:
        print(f"\n  ✅ {len(samples)} échantillon(s) prêt(s) pour l'entraînement")

    if not samples:
        print("❌ Aucun échantillon valide (vérifiez que les zones Word ne sont pas toutes nulles).")
        sys.exit(1)

    # ── 5. Auto-calibration ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ÉTAPE 4 — Auto-calibration (grille de recherche)")
    print(f"{'─'*60}")

    from trainer import auto_calibrate, PARAM2_GRID, CLAHE_CLIP_GRID, THRESHOLD_GRID
    n_combos = len(PARAM2_GRID) * len(CLAHE_CLIP_GRID) + len(THRESHOLD_GRID)
    print(f"  Combinaisons à tester : {n_combos} × {len(samples)} images")
    print(f"  Paramètres : param2={PARAM2_GRID}, clahe={CLAHE_CLIP_GRID}")
    print(f"  Seuil zone : {THRESHOLD_GRID}")

    _last_msg = [""]
    t0 = time.time()

    def progress(frac, msg):
        elapsed = time.time() - t0
        bar_len = 40
        filled = int(bar_len * frac)
        bar = "█" * filled + "░" * (bar_len - filled)
        eta = (elapsed / frac * (1 - frac)) if frac > 0.01 else 0
        print(f"\r  [{bar}] {frac*100:5.1f}%  {msg[:50]:<50}  ETA {eta:.0f}s", end="", flush=True)

    result = auto_calibrate(samples, progress_cb=progress)
    print()  # newline after progress bar

    elapsed = time.time() - t0
    print(f"\n  Durée : {elapsed:.1f}s")

    # ── 6. Résultats ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  RÉSULTATS")
    print(f"{'─'*60}")
    print(f"  MAE baseline (paramètres actuels) : {result.baseline_mae:.2f} mm")
    print(f"  MAE après calibration             : {result.best_mae:.2f} mm")
    print(f"  Amélioration                      : {result.improvement_pct:+.1f}%")

    print(f"\n  Meilleurs paramètres :")
    for k in ["hough_param2", "clahe_clip", "zone_threshold_ratio"]:
        print(f"    {k:<30} = {result.best_params[k]}")

    print(f"\n  Grille de recherche (top 5) :")
    print(result.grid_df.head(5).to_string(index=False))

    if not result.per_sample_df.empty:
        print(f"\n  Détail par échantillon :")
        print(result.per_sample_df.to_string(index=False))

    # ── 7. Sauvegarder la calibration ─────────────────────────────────────────
    from antibiogram_processor import save_params
    save_params(result.best_params)
    print(f"\n  ✅ Calibration sauvegardée → {args.output}")
    print(f"{'='*60}\n")

    # Aussi sauvegarder rapport JSON
    report = {
        "baseline_mae": result.baseline_mae,
        "best_mae": result.best_mae,
        "improvement_pct": result.improvement_pct,
        "best_params": result.best_params,
        "n_samples": len(samples),
    }
    report_path = Path("data/training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  📄 Rapport → {report_path}")


if __name__ == "__main__":
    main()
