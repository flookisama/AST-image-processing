#!/usr/bin/env python3
"""
train_ml_model.py — Train the HOG+SVM disk classifier from antibiogram images.

Usage:
    python3 train_ml_model.py --images <folder>

The script:
  1. Loads all images in the folder
  2. Runs the calibrated Hough detector to find disk candidates (positives)
  3. Generates random background patches (negatives)
  4. Trains a HOG + SVM binary classifier
  5. Saves the model to data/disk_classifier.pkl
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Entraînement du classifieur de disques ML")
    parser.add_argument("--images", required=True, help="Dossier contenant les images (.jpg/.png)")
    parser.add_argument("--max-size", type=int, default=512, help="Redimensionner les images à max N px (défaut: 512)")
    parser.add_argument("--max-samples", type=int, default=0, help="Limiter à N images (0 = toutes)")
    parser.add_argument("--output", default="data/disk_classifier.pkl", help="Chemin du modèle à écrire")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"❌ Dossier introuvable : {images_dir}")
        sys.exit(1)

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".bmp")
    )
    if not image_files:
        print("❌ Aucune image trouvée.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT DU CLASSIFIEUR DE DISQUES ML")
    print(f"{'='*60}")
    print(f"  Images trouvées : {len(image_files)}")

    # Optional: limit samples
    if args.max_samples and args.max_samples < len(image_files):
        import random; random.seed(42)
        image_files = random.sample(image_files, args.max_samples)
        print(f"  Sous-ensemble   : {len(image_files)} images (--max-samples)")

    # ── Load images ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ÉTAPE 1 — Chargement et redimensionnement des images")
    print(f"{'─'*60}")

    max_size = args.max_size
    images_arrays = {}
    t0 = time.time()

    for i, img_path in enumerate(image_files):
        try:
            pil = Image.open(img_path).convert("RGB")
            if max(pil.size) > max_size:
                pil.thumbnail((max_size, max_size), Image.LANCZOS)
            arr = np.array(pil)
            images_arrays[img_path.name] = arr
            h, w = arr.shape[:2]
            print(f"  [{i+1:3d}/{len(image_files)}] {img_path.name:<40} {w}×{h}px")
        except Exception as e:
            print(f"  ❌ {img_path.name}: {e}")

    print(f"\n  ✅ {len(images_arrays)} image(s) chargée(s) en {time.time()-t0:.1f}s")

    # ── Generate patches & train ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ÉTAPE 2 — Génération des patches et entraînement")
    print(f"{'─'*60}")
    print(f"  (Hough → disques positifs + patches aléatoires négatifs)")

    from ml_detector import DiskClassifier, generate_training_patches

    t1 = time.time()
    try:
        X, y = generate_training_patches(images_arrays, verbose=True)
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération des patches: {e}")
        sys.exit(1)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"\n  Patches totaux : {len(X)}  (positifs={n_pos}, négatifs={n_neg})")

    print(f"\n{'─'*60}")
    print("  ÉTAPE 3 — Entraînement SVM")
    print(f"{'─'*60}")

    clf = DiskClassifier()
    metrics = clf.fit(X, y)

    print(f"  F1 score (cross-validation) : {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}")
    print(f"  Durée totale : {time.time()-t1:.1f}s")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(out_path)

    print(f"\n{'─'*60}")
    print(f"  ✅ Modèle sauvegardé → {out_path}  ({out_path.stat().st_size // 1024} KB)")
    print(f"{'='*60}\n")
    print("  Prochaine étape — pousser le modèle vers le serveur :")
    print(f"    git add {out_path}")
    print( "    git commit -m \"Add trained ML disk classifier\"")
    print( "    git push mygithub master")
    print()


if __name__ == "__main__":
    main()
