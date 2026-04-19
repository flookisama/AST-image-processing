"""
ML Models page — train the disk classifier and test label recognition.
"""
import io
import json
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Modèles ML", layout="wide", page_icon="🤖")
st.title("🤖 Modèles ML")
st.markdown(
    "Entraînez le détecteur de disques par apprentissage automatique "
    "et testez la reconnaissance automatique des étiquettes d'antibiotiques."
)

MODEL_PATH = Path(__file__).parent.parent / "data" / "disk_classifier.pkl"
GT_PATH    = Path(__file__).parent.parent / "data" / "ground_truth.json"

tab_detect, tab_ocr = st.tabs([
    "🔬 1. Détecteur de disques ML",
    "🏷️ 2. Reconnaissance des étiquettes",
])

# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.subheader("Détecteur de disques HOG + SVM")
    st.markdown(
        "Ce modèle apprend à distinguer les disques antibiotiques des faux positifs "
        "(reflets, bulles, artefacts). Il filtre les détections Hough après entraînement."
    )

    # ── Status ────────────────────────────────────────────────────────────────
    if MODEL_PATH.is_file():
        st.success(f"✅ Modèle entraîné disponible — `{MODEL_PATH.name}`  "
                   f"({MODEL_PATH.stat().st_size // 1024} KB)")
        if st.button("🗑️ Supprimer le modèle", key="del_clf"):
            MODEL_PATH.unlink()
            st.rerun()
    else:
        st.info("Aucun modèle entraîné.")

    # ── Upload pre-trained model ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Importer un modèle entraîné")
    st.caption("Si vous avez entraîné le modèle localement (train_ml_model.py), importez le fichier .pkl directement.")
    pkl_upload = st.file_uploader("disk_classifier.pkl", type=["pkl"], key="pkl_upload")
    if pkl_upload:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_bytes(pkl_upload.read())
        st.success(f"✅ Modèle importé ({MODEL_PATH.stat().st_size // 1024} KB) — actif immédiatement.")
        st.rerun()

    st.markdown("---")

    # ── Upload images for training ─────────────────────────────────────────
    st.markdown("### Images d'entraînement")
    st.caption(
        "Les images sont traitées par le détecteur Hough calibré pour générer "
        "automatiquement des exemples positifs (disques) et négatifs (fond)."
    )

    uploaded = st.file_uploader(
        "Images de boîtes de Pétri",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=True,
        key="ml_train_imgs",
    )

    if "ml_images" not in st.session_state:
        st.session_state.ml_images = {}

    if uploaded:
        for f in uploaded:
            st.session_state.ml_images[f.name] = f.read()

    imgs = st.session_state.ml_images
    if imgs:
        st.info(f"{len(imgs)} image(s) disponible(s) pour l'entraînement.")
        col_clear, _ = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ Vider les images", key="clear_ml_imgs"):
                st.session_state.ml_images = {}
                st.rerun()

    # ── Train button ───────────────────────────────────────────────────────
    can_train = len(imgs) >= 3
    if not can_train:
        st.warning("Importez au moins 3 images pour entraîner le modèle.")

    if can_train and st.button("🚀 Entraîner le classifieur", type="primary", key="train_clf"):
        from ml_detector import DiskClassifier

        progress = st.progress(0.0)
        status   = st.empty()

        images_arrays = {}
        for name, raw in imgs.items():
            try:
                images_arrays[name] = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception as e:
                st.warning(f"Impossible de charger {name}: {e}")

        def _cb(frac, msg):
            progress.progress(min(frac, 1.0))
            status.caption(f"⏳ {msg}")

        with st.spinner("Génération des patches et entraînement SVM…"):
            clf = DiskClassifier()
            try:
                metrics = clf.fit_from_images(images_arrays, verbose=False, progress_cb=_cb)
                clf.save()
                progress.progress(1.0)
                status.empty()
                st.success(
                    f"✅ Modèle entraîné et sauvegardé !\n\n"
                    f"F1 score (validation croisée) : **{metrics['f1_mean']:.3f}** ± {metrics['f1_std']:.3f}  \n"
                    f"Patches positifs : {metrics['n_pos']}  |  Négatifs : {metrics['n_neg']}"
                )
            except Exception as e:
                st.error(f"Erreur d'entraînement : {e}")

    # ── Test on single image ───────────────────────────────────────────────
    if MODEL_PATH.is_file():
        st.markdown("---")
        st.markdown("### Tester le modèle")
        test_img = st.file_uploader(
            "Image de test", type=["jpg","jpeg","png","tiff","bmp"], key="ml_test_img"
        )
        if test_img:
            arr = np.array(Image.open(test_img).convert("RGB"))
            col_orig, col_ml = st.columns(2)

            with col_orig:
                from antibiogram_processor import find_disks, draw_results, process_antibiogram
                # Hough only
                with st.spinner("Détection Hough…"):
                    disks_hough, scale = process_antibiogram.__wrapped__(arr) if hasattr(process_antibiogram, '__wrapped__') else process_antibiogram(arr)
                annotated_hough = draw_results(arr, disks_hough, scale)
                st.image(annotated_hough, caption=f"Hough seul : {len(disks_hough)} disque(s)", use_container_width=True)

            with col_ml:
                from ml_detector import DiskClassifier
                from antibiogram_processor import find_disks as _find, _to_gray_uint8
                with st.spinner("Détection ML…"):
                    clf = DiskClassifier.load()
                    candidates = _find(arr, use_ml=False)
                    filtered   = clf.filter(arr, candidates)
                n_removed = len(candidates) - len(filtered)
                st.info(f"ML a supprimé **{n_removed}** faux positif(s) parmi {len(candidates)} candidats.")

                # Re-run full pipeline with ML-filtered result (show side effect)
                with st.spinner("Pipeline complet avec ML…"):
                    from antibiogram_processor import process_antibiogram
                    disks_ml, scale_ml = process_antibiogram(arr)
                annotated_ml = draw_results(arr, disks_ml, scale_ml)
                st.image(annotated_ml, caption=f"Hough + ML : {len(disks_ml)} disque(s)", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_ocr:
    st.subheader("Reconnaissance des étiquettes d'antibiotiques")
    st.markdown(
        "Détecte automatiquement les codes imprimés sur les disques "
        "(ex : **AMX**, **CIP**, **GEN**…) en utilisant EasyOCR."
    )

    from label_recognizer import _get_easyocr, _known_codes
    reader = _get_easyocr()
    if reader is not None:
        st.success("✅ EasyOCR chargé — reconnaissance haute précision disponible.")
    else:
        st.warning("⚠️ EasyOCR non disponible. Tesseract sera utilisé en fallback.")

    st.markdown(f"**{len(_known_codes())} codes d'antibiotiques** connus dans la base de référence.")

    st.markdown("---")
    st.markdown("### Tester sur une image")

    test_img2 = st.file_uploader(
        "Image antibiogramme",
        type=["jpg","jpeg","png","tiff","bmp"],
        key="ocr_test_img",
    )

    if test_img2:
        arr2 = np.array(Image.open(test_img2).convert("RGB"))

        with st.spinner("Détection des disques et reconnaissance des étiquettes…"):
            from antibiogram_processor import process_antibiogram
            from label_recognizer import recognize_all_disks
            disks2, scale2 = process_antibiogram(arr2)
            if disks2:
                labels = recognize_all_disks(arr2, disks2)

        if not disks2:
            st.warning("Aucun disque détecté dans cette image.")
        else:
            st.success(f"{len(disks2)} disque(s) détecté(s).")

            cols = st.columns(min(len(disks2), 4))
            for i, (disk, (code, conf)) in enumerate(zip(disks2, labels)):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    # Crop preview
                    h, w = arr2.shape[:2]
                    r = disk.radius_px
                    pad = int(r * 1.3)
                    x1 = max(0, int(disk.center[0]) - pad)
                    y1 = max(0, int(disk.center[1]) - pad)
                    x2 = min(w, int(disk.center[0]) + pad)
                    y2 = min(h, int(disk.center[1]) + pad)
                    crop = arr2[y1:y2, x1:x2]
                    if crop.size > 0:
                        thumb = Image.fromarray(crop).resize((100, 100), Image.LANCZOS)
                        st.image(np.array(thumb), use_container_width=False, width=100)

                    if code:
                        conf_icon = "🟢" if conf > 0.65 else "🟡" if conf > 0.35 else "🔴"
                        st.markdown(f"**{code}** {conf_icon} {conf:.0%}")
                    else:
                        st.markdown("*Non reconnu*")

            # Show full results table
            rows = []
            for i, (disk, (code, conf)) in enumerate(zip(disks2, labels)):
                rows.append({
                    "Disque": i + 1,
                    "Code reconnu": code or "—",
                    "Confiance OCR": f"{conf:.0%}",
                    "Zone (mm)": f"{disk.zone_diameter_mm:.1f}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Tester un crop de disque unique")
    st.caption("Collez ou importez une image recadrée autour d'un seul disque.")

    crop_file = st.file_uploader("Image du disque", type=["jpg","jpeg","png"], key="crop_test")
    if crop_file:
        crop_arr = np.array(Image.open(crop_file).convert("RGB"))
        h, w = crop_arr.shape[:2]
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2.5

        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(crop_arr, caption="Disque importé", use_container_width=True)
        with col_res:
            from label_recognizer import recognize_label
            code, conf = recognize_label(crop_arr, cx, cy, r)
            if code:
                st.markdown(f"### Code reconnu : **{code}**")
                st.markdown(f"Confiance : {conf:.0%}")
            else:
                st.markdown("*Aucun code reconnu*")
