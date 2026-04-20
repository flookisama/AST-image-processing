"""
Calibration Page
Evaluate detection accuracy against ground truth, tune parameters, and save calibration.
"""
import io
import json
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title="Calibration", layout="wide", page_icon="⚙️")
st.title("⚙️ Calibration de la détection")
st.markdown(
    "Évaluez la précision de la détection sur vos images de référence. "
    "Ajustez les paramètres et sauvegardez la configuration optimale."
)

GROUND_TRUTH_PATH = Path(__file__).parent.parent / "data" / "ground_truth.json"
CONFIG_PATH = Path(__file__).parent.parent / "data" / "calibration.json"

from antibiogram_processor import (
    load_params,
    save_params,
    reset_params,
    process_antibiogram,
    process_with_params,
    _DEFAULT_PARAMS,
)


def _load_gt() -> list:
    if GROUND_TRUTH_PATH.is_file():
        try:
            with open(GROUND_TRUTH_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return []


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_eval, tab_params, tab_test, tab_eucast = st.tabs([
    "📊 Évaluation", "🔧 Paramètres", "🖼️ Test en direct", "📋 EUCAST Excel"
])

# ══════════════════════════════════════════════════════════════════════════════
with tab_params:
    st.subheader("Paramètres de détection")

    current = load_params()

    st.markdown("**Prétraitement (CLAHE)**")
    c1, c2 = st.columns(2)
    with c1:
        clahe_clip = st.slider(
            "Limite de contraste (clahe_clip)",
            min_value=1.0, max_value=8.0, step=0.5,
            value=float(current.get("clahe_clip", 2.0)),
            help="Valeur plus élevée = contraste plus agressif",
        )
    with c2:
        clahe_grid = st.select_slider(
            "Taille de grille CLAHE",
            options=[4, 8, 16, 32],
            value=int(current.get("clahe_grid", 8)),
        )

    st.markdown("**Transformée de Hough (détection de cercles)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        hough_param1 = st.slider(
            "param1 (seuil gradient)",
            min_value=20, max_value=200, step=5,
            value=int(current.get("hough_param1", 80)),
            help="Seuil du détecteur de contours Canny",
        )
    with c2:
        hough_param2 = st.slider(
            "param2 (sensibilité)",
            min_value=10, max_value=80, step=5,
            value=int(current.get("hough_param2", 35)),
            help="Valeur plus faible = plus de cercles détectés (plus de faux positifs)",
        )
    with c3:
        hough_dp = st.select_slider(
            "dp (résolution inverse)",
            options=[1, 1.5, 2],
            value=float(current.get("hough_dp", 1)),
        )

    c1, c2 = st.columns(2)
    with c1:
        min_r_ratio = st.slider(
            "Rayon minimum / taille image (%)",
            min_value=1, max_value=8, step=1,
            value=int(float(current.get("hough_min_r_ratio", 0.02)) * 100),
        ) / 100.0
    with c2:
        max_r_ratio = st.slider(
            "Rayon maximum / taille image (%)",
            min_value=8, max_value=25, step=1,
            value=int(float(current.get("hough_max_r_ratio", 0.15)) * 100),
        ) / 100.0

    st.markdown("**Mesure de la zone d'inhibition**")
    zone_thresh = st.slider(
        "Seuil de détection du bord de zone (%)",
        min_value=20, max_value=70, step=5,
        value=int(float(current.get("zone_threshold_ratio", 0.40)) * 100),
        help="Pourcentage de l'amplitude d'intensité à partir duquel on détecte le bord de la zone",
    ) / 100.0

    st.markdown("**Validation des disques**")
    dark_val = st.checkbox(
        "Activer la validation 'disque sombre'",
        value=bool(current.get("dark_validation", True)),
        help="Rejette les cercles brillants (reflets, bulles). Désactivez si les disques sont clairs.",
    )
    dark_perc = st.slider(
        "Percentile d'intensité extérieure pour validation",
        min_value=10, max_value=60, step=5,
        value=int(current.get("dark_percentile", 40)),
    )

    new_params = {
        "clahe_clip": clahe_clip,
        "clahe_grid": clahe_grid,
        "hough_dp": hough_dp,
        "hough_param1": hough_param1,
        "hough_param2": hough_param2,
        "hough_min_r_ratio": min_r_ratio,
        "hough_max_r_ratio": max_r_ratio,
        "zone_threshold_ratio": zone_thresh,
        "dark_validation": dark_val,
        "dark_percentile": dark_perc,
    }

    col_save, col_reset = st.columns([1, 1])
    with col_save:
        if st.button("💾 Sauvegarder les paramètres", type="primary"):
            save_params(new_params)
            st.success(f"Paramètres sauvegardés dans {CONFIG_PATH}")
    with col_reset:
        if st.button("↩️ Réinitialiser aux valeurs par défaut"):
            reset_params()
            st.success("Paramètres réinitialisés.")
            st.rerun()

    with st.expander("Paramètres actuels (JSON)"):
        st.json(current)

# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("Évaluation sur la vérité terrain")
    gt = _load_gt()

    if not gt:
        st.info(
            "Aucune vérité terrain disponible. "
            "Allez sur la page **Données d'entraînement** pour importer et associer vos fichiers."
        )
    else:
        st.metric("Fiches de référence disponibles", len(gt))

        # Upload images for evaluation
        st.markdown("**Importez les images pour l'évaluation :**")
        eval_imgs = st.file_uploader(
            "Images (les noms de fichiers doivent correspondre à ceux de la vérité terrain)",
            type=["jpg", "jpeg", "png", "tiff", "bmp"],
            accept_multiple_files=True,
            key="eval_uploader",
        )

        if eval_imgs and st.button("▶️ Lancer l'évaluation", type="primary"):
            img_map: Dict[str, np.ndarray] = {}
            for f in eval_imgs:
                arr = np.array(Image.open(f).convert("RGB"))
                img_map[f.name] = arr

            results = []
            matched = 0

            progress = st.progress(0)
            for idx, rec in enumerate(gt):
                img_name = rec.get("image_name", "")
                if img_name not in img_map:
                    progress.progress((idx + 1) / len(gt))
                    continue

                img = img_map[img_name]
                entries = rec.get("entries", [])
                gt_zones = [e["zone_mm"] for e in entries if e.get("zone_mm") is not None]

                try:
                    disks, px_per_mm = process_antibiogram(img)
                    detected_zones = sorted([d.zone_diameter_mm for d in disks])
                    gt_zones_sorted = sorted(gt_zones)

                    # Match detected to ground truth (best effort by count)
                    n = min(len(detected_zones), len(gt_zones_sorted))
                    for j in range(n):
                        diff = abs(detected_zones[j] - gt_zones_sorted[j])
                        results.append({
                            "Image": img_name,
                            "GT Zone (mm)": round(gt_zones_sorted[j], 1),
                            "Détectée (mm)": round(detected_zones[j], 1),
                            "Erreur (mm)": round(diff, 1),
                            "OK (≤2mm)": "✅" if diff <= 2.0 else "❌",
                        })
                    matched += 1
                except Exception as exc:
                    st.warning(f"{img_name} : erreur — {exc}")

                progress.progress((idx + 1) / len(gt))

            progress.empty()

            if not results:
                st.warning("Aucune correspondance image ↔ vérité terrain. Vérifiez les noms de fichiers.")
            else:
                df = pd.DataFrame(results)
                mae = float(df["Erreur (mm)"].mean())
                rmse = float(np.sqrt((df["Erreur (mm)"] ** 2).mean()))
                ok_rate = float((df["OK (≤2mm)"] == "✅").mean()) * 100

                st.markdown("### Métriques de précision")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Images évaluées", matched)
                m2.metric("Erreur moyenne (MAE)", f"{mae:.1f} mm")
                m3.metric("RMSE", f"{rmse:.1f} mm")
                m4.metric("Zones correctes (±2mm)", f"{ok_rate:.0f}%")

                st.markdown("### Résultats détaillés")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Download report
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Télécharger rapport CSV",
                    data=csv,
                    file_name="evaluation_rapport.csv",
                    mime="text/csv",
                )

# ══════════════════════════════════════════════════════════════════════════════
with tab_test:
    st.subheader("Test en direct avec les paramètres actuels")
    st.caption("Modifiez les paramètres dans l'onglet Paramètres, puis testez ici avant de sauvegarder.")

    test_img = st.file_uploader(
        "Image de test", type=["jpg", "jpeg", "png", "tiff", "bmp"], key="test_img"
    )

    if test_img:
        from antibiogram_processor import draw_results

        img_arr = np.array(Image.open(test_img).convert("RGB"))

        with st.spinner("Détection en cours…"):
            try:
                disks, px_per_mm = process_with_params(img_arr, new_params)
            except Exception as exc:
                import traceback
                st.error(f"Erreur : {exc}")
                with st.expander("Détails de l'erreur"):
                    st.code(traceback.format_exc())
                disks, px_per_mm = [], 0.0

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_arr, use_container_width=True, caption="Image originale")
        with c2:
            if disks:
                annotated = draw_results(img_arr, disks, px_per_mm)
                st.image(annotated, use_container_width=True, caption=f"{len(disks)} disque(s) détecté(s)")

                rows = []
                for i, d in enumerate(disks):
                    rows.append({
                        "Disque": i + 1,
                        "Zone (mm)": round(d.zone_diameter_mm, 1),
                        "Confiance": f"{d.confidence:.0%}",
                        "Rayon disque (px)": round(d.radius_px, 1),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.success(f"Échelle : {px_per_mm:.1f} px/mm")
            else:
                st.warning("Aucun disque détecté. Ajustez les paramètres dans l'onglet Paramètres.")

# ══════════════════════════════════════════════════════════════════════════════
with tab_eucast:
    st.subheader("Importer le fichier Excel EUCAST")
    st.markdown(
        "Importez le fichier officiel **EUCAST Breakpoint Tables** (format `.xlsx`) "
        "pour bénéficier des seuils S/I/R les plus récents pour toutes les espèces."
    )

    EUCAST_PATH = Path(__file__).parent.parent / "data" / "eucast_breakpoints.xlsx"

    # ── Status ────────────────────────────────────────────────────────────────
    if EUCAST_PATH.is_file():
        st.success(f"✅ Fichier EUCAST présent : `{EUCAST_PATH.name}`  "
                   f"({EUCAST_PATH.stat().st_size // 1024} KB)")
        if st.button("🗑️ Supprimer", key="del_eucast"):
            EUCAST_PATH.unlink()
            st.rerun()
    else:
        st.info("Aucun fichier EUCAST importé. Téléchargez-le sur eucast.org → Clinical Breakpoints.")

    st.markdown("---")
    uploaded_xl = st.file_uploader(
        "Fichier EUCAST BreakpointTables (.xlsx)",
        type=["xlsx"],
        key="eucast_upload",
    )

    if uploaded_xl:
        EUCAST_PATH.parent.mkdir(parents=True, exist_ok=True)
        EUCAST_PATH.write_bytes(uploaded_xl.read())
        st.success(f"✅ Fichier sauvegardé ({EUCAST_PATH.stat().st_size // 1024} KB). Chargement…")
        st.rerun()

    # ── Preview loaded breakpoints ────────────────────────────────────────────
    if EUCAST_PATH.is_file():
        st.markdown("### Contenu chargé")
        from eucast_loader import load_eucast_excel
        with st.spinner("Lecture du fichier Excel…"):
            bp_dict, codes = load_eucast_excel(EUCAST_PATH)

        if not bp_dict:
            st.error("Impossible de lire les breakpoints. Vérifiez que c'est le bon fichier EUCAST.")
        else:
            species_list = sorted({s for (_, s) in bp_dict})
            st.success(f"✅ {len(species_list)} espèce(s) — {len(codes)} antibiotique(s) chargés")

            for sp in species_list:
                key = ("EUCAST", sp)
                abx = bp_dict.get(key, {})
                with st.expander(f"**{sp}** — {len(abx)} antibiotiques"):
                    rows = [{"Code": k, "R_max": v[0], "S_min": v[1]} for k, v in sorted(abx.items())]
                    if rows:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
