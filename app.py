import io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from antibiogram_processor import process_antibiogram, draw_results, DetectedDisk
from breakpoints import get_breakpoint_info, get_antibiotic_options, SPECIES_LIST

st.set_page_config(page_title="Antibiogram Reader", layout="wide", page_icon="🧫")

st.title("🧫 Antibiogram Reader")
st.markdown(
    "Photographiez ou importez une boîte de Pétri. L'application détecte les disques, "
    "mesure les zones d'inhibition et interprète S/I/R selon les normes EUCAST ou CLSI."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Paramètres du test")
species = st.sidebar.selectbox("Espèce bactérienne :", sorted(SPECIES_LIST))
guideline = st.sidebar.radio("Référentiel clinique :", ["EUCAST", "CLSI"])
ANTIBIOTIC_OPTIONS = get_antibiotic_options(guideline, species)

st.sidebar.markdown("---")
st.sidebar.markdown("**Autres pages**")
st.sidebar.page_link("pages/1_Données_Entraînement.py", label="📂 Données d'entraînement")
st.sidebar.page_link("pages/2_Calibration.py", label="⚙️ Calibration")

# ── Session state ──────────────────────────────────────────────────────────────
if "disks" not in st.session_state:
    st.session_state.disks: list[DetectedDisk] = []
if "px_per_mm" not in st.session_state:
    st.session_state.px_per_mm: float = 0.0
if "img_data" not in st.session_state:
    st.session_state.img_data = None
if "img_key" not in st.session_state:
    st.session_state.img_key: str = ""

# ── Helper: OCR label from disk region ────────────────────────────────────────
def _try_ocr(img_rgb: np.ndarray, cx: float, cy: float, r: float) -> str:
    """Try pytesseract OCR on the disk region. Returns empty string if unavailable."""
    try:
        import pytesseract
        import cv2
        h, w = img_rgb.shape[:2]
        pad = int(r * 1.2)
        x1, y1 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x2, y2 = min(w, int(cx + pad)), min(h, int(cy + pad))
        crop = img_rgb[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cfg = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(bw, config=cfg).strip()
        return text[:8] if text else ""
    except Exception:
        return ""


# ── Main layout ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Capture ou import de l'image")
    input_method = st.radio("Source :", ["📷 Caméra", "📁 Importer une image"], horizontal=True)

    raw_img = None
    if input_method == "📷 Caméra":
        cam = st.camera_input("Photographiez la boîte de Pétri")
        if cam is not None:
            raw_img = cam
            img_key = cam.name if hasattr(cam, "name") else str(cam.file_id)
    else:
        up = st.file_uploader("Image antibiogramme", type=["jpg", "jpeg", "png", "tiff", "bmp"])
        if up is not None:
            raw_img = up
            img_key = up.name

    if raw_img is not None:
        img_arr = np.array(Image.open(raw_img).convert("RGB"))
        st.image(img_arr, use_container_width=True, caption="Image originale")

        # Run detection only when image changes
        if img_key != st.session_state.img_key:
            st.session_state.img_key = img_key
            with st.spinner("Détection des disques et mesure des zones…"):
                try:
                    disks, px_per_mm = process_antibiogram(img_arr)
                except Exception as exc:
                    st.error(f"Erreur de traitement : {exc}")
                    disks, px_per_mm = [], 0.0
            st.session_state.disks = disks
            st.session_state.px_per_mm = px_per_mm
            st.session_state.img_data = img_arr
            # Reset any manual corrections
            for k in list(st.session_state.keys()):
                if k.startswith("zone_corr_") or k.startswith("label_corr_"):
                    del st.session_state[k]

        disks = st.session_state.disks
        px_per_mm = st.session_state.px_per_mm

        if disks:
            st.success(f"✅ {len(disks)} disque(s) détecté(s) — échelle : {px_per_mm:.1f} px/mm")

            # Draw annotated image using current (possibly corrected) zones
            corrected_disks = []
            for i, d in enumerate(disks):
                corr_mm = st.session_state.get(f"zone_corr_{i}", d.zone_diameter_mm)
                corr_label = st.session_state.get(f"label_corr_{i}", d.label)
                import copy
                cd = copy.copy(d)
                cd.zone_diameter_mm = float(corr_mm)
                cd.label = corr_label
                cd.zone_radius_px = (float(corr_mm) / 2.0) * px_per_mm
                corrected_disks.append(cd)

            annotated = draw_results(img_arr, corrected_disks, px_per_mm)
            st.image(annotated, use_container_width=True, caption="Résultat (rouge = disque, vert = zone)")
            st.caption("🟢 confiance élevée  🟡 confiance moyenne  🔴 confiance faible")
        else:
            st.warning(
                "Aucun disque détecté. Améliorez l'éclairage, centrez la boîte et "
                "évitez les reflets. Consultez la page Calibration si le problème persiste."
            )

with col_right:
    st.subheader("2. Interprétation et correction")
    st.write(f"**Espèce :** {species}   |   **Référentiel :** {guideline}")

    disks = st.session_state.disks
    px_per_mm = st.session_state.px_per_mm
    img_data = st.session_state.img_data

    if img_data is not None and disks:
        rows = []
        for i, d in enumerate(disks):
            st.markdown(f"---\n**Disque {i + 1}**")

            # Confidence indicator
            conf = d.confidence
            conf_label = "🟢 Élevée" if conf > 0.65 else "🟡 Moyenne" if conf > 0.35 else "🔴 Faible"
            st.caption(f"Confiance de détection : {conf_label} ({conf:.0%})")

            c1, c2 = st.columns([1, 1])
            with c1:
                # Label: try OCR then manual
                ocr_suggestion = _try_ocr(img_data, d.center[0], d.center[1], d.radius_px)
                default_label = ocr_suggestion if ocr_suggestion else d.label
                label = st.text_input(
                    "Étiquette du disque",
                    value=st.session_state.get(f"label_corr_{i}", default_label),
                    key=f"label_widget_{i}",
                )
                st.session_state[f"label_corr_{i}"] = label

            with c2:
                zone_corr = st.number_input(
                    "Zone (mm) — correction manuelle",
                    min_value=6.0,
                    max_value=50.0,
                    value=float(st.session_state.get(f"zone_corr_{i}", round(d.zone_diameter_mm, 1))),
                    step=0.5,
                    key=f"zone_widget_{i}",
                )
                st.session_state[f"zone_corr_{i}"] = zone_corr

            ab_choice = st.selectbox(
                "Antibiotique",
                ANTIBIOTIC_OPTIONS,
                key=f"ab_{i}",
            )

            if ab_choice and ab_choice != "Unknown":
                info = get_breakpoint_info(guideline, species, ab_choice, zone_corr)
                sir = info["sir"]
                sir_label = info["label"]
                color_icon = {"S": "🟢", "I": "🟡", "R": "🔴"}.get(sir, "⚪")
                st.markdown(f"**Résultat : {color_icon} {sir_label} ({sir})**")
                rows.append(
                    {
                        "Disk": i + 1,
                        "Label": label,
                        "Antibiotic": ab_choice,
                        "Zone (mm)": round(float(zone_corr), 1),
                        "Confidence": f"{conf:.0%}",
                        "Interpretation": f"{color_icon} {sir_label}",
                        "SIR": sir,
                    }
                )
            else:
                rows.append(
                    {
                        "Disk": i + 1,
                        "Label": label,
                        "Antibiotic": ab_choice or "—",
                        "Zone (mm)": round(float(zone_corr), 1),
                        "Confidence": f"{conf:.0%}",
                        "Interpretation": "—",
                        "SIR": "?",
                    }
                )

        # ── Summary table ──────────────────────────────────────────────────
        if rows:
            st.markdown("---")
            st.subheader("Tableau récapitulatif")
            display_cols = ["Disk", "Label", "Antibiotic", "Zone (mm)", "Confidence", "Interpretation"]
            df = pd.DataFrame(rows)[display_cols]
            st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Export ────────────────────────────────────────────────────
            st.markdown("**Exporter les résultats**")
            ec, pc = st.columns(2)

            with ec:
                from export import to_csv_bytes
                csv_bytes = to_csv_bytes(rows)
                st.download_button(
                    "⬇️ Télécharger CSV",
                    data=csv_bytes,
                    file_name="antibiogramme_resultats.csv",
                    mime="text/csv",
                )

            with pc:
                try:
                    from export import to_pdf_bytes
                    pdf_bytes = to_pdf_bytes(rows, species=species, guideline=guideline)
                    st.download_button(
                        "⬇️ Télécharger PDF",
                        data=pdf_bytes,
                        file_name="antibiogramme_rapport.pdf",
                        mime="application/pdf",
                    )
                except ImportError:
                    st.caption("PDF désactivé (installez fpdf2)")
    else:
        st.info(
            "Capturez ou importez une image de boîte de Pétri pour voir "
            "la détection de zones et l'interprétation S/I/R ici."
        )
