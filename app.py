import copy
import unicodedata
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
st.sidebar.markdown("**Détection**")
use_ml_filter = st.sidebar.checkbox(
    "Filtre ML (HOG+SVM)",
    value=True,
    help="Désactivez si aucun disque n'est détecté — le filtre ML peut être trop strict sur certaines images.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Autres pages**")
st.sidebar.page_link("pages/1_Données_Entraînement.py", label="📂 Données d'entraînement")
st.sidebar.page_link("pages/2_Calibration.py", label="⚙️ Calibration")
st.sidebar.page_link("pages/3_Entraînement.py", label="🎓 Entraînement")
st.sidebar.page_link("pages/4_Modèles_ML.py", label="🤖 Modèles ML")

# ── Antibiotic name → code fuzzy lookup ───────────────────────────────────────
def _build_name_map() -> dict:
    """Build a lowercase name → code lookup from eucast_loader."""
    try:
        from eucast_loader import AGENT_NAME_TO_CODE
        m = {}
        for full_name, code in AGENT_NAME_TO_CODE:
            key = unicodedata.normalize("NFKD", full_name).encode("ascii", "ignore").decode().lower()
            m[key] = code
        return m
    except Exception:
        return {}

_NAME_MAP = _build_name_map()

def _resolve_antibiotic(text: str) -> str:
    """
    Convert free-text antibiotic name to a known code.
    Returns the input unchanged if no match is found (allows unknown antibiotics).
    """
    if not text:
        return text
    t = text.strip().upper()
    # 1. Direct code match
    if t in ANTIBIOTIC_OPTIONS:
        return t
    # 2. Lowercase normalized lookup (full name or prefix)
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower().strip()
    if norm in _NAME_MAP:
        return _NAME_MAP[norm]
    # 3. Prefix match
    for key, code in _NAME_MAP.items():
        if key.startswith(norm) or norm.startswith(key[:6]):
            return code
    # 4. Contains match (at least 4 chars)
    if len(norm) >= 4:
        for key, code in _NAME_MAP.items():
            if norm in key or key[:len(norm)] == norm:
                return code
    # No match — return as-is so user can still store it
    return t

# ── Label recognition (EasyOCR → tesseract → empty) ───────────────────────────
def _try_ocr(img_rgb: np.ndarray, cx: float, cy: float, r: float) -> str:
    try:
        from label_recognizer import recognize_label
        code, conf = recognize_label(img_rgb, cx, cy, r)
        return code if conf > 0.3 else ""
    except Exception:
        return ""

# ── Disk crop preview ──────────────────────────────────────────────────────────
def _disk_crop(img: np.ndarray, cx: float, cy: float, r: float, size: int = 110) -> np.ndarray:
    h, w = img.shape[:2]
    pad = int(r * 1.6)
    x1, y1 = max(0, int(cx) - pad), max(0, int(cy) - pad)
    x2, y2 = min(w, int(cx) + pad), min(h, int(cy) + pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return img[:size, :size]
    pil = Image.fromarray(crop).resize((size, size), Image.LANCZOS)
    return np.array(pil)

# ── Session state ──────────────────────────────────────────────────────────────
for key in ("disks", "px_per_mm", "img_data", "img_key"):
    if key not in st.session_state:
        st.session_state[key] = [] if key == "disks" else (0.0 if key == "px_per_mm" else None if key == "img_data" else "")

# ── Image input ────────────────────────────────────────────────────────────────
st.subheader("1. Capture ou import de l'image")
input_method = st.radio("Source :", ["📷 Caméra", "📁 Importer une image"], horizontal=True)

raw_img = None
img_key = ""
if input_method == "📷 Caméra":
    cam = st.camera_input("Photographiez la boîte de Pétri")
    if cam is not None:
        raw_img = cam
        img_key = cam.name if hasattr(cam, "name") else str(getattr(cam, "file_id", "cam"))
else:
    up = st.file_uploader("Image antibiogramme", type=["jpg", "jpeg", "png", "tiff", "bmp"])
    if up is not None:
        raw_img = up
        img_key = up.name

if raw_img is not None:
    img_arr = np.array(Image.open(raw_img).convert("RGB"))

    # Run detection only when the image changes or ML toggle changes
    detection_key = f"{img_key}_{use_ml_filter}"
    if detection_key != st.session_state.img_key:
        st.session_state.img_key = detection_key
        with st.spinner("Détection des disques et mesure des zones…"):
            try:
                disks, px_per_mm = process_antibiogram(img_arr, use_ml=use_ml_filter)
            except Exception as exc:
                st.error(f"Erreur de traitement : {exc}")
                disks, px_per_mm = [], 0.0
        st.session_state.disks = disks
        st.session_state.px_per_mm = px_per_mm
        st.session_state.img_data = img_arr
        # Clear all per-disk widget state so new defaults apply
        for k in list(st.session_state.keys()):
            if any(k.startswith(p) for p in ("zone_w_", "label_w_", "ab_mode_", "ab_sel_", "ab_free_")):
                del st.session_state[k]

    disks: list = st.session_state.disks
    px_per_mm: float = st.session_state.px_per_mm
    img_data: np.ndarray = st.session_state.img_data

    # ── Pre-initialise widget states so col_left image is always up-to-date ──
    if disks:
        for i, d in enumerate(disks):
            if f"zone_w_{i}" not in st.session_state:
                st.session_state[f"zone_w_{i}"] = round(float(d.zone_diameter_mm), 1)
            if f"label_w_{i}" not in st.session_state:
                ocr = _try_ocr(img_data, d.center[0], d.center[1], d.radius_px)
                st.session_state[f"label_w_{i}"] = ocr if ocr else d.label
            if f"ab_mode_{i}" not in st.session_state:
                st.session_state[f"ab_mode_{i}"] = "Liste"
            if f"ab_sel_{i}" not in st.session_state:
                st.session_state[f"ab_sel_{i}"] = "Unknown"
            if f"ab_free_{i}" not in st.session_state:
                st.session_state[f"ab_free_{i}"] = ""

    # ── Left/Right columns ────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.image(img_arr, use_container_width=True, caption="Image originale")
        if disks:
            # Build corrected disks using current widget values
            corrected = []
            for i, d in enumerate(disks):
                cd = copy.copy(d)
                cd.zone_diameter_mm = float(st.session_state.get(f"zone_w_{i}", d.zone_diameter_mm))
                cd.label = st.session_state.get(f"label_w_{i}", d.label)
                cd.zone_radius_px = (cd.zone_diameter_mm / 2.0) * px_per_mm
                corrected.append(cd)
            annotated = draw_results(img_data, corrected, px_per_mm)
            st.image(annotated, use_container_width=True, caption="Zones détectées (se met à jour en temps réel)")
            st.caption("🟢 confiance ≥ 65%  •  🟡 35–65%  •  🔴 < 35%")
            st.success(f"✅ {len(disks)} disque(s) — échelle : {px_per_mm:.1f} px/mm")
        else:
            st.warning(
                "Aucun disque détecté. Améliorez l'éclairage, centrez la boîte "
                "et évitez les reflets. Utilisez la page ⚙️ Calibration si besoin."
            )

    with col_right:
        st.subheader("2. Correction et interprétation")
        st.write(f"**Espèce :** {species}   |   **Référentiel :** {guideline}")

        if not disks:
            st.info("En attente d'une image avec des disques détectés.")
        else:
            rows = []
            for i, d in enumerate(disks):
                conf = d.confidence
                conf_icon = "🟢" if conf > 0.65 else "🟡" if conf > 0.35 else "🔴"

                with st.expander(
                    f"**Disque {i + 1}** — {st.session_state.get(f'label_w_{i}', d.label)}   "
                    f"{conf_icon} {conf:.0%}",
                    expanded=True,
                ):
                    # ── Disk crop preview + zone slider ──────────────────────
                    crop_col, ctrl_col = st.columns([1, 2])

                    with crop_col:
                        crop_img = _disk_crop(img_data, d.center[0], d.center[1], d.radius_px)
                        st.image(crop_img, caption=f"Disque {i+1}", use_container_width=True)

                    with ctrl_col:
                        # ── Zone slider ──────────────────────────────────────
                        st.markdown("**Zone d'inhibition (mm)**")
                        zone_val = st.slider(
                            "Ajuster la zone",
                            min_value=6.0,
                            max_value=50.0,
                            step=0.5,
                            key=f"zone_w_{i}",
                            label_visibility="collapsed",
                        )
                        st.caption(f"Zone détectée automatiquement : {d.zone_diameter_mm:.1f} mm  |  Valeur courante : **{zone_val:.1f} mm**")
                        if abs(zone_val - d.zone_diameter_mm) > 0.4:
                            if st.button("↩️ Réinitialiser", key=f"reset_zone_{i}"):
                                st.session_state[f"zone_w_{i}"] = round(float(d.zone_diameter_mm), 1)
                                st.rerun()

                    # ── Label / OCR ──────────────────────────────────────────
                    label = st.text_input(
                        "Étiquette du disque (code imprimé)",
                        key=f"label_w_{i}",
                        help="Saisissez ou corrigez le code inscrit sur le disque (ex: AMX25, CIP5…)",
                    )

                    # ── Antibiotic selection: dropdown OR free text ──────────
                    st.markdown("**Antibiotique pour interprétation S/I/R**")
                    mode = st.radio(
                        "Mode de saisie",
                        ["Liste déroulante", "Saisie libre"],
                        horizontal=True,
                        key=f"ab_mode_{i}",
                        label_visibility="collapsed",
                    )

                    if mode == "Liste déroulante":
                        ab_code = st.selectbox(
                            "Antibiotique",
                            ANTIBIOTIC_OPTIONS,
                            key=f"ab_sel_{i}",
                            label_visibility="collapsed",
                        )
                    else:
                        raw_ab = st.text_input(
                            "Nom ou code de l'antibiotique",
                            key=f"ab_free_{i}",
                            placeholder="ex: Ciprofloxacin, CIP, Amoxicillin…",
                            label_visibility="collapsed",
                        )
                        ab_code = _resolve_antibiotic(raw_ab) if raw_ab.strip() else "Unknown"
                        if raw_ab.strip() and ab_code != raw_ab.strip().upper():
                            st.caption(f"Correspondance trouvée : **{ab_code}**")
                        elif raw_ab.strip():
                            st.caption(f"Code utilisé tel quel : **{ab_code}** (aucune correspondance dans les référentiels)")

                    # ── SIR result ───────────────────────────────────────────
                    if ab_code and ab_code != "Unknown":
                        info = get_breakpoint_info(guideline, species, ab_code, zone_val)
                        sir = info["sir"]
                        sir_label = info["label"]
                        color_icon = {"S": "🟢", "I": "🟡", "R": "🔴"}.get(sir, "⚪")
                        st.markdown(f"### {color_icon} {sir_label} ({sir})")
                        rows.append({
                            "Disk": i + 1,
                            "Label": label,
                            "Antibiotic": ab_code,
                            "Zone (mm)": round(zone_val, 1),
                            "Confidence": f"{conf:.0%}",
                            "Interpretation": f"{sir_label}",
                            "SIR": sir,
                        })
                    else:
                        st.markdown("*Sélectionnez ou saisissez un antibiotique pour obtenir l'interprétation.*")
                        rows.append({
                            "Disk": i + 1,
                            "Label": label,
                            "Antibiotic": "—",
                            "Zone (mm)": round(zone_val, 1),
                            "Confidence": f"{conf:.0%}",
                            "Interpretation": "—",
                            "SIR": "?",
                        })

            # ── Summary table + export ────────────────────────────────────────
            if rows:
                st.markdown("---")
                st.subheader("Tableau récapitulatif")
                df = pd.DataFrame(rows)[["Disk", "Label", "Antibiotic", "Zone (mm)", "Confidence", "Interpretation"]]
                st.dataframe(df, use_container_width=True, hide_index=True)

                ec, pc = st.columns(2)
                with ec:
                    from export import to_csv_bytes
                    st.download_button(
                        "⬇️ Télécharger CSV",
                        data=to_csv_bytes(rows),
                        file_name="antibiogramme_resultats.csv",
                        mime="text/csv",
                    )
                with pc:
                    try:
                        from export import to_pdf_bytes
                        st.download_button(
                            "⬇️ Télécharger PDF",
                            data=to_pdf_bytes(rows, species=species, guideline=guideline),
                            file_name="antibiogramme_rapport.pdf",
                            mime="application/pdf",
                        )
                    except ImportError:
                        st.caption("PDF désactivé (installez fpdf2)")

else:
    st.info("Capturez ou importez une image de boîte de Pétri pour commencer.")
