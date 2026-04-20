"""
Training Data Management Page
Import Word antibiogram documents + images, match them, and save ground truth.
"""
import io
import json
import copy
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title="Données d'entraînement", layout="wide", page_icon="📂")
st.title("📂 Données d'entraînement")
st.markdown(
    "Importez vos images d'antibiogrammes et vos tableaux Word. "
    "Associez chaque image à ses données de référence, puis sauvegardez la vérité terrain."
)

GROUND_TRUTH_PATH = Path(__file__).parent.parent / "data" / "ground_truth.json"
GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_gt() -> list:
    if GROUND_TRUTH_PATH.is_file():
        try:
            with open(GROUND_TRUTH_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_gt(data: list) -> None:
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Session state ──────────────────────────────────────────────────────────────
if "gt_data" not in st.session_state:
    st.session_state.gt_data = _load_gt()
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}   # name → bytes
if "word_records" not in st.session_state:
    st.session_state.word_records = []      # list of AntibiogramRecord dicts

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["1. Import fichiers", "2. Association image ↔ données", "3. Vérité terrain"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Importer les images")
    uploaded_imgs = st.file_uploader(
        "Images de boîtes de Pétri (jpg, png, tiff…)",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=True,
        key="img_uploader",
    )
    if uploaded_imgs:
        for f in uploaded_imgs:
            st.session_state.uploaded_images[f.name] = f.read()
        st.success(f"{len(st.session_state.uploaded_images)} image(s) chargée(s).")

    st.markdown("---")
    st.subheader("Importer les documents Word (.docx)")
    uploaded_words = st.file_uploader(
        "Tableaux d'antibiogramme Word",
        type=["docx"],
        accept_multiple_files=True,
        key="word_uploader",
    )

    if uploaded_words:
        try:
            import tempfile
            from word_importer import parse_word_file, AntibiogramRecord
            records = []
            errors = []
            for wf in uploaded_words:
                suffix = Path(wf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(wf.read())
                    tmp_path = Path(tmp.name)
                try:
                    recs = parse_word_file(tmp_path)
                    for r in recs:
                        records.append(r.to_dict())
                except Exception as exc:
                    errors.append(f"{wf.name} : {exc}")
                finally:
                    tmp_path.unlink(missing_ok=True)

            st.session_state.word_records = records

            if errors:
                for e in errors:
                    st.warning(e)
            if records:
                st.success(f"{len(records)} fiche(s) extraite(s) de {len(uploaded_words)} fichier(s).")
            else:
                st.warning("Aucune fiche extraite. Vérifiez le format des tableaux Word.")

        except ImportError:
            st.error("python-docx est requis : `pip install python-docx`")

    # Show parsed Word records summary
    if st.session_state.word_records:
        st.markdown("**Aperçu des fiches extraites :**")
        summary = []
        for i, r in enumerate(st.session_state.word_records):
            summary.append({
                "#": i + 1,
                "Fichier": r.get("source_file", ""),
                "Patient": r.get("patient_id") or "—",
                "Bactérie": r.get("bacteria") or "—",
                "Date": r.get("date") or "—",
                "Entrées": len(r.get("entries", [])),
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Associer chaque image à une fiche")

    images = st.session_state.uploaded_images
    word_records = st.session_state.word_records

    if not images:
        st.info("Importez d'abord des images dans l'onglet 1.")
    elif not word_records:
        st.info("Importez des fichiers Word ou saisissez les données manuellement ci-dessous.")
    else:
        record_labels = ["— Aucune —"] + [
            f"#{i+1} | {r.get('source_file','')} | {r.get('bacteria') or 'bactérie inconnue'} | {r.get('patient_id') or ''}"
            for i, r in enumerate(word_records)
        ]

        st.markdown("Pour chaque image, sélectionnez la fiche correspondante :")
        pairs = []
        for img_name, img_bytes in images.items():
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(img_pil, caption=img_name, use_container_width=True)
            with c2:
                choice = st.selectbox(f"Fiche pour **{img_name}** :", record_labels, key=f"match_{img_name}")
                idx = record_labels.index(choice) - 1
                if idx >= 0:
                    rec = word_records[idx]
                    st.markdown(
                        f"- **Bactérie :** {rec.get('bacteria') or '?'}\n"
                        f"- **Patient :** {rec.get('patient_id') or '?'}\n"
                        f"- **{len(rec.get('entries', []))} antibiogramme(s)**"
                    )
                    entries_df = pd.DataFrame(rec.get("entries", []))
                    if not entries_df.empty:
                        st.dataframe(entries_df, use_container_width=True, hide_index=True)
                    pairs.append({"image_name": img_name, "record_index": idx})
                else:
                    pairs.append({"image_name": img_name, "record_index": None})

        st.markdown("---")
        if st.button("💾 Sauvegarder les associations", type="primary"):
            gt_entries = []
            for pair in pairs:
                if pair["record_index"] is None:
                    continue
                rec = copy.deepcopy(word_records[pair["record_index"]])
                rec["image_name"] = pair["image_name"]
                gt_entries.append(rec)

            # Merge with existing ground truth (update by image_name)
            existing = {e.get("image_name"): e for e in _load_gt()}
            for e in gt_entries:
                existing[e["image_name"]] = e
            merged = list(existing.values())
            _save_gt(merged)
            st.session_state.gt_data = merged
            st.success(f"✅ {len(gt_entries)} association(s) sauvegardée(s) dans {GROUND_TRUTH_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Vérité terrain enregistrée")
    gt = st.session_state.gt_data

    if not gt:
        st.info("Aucune donnée enregistrée. Associez des images dans l'onglet 2.")
    else:
        st.metric("Fiches enregistrées", len(gt))

        # Table summary
        rows = []
        for rec in gt:
            rows.append({
                "Image": rec.get("image_name", "?"),
                "Bactérie": rec.get("bacteria") or "?",
                "Patient": rec.get("patient_id") or "?",
                "Date": rec.get("date") or "?",
                "Entrées": len(rec.get("entries", [])),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Detail expander per record
        for i, rec in enumerate(gt):
            with st.expander(f"Détail — {rec.get('image_name', f'Fiche {i+1}')}"):
                st.json(rec)

        # Export ground truth
        gt_json = json.dumps(gt, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger la vérité terrain (JSON)",
            data=gt_json,
            file_name="ground_truth.json",
            mime="application/json",
        )

        # Upload existing ground truth
        st.markdown("---")
        uploaded_gt = st.file_uploader("Importer un fichier ground_truth.json existant", type=["json"])
        if uploaded_gt is not None:
            try:
                imported = json.loads(uploaded_gt.read().decode("utf-8"))
                _save_gt(imported)
                st.session_state.gt_data = imported
                st.success(f"✅ {len(imported)} fiche(s) importée(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Erreur : {exc}")

        # Delete button
        if st.button("🗑️ Effacer toute la vérité terrain", type="secondary"):
            _save_gt([])
            st.session_state.gt_data = []
            st.rerun()
