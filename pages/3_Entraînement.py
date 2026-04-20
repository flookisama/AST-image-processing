"""
Training Pipeline Page — auto-calibration from your labelled dataset.

Workflow:
  Step 1 · Upload images + Word files (or reuse data from page 1)
  Step 2 · Review / correct image ↔ Word matching
  Step 3 · Inspect extracted ground-truth
  Step 4 · Run auto-calibration (grid search)
  Step 5 · Analyse results and apply best parameters
"""
import io
import json
import copy
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title="Entraînement", layout="wide", page_icon="🎓")
st.title("🎓 Entraînement et auto-calibration")
st.markdown(
    "Importez votre base de données (images + tableaux Word) pour trouver automatiquement "
    "les meilleurs paramètres de détection grâce à une recherche par grille."
)

GROUND_TRUTH_PATH = Path(__file__).parent.parent / "data" / "ground_truth.json"

# ── Helper: load saved ground truth ───────────────────────────────────────────
def _load_gt() -> list:
    if GROUND_TRUTH_PATH.is_file():
        try:
            with open(GROUND_TRUTH_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _save_gt(data: list) -> None:
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── Session state ──────────────────────────────────────────────────────────────
if "train_images"   not in st.session_state: st.session_state.train_images   = {}   # name→bytes
if "train_words"    not in st.session_state: st.session_state.train_words    = {}   # name→bytes
if "word_records"   not in st.session_state: st.session_state.word_records   = []   # parsed records
if "gt_records"     not in st.session_state: st.session_state.gt_records     = _load_gt()
if "calib_result"   not in st.session_state: st.session_state.calib_result   = None
if "match_map"      not in st.session_state: st.session_state.match_map      = {}   # img→word

# ══════════════════════════════════════════════════════════════════════════════
step1, step2, step3, step4, step5 = st.tabs([
    "📁 1. Import",
    "🔗 2. Association",
    "🔍 3. Vérité terrain",
    "🚀 4. Entraînement",
    "📊 5. Résultats",
])

# ══════════════════════════════════════════════════════════════════════════════
with step1:
    st.subheader("Importer votre dataset")

    col_img, col_word = st.columns(2)

    with col_img:
        st.markdown("**Images de boîtes de Pétri**")
        imgs = st.file_uploader(
            "Formats acceptés : jpg, png, tiff, bmp",
            type=["jpg","jpeg","png","tiff","bmp"],
            accept_multiple_files=True,
            key="train_img_upload",
        )
        if imgs:
            for f in imgs:
                st.session_state.train_images[f.name] = f.read()
            st.success(f"{len(st.session_state.train_images)} image(s) chargée(s)")

        if st.session_state.train_images:
            st.dataframe(
                pd.DataFrame({"Fichier": list(st.session_state.train_images.keys())}),
                use_container_width=True, hide_index=True,
            )
            if st.button("🗑️ Effacer les images", key="clear_imgs"):
                st.session_state.train_images = {}
                st.rerun()

    with col_word:
        st.markdown("**Documents Word (.docx)**")
        words = st.file_uploader(
            "Tableaux antibiogramme en format Word",
            type=["docx"],
            accept_multiple_files=True,
            key="train_word_upload",
        )
        if words:
            from word_importer import parse_word_file, HAS_DOCX
            if not HAS_DOCX:
                st.error("python-docx requis — pip install python-docx")
            else:
                import tempfile
                records, errors = [], []
                for wf in words:
                    data = wf.read()
                    st.session_state.train_words[wf.name] = data
                    suffix = Path(wf.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
                        tmp_f.write(data)
                        tmp = Path(tmp_f.name)
                    try:
                        recs = parse_word_file(tmp)
                        for r in recs:
                            r_dict = r.to_dict()
                            r_dict["_word_file"] = wf.name
                            records.append(r_dict)
                    except Exception as e:
                        errors.append(f"{wf.name}: {e}")
                    finally:
                        tmp.unlink(missing_ok=True)

                st.session_state.word_records = records
                for e in errors: st.warning(e)
                if records:
                    st.success(f"{len(records)} fiche(s) extraite(s) de {len(words)} fichier(s)")

        if st.session_state.word_records:
            summary = [{"Fichier Word": r.get("_word_file","?"),
                        "Patient": r.get("patient_id") or "—",
                        "Bactérie": r.get("bacteria") or "—",
                        "Entrées": len(r.get("entries",[]))} for r in st.session_state.word_records]
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Ou importez directement un fichier `ground_truth.json` existant**")
    gt_upload = st.file_uploader("ground_truth.json", type=["json"], key="gt_upload")
    if gt_upload:
        try:
            data = json.loads(gt_upload.read())
            _save_gt(data)
            st.session_state.gt_records = data
            st.success(f"✅ {len(data)} fiche(s) importée(s) — passez à l'étape 4.")
        except Exception as e:
            st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
with step2:
    st.subheader("Association image ↔ fiche Word")

    images = st.session_state.train_images
    records = st.session_state.word_records

    if not images:
        st.info("Importez des images dans l'étape 1.")
    elif not records:
        st.info("Importez des fichiers Word dans l'étape 1, ou passez directement à l'étape 4 si vous avez déjà un ground_truth.json.")
    else:
        # Auto-match by filename similarity
        from trainer import auto_match_files
        if st.button("🔄 Association automatique par nom de fichier"):
            word_names = [r.get("_word_file","") for r in records]
            auto = auto_match_files(list(images.keys()), word_names)
            st.session_state.match_map = auto

        record_labels = ["— Aucune —"] + [
            f"#{i+1} | {r.get('_word_file','')} | {r.get('bacteria') or '?'} | {r.get('patient_id') or '?'}"
            for i, r in enumerate(records)
        ]

        match_map = st.session_state.match_map
        new_map = {}

        for img_name in images:
            c1, c2 = st.columns([1, 3])
            with c1:
                arr = np.array(Image.open(io.BytesIO(images[img_name])).convert("RGB"))
                thumb = Image.fromarray(arr).resize((120, 90), Image.LANCZOS)
                st.image(np.array(thumb), caption=img_name, use_container_width=False)
            with c2:
                # Find default index from auto-match
                auto_word = match_map.get(img_name)
                default_idx = 0
                if auto_word:
                    for i, r in enumerate(records):
                        if r.get("_word_file") == auto_word:
                            default_idx = i + 1
                            break

                choice = st.selectbox(
                    f"Fiche pour {img_name}",
                    record_labels,
                    index=default_idx,
                    key=f"match_sel_{img_name}",
                    label_visibility="collapsed",
                )
                idx = record_labels.index(choice) - 1
                new_map[img_name] = idx

                if idx >= 0:
                    r = records[idx]
                    zones = [e["zone_mm"] for e in r.get("entries",[]) if e.get("zone_mm")]
                    antibs = [e["antibiotic"] for e in r.get("entries",[]) if e.get("antibiotic")]
                    st.caption(
                        f"Bactérie: **{r.get('bacteria') or '?'}** · "
                        f"{len(r.get('entries',[]))} antibiotiques · "
                        f"Zones: {', '.join(f'{z:.0f}mm' for z in zones[:5])} …"
                    )

        st.session_state.match_map = new_map

        if st.button("💾 Sauvegarder les associations", type="primary"):
            gt_entries = []
            for img_name, idx in new_map.items():
                if idx < 0:
                    continue
                rec = copy.deepcopy(records[idx])
                rec["image_name"] = img_name
                gt_entries.append(rec)

            existing = {e.get("image_name"): e for e in _load_gt()}
            for e in gt_entries:
                existing[e["image_name"]] = e
            merged = list(existing.values())
            _save_gt(merged)
            st.session_state.gt_records = merged
            st.success(f"✅ {len(gt_entries)} paire(s) sauvegardée(s). Passez à l'étape 4.")

# ══════════════════════════════════════════════════════════════════════════════
with step3:
    st.subheader("Aperçu de la vérité terrain")
    gt = st.session_state.gt_records

    if not gt:
        st.info("Aucune vérité terrain. Complétez les étapes 1 et 2.")
    else:
        st.metric("Fiches disponibles", len(gt))
        total_zones = sum(
            len([e for e in r.get("entries",[]) if e.get("zone_mm") is not None])
            for r in gt
        )
        st.metric("Zones de référence totales", total_zones)

        for rec in gt:
            with st.expander(f"📋 {rec.get('image_name','?')}  —  {rec.get('bacteria') or 'bactérie?'}"):
                entries = rec.get("entries", [])
                if entries:
                    df = pd.DataFrame(entries)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("Aucune entrée.")

        col_dl, col_del = st.columns([1,1])
        with col_dl:
            st.download_button(
                "⬇️ Télécharger ground_truth.json",
                data=json.dumps(gt, ensure_ascii=False, indent=2).encode(),
                file_name="ground_truth.json",
                mime="application/json",
            )
        with col_del:
            if st.button("🗑️ Effacer la vérité terrain"):
                _save_gt([])
                st.session_state.gt_records = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
with step4:
    st.subheader("Lancer l'auto-calibration")

    gt = st.session_state.gt_records
    images_bytes = st.session_state.train_images

    # Allow uploading images separately if not in session
    if not images_bytes:
        st.markdown("**Images du dataset (si pas encore importées)**")
        train_imgs_up = st.file_uploader(
            "Images", type=["jpg","jpeg","png","tiff","bmp"],
            accept_multiple_files=True, key="train_imgs_step4"
        )
        if train_imgs_up:
            for f in train_imgs_up:
                st.session_state.train_images[f.name] = f.read()
            images_bytes = st.session_state.train_images

    if not gt:
        st.warning("Aucune vérité terrain. Complétez les étapes précédentes ou importez un ground_truth.json dans l'étape 1.")
    else:
        # Build samples list
        from trainer import samples_from_ground_truth, TrainingSample
        images_arrays = {}
        for name, raw in images_bytes.items():
            try:
                images_arrays[name] = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception:
                pass

        samples = samples_from_ground_truth(gt, images_arrays)

        # Show what we have
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Fiches de référence", len(gt))
        col_b.metric("Images chargées", len(images_arrays))
        col_c.metric("Paires prêtes à l'entraînement", len(samples))

        if len(samples) == 0:
            st.warning(
                "Aucune paire image+vérité terrain valide. "
                "Vérifiez que les noms de fichiers image correspondent aux champs `image_name` "
                "dans ground_truth.json, et que les entrées contiennent des `zone_mm` non nuls."
            )
        else:
            # Configuration
            with st.expander("⚙️ Configuration de la recherche par grille"):
                from trainer import PARAM2_GRID, CLAHE_CLIP_GRID, THRESHOLD_GRID
                st.markdown(f"**Étape 1** — `hough_param2` × `clahe_clip` : {len(PARAM2_GRID)} × {len(CLAHE_CLIP_GRID)} = {len(PARAM2_GRID)*len(CLAHE_CLIP_GRID)} combinaisons")
                st.markdown(f"**Étape 2** — `zone_threshold_ratio` : {len(THRESHOLD_GRID)} valeurs")
                n_runs = (len(PARAM2_GRID) * len(CLAHE_CLIP_GRID) + len(THRESHOLD_GRID)) * len(samples)
                st.info(f"Total estimé : **{n_runs} exécutions** sur {len(samples)} image(s). Temps estimé : ~{n_runs * 0.5 / 60:.0f}–{n_runs * 2 / 60:.0f} min.")

            if st.button("🚀 Démarrer l'entraînement", type="primary"):
                status_box = st.empty()
                progress_bar = st.progress(0.0)

                def on_progress(frac: float, msg: str) -> None:
                    progress_bar.progress(min(frac, 1.0))
                    status_box.caption(f"⏳ {msg}")

                from trainer import auto_calibrate
                with st.spinner("Calibration en cours…"):
                    result = auto_calibrate(samples, progress_cb=on_progress)

                progress_bar.progress(1.0)
                status_box.empty()
                st.session_state.calib_result = result
                st.success(
                    f"✅ Calibration terminée ! "
                    f"MAE baseline : **{result.baseline_mae:.2f} mm** → "
                    f"après calibration : **{result.best_mae:.2f} mm** "
                    f"(amélioration : **{result.improvement_pct:+.1f}%**)"
                )
                st.info("Consultez l'étape 5 pour analyser les résultats et appliquer la calibration.")

# ══════════════════════════════════════════════════════════════════════════════
with step5:
    st.subheader("Résultats et application de la calibration")

    result = st.session_state.calib_result

    if result is None:
        st.info("Lancez l'entraînement à l'étape 4 pour voir les résultats ici.")
    else:
        from antibiogram_processor import load_params, save_params

        # ── Summary metrics ────────────────────────────────────────────────
        st.markdown("### Résumé")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE avant calibration", f"{result.baseline_mae:.2f} mm")
        m2.metric("MAE après calibration", f"{result.best_mae:.2f} mm",
                  delta=f"{result.best_mae - result.baseline_mae:+.2f} mm")
        m3.metric("Amélioration", f"{result.improvement_pct:+.1f}%")

        # ── Best parameters ────────────────────────────────────────────────
        st.markdown("### Meilleurs paramètres trouvés")
        current = load_params()
        param_rows = []
        for k in ["hough_param2", "clahe_clip", "zone_threshold_ratio"]:
            param_rows.append({
                "Paramètre": k,
                "Avant": current.get(k, "—"),
                "Après": result.best_params.get(k, "—"),
            })
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

        # ── Grid search results ────────────────────────────────────────────
        st.markdown("### Résultats de la grille de recherche (Étape 2 — seuil de zone)")
        st.dataframe(result.grid_df, use_container_width=True, hide_index=True)

        # ── Per-sample results ─────────────────────────────────────────────
        st.markdown("### Détail par échantillon")
        if not result.per_sample_df.empty:
            # Numeric MAE only (exclude N/D)
            numeric_df = result.per_sample_df[result.per_sample_df["Erreur (mm)"] != "N/D"].copy()
            if not numeric_df.empty:
                numeric_df["Erreur (mm)"] = numeric_df["Erreur (mm)"].astype(float)
                ok_rate = (numeric_df["OK (≤2mm)"] == "✅").mean() * 100
                st.caption(f"Zones correctes à ±2mm : **{ok_rate:.0f}%**")
            st.dataframe(result.per_sample_df, use_container_width=True, hide_index=True)

            csv_report = result.per_sample_df.to_csv(index=False).encode()
            st.download_button("⬇️ Télécharger rapport CSV", data=csv_report,
                               file_name="calibration_rapport.csv", mime="text/csv")

        st.markdown("---")
        # ── Apply / discard ────────────────────────────────────────────────
        col_apply, col_discard = st.columns(2)
        with col_apply:
            if st.button("✅ Appliquer cette calibration", type="primary"):
                save_params(result.best_params)
                st.success(
                    f"Calibration appliquée ! Paramètres sauvegardés dans `data/calibration.json`.\n\n"
                    f"L'application principale utilisera désormais ces paramètres optimisés."
                )
        with col_discard:
            if st.button("🗑️ Ignorer et conserver les paramètres actuels"):
                st.session_state.calib_result = None
                st.info("Résultats ignorés. Les paramètres actuels sont inchangés.")
                st.rerun()
