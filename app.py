import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from antibiogram_processor import process_antibiogram, draw_results, DetectedDisk
from breakpoints import get_breakpoint_info, get_antibiotic_options

# Set up the page layout
st.set_page_config(page_title="Antibiogram Reader", layout="wide")

# App Header
st.title("🧫 Antibiogram Reader")
st.markdown(
    "Photograph or upload a petri dish with antibiotic disks. The app will detect zones, "
    "measure diameters, and interpret susceptibility (S/I/R) using EUCAST or CLSI breakpoints."
)

# Sidebar for test parameters
st.sidebar.header("Test Parameters")
species = st.sidebar.selectbox(
    "Select Bacterial Species:",
    [
        "Escherichia coli",
        "Staphylococcus aureus",
        "Pseudomonas aeruginosa",
        "Streptococcus pneumoniae",
    ],
)
guideline = st.sidebar.radio("Clinical Guideline:", ["EUCAST", "CLSI"])

# Antibiotic list from breakpoints (EUCAST from Excel when available)
ANTIBIOTIC_OPTIONS = get_antibiotic_options(guideline, species)

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Capture or upload")
    input_method = st.radio(
        "Input:",
        ["📷 Camera", "📁 Upload image"],
        horizontal=True,
    )

    img_data = None
    if input_method == "📷 Camera":
        cam_img = st.camera_input("Take a photo of the petri dish")
        if cam_img is not None:
            img_data = np.array(Image.open(cam_img).convert("RGB"))
    else:
        uploaded_file = st.file_uploader(
            "Upload antibiogram image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            img_data = np.array(Image.open(uploaded_file).convert("RGB"))

    if img_data is not None:
        st.image(img_data, use_container_width=True, channels="RGB")
        with st.spinner("Detecting disks and measuring zones…"):
            try:
                disks, px_per_mm = process_antibiogram(img_data)
            except Exception as e:
                st.error(f"Processing error: {e}")
                disks = []
                px_per_mm = 0.0

        if disks:
            st.success(f"Found {len(disks)} disk(s). Scale: {px_per_mm:.1f} px/mm")
            annotated = draw_results(img_data, disks, px_per_mm)
            st.image(annotated, use_container_width=True, channels="RGB")
            st.caption("Red: disk outline. Green: measured zone.")
        else:
            st.warning("No disks detected. Try better lighting, centering the plate, or upload a clearer image.")

with col2:
    st.subheader("2. Interpretation")
    st.write(f"**Species:** {species}")
    st.write(f"**Guideline:** {guideline}")

    if img_data is not None and disks:
        # Per-disk antibiotic selection and SIR
        st.markdown("**Assign antibiotic and view S/I/R**")
        rows = []
        for i, d in enumerate(disks):
            ab_choice = st.selectbox(
                f"Disk {i + 1} — {d.zone_diameter_mm:.1f} mm",
                ANTIBIOTIC_OPTIONS,
                key=f"ab_{i}",
            )
            if ab_choice and ab_choice != "Unknown":
                info = get_breakpoint_info(guideline, species, ab_choice, d.zone_diameter_mm)
                sir = info["sir"]
                label = info["label"]
                color = {"S": "🟢", "I": "🟡", "R": "🔴"}.get(sir, "⚪")
                rows.append(
                    {
                        "Disk": i + 1,
                        "Antibiotic": ab_choice,
                        "Zone (mm)": round(d.zone_diameter_mm, 1),
                        "Interpretation": f"{color} {label}",
                    }
                )
            else:
                rows.append(
                    {
                        "Disk": i + 1,
                        "Antibiotic": ab_choice or "—",
                        "Zone (mm)": round(d.zone_diameter_mm, 1),
                        "Interpretation": "—",
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Capture or upload a petri dish image to see zone detection and S/I/R interpretation here.")
