from  utils import *
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO


use_case_map = {
    "Roughness": Detect_Roughness,
    "Lumps":detect_lumps,
    "Black Shade": Detect_shade,
    "Sunlight": Detect_sunlight,
    "Potholes": Detect_potholes,
    "Rust":Detect_non_blue_regions,
    "Diameter": Detect_diameter,
    "Material-Mixing":Detect_material_mix,
    "Joint-Cut": joint_cut,
    "Mould-Joint-Mismatch": mould_joint_mismatch,
    "Black Lining": black_lining,
    "Burning Marks": burn_white_mark_detection,
    "Damage": detect_damage,
    "Improper Weld Finishing": improper_weld_finishing,
    "Blow Hole": detect_blowhole,
    "Blow Hole due to Contamination": detect_blowhole_contamination
}

st.title("AquaDefect AI")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save image to a temp location for OpenCV use
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        tmp_image_path = tmp_file.name

    # Select use case
    use_case = st.selectbox("Select use case to detect", list(use_case_map.keys()))

    if st.button("Detect"):
        # Run selected detection
        st.write(f"Running detection for: {use_case}")
        result_image = use_case_map[use_case](tmp_image_path)

        # Convert result for Streamlit
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption=f"{use_case} Detection Result", use_container_width=True)

        # Clean up temp file
        os.remove(tmp_image_path)
