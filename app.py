import streamlit as st
import torch
from model_utils import generate_3d_from_text

st.set_page_config(layout="centered")
st.title("🧊 Text to 3D Model Generator")

prompt = st.text_input("Enter a short text prompt", "a small toy car")

if st.button("Generate 3D Model"):
    with st.spinner("Generating 3D model..."):
        obj_path = generate_3d_from_text(prompt)
        st.success("Model generated!")
        st.download_button("Download .obj file", open(obj_path, "rb"), file_name="model.obj")
