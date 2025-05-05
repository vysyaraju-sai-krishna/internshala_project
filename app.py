import streamlit as st
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_models():
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    return xm, model, diffusion

st.title("🧊 Text to 3D Model Generator")

prompt = st.text_input("Enter a text prompt", "a small toy car")

if st.button("Generate"):
    with st.spinner("Generating 3D model..."):
        xm, model, diffusion = load_models()
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=15.0,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
        filename = f"{prompt.replace(' ', '_')}.obj"
        mesh.write_obj(open(filename, "w"))
        st.success("3D model generated!")
        st.download_button("Download .obj", open(filename, "rb"), filename)
