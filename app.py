import streamlit as st
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh

st.title("🧠🔄 Prompt to 3D Generator (Shap-E)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models (cached)
@st.cache_resource
def load_models():
    print("Loading models...")
    xm = load_model('text300M', device)
    dm = diffusion_from_config('text300M')
    return xm, dm

xm, diffusion_model = load_models()

# Input prompt
prompt = st.text_input("Enter a text prompt to generate a 3D model", "a chair shaped like an avocado")

if st.button("Generate 3D Model"):
    with st.spinner("Generating... this may take a minute ⏳"):
        batch_size = 1
        guidance_scale = 15.0

        latents = sample_latents(
            batch_size=batch_size,
            model=diffusion_model,
            model_kwargs=dict(texts=[prompt]),
            guidance_scale=guidance_scale,
            progress=True,
            device=device,
        )

        # Decode into mesh
        mesh = decode_latent_mesh(xm, latents[0], device)
        output_path = f"{prompt.replace(' ', '_')}.obj"
        with open(output_path, "w") as f:
            mesh.write_obj(f)

        st.success("✅ 3D Model Generated!")
        st.download_button("Download .obj file", data=open(output_path, "rb"), file_name=output_path)

        # Optional: preview as images
        cameras = create_pan_cameras()
        images = decode_latent_images(xm, latents[0], cameras, device)
        for img in images:
            st.image(img, use_column_width=True)
