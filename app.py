import streamlit as st
import torch
import os
import tempfile

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model
from shap_e.util.rendering import create_pan_cameras, render_mesh

# Set up Streamlit page
st.set_page_config(page_title="Shap-E 3D Generator", layout="centered")
st.title("🧠 Shap-E: Text to 3D Model Generator")

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and diffusion (cached)
@st.cache_resource
def load_models():
    st.info("Loading Shap-E models...")
    xm = load_model('text300M', device)
    diffusion = diffusion_from_config('text300M')
    return xm, diffusion

xm, diffusion_model = load_models()

# User input
prompt = st.text_input("🔤 Enter a text prompt to generate a 3D model:", "a cute robot")

if st.button("🚀 Generate 3D Model"):
    with st.spinner("Generating 3D model from text..."):
        try:
            # Step 1: Sample latent representation
            latents = sample_latents(
                batch_size=1,
                model=diffusion_model,
                model_kwargs=dict(texts=[prompt]),
                guidance_scale=15.0,
                progress=True,
                device=device,
            )

            latent = latents[0]
            mesh = xm.decode_latent_mesh(latent).tri_mesh()

            with tempfile.TemporaryDirectory() as tmpdir:
                obj_path = os.path.join(tmpdir, "model.obj")
                stl_path = os.path.join(tmpdir, "model.stl")

                mesh.write_obj(obj_path)
                mesh.write_stl(stl_path)

                st.success("✅ 3D model generated!")

                # Download links
                st.download_button("⬇️ Download .OBJ", open(obj_path, "rb"), "model.obj")
                st.download_button("⬇️ Download .STL", open(stl_path, "rb"), "model.stl")

                # Step 2: Render preview images
                st.subheader("🖼️ 3D Model Preview")
                cameras = create_pan_cameras()
                images = render_mesh(mesh, cameras=cameras, resolution=256)

                for i, img in enumerate(images[:3]):
                    st.image(img, caption=f"View {i + 1}", use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
