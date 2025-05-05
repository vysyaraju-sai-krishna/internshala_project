import streamlit as st
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh, decode_latent_images, create_pan_cameras
import tempfile
import os

st.set_page_config(page_title="Shap-E: Text to 3D", layout="centered")
st.title("🧠 Text to 3D Generator using OpenAI's Shap-E")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_models():
    st.info("Loading models...")
    xm = load_model('text300M', device)
    diffusion_model = diffusion_from_config('text300M')
    return xm, diffusion_model

xm, diffusion_model = load_models()

prompt = st.text_input("🔤 Enter your 3D prompt", value="a spaceship shaped like a banana")

if st.button("🚀 Generate 3D Model"):
    with st.spinner("Generating 3D model... This takes a minute or two ⏳"):
        try:
            # Step 1: Generate latents
            latents = sample_latents(
                batch_size=1,
                model=diffusion_model,
                model_kwargs=dict(texts=[prompt]),
                guidance_scale=15.0,
                progress=True,
                device=device,
            )

            # Step 2: Decode mesh
            mesh = decode_latent_mesh(xm, latents[0], device)

            # Step 3: Write to OBJ and STL
            with tempfile.TemporaryDirectory() as tmpdir:
                obj_path = os.path.join(tmpdir, f"{prompt.replace(' ', '_')}.obj")
                stl_path = os.path.join(tmpdir, f"{prompt.replace(' ', '_')}.stl")

                with open(obj_path, "w") as f:
                    mesh.write_obj(f)
                with open(stl_path, "w") as f:
                    mesh.write_stl(f)

                # Step 4: Display download buttons
                st.success("✅ Model generation complete!")
                st.download_button("⬇️ Download OBJ", open(obj_path, "rb"), file_name=os.path.basename(obj_path))
                st.download_button("⬇️ Download STL", open(stl_path, "rb"), file_name=os.path.basename(stl_path))

                # Step 5: Preview images
                st.subheader("🖼️ Preview Images")
                cameras = create_pan_cameras()
                images = decode_latent_images(xm, latents[0], cameras, device)
                for i, image in enumerate(images[:3]):  # Show 3 preview images
                    st.image(image, caption=f"View {i+1}", use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

