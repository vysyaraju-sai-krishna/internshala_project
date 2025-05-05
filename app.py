import streamlit as st
import trimesh
import pyrender
import numpy as np
import os
import uuid

# Function to generate a 3D model from text prompt (mock point cloud)
def text_to_3d(prompt):
    # Mock implementation: create a cube or sphere based on keywords
    if "car" in prompt.lower():
        # Create a simple box (mock car shape)
        mesh = trimesh.creation.box(extents=[0.5, 0.3, 0.2])
    elif "chair" in prompt.lower():
        # Create a simple chair-like shape
        mesh = trimesh.creation.box(extents=[0.3, 0.3, 0.5])
    elif "toy" in prompt.lower():
        # Create a simple toy-like shape
        mesh = trimesh.creation.icosphere(radius=0.2)
    else:
        # Default to a sphere
        mesh = trimesh.creation.icosphere(radius=0.3)
    return mesh

# Function to visualize and save 3D model
def visualize_and_save(mesh, output_format="obj"):
    # Save mesh to file
    output_file = f"model_{uuid.uuid4()}.{output_format}"
    mesh.export(output_file)
    
    # Visualize using pyrender
    scene = pyrender.Scene()
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyrender)
    
    # Set up camera and lighting
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light)
    
    # Render
    r = pyrender.OffscreenRenderer(400, 400)
    color, _ = r.render(scene)
    r.delete()
    
    return output_file, color

# Streamlit app
st.title("Text to 3D Model Generator")
st.write("Enter a text prompt to generate a 3D model (e.g., 'A small toy car').")

# Text input
text_prompt = st.text_input("Enter a text prompt")
if text_prompt and st.button("Generate 3D Model"):
    with st.spinner("Generating 3D model from text..."):
        # Generate 3D model from text
        mesh = text_to_3d(text_prompt)
        
        # Visualize and save
        output_file, rendered_image = visualize_and_save(mesh, "obj")
        
        # Display rendered 3D model
        st.image(rendered_image, caption="Generated 3D Model", use_column_width=True)
        
        # Provide download link
        with open(output_file, "rb") as f:
            st.download_button(
                label="Download 3D Model (.obj)",
                data=f,
                file_name=output_file,
                mime="application/octet-stream"
            )
        os.remove(output_file)  # Clean up
