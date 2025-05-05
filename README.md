# Text to 3D Model Generator

This app takes a short text prompt and generates a downloadable 3D model (.obj) using OpenAI's Shap-E.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   streamlit run app.py

## Libraries Used
- torch
- shap-e
- streamlit
- numpy

## Thought Process
I choose Shap-E for its ability to convert text into 3D meshes. Streamlit was used for fast prototyping and interactivity.
