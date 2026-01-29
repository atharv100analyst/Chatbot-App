import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(page_title="Prompt Comparison", layout="wide")
st.title("🖼️ Normal vs Professional Prompt Comparison (CPU Safe)")

st.warning(
    "⚠️ This app runs on CPU (Streamlit Cloud has no GPU). "
    "Image generation may take 30–90 seconds."
)

# ---------------------------------
# Load Model (CPU Optimized)
# ---------------------------------
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,   # CPU SAFE
        safety_checker=None
    )
    pipe.enable_attention_slicing()  # Reduce RAM usage
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()

# ---------------------------------
# User Inputs
# ---------------------------------
normal_prompt = st.text_area(
    "Normal Prompt",
    value="an apple on table"
)

pro_prompt = st.text_area(
    "Professional Prompt",
    value="ultra realistic red apple on wooden table, studio lighting, soft shadow, product photography"
)

# ---------------------------------
# Generate Button
# ---------------------------------
if st.button("🚀 Generate Images"):
    with st.spinner("Generating images on CPU... Please wait ⏳"):
        img_normal = pipe(
            normal_prompt,
            num_inference_steps=15,     # Reduced steps
            guidance_scale=7.0,
            height=384,
            width=384
        ).images[0]

        img_pro = pipe(
            pro_prompt,
            num_inference_steps=20,
            guidance_scale=8.0,
            height=384,
            width=384
        ).images[0]

    # ---------------------------------
    # Display Side by Side
    # ---------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Normal Prompt Output")
        st.image(img_normal, use_column_width=True)

    with col2:
        st.subheader("Professional Prompt Output")
        st.image(img_pro, use_column_width=True)
