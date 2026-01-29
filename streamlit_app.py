import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Prompt Comparison", layout="wide")
st.title("🖼️ Normal vs Professional Prompt Comparison")

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

normal_prompt = st.text_area(
    "Normal Prompt",
    value="an apple on table"
)

pro_prompt = st.text_area(
    "Professional Prompt",
    value="ultra realistic red apple on wooden table, studio lighting, soft shadow, product photography, high detail, 4k"
)

if st.button("🚀 Generate Images"):
    with st.spinner("Generating images..."):
        img1 = pipe(normal_prompt).images[0]
        img2 = pipe(pro_prompt).images[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Normal Prompt")
        st.image(img1, use_column_width=True)

    with col2:
        st.subheader("Professional Prompt")
        st.image(img2, use_column_width=True)
