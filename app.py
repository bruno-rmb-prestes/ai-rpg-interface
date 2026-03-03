import os
import ssl
from dotenv import load_dotenv

load_dotenv()

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
ssl._create_default_https_context = lambda: ssl_context

import httpcore._backends.sync
_original_start_tls = httpcore._backends.sync.SyncStream.start_tls

def _patched_start_tls(self, ssl_context, server_hostname=None, timeout=None):
    patched_context = ssl.create_default_context()
    patched_context.check_hostname = False
    patched_context.verify_mode = ssl.CERT_NONE
    return _original_start_tls(self, patched_context, server_hostname, timeout)

httpcore._backends.sync.SyncStream.start_tls = _patched_start_tls

import streamlit as st
from gradio_client import Client

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.title("AI Image Generator")
st.markdown("Generate images using Z-Image-Turbo on Hugging Face")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    with st.spinner("Connecting to Hugging Face Space..."):
        try:
            st.session_state.client = Client("mrfakename/Z-Image-Turbo")
        except Exception as e:
            st.error(f"Failed to connect to HF Space: {e}")
            st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            if "image" in message:
                st.image(message["image"], caption=message.get("prompt", ""))
                if "seed" in message:
                    st.caption(f"Seed: {message['seed']}")
            if "error" in message:
                st.error(message["error"])

if prompt := st.chat_input("Describe the image you want to generate..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Generating image..."):
            try:
                result = st.session_state.client.predict(
                    prompt=prompt,
                    height=st.session_state.get("height", 1024),
                    width=st.session_state.get("width", 1024),
                    num_inference_steps=st.session_state.get("steps", 9),
                    randomize_seed=True,
                    api_name="/generate_image"
                )
                
                image_data, seed_used = result
                
                if isinstance(image_data, dict):
                    image_path = image_data.get("path") or image_data.get("url")
                elif isinstance(image_data, str):
                    image_path = image_data
                else:
                    image_path = image_data
                
                st.image(image_path, caption=prompt)
                st.caption(f"Seed: {int(seed_used)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "image": image_path,
                    "prompt": prompt,
                    "seed": int(seed_used)
                })
                
            except Exception as e:
                error_msg = f"Error generating image: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "error": error_msg
                })

with st.sidebar:
    st.header("Settings")
    
    st.subheader("Image Settings")
    st.session_state.width = st.select_slider(
        "Width",
        options=[512, 768, 1024, 1280],
        value=1024
    )
    st.session_state.height = st.select_slider(
        "Height", 
        options=[512, 768, 1024, 1280],
        value=1024
    )
    st.session_state.steps = st.slider(
        "Inference Steps",
        min_value=4,
        max_value=20,
        value=9,
        help="More steps = better quality but slower"
    )
    
    st.markdown("---")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This app uses [Z-Image-Turbo](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo) "
        "on Hugging Face for fast image generation."
    )
    st.markdown("---")
    st.markdown("### Tips")
    st.markdown(
        "- Be descriptive with your prompts\n"
        "- Include style keywords (e.g., 'photorealistic', 'anime', 'oil painting')\n"
        "- Specify lighting and mood"
    )
