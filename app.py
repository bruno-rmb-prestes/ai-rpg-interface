import os
import random
import ssl
from dotenv import load_dotenv
import streamlit as st
from gradio_client import Client
from openai import OpenAI
import httpcore._backends.sync

load_dotenv()

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
ssl._create_default_https_context = lambda: ssl_context

_original_start_tls = httpcore._backends.sync.SyncStream.start_tls

def _patched_start_tls(self, ssl_context, server_hostname=None, timeout=None):
    patched_context = ssl.create_default_context()
    patched_context.check_hostname = False
    patched_context.verify_mode = ssl.CERT_NONE
    return _original_start_tls(self, patched_context, server_hostname, timeout)

httpcore._backends.sync.SyncStream.start_tls = _patched_start_tls


HF_INFERENCE_MODEL = "Qwen/Qwen3.5-35B-A3B:novita"
IMPROVE_SYSTEM = (
    "###ROLE"
"You are a expert prompt enhacer, focusing in detailed descriptions for d&d places and characters. "
"###MISSION"
"You will receive a prompt and you need to enhance its description, giving more details to make the prompt as complete as it can be."
"###OUTPUT"
"your output will ALWAYS be ONLY the improved prompt."
"###PROMPT"
"{prompt}"
)


def get_hf_openai_client():
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )


# Thumbnail width for side-by-side display
IMAGE_THUMB_WIDTH = 220

@st.dialog("Full size image")
def show_full_size(image_path: str):
    st.image(image_path)
    if st.button("Close"):
        st.session_state.lightbox_path = None
        st.rerun()

def render_image_gallery(image_paths, key_prefix="gallery"):
    """Render images side by side, at most 4 per row; 'View full size' opens dialog."""
    paths = image_paths if isinstance(image_paths, list) else [image_paths]
    n = len(paths)
    num_cols = min(n, 4)
    cols = st.columns(num_cols)
    for i, path in enumerate(paths):
        with cols[i % num_cols]:
            st.image(path, width=IMAGE_THUMB_WIDTH)
            if st.button("View full size", key=f"{key_prefix}_{i}", use_container_width=True):
                st.session_state.lightbox_path = path
                st.rerun()


def improve_prompt(raw_prompt: str) -> str:
    if not raw_prompt or not raw_prompt.strip():
        return ""
    prompt_text = raw_prompt.strip()
    system_content = IMPROVE_SYSTEM.replace("{prompt}", prompt_text)
    client = get_hf_openai_client()
    completion = client.chat.completions.create(
        model=HF_INFERENCE_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=1024,
    )
    return (completion.choices[0].message.content or "").strip()

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.title("AI Image Generator")
st.markdown("Generate images using Z-Image-Turbo on Hugging Face")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompt_draft" not in st.session_state:
    st.session_state.prompt_draft = ""
if "pending_improve" not in st.session_state:
    st.session_state.pending_improve = None  # str: prompt being improved, or None
if "pending_generate" not in st.session_state:
    st.session_state.pending_generate = None  # str: prompt being generated, or None
# Qwen improved result: applied at start of run so the text area shows it (see "where the answer goes" below)
if "improved_prompt_to_apply" not in st.session_state:
    st.session_state.improved_prompt_to_apply = None
# Full-size image dialog (set by "View full size" button in gallery)
if "lightbox_path" not in st.session_state:
    st.session_state.lightbox_path = None
# Clear input on next run (set by generate flow; we apply it here before any widget uses prompt_input)
if "clear_prompt_on_next_run" not in st.session_state:
    st.session_state.clear_prompt_on_next_run = False

# Apply clear or improved prompt at start of run (before widget); set both draft and widget key so display is correct
if st.session_state.clear_prompt_on_next_run:
    st.session_state.prompt_draft = ""
    st.session_state.prompt_box = ""
    st.session_state.clear_prompt_on_next_run = False
if st.session_state.improved_prompt_to_apply is not None:
    st.session_state.prompt_draft = st.session_state.improved_prompt_to_apply
    st.session_state.prompt_box = st.session_state.improved_prompt_to_apply
    st.session_state.improved_prompt_to_apply = None

if "client" not in st.session_state:
    with st.spinner("Connecting to Hugging Face Space..."):
        try:
            hf_token = os.getenv("HF_TOKEN")
            st.session_state.client = Client(
                "mrfakename/Z-Image-Turbo",
                token=hf_token
            )
        except Exception as e:
            st.error(f"Failed to connect to HF Space: {e}")
            st.stop()

for msg_i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            if "image" in message:
                img = message["image"]
                render_image_gallery(img if isinstance(img, list) else [img], key_prefix=f"msg_{msg_i}")
            if "error" in message:
                st.error(message["error"])

if st.session_state.get("lightbox_path"):
    show_full_size(st.session_state.lightbox_path)

# Block input and buttons while waiting for an answer
is_busy = (
    st.session_state.pending_improve is not None
    or st.session_state.pending_generate is not None
)

# Message input: text area + Improve prompt + Generate (key "prompt_box" so we never write to widget state)
prompt_input = st.text_area(
    "Describe the image you want to generate...",
    value=st.session_state.prompt_draft,
    height=100,
    key="prompt_box",
    placeholder="e.g. a cat wearing a wizard hat, digital art, vibrant colors",
    disabled=is_busy,
)
st.session_state.prompt_draft = prompt_input

col_improve, col_generate, _ = st.columns([1, 1, 4])
with col_improve:
    improve_clicked = st.button(
        "✨ Improve prompt", use_container_width=True, disabled=is_busy
    )
with col_generate:
    generate_clicked = st.button(
        "🎨 Generate", type="primary", use_container_width=True, disabled=is_busy
    )

# Run improve: blocked UI; Qwen answer goes into improved_prompt_to_apply, then we rerun
# so the next run applies it to the input (see block at top of script)
if st.session_state.pending_improve is not None:
    prompt_to_improve = st.session_state.pending_improve
    with st.spinner("Improving prompt with Qwen..."):
        try:
            improved = improve_prompt(prompt_to_improve)
            if improved:
                st.session_state.improved_prompt_to_apply = improved
        except Exception as e:
            st.error(f"Failed to improve prompt: {str(e)}")
    st.session_state.pending_improve = None
    st.rerun()

# Run generate: blocked UI until image(s) ready; call space once per image when num_images > 1
if st.session_state.pending_generate is not None:
    prompt = st.session_state.pending_generate
    st.session_state.clear_prompt_on_next_run = True  # clear input at start of next run (before widget)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        num_images = st.session_state.get("num_images", 1)
        use_random_seed = st.session_state.get("use_random_seed", True)
        fixed_seed = st.session_state.get("seed", 0)
        # For a single image we can use fixed seed if user chose it; for multiple we always use distinct random seeds
        use_fixed_seed = num_images == 1 and not use_random_seed and (fixed_seed is not None and fixed_seed >= 0)
        # For multiple images, generate completely different seeds so each image is unique
        if num_images > 1:
            seeds_to_use = random.sample(range(0, 2**32), num_images)
        else:
            seeds_to_use = [int(fixed_seed)] if use_fixed_seed else None

        try:
            image_paths = []
            seeds_used = []
            for i in range(num_images):
                with st.spinner(f"Generating image {i + 1} of {num_images}..."):
                    if seeds_to_use is not None:
                        # Explicit seed(s): use them so each image has a different seed
                        result = st.session_state.client.predict(
                            prompt=prompt,
                            height=st.session_state.get("height", 1024),
                            width=st.session_state.get("width", 1024),
                            num_inference_steps=st.session_state.get("steps", 9),
                            randomize_seed=False,
                            seed=int(seeds_to_use[i]),
                            api_name="/generate_image"
                        )
                    else:
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

                image_paths.append(image_path)
                seeds_used.append(int(seed_used))

            render_image_gallery(image_paths, key_prefix="new_gen")
            st.session_state.messages.append({
                "role": "assistant",
                "image": image_paths[0] if len(image_paths) == 1 else image_paths,
                "prompt": prompt,
                "seed": seeds_used[0] if len(seeds_used) == 1 else seeds_used,
            })

        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "error": error_msg
            })
    st.session_state.pending_generate = None
    st.rerun()

# Queue improve: set pending and rerun so next run shows blocked UI and runs API
if not is_busy and improve_clicked and prompt_input.strip():
    st.session_state.pending_improve = prompt_input.strip()
    st.rerun()
elif not is_busy and improve_clicked and not prompt_input.strip():
    st.warning("Enter a prompt first, then click Improve prompt.")

# Queue generate: set pending and rerun so next run shows blocked UI and runs API
if not is_busy and generate_clicked and prompt_input.strip():
    st.session_state.pending_generate = prompt_input.strip()
    st.rerun()
elif not is_busy and generate_clicked and not prompt_input.strip():
    st.warning("Enter a prompt first, then click Generate.")

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
    st.session_state.num_images = st.slider(
        "Number of images",
        min_value=1,
        max_value=8,
        value=1,
        help="How many images to generate. For more than 1, each uses a random seed and the space is called once per image."
    )
    use_random_seed = st.checkbox(
        "Random seed",
        value=True,
        help="Use a random seed (only applies when generating 1 image). Uncheck and set seed below for reproducible results."
    )
    st.session_state.use_random_seed = use_random_seed
    st.session_state.seed = st.number_input(
        "Seed (for 1 image only)",
        min_value=0,
        max_value=2**32 - 1,
        value=0,
        step=1,
        help="Fixed seed for reproducibility when generating a single image. Ignored when Number of images > 1 or Random seed is on."
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
