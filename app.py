# app.py — Streamlit UI for the AI Image Generator.
# All layout, session state, and user interaction live here; business logic
# (Hugging Face API, prompt improvement, image generation) is in services.py.

import os
import ssl
from dotenv import load_dotenv
import streamlit as st
from gradio_client import Client
import httpcore._backends.sync

from services import improve_prompt, generate_images, edit_image, IMPROVE_SYSTEM

load_dotenv()


# -----------------------------------------------------------------------------
# SSL and TLS configuration for Hugging Face / Gradio
# -----------------------------------------------------------------------------
# Disables certificate verification so the app can reach Hugging Face endpoints
# in environments where SSL verification would otherwise fail (e.g. some proxies).
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


# -----------------------------------------------------------------------------
# Page configuration and global layout
# -----------------------------------------------------------------------------
# Sets the browser tab title, favicon, and wide layout so the image gallery
# and chat have more horizontal space.
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
)

st.title("AI Image Generator")
st.markdown("Generate images using Z-Image-Turbo on Hugging Face")

# Hides Streamlit’s built-in “View fullscreen” button on images to avoid
# clutter in the gallery.
st.markdown(
    '<style>button[title="View fullscreen"]{display:none !important;}</style>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------
# Ensures all keys used by the chat, prompt input, and async flows exist before
# any widget or handler runs. Prevents KeyError and keeps state consistent
# across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompt_draft" not in st.session_state:
    st.session_state.prompt_draft = ""
if "pending_improve" not in st.session_state:
    st.session_state.pending_improve = None  # prompt being improved, or None
if "pending_generate" not in st.session_state:
    st.session_state.pending_generate = None  # prompt being generated, or None
if "improved_prompt_to_apply" not in st.session_state:
    st.session_state.improved_prompt_to_apply = None  # Qwen result applied at next run
if "clear_prompt_on_next_run" not in st.session_state:
    st.session_state.clear_prompt_on_next_run = False
if "pending_edit" not in st.session_state:
    st.session_state.pending_edit = None
# Custom system prompt for the prompt enhancer, seeded with the module default.
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = IMPROVE_SYSTEM

# Applies “clear input” or “use improved prompt” at the start of each run,
# before the text area is rendered, so the user sees the correct value.
if st.session_state.clear_prompt_on_next_run:
    st.session_state.prompt_draft = ""
    st.session_state.prompt_box = ""
    st.session_state.clear_prompt_on_next_run = False
if st.session_state.improved_prompt_to_apply is not None:
    st.session_state.prompt_draft = st.session_state.improved_prompt_to_apply
    st.session_state.prompt_box = st.session_state.improved_prompt_to_apply
    st.session_state.improved_prompt_to_apply = None


# -----------------------------------------------------------------------------
# Hugging Face Gradio client (lazy init)
# -----------------------------------------------------------------------------
# Connects to the Z-Image-Turbo Space once and stores the client in session
# state so we don’t reconnect on every rerun. Stops the app with an error
# if the token is missing or the connection fails.
if "client" not in st.session_state:
    with st.spinner("Connecting to Hugging Face Space..."):
        try:
            hf_token = os.getenv("HF_TOKEN")
            st.session_state.client = Client(
                "mrfakename/Z-Image-Turbo",
                token=hf_token,
            )
        except Exception as e:
            st.error(f"Failed to connect to HF Space: {e}")
            st.stop()


# -----------------------------------------------------------------------------
# UI helper: image gallery
# -----------------------------------------------------------------------------
# Renders a list of image paths in a responsive grid (up to 4 per row) so
# single and multiple generated images display consistently in the chat.
IMAGES_PER_ROW = 4


def render_image_gallery(image_paths, key_prefix="gallery"):
    """Show images at full size in a grid, up to 4 per row."""
    paths = image_paths if isinstance(image_paths, list) else [image_paths]
    n = len(paths)
    num_cols = min(n, IMAGES_PER_ROW)
    cols = st.columns(num_cols)
    for i, path in enumerate(paths):
        with cols[i % num_cols]:
            st.image(path, width="stretch")


# -----------------------------------------------------------------------------
# Busy state: block input while improve, generate, or edit is running
# -----------------------------------------------------------------------------
is_busy = (
    st.session_state.pending_improve is not None
    or st.session_state.pending_generate is not None
    or st.session_state.pending_edit is not None
)


# -----------------------------------------------------------------------------
# Chat history display
# -----------------------------------------------------------------------------
# Renders all stored messages: user prompts as markdown and assistant replies
# as image galleries or error text. Each assistant image also gets inline edit
# controls (text input + button) so the user can request modifications.
edit_request = None

for msg_i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            if "image" in message:
                img = message["image"]
                render_image_gallery(
                    img if isinstance(img, list) else [img],
                    key_prefix=f"msg_{msg_i}",
                )
                edit_col_input, edit_col_btn = st.columns([4, 1])
                with edit_col_input:
                    st.text_input(
                        "Edit instruction",
                        key=f"edit_text_{msg_i}",
                        placeholder="Describe the modification...",
                        disabled=is_busy,
                        label_visibility="collapsed",
                    )
                with edit_col_btn:
                    edit_clicked = st.button(
                        "✏️ Edit",
                        key=f"edit_btn_{msg_i}",
                        disabled=is_busy,
                        use_container_width=True,
                    )
                if edit_clicked and not is_busy:
                    edit_text = st.session_state.get(
                        f"edit_text_{msg_i}", ""
                    ).strip()
                    if edit_text:
                        img_path = img[0] if isinstance(img, list) else img
                        edit_request = {
                            "image_path": img_path,
                            "prompt": edit_text,
                        }
            if "error" in message:
                st.error(message["error"])


# -----------------------------------------------------------------------------
# Prompt input and action buttons
# -----------------------------------------------------------------------------
# Text area holds the current prompt (synced with prompt_draft). Improve and
# Generate buttons trigger pending flows; we use a separate key for the widget
# so we can programmatically set prompt_draft/prompt_box without conflicts.
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


# -----------------------------------------------------------------------------
# Pending: run prompt improvement (Qwen)
# -----------------------------------------------------------------------------
# When the user clicked “Improve prompt”, we stored the prompt in
# pending_improve. Here we call the improve API and put the result in
# improved_prompt_to_apply; the next rerun will apply it to the text area.
if st.session_state.pending_improve is not None:
    prompt_to_improve = st.session_state.pending_improve
    with st.spinner("Improving prompt with Qwen..."):
        try:
            # Pass the user-defined system prompt so customisations are applied.
            improved = improve_prompt(
                prompt_to_improve,
                custom_system_prompt=st.session_state.system_prompt,
            )
            if improved:
                st.session_state.improved_prompt_to_apply = improved
        except Exception as e:
            st.error(f"Failed to improve prompt: {str(e)}")
    st.session_state.pending_improve = None
    st.rerun()


# -----------------------------------------------------------------------------
# Pending: run image generation (Z-Image-Turbo)
# -----------------------------------------------------------------------------
# When the user clicked “Generate”, we stored the prompt in pending_generate.
# We append the user message, then call the Space (via services) for each
# image, show the gallery, and append the assistant message. We clear the
# prompt input on the next run and rerun so the UI updates in one step.
if st.session_state.pending_generate is not None:
    prompt = st.session_state.pending_generate
    st.session_state.clear_prompt_on_next_run = True
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        num_images = st.session_state.get("num_images", 1)
        use_random_seed = st.session_state.get("use_random_seed", True)
        fixed_seed = st.session_state.get("seed", 0)

        try:
            image_paths, seeds_used = generate_images(
                st.session_state.client,
                prompt,
                num_images=num_images,
                width=st.session_state.get("width", 1024),
                height=st.session_state.get("height", 1024),
                steps=st.session_state.get("steps", 9),
                use_random_seed=use_random_seed,
                fixed_seed=fixed_seed,
            )

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
                "error": error_msg,
            })

    st.session_state.pending_generate = None
    st.rerun()


# -----------------------------------------------------------------------------
# Pending: run image edit (FireRed-Image-Edit)
# -----------------------------------------------------------------------------
# When the user clicked "Edit" on an assistant image, we stored the image path
# and edit instruction in pending_edit. Here we lazily connect to the FireRed
# Space, call edit_image, and append the result to chat history.
if st.session_state.pending_edit is not None:
    edit_data = st.session_state.pending_edit

    if "edit_client" not in st.session_state:
        with st.spinner("Connecting to FireRed Image Edit Space..."):
            try:
                hf_token = os.getenv("HF_TOKEN")
                st.session_state.edit_client = Client(
                    "macgaga/FireRed-Image-Edit-1.0-Fast",
                    token=hf_token,
                )
            except Exception as e:
                st.error(f"Failed to connect to FireRed Edit Space: {e}")
                st.session_state.pending_edit = None
                st.rerun()

    st.session_state.messages.append({
        "role": "user",
        "content": f"✏️ Edit: {edit_data['prompt']}",
    })

    with st.chat_message("user"):
        st.markdown(f"✏️ Edit: {edit_data['prompt']}")

    with st.chat_message("assistant"):
        with st.spinner("Editing image with FireRed..."):
            try:
                edited_path, seed_used = edit_image(
                    st.session_state.edit_client,
                    edit_data["image_path"],
                    edit_data["prompt"],
                )
                if edited_path:
                    render_image_gallery([edited_path], key_prefix="edit_result")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "image": edited_path,
                        "prompt": edit_data["prompt"],
                        "seed": seed_used,
                    })
                else:
                    error_msg = "Edit returned no image."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "error": error_msg,
                    })
            except Exception as e:
                error_msg = f"Error editing image: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "error": error_msg,
                })

    st.session_state.pending_edit = None
    st.rerun()


# -----------------------------------------------------------------------------
# Button handlers: queue improve or generate
# -----------------------------------------------------------------------------
# When not busy, clicking Improve or Generate sets the corresponding pending
# flag and reruns; the blocks above then run the API and update the UI.
# Empty prompt is rejected with a short warning.
if not is_busy and improve_clicked and prompt_input.strip():
    st.session_state.pending_improve = prompt_input.strip()
    st.rerun()
elif not is_busy and improve_clicked and not prompt_input.strip():
    st.warning("Enter a prompt first, then click Improve prompt.")

if not is_busy and generate_clicked and prompt_input.strip():
    st.session_state.pending_generate = prompt_input.strip()
    st.rerun()
elif not is_busy and generate_clicked and not prompt_input.strip():
    st.warning("Enter a prompt first, then click Generate.")

if edit_request is not None and not is_busy:
    st.session_state.pending_edit = edit_request
    st.rerun()


# -----------------------------------------------------------------------------
# Sidebar: image settings and app info
# -----------------------------------------------------------------------------
# Resolution, inference steps, number of images, and seed options. Stored in
# session state so the generate flow can read them. Clear chat and static
# “About” / “Tips” content live here as well.
with st.sidebar:
    st.header("Settings")

    # -------------------------------------------------------------------------
    # Prompt Enhancer Settings
    # -------------------------------------------------------------------------
    # Allows the user to customise the system prompt sent to Qwen when "Improve
    # prompt" is clicked. Edits are stored in session state and forwarded to
    # improve_prompt() at runtime. Resetting restores the module-level default.
    with st.expander("✨ Prompt Enhancer Settings", expanded=False):
        st.caption(
            "Customise the system prompt sent to the AI when you click "
            "**✨ Improve prompt**. Use `{user_prompt}` as the placeholder "
            "for the user's original text."
        )
        edited_system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            height=300,
            key="system_prompt_editor",
            label_visibility="collapsed",
            help="The system prompt used by the Qwen model to expand your description.",
        )
        col_apply, col_reset = st.columns(2)
        with col_apply:
            if st.button(
                "Apply",
                use_container_width=True,
                help="Save your changes to the system prompt.",
            ):
                st.session_state.system_prompt = edited_system_prompt
                st.success("System prompt updated!")
        with col_reset:
            if st.button(
                "Reset to default",
                use_container_width=True,
                help="Restore the built-in D&D system prompt.",
            ):
                st.session_state.system_prompt = IMPROVE_SYSTEM
                st.rerun()

    st.markdown("---")

    st.subheader("Image Settings")
    st.session_state.width = st.select_slider(
        "Width",
        options=[512, 768, 1024, 1280],
        value=1024,
    )
    st.session_state.height = st.select_slider(
        "Height",
        options=[512, 768, 1024, 1280],
        value=1024,
    )
    st.session_state.num_images = st.slider(
        "Number of images",
        min_value=1,
        max_value=8,
        value=1,
        help="How many images to generate. For more than 1, each uses a random seed and the space is called once per image.",
    )
    use_random_seed = st.checkbox(
        "Random seed",
        value=True,
        help="Use a random seed (only applies when generating 1 image). Uncheck and set seed below for reproducible results.",
    )
    st.session_state.use_random_seed = use_random_seed
    st.session_state.seed = st.number_input(
        "Seed (for 1 image only)",
        min_value=0,
        max_value=2**32 - 1,
        value=0,
        step=1,
        help="Fixed seed for reproducibility when generating a single image. Ignored when Number of images > 1 or Random seed is on.",
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
