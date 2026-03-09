# services.py — Business logic and API clients for the AI Image Generator app.
# This module keeps all Hugging Face API calls, prompt improvement, and image generation
# logic separate from the Streamlit UI so the front-end stays readable and testable.

import os
import random
from openai import OpenAI


# -----------------------------------------------------------------------------
# Hugging Face inference configuration
# -----------------------------------------------------------------------------
# Model used for enhancing user prompts with richer D&D-style descriptions.
# The system prompt below instructs the model to return only the improved text.
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


# -----------------------------------------------------------------------------
# OpenAI-compatible client for Hugging Face
# -----------------------------------------------------------------------------
# Returns a client configured to use Hugging Face's inference API so we can
# call chat models (e.g. Qwen) for prompt improvement without hosting our own.
def get_hf_openai_client():
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )


# -----------------------------------------------------------------------------
# Prompt improvement via Hugging Face chat model
# -----------------------------------------------------------------------------
# Sends the raw user prompt to the configured model with the IMPROVE_SYSTEM
# instructions so the model returns a single enhanced prompt string. Used when
# the user clicks "Improve prompt" to get a more detailed description.
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


# -----------------------------------------------------------------------------
# Image generation via Hugging Face Space (Gradio client)
# -----------------------------------------------------------------------------
# Calls the Z-Image-Turbo Space once per requested image. For a single image
# we can use a fixed seed when the user disables "Random seed"; for multiple
# images we always use distinct random seeds so each result is unique. Returns
# (list of image paths, list of seeds used) for the UI to display and store.
def generate_images(
    client,
    prompt: str,
    *,
    num_images: int = 1,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    use_random_seed: bool = True,
    fixed_seed: int = 0,
):
    use_fixed_seed = (
        num_images == 1
        and not use_random_seed
        and fixed_seed is not None
        and fixed_seed >= 0
    )
    if num_images > 1:
        seeds_to_use = random.sample(range(0, 2**32), num_images)
    else:
        seeds_to_use = [int(fixed_seed)] if use_fixed_seed else None

    image_paths = []
    seeds_used = []

    for i in range(num_images):
        if seeds_to_use is not None:
            result = client.predict(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                randomize_seed=False,
                seed=int(seeds_to_use[i]),
                api_name="/generate_image",
            )
        else:
            result = client.predict(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                randomize_seed=True,
                api_name="/generate_image",
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

    return image_paths, seeds_used
