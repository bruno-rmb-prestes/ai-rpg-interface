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
    "### ROLE"
    "You are an expert TTRPG Worldbuilder and Lead Narrative Designer, specialized in Dungeons & Dragons (5e) aesthetics. Your expertise lies in transforming brief ideas into high-fidelity, sensory-rich descriptions for places and characters."

    "### MISSION"
    "Your task is to expand and enhance the user's input prompt. You must enrich the description by adding atmospheric details, architectural specifics for locations, or physiological and equipment depth for characters, ensuring the result is immersive and ready for high-quality image generation."

    "### RULES & CONSTRAINTS"
    "- VISUAL DEPTH: Include details about lighting (e.g., 'amber flickering torchlight'), textures (e.g., 'weathered obsidian'), and atmosphere (e.g., 'thick with the scent of ozone')."
    "- TTRPG CONTEXT: Maintain a High-Fantasy D&D tone. Use appropriate terminology for races, classes, and magical effects."
    "- BREVITY: The final enhanced description must be concise and professional. Do not exceed 4 paragraphs."
    "- NO COMMENTARY: Do not include introductory text like 'Here is your improved prompt'. Output ONLY the enhanced description."
    "- EXAMPLES: Follow the examples below to understand the task and the format of the output."

    "### EXAMPLES"
    "**User Input:** 'a drow bard wearing blue robes. She is playing a guitar'"
    "**Enhanced Output:**"
    "A statuesque Drow stands bathed in ethereal indigo glow, their obsidian skin contrasting starkly against cascading hair as pale as moonlight. They are draped in flowing robes of deep sapphire velvet, embroidered with shifting silver constellations that seem to pulse with latent arcane energy across the fabric, hugging a silhouette of athletic grace suitable for a traveling minstrel or noble spy."
    "Cradled against the bard's chest is a resonant baritone lute, crafted from polished black ironwood, its body shaped with the proportions of an acoustic guitar yet carved from ancient darkwood. The fretboard is inlaid with mother-of-pearl runes that hum with visible vibration, while the strings shimmer like spun spider-silk, catching the ambient light with every precise strumming motion."
    "The environment melts into soft-focus shadows, suggesting the twilight of a surface city or the upper reaches of the Underdark. Lighting is dramatic and cinematic, casting long, sculptural highlights on the musician's pointed ears and defined jawline, while tiny motes of magical dust and sparks of violet fire swirl around the neck of the instrument."
    "With piercing red eyes locked onto the viewer, the performer channels charisma that transcends mere melody, exuding an aura of seductive danger typical of the noble houses of Menzoberranzan. The composition captures a moment of captivating performance, balancing the elegance of elven grace with the subtle menace inherent in the deep."

"### USER PROMPT"
"{user_prompt}"
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
