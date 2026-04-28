# services.py — Business logic and API clients for the AI Image Generator app.
# This module keeps all Hugging Face API calls, prompt improvement, and image generation
# logic separate from the Streamlit UI so the front-end stays readable and testable.

import os
import random
from openai import OpenAI


# -----------------------------------------------------------------------------
# Hugging Face inference configuration
# Local model to use for inference. Can be overridden via LOCAL_MODEL env var.
# Assuming you run Ollama with a command like: `ollama run mistral-nemo`
LOCAL_INFERENCE_MODEL = os.getenv("LOCAL_MODEL", "mistral-nemo")

IMPROVE_SYSTEM = (
  "### ROLE"
"You are an expert TTRPG Worldbuilder and AI Image Prompt Engineer, specialized in Dungeons & Dragons (5e) aesthetics and high-end digital concept art."
"### MISSION"
"Your task is to transform brief user inputs into high-fidelity, sensory-rich descriptions optimized for image generation. You must merge the user's core idea with a specific 'Epic Fantasy Concept Art' style to create a masterwork-quality visual prompt."
"### VISUAL STYLE GUIDE (Mandatory)"
"Every output must be infused with the following stylistic elements:"
"High-end digital painting concept art style, epic fantasy RPG design, cinematic lighting with strong rim light, hyper-detailed textures, volumetric atmosphere with magical particles, vibrant color grading, sharp focus, masterwork quality, stylized realism, influenced by ArtStation trending fantasy artists."
"### RULES & CONSTRAINTS"
"VISUAL DEPTH: Include specific details about lighting (e.g., 'dramatic rim lighting'), textures of materials mentioned (e.g., 'weathered leather', 'engraved plate armor'), and atmosphere."
"TTRPG CONTEXT: Maintain a High-Fantasy D&D tone. Use appropriate terminology for races, classes, and magical effects."
"STRUCTURE: Organize the description into logical segments (Character/Subject, Gear/Clothing, Environment, and Lighting/Atmosphere)." 
"LIMITS: Do not exceed 4 paragraphs."
"NO COMMENTARY: Output ONLY the enhanced description."
"### EXAMPLES"
"User Input: 'a drow bard wearing blue robes. She is playing a guitar'"
"Enhanced Output:"
"A statuesque Drow stands bathed in cinematic indigo rim light, her obsidian skin rendered in stylized realism against cascading hair as pale as moonlight. She is draped in flowing robes of deep sapphire velvet, featuring hyper-detailed textures of embroidered silver constellations that pulse with latent arcane energy. The silhouette is one of athletic grace, blending the elegance of a noble house with the practical gear of a traveling minstrel, all in a high-end digital painting style."
"Cradled against the bard's chest is a resonant baritone lute-guitar crafted from polished black ironwood. The fretboard is inlaid with mother-of-pearl runes that hum with visible vibration, while the strings shimmer like spun spider-silk. Every strum sends magical particles swirling into the air, catching the vibrant color grading of the scene in a display of masterwork-quality concept art."
"The environment melts into a volumetric atmosphere of soft-focus shadows, suggesting the twilight of the Underdark. The lighting is dramatic and sculptural, casting sharp highlights on the musician's pointed ears and defined jawline. Tiny motes of violet fire and magical dust swirl around the neck of the instrument, enhancing the epic fantasy RPG aesthetic."
"With piercing red eyes locked onto the viewer, the performer channels a charisma that transcends mere melody. The composition captures a moment of captivating performance, balancing elven grace with the subtle menace of Menzoberranzan, presented with the sharp focus and vibrant energy characteristic of trending ArtStation fantasy masterpieces."
"### USER REQUEST"
"{user_prompt}"
)


# -----------------------------------------------------------------------------
# OpenAI-compatible client for local inference engine (e.g. Ollama, vLLM)
# -----------------------------------------------------------------------------
# Returns a client configured to use a local API server so we can
# call chat models (e.g. Mistral) for prompt improvement completely offline.
def get_local_openai_client():
    return OpenAI(
        base_url=os.getenv("LOCAL_API_BASE", "http://127.0.0.1:11434/v1"),
        api_key=os.getenv("LOCAL_API_KEY", "ollama"),
    )


# -----------------------------------------------------------------------------
# Prompt improvement via Local chat model
# -----------------------------------------------------------------------------
# Sends the raw user prompt to the configured model with the IMPROVE_SYSTEM
# instructions so the model returns a single enhanced prompt string. Used when
# the user clicks "Improve prompt" to get a more detailed description.
def improve_prompt(raw_prompt: str) -> str:
    if not raw_prompt or not raw_prompt.strip():
        return ""
    prompt_text = raw_prompt.strip()
    system_content = IMPROVE_SYSTEM.replace("{prompt}", prompt_text)
    client = get_local_openai_client()
    completion = client.chat.completions.create(
        model=LOCAL_INFERENCE_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=1024,
    )
    return (completion.choices[0].message.content or "").strip()


# -----------------------------------------------------------------------------
# Image generation via Local Server (Gradio client)
# -----------------------------------------------------------------------------
# Calls the local image generation backend once per requested image. For a single image
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


# -----------------------------------------------------------------------------
# Image editing via Local Server (FireRed-Image-Edit)
# -----------------------------------------------------------------------------
# Sends an existing image and an edit instruction to the local FireRed-Image-Edit
# Space.  The Gallery input requires each item wrapped as a GalleryImage dict
# ({"image": handle_file(...)}).  The output is (ImageData, seed).
# Returns (edited_image_path, seed_used).
def edit_image(
    client,
    image_path: str,
    prompt: str,
    *,
    seed: int = 0,
    randomize_seed: bool = True,
    guidance_scale: float = 1.0,
    steps: int = 4,
):
    from gradio_client import handle_file

    result = client.predict(
        images=[{"image": handle_file(image_path)}],
        prompt=prompt,
        seed=seed,
        randomize_seed=randomize_seed,
        guidance_scale=guidance_scale,
        steps=steps,
        api_name="/infer",
    )

    if isinstance(result, tuple):
        image_data = result[0]
        seed_used = result[1] if len(result) > 1 else seed
    else:
        image_data = result
        seed_used = seed

    if isinstance(image_data, dict):
        edited_path = image_data.get("path") or image_data.get("url")
    elif isinstance(image_data, str):
        edited_path = image_data
    else:
        edited_path = image_data

    return edited_path, int(seed_used)
