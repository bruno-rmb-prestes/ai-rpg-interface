import os
import gc
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image
from diffusers import DiffusionPipeline

from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

# --- Config & Init ---
device_gen = "cuda:0"
device_edit = "cuda:0"
dtype = torch.bfloat16

# Global references to the pipelines
active_pipe = None
current_loaded = None

from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

def init_models():
    """
    Initializes by ensuring the generation model is loaded first.
    """
    ensure_gen_loaded()
    print("Backend initialized successfully.")

# --- Pipeline Swapping Logic ---
def ensure_gen_loaded():
    """
    Ensures that Z-Image-Turbo is loaded. If FireRed is present, it deletes it to free VRAM.
    """
    global active_pipe, current_loaded
    if current_loaded == "gen":
        return
        
    print("Swapping models: Loading Z-Image-Turbo to GPU (8-bit quantized)...")
    if active_pipe is not None:
        del active_pipe
        gc.collect()
        torch.cuda.empty_cache()
        
    quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit", 
        quant_kwargs={"load_in_8bit": True}
    )
    
    active_pipe = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=dtype,
        quantization_config=quant_config,
    )
    active_pipe.to(device_gen)
    current_loaded = "gen"
    print("Z-Image-Turbo is now active.")

def ensure_edit_loaded():
    """
    Ensures that FireRed is loaded. If Z-Image-Turbo is present, it deletes it to free VRAM.
    """
    global active_pipe, current_loaded
    if current_loaded == "edit":
        return
        
    print("Swapping models: Loading FireRed-Image-Edit-1.1 to GPU (8-bit quantized)...")
    if active_pipe is not None:
        del active_pipe
        gc.collect()
        torch.cuda.empty_cache()
        
    quant_config_edit = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit", 
        quant_kwargs={"load_in_8bit": True}
    )
    quant_config_model = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    
    active_pipe = QwenImageEditPlusPipeline.from_pretrained(
        "FireRedTeam/FireRed-Image-Edit-1.1",
        transformer=QwenImageTransformer2DModel.from_pretrained(
            "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
            torch_dtype=dtype,
            quantization_config=quant_config_model,
        ),
        torch_dtype=dtype,
        quantization_config=quant_config_edit,
    )
    active_pipe.to(device_gen)  # Map to the only visible GPU
    
    try:
        active_pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        print("Flash Attention 3 Processor set successfully.")
    except Exception as e:
        print(f"Warning: Could not set FA3 processor: {e}")
        
    current_loaded = "edit"
    print("FireRed-Image-Edit is now active.")


# --- Inference Functions ---
def generate_image(prompt, height, width, num_inference_steps, seed, randomize_seed, progress=gr.Progress(track_tqdm=True)):
    """
    Generates a new image based on a text prompt.
    Automatically ensures the correct model is loaded into VRAM before generating.
    
    Args:
        prompt (str): The text description of the image.
        height (int): Image height.
        width (int): Image width.
        num_inference_steps (int): Diffusion steps.
        seed (int): The generation seed.
        randomize_seed (bool): Whether to ignore the seed and generate randomly.
        progress (gr.Progress): Gradio progress tracker.
        
    Returns:
        tuple: A tuple containing the generated PIL Image and the exact seed used.
    """
    ensure_gen_loaded()
    
    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator(device_gen).manual_seed(int(seed))
    image = active_pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    
    return image, seed

MAX_SEED = np.iinfo(np.int32).max

def update_dimensions_on_upload(image):
    """
    Calculates appropriate dimensions for an uploaded image to maintain aspect ratio 
    while clamping the maximum side to 1024px and ensuring the result is divisible by 8.
    
    Args:
        image (Image.Image): The uploaded PIL Image.
        
    Returns:
        tuple: A tuple of (new_width, new_height).
    """
    if image is None:
        return 1024, 1024
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    return new_width, new_height

def infer(
    images,
    prompt,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Performs an instruction-based image edit on an uploaded image.
    Automatically ensures the correct FireRed model is loaded into VRAM before editing.
    
    Args:
        images (list): A list of uploaded images (from Gradio Gallery).
        prompt (str): The text instructions for editing.
        seed (int): The generation seed.
        randomize_seed (bool): Whether to randomize the seed.
        guidance_scale (float): Scale for classifier-free guidance.
        steps (int): Number of inference steps for the edit.
        progress (gr.Progress): Gradio progress tracker.
        
    Returns:
        tuple: A tuple containing the edited PIL Image and the exact seed used.
    """
    ensure_edit_loaded()
    
    if not images:
        raise gr.Error("Please upload at least one image to edit.")

    pil_images = []
    if images is not None:
        for item in images:
            try:
                if isinstance(item, tuple) or isinstance(item, list):
                    path_or_img = item[0]
                else:
                    path_or_img = item

                if isinstance(path_or_img, str):
                    pil_images.append(Image.open(path_or_img).convert("RGB"))
                elif isinstance(path_or_img, Image.Image):
                    pil_images.append(path_or_img.convert("RGB"))
                else:
                    pil_images.append(Image.open(path_or_img.name).convert("RGB"))
            except Exception as e:
                print(f"Skipping invalid image item: {e}")
                continue

    if not pil_images:
        raise gr.Error("Could not process uploaded images.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device_edit).manual_seed(seed)
    negative_prompt = "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

    width, height = update_dimensions_on_upload(pil_images[0])

    try:
        result_image = active_pipe(
            image=pil_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            true_cfg_scale=guidance_scale,
        ).images[0]
        return result_image, seed
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()

# Load models on startup
init_models()

# --- Gradio UI & Launch ---
# Build the Gradio interface
with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# 🎨 Unified Local Backend Manager")
    
    with gr.Tab("Generate Image"):
        with gr.Row():
            prompt_gen = gr.Textbox(label="Prompt", lines=3)
            with gr.Column():
                height_gen = gr.Slider(minimum=512, maximum=2048, value=1024, step=64, label="Height")
                width_gen = gr.Slider(minimum=512, maximum=2048, value=1024, step=64, label="Width")
                num_inference_steps_gen = gr.Slider(minimum=1, maximum=20, value=9, step=1, label="Inference Steps")
                randomize_seed_gen = gr.Checkbox(label="Randomize Seed", value=True)
                seed_gen = gr.Number(label="Seed", value=42, precision=0)
                generate_btn = gr.Button("Generate")

        with gr.Row():
            output_image_gen = gr.Image(label="Generated Image", type="pil")
            used_seed_gen = gr.Number(label="Seed Used")
        
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_gen, height_gen, width_gen, num_inference_steps_gen, seed_gen, randomize_seed_gen],
            outputs=[output_image_gen, used_seed_gen],
            api_name="generate_image"
        )
        
    with gr.Tab("Edit Image"):
        with gr.Row():
            with gr.Column():
                images_edit = gr.Gallery(label="Upload Images", type="filepath", columns=2, rows=1, height=300)
                prompt_edit = gr.Text(label="Edit Prompt")
                edit_btn = gr.Button("Edit Image", variant="primary")

            with gr.Column():
                output_image_edit = gr.Image(label="Output Image", interactive=False, type="pil")
                seed_edit = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed_edit = gr.Checkbox(label="Randomize Seed", value=True)
                guidance_scale_edit = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                steps_edit = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=4)

        edit_btn.click(
            fn=infer,
            inputs=[images_edit, prompt_edit, seed_edit, randomize_seed_edit, guidance_scale_edit, steps_edit],
            outputs=[output_image_edit, seed_edit],
            api_name="infer"
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
