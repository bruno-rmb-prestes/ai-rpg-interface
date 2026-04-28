import torch
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print("Loading text encoder in 8-bit...")
text_encoder = T5EncoderModel.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    subfolder="text_encoder", 
    quantization_config=quantization_config, 
    torch_dtype=torch.bfloat16
)

print("Loading pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
print("VRAM loaded:", torch.cuda.memory_allocated() / 1024**3)
try:
    pipe.to("cuda")
    pipe("A magical sword", height=1024, width=1024, num_inference_steps=2).images[0]
    print("VRAM after gen:", torch.cuda.max_memory_allocated() / 1024**3)
except Exception as e:
    print("Error:", e)
