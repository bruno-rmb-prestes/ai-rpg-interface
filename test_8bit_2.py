import torch
from diffusers import DiffusionPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)

pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config
)
print("VRAM loaded:", torch.cuda.memory_allocated() / 1024**3)
try:
    pipe("A magical sword", height=1024, width=1024, num_inference_steps=2).images[0]
    print("VRAM after gen:", torch.cuda.max_memory_allocated() / 1024**3)
except Exception as e:
    print("Error:", e)
