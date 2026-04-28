import torch
from diffusers import DiffusionPipeline
import time

print("Loading model balanced...")
pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)
print("VRAM GPU0:", torch.cuda.memory_allocated(0) / 1024**3)
print("VRAM GPU1:", torch.cuda.memory_allocated(1) / 1024**3)
print("Finished!")
