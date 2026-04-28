import torch
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)
print("VRAM loaded GPU0:", torch.cuda.memory_allocated(0) / 1024**3)
print("VRAM loaded GPU1:", torch.cuda.memory_allocated(1) / 1024**3)
try:
    pipe("A cat", height=1024, width=1024, num_inference_steps=2).images[0]
    print("VRAM after gen GPU0:", torch.cuda.max_memory_allocated(0) / 1024**3)
    print("VRAM after gen GPU1:", torch.cuda.max_memory_allocated(1) / 1024**3)
except Exception as e:
    print("Error:", e)
