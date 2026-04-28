import torch
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("Z-Image-Turbo VRAM:", torch.cuda.memory_allocated() / 1024**3)
