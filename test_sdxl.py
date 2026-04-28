import torch
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("VRAM loaded:", torch.cuda.memory_allocated() / 1024**3)
pipe("A magic sword", num_inference_steps=2, guidance_scale=0.0).images[0]
print("VRAM after gen:", torch.cuda.max_memory_allocated() / 1024**3)
